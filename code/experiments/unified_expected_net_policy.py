"""Fixed zero-net two-of-three ensemble policy for Notebook 04f."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_models import MODEL_NAMES
from experiments.unified_2021_ensemble_policy import (
    REFRACTORY,
    causal_crossings,
    replay_selected_paths,
)


def _prediction_columns() -> tuple[str, ...]:
    return tuple(
        f"pred_{side}_{model}"
        for side in ("long", "short")
        for model in MODEL_NAMES
    )


def model_side_votes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply each model's strict positive and side-dominance vote rule."""
    missing = sorted(set(_prediction_columns()).difference(predictions.columns))
    if missing:
        raise ValueError(f"expected-net voting lacks predictions: {missing}")
    votes = pd.DataFrame(index=predictions.index)
    for model in MODEL_NAMES:
        long = pd.to_numeric(
            predictions[f"pred_long_{model}"], errors="coerce"
        ).to_numpy(float)
        short = pd.to_numeric(
            predictions[f"pred_short_{model}"], errors="coerce"
        ).to_numpy(float)
        if not np.isfinite(long).all() or not np.isfinite(short).all():
            raise ValueError(f"{model} expected-net predictions must be finite")
        votes[f"vote_{model}"] = np.where(
            (long > 0.0) & (long > short),
            "long",
            np.where((short > 0.0) & (short > long), "short", "wait"),
        )
    return votes


def score_fixed_expected_net_routes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score the one preregistered policy; no threshold or rate is selected."""
    scored = predictions.copy()
    votes = model_side_votes(scored)
    for column in votes:
        scored[column] = votes[column].to_numpy()

    long_votes = sum(votes[f"vote_{model}"].eq("long") for model in MODEL_NAMES)
    short_votes = sum(votes[f"vote_{model}"].eq("short") for model in MODEL_NAMES)
    provisional_side = np.where(
        long_votes >= 2,
        "long",
        np.where(short_votes >= 2, "short", "wait"),
    )

    candidate_side: list[str] = []
    agreeing_models: list[str] = []
    agreeing_votes: list[int] = []
    predicted_net: list[float] = []
    routes: list[str] = []
    for row_position, side in enumerate(provisional_side):
        if side == "wait":
            candidate_side.append("wait")
            agreeing_models.append("")
            agreeing_votes.append(0)
            predicted_net.append(0.0)
            routes.append("none")
            continue
        agreeing = [
            model
            for model in MODEL_NAMES
            if votes.iloc[row_position][f"vote_{model}"] == side
        ]
        side_values = np.asarray(
            [
                float(scored.iloc[row_position][f"pred_{side}_{model}"])
                for model in agreeing
            ],
            dtype=float,
        )
        median = float(np.median(side_values))
        if len(agreeing) < 2 or median <= 0.0:
            candidate_side.append("wait")
            agreeing_models.append(",".join(agreeing))
            agreeing_votes.append(len(agreeing))
            predicted_net.append(median)
            routes.append("none")
            continue
        candidate_side.append(side)
        agreeing_models.append(",".join(agreeing))
        agreeing_votes.append(len(agreeing))
        predicted_net.append(median)
        if len(agreeing) == 3:
            routes.append("unanimous")
        elif "xgboost" in agreeing:
            routes.append("xgboost_decisive")
        else:
            routes.append("without_xgboost")

    scored["long_votes"] = np.asarray(long_votes, dtype=int)
    scored["short_votes"] = np.asarray(short_votes, dtype=int)
    scored["candidate_side"] = candidate_side
    scored["agreeing_models"] = agreeing_models
    scored["agreeing_votes"] = agreeing_votes
    scored["predicted_net_bps"] = predicted_net
    scored["route"] = routes
    scored["xgboost_solo"] = False
    if scored["route"].eq("xgboost_solo").any():
        raise AssertionError("the fixed expected-net policy cannot route XGBoost solo")
    return scored


def fixed_policy_activations(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay side-specific cold crossings and the shared 60-minute refractory."""
    scored = score_fixed_expected_net_routes(predictions).sort_values(
        "decision_time", kind="stable"
    )
    scored["decision_time"] = pd.to_datetime(
        scored["decision_time"], utc=True, errors="raise"
    )
    time = pd.DatetimeIndex(scored["decision_time"])
    if not time.is_monotonic_increasing or not time.is_unique:
        raise ValueError("fixed policy needs unique increasing decision times")
    by_time = scored.set_index("decision_time", drop=False)
    events: list[dict[str, object]] = []
    for side in ("long", "short"):
        signal = pd.Series(
            scored["candidate_side"].eq(side).to_numpy(float),
            index=time,
            name=f"candidate_{side}",
        )
        for timestamp in causal_crossings(signal, 0.5, REFRACTORY):
            row = by_time.loc[timestamp]
            if isinstance(row, pd.DataFrame):
                raise AssertionError("decision timestamps must be unique")
            events.append(
                {
                    "decision_time": timestamp,
                    "row_key": row["row_key"],
                    "selected_side": side,
                    "route": row["route"],
                    "agreeing_models": row["agreeing_models"],
                    "agreeing_votes": int(row["agreeing_votes"]),
                    "predicted_net_bps": float(row["predicted_net_bps"]),
                    "xgboost_solo": False,
                }
            )
    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        funnel = scored.copy()
        funnel["candidate"] = False
        funnel["accepted"] = False
        funnel["decision_reason"] = "no_cold_crossing"
        empty = funnel.iloc[0:0].copy()
        empty["selected_side"] = pd.Series(dtype=str)
        return empty, funnel.reset_index(drop=True)

    event_frame = event_frame.sort_values("decision_time", kind="stable")
    accepted: list[bool] = []
    previous_candidate: pd.Timestamp | None = None
    for timestamp in pd.to_datetime(event_frame["decision_time"], utc=True):
        permitted = (
            previous_candidate is None
            or timestamp - previous_candidate >= pd.Timedelta(REFRACTORY)
        )
        accepted.append(permitted)
        previous_candidate = timestamp
    event_frame["accepted"] = accepted
    accepted_events = event_frame.loc[event_frame["accepted"]].drop(
        columns="accepted"
    )
    activations = scored.merge(
        accepted_events,
        on=["row_key", "decision_time"],
        how="inner",
        validate="one_to_one",
        suffixes=("", "_activation"),
    )
    for column in (
        "route",
        "agreeing_models",
        "agreeing_votes",
        "predicted_net_bps",
        "xgboost_solo",
    ):
        activation_column = f"{column}_activation"
        if activation_column in activations:
            activations[column] = activations.pop(activation_column)

    candidate_keys = set(event_frame["row_key"].astype(str))
    accepted_keys = set(accepted_events["row_key"].astype(str))
    funnel = scored.copy()
    keys = funnel["row_key"].astype(str)
    funnel["candidate"] = keys.isin(candidate_keys)
    funnel["accepted"] = keys.isin(accepted_keys)
    funnel["decision_reason"] = np.where(
        funnel["accepted"],
        "accepted",
        np.where(
            funnel["candidate"], "global_refractory_rejection", "no_cold_crossing"
        ),
    )
    return (
        activations.sort_values("decision_time", kind="stable").reset_index(drop=True),
        funnel.reset_index(drop=True),
    )


def apply_fixed_expected_net_policy(
    predictions: pd.DataFrame,
    economic_paths: pd.DataFrame,
) -> pd.DataFrame:
    """Return the exact single-position ledger for the registered fixed policy."""
    activations, _ = fixed_policy_activations(predictions)
    ledger = replay_selected_paths(activations, economic_paths)
    if not ledger.empty:
        if ledger["route"].eq("xgboost_solo").any():
            raise AssertionError("XGBoost solo escaped the fixed policy")
        selected = ledger["selected_side"].astype(str)
        direction = ledger["direction"].astype(str)
        if not selected.equals(direction):
            raise AssertionError("selected side and replayed path direction differ")
    return ledger


def _positive_fold_count(
    ledger: pd.DataFrame,
    mask: pd.Series | np.ndarray,
) -> int:
    selected = ledger.loc[np.asarray(mask, dtype=bool)]
    if selected.empty:
        return 0
    by_fold = selected.groupby("fold_id", sort=True)["net_return"].sum()
    return int(by_fold.gt(0.0).sum())


def _audit_flag(audit: Mapping[str, object], name: str) -> bool:
    return bool(audit.get(name, False))


def evaluate_expected_net_development(
    ledger: pd.DataFrame,
    audit: Mapping[str, object],
) -> dict[str, object]:
    """Apply every registered absolute development gate without tuning."""
    if ledger.empty:
        direction = pd.Series(dtype=str)
        net = pd.Series(dtype=float)
        total_mask = np.zeros(0, dtype=bool)
        solo = pd.Series(dtype=bool)
    else:
        required = {"fold_id", "direction", "net_return"}
        missing = sorted(required.difference(ledger.columns))
        if missing:
            raise ValueError(f"development ledger lacks columns: {missing}")
        direction = ledger["direction"].astype(str).str.lower()
        net = pd.to_numeric(ledger["net_return"], errors="raise")
        if not np.isfinite(net.to_numpy(float)).all():
            raise ValueError("development ledger returns must be finite")
        total_mask = np.ones(len(ledger), dtype=bool)
        route = ledger.get("route", pd.Series("none", index=ledger.index)).astype(str)
        solo = route.eq("xgboost_solo")

    trades = int(len(ledger))
    long_mask = direction.eq("long")
    short_mask = direction.eq("short")
    long_trades = int(long_mask.sum())
    short_trades = int(short_mask.sum())
    required_side_trades = max(15, int(np.ceil(0.20 * trades)))
    total_net = float(net.sum())
    long_net = float(net.loc[long_mask].sum())
    short_net = float(net.loc[short_mask].sum())
    total_positive_folds = _positive_fold_count(ledger, total_mask)
    long_positive_folds = _positive_fold_count(ledger, long_mask)
    short_positive_folds = _positive_fold_count(ledger, short_mask)
    xgboost_solo_trades = int(solo.sum())

    leakage_clean = _audit_flag(audit, "leakage_clean")
    reconciliation_clean = _audit_flag(audit, "reconciliation_clean")
    path_contract_clean = _audit_flag(audit, "path_contract_clean")
    cost_contract_clean = _audit_flag(audit, "cost_contract_clean")
    unique_trade_keys = bool(
        ledger.empty
        or "row_key" not in ledger
        or not ledger["row_key"].astype(str).duplicated().any()
    )
    gates = {
        "frequency_gate": trades >= 132,
        "side_count_gate": (
            long_trades >= required_side_trades
            and short_trades >= required_side_trades
        ),
        "total_net_gate": total_net > 0.0,
        "long_net_gate": long_net > 0.0,
        "short_net_gate": short_net > 0.0,
        "total_fold_gate": total_positive_folds >= 3,
        "long_fold_gate": long_positive_folds >= 3,
        "short_fold_gate": short_positive_folds >= 3,
        "xgboost_solo_zero_gate": xgboost_solo_trades == 0,
        "leakage_gate": leakage_clean,
        "reconciliation_gate": reconciliation_clean,
        "path_contract_gate": path_contract_clean,
        "cost_contract_gate": cost_contract_clean,
        "unique_trade_keys_gate": unique_trade_keys,
    }
    development_pass = bool(all(gates.values()))
    return {
        "trades": trades,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "required_side_trades": required_side_trades,
        "net_return": total_net,
        "long_net_return": long_net,
        "short_net_return": short_net,
        "total_positive_folds": total_positive_folds,
        "long_positive_folds": long_positive_folds,
        "short_positive_folds": short_positive_folds,
        "xgboost_solo_trades": xgboost_solo_trades,
        "leakage_clean": leakage_clean,
        "reconciliation_clean": reconciliation_clean,
        "path_contract_clean": path_contract_clean,
        "cost_contract_clean": cost_contract_clean,
        "unique_trade_keys": unique_trade_keys,
        **gates,
        "development_pass": development_pass,
        "decision": "development_pass" if development_pass else "development_fail",
    }


__all__ = [
    "apply_fixed_expected_net_policy",
    "evaluate_expected_net_development",
    "fixed_policy_activations",
    "model_side_votes",
    "score_fixed_expected_net_routes",
]
