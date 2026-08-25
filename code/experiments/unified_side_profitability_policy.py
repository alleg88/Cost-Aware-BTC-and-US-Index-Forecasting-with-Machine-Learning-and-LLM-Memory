"""Nested risk/coverage policy and XGBoost attribution for Notebook 04e."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_models import MODEL_NAMES
from experiments.unified_2021_ensemble_policy import (
    REFRACTORY,
    causal_crossings,
    replay_selected_paths,
)


@dataclass(frozen=True, order=True)
class RiskCoveragePolicy:
    regular_side_rate_cap: float
    xgb_side_rate_cap: float


POLICY_GRID = tuple(
    RiskCoveragePolicy(regular, satellite)
    for regular in (0.5, 1.0, 1.5)
    for satellite in (0.125, 0.25, 0.5)
)
assert len(POLICY_GRID) == 9


@dataclass
class FoldPolicySelection:
    fold_id: int
    policy: RiskCoveragePolicy
    thresholds: dict[str, float]
    selection_passed: bool
    source_role: str
    policy_grid: pd.DataFrame
    threshold_frontier: pd.DataFrame


def _probability_columns() -> tuple[str, ...]:
    return tuple(
        f"p_{side}_{model}"
        for side in ("long", "short")
        for model in MODEL_NAMES
    )


def score_routes(predictions: pd.DataFrame) -> pd.DataFrame:
    """Derive majority and guarded XGBoost-solo scores from calibrated heads."""
    missing = sorted(set(_probability_columns()).difference(predictions.columns))
    if missing:
        raise ValueError(f"route scoring lacks probabilities: {missing}")
    scored = predictions.copy()
    for column in _probability_columns():
        values = pd.to_numeric(scored[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
            raise ValueError(f"{column} must contain finite probabilities")
        scored[column] = values

    preferences: dict[str, np.ndarray] = {}
    signed_edges: dict[str, np.ndarray] = {}
    for model in MODEL_NAMES:
        edge = (
            scored[f"p_long_{model}"].to_numpy(float)
            - scored[f"p_short_{model}"].to_numpy(float)
        )
        signed_edges[model] = edge
        preferences[model] = np.where(
            edge > 1e-12,
            "long",
            np.where(edge < -1e-12, "short", "none"),
        )
        scored[f"preferred_side_{model}"] = preferences[model]
        scored[f"side_edge_{model}"] = np.abs(edge)

    long_votes = sum(preferences[model] == "long" for model in MODEL_NAMES)
    short_votes = sum(preferences[model] == "short" for model in MODEL_NAMES)
    regular_side = np.where(
        long_votes >= 2,
        "long",
        np.where(short_votes >= 2, "short", "none"),
    )
    regular_votes = np.maximum(long_votes, short_votes).astype(int)
    scored["regular_side"] = regular_side
    scored["regular_votes"] = regular_votes

    side_probability = np.empty((len(scored), len(MODEL_NAMES)), dtype=float)
    aligned_edge = np.empty_like(side_probability)
    for model_index, model in enumerate(MODEL_NAMES):
        side_probability[:, model_index] = np.where(
            regular_side == "long",
            scored[f"p_long_{model}"].to_numpy(float),
            np.where(
                regular_side == "short",
                scored[f"p_short_{model}"].to_numpy(float),
                0.0,
            ),
        )
        aligned_edge[:, model_index] = np.where(
            regular_side == "long",
            np.maximum(signed_edges[model], 0.0),
            np.where(
                regular_side == "short",
                np.maximum(-signed_edges[model], 0.0),
                0.0,
            ),
        )
    regular_score = np.median(side_probability, axis=1) + np.median(
        aligned_edge, axis=1
    )
    scored["regular_score"] = np.where(regular_side != "none", regular_score, 0.0)

    xgb_side = preferences["xgboost"]
    opposite = np.where(xgb_side == "long", "short", "long")
    two_opponents = (
        (preferences["lstm"] == opposite)
        & (preferences["svm_linear"] == opposite)
        & (xgb_side != "none")
    )
    # Eligibility is evaluated before route thresholds.  A regular majority can
    # exist but fail its stricter confidence cutoff; the satellite may then
    # admit a sufficiently strong XGBoost signal.  It remains vetoed when both
    # independent models point to the opposite side.
    satellite_eligible = (xgb_side != "none") & ~two_opponents
    scored["xgb_satellite_eligible"] = satellite_eligible
    scored["xgb_satellite_side"] = np.where(satellite_eligible, xgb_side, "none")
    scored["xgb_satellite_score"] = np.where(
        satellite_eligible, np.abs(signed_edges["xgboost"]), 0.0
    )

    regular_route = np.full(len(scored), "none", dtype=object)
    unanimous = (regular_side != "none") & (regular_votes == 3)
    xgb_decisive = (
        (regular_side != "none")
        & (regular_votes == 2)
        & (preferences["xgboost"] == regular_side)
    )
    without_xgb = (
        (regular_side != "none") & (regular_votes == 2) & ~xgb_decisive
    )
    regular_route[unanimous] = "regular_unanimous"
    regular_route[xgb_decisive] = "regular_xgb_decisive"
    regular_route[without_xgb] = "regular_without_xgb"
    scored["regular_route"] = regular_route
    return scored


def _route_score_series(scored: pd.DataFrame, route: str, side: str) -> pd.Series:
    time = pd.DatetimeIndex(pd.to_datetime(scored["decision_time"], utc=True))
    if not time.is_monotonic_increasing or not time.is_unique:
        raise ValueError("policy scores need unique increasing decision times")
    if route == "regular":
        mask = scored["regular_side"].eq(side).to_numpy(bool)
        values = scored["regular_score"].to_numpy(float)
    elif route == "xgb":
        mask = (
            scored["xgb_satellite_eligible"].astype(bool)
            & scored["xgb_satellite_side"].eq(side)
        ).to_numpy(bool)
        values = scored["xgb_satellite_score"].to_numpy(float)
    else:
        raise KeyError(f"unknown route: {route}")
    return pd.Series(np.where(mask, values, 0.0), index=time, name=f"{route}_{side}")


def _candidate_cutoffs(scores: pd.Series, maximum: int = 257) -> np.ndarray:
    positive = np.sort(scores.to_numpy(float)[scores.to_numpy(float) > 0.0])
    if not len(positive):
        return np.array([1.0], dtype=float)
    unique = np.unique(positive)
    if len(unique) > maximum:
        quantile = np.linspace(0.0, 1.0, maximum)
        unique = np.unique(np.quantile(unique, quantile))
    return np.append(unique, np.nextafter(unique.max(), np.inf))


def _select_score_threshold(
    scores: pd.Series,
    rate_cap: float,
) -> tuple[float, pd.DataFrame]:
    if not np.isfinite(rate_cap) or rate_cap <= 0.0:
        raise ValueError("rate cap must be finite and positive")
    observed_days = int(scores.index.normalize().nunique())
    if observed_days < 1:
        raise ValueError("threshold selection needs at least one observed day")
    rows: list[dict[str, object]] = []
    for cutoff in _candidate_cutoffs(scores):
        crossings = causal_crossings(scores, float(cutoff), REFRACTORY)
        rate = len(crossings) / observed_days
        rows.append(
            {
                "threshold": float(cutoff),
                "crossings": len(crossings),
                "observed_days": observed_days,
                "crossings_per_observed_day": rate,
                "within_rate_cap": rate <= rate_cap + 1e-12,
            }
        )
    frontier = pd.DataFrame(rows)
    eligible = frontier.loc[frontier["within_rate_cap"]]
    if eligible.empty:
        raise RuntimeError("no score cutoff satisfies the registered rate cap")
    selected_index = eligible.sort_values(
        ["crossings", "threshold"],
        ascending=[False, False],
        kind="stable",
    ).index[0]
    frontier["selected"] = frontier.index == selected_index
    return float(frontier.loc[selected_index, "threshold"]), frontier


def _thresholds_for_policy(
    scored: pd.DataFrame,
    policy: RiskCoveragePolicy,
    cache: dict[tuple[str, str, float], tuple[float, pd.DataFrame]],
) -> tuple[dict[str, float], list[pd.DataFrame]]:
    thresholds: dict[str, float] = {}
    frontiers: list[pd.DataFrame] = []
    for route, cap, output_prefix in (
        ("regular", policy.regular_side_rate_cap, "regular"),
        ("xgb", policy.xgb_side_rate_cap, "xgb"),
    ):
        for side in ("long", "short"):
            key = (route, side, float(cap))
            if key not in cache:
                threshold, frontier = _select_score_threshold(
                    _route_score_series(scored, route, side), cap
                )
                frontier = frontier.assign(route=route, side=side, rate_cap=cap)
                cache[key] = threshold, frontier
            threshold, frontier = cache[key]
            thresholds[f"{output_prefix}_{side}"] = threshold
            frontiers.append(frontier.copy())
    return thresholds, frontiers


def apply_frozen_fold_policy(
    predictions: pd.DataFrame,
    selection: FoldPolicySelection,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply frozen side cut-offs, route priority and global refractory."""
    scored = score_routes(predictions).sort_values("decision_time", kind="stable")
    scored["decision_time"] = pd.to_datetime(scored["decision_time"], utc=True)
    by_time = scored.set_index("decision_time", drop=False)
    events: list[dict[str, object]] = []
    for route, score_column, side_column, threshold_prefix, priority in (
        ("regular", "regular_score", "regular_side", "regular", 0),
        ("xgb", "xgb_satellite_score", "xgb_satellite_side", "xgb", 1),
    ):
        for side in ("long", "short"):
            threshold = float(selection.thresholds[f"{threshold_prefix}_{side}"])
            series = _route_score_series(scored, route, side)
            for timestamp in causal_crossings(series, threshold, REFRACTORY):
                row = by_time.loc[timestamp]
                if isinstance(row, pd.DataFrame):
                    raise AssertionError("decision timestamps must be unique")
                if route == "regular":
                    route_name = str(row["regular_route"])
                else:
                    route_name = "xgb_satellite"
                events.append(
                    {
                        "decision_time": timestamp,
                        "row_key": row["row_key"],
                        "selected_side": side,
                        "route": route_name,
                        "route_score": float(row[score_column]),
                        "priority": priority,
                    }
                )
    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        funnel = scored.copy()
        funnel["candidate"] = False
        funnel["accepted"] = False
        funnel["decision_reason"] = "below_threshold"
        return funnel.iloc[0:0].copy(), funnel

    event_frame = event_frame.sort_values(
        ["decision_time", "priority"], kind="stable"
    ).drop_duplicates("decision_time", keep="first")
    accepted: list[bool] = []
    previous_candidate: pd.Timestamp | None = None
    for timestamp in pd.to_datetime(event_frame["decision_time"], utc=True):
        permitted = (
            previous_candidate is None
            or timestamp - previous_candidate >= REFRACTORY
        )
        accepted.append(permitted)
        previous_candidate = timestamp
    event_frame["accepted"] = accepted
    accepted_events = event_frame.loc[event_frame["accepted"]].drop(
        columns="priority"
    )
    activations = scored.merge(
        accepted_events,
        on=["row_key", "decision_time"],
        how="inner",
        validate="one_to_one",
    ).sort_values("decision_time", kind="stable")

    candidate_keys = set(event_frame["row_key"].astype(str))
    accepted_keys = set(accepted_events["row_key"].astype(str))
    funnel = scored.copy()
    key = funnel["row_key"].astype(str)
    funnel["candidate"] = key.isin(candidate_keys)
    funnel["accepted"] = key.isin(accepted_keys)
    funnel["decision_reason"] = np.where(
        funnel["accepted"],
        "accepted",
        np.where(funnel["candidate"], "refractory_rejection", "below_threshold"),
    )
    return activations.reset_index(drop=True), funnel.reset_index(drop=True)


def _ledger_metrics(ledger: pd.DataFrame) -> dict[str, object]:
    if ledger.empty:
        return {
            "trades": 0,
            "long_trades": 0,
            "short_trades": 0,
            "net_return": 0.0,
            "long_net_return": 0.0,
            "short_net_return": 0.0,
            "xgb_solo_trades": 0,
            "xgb_solo_net_return": 0.0,
        }
    side = ledger["direction"].astype(str)
    net = pd.to_numeric(ledger["net_return"], errors="raise")
    solo = ledger["route"].eq("xgb_satellite")
    return {
        "trades": len(ledger),
        "long_trades": int(side.eq("long").sum()),
        "short_trades": int(side.eq("short").sum()),
        "net_return": float(net.sum()),
        "long_net_return": float(net.loc[side.eq("long")].sum()),
        "short_net_return": float(net.loc[side.eq("short")].sum()),
        "xgb_solo_trades": int(solo.sum()),
        "xgb_solo_net_return": float(net.loc[solo].sum()),
    }


def select_fold_policy(
    policy_predictions: pd.DataFrame,
    economic_paths: pd.DataFrame,
    fold_id: int,
) -> FoldPolicySelection:
    """Select one policy using only the fold's dedicated policy role."""
    if set(policy_predictions.get("source_role", pd.Series(dtype=str))) != {
        "policy_selection"
    }:
        raise ValueError("fold policy selection accepts policy_selection rows only")
    if set(pd.to_numeric(policy_predictions["fold_id"], errors="raise")) != {
        fold_id
    }:
        raise ValueError("policy predictions do not match the requested fold")
    scored = score_routes(policy_predictions).sort_values(
        "decision_time", kind="stable"
    )
    threshold_cache: dict[
        tuple[str, str, float], tuple[float, pd.DataFrame]
    ] = {}
    grid_rows: list[dict[str, object]] = []
    thresholds_by_policy: dict[RiskCoveragePolicy, dict[str, float]] = {}
    frontier_frames: list[pd.DataFrame] = []
    for policy in POLICY_GRID:
        thresholds, frontiers = _thresholds_for_policy(
            scored, policy, threshold_cache
        )
        thresholds_by_policy[policy] = thresholds
        for frontier in frontiers:
            frontier_frames.append(
                frontier.assign(
                    regular_side_rate_cap=policy.regular_side_rate_cap,
                    xgb_side_rate_cap=policy.xgb_side_rate_cap,
                )
            )
        provisional = FoldPolicySelection(
            fold_id=fold_id,
            policy=policy,
            thresholds=thresholds,
            selection_passed=False,
            source_role="policy_selection",
            policy_grid=pd.DataFrame(),
            threshold_frontier=pd.DataFrame(),
        )
        activations, _ = apply_frozen_fold_policy(scored, provisional)
        ledger = replay_selected_paths(activations, economic_paths)
        metrics = _ledger_metrics(ledger)
        qualifies = bool(
            metrics["trades"] >= 12
            and metrics["long_trades"] >= 3
            and metrics["short_trades"] >= 3
            and metrics["net_return"] > 0.0
            and metrics["long_net_return"] >= 0.0
            and metrics["short_net_return"] >= 0.0
            and metrics["xgb_solo_net_return"] >= 0.0
        )
        grid_rows.append(
            {
                "fold_id": fold_id,
                "regular_side_rate_cap": policy.regular_side_rate_cap,
                "xgb_side_rate_cap": policy.xgb_side_rate_cap,
                **thresholds,
                **metrics,
                "qualifies": qualifies,
            }
        )
    grid = pd.DataFrame(grid_rows)
    grid["selected"] = False
    qualified = grid.loc[grid["qualifies"]]
    selection_passed = not qualified.empty
    if selection_passed:
        selected_index = qualified.sort_values(
            [
                "trades",
                "net_return",
                "regular_side_rate_cap",
                "xgb_side_rate_cap",
            ],
            ascending=[False, False, True, True],
            kind="stable",
        ).index[0]
    else:
        selected_index = grid.sort_values(
            ["regular_side_rate_cap", "xgb_side_rate_cap"], kind="stable"
        ).index[0]
    grid.loc[selected_index, "selected"] = True
    selected_row = grid.loc[selected_index]
    selected_policy = RiskCoveragePolicy(
        float(selected_row["regular_side_rate_cap"]),
        float(selected_row["xgb_side_rate_cap"]),
    )
    frontier = pd.concat(frontier_frames, ignore_index=True).drop_duplicates(
        ["route", "side", "rate_cap", "threshold"], keep="first"
    )
    return FoldPolicySelection(
        fold_id=fold_id,
        policy=selected_policy,
        thresholds=thresholds_by_policy[selected_policy],
        selection_passed=selection_passed,
        source_role="policy_selection",
        policy_grid=grid,
        threshold_frontier=frontier,
    )


def _positive_fold_count(
    ledger: pd.DataFrame,
    mask: pd.Series | np.ndarray,
) -> int:
    selected = ledger.loc[np.asarray(mask, dtype=bool)]
    if selected.empty:
        return 0
    by_fold = selected.groupby("fold_id", sort=True)["net_return"].sum()
    return int(by_fold.gt(0.0).sum())


def evaluate_development_ledger(
    ledger: pd.DataFrame,
    fold_selections: list[FoldPolicySelection],
) -> dict[str, object]:
    """Apply the preregistered 87/132, side, fold and XGBoost gates."""
    metrics = _ledger_metrics(ledger)
    if ledger.empty:
        direction = pd.Series(dtype=str)
        solo = pd.Series(dtype=bool)
        total_mask = np.zeros(0, dtype=bool)
    else:
        direction = ledger["direction"].astype(str)
        solo = ledger["route"].eq("xgb_satellite")
        total_mask = np.ones(len(ledger), dtype=bool)
    total_positive_folds = _positive_fold_count(ledger, total_mask)
    long_positive_folds = _positive_fold_count(ledger, direction.eq("long"))
    short_positive_folds = _positive_fold_count(ledger, direction.eq("short"))
    xgb_positive_folds = _positive_fold_count(ledger, solo)
    xgb_folds = int(ledger.loc[solo, "fold_id"].nunique()) if len(ledger) else 0
    selection_passes = sum(selection.selection_passed for selection in fold_selections)
    quality = bool(
        metrics["trades"] >= 87
        and metrics["long_trades"] >= 15
        and metrics["short_trades"] >= 15
        and metrics["net_return"] > 0.0
        and metrics["long_net_return"] > 0.0
        and metrics["short_net_return"] > 0.0
        and total_positive_folds >= 3
        and long_positive_folds >= 3
        and short_positive_folds >= 3
        and metrics["xgb_solo_trades"] >= 10
        and metrics["xgb_solo_net_return"] > 0.0
        and xgb_folds >= 3
        and xgb_positive_folds >= 3
        and selection_passes >= 4
    )
    development_pass = bool(quality and metrics["trades"] >= 132)
    if development_pass:
        decision = "development_pass"
    elif quality:
        decision = "development_non_regression_only"
    else:
        decision = "development_fail"
    return {
        **metrics,
        "total_positive_folds": total_positive_folds,
        "long_positive_folds": long_positive_folds,
        "short_positive_folds": short_positive_folds,
        "xgb_solo_folds": xgb_folds,
        "xgb_solo_positive_folds": xgb_positive_folds,
        "fold_policy_selections_passed": selection_passes,
        "development_non_regression": quality,
        "development_pass": development_pass,
        "decision": decision,
    }


def xgb_satellite_ablation(ledger: pd.DataFrame) -> dict[str, object]:
    """Report the exact incremental contribution of the disjoint solo route."""
    if ledger.empty:
        return {
            "satellite_trades": 0,
            "without_satellite_trades": 0,
            "incremental_net_return": 0.0,
            "without_satellite_net_return": 0.0,
            "full_net_return": 0.0,
        }
    net = pd.to_numeric(ledger["net_return"], errors="raise")
    satellite = ledger["route"].eq("xgb_satellite")
    return {
        "satellite_trades": int(satellite.sum()),
        "without_satellite_trades": int((~satellite).sum()),
        "incremental_net_return": float(net.loc[satellite].sum()),
        "without_satellite_net_return": float(net.loc[~satellite].sum()),
        "full_net_return": float(net.sum()),
    }


__all__ = [
    "POLICY_GRID",
    "FoldPolicySelection",
    "RiskCoveragePolicy",
    "apply_frozen_fold_policy",
    "evaluate_development_ledger",
    "score_routes",
    "select_fold_policy",
    "xgb_satellite_ablation",
]
