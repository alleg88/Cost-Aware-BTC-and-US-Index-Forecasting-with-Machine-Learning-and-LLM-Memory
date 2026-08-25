"""Run the registered dev-only causal outcome-EV Fast-T2 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.channel_window_validation import (
    PurgedFold,
    effective_sample_size,
    expanding_purged_folds,
    interval_uniqueness,
)
from evaluation.fast_t2_action_policy import (
    first_crossing_entries,
    replay_entry_capacity,
)
from evaluation.fast_t2_causal_ev_policy import (
    expected_net_r,
    first_positive_ev_entries,
)
from experiments.fast_t2_causal_ev_dataset import annotate_causal_ev_rows
from experiments.fast_t2_causal_ev_models import (
    EV_MODEL_NAMES,
    TEMPERATURE_GRID,
    apply_temperature,
    fit_predict_causal_ev_model,
    select_temperature,
)
from experiments.fast_t2_entry_dataset import ENTRY_FEATURE_COLUMNS
from experiments.fast_t2_study import DEV_END, DEV_START
from experiments.run_fast_t2_entry_policy import (
    MINUTE_SOURCE,
    inner_episode_purged_indices,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIR = CODE_ROOT / "experiments" / "cache" / "fast_t2_entry_policy" / "dev"
OUT = CODE_ROOT / "experiments" / "cache" / "fast_t2_causal_ev" / "dev"
MODELS = EV_MODEL_NAMES
POLICIES: tuple[tuple[str, float, bool], ...] = (
    ("primary_25bps", 25.0, False),
    ("sensitivity_40bps", 40.0, False),
    ("target_cancel_25bps", 25.0, True),
)
MINIMUM_TRADES_PER_DAY = 1.0


def _json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_models(models: Iterable[str]) -> tuple[str, ...]:
    requested = tuple(dict.fromkeys(models))
    unknown = sorted(set(requested).difference(MODELS))
    if unknown:
        raise ValueError(f"unsupported causal EV models: {unknown}")
    if not requested:
        raise ValueError("at least one causal EV model is required")
    return tuple(name for name in MODELS if name in requested)


def build_protocol(
    entry_protocol: dict[str, object],
    manifest: dict[str, object],
) -> dict[str, object]:
    """Build the path-independent, fixed-zero-EV Notebook H protocol."""
    if bool(entry_protocol.get("forward_or_lockbox_loaded")):
        raise ValueError("forward and lockbox must remain sealed")
    if entry_protocol.get("period_end_exclusive") != DEV_END.isoformat():
        raise ValueError("Notebook H must remain dev-only")
    if pd.Timestamp(manifest["max_loaded_timestamp"]) >= DEV_END:
        raise ValueError("source manifest reaches the sealed period")
    payload: dict[str, object] = {
        "stage": "development",
        "source_entry_protocol_hash": entry_protocol["protocol_hash"],
        "source_dataset_hash": manifest["dataset_hash"],
        "source_decision_ledger_hash": manifest["decision_ledger_hash"],
        "period_start": entry_protocol["period_start"],
        "period_end_exclusive": entry_protocol["period_end_exclusive"],
        "models": list(MODELS),
        "model_form": "pooled long and short with side-normalised features",
        "features": list(ENTRY_FEATURE_COLUMNS),
        "outcome_classes": ["sl", "tp", "timeout"],
        "timeout_target": "gross R, clipped at training-only 1st/99th percentiles",
        "ev_formula": "P(TP)*planned_RR - P(SL) + P(timeout)*E[timeout_gross_R] - planned_cost_R",
        "planned_round_trip_cost_bps": 10.0,
        "primary_min_risk_bps": 25.0,
        "sensitivity_min_risk_bps": 40.0,
        "window_cancellation": "permanent after completed 1m stop touch",
        "target_touch_cancellation": "diagnostic sensitivity only",
        "entry_rule": "strict predicted EV > 0; no threshold tuning",
        "calibration": {
            "method": "train-only scalar temperature",
            "grid": list(TEMPERATURE_GRID),
            "selection": "inner chronological episode-purged weighted log loss",
        },
        "validation": "seven expanding six-month episode-purged folds",
        "uniqueness_weighting": "training labels only; no class balancing",
        "minimum_trades_per_day": MINIMUM_TRADES_PER_DAY,
        "frequency_is_admission_only": True,
        "capacity": "unlimited; no capacity selection",
        "frozen_baseline": "Notebook E LogReg",
        "catastrophic_classifier": "not fitted when the primary causal sample contains zero R<-2 labels",
        "forward_or_lockbox_loaded": False,
    }
    payload["protocol_hash"] = _json_hash(payload)
    return payload


def _write_protocol(
    output_dir: Path,
    entry_protocol: dict[str, object],
    manifest: dict[str, object],
    *,
    rebuild: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = build_protocol(entry_protocol, manifest)
    path = output_dir / "protocol.json"
    if path.exists() and not rebuild:
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored.get("protocol_hash") != expected["protocol_hash"]:
            raise ValueError("Notebook H protocol hash mismatch; use --rebuild")
        return stored
    path.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return expected


def _load_entry_inputs() -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    required = {
        "run_state.json",
        "protocol.json",
        "dataset_manifest.json",
        "entry_decisions.parquet",
        "entry_freeze.json",
    }
    missing = sorted(name for name in required if not (ENTRY_DIR / name).exists())
    if missing:
        raise FileNotFoundError(f"frozen Notebook E artifacts missing: {missing}")
    state = json.loads((ENTRY_DIR / "run_state.json").read_text(encoding="utf-8"))
    protocol = json.loads((ENTRY_DIR / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (ENTRY_DIR / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    if state.get("status") != "complete" or not state.get("consolidated"):
        raise ValueError("Notebook H requires complete frozen Notebook E artifacts")
    if bool(state.get("forward_or_lockbox_loaded")) or bool(
        protocol.get("forward_or_lockbox_loaded")
    ):
        raise ValueError("forward and lockbox must remain sealed")
    decisions = pd.read_parquet(ENTRY_DIR / "entry_decisions.parquet")
    decisions = decisions.sort_values(
        ["decision_time", "decision_id"], kind="stable"
    ).reset_index(drop=True)
    if _json_hash(decisions["decision_id"].astype(str).tolist()) != manifest.get(
        "decision_ledger_hash"
    ):
        raise ValueError("Notebook E decision ledger hash mismatch")
    for column in ("t2_time", "decision_time", "label_start", "label_end"):
        decisions[column] = pd.to_datetime(decisions[column], utc=True, errors="raise")
    if decisions["label_end"].max() >= DEV_END:
        raise AssertionError("Notebook H labels reach the sealed post-dev period")
    return decisions, protocol, manifest


def _load_dev_minutes() -> pd.DataFrame:
    minute = pd.read_parquet(
        MINUTE_SOURCE,
        columns=["high", "low"],
        filters=[("timestamp", ">=", DEV_START), ("timestamp", "<", DEV_END)],
    ).sort_index(kind="stable")
    minute = minute[(minute.index >= DEV_START) & (minute.index < DEV_END)]
    if minute.index.has_duplicates:
        raise ValueError("native one-minute source has duplicate timestamps")
    if len(minute) and minute.index.max() >= DEV_END:
        raise AssertionError("Notebook H loaded data at or beyond the dev boundary")
    return minute


def _prepare_causal_dataset(
    decisions: pd.DataFrame,
    *,
    output_dir: Path,
    protocol: dict[str, object],
    source_manifest: dict[str, object],
    rebuild: bool,
) -> tuple[pd.DataFrame, dict[str, object]]:
    path = output_dir / "causal_ev_decisions.parquet"
    manifest_path = output_dir / "causal_dataset_manifest.json"
    if path.exists() and manifest_path.exists() and not rebuild:
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored.get("protocol_hash") != protocol["protocol_hash"]:
            raise ValueError("cached causal dataset belongs to another protocol")
        annotated = pd.read_parquet(path)
        if _json_hash(annotated["decision_id"].astype(str).tolist()) != stored.get(
            "annotated_ledger_hash"
        ):
            raise ValueError("cached causal decision ledger hash mismatch")
        return annotated, stored

    minute = _load_dev_minutes()
    annotated = annotate_causal_ev_rows(decisions, minute)
    annotated.to_parquet(path, index=False)
    primary = annotated[annotated["primary_eligible"].astype(bool)]
    dataset_manifest: dict[str, object] = {
        "protocol_hash": protocol["protocol_hash"],
        "source_decision_ledger_hash": source_manifest["decision_ledger_hash"],
        "annotated_ledger_hash": _json_hash(
            annotated["decision_id"].astype(str).tolist()
        ),
        "rows": int(len(annotated)),
        "windows": int(annotated["window_id"].nunique()),
        "stop_cancelled_rows": int(
            annotated["stop_invalidated_before_decision"].sum()
        ),
        "stop_cancelled_windows": int(
            annotated.loc[
                annotated["stop_invalidated_before_decision"], "window_id"
            ].nunique()
        ),
        "target_pre_hit_rows": int(
            annotated["target_reached_before_decision"].sum()
        ),
        "primary_rows": int(len(primary)),
        "primary_windows": int(primary["window_id"].nunique()),
        "sensitivity_40_rows": int(
            annotated["sensitivity_40_eligible"].sum()
        ),
        "primary_catastrophic_labels": int(primary["catastrophic_loss"].sum()),
        "max_loaded_timestamp": source_manifest["max_loaded_timestamp"],
        "forward_or_lockbox_loaded": False,
    }
    manifest_path.write_text(
        json.dumps(dataset_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return annotated, dataset_manifest


def _eligible_positions(decisions: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    mask = decisions.iloc[positions]["primary_eligible"].to_numpy(dtype=bool)
    return positions[mask]


def _fit_raw_prediction(
    decisions: pd.DataFrame,
    *,
    model_name: str,
    train_index: np.ndarray,
    score_index: np.ndarray,
):
    train_index = _eligible_positions(decisions, train_index)
    score_index = _eligible_positions(decisions, score_index)
    if not len(train_index) or not len(score_index):
        raise ValueError("causal EV fit and score rows cannot be empty")
    train_episodes = set(decisions.iloc[train_index]["channel_episode_id"])
    score_episodes = set(decisions.iloc[score_index]["channel_episode_id"])
    if train_episodes.intersection(score_episodes):
        raise AssertionError("causal EV fit and score rows share episodes")
    raw_uniqueness = interval_uniqueness(decisions, train_index, normalize=False)
    weights = raw_uniqueness / raw_uniqueness.mean()
    prediction = fit_predict_causal_ev_model(
        model_name,
        decisions.iloc[train_index][list(ENTRY_FEATURE_COLUMNS)].to_numpy(float),
        decisions.iloc[train_index]["outcome_class"].to_numpy(np.int64),
        decisions.iloc[train_index]["observed_gross_r"].to_numpy(float),
        weights,
        decisions.iloc[score_index][list(ENTRY_FEATURE_COLUMNS)].to_numpy(float),
    )
    return prediction, train_index, score_index, raw_uniqueness


def _predictive_diagnostics(
    scored: pd.DataFrame,
    *,
    model_name: str,
    fold: PurgedFold,
    temperature: float,
    train_index: np.ndarray,
    inner_fit: np.ndarray,
    inner_valid: np.ndarray,
    uniqueness: np.ndarray,
) -> dict[str, object]:
    from sklearn.metrics import log_loss, roc_auc_score

    target = scored["outcome_class"].to_numpy(np.int64)
    probabilities = scored[["p_sl", "p_tp", "p_timeout"]].to_numpy(float)
    one_hot = np.eye(3, dtype=float)[target]
    timeout = target == 2
    try:
        macro_auc = float(
            roc_auc_score(target, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        macro_auc = np.nan
    return {
        "model": model_name,
        "fold_id": fold.fold_id,
        "train_rows": int(len(train_index)),
        "train_episodes": int(
            scored.attrs.get("train_episodes", 0)
        ),
        "inner_fit_rows": int(len(inner_fit)),
        "inner_validation_rows": int(len(inner_valid)),
        "validation_rows": int(len(scored)),
        "validation_windows": int(scored["window_id"].nunique()),
        "validation_sl": int((target == 0).sum()),
        "validation_tp": int((target == 1).sum()),
        "validation_timeout": int(timeout.sum()),
        "temperature": float(temperature),
        "weighted_train_uniqueness_mean": float(uniqueness.mean()),
        "effective_train_rows": float(effective_sample_size(uniqueness)),
        "multiclass_log_loss": float(log_loss(target, probabilities, labels=[0, 1, 2])),
        "multiclass_brier": float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))),
        "macro_ovr_auc_diagnostic_only": macro_auc,
        "timeout_mae_gross_r": float(
            np.mean(
                np.abs(
                    scored.loc[timeout, "observed_gross_r"]
                    - scored.loc[timeout, "timeout_gross_prediction"]
                )
            )
        )
        if timeout.any()
        else np.nan,
        "ev_rank_ic_diagnostic_only": float(
            scored["ev_score"].corr(scored["r_net"], method="spearman")
        ),
        "catastrophic_validation_labels": int(scored["catastrophic_loss"].sum()),
        "validation_start": fold.valid_start.isoformat(),
        "validation_end_exclusive": fold.valid_end.isoformat(),
    }


def score_outer_fold(
    decisions: pd.DataFrame,
    fold: PurgedFold,
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Calibrate on inner history and score one untouched outer fold."""
    inner_cut = fold.valid_start - pd.DateOffset(months=6)
    inner_fit, inner_valid = inner_episode_purged_indices(
        decisions,
        fold.train,
        inner_cut=inner_cut,
        outer_valid_start=fold.valid_start,
    )
    inner_prediction, inner_fit, inner_valid, _ = _fit_raw_prediction(
        decisions,
        model_name=model_name,
        train_index=inner_fit,
        score_index=inner_valid,
    )
    calibration_uniqueness = interval_uniqueness(
        decisions, inner_valid, normalize=False
    )
    temperature = select_temperature(
        inner_prediction.outcome_probabilities,
        decisions.iloc[inner_valid]["outcome_class"].to_numpy(np.int64),
        calibration_uniqueness,
    )

    prediction, train_index, valid_index, uniqueness = _fit_raw_prediction(
        decisions,
        model_name=model_name,
        train_index=fold.train,
        score_index=fold.valid,
    )
    probabilities = apply_temperature(
        prediction.outcome_probabilities, temperature
    )
    scored = decisions.iloc[valid_index].copy()
    scored.insert(0, "model", model_name)
    scored.insert(1, "fold_id", fold.fold_id)
    scored["p_sl"] = probabilities[:, 0]
    scored["p_tp"] = probabilities[:, 1]
    scored["p_timeout"] = probabilities[:, 2]
    scored["timeout_gross_prediction"] = prediction.timeout_gross_r
    scored["temperature"] = temperature
    scored["ev_score"] = expected_net_r(
        probabilities,
        planned_tp_gross_r=scored["planned_tp_gross_r"].to_numpy(float),
        timeout_gross_r=prediction.timeout_gross_r,
        planned_cost_r=scored["planned_cost_r_10bps"].to_numpy(float),
    )
    scored["score"] = scored["ev_score"]
    scored.attrs["train_episodes"] = int(
        decisions.iloc[train_index]["channel_episode_id"].nunique()
    )

    entry_frames: list[pd.DataFrame] = []
    action_frames: list[pd.DataFrame] = []
    for policy, min_risk, cancel_target in POLICIES:
        entries, actions = first_positive_ev_entries(
            scored,
            min_risk_bps=min_risk,
            cancel_after_target=cancel_target,
        )
        for frame in (entries, actions):
            frame["policy"] = policy
            frame["model"] = model_name
            frame["fold_id"] = fold.fold_id
        entry_frames.append(entries)
        action_frames.append(actions)
    diagnostics = _predictive_diagnostics(
        scored,
        model_name=model_name,
        fold=fold,
        temperature=temperature,
        train_index=train_index,
        inner_fit=inner_fit,
        inner_valid=inner_valid,
        uniqueness=uniqueness,
    )
    diagnostics.update(
        {
            "train_episodes": int(
                decisions.iloc[train_index]["channel_episode_id"].nunique()
            ),
            "validation_episodes": int(
                decisions.iloc[valid_index]["channel_episode_id"].nunique()
            ),
            "timeout_target_low": prediction.timeout_target_low,
            "timeout_target_high": prediction.timeout_target_high,
            "episode_overlap": int(
                len(
                    set(decisions.iloc[train_index]["channel_episode_id"]).intersection(
                        decisions.iloc[valid_index]["channel_episode_id"]
                    )
                )
            ),
        }
    )
    if diagnostics["episode_overlap"]:
        raise AssertionError("outer causal EV fold shares episodes")
    return (
        scored.reset_index(drop=True),
        pd.concat(entry_frames, ignore_index=True),
        pd.concat(action_frames, ignore_index=True),
        diagnostics,
    )


def _artifact_paths(output_dir: Path, fold_id: str, model_name: str) -> dict[str, Path]:
    stem = f"{fold_id}_{model_name}"
    fold_dir = output_dir / "folds"
    return {
        "scores": fold_dir / f"scores_{stem}.parquet",
        "entries": fold_dir / f"entries_{stem}.parquet",
        "actions": fold_dir / f"actions_{stem}.parquet",
        "audit": fold_dir / f"audit_{stem}.json",
    }


def _artifacts_exist(paths: dict[str, Path]) -> bool:
    return all(path.exists() for path in paths.values())


def _evaluation_days(folds: list[PurgedFold]) -> int:
    return int(sum((fold.valid_end - fold.valid_start).days for fold in folds))


def _summarise_orders(
    orders: pd.DataFrame,
    *,
    evaluation_days: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    replay = replay_entry_capacity(
        orders, capacity=None, evaluation_days=evaluation_days
    )
    return replay.orders, replay.metrics


def _geometry_ablation(
    decisions: pd.DataFrame,
    *,
    evaluation_days: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    score_path = ENTRY_DIR / "oof_decisions_logreg.parquet"
    frontier_path = ENTRY_DIR / "threshold_frontier_logreg.csv"
    order_path = ENTRY_DIR / "orders_logreg_unlimited.parquet"
    for path in (score_path, frontier_path, order_path):
        if not path.exists():
            raise FileNotFoundError(path)
    frozen_orders = pd.read_parquet(order_path)
    _, frozen_metrics = _summarise_orders(
        frozen_orders, evaluation_days=evaluation_days
    )
    rows: list[dict[str, object]] = [
        {
            "step": "Frozen Notebook E LogReg",
            "minimum_risk_bps": np.nan,
            "cancel_after_stop": False,
            "cancel_after_target": False,
            **frozen_metrics,
        }
    ]
    scores = pd.read_parquet(score_path)
    extra = decisions[
        [
            "decision_id",
            "stop_invalidated_before_decision",
            "target_reached_before_decision",
            "distance_to_stop_bps",
            "distance_to_target_bps",
            "outcome_class",
        ]
    ]
    scores = scores.merge(extra, on="decision_id", how="left", validate="one_to_one")
    thresholds = pd.read_csv(frontier_path).groupby("fold_id", sort=False)[
        "outer_threshold"
    ].first()
    stages = (
        ("Causal stop cancellation", None, False),
        ("Causal stop + minimum risk 25 bps", 25.0, False),
        ("Causal stop + minimum risk 40 bps", 40.0, False),
        ("Target-touch cancellation sensitivity", 25.0, True),
    )
    last_metrics = frozen_metrics
    for label, min_risk, cancel_target in stages:
        ledgers: list[pd.DataFrame] = []
        for fold_id, fold_scores in scores.groupby("fold_id", sort=False):
            work = fold_scores.copy()
            eligible = ~work["stop_invalidated_before_decision"].astype(bool)
            if min_risk is not None:
                eligible &= work["distance_to_stop_bps"].ge(min_risk)
                eligible &= work["distance_to_target_bps"].gt(0.0)
                eligible &= work["outcome_class"].ge(0)
                eligible &= work["filled"].astype(bool)
            if cancel_target:
                eligible &= ~work["target_reached_before_decision"].astype(bool)
            work.loc[~eligible, "score"] = -np.inf
            entries, _ = first_crossing_entries(
                work, float(thresholds.loc[fold_id])
            )
            ledgers.append(entries)
        ledger = pd.concat(ledgers, ignore_index=True)
        _, last_metrics = _summarise_orders(
            ledger, evaluation_days=evaluation_days
        )
        rows.append(
            {
                "step": label,
                "minimum_risk_bps": min_risk,
                "cancel_after_stop": True,
                "cancel_after_target": cancel_target,
                **last_metrics,
            }
        )
    table = pd.DataFrame(rows)
    table["frequency_eligible"] = table["trades_per_day"].ge(
        MINIMUM_TRADES_PER_DAY
    )
    return table, frozen_metrics


def _consolidate(
    decisions: pd.DataFrame,
    *,
    output_dir: Path,
    models: tuple[str, ...],
    folds: list[PurgedFold],
    protocol: dict[str, object],
    dataset_manifest: dict[str, object],
) -> None:
    days = _evaluation_days(folds)
    summaries: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    primary_ledgers: dict[str, pd.DataFrame] = {}
    model_labels = {
        "catboost": "CatBoost pooled",
        "xgboost": "XGBoost pooled",
    }
    for model_name in models:
        scores = pd.concat(
            [
                pd.read_parquet(_artifact_paths(output_dir, fold.fold_id, model_name)["scores"])
                for fold in folds
            ],
            ignore_index=True,
        )
        entries = pd.concat(
            [
                pd.read_parquet(_artifact_paths(output_dir, fold.fold_id, model_name)["entries"])
                for fold in folds
            ],
            ignore_index=True,
        )
        actions = pd.concat(
            [
                pd.read_parquet(_artifact_paths(output_dir, fold.fold_id, model_name)["actions"])
                for fold in folds
            ],
            ignore_index=True,
        )
        for fold in folds:
            diagnostic_rows.append(
                json.loads(
                    _artifact_paths(output_dir, fold.fold_id, model_name)["audit"].read_text(
                        encoding="utf-8"
                    )
                )
            )
        scores.to_parquet(output_dir / f"oof_scores_{model_name}.parquet", index=False)
        entries.to_parquet(output_dir / f"oof_entries_{model_name}.parquet", index=False)
        actions.to_parquet(output_dir / f"oof_actions_{model_name}.parquet", index=False)
        for policy, _, _ in POLICIES:
            ledger = entries[entries["policy"].eq(policy)].copy()
            orders, metrics = _summarise_orders(ledger, evaluation_days=days)
            orders.to_parquet(
                output_dir / f"orders_{model_name}_{policy}.parquet", index=False
            )
            filled = orders[orders["filled"].astype(bool)]
            summaries.append(
                {
                    "model": model_labels[model_name],
                    "model_key": model_name,
                    "policy": policy,
                    "strict_positive_ev": True,
                    "mean_predicted_ev": float(filled["ev_score"].mean())
                    if len(filled)
                    else np.nan,
                    "catastrophic_trades": int(filled["catastrophic_loss"].sum())
                    if len(filled)
                    else 0,
                    "target_pre_hit_trades": int(
                        filled["target_reached_before_decision"].sum()
                    )
                    if len(filled)
                    else 0,
                    **metrics,
                }
            )
            if policy == "primary_25bps":
                primary_ledgers[model_name] = orders

    geometry, frozen_metrics = _geometry_ablation(
        decisions, evaluation_days=days
    )
    geometry.to_csv(output_dir / "geometry_ablation.csv", index=False)
    summaries.insert(
        0,
        {
            "model": "Frozen Notebook E LogReg",
            "model_key": "notebook_e_logreg",
            "policy": "frozen",
            "strict_positive_ev": False,
            "mean_predicted_ev": np.nan,
            "catastrophic_trades": np.nan,
            "target_pre_hit_trades": np.nan,
            **frozen_metrics,
        },
    )
    summary = pd.DataFrame(summaries)
    summary["frequency_eligible"] = summary["trades_per_day"].ge(
        MINIMUM_TRADES_PER_DAY
    )
    summary["two_sided_support"] = summary["long_trades"].gt(0) & summary[
        "short_trades"
    ].gt(0)
    baseline_mean = float(frozen_metrics["mean_net_r"])
    summary["improvement_vs_frozen_logreg_r"] = summary["mean_net_r"] - baseline_mean
    summary["economic_success"] = (
        summary["policy"].eq("primary_25bps")
        & summary["frequency_eligible"]
        & summary["mean_net_r"].ge(0.05)
        & summary["bootstrap_low"].gt(0.0)
        & summary["mean_net_r"].gt(baseline_mean)
    )
    summary.to_csv(output_dir / "causal_ev_policy_summary.csv", index=False)
    diagnostics = pd.DataFrame(diagnostic_rows)
    diagnostics.to_csv(output_dir / "predictive_diagnostics.csv", index=False)
    diagnostics.to_csv(output_dir / "fold_audit.csv", index=False)

    primary = summary[summary["policy"].eq("primary_25bps")].copy()
    eligible = primary[primary["frequency_eligible"]]
    ranked = (eligible if not eligible.empty else primary).sort_values(
        ["mean_net_r", "bootstrap_low", "model_key"],
        ascending=[False, False, True],
        kind="stable",
    )
    leader = ranked.iloc[0]
    leader_key = str(leader["model_key"])
    leader_orders = primary_ledgers[leader_key]
    leader_orders.to_parquet(output_dir / "development_leader_orders.parquet", index=False)
    result = {
        "protocol_hash": protocol["protocol_hash"],
        "development_leader": leader_key,
        "development_leader_is_not_forward_promotion": True,
        "primary_policy": "first strict predicted EV > 0, risk >=25 bps",
        "sensitivity_policy": "same scores, risk >=40 bps",
        "target_cancel_policy": "diagnostic only",
        "minimum_trades_per_day": MINIMUM_TRADES_PER_DAY,
        "frequency_is_admission_only": True,
        "leader_filled_trades": int(leader["filled_trades"]),
        "leader_trades_per_day": float(leader["trades_per_day"]),
        "leader_mean_net_r": float(leader["mean_net_r"]),
        "leader_bootstrap_low": float(leader["bootstrap_low"]),
        "leader_bootstrap_high": float(leader["bootstrap_high"]),
        "frozen_logreg_mean_net_r": baseline_mean,
        "leader_improvement_vs_frozen_logreg_r": float(
            leader["mean_net_r"] - baseline_mean
        ),
        "economic_success": bool(leader["economic_success"]),
        "status": "eligible_for_registered_forward" if bool(leader["economic_success"]) else "development_failure",
        "primary_catastrophic_labels": int(
            dataset_manifest["primary_catastrophic_labels"]
        ),
        "catastrophic_classifier_fitted": False,
        "tail_risk_proxy": "P(SL); separate catastrophe head is unidentifiable after causal risk gate",
        "leader_ledger_hash": _json_hash(
            sorted(leader_orders["decision_id"].astype(str).tolist())
        ),
        "primary_trials": len(models),
        "removed_from_active_path": [
            "learned_early_exit",
            "split_long_short_catboost",
            "gru",
            "lstm",
            "hard_rr_gate",
            "capacity_tuning",
            "accuracy_auc_win_rate_model_selection",
            "forced_threshold_reduction_for_frequency",
        ],
        "forward_or_lockbox_loaded": False,
    }
    (output_dir / "causal_ev_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _write_state(output_dir: Path, **values: object) -> None:
    (output_dir / "run_state.json").write_text(
        json.dumps(values, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def run(
    *,
    models: Iterable[str] = MODELS,
    limit_folds: int | None = None,
    rebuild: bool = False,
    output_dir: Path = OUT,
) -> None:
    """Run resumable fold/model cells and consolidate only the full matrix."""
    model_names = _normalise_models(models)
    if limit_folds is not None and limit_folds < 1:
        raise ValueError("limit_folds must be positive")
    output_dir = Path(output_dir)
    decisions, entry_protocol, entry_manifest = _load_entry_inputs()
    protocol = _write_protocol(
        output_dir,
        entry_protocol,
        entry_manifest,
        rebuild=rebuild,
    )
    decisions, dataset_manifest = _prepare_causal_dataset(
        decisions,
        output_dir=output_dir,
        protocol=protocol,
        source_manifest=entry_manifest,
        rebuild=rebuild,
    )
    folds = expanding_purged_folds(decisions)
    if len(folds) != 7:
        raise AssertionError(f"expected seven development folds, found {len(folds)}")
    selected_folds = folds[:limit_folds] if limit_folds is not None else folds
    (output_dir / "folds").mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    _write_state(
        output_dir,
        status="running",
        active="folds",
        completed=completed,
        consolidated=False,
        protocol_hash=protocol["protocol_hash"],
        forward_or_lockbox_loaded=False,
    )
    for fold in selected_folds:
        for model_name in model_names:
            key = f"{fold.fold_id}:{model_name}"
            paths = _artifact_paths(output_dir, fold.fold_id, model_name)
            if _artifacts_exist(paths) and not rebuild:
                completed.append(key)
                print(f"resume {key}", flush=True)
                continue
            _write_state(
                output_dir,
                status="running",
                active=key,
                completed=completed,
                consolidated=False,
                protocol_hash=protocol["protocol_hash"],
                forward_or_lockbox_loaded=False,
            )
            print(f"fit {key}", flush=True)
            scores, entries, actions, audit = score_outer_fold(
                decisions, fold, model_name
            )
            scores.to_parquet(paths["scores"], index=False)
            entries.to_parquet(paths["entries"], index=False)
            actions.to_parquet(paths["actions"], index=False)
            paths["audit"].write_text(
                json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            completed.append(key)

    full_matrix = (
        limit_folds is None
        and model_names == MODELS
        and all(
            _artifacts_exist(_artifact_paths(output_dir, fold.fold_id, model_name))
            for fold in folds
            for model_name in MODELS
        )
    )
    if full_matrix:
        _consolidate(
            decisions,
            output_dir=output_dir,
            models=model_names,
            folds=folds,
            protocol=protocol,
            dataset_manifest=dataset_manifest,
        )
    _write_state(
        output_dir,
        status="complete",
        active=None,
        completed=completed,
        expected=int(len(selected_folds) * len(model_names)),
        consolidated=full_matrix,
        protocol_hash=protocol["protocol_hash"],
        max_loaded_timestamp=entry_manifest["max_loaded_timestamp"],
        forward_or_lockbox_loaded=False,
    )
    print(
        f"complete: {len(completed)}/{len(selected_folds) * len(model_names)} "
        f"fold-model cells; consolidated={full_matrix}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--limit-folds", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()
    run(
        models=tuple(args.models),
        limit_folds=args.limit_folds,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
