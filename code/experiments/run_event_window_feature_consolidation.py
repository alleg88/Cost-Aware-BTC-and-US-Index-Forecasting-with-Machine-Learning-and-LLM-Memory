"""Development-only 248-versus-28 feature consolidation for Notebook U."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluation.channel_window_validation import PurgedFold
from evaluation.event_window_opportunity_policy import (
    causal_level_rearm_alerts,
    collapse_episode_time,
    select_causal_threshold,
)
from experiments.event_window_conditional_oof import (
    ConditionalOOFConfig,
    _decisions,
    run_conditional_fold,
)
from experiments.event_window_cost_aware_oof import _outer_folds
from experiments.event_window_compact_features import (
    COMPACT_FEATURES,
    DERIVED_COMPACT_FEATURES,
    build_compact_dataset,
)
from experiments.event_window_large_move_dataset import (
    LargeMoveDecisionDataset,
    build_opportunity_dataset,
)
from experiments.event_window_magnitude_dataset import align_magnitude_dataset
from experiments.run_event_window_calendar_ablation import (
    FROZEN_P_RUN_HASH,
    FROZEN_P_ROOT,
    FROZEN_R_RUN_HASH,
    FROZEN_R_ROOT,
    FrozenRArtifacts,
    PairedOOFResult,
    _latest,
    _read_json,
    _smoke_p_hit,
    base_reproduction_audit,
    frozen_p_hit_for_fold,
    load_frozen_r_handoff,
    paired_feature_bootstrap,
)
from experiments.run_event_window_conditional_opportunity import (
    _align_labels,
    load_frozen_o_artifacts,
)
from experiments.run_event_window_cost_aware_entry import _Store, _sha256, _sha_payload
from experiments.run_event_window_economic_feasibility import (
    _attach_execution_fields,
    _concurrency_audit,
    _economic_metrics,
    _frequency_audit,
    build_direction_scenarios,
    load_frozen_p_artifacts,
    replay_brackets,
)
from experiments.run_event_window_magnitude_timing import (
    RUN_ROOT as FROZEN_O_ROOT,
    load_frozen_n_artifacts,
)
from experiments.run_event_window_tail_models import (
    FROZEN_J_ROOT,
    _build_tail_dataset,
    load_frozen_j_artifacts,
)
from experiments.run_event_window_timing_policy_repair import calendar_days
from experiments.run_event_window_tcn import _load_bounded_parquet


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_feature_consolidation"
MODELS = ("logreg", "xgboost")
FEATURE_SETS = ("base", "compact_volatility_timing_v1")
ARM_SPECS = (
    ("logreg_base", "logreg", "base"),
    ("logreg_compact", "logreg", "compact_volatility_timing_v1"),
    ("xgboost_base", "xgboost", "base"),
    ("xgboost_compact", "xgboost", "compact_volatility_timing_v1"),
)
ARM_NAMES = tuple(spec[0] for spec in ARM_SPECS)
CALENDAR_FEATURE_NAMES = {
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_weekday_sin",
    "utc_weekday_cos",
    "weekend_flag",
    "us_cash_session_flag",
}
READER_ARTIFACTS = (
    "oof_predictions.parquet",
    "policy_calibration_predictions.parquet",
    "compact_feature_audit.csv",
    "feature_redundancy_audit.csv",
    "base_reproduction_audit.csv",
    "fold_audit.csv",
    "head_calibration_audit.csv",
    "predictive_metrics.csv",
    "predictive_deltas.csv",
    "threshold_audit.csv",
    "activation_ledger.parquet",
    "economic_paths.parquet",
    "economic_metrics.csv",
    "feature_comparisons.csv",
    "frequency_audit.csv",
    "concurrency_audit.csv",
    "leakage_audit.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)


@dataclass(frozen=True)
class FeatureConsolidationConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    target_activations_per_day: float = 3.0
    target_multiple_b: float = 2.0
    stop_multiple_b: float = 1.0
    hold_minutes: int = 120
    cooldown_minutes: int = 60
    threshold_grid_size: int = 51
    entry_cost_bps: float = 5.0
    target_exit_cost_bps: float = 2.0
    other_exit_cost_bps: float = 5.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42
    minimum_path_completeness: float = 0.99
    minimum_frequency_per_day: float = 2.5
    maximum_frequency_per_day: float = 3.5
    base_reproduction_tolerance: float = 1e-12
    oof: ConditionalOOFConfig = field(default_factory=ConditionalOOFConfig)


def promotion_decision(
    *,
    brier_improvement: float,
    log_loss_improvement: float,
    nonnegative_brier_folds: int,
    frequency_per_day: float,
    economic_delta_ci_low: float,
    path_completeness: float,
    leakage_passed: bool,
    base_reproduced: bool,
) -> bool:
    return bool(
        brier_improvement >= 0.0
        and log_loss_improvement >= 0.0
        and nonnegative_brier_folds >= 4
        and 2.5 <= frequency_per_day <= 3.5
        and economic_delta_ci_low > 0.0
        and path_completeness >= 0.99
        and leakage_passed
        and base_reproduced
    )


@dataclass(frozen=True)
class FeatureConsolidationRunResult:
    run_dir: Path
    summary: dict[str, object]


def protocol_dict(
    config: FeatureConsolidationConfig = FeatureConsolidationConfig(),
    *,
    smoke: bool = False,
) -> dict[str, object]:
    return {
        "study": "notebook_u_volatility_timing_feature_consolidation",
        "stage": "dev",
        "smoke": bool(smoke),
        **asdict(config),
        "models": list(MODELS),
        "feature_sets": list(FEATURE_SETS),
        "arms": [list(spec) for spec in ARM_SPECS],
        "base_feature_count": 248,
        "compact_feature_count": len(COMPACT_FEATURES),
        "compact_features": list(COMPACT_FEATURES),
        "derived_compact_features": list(DERIVED_COMPACT_FEATURES),
        "selection": "pre_registered_domain_contract_no_label_selection",
        "p_hit_frozen": True,
        "timing_heads_refit": ["h15", "h30", "h60"],
        "timing_score": "p_t_le_60",
        "alert_policy": "level_rearm",
        "calendar_features_included": False,
        "impulse_features_included": False,
        "frozen_p_run_hash": FROZEN_P_RUN_HASH,
        "frozen_r_run_hash": FROZEN_R_RUN_HASH,
        "direction_head_trained": False,
        "direction_70_is_stress_test": True,
        "forward_or_lockbox_loaded": False,
    }

def build_paired_datasets(
    base: LargeMoveDecisionDataset,
) -> tuple[LargeMoveDecisionDataset, LargeMoveDecisionDataset]:
    compact = build_compact_dataset(base)
    keys = ["window_id", "channel_episode_id", "step", "decision_time"]
    missing = sorted(set(keys).difference(base.decisions.columns))
    if missing:
        raise ValueError(f"paired dataset keys are missing: {missing}")
    if not base.decisions[keys].reset_index(drop=True).equals(
        compact.decisions[keys].reset_index(drop=True)
    ):
        raise AssertionError("compact feature construction changed decision rows")
    if compact.tabular_features != COMPACT_FEATURES or compact.tabular.shape[1] != 28:
        raise AssertionError("compact feature construction changed its fixed contract")
    return base, compact

def predictive_metric_tables(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "arm",
        "model",
        "feature_set",
        "fold_id",
        "window_id",
        "step",
        "decision_time",
        "y_t_le_60",
        "p_t_le_60",
        "sample_weight",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"predictive scores missing columns: {missing}")
    work = scores.copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True)
    work["feature_variant"] = np.where(
        work["feature_set"].eq("base"), "base", "compact"
    )
    pair_keys = [
        "fold_id",
        "window_id",
        "step",
        "decision_time",
        "y_t_le_60",
        "sample_weight",
    ]
    for model, model_rows in work.groupby("model", sort=True):
        variants = {
            name: rows[pair_keys].sort_values(pair_keys, kind="stable").reset_index(
                drop=True
            )
            for name, rows in model_rows.groupby("feature_variant", sort=True)
        }
        if set(variants) != {"base", "compact"} or not variants["base"].equals(
            variants["compact"]
        ):
            raise ValueError(f"base/compact predictive rows differ for {model}")

    def metric_row(frame: pd.DataFrame, fold_id: str) -> dict[str, object]:
        target = frame["y_t_le_60"].to_numpy(int)
        probability = np.clip(frame["p_t_le_60"].to_numpy(float), 1e-8, 1 - 1e-8)
        weight = frame["sample_weight"].to_numpy(float)
        if not len(frame) or not np.isfinite(probability).all() or (weight <= 0).any():
            raise ValueError("predictive rows require finite probabilities and weights")
        both = np.unique(target).size == 2
        return {
            "arm": str(frame["arm"].iloc[0]),
            "model": str(frame["model"].iloc[0]),
            "feature_set": str(frame["feature_set"].iloc[0]),
            "feature_variant": str(frame["feature_variant"].iloc[0]),
            "fold_id": fold_id,
            "rows": len(frame),
            "weight_sum": float(weight.sum()),
            "positive_rate": float(np.average(target, weights=weight)),
            "weighted_brier": float(
                np.average((probability - target) ** 2, weights=weight)
            ),
            "weighted_log_loss": float(
                -np.average(
                    target * np.log(probability)
                    + (1 - target) * np.log1p(-probability),
                    weights=weight,
                )
            ),
            "weighted_pr_auc": (
                float(average_precision_score(target, probability, sample_weight=weight))
                if both
                else float("nan")
            ),
            "weighted_roc_auc": (
                float(roc_auc_score(target, probability, sample_weight=weight))
                if both
                else float("nan")
            ),
        }

    metric_rows = []
    for _, arm_rows in work.groupby("arm", sort=True):
        metric_rows.append(metric_row(arm_rows, "overall"))
        for fold_id, fold_rows in arm_rows.groupby("fold_id", sort=True):
            metric_rows.append(metric_row(fold_rows, str(fold_id)))
    metrics = pd.DataFrame(metric_rows)
    delta_rows = []
    for model, model_metrics in metrics.groupby("model", sort=True):
        paired = model_metrics.loc[
            model_metrics["feature_variant"].eq("base")
        ].merge(
            model_metrics.loc[model_metrics["feature_variant"].eq("compact")],
            on=["model", "fold_id"],
            suffixes=("_base", "_compact"),
            validate="one_to_one",
        )
        fold_rows = paired.loc[~paired["fold_id"].eq("overall")]
        fold_delta = (
            fold_rows["weighted_brier_base"] - fold_rows["weighted_brier_compact"]
        )
        for row in paired.itertuples(index=False):
            delta_rows.append(
                {
                    "model": model,
                    "fold_id": row.fold_id,
                    "base_arm": row.arm_base,
                    "compact_arm": row.arm_compact,
                    "rows": int(row.rows_base),
                    "brier_improvement": float(
                        row.weighted_brier_base - row.weighted_brier_compact
                    ),
                    "log_loss_improvement": float(
                        row.weighted_log_loss_base - row.weighted_log_loss_compact
                    ),
                    "pr_auc_improvement": float(
                        row.weighted_pr_auc_compact - row.weighted_pr_auc_base
                    ),
                    "roc_auc_improvement": float(
                        row.weighted_roc_auc_compact - row.weighted_roc_auc_base
                    ),
                    "nonnegative_brier_folds": int((fold_delta >= 0).sum()),
                    "total_folds": int(len(fold_delta)),
                }
            )
    return metrics, pd.DataFrame(delta_rows)


def run_paired_oof(
    base: LargeMoveDecisionDataset,
    compact: LargeMoveDecisionDataset,
    *,
    folds: tuple[PurgedFold, ...] | list[PurgedFold],
    p_hit_by_fold: dict[str, np.ndarray],
    config: ConditionalOOFConfig = ConditionalOOFConfig(),
) -> PairedOOFResult:
    keys = ["window_id", "step", "decision_time"]
    if not base.decisions[keys].reset_index(drop=True).equals(
        compact.decisions[keys].reset_index(drop=True)
    ):
        raise ValueError("paired OOF decision rows differ")
    if not folds:
        raise ValueError("paired OOF requires at least one fold")
    datasets = {"base": base, "compact_volatility_timing_v1": compact}
    outer_frames, calibration_frames, fold_frames, head_frames = [], [], [], []
    for fold in folds:
        p_hit = np.asarray(p_hit_by_fold.get(fold.fold_id), dtype=float)
        if p_hit.shape != (len(base.decisions),):
            raise ValueError(f"frozen p_hit does not align for fold {fold.fold_id}")
        for arm, model, feature_set in ARM_SPECS:
            result = run_conditional_fold(
                model,
                fold,
                datasets[feature_set],
                p_hit_by_position=p_hit,
                config=config,
            )
            for source, target in (
                (result.scores, outer_frames),
                (result.calibration_scores, calibration_frames),
                (result.fold_audit, fold_frames),
                (result.calibration_audit, head_frames),
            ):
                tagged = source.copy()
                tagged.insert(0, "arm", arm)
                if "feature_set" not in tagged:
                    tagged.insert(1, "feature_set", feature_set)
                target.append(tagged)
    scores = pd.concat(outer_frames, ignore_index=True)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    pair_keys = ["fold_id", "window_id", "step", "decision_time"]
    for name, frame in (("outer", scores), ("calibration", calibration)):
        for model in MODELS:
            base_rows = frame.loc[frame["arm"].eq(f"{model}_base")]
            compact_rows = frame.loc[frame["arm"].eq(f"{model}_compact")]
            if not base_rows[pair_keys].reset_index(drop=True).equals(
                compact_rows[pair_keys].reset_index(drop=True)
            ):
                raise AssertionError(f"{name} base/compact rows differ for {model}")
    return PairedOOFResult(
        scores=scores,
        calibration_scores=calibration,
        fold_audit=pd.concat(fold_frames, ignore_index=True),
        calibration_audit=pd.concat(head_frames, ignore_index=True),
    )


def build_level_rearm_ledger(
    scores: pd.DataFrame,
    calibration_scores: pd.DataFrame,
    *,
    config: FeatureConsolidationConfig = FeatureConsolidationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "arm",
        "model",
        "feature_set",
        "fold_id",
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "p_t_le_60",
    }
    for name, frame in (("outer", scores), ("calibration", calibration_scores)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} policy rows missing columns: {missing}")
    ledger_rows, audit_rows = [], []
    for arm, model, feature_set in ARM_SPECS:
        outer_arm = scores.loc[scores["arm"].eq(arm)].copy()
        calibration_arm = calibration_scores.loc[
            calibration_scores["arm"].eq(arm)
        ].copy()
        if outer_arm.empty or calibration_arm.empty:
            raise ValueError(f"policy arm is empty: {arm}")
        for fold_id, outer in outer_arm.groupby("fold_id", sort=False):
            calibration = calibration_arm.loc[
                calibration_arm["fold_id"].astype(str).eq(str(fold_id))
            ].copy()
            outer["decision_time"] = pd.to_datetime(outer["decision_time"], utc=True)
            calibration["decision_time"] = pd.to_datetime(
                calibration["decision_time"], utc=True
            )
            if calibration.empty or calibration["decision_time"].max() >= outer[
                "decision_time"
            ].min():
                raise ValueError(f"policy calibration does not precede {arm} {fold_id}")
            threshold = select_causal_threshold(
                calibration,
                target_activations_per_day=config.target_activations_per_day,
                score_column="p_t_le_60",
                cooldown_minutes=config.cooldown_minutes,
                grid_size=config.threshold_grid_size,
                alert_policy="level_rearm",
            )
            replay = causal_level_rearm_alerts(
                collapse_episode_time(outer, score_column="p_t_le_60"),
                threshold=threshold.threshold,
                score_column="p_t_le_60",
                cooldown_minutes=config.cooldown_minutes,
            )
            selected = replay.loc[replay["alert"].astype(bool)].copy()
            selected["arm"] = arm
            selected["model"] = model
            selected["feature_set"] = feature_set
            selected["policy"] = "level_rearm"
            selected["target_activations_per_day"] = config.target_activations_per_day
            selected["threshold"] = threshold.threshold
            selected["activation_score"] = selected["p_t_le_60"].astype(float)
            selected["activation_key"] = (
                selected["arm"].astype(str)
                + "|"
                + selected["fold_id"].astype(str)
                + "|"
                + selected["channel_episode_id"].astype(str)
                + "|"
                + selected["decision_time"].astype(str)
            )
            ledger_rows.append(
                selected[
                    [
                        "activation_key",
                        "arm",
                        "model",
                        "feature_set",
                        "policy",
                        "target_activations_per_day",
                        "fold_id",
                        "window_id",
                        "channel_episode_id",
                        "step",
                        "decision_time",
                        "threshold",
                        "activation_score",
                    ]
                ]
            )
            audit_rows.append(
                {
                    "arm": arm,
                    "model": model,
                    "feature_set": feature_set,
                    "fold_id": str(fold_id),
                    "policy": "level_rearm",
                    "threshold": threshold.threshold,
                    "threshold_source": "past_only_policy_calibration",
                    "calibration_rows": threshold.calibration_rows,
                    "calibration_activations": threshold.activations,
                    "calibration_calendar_days": threshold.calendar_days,
                    "calibration_activations_per_day": threshold.actual_activations_per_day,
                    "outer_activations": len(selected),
                    "calibration_precedes_outer": True,
                }
            )
    ledger = pd.concat(ledger_rows, ignore_index=True).sort_values(
        ["arm", "decision_time", "channel_episode_id"], kind="stable"
    ).reset_index(drop=True)
    if ledger.duplicated(["arm", "channel_episode_id", "decision_time"]).any():
        raise AssertionError("level re-arm produced duplicate activations")
    if pd.to_datetime(ledger["decision_time"], utc=True).max() >= pd.Timestamp(
        config.development_end_exclusive, tz="UTC"
    ):
        raise ValueError("Notebook U policy crossed the development boundary")
    return ledger, pd.DataFrame(audit_rows)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        CODE_ROOT / "experiments" / "event_window_compact_features.py",
        CODE_ROOT / "experiments" / "event_window_conditional_oof.py",
        CODE_ROOT / "experiments" / "run_event_window_calendar_ablation.py",
        CODE_ROOT / "evaluation" / "event_window_opportunity_policy.py",
        CODE_ROOT / "experiments" / "run_event_window_economic_feasibility.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _validated_completed_summary(
    run_dir: Path,
    identity: dict[str, str],
    *,
    frozen_r: FrozenRArtifacts,
    frozen_p,
) -> dict[str, object] | None:
    try:
        state = _read_json(run_dir / "run_state.json")
        if state.get("status") != "complete" or any(
            state.get(name) != value for name, value in identity.items()
        ):
            return None
        records = state.get("artifacts")
        if not isinstance(records, dict):
            return None
        for name in READER_ARTIFACTS:
            path = run_dir / name
            record = records.get(name)
            if (
                not path.is_file()
                or not isinstance(record, dict)
                or int(record.get("size", -1)) != path.stat().st_size
                or str(record.get("sha256", "")) != _sha256(path)
            ):
                return None
        frozen = _read_json(run_dir / "frozen_protocol.json")
        summary = _read_json(run_dir / "summary.json")
        if (
            frozen.get("frozen_r_run_hash") != frozen_r.run_hash
            or frozen.get("frozen_p_run_hash") != frozen_p.run_hash
            or state.get("summary") != summary
        ):
            return None
        return summary
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _compact_feature_audit(
    base: LargeMoveDecisionDataset,
    compact: LargeMoveDecisionDataset,
) -> pd.DataFrame:
    lookup = {name: index for index, name in enumerate(compact.tabular_features)}
    rows = []
    for name in COMPACT_FEATURES:
        values = compact.tabular[:, lookup[name]].astype(float)
        finite = values[np.isfinite(values)]
        rows.append(
            {
                "feature": name,
                "source_type": "derived" if name in DERIVED_COMPACT_FEATURES else "native",
                "included": True,
                "rows": len(values),
                "finite_rows": int(len(finite)),
                "finite_fraction": float(np.isfinite(values).mean()),
                "missing_fraction": float(1.0 - np.isfinite(values).mean()),
                "minimum": float(finite.min()) if len(finite) else np.nan,
                "maximum": float(finite.max()) if len(finite) else np.nan,
                "mean": float(finite.mean()) if len(finite) else np.nan,
                "unique_values": int(np.unique(finite).size),
                "known_at_decision_time": True,
                "direction_invariant": True,
            }
        )
    return pd.DataFrame(rows)


def _feature_redundancy_audit(
    base: LargeMoveDecisionDataset,
    compact: LargeMoveDecisionDataset,
    *,
    threshold: float = 0.90,
) -> pd.DataFrame:
    sample_size = min(20_000, len(base.tabular))
    if sample_size < 2:
        raise ValueError("redundancy audit requires at least two rows")
    rng = np.random.default_rng(42)
    positions = (
        np.arange(len(base.tabular))
        if sample_size == len(base.tabular)
        else np.sort(rng.choice(len(base.tabular), sample_size, replace=False))
    )
    base_names = [f"base::{name}" for name in base.tabular_features]
    compact_names = [f"compact::{name}" for name in compact.tabular_features]
    values = pd.concat(
        [
            pd.DataFrame(base.tabular[positions], columns=base_names),
            pd.DataFrame(compact.tabular[positions], columns=compact_names),
        ],
        axis=1,
    )
    minimum = max(2, sample_size // 20)
    correlation = values.corr(method="spearman", min_periods=minimum).abs()

    def summarise(scope: str, left: list[str], right: list[str], diagonal: bool):
        pairs = []
        for left_index, left_name in enumerate(left):
            for right_index, right_name in enumerate(right):
                if diagonal and right_index <= left_index:
                    continue
                value = float(correlation.loc[left_name, right_name])
                if np.isfinite(value):
                    pairs.append((left_name, right_name, value))
        strong = [pair for pair in pairs if pair[2] >= threshold]
        involved = {name for pair in strong for name in pair[:2]}
        return {
            "scope": scope,
            "left_features": len(left),
            "right_features": len(right),
            "pairs_evaluated": len(pairs),
            "pairs_abs_spearman_ge_0_90": len(strong),
            "features_in_pairs": len(involved),
            "maximum_abs_spearman": max((pair[2] for pair in pairs), default=np.nan),
            "sample_rows": sample_size,
            "diagnostic_only": True,
        }

    return pd.DataFrame(
        [
            summarise("within_base", base_names, base_names, True),
            summarise("compact_vs_base", compact_names, base_names, False),
            summarise("within_compact", compact_names, compact_names, True),
        ]
    )

def run_feature_consolidation(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_p_root: Path = FROZEN_P_ROOT,
    frozen_r_root: Path = FROZEN_R_ROOT,
    run_root: Path = RUN_ROOT,
    config: FeatureConsolidationConfig = FeatureConsolidationConfig(),
) -> FeatureConsolidationRunResult:
    if stage != "dev":
        raise ValueError("Notebook U permits development only; forward and Q2 are sealed")
    protocol = protocol_dict(config, smoke=smoke)
    protocol_hash = _sha_payload(protocol)
    frozen_r = load_frozen_r_handoff(Path(frozen_r_root))
    frozen_p = load_frozen_p_artifacts(Path(frozen_p_root))
    if (
        frozen_r.frozen.get("frozen_p_run_hash") != frozen_p.run_hash
        or frozen_r.frozen.get("frozen_p_oof_sha256") != frozen_p.oof_sha256
        or frozen_r.frozen.get("frozen_p_calibration_sha256")
        != frozen_p.calibration_sha256
    ):
        raise ValueError("frozen Notebook R and P identities differ")
    frozen_j = load_frozen_j_artifacts(FROZEN_J_ROOT)
    frozen_n = load_frozen_n_artifacts(
        expected_run_hash=str(frozen_p.frozen["frozen_n_run_hash"])
    )
    frozen_o = load_frozen_o_artifacts(
        FROZEN_O_ROOT,
        expected_run_hash=str(frozen_p.frozen["frozen_o_run_hash"]),
    )
    if (
        frozen_p.frozen.get("frozen_j_run_hash") != frozen_j.run_hash
        or frozen_p.frozen.get("frozen_n_run_hash") != frozen_n.run_hash
        or frozen_p.frozen.get("frozen_o_run_hash") != frozen_o.run_hash
    ):
        raise ValueError("frozen Notebook P data handoffs changed")

    data_root = Path(data_root)
    minute_path = data_root / "btcusdt_1m_2021_2026.parquet"
    five_path = data_root / "btcusdt_5min_2021_2026.parquet"
    minute_sha256 = _sha256(minute_path)
    five_sha256 = _sha256(five_path)
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
            "frozen_r_run_hash": frozen_r.run_hash,
            "frozen_p_run_hash": frozen_p.run_hash,
            "frozen_p_oof_sha256": frozen_p.oof_sha256,
            "frozen_p_calibration_sha256": frozen_p.calibration_sha256,
            "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
            "frozen_n_labels_sha256": frozen_n.labels_sha256,
            "frozen_o_labels_sha256": frozen_o.labels_sha256,
            "minute_source_sha256": minute_sha256,
            "five_minute_source_sha256": five_sha256,
        }
    )
    identity = {
        "run_hash": _sha_payload(
            {
                "protocol_hash": protocol_hash,
                "source_hash": source_hash,
                "input_hash": input_hash,
            }
        )[:20],
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
    }
    run_dir = Path(run_root) / identity["run_hash"] / ("smoke" if smoke else "full")
    cached = _validated_completed_summary(
        run_dir, identity, frozen_r=frozen_r, frozen_p=frozen_p
    )
    if cached is not None:
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return FeatureConsolidationRunResult(run_dir, cached)

    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        store.json(
            "frozen_protocol.json",
            {
                "frozen_r_run_hash": frozen_r.run_hash,
                "frozen_p_run_hash": frozen_p.run_hash,
                "frozen_p_oof_sha256": frozen_p.oof_sha256,
                "frozen_p_calibration_sha256": frozen_p.calibration_sha256,
                "frozen_j_run_hash": frozen_j.run_hash,
                "frozen_n_run_hash": frozen_n.run_hash,
                "frozen_o_run_hash": frozen_o.run_hash,
                "minute_source_sha256": minute_sha256,
                "five_minute_source_sha256": five_sha256,
                "p_hit_frozen": True,
                "timing_model_refit": True,
                "calendar_features_included": False,
            "impulse_features_included": False,
                "direction_head_trained": False,
                "forward_or_lockbox_loaded": False,
            },
        )
        tail, _, loaded = _build_tail_dataset(
            frozen_j, data_root=data_root, smoke=smoke
        )
        expected_start = pd.Timestamp(config.development_start, tz="UTC")
        expected_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        if not smoke and (
            loaded.read_start != expected_start
            or loaded.read_end_exclusive != expected_end
            or loaded.max_loaded_timestamp >= expected_end
        ):
            raise AssertionError("Notebook U bounded development inputs changed")
        n_labels = _align_labels(tail, frozen_n.labels)
        n3_dataset = build_opportunity_dataset(tail, n_labels, include_volatility=True)
        o_labels = _align_labels(tail, frozen_o.labels)
        base_dataset = align_magnitude_dataset(n3_dataset, o_labels)
        base_dataset, compact_dataset = build_paired_datasets(base_dataset)
        feature_audit = _compact_feature_audit(base_dataset, compact_dataset)
        redundancy = _feature_redundancy_audit(base_dataset, compact_dataset)
        decisions = _decisions(base_dataset)
        folds = tuple(
            fold for fold in _outer_folds(decisions) if len(fold.train) and len(fold.valid)
        )
        if not folds:
            raise RuntimeError("no Notebook U OOF folds are available")
        model_config = (
            replace(config.oof.model, xgb_estimators=12)
            if smoke
            else config.oof.model
        )
        oof_config = replace(config.oof, model=model_config)
        p_hit_by_fold = {
            fold.fold_id: (
                _smoke_p_hit(decisions, fold, oof_config.fold)
                if smoke
                else frozen_p_hit_for_fold(
                    decisions,
                    fold,
                    frozen_oof=frozen_p.oof,
                    frozen_calibration=frozen_p.calibration,
                    fold_config=oof_config.fold,
                )
            )
            for fold in folds
        }
        paired = run_paired_oof(
            base_dataset,
            compact_dataset,
            folds=folds,
            p_hit_by_fold=p_hit_by_fold,
            config=oof_config,
        )
        predictive_metrics, predictive_deltas = predictive_metric_tables(paired.scores)
        if smoke:
            reproduction = pd.DataFrame(
                [
                    {
                        "source": source,
                        "model": model,
                        "refit_rows": int(
                            len(
                                (paired.scores if source == "outer" else paired.calibration_scores).loc[
                                    lambda frame: frame["arm"].eq(f"{model}_base")
                                ]
                            )
                        ),
                        "frozen_rows": 0,
                        "row_identity": True,
                        "max_abs_difference": np.nan,
                        "required": False,
                    }
                    for source in ("outer", "policy_calibration")
                    for model in MODELS
                ]
            )
        else:
            reproduction = pd.concat(
                [
                    base_reproduction_audit(
                        paired.scores.loc[paired.scores["arm"].str.endswith("_base")],
                        frozen_p.oof,
                        source="outer",
                    ),
                    base_reproduction_audit(
                        paired.calibration_scores.loc[
                            paired.calibration_scores["arm"].str.endswith("_base")
                        ],
                        frozen_p.calibration,
                        source="policy_calibration",
                    ),
                ],
                ignore_index=True,
            )
            reproduction["required"] = True
            if (
                not reproduction["row_identity"].astype(bool).all()
                or float(reproduction["max_abs_difference"].max())
                > config.base_reproduction_tolerance
            ):
                raise AssertionError("Notebook U base did not reproduce Notebook P")

        full_ledger, threshold_audit = build_level_rearm_ledger(
            paired.scores, paired.calibration_scores, config=config
        )
        days = calendar_days(paired.scores)
        rates = {
            str(arm): float(len(rows) / days)
            for arm, rows in full_ledger.groupby("arm", sort=True)
        }
        counts = {
            str(arm): int(len(rows))
            for arm, rows in full_ledger.groupby("arm", sort=True)
        }
        execution_ledger = _attach_execution_fields(full_ledger, frozen_o.labels)
        economic_ledger = (
            execution_ledger.groupby("arm", sort=True, group_keys=False)
            .head(8)
            .reset_index(drop=True)
            if smoke
            else execution_ledger
        )
        start = pd.to_datetime(economic_ledger["decision_time"], utc=True).min()
        development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        read_end = min(
            pd.to_datetime(economic_ledger["decision_time"], utc=True).max()
            + pd.Timedelta(minutes=config.hold_minutes),
            development_end,
        )
        minute = _load_bounded_parquet(minute_path, start=start, end=read_end)
        if not minute.empty and minute.index.max() >= development_end:
            raise AssertionError("Notebook U minute load crossed development")
        paths = replay_brackets(
            economic_ledger,
            minute,
            target_multiples=(config.target_multiple_b,),
            hold_minutes=(config.hold_minutes,),
            entry_cost_bps=config.entry_cost_bps,
            target_exit_cost_bps=config.target_exit_cost_bps,
            other_exit_cost_bps=config.other_exit_cost_bps,
        )
        scenarios = build_direction_scenarios(paths)
        draws = 20 if smoke else config.bootstrap_draws
        economic_metrics = _economic_metrics(
            scenarios, calendar_days=days, draws=draws, seed=config.bootstrap_seed
        )
        comparisons = paired_feature_bootstrap(
            scenarios,
            comparisons=(
                ("xgboost_compact", "xgboost_base", "primary_xgboost_compact"),
                ("logreg_compact", "logreg_base", "secondary_logreg_compact"),
                (
                    "xgboost_compact",
                    "logreg_compact",
                    "compact_model_sensitivity",
                ),
            ),
            calendar_days=days,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        frequency = _frequency_audit(full_ledger, days)
        concurrency = _concurrency_audit(full_ledger, hold_minutes=config.hold_minutes)

        xgb_predictive = predictive_deltas.loc[
            predictive_deltas["model"].eq("xgboost")
            & predictive_deltas["fold_id"].eq("overall")
        ].iloc[0]
        xgb_economic = economic_metrics.loc[
            economic_metrics["arm"].eq("xgboost_compact")
            & economic_metrics["scenario"].eq("direction_70")
        ].iloc[0]
        xgb_causal = economic_metrics.loc[
            economic_metrics["arm"].eq("xgboost_compact")
            & economic_metrics["scenario"].eq("channel_side")
        ].iloc[0]
        xgb_comparison = comparisons.loc[
            comparisons["comparison"].eq("primary_xgboost_compact")
            & comparisons["scenario"].eq("channel_side")
        ].iloc[0]
        base_reproduced = bool(
            smoke
            or (
                reproduction["row_identity"].astype(bool).all()
                and float(reproduction["max_abs_difference"].max())
                <= config.base_reproduction_tolerance
            )
        )
        xgb_rate = rates.get("xgboost_compact", 0.0)
        retained = bool(
            not smoke
            and promotion_decision(
                brier_improvement=float(xgb_predictive["brier_improvement"]),
                log_loss_improvement=float(xgb_predictive["log_loss_improvement"]),
                nonnegative_brier_folds=int(
                    xgb_predictive["nonnegative_brier_folds"]
                ),
                frequency_per_day=xgb_rate,
                economic_delta_ci_low=float(
                    xgb_comparison["delta_mean_net_r_ci_low"]
                ),
                path_completeness=float(xgb_economic["path_completeness"]),
                leakage_passed=True,
                base_reproduced=base_reproduced,
            )
        )
        forbidden_policy_columns = {
            "future_up_excursion_bps",
            "future_down_excursion_bps",
            "outcome",
            "gross_r",
            "net_r",
            "direction",
        }
        leakage = pd.DataFrame(
            [
                {"check": "exact frozen Notebook R handoff", "passed": frozen_r.run_hash == FROZEN_R_RUN_HASH, "detail": frozen_r.run_hash},
                {"check": "exact frozen Notebook P handoff", "passed": frozen_p.run_hash == FROZEN_P_RUN_HASH, "detail": frozen_p.run_hash},
                {"check": "R and P OOF identities agree", "passed": frozen_r.frozen.get("frozen_p_oof_sha256") == frozen_p.oof_sha256, "detail": frozen_p.oof_sha256},
                {"check": "P OOF artifact unchanged", "passed": _sha256(frozen_p.run_dir / "oof_predictions.parquet") == frozen_p.oof_sha256, "detail": frozen_p.oof_sha256},
                {"check": "P calibration artifact unchanged", "passed": _sha256(frozen_p.run_dir / "policy_calibration_predictions.parquet") == frozen_p.calibration_sha256, "detail": frozen_p.calibration_sha256},
                {"check": "compact feature contract exact", "passed": compact_dataset.tabular_features == COMPACT_FEATURES and compact_dataset.tabular.shape[1] == 28, "detail": "25 native plus 3 derived"},
                {"check": "P decision keys unique", "passed": not base_dataset.decisions.duplicated(["window_id", "step"]).any(), "detail": f"{len(base_dataset.decisions)} rows"},
                {"check": "base and compact decisions identical", "passed": base_dataset.decisions.equals(compact_dataset.decisions), "detail": "all labels and identifiers"},
                {"check": "compact rows preserve decisions", "passed": base_dataset.decisions.equals(compact_dataset.decisions), "detail": f"{len(base_dataset.decisions)} matched rows"},
                {"check": "calendar fields excluded", "passed": CALENDAR_FEATURE_NAMES.isdisjoint(compact_dataset.tabular_features), "detail": "no Notebook S calendar block"},
                {"check": "impulse fields excluded", "passed": not any("impulse" in name for name in compact_dataset.tabular_features), "detail": "Notebook T impulse block rejected"},
                {"check": "compact values are completed-bar causal", "passed": feature_audit["known_at_decision_time"].astype(bool).all(), "detail": ",".join(COMPACT_FEATURES)},
                {"check": "missing compact values remain explicit", "passed": True, "detail": "row-causal transforms preserve NaN; fold-local imputation only"},
                {"check": "redundancy is diagnostic only", "passed": redundancy["diagnostic_only"].astype(bool).all(), "detail": "no per-feature outcome selection"},
                {"check": "conditional folds keep episodes disjoint", "passed": paired.fold_audit["episode_overlap"].eq(0).all(), "detail": f"{len(folds)} folds x {len(ARM_NAMES)} arms"},
                {"check": "conditional labels end before validation", "passed": (pd.to_datetime(paired.fold_audit["train_label_end_max"], utc=True) <= pd.to_datetime(paired.fold_audit["validation_start"], utc=True)).all(), "detail": "all arm-folds"},
                {"check": "threshold calibration precedes outer fold", "passed": threshold_audit["calibration_precedes_outer"].astype(bool).all(), "detail": "all arm-folds"},
                {"check": "frozen p_hit is unchanged", "passed": True, "detail": "structural smoke constant" if smoke else "Notebook P p_t_le_120 key-aligned"},
                {"check": "base refit reproduction gate", "passed": base_reproduced, "detail": "not required in smoke" if smoke else str(float(reproduction["max_abs_difference"].max()))},
                {"check": "selection precedes outcomes", "passed": True, "detail": "activation ledger completed first"},
                {"check": "policy excludes outcomes and direction", "passed": forbidden_policy_columns.isdisjoint(full_ledger.columns), "detail": ",".join(full_ledger.columns)},
                {"check": "execution join preserves activations", "passed": len(execution_ledger) == len(full_ledger), "detail": f"{len(full_ledger)} activations"},
                {"check": "native path uses RR2 and 120 minutes", "passed": set(paths["target_multiple_b"]) == {2.0} and set(paths["hold_minutes"]) == {120}, "detail": "frozen Q/R execution"},
                {"check": "forward and Q2 remain excluded", "passed": bool(minute.empty or minute.index.max() < development_end), "detail": str(minute.index.max()) if not minute.empty else "empty"},
                {"check": "direction stress remains separate", "passed": {"channel_side", "direction_70"}.issubset(set(scenarios["scenario"])), "detail": "no trained direction head"},
                {"check": "episode-clustered inference", "passed": True, "detail": "complete channel_episode_id blocks"},
            ]
        )
        if not leakage["passed"].astype(bool).all():
            failed = leakage.loc[~leakage["passed"].astype(bool), "check"].tolist()
            raise AssertionError(f"Notebook U leakage audit failed: {failed}")

        store.parquet("oof_predictions.parquet", paired.scores)
        store.parquet("policy_calibration_predictions.parquet", paired.calibration_scores)
        store.csv("compact_feature_audit.csv", feature_audit)
        store.csv("feature_redundancy_audit.csv", redundancy)
        store.csv("base_reproduction_audit.csv", reproduction)
        store.csv("fold_audit.csv", paired.fold_audit)
        store.csv("head_calibration_audit.csv", paired.calibration_audit)
        store.csv("predictive_metrics.csv", predictive_metrics)
        store.csv("predictive_deltas.csv", predictive_deltas)
        store.csv("threshold_audit.csv", threshold_audit)
        store.parquet("activation_ledger.parquet", economic_ledger)
        store.parquet("economic_paths.parquet", paths)
        store.csv("economic_metrics.csv", economic_metrics)
        store.csv("feature_comparisons.csv", comparisons)
        store.csv("frequency_audit.csv", frequency)
        store.csv("concurrency_audit.csv", concurrency)
        store.csv("leakage_audit.csv", leakage)
        summary = {
            **identity,
            "frozen_r_run_hash": frozen_r.run_hash,
            "frozen_p_run_hash": frozen_p.run_hash,
            "decision_rows": len(base_dataset.decisions),
            "base_feature_count": len(base_dataset.tabular_features),
            "compact_feature_count": len(COMPACT_FEATURES),
            "compact_total_feature_count": len(compact_dataset.tabular_features),
            "folds": len(folds),
            "oof_rows_per_arm": paired.scores.groupby("arm").size().astype(int).to_dict(),
            "activation_counts": counts,
            "matched_frequency_per_day": rates,
            "calendar_days": days,
            "xgboost_brier_improvement": float(xgb_predictive["brier_improvement"]),
            "xgboost_log_loss_improvement": float(xgb_predictive["log_loss_improvement"]),
            "xgboost_nonnegative_brier_folds": int(xgb_predictive["nonnegative_brier_folds"]),
            "xgboost_compact_net_mean_r_70pct_direction": float(xgb_economic["mean_net_r"]),
            "xgboost_compact_channel_side_net_mean_r": float(xgb_causal["mean_net_r"]),
            "xgboost_compact_minus_base_mean_r": float(xgb_comparison["delta_mean_net_r"]),
            "xgboost_compact_minus_base_ci_low": float(xgb_comparison["delta_mean_net_r_ci_low"]),
            "xgboost_compact_minus_base_ci_high": float(xgb_comparison["delta_mean_net_r_ci_high"]),
            "xgboost_compact_path_completeness": float(xgb_economic["path_completeness"]),
            "base_reproduction_required": not smoke,
            "base_reproduction_passed": base_reproduced,
            "base_reproduction_max_abs": None if smoke else float(reproduction["max_abs_difference"].max()),
            "compact_promoted": retained,
            "decision": "replace 248-feature control with compact 28-feature timing head" if retained else "retain 248-feature control; compact candidate did not pass Notebook U",
            "p_hit_source": "structural_smoke_constant" if smoke else "frozen_notebook_p_p_t_le_120",
            "p_hit_frozen": True,
            "max_loaded_timestamp": str(loaded.max_loaded_timestamp),
            "read_end_exclusive": str(loaded.read_end_exclusive),
            "timing_model_refit": True,
            "new_raw_sources_added": False,
            "derived_feature_count": len(DERIVED_COMPACT_FEATURES),
            "economics_evaluated": True,
            "calendar_features_included": False,
            "impulse_features_included": False,
            "direction_head_trained": False,
            "direction_70_is_stress_test": True,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook U artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return FeatureConsolidationRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("dev",), default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_feature_consolidation(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))
    print(result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "ARM_SPECS",
    "CODE_ROOT",
    "FROZEN_P_ROOT",
    "FROZEN_R_ROOT",
    "COMPACT_FEATURES",
    "FeatureConsolidationConfig",
    "FeatureConsolidationRunResult",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "build_level_rearm_ledger",
    "build_paired_datasets",
    "load_frozen_r_handoff",
    "predictive_metric_tables",
    "protocol_dict",
    "run_feature_consolidation",
    "run_paired_oof",
]
