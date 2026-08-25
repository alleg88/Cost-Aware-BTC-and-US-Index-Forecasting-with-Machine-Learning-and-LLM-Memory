"""Development-only paired calendar-feature ablation for Notebook S."""
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
from experiments.event_window_cost_aware_oof import (
    CostAwareFoldConfig,
    _outer_folds,
    _partitions,
)
from experiments.event_window_calendar_features import (
    CALENDAR_FEATURE_COLUMNS,
    append_calendar_features,
)
from experiments.event_window_large_move_dataset import (
    LargeMoveDecisionDataset,
    build_opportunity_dataset,
)
from experiments.event_window_magnitude_dataset import align_magnitude_dataset
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
from experiments.run_event_window_timing_policy_repair import (
    READER_ARTIFACTS as FROZEN_R_READER_ARTIFACTS,
    calendar_days,
    paired_policy_bootstrap,
)
from experiments.run_event_window_tcn import _load_bounded_parquet


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
FROZEN_P_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_conditional_opportunity"
FROZEN_R_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_timing_policy_repair"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_calendar_ablation"
FROZEN_P_RUN_HASH = "0474798f6d0eb56e64d3"
FROZEN_R_RUN_HASH = "f8de389583c836f48140"
MODELS = ("logreg", "xgboost")
FEATURE_SETS = ("base", "base+calendar_v1")
ARM_SPECS = (
    ("logreg_base", "logreg", "base"),
    ("logreg_calendar", "logreg", "base+calendar_v1"),
    ("xgboost_base", "xgboost", "base"),
    ("xgboost_calendar", "xgboost", "base+calendar_v1"),
)
ARM_NAMES = tuple(spec[0] for spec in ARM_SPECS)
READER_ARTIFACTS = (
    "oof_predictions.parquet",
    "policy_calibration_predictions.parquet",
    "calendar_feature_audit.csv",
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
class CalendarAblationConfig:
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


@dataclass(frozen=True)
class FrozenRArtifacts:
    run_hash: str
    protocol_hash: str
    run_dir: Path
    protocol: dict[str, object]
    frozen: dict[str, object]
    summary: dict[str, object]
    state: dict[str, object]


@dataclass(frozen=True)
class PairedOOFResult:
    scores: pd.DataFrame
    calibration_scores: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame


@dataclass(frozen=True)
class CalendarAblationRunResult:
    run_dir: Path
    summary: dict[str, object]


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path.name}")
    return value


def protocol_dict(
    config: CalendarAblationConfig = CalendarAblationConfig(),
    *,
    smoke: bool = False,
) -> dict[str, object]:
    return {
        "study": "notebook_s_calendar_feature_ablation",
        "stage": "dev",
        "smoke": bool(smoke),
        **asdict(config),
        "models": list(MODELS),
        "feature_sets": list(FEATURE_SETS),
        "arms": [list(spec) for spec in ARM_SPECS],
        "calendar_features": list(CALENDAR_FEATURE_COLUMNS),
        "timing_score": "p_t_le_60",
        "alert_policy": "level_rearm",
        "frozen_p_run_hash": FROZEN_P_RUN_HASH,
        "frozen_r_run_hash": FROZEN_R_RUN_HASH,
        "direction_head_trained": False,
        "direction_70_is_stress_test": True,
        "forward_or_lockbox_loaded": False,
    }


def build_paired_datasets(
    base: LargeMoveDecisionDataset,
) -> tuple[LargeMoveDecisionDataset, LargeMoveDecisionDataset]:
    """Return the untouched base matrix and its row-identical calendar extension."""
    calendar = append_calendar_features(base)
    keys = ["window_id", "step", "decision_time"]
    missing = sorted(set(keys).difference(base.decisions.columns))
    if missing:
        raise ValueError(f"paired dataset keys are missing: {missing}")
    if not base.decisions[keys].reset_index(drop=True).equals(
        calendar.decisions[keys].reset_index(drop=True)
    ):
        raise AssertionError("calendar feature construction changed decision rows")
    if not np.array_equal(
        np.asarray(base.tabular, dtype=np.float32),
        calendar.tabular[:, : base.tabular.shape[1]],
        equal_nan=True,
    ):
        raise AssertionError("calendar feature construction changed base values")
    return base, calendar


def base_reproduction_audit(
    refit: pd.DataFrame,
    frozen: pd.DataFrame,
    *,
    source: str,
) -> pd.DataFrame:
    """Compare refitted base timing probabilities to frozen Notebook P by key."""
    keys = ("model", "fold_id", "window_id", "step", "decision_time")
    probabilities = ("p_t_le_15", "p_t_le_30", "p_t_le_60", "p_t_le_120")
    required = set((*keys, *probabilities))
    for name, frame in (("refit", refit), ("frozen", frozen)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} reproduction frame missing columns: {missing}")
        if frame.duplicated(list(keys)).any():
            raise ValueError(f"{name} reproduction keys are not unique")
    left = refit.copy()
    right = frozen.copy()
    for frame in (left, right):
        frame["decision_time"] = pd.to_datetime(
            frame["decision_time"], utc=True, errors="raise"
        )
    rows: list[dict[str, object]] = []
    models = sorted(set(left["model"]).union(right["model"]))
    for model in models:
        left_model = left.loc[left["model"].eq(model)].sort_values(
            list(keys), kind="stable"
        )
        right_model = right.loc[right["model"].eq(model)].sort_values(
            list(keys), kind="stable"
        )
        row_identity = left_model[list(keys)].reset_index(drop=True).equals(
            right_model[list(keys)].reset_index(drop=True)
        )
        if row_identity and len(left_model):
            difference = np.abs(
                left_model[list(probabilities)].to_numpy(float)
                - right_model[list(probabilities)].to_numpy(float)
            )
            max_abs = float(difference.max())
        else:
            max_abs = float("inf")
        rows.append(
            {
                "source": source,
                "model": model,
                "refit_rows": len(left_model),
                "frozen_rows": len(right_model),
                "row_identity": row_identity,
                "max_abs_difference": max_abs,
            }
        )
    return pd.DataFrame(rows)


def predictive_metric_tables(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return proper scores and paired calendar-minus-base improvements."""
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
    work["decision_time"] = pd.to_datetime(
        work["decision_time"], utc=True, errors="raise"
    )
    work["feature_variant"] = np.where(
        work["feature_set"].eq("base"), "base", "calendar"
    )
    if not set(work["feature_variant"]) <= {"base", "calendar"}:
        raise ValueError("predictive feature variants are invalid")
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
            name: group[pair_keys].sort_values(pair_keys, kind="stable").reset_index(
                drop=True
            )
            for name, group in model_rows.groupby("feature_variant", sort=True)
        }
        if set(variants) != {"base", "calendar"} or not variants["base"].equals(
            variants["calendar"]
        ):
            raise ValueError(f"base/calendar predictive rows differ for {model}")

    def metric_row(frame: pd.DataFrame, *, fold_id: str) -> dict[str, object]:
        target = frame["y_t_le_60"].to_numpy(int)
        probability = np.clip(frame["p_t_le_60"].to_numpy(float), 1e-8, 1.0 - 1e-8)
        weight = frame["sample_weight"].to_numpy(float)
        if (
            not len(frame)
            or not np.isfinite(probability).all()
            or not np.isfinite(weight).all()
            or (weight <= 0.0).any()
        ):
            raise ValueError("predictive metric rows require finite probabilities and weights")
        weighted_brier = float(np.average((probability - target) ** 2, weights=weight))
        weighted_log_loss = float(
            -np.average(
                target * np.log(probability)
                + (1 - target) * np.log1p(-probability),
                weights=weight,
            )
        )
        both_classes = np.unique(target).size == 2
        return {
            "arm": str(frame["arm"].iloc[0]),
            "model": str(frame["model"].iloc[0]),
            "feature_set": str(frame["feature_set"].iloc[0]),
            "feature_variant": str(frame["feature_variant"].iloc[0]),
            "fold_id": fold_id,
            "rows": len(frame),
            "weight_sum": float(weight.sum()),
            "positive_rate": float(np.average(target, weights=weight)),
            "weighted_brier": weighted_brier,
            "weighted_log_loss": weighted_log_loss,
            "weighted_pr_auc": (
                float(average_precision_score(target, probability, sample_weight=weight))
                if both_classes
                else float("nan")
            ),
            "weighted_roc_auc": (
                float(roc_auc_score(target, probability, sample_weight=weight))
                if both_classes
                else float("nan")
            ),
        }

    metric_rows: list[dict[str, object]] = []
    for _, arm_rows in work.groupby("arm", sort=True):
        metric_rows.append(metric_row(arm_rows, fold_id="overall"))
        for fold_id, fold_rows in arm_rows.groupby("fold_id", sort=True):
            metric_rows.append(metric_row(fold_rows, fold_id=str(fold_id)))
    metrics = pd.DataFrame(metric_rows)

    delta_rows: list[dict[str, object]] = []
    for model, model_metrics in metrics.groupby("model", sort=True):
        base = model_metrics.loc[model_metrics["feature_variant"].eq("base")]
        calendar = model_metrics.loc[
            model_metrics["feature_variant"].eq("calendar")
        ]
        paired = base.merge(
            calendar,
            on=["model", "fold_id"],
            suffixes=("_base", "_calendar"),
            validate="one_to_one",
        )
        fold_improvements = (
            paired.loc[~paired["fold_id"].eq("overall"), "weighted_brier_base"]
            - paired.loc[
                ~paired["fold_id"].eq("overall"), "weighted_brier_calendar"
            ]
        )
        nonnegative = int((fold_improvements >= 0.0).sum())
        total_folds = int(len(fold_improvements))
        for row in paired.itertuples(index=False):
            delta_rows.append(
                {
                    "model": model,
                    "fold_id": row.fold_id,
                    "base_arm": row.arm_base,
                    "calendar_arm": row.arm_calendar,
                    "rows": int(row.rows_base),
                    "brier_improvement": float(
                        row.weighted_brier_base - row.weighted_brier_calendar
                    ),
                    "log_loss_improvement": float(
                        row.weighted_log_loss_base - row.weighted_log_loss_calendar
                    ),
                    "pr_auc_improvement": float(
                        row.weighted_pr_auc_calendar - row.weighted_pr_auc_base
                    ),
                    "roc_auc_improvement": float(
                        row.weighted_roc_auc_calendar - row.weighted_roc_auc_base
                    ),
                    "nonnegative_brier_folds": nonnegative,
                    "total_folds": total_folds,
                }
            )
    return metrics, pd.DataFrame(delta_rows)


def run_paired_oof(
    base: LargeMoveDecisionDataset,
    calendar: LargeMoveDecisionDataset,
    *,
    folds: tuple[PurgedFold, ...] | list[PurgedFold],
    p_hit_by_fold: dict[str, np.ndarray],
    config: ConditionalOOFConfig = ConditionalOOFConfig(),
) -> PairedOOFResult:
    """Refit base/calendar timing heads on identical frozen fold contracts."""
    keys = ["window_id", "step", "decision_time"]
    if not base.decisions[keys].reset_index(drop=True).equals(
        calendar.decisions[keys].reset_index(drop=True)
    ):
        raise ValueError("paired OOF decision rows differ")
    if not np.array_equal(
        np.asarray(base.tabular, dtype=np.float32),
        calendar.tabular[:, : base.tabular.shape[1]],
        equal_nan=True,
    ):
        raise ValueError("paired OOF calendar matrix changed base values")
    if not folds:
        raise ValueError("paired OOF requires at least one fold")

    datasets = {"base": base, "base+calendar_v1": calendar}
    score_frames: list[pd.DataFrame] = []
    calibration_frames: list[pd.DataFrame] = []
    fold_frames: list[pd.DataFrame] = []
    head_frames: list[pd.DataFrame] = []
    for fold in folds:
        if fold.fold_id not in p_hit_by_fold:
            raise ValueError(f"missing frozen p_hit for fold {fold.fold_id}")
        p_hit = np.asarray(p_hit_by_fold[fold.fold_id], dtype=float)
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
                (result.scores, score_frames),
                (result.calibration_scores, calibration_frames),
                (result.fold_audit, fold_frames),
                (result.calibration_audit, head_frames),
            ):
                tagged = source.copy()
                tagged.insert(0, "arm", arm)
                if "feature_set" not in tagged:
                    tagged.insert(1, "feature_set", feature_set)
                target.append(tagged)

    scores = pd.concat(score_frames, ignore_index=True)
    calibration_scores = pd.concat(calibration_frames, ignore_index=True)
    fold_audit = pd.concat(fold_frames, ignore_index=True)
    calibration_audit = pd.concat(head_frames, ignore_index=True)
    pair_keys = ["fold_id", "window_id", "step", "decision_time"]
    for frame_name, frame in (
        ("outer", scores),
        ("policy calibration", calibration_scores),
    ):
        for model in MODELS:
            base_rows = frame.loc[frame["arm"].eq(f"{model}_base")]
            calendar_rows = frame.loc[frame["arm"].eq(f"{model}_calendar")]
            if not base_rows[pair_keys].reset_index(drop=True).equals(
                calendar_rows[pair_keys].reset_index(drop=True)
            ):
                raise AssertionError(f"{frame_name} base/calendar OOF rows differ for {model}")
    return PairedOOFResult(
        scores=scores,
        calibration_scores=calibration_scores,
        fold_audit=fold_audit,
        calibration_audit=calibration_audit,
    )


def frozen_p_hit_for_fold(
    decisions: pd.DataFrame,
    fold: PurgedFold,
    *,
    frozen_oof: pd.DataFrame,
    frozen_calibration: pd.DataFrame,
    fold_config: CostAwareFoldConfig,
) -> np.ndarray:
    """Align frozen Notebook P incidence to reserved and outer fold positions."""
    required_decisions = {"window_id", "step", "decision_time", "model_target_valid"}
    missing_decisions = sorted(required_decisions.difference(decisions.columns))
    if missing_decisions:
        raise ValueError(f"p_hit decisions missing columns: {missing_decisions}")
    _, _, reserved = _partitions(decisions, fold.train, fold_config)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    result = np.full(len(decisions), np.nan, dtype=float)

    def values_for(
        frame: pd.DataFrame,
        positions: np.ndarray,
        *,
        source: str,
    ) -> np.ndarray:
        required = {
            "model",
            "fold_id",
            "window_id",
            "step",
            "decision_time",
            "p_t_le_120",
        }
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"frozen P {source} missing columns: {missing}")
        expected = decisions.iloc[positions][
            ["window_id", "step", "decision_time"]
        ].reset_index(drop=True)
        expected["decision_time"] = pd.to_datetime(
            expected["decision_time"], utc=True, errors="raise"
        )
        by_model: dict[str, np.ndarray] = {}
        for model in MODELS:
            selected = frame.loc[
                frame["model"].eq(model)
                & frame["fold_id"].astype(str).eq(str(fold.fold_id)),
                ["window_id", "step", "decision_time", "p_t_le_120"],
            ].copy()
            selected["decision_time"] = pd.to_datetime(
                selected["decision_time"], utc=True, errors="raise"
            )
            keys = ["window_id", "step", "decision_time"]
            if selected.duplicated(keys).any():
                raise ValueError(f"frozen P {source} keys are not unique for {model}")
            aligned = expected.merge(selected, on=keys, how="left", validate="one_to_one")
            if len(selected) != len(expected) or aligned["p_t_le_120"].isna().any():
                raise ValueError(f"frozen P {source} rows do not match {model} fold keys")
            probability = aligned["p_t_le_120"].to_numpy(float)
            if (
                not np.isfinite(probability).all()
                or (probability < 0.0).any()
                or (probability > 1.0).any()
            ):
                raise ValueError(f"frozen P {source} probabilities are invalid")
            by_model[model] = probability
        if not np.array_equal(by_model["logreg"], by_model["xgboost"]):
            raise ValueError(f"frozen P {source} conditional models disagree on p_hit")
        return by_model["xgboost"]

    result[reserved] = values_for(
        frozen_calibration, reserved, source="policy calibration"
    )
    result[outer] = values_for(frozen_oof, outer, source="outer OOF")
    return result


def build_level_rearm_ledger(
    scores: pd.DataFrame,
    calibration_scores: pd.DataFrame,
    *,
    config: CalendarAblationConfig = CalendarAblationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select fold-local level-rearm activations before any outcome is attached."""
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
    ledger_rows: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
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
            outer["decision_time"] = pd.to_datetime(
                outer["decision_time"], utc=True, errors="raise"
            )
            calibration["decision_time"] = pd.to_datetime(
                calibration["decision_time"], utc=True, errors="raise"
            )
            if (
                calibration.empty
                or calibration["decision_time"].max()
                >= outer["decision_time"].min()
            ):
                raise ValueError(f"policy calibration does not precede {arm} {fold_id}")
            threshold = select_causal_threshold(
                calibration,
                target_activations_per_day=config.target_activations_per_day,
                score_column="p_t_le_60",
                cooldown_minutes=config.cooldown_minutes,
                grid_size=config.threshold_grid_size,
                alert_policy="level_rearm",
            )
            collapsed = collapse_episode_time(outer, score_column="p_t_le_60")
            replay = causal_level_rearm_alerts(
                collapsed,
                threshold=threshold.threshold,
                score_column="p_t_le_60",
                cooldown_minutes=config.cooldown_minutes,
            )
            selected = replay.loc[replay["alert"].astype(bool)].copy()
            selected["arm"] = arm
            selected["model"] = model
            selected["feature_set"] = feature_set
            selected["policy"] = "level_rearm"
            selected["target_activations_per_day"] = (
                config.target_activations_per_day
            )
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
                    "calibration_activations_per_day": (
                        threshold.actual_activations_per_day
                    ),
                    "outer_activations": len(selected),
                    "calibration_precedes_outer": True,
                }
            )
    ledger = pd.concat(ledger_rows, ignore_index=True).sort_values(
        ["arm", "decision_time", "channel_episode_id"], kind="stable"
    ).reset_index(drop=True)
    if ledger.duplicated(["arm", "channel_episode_id", "decision_time"]).any():
        raise AssertionError("level-rearm replay produced duplicate activations")
    development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
    if pd.to_datetime(ledger["decision_time"], utc=True).max() >= development_end:
        raise ValueError("Notebook S policy crossed the development boundary")
    return ledger, pd.DataFrame(audit_rows)


def paired_feature_bootstrap(
    scenarios: pd.DataFrame,
    *,
    comparisons: tuple[tuple[str, str, str], ...],
    calendar_days: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    """Apply Notebook R's episode-complete inference to feature ablations."""
    return paired_policy_bootstrap(
        scenarios,
        comparisons=comparisons,
        calendar_days=calendar_days,
        draws=draws,
        seed=seed,
    )


def load_frozen_r_handoff(run_root: Path = FROZEN_R_ROOT) -> FrozenRArtifacts:
    """Validate every registered artifact from the exact completed Notebook R run."""
    root = Path(run_root)
    pointer = _read_json(root / "latest_dev.json")
    expected_relative = f"{FROZEN_R_RUN_HASH}/full"
    if (
        pointer.get("run_hash") != FROZEN_R_RUN_HASH
        or pointer.get("relative_path") != expected_relative
    ):
        raise ValueError("frozen Notebook R pointer changed")
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ValueError("frozen Notebook R path escaped its root")
    state = _read_json(run_dir / "run_state.json")
    protocol_hash = str(pointer.get("protocol_hash", ""))
    if (
        state.get("status") != "complete"
        or state.get("run_hash") != FROZEN_R_RUN_HASH
        or state.get("protocol_hash") != protocol_hash
    ):
        raise ValueError("frozen Notebook R state is incomplete or changed")
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ValueError("frozen Notebook R artifact registry is invalid")
    for name in FROZEN_R_READER_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or int(record.get("size", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            raise ValueError(f"frozen Notebook R artifact changed: {name}")
    protocol = _read_json(run_dir / "protocol.json")
    frozen = _read_json(run_dir / "frozen_protocol.json")
    summary = _read_json(run_dir / "summary.json")
    if (
        state.get("summary") != summary
        or summary.get("frozen_p_run_hash") != FROZEN_P_RUN_HASH
        or frozen.get("frozen_p_run_hash") != FROZEN_P_RUN_HASH
        or summary.get("level_rearm_selected_for_direction_head") is not True
        or summary.get("forward_or_lockbox_loaded") is not False
    ):
        raise ValueError("frozen Notebook R methodological handoff changed")
    return FrozenRArtifacts(
        run_hash=FROZEN_R_RUN_HASH,
        protocol_hash=protocol_hash,
        run_dir=run_dir,
        protocol=protocol,
        frozen=frozen,
        summary=summary,
        state=state,
    )


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        CODE_ROOT / "experiments" / "event_window_calendar_features.py",
        CODE_ROOT / "experiments" / "event_window_conditional_oof.py",
        CODE_ROOT / "evaluation" / "event_window_opportunity_policy.py",
        CODE_ROOT / "experiments" / "run_event_window_economic_feasibility.py",
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _latest(root: Path, run_hash: str, protocol_hash: str) -> None:
    payload = {
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "relative_path": f"{run_hash}/full",
    }
    path = Path(root) / "latest_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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


def _calendar_feature_audit(
    base: LargeMoveDecisionDataset,
    calendar: LargeMoveDecisionDataset,
) -> pd.DataFrame:
    offset = len(base.tabular_features)
    rows = []
    for index, name in enumerate(CALENDAR_FEATURE_COLUMNS, start=offset):
        values = calendar.tabular[:, index].astype(float)
        rows.append(
            {
                "feature": name,
                "rows": len(values),
                "finite": bool(np.isfinite(values).all()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
                "mean": float(values.mean()),
                "unique_values": int(np.unique(values).size),
                "known_at_decision_time": True,
                "source": "decision_time_only",
            }
        )
    return pd.DataFrame(rows)


def _smoke_p_hit(
    decisions: pd.DataFrame,
    fold: PurgedFold,
    fold_config: CostAwareFoldConfig,
) -> np.ndarray:
    """Supply a structural-only incidence constant for non-evidential smoke."""
    _, _, reserved = _partitions(decisions, fold.train, fold_config)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    values = np.full(len(decisions), np.nan, dtype=float)
    values[np.concatenate([reserved, outer])] = 0.5
    return values


def run_calendar_ablation(
    *,
    stage: str = "dev",
    smoke: bool = False,
    data_root: Path = DEFAULT_DATA_ROOT,
    frozen_p_root: Path = FROZEN_P_ROOT,
    frozen_r_root: Path = FROZEN_R_ROOT,
    run_root: Path = RUN_ROOT,
    config: CalendarAblationConfig = CalendarAblationConfig(),
) -> CalendarAblationRunResult:
    if stage != "dev":
        raise ValueError("Notebook S permits development only; forward and Q2 are sealed")
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

    minute_path = Path(data_root) / "btcusdt_1m_2021_2026.parquet"
    minute_sha256 = _sha256(minute_path)
    source_hash = _source_hash()
    input_hash = _sha_payload(
        {
            "frozen_r_run_hash": frozen_r.run_hash,
            "frozen_r_protocol_hash": frozen_r.protocol_hash,
            "frozen_p_run_hash": frozen_p.run_hash,
            "frozen_p_oof_sha256": frozen_p.oof_sha256,
            "frozen_p_calibration_sha256": frozen_p.calibration_sha256,
            "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
            "frozen_n_labels_sha256": frozen_n.labels_sha256,
            "frozen_o_labels_sha256": frozen_o.labels_sha256,
            "minute_source_sha256": minute_sha256,
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
        return CalendarAblationRunResult(run_dir, cached)

    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        store.json(
            "frozen_protocol.json",
            {
                "frozen_r_run_hash": frozen_r.run_hash,
                "frozen_r_protocol_hash": frozen_r.protocol_hash,
                "frozen_p_run_hash": frozen_p.run_hash,
                "frozen_p_protocol_hash": frozen_p.protocol_hash,
                "frozen_p_oof_sha256": frozen_p.oof_sha256,
                "frozen_p_calibration_sha256": frozen_p.calibration_sha256,
                "frozen_j_run_hash": frozen_j.run_hash,
                "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
                "frozen_n_run_hash": frozen_n.run_hash,
                "frozen_n_labels_sha256": frozen_n.labels_sha256,
                "frozen_o_run_hash": frozen_o.run_hash,
                "frozen_o_labels_sha256": frozen_o.labels_sha256,
                "minute_source_sha256": minute_sha256,
                "timing_model_refit": True,
                "new_features_added": True,
                "direction_head_trained": False,
                "forward_or_lockbox_loaded": False,
            },
        )

        tail, _, loaded = _build_tail_dataset(
            frozen_j, data_root=Path(data_root), smoke=smoke
        )
        expected_start = pd.Timestamp(config.development_start, tz="UTC")
        expected_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        if not smoke and (
            loaded.read_start != expected_start
            or loaded.read_end_exclusive != expected_end
            or loaded.max_loaded_timestamp >= expected_end
        ):
            raise AssertionError("Notebook S bounded development inputs changed")
        n_labels = _align_labels(tail, frozen_n.labels)
        n3_dataset = build_opportunity_dataset(
            tail, n_labels, include_volatility=True
        )
        o_labels = _align_labels(tail, frozen_o.labels)
        base_dataset = align_magnitude_dataset(n3_dataset, o_labels)
        base_dataset, calendar_dataset = build_paired_datasets(base_dataset)
        feature_audit = _calendar_feature_audit(base_dataset, calendar_dataset)
        decisions = _decisions(base_dataset)
        folds = tuple(
            fold
            for fold in _outer_folds(decisions)
            if len(fold.train) and len(fold.valid)
        )
        if not folds:
            raise RuntimeError("no Notebook S OOF folds are available")
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
            calendar_dataset,
            folds=folds,
            p_hit_by_fold=p_hit_by_fold,
            config=oof_config,
        )
        predictive_metrics, predictive_deltas = predictive_metric_tables(
            paired.scores
        )
        if smoke:
            reproduction = pd.DataFrame(
                [
                    {
                        "source": source,
                        "model": model,
                        "refit_rows": int(
                            len(
                                (
                                    paired.scores
                                    if source == "outer"
                                    else paired.calibration_scores
                                ).loc[lambda frame: frame["arm"].eq(f"{model}_base")]
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
                raise AssertionError("Notebook S base refit did not reproduce Notebook P")

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

        # Selection is complete before outcome geometry or native-minute paths attach.
        execution_ledger = _attach_execution_fields(full_ledger, frozen_o.labels)
        economic_ledger = (
            execution_ledger.groupby("arm", sort=True, group_keys=False)
            .head(8)
            .reset_index(drop=True)
            if smoke
            else execution_ledger
        )
        start = pd.to_datetime(
            economic_ledger["decision_time"], utc=True, errors="raise"
        ).min()
        development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        requested_end = pd.to_datetime(
            economic_ledger["decision_time"], utc=True, errors="raise"
        ).max() + pd.Timedelta(minutes=config.hold_minutes)
        read_end = min(requested_end, development_end)
        minute = _load_bounded_parquet(minute_path, start=start, end=read_end)
        if not minute.empty and minute.index.max() >= development_end:
            raise AssertionError("Notebook S minute load crossed development")
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
            scenarios,
            calendar_days=days,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        comparisons = paired_feature_bootstrap(
            scenarios,
            comparisons=(
                ("xgboost_calendar", "xgboost_base", "primary_xgboost_calendar"),
                ("logreg_calendar", "logreg_base", "secondary_logreg_calendar"),
                ("xgboost_calendar", "logreg_calendar", "calendar_model_sensitivity"),
            ),
            calendar_days=days,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        frequency = _frequency_audit(full_ledger, days)
        concurrency = _concurrency_audit(
            full_ledger, hold_minutes=config.hold_minutes
        )

        xgb_predictive = predictive_deltas.loc[
            predictive_deltas["model"].eq("xgboost")
            & predictive_deltas["fold_id"].eq("overall")
        ].iloc[0]
        xgb_economic = economic_metrics.loc[
            economic_metrics["arm"].eq("xgboost_calendar")
            & economic_metrics["scenario"].eq("direction_70")
        ].iloc[0]
        xgb_causal = economic_metrics.loc[
            economic_metrics["arm"].eq("xgboost_calendar")
            & economic_metrics["scenario"].eq("channel_side")
        ].iloc[0]
        xgb_comparison = comparisons.loc[
            comparisons["comparison"].eq("primary_xgboost_calendar")
            & comparisons["scenario"].eq("direction_70")
        ].iloc[0]
        base_reproduced = bool(
            smoke
            or (
                reproduction["row_identity"].astype(bool).all()
                and float(reproduction["max_abs_difference"].max())
                <= config.base_reproduction_tolerance
            )
        )
        xgb_rate = rates.get("xgboost_calendar", 0.0)
        retained = bool(
            not smoke
            and base_reproduced
            and float(xgb_predictive["brier_improvement"]) >= 0.0
            and int(xgb_predictive["nonnegative_brier_folds"]) >= 4
            and config.minimum_frequency_per_day
            <= xgb_rate
            <= config.maximum_frequency_per_day
            and float(xgb_economic["path_completeness"])
            >= config.minimum_path_completeness
            and float(xgb_comparison["delta_mean_net_r_ci_low"]) > 0.0
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
                {"check": "frozen J identity unchanged", "passed": frozen_p.frozen.get("frozen_j_run_hash") == frozen_j.run_hash, "detail": frozen_j.run_hash},
                {"check": "frozen N identity unchanged", "passed": frozen_p.frozen.get("frozen_n_run_hash") == frozen_n.run_hash, "detail": frozen_n.run_hash},
                {"check": "frozen O identity unchanged", "passed": frozen_p.frozen.get("frozen_o_run_hash") == frozen_o.run_hash, "detail": frozen_o.run_hash},
                {"check": "base and calendar decision rows identical", "passed": base_dataset.decisions[["window_id", "step", "decision_time"]].equals(calendar_dataset.decisions[["window_id", "step", "decision_time"]]), "detail": f"{len(base_dataset.decisions)} rows"},
                {"check": "calendar block preserves base matrix", "passed": np.array_equal(base_dataset.tabular, calendar_dataset.tabular[:, : base_dataset.tabular.shape[1]], equal_nan=True), "detail": f"{base_dataset.tabular.shape[1]} base plus {len(CALENDAR_FEATURE_COLUMNS)} calendar"},
                {"check": "calendar features are timestamp-only and finite", "passed": feature_audit["finite"].astype(bool).all() and feature_audit["known_at_decision_time"].astype(bool).all(), "detail": ",".join(CALENDAR_FEATURE_COLUMNS)},
                {"check": "conditional folds keep episodes disjoint", "passed": paired.fold_audit["episode_overlap"].eq(0).all(), "detail": f"{len(folds)} folds x {len(ARM_NAMES)} arms"},
                {"check": "conditional labels end before validation", "passed": (pd.to_datetime(paired.fold_audit["train_label_end_max"], utc=True) <= pd.to_datetime(paired.fold_audit["validation_start"], utc=True)).all(), "detail": "all arm-folds"},
                {"check": "threshold calibration precedes outer fold", "passed": threshold_audit["calibration_precedes_outer"].astype(bool).all(), "detail": "all arm-folds"},
                {"check": "frozen N3 incidence is unchanged in full mode", "passed": True, "detail": "structural 0.5 smoke only" if smoke else "P p_t_le_120 key-aligned"},
                {"check": "base refit reproduction gate", "passed": base_reproduced, "detail": "not required in smoke" if smoke else str(float(reproduction["max_abs_difference"].max()))},
                {"check": "selection precedes outcome and minute attachment", "passed": True, "detail": "activation ledger completed first"},
                {"check": "policy excludes outcomes and direction", "passed": forbidden_policy_columns.isdisjoint(full_ledger.columns), "detail": ",".join(full_ledger.columns)},
                {"check": "execution join preserves activations", "passed": len(execution_ledger) == len(full_ledger), "detail": f"{len(full_ledger)} activations"},
                {"check": "native path uses RR2 and 120 minutes", "passed": set(paths["target_multiple_b"]) == {2.0} and set(paths["hold_minutes"]) == {120}, "detail": "frozen Q/R execution"},
                {"check": "forward and Q2 remain excluded", "passed": bool(minute.empty or minute.index.max() < development_end), "detail": str(minute.index.max()) if not minute.empty else "empty"},
                {"check": "direction stress is separate from causal side", "passed": {"channel_side", "direction_70"}.issubset(set(scenarios["scenario"])), "detail": "no trained direction head"},
                {"check": "episode-clustered feature inference", "passed": True, "detail": "complete channel_episode_id blocks"},
            ]
        )
        if not leakage["passed"].astype(bool).all():
            failed = leakage.loc[~leakage["passed"].astype(bool), "check"].tolist()
            raise AssertionError(f"Notebook S leakage audit failed: {failed}")

        store.parquet("oof_predictions.parquet", paired.scores)
        store.parquet("policy_calibration_predictions.parquet", paired.calibration_scores)
        store.csv("calendar_feature_audit.csv", feature_audit)
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
            "calendar_feature_count": len(CALENDAR_FEATURE_COLUMNS),
            "calendar_total_feature_count": len(calendar_dataset.tabular_features),
            "folds": len(folds),
            "oof_rows_per_arm": paired.scores.groupby("arm").size().astype(int).to_dict(),
            "activation_counts": counts,
            "matched_frequency_per_day": rates,
            "calendar_days": days,
            "xgboost_brier_improvement": float(xgb_predictive["brier_improvement"]),
            "xgboost_log_loss_improvement": float(xgb_predictive["log_loss_improvement"]),
            "xgboost_nonnegative_brier_folds": int(xgb_predictive["nonnegative_brier_folds"]),
            "xgboost_calendar_net_mean_r_70pct_direction": float(xgb_economic["mean_net_r"]),
            "xgboost_calendar_channel_side_net_mean_r": float(xgb_causal["mean_net_r"]),
            "xgboost_calendar_minus_base_mean_r": float(xgb_comparison["delta_mean_net_r"]),
            "xgboost_calendar_minus_base_ci_low": float(xgb_comparison["delta_mean_net_r_ci_low"]),
            "xgboost_calendar_minus_base_ci_high": float(xgb_comparison["delta_mean_net_r_ci_high"]),
            "xgboost_calendar_path_completeness": float(xgb_economic["path_completeness"]),
            "base_reproduction_required": not smoke,
            "base_reproduction_passed": base_reproduced,
            "base_reproduction_max_abs": None if smoke else float(reproduction["max_abs_difference"].max()),
            "calendar_retained_for_next_experiment": retained,
            "decision": "retain calendar block for the next direction-head study" if retained else "do not promote calendar block from Notebook S",
            "p_hit_source": "structural_smoke_constant" if smoke else "frozen_notebook_p_p_t_le_120",
            "max_loaded_timestamp": str(loaded.max_loaded_timestamp),
            "read_end_exclusive": str(loaded.read_end_exclusive),
            "timing_model_refit": True,
            "new_features_added": True,
            "economics_evaluated": True,
            "direction_head_trained": False,
            "direction_70_is_stress_test": True,
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        missing = [name for name in READER_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Notebook S artifacts missing: {missing}")
        store.complete(summary)
        if not smoke:
            _latest(Path(run_root), identity["run_hash"], protocol_hash)
        return CalendarAblationRunResult(run_dir, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frozen-p-root", type=Path, default=FROZEN_P_ROOT)
    parser.add_argument("--frozen-r-root", type=Path, default=FROZEN_R_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_calendar_ablation(
        stage=args.stage,
        smoke=args.smoke,
        data_root=args.data_root,
        frozen_p_root=args.frozen_p_root,
        frozen_r_root=args.frozen_r_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARM_NAMES",
    "ARM_SPECS",
    "CALENDAR_FEATURE_COLUMNS",
    "CODE_ROOT",
    "CalendarAblationConfig",
    "CalendarAblationRunResult",
    "FEATURE_SETS",
    "FROZEN_P_ROOT",
    "FROZEN_R_READER_ARTIFACTS",
    "FROZEN_R_ROOT",
    "FrozenRArtifacts",
    "MODELS",
    "PairedOOFResult",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "base_reproduction_audit",
    "build_level_rearm_ledger",
    "build_paired_datasets",
    "frozen_p_hit_for_fold",
    "load_frozen_r_handoff",
    "paired_feature_bootstrap",
    "predictive_metric_tables",
    "protocol_dict",
    "run_paired_oof",
    "run_calendar_ablation",
]
