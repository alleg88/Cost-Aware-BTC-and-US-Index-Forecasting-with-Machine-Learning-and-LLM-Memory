"""Purged walk-forward OOF evaluation for Notebook O magnitude bins."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_cost_aware_oof import (
    CostAwareFoldConfig,
    _fit_temperature,
    _outer_folds,
    _partitions,
    _probabilities,
)
from experiments.event_window_large_move_dataset import (
    LargeMoveDecisionDataset,
)
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_magnitude_dataset import (
    MAGNITUDE_BIN_LABELS,
    assert_magnitude_feature_isolation,
)
from experiments.event_window_magnitude_models import fit_predict_magnitude_xgboost
from experiments.event_window_tail_oof import _half_open_uniqueness


CUMULATIVE_CLASS_STARTS = {
    "075": 1,
    "100": 2,
    "150": 3,
    "200": 4,
}


@dataclass(frozen=True)
class MagnitudeOOFConfig:
    fold: CostAwareFoldConfig = field(default_factory=CostAwareFoldConfig)
    model: LargeMoveModelConfig = field(default_factory=LargeMoveModelConfig)


@dataclass(frozen=True)
class MagnitudeOOFResult:
    scores: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame


def cumulative_probabilities(probabilities: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(probabilities, dtype=float)
    class_count = len(MAGNITUDE_BIN_LABELS)
    if values.ndim != 2 or values.shape[1] != class_count:
        raise ValueError("magnitude probabilities must have five columns")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("magnitude probabilities must be finite and non-negative")
    row_sum = values.sum(axis=1)
    if not np.allclose(row_sum, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("magnitude probability rows must sum to one")
    return {
        name: values[:, start:].sum(axis=1)
        for name, start in CUMULATIVE_CLASS_STARTS.items()
    }


def monotonic_violation_count(cumulative: dict[str, np.ndarray]) -> int:
    ordered = np.column_stack(
        [cumulative[name] for name in ("075", "100", "150", "200")]
    )
    return int((np.diff(ordered, axis=1) > 1e-12).any(axis=1).sum())


def ranked_probability_score(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> float:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    cut_points = np.arange(len(MAGNITUDE_BIN_LABELS) - 1)
    predicted_cdf = np.cumsum(p, axis=1)[:, :-1]
    observed_cdf = y[:, None] <= cut_points[None, :]
    row_score = np.square(predicted_cdf - observed_cdf).mean(axis=1)
    return float(np.average(row_score, weights=sample_weight))


def magnitude_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    sample_weight = np.asarray(weights, dtype=float)
    if not len(y) or len(y) != len(p) or len(p) != len(sample_weight):
        raise ValueError("magnitude metric arrays must be non-empty and aligned")
    if not set(np.unique(y)).issubset(set(range(len(MAGNITUDE_BIN_LABELS)))):
        raise ValueError("magnitude metric labels are outside the registered bins")
    if not np.isfinite(sample_weight).all() or (sample_weight <= 0.0).any():
        raise ValueError("magnitude metric weights must be positive and finite")
    cumulative = cumulative_probabilities(p)
    metrics: dict[str, float] = {
        "multiclass_logloss": float(
            log_loss(
                y,
                p,
                labels=list(range(len(MAGNITUDE_BIN_LABELS))),
                sample_weight=sample_weight,
            )
        ),
        "ranked_probability_score": ranked_probability_score(y, p, sample_weight),
        "monotonic_violation_rate": float(
            monotonic_violation_count(cumulative) / len(y)
        ),
    }
    for name, start in CUMULATIVE_CLASS_STARTS.items():
        target = (y >= start).astype(int)
        probability = np.clip(cumulative[name], 1e-8, 1.0 - 1e-8)
        prevalence = float(np.average(target, weights=sample_weight))
        threshold = float(np.quantile(probability, 0.9))
        top = probability >= threshold
        top_rate = float(np.average(target[top], weights=sample_weight[top]))
        metrics[f"ge_{name}_prevalence"] = prevalence
        metrics[f"ge_{name}_brier"] = float(
            np.average(np.square(probability - target), weights=sample_weight)
        )
        metrics[f"ge_{name}_logloss"] = float(
            log_loss(
                target,
                np.column_stack([1.0 - probability, probability]),
                labels=[0, 1],
                sample_weight=sample_weight,
            )
        )
        metrics[f"ge_{name}_pr_auc"] = (
            float(average_precision_score(target, probability, sample_weight=sample_weight))
            if np.unique(target).size == 2
            else np.nan
        )
        metrics[f"ge_{name}_roc_auc"] = (
            float(roc_auc_score(target, probability, sample_weight=sample_weight))
            if np.unique(target).size == 2
            else np.nan
        )
        metrics[f"ge_{name}_top_decile_rate"] = top_rate
        metrics[f"ge_{name}_top_decile_lift"] = (
            top_rate / prevalence if prevalence > 0.0 else np.nan
        )
    return metrics


def _decisions(dataset: LargeMoveDecisionDataset) -> pd.DataFrame:
    work = dataset.decisions.copy().reset_index(drop=True)
    required = {
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "label_start",
        "label_end",
        "magnitude_class",
        "magnitude_target_valid",
    }
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"magnitude decisions missing columns: {missing}")
    if len(work) != len(dataset.tabular):
        raise ValueError("magnitude decisions and feature rows do not align")
    if work.duplicated(["window_id", "step"]).any():
        raise ValueError("magnitude decision keys must be unique")
    for column in ("decision_time", "label_start", "label_end"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    work["model_target_valid"] = work["magnitude_target_valid"].astype(bool)
    valid = work["model_target_valid"]
    if not work.loc[valid, "magnitude_class"].isin(range(5)).all():
        raise ValueError("valid magnitude rows require a registered class")
    expected_end = work["decision_time"] + pd.Timedelta(minutes=120)
    if not work["label_end"].eq(expected_end).all():
        raise ValueError("all magnitude label intervals must end at t+120 minutes")
    assert_magnitude_feature_isolation(dataset.tabular_features)
    return work


def _fit_hash(
    x: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    config: LargeMoveModelConfig,
) -> str:
    digest = hashlib.sha256(b"O_magnitude_xgboost")
    digest.update(np.ascontiguousarray(x).view(np.uint8))
    digest.update(np.asarray(labels, dtype=np.int8).tobytes())
    digest.update(np.asarray(weights, dtype=np.float64).tobytes())
    digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _score_frame(
    decisions: pd.DataFrame,
    positions: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
    fold_id: str,
) -> pd.DataFrame:
    rows = decisions.iloc[positions].reset_index(drop=True)
    cumulative = cumulative_probabilities(probabilities)
    if monotonic_violation_count(cumulative):
        raise AssertionError("cumulative magnitude probabilities are not nested")
    result = pd.DataFrame(
        {
            "model": "xgboost_multiclass_magnitude",
            "fold_id": fold_id,
            "window_id": rows["window_id"],
            "channel_episode_id": rows["channel_episode_id"],
            "side": rows["side"],
            "step": rows["step"].astype(int),
            "decision_time": rows["decision_time"],
            "label_start": rows["label_start"],
            "label_end": rows["label_end"],
            "magnitude_class": rows["magnitude_class"].astype(int),
            "magnitude_ratio": rows["magnitude_ratio"].astype(float),
            "tth_100_min": rows["tth_100_min"].astype(float),
            "sample_weight": weights,
        }
    )
    for class_number in range(len(MAGNITUDE_BIN_LABELS)):
        result[f"p_bin_{class_number}"] = probabilities[:, class_number]
    for name, values in cumulative.items():
        result[f"p_ge_{name}"] = values
        result[f"y_ge_{name}"] = (
            result["magnitude_class"] >= CUMULATIVE_CLASS_STARTS[name]
        ).astype(np.int8)
    return result


def run_magnitude_fold(
    fold: PurgedFold,
    dataset: LargeMoveDecisionDataset,
    config: MagnitudeOOFConfig = MagnitudeOOFConfig(),
) -> MagnitudeOOFResult:
    """Fit on past episodes, calibrate on early episodes, score untouched outer rows."""
    decisions = _decisions(dataset)
    fit, early, reserved = _partitions(decisions, fold.train, config.fold)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    if not len(outer):
        raise ValueError(f"fold {fold.fold_id} has no valid magnitude rows")
    fit_weights = _half_open_uniqueness(decisions, fit)
    score_positions = np.concatenate([early, outer])
    raw = fit_predict_magnitude_xgboost(
        train_x=dataset.tabular[fit],
        labels=decisions.iloc[fit]["magnitude_class"].to_numpy(int),
        sample_weight=fit_weights,
        score_x=dataset.tabular[score_positions],
        config=config.model,
    )
    early_weights = _half_open_uniqueness(decisions, early)
    temperature = _fit_temperature(
        raw.logits[: len(early)],
        decisions.iloc[early]["magnitude_class"].to_numpy(int),
        early_weights,
    )
    outer_probability = _probabilities(raw.logits[len(early):], temperature)
    outer_weights = _half_open_uniqueness(decisions, outer)
    scores = _score_frame(
        decisions, outer, outer_probability, outer_weights, fold.fold_id
    )
    train_positions = np.concatenate([fit, early, reserved])
    train_episodes = set(decisions.iloc[train_positions]["channel_episode_id"])
    outer_episodes = set(decisions.iloc[outer]["channel_episode_id"])
    metric = magnitude_metrics(
        scores["magnitude_class"].to_numpy(int),
        scores[[f"p_bin_{number}" for number in range(5)]].to_numpy(float),
        scores["sample_weight"].to_numpy(float),
    )
    fold_audit = pd.DataFrame(
        [
            {
                "model": "xgboost_multiclass_magnitude",
                "fold": fold.fold_id,
                "feature_set": dataset.feature_set,
                "features": len(dataset.tabular_features),
                "fit_rows": len(fit),
                "early_rows": len(early),
                "reserved_calibration_rows": len(reserved),
                "validation_rows": len(outer),
                "fit_episodes": decisions.iloc[fit]["channel_episode_id"].nunique(),
                "early_episodes": decisions.iloc[early]["channel_episode_id"].nunique(),
                "reserved_calibration_episodes": decisions.iloc[reserved][
                    "channel_episode_id"
                ].nunique(),
                "validation_episodes": decisions.iloc[outer][
                    "channel_episode_id"
                ].nunique(),
                "episode_overlap": len(train_episodes & outer_episodes),
                "train_label_end_max": decisions.iloc[train_positions]["label_end"].max(),
                "validation_start": fold.valid_start,
                "fit_model_hash": _fit_hash(
                    dataset.tabular[fit],
                    decisions.iloc[fit]["magnitude_class"].to_numpy(int),
                    fit_weights,
                    config.model,
                ),
                **metric,
            }
        ]
    )
    if fold_audit.iloc[0]["episode_overlap"] != 0:
        raise AssertionError("magnitude OOF split channel episodes")
    if fold_audit.iloc[0]["train_label_end_max"] > fold.valid_start:
        raise AssertionError("magnitude training label crosses validation")
    effective_weight = float(
        early_weights.sum() ** 2 / np.square(early_weights).sum()
    )
    calibration_audit = pd.DataFrame(
        [
            {
                "model": "xgboost_multiclass_magnitude",
                "fold": fold.fold_id,
                "temperature": temperature,
                "early_rows": len(early),
                "early_episodes": decisions.iloc[early][
                    "channel_episode_id"
                ].nunique(),
                "early_effective_weight": effective_weight,
                "reserved_calibration_rows": len(reserved),
                "calibration_target": "one temperature for all five bins",
                "threshold_status": "deferred; Notebook O is threshold-free",
            }
        ]
    )
    return MagnitudeOOFResult(scores, fold_audit, calibration_audit)


def run_magnitude_oof(
    dataset: LargeMoveDecisionDataset,
    config: MagnitudeOOFConfig = MagnitudeOOFConfig(),
) -> MagnitudeOOFResult:
    decisions = _decisions(dataset)
    folds = [fold for fold in _outer_folds(decisions) if len(fold.train) and len(fold.valid)]
    if not folds:
        raise RuntimeError("no magnitude OOF folds are available")
    results = [run_magnitude_fold(fold, dataset, config) for fold in folds]
    scores = pd.concat([result.scores for result in results], ignore_index=True)
    if scores.duplicated(["window_id", "step"]).any():
        raise AssertionError("magnitude OOF keys overlap between folds")
    if monotonic_violation_count(
        {
            name: scores[f"p_ge_{name}"].to_numpy(float)
            for name in CUMULATIVE_CLASS_STARTS
        }
    ):
        raise AssertionError("published magnitude probabilities are not nested")
    return MagnitudeOOFResult(
        scores,
        pd.concat([result.fold_audit for result in results], ignore_index=True),
        pd.concat([result.calibration_audit for result in results], ignore_index=True),
    )


__all__ = [
    "CUMULATIVE_CLASS_STARTS",
    "MagnitudeOOFConfig",
    "MagnitudeOOFResult",
    "cumulative_probabilities",
    "magnitude_metrics",
    "monotonic_violation_count",
    "ranked_probability_score",
    "run_magnitude_fold",
    "run_magnitude_oof",
]
