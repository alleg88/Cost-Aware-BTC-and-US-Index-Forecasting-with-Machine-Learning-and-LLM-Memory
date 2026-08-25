"""Purged OOF evaluation for the standalone Notebook N opportunity head."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_cost_aware_oof import (
    CostAwareFoldConfig,
    _outer_folds,
    _partitions,
)
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.event_window_large_move_models import (
    LargeMoveModelConfig,
    fit_predict_opportunity,
)
from experiments.event_window_large_move_oof import _fit_binary_platt
from experiments.event_window_tail_oof import _half_open_uniqueness


@dataclass(frozen=True)
class OpportunityOOFConfig:
    fold: CostAwareFoldConfig = field(default_factory=CostAwareFoldConfig)
    model: LargeMoveModelConfig = field(default_factory=LargeMoveModelConfig)


@dataclass(frozen=True)
class OpportunityOOFResult:
    arm: str
    model_name: str
    scores: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame


def _fit_hash(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    arm: str,
    model_name: str,
) -> str:
    digest = hashlib.sha256(f"{arm}:{model_name}".encode("utf-8"))
    digest.update(np.ascontiguousarray(x).view(np.uint8))
    digest.update(np.asarray(y, dtype=np.int8).tobytes())
    digest.update(np.asarray(weights, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _sigmoid(logit: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    value = slope * np.asarray(logit, dtype=float) + intercept
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))


def opportunity_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    weights: np.ndarray,
) -> dict[str, float]:
    """Weighted threshold-free magnitude diagnostics."""
    y = np.asarray(labels, dtype=int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    weight = np.asarray(weights, dtype=float)
    if not len(y) or len(y) != len(p) or len(p) != len(weight):
        raise ValueError("metric arrays must be non-empty and aligned")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("opportunity metrics require both binary classes")
    prevalence = float(np.average(y, weights=weight))
    threshold = float(np.quantile(p, 0.9))
    top = p >= threshold
    top_rate = float(np.average(y[top], weights=weight[top]))
    return {
        "opportunity_pr_auc": float(average_precision_score(y, p, sample_weight=weight)),
        "opportunity_roc_auc": float(roc_auc_score(y, p, sample_weight=weight)),
        "opportunity_brier": float(np.average(np.square(p - y), weights=weight)),
        "opportunity_logloss": float(
            log_loss(
                y,
                np.column_stack([1.0 - p, p]),
                labels=[0, 1],
                sample_weight=weight,
            )
        ),
        "opportunity_prevalence": prevalence,
        "top_decile_opportunity_rate": top_rate,
        "top_decile_lift": top_rate / prevalence if prevalence > 0.0 else np.nan,
    }


def _opportunity_decisions(dataset: LargeMoveDecisionDataset) -> pd.DataFrame:
    decisions = dataset.decisions.copy().reset_index(drop=True)
    required = {
        "opportunity_code",
        "opportunity_target_valid",
        "decision_time",
        "label_start",
        "label_end",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"opportunity decisions missing columns: {missing}")
    for column in ("decision_time", "label_start", "label_end"):
        decisions[column] = pd.to_datetime(decisions[column], utc=True, errors="raise")
    decisions["model_target_valid"] = decisions["opportunity_target_valid"].astype(bool)
    return decisions


def run_opportunity_fold(
    arm: str,
    model_name: str,
    fold: PurgedFold,
    dataset: LargeMoveDecisionDataset,
    config: OpportunityOOFConfig = OpportunityOOFConfig(),
) -> OpportunityOOFResult:
    """Fit and Platt-calibrate one binary head; outer labels never affect fit."""
    if model_name not in {"logreg", "xgboost"}:
        raise ValueError("opportunity model must be logreg or xgboost")
    decisions = _opportunity_decisions(dataset)
    fit, early, calibration = _partitions(decisions, fold.train, config.fold)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    if not len(outer):
        raise ValueError(f"fold {fold.fold_id} has no valid opportunity labels")
    fit_weights = _half_open_uniqueness(decisions, fit)
    score_positions = np.concatenate([early, outer])
    raw = fit_predict_opportunity(
        model_name,
        train_x=dataset.tabular[fit],
        labels=decisions.iloc[fit]["opportunity_code"].to_numpy(int),
        sample_weight=fit_weights,
        score_x=dataset.tabular[score_positions],
        config=config.model,
    )
    if raw.opportunity_logit is None:
        raise AssertionError("binary opportunity model did not expose its logit")
    early_weights = _half_open_uniqueness(decisions, early)
    slope, intercept, fallback = _fit_binary_platt(
        raw.opportunity_logit[: len(early)],
        decisions.iloc[early]["opportunity_code"].to_numpy(int),
        early_weights,
    )
    outer_probability = _sigmoid(raw.opportunity_logit[len(early) :], slope, intercept)
    outer_weights = _half_open_uniqueness(decisions, outer)
    rows = decisions.iloc[outer].reset_index(drop=True)
    scores = pd.DataFrame(
        {
            "arm": arm,
            "model": model_name,
            "fold_id": fold.fold_id,
            "window_id": rows["window_id"],
            "channel_episode_id": rows["channel_episode_id"],
            "step": rows["step"].astype(int),
            "decision_time": rows["decision_time"],
            "label_start": rows["label_start"],
            "label_end": rows["label_end"],
            "opportunity_code": rows["opportunity_code"].astype(int),
            "p_hit": outer_probability,
            "sample_weight": outer_weights,
        }
    )
    train_positions = np.concatenate([fit, early, calibration])
    train_episodes = set(decisions.iloc[train_positions]["channel_episode_id"])
    outer_episodes = set(decisions.iloc[outer]["channel_episode_id"])
    metric = opportunity_metrics(
        scores["opportunity_code"].to_numpy(int),
        scores["p_hit"].to_numpy(float),
        scores["sample_weight"].to_numpy(float),
    )
    fold_audit = pd.DataFrame(
        [
            {
                "arm": arm,
                "model": model_name,
                "fold": fold.fold_id,
                "feature_set": dataset.feature_set,
                "features": len(dataset.tabular_features),
                "fit_rows": len(fit),
                "early_rows": len(early),
                "reserved_calibration_rows": len(calibration),
                "validation_rows": len(outer),
                "fit_episodes": decisions.iloc[fit]["channel_episode_id"].nunique(),
                "early_episodes": decisions.iloc[early]["channel_episode_id"].nunique(),
                "reserved_calibration_episodes": decisions.iloc[calibration][
                    "channel_episode_id"
                ].nunique(),
                "validation_episodes": decisions.iloc[outer]["channel_episode_id"].nunique(),
                "episode_overlap": len(train_episodes & outer_episodes),
                "train_label_end_max": decisions.iloc[train_positions]["label_end"].max(),
                "validation_start": fold.valid_start,
                "fit_model_hash": _fit_hash(
                    dataset.tabular[fit],
                    decisions.iloc[fit]["opportunity_code"].to_numpy(int),
                    fit_weights,
                    arm=arm,
                    model_name=model_name,
                ),
                **metric,
            }
        ]
    )
    if fold_audit.iloc[0]["episode_overlap"] != 0:
        raise AssertionError("opportunity OOF split channel episodes")
    if fold_audit.iloc[0]["train_label_end_max"] > fold.valid_start:
        raise AssertionError("opportunity training label crosses validation")
    effective_weight = float(
        early_weights.sum() ** 2 / np.square(early_weights).sum()
    )
    calibration_audit = pd.DataFrame(
        [
            {
                "arm": arm,
                "model": model_name,
                "fold": fold.fold_id,
                "platt_slope": slope,
                "platt_intercept": intercept,
                "identity_fallback": fallback,
                "early_rows": len(early),
                "early_episodes": decisions.iloc[early]["channel_episode_id"].nunique(),
                "early_effective_weight": effective_weight,
                "reserved_calibration_rows": len(calibration),
                "threshold_status": "deferred; Notebook N is threshold-free",
            }
        ]
    )
    return OpportunityOOFResult(arm, model_name, scores, fold_audit, calibration_audit)


def run_opportunity_model_oof(
    arm: str,
    model_name: str,
    dataset: LargeMoveDecisionDataset,
    config: OpportunityOOFConfig = OpportunityOOFConfig(),
) -> OpportunityOOFResult:
    decisions = _opportunity_decisions(dataset)
    folds = [fold for fold in _outer_folds(decisions) if len(fold.train) and len(fold.valid)]
    if not folds:
        raise RuntimeError("no opportunity OOF folds are available")
    results = [run_opportunity_fold(arm, model_name, fold, dataset, config) for fold in folds]
    scores = pd.concat([result.scores for result in results], ignore_index=True)
    if scores.duplicated(["window_id", "step"]).any():
        raise AssertionError("opportunity OOF keys overlap between folds")
    return OpportunityOOFResult(
        arm,
        model_name,
        scores,
        pd.concat([result.fold_audit for result in results], ignore_index=True),
        pd.concat([result.calibration_audit for result in results], ignore_index=True),
    )


def assert_identical_opportunity_keys(results: list[OpportunityOOFResult]) -> None:
    if not results:
        raise ValueError("results cannot be empty")
    reference = set(
        results[0].scores[["window_id", "step"]].itertuples(index=False, name=None)
    )
    for result in results[1:]:
        keys = set(result.scores[["window_id", "step"]].itertuples(index=False, name=None))
        if keys != reference:
            raise AssertionError("opportunity model OOF keys differ")


__all__ = [
    "OpportunityOOFConfig",
    "OpportunityOOFResult",
    "assert_identical_opportunity_keys",
    "opportunity_metrics",
    "run_opportunity_fold",
    "run_opportunity_model_oof",
]
