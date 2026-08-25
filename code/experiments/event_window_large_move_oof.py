"""Purged walk-forward OOF orchestration for adaptive large moves."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression

from evaluation.channel_window_validation import PurgedFold
from evaluation.event_window_large_move_policy import (
    LargeMovePolicyConfig,
    select_calibration_threshold,
)
from experiments.event_window_cost_aware_oof import (
    CostAwareFoldConfig,
    _fit_temperature,
    _outer_folds,
    _partitions,
    _probabilities,
)
from experiments.event_window_large_move_dataset import (
    AdaptiveMoveConfig,
    LargeMoveDecisionDataset,
)
from experiments.event_window_large_move_models import (
    LargeMoveModelConfig,
    fit_predict_multiclass,
    fit_predict_two_stage_xgboost,
)
from experiments.event_window_tail_oof import _half_open_uniqueness


@dataclass(frozen=True)
class LargeMoveOOFConfig:
    fold: CostAwareFoldConfig = field(default_factory=CostAwareFoldConfig)
    model: LargeMoveModelConfig = field(default_factory=LargeMoveModelConfig)
    policy: LargeMovePolicyConfig = field(default_factory=LargeMovePolicyConfig)
    target: AdaptiveMoveConfig = field(default_factory=AdaptiveMoveConfig)


@dataclass(frozen=True)
class LargeMoveOOFResult:
    model_name: str
    scores: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame
    threshold_frontier: pd.DataFrame


def _fit_hash(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, model_name: str
) -> str:
    digest = hashlib.sha256(model_name.encode("utf-8"))
    digest.update(np.ascontiguousarray(x).view(np.uint8))
    digest.update(np.asarray(y, dtype=np.int8).tobytes())
    digest.update(np.asarray(weights, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _score_frame(
    decisions: pd.DataFrame,
    positions: np.ndarray,
    probabilities: np.ndarray,
    *,
    model_name: str,
    fold_id: str,
) -> pd.DataFrame:
    rows = decisions.iloc[positions].reset_index(drop=True)
    return pd.DataFrame(
        {
            "model": model_name,
            "fold_id": fold_id,
            "window_id": rows["window_id"],
            "channel_episode_id": rows["channel_episode_id"],
            "side": rows["side"],
            "step": rows["step"].astype(int),
            "decision_time": rows["decision_time"],
            "p_no_big": probabilities[:, 0],
            "p_up_big": probabilities[:, 1],
            "p_down_big": probabilities[:, 2],
        }
    )


def _fit_binary_platt(
    logit: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[float, float, bool]:
    values = np.asarray(logit, dtype=float)
    target = np.asarray(labels, dtype=int)
    sample_weight = np.asarray(weights, dtype=float)
    if not len(values) or len(values) != len(target) or len(target) != len(sample_weight):
        raise ValueError("binary calibration arrays must be non-empty and aligned")
    if set(np.unique(target)) != {0, 1}:
        return 1.0, 0.0, True
    calibrator = LogisticRegression(C=1_000.0, solver="lbfgs", max_iter=1_000)
    calibrator.fit(values.reshape(-1, 1), target, sample_weight=sample_weight)
    return (
        float(calibrator.coef_[0, 0]),
        float(calibrator.intercept_[0]),
        False,
    )


def _two_stage_probabilities(
    opportunity_logit: np.ndarray,
    direction_logit: np.ndarray,
    *,
    opportunity_slope: float,
    opportunity_intercept: float,
    direction_slope: float,
    direction_intercept: float,
) -> np.ndarray:
    opportunity = (
        opportunity_slope * np.asarray(opportunity_logit, dtype=float)
        + opportunity_intercept
    )
    direction = (
        direction_slope * np.asarray(direction_logit, dtype=float)
        + direction_intercept
    )
    p_big = 1.0 / (1.0 + np.exp(-np.clip(opportunity, -40.0, 40.0)))
    p_up_given_big = 1.0 / (1.0 + np.exp(-np.clip(direction, -40.0, 40.0)))
    probabilities = np.column_stack(
        [1.0 - p_big, p_big * p_up_given_big, p_big * (1.0 - p_up_given_big)]
    )
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def _metrics(
    probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    weight = np.asarray(weights, dtype=float)
    big = y != 0
    p_big = p[:, 1] + p[:, 2]
    direction = np.where(p[:, 1] >= p[:, 2], 1, 2)
    top = p_big >= np.quantile(p_big, 0.9)
    prevalence = float(np.average(big, weights=weight))
    p_up_given_big = np.divide(
        p[:, 1],
        p_big,
        out=np.full(len(p), 0.5, dtype=float),
        where=p_big > 0.0,
    )
    direction_target = y[big] == 1
    return {
        "multiclass_logloss": float(
            log_loss(y, p, labels=[0, 1, 2], sample_weight=weight)
        ),
        "big_roc_auc": float(roc_auc_score(big, p_big, sample_weight=weight)),
        "big_pr_auc": float(
            average_precision_score(big, p_big, sample_weight=weight)
        ),
        "big_logloss": float(
            log_loss(big, np.column_stack([1.0 - p_big, p_big]), labels=[0, 1], sample_weight=weight)
        ),
        "big_brier": float(np.average((p_big - big.astype(float)) ** 2, weights=weight)),
        "big_prevalence": prevalence,
        "top_decile_big_rate": float(np.average(big[top], weights=weight[top])),
        "top_decile_lift": float(
            np.average(big[top], weights=weight[top]) / prevalence
        )
        if prevalence
        else np.nan,
        "direction_accuracy_on_big": float(
            np.average(direction[big] == y[big], weights=weight[big])
        )
        if big.any()
        else np.nan,
        "direction_logloss_on_big": float(
            log_loss(
                direction_target,
                np.column_stack([1.0 - p_up_given_big[big], p_up_given_big[big]]),
                labels=[0, 1],
                sample_weight=weight[big],
            )
        )
        if big.any()
        else np.nan,
        "direction_brier_on_big": float(
            np.average(
                (p_up_given_big[big] - direction_target.astype(float)) ** 2,
                weights=weight[big],
            )
        )
        if big.any()
        else np.nan,
    }


def run_large_move_fold(
    model_name: str,
    fold: PurgedFold,
    dataset: LargeMoveDecisionDataset,
    config: LargeMoveOOFConfig = LargeMoveOOFConfig(),
) -> LargeMoveOOFResult:
    """Fit, calibrate, select a threshold, and score one untouched outer fold."""
    if model_name not in {"logreg", "xgboost", "xgboost_two_stage"}:
        raise ValueError("unsupported large-move model")
    decisions = dataset.decisions.copy().reset_index(drop=True)
    for column in ("decision_time", "label_start", "label_end"):
        decisions[column] = pd.to_datetime(decisions[column], utc=True, errors="coerce")
    fit, early, calibration = _partitions(decisions, fold.train, config.fold)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    if not len(outer):
        raise ValueError(f"fold {fold.fold_id} has no valid outer labels")
    weights = _half_open_uniqueness(decisions, fit)
    combined = np.concatenate([early, calibration, outer])
    fit_kwargs = dict(
        train_x=dataset.tabular[fit],
        labels=decisions.iloc[fit]["move_code"].to_numpy(int),
        sample_weight=weights,
        score_x=dataset.tabular[combined],
        config=config.model,
    )
    raw = (
        fit_predict_two_stage_xgboost(**fit_kwargs)
        if model_name == "xgboost_two_stage"
        else fit_predict_multiclass(model_name, **fit_kwargs)
    )
    early_stop = len(early)
    calibration_stop = early_stop + len(calibration)
    early_weights = _half_open_uniqueness(decisions, early)
    opportunity_slope = np.nan
    opportunity_intercept = np.nan
    direction_slope = np.nan
    direction_intercept = np.nan
    opportunity_identity_fallback = False
    direction_identity_fallback = False
    direction_early_rows = 0
    direction_early_episodes = 0
    direction_early_up = 0
    direction_early_down = 0
    early_effective_weight = float(
        early_weights.sum() ** 2 / np.square(early_weights).sum()
    )
    direction_early_effective_weight = 0.0
    if model_name == "xgboost_two_stage":
        if raw.opportunity_logit is None or raw.direction_logit is None:
            raise AssertionError("two-stage model must expose both binary logits")
        early_labels = decisions.iloc[early]["move_code"].to_numpy(int)
        (
            opportunity_slope,
            opportunity_intercept,
            opportunity_identity_fallback,
        ) = _fit_binary_platt(
            raw.opportunity_logit[:early_stop],
            (early_labels != 0).astype(int),
            early_weights,
        )
        early_big = early_labels != 0
        direction_early_rows = int(early_big.sum())
        direction_early_up = int((early_labels[early_big] == 1).sum())
        direction_early_down = int((early_labels[early_big] == 2).sum())
        direction_early_episodes = int(
            decisions.iloc[early[early_big]]["channel_episode_id"].nunique()
        )
        direction_weights = early_weights[early_big]
        if len(direction_weights):
            direction_early_effective_weight = float(
                direction_weights.sum() ** 2 / np.square(direction_weights).sum()
            )
        (
            direction_slope,
            direction_intercept,
            direction_identity_fallback,
        ) = _fit_binary_platt(
            raw.direction_logit[:early_stop][early_big],
            (early_labels[early_big] == 1).astype(int),
            direction_weights,
        )
        calibration_probabilities = _two_stage_probabilities(
            raw.opportunity_logit[early_stop:calibration_stop],
            raw.direction_logit[early_stop:calibration_stop],
            opportunity_slope=opportunity_slope,
            opportunity_intercept=opportunity_intercept,
            direction_slope=direction_slope,
            direction_intercept=direction_intercept,
        )
        outer_probabilities = _two_stage_probabilities(
            raw.opportunity_logit[calibration_stop:],
            raw.direction_logit[calibration_stop:],
            opportunity_slope=opportunity_slope,
            opportunity_intercept=opportunity_intercept,
            direction_slope=direction_slope,
            direction_intercept=direction_intercept,
        )
        temperature = np.nan
    else:
        temperature = _fit_temperature(
            raw.logits[:early_stop],
            decisions.iloc[early]["move_code"].to_numpy(int),
            early_weights,
        )
        calibration_probabilities = _probabilities(
            raw.logits[early_stop:calibration_stop], temperature
        )
        outer_probabilities = _probabilities(raw.logits[calibration_stop:], temperature)
    calibration_scores = _score_frame(
        decisions,
        calibration,
        calibration_probabilities,
        model_name=model_name,
        fold_id=fold.fold_id,
    )
    threshold, frontier = select_calibration_threshold(
        calibration_scores,
        decisions.iloc[calibration],
        config=config.policy,
        execution=config.target,
    )
    frontier.insert(0, "fold", fold.fold_id)
    frontier.insert(0, "model", model_name)
    scores = _score_frame(
        decisions,
        outer,
        outer_probabilities,
        model_name=model_name,
        fold_id=fold.fold_id,
    )
    scores["calibration_threshold"] = threshold
    outer_weights = _half_open_uniqueness(decisions, outer)
    metric = _metrics(
        outer_probabilities,
        decisions.iloc[outer]["move_code"].to_numpy(int),
        outer_weights,
    )
    train_positions = np.concatenate([fit, early, calibration])
    train_episodes = set(decisions.iloc[train_positions]["channel_episode_id"])
    outer_episodes = set(decisions.iloc[outer]["channel_episode_id"])
    fold_audit = pd.DataFrame(
        [
            {
                "model": model_name,
                "fold": fold.fold_id,
                "fit_rows": len(fit),
                "early_rows": len(early),
                "calibration_rows": len(calibration),
                "validation_rows": len(outer),
                "fit_episodes": decisions.iloc[fit]["channel_episode_id"].nunique(),
                "early_episodes": decisions.iloc[early]["channel_episode_id"].nunique(),
                "calibration_episodes": decisions.iloc[calibration]["channel_episode_id"].nunique(),
                "validation_episodes": decisions.iloc[outer]["channel_episode_id"].nunique(),
                "episode_overlap": len(train_episodes & outer_episodes),
                "train_label_end_max": decisions.iloc[train_positions]["label_end"].max(),
                "validation_start": fold.valid_start,
                "fit_model_hash": _fit_hash(
                    dataset.tabular[fit],
                    decisions.iloc[fit]["move_code"].to_numpy(int),
                    weights,
                    model_name,
                ),
                **metric,
            }
        ]
    )
    if fold_audit.iloc[0]["episode_overlap"] != 0:
        raise AssertionError("large-move OOF split channel episodes")
    if fold_audit.iloc[0]["train_label_end_max"] > fold.valid_start:
        raise AssertionError("large-move training label crosses validation")
    calibration_audit = pd.DataFrame(
        [
            {
                "model": model_name,
                "fold": fold.fold_id,
                "temperature": temperature,
                "opportunity_platt_slope": opportunity_slope,
                "opportunity_platt_intercept": opportunity_intercept,
                "opportunity_identity_fallback": opportunity_identity_fallback,
                "direction_platt_slope": direction_slope,
                "direction_platt_intercept": direction_intercept,
                "direction_identity_fallback": direction_identity_fallback,
                "direction_early_rows": direction_early_rows,
                "direction_early_up": direction_early_up,
                "direction_early_down": direction_early_down,
                "early_episodes": decisions.iloc[early]["channel_episode_id"].nunique(),
                "direction_early_episodes": direction_early_episodes,
                "early_effective_weight": early_effective_weight,
                "direction_early_effective_weight": direction_early_effective_weight,
                "selected_threshold": threshold,
                "early_rows": len(early),
                "calibration_rows": len(calibration),
            }
        ]
    )
    return LargeMoveOOFResult(
        model_name=model_name,
        scores=scores,
        fold_audit=fold_audit,
        calibration_audit=calibration_audit,
        threshold_frontier=frontier,
    )


def run_large_move_model_oof(
    model_name: str,
    dataset: LargeMoveDecisionDataset,
    config: LargeMoveOOFConfig = LargeMoveOOFConfig(),
) -> LargeMoveOOFResult:
    decisions = dataset.decisions.copy().reset_index(drop=True)
    folds = [fold for fold in _outer_folds(decisions) if len(fold.train) and len(fold.valid)]
    if not folds:
        raise RuntimeError("no large-move OOF folds are available")
    results = [run_large_move_fold(model_name, fold, dataset, config) for fold in folds]
    scores = pd.concat([result.scores for result in results], ignore_index=True)
    if scores.duplicated(["window_id", "step"]).any():
        raise AssertionError("large-move OOF keys overlap between folds")
    return LargeMoveOOFResult(
        model_name=model_name,
        scores=scores,
        fold_audit=pd.concat([result.fold_audit for result in results], ignore_index=True),
        calibration_audit=pd.concat(
            [result.calibration_audit for result in results], ignore_index=True
        ),
        threshold_frontier=pd.concat(
            [result.threshold_frontier for result in results], ignore_index=True
        ),
    )


def assert_identical_large_move_keys(results: list[LargeMoveOOFResult]) -> None:
    if not results:
        raise ValueError("results cannot be empty")
    reference = set(
        results[0].scores[["window_id", "step"]].itertuples(index=False, name=None)
    )
    for result in results[1:]:
        keys = set(result.scores[["window_id", "step"]].itertuples(index=False, name=None))
        if keys != reference:
            raise AssertionError("large-move model OOF keys differ")


__all__ = [
    "LargeMoveOOFConfig",
    "LargeMoveOOFResult",
    "assert_identical_large_move_keys",
    "run_large_move_fold",
    "run_large_move_model_oof",
]
