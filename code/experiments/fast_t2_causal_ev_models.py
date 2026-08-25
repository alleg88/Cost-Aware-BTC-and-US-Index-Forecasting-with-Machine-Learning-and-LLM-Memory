"""Matched pooled CatBoost/XGBoost models for causal outcome EV."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EV_MODEL_NAMES = ("catboost", "xgboost")
TEMPERATURE_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


@dataclass(frozen=True)
class CausalEVPrediction:
    outcome_probabilities: np.ndarray
    timeout_gross_r: np.ndarray
    timeout_target_low: float
    timeout_target_high: float


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply scalar temperature to already-normalised probabilities."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("outcome probabilities must have shape (n, 3)")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    clipped = np.clip(values, 1e-12, 1.0)
    logits = np.log(clipped) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    calibrated = np.exp(logits)
    calibrated /= calibrated.sum(axis=1, keepdims=True)
    return calibrated


def select_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    """Choose a registered temperature by weighted multiclass log loss."""
    values = np.asarray(probabilities, dtype=float)
    target = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("outcome probabilities must have shape (n, 3)")
    if not (len(values) == len(target) == len(weights)) or not len(values):
        raise ValueError("calibration arrays must be non-empty and aligned")
    if not set(np.unique(target)).issubset({0, 1, 2}):
        raise ValueError("outcome labels must be 0, 1, or 2")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("calibration weights must be positive and finite")
    losses: list[tuple[float, float]] = []
    for temperature in TEMPERATURE_GRID:
        calibrated = apply_temperature(values, temperature)
        chosen = np.clip(calibrated[np.arange(len(target)), target], 1e-12, 1.0)
        loss = float(np.average(-np.log(chosen), weights=weights))
        losses.append((loss, float(temperature)))
    return min(losses, key=lambda item: (item[0], item[1]))[1]


def _validate_inputs(
    model_name: str,
    train_x: np.ndarray,
    labels: np.ndarray,
    gross_r: np.ndarray,
    weights: np.ndarray,
    score_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model_name not in EV_MODEL_NAMES:
        raise ValueError(f"unsupported causal EV model: {model_name}")
    train_x = np.asarray(train_x, dtype=float)
    score_x = np.asarray(score_x, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    gross_r = np.asarray(gross_r, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if train_x.ndim != 2 or score_x.ndim != 2:
        raise ValueError("causal EV features must be two-dimensional")
    if train_x.shape[1] != score_x.shape[1]:
        raise ValueError("training and scoring feature counts must match")
    if not (len(train_x) == len(labels) == len(gross_r) == len(weights)):
        raise ValueError("causal EV training arrays must align")
    if not len(train_x) or not len(score_x):
        raise ValueError("causal EV inputs cannot be empty")
    if set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError("causal EV training needs SL, TP, and timeout classes")
    if not np.isfinite(gross_r).all():
        raise ValueError("gross-R target must be finite")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("training weights must be positive and finite")
    return train_x, labels, gross_r, weights, score_x


def _classifier(model_name: str):
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            loss_function="MultiClass",
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
            thread_count=1,
        )
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def _regressor(model_name: str):
    if model_name == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=300,
            depth=4,
            learning_rate=0.03,
            l2_leaf_reg=10.0,
            loss_function="RMSE",
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
            thread_count=1,
        )
    from xgboost import XGBRegressor

    return XGBRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )


def fit_predict_causal_ev_model(
    model_name: str,
    train_x: np.ndarray,
    labels: np.ndarray,
    gross_r: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
) -> CausalEVPrediction:
    """Fit one pooled outcome classifier plus its timeout-magnitude regressor."""
    from sklearn.impute import SimpleImputer

    train_x, labels, gross_r, weights, score_x = _validate_inputs(
        model_name, train_x, labels, gross_r, sample_weight, score_x
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train_x)
    predict_x = imputer.transform(score_x)

    classifier = _classifier(model_name)
    classifier.fit(fit_x, labels, sample_weight=weights)
    probabilities = np.asarray(classifier.predict_proba(predict_x), dtype=float)
    if probabilities.shape != (len(score_x), 3):
        raise AssertionError("classifier did not return all three outcome classes")

    timeout_mask = labels == 2
    if int(timeout_mask.sum()) < 10:
        raise ValueError("timeout regression needs at least ten training outcomes")
    timeout_target = gross_r[timeout_mask]
    low, high = np.quantile(timeout_target, [0.01, 0.99])
    low = float(low)
    high = float(high)
    if low >= high:
        centre = float(np.mean(timeout_target))
        low, high = centre - 1e-9, centre + 1e-9
    robust_timeout = np.clip(timeout_target, low, high)
    regressor = _regressor(model_name)
    regressor.fit(
        fit_x[timeout_mask],
        robust_timeout,
        sample_weight=weights[timeout_mask],
    )
    timeout_prediction = np.asarray(regressor.predict(predict_x), dtype=float)
    timeout_prediction = np.clip(timeout_prediction, low, high)
    return CausalEVPrediction(
        outcome_probabilities=probabilities,
        timeout_gross_r=timeout_prediction,
        timeout_target_low=low,
        timeout_target_high=high,
    )
