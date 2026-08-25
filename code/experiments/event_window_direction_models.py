"""Fold-local forced-choice direction models for Notebook V."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DirectionModelConfig:
    xgb_estimators: int = 300
    xgb_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_min_child_weight: float = 20.0
    xgb_reg_lambda: float = 10.0
    logreg_c: float = 1.0
    logreg_max_iter: int = 2_000
    random_seed: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class ValueLogRegPrediction:
    score: np.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class DeltaXGBoostPrediction:
    predicted_delta_r: np.ndarray
    metadata: dict[str, object]


def _non_tie_arrays(
    train_x: np.ndarray,
    delta_r: np.ndarray,
    uniqueness: np.ndarray,
    score_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=float)
    score = np.asarray(score_x, dtype=float)
    delta = np.asarray(delta_r, dtype=float)
    weights = np.asarray(uniqueness, dtype=float)
    if train.ndim != 2 or score.ndim != 2 or train.shape[1] != score.shape[1]:
        raise ValueError("training and score features must be aligned matrices")
    if not len(train) or not len(score) or len(train) != len(delta) or len(delta) != len(weights):
        raise ValueError("training arrays must be non-empty and aligned")
    if not np.isfinite(delta).all():
        raise ValueError("delta_r must be finite")

    non_tie = np.abs(delta) > _TIE_TOLERANCE
    fit_delta = delta[non_tie]
    fit_weights = weights[non_tie]
    if set(np.unique(fit_delta > 0.0)) != {False, True}:
        raise ValueError("fit rows require both direction classes")
    if not np.isfinite(fit_weights).all() or (fit_weights <= 0.0).any():
        raise ValueError("non-tie uniqueness weights must be positive and finite")
    return (
        np.where(np.isfinite(train[non_tie]), train[non_tie], np.nan),
        fit_delta,
        fit_weights,
        np.where(np.isfinite(score), score, np.nan),
    )


def _impute(
    train_x: np.ndarray, score_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    return imputer.fit_transform(train_x), imputer.transform(score_x), imputer


def fit_predict_value_logreg(
    *,
    train_x: np.ndarray,
    delta_r: np.ndarray,
    uniqueness: np.ndarray,
    score_x: np.ndarray,
    config: DirectionModelConfig = DirectionModelConfig(),
) -> ValueLogRegPrediction:
    """Fit value-weighted binary LogReg and score every supplied activation."""
    train, delta, weights, score = _non_tie_arrays(
        train_x, delta_r, uniqueness, score_x
    )
    fit_x, predict_x, imputer = _impute(train, score)
    scaler = StandardScaler()
    fit_x = scaler.fit_transform(fit_x)
    predict_x = scaler.transform(predict_x)
    model = LogisticRegression(
        C=config.logreg_c,
        solver="lbfgs",
        max_iter=config.logreg_max_iter,
        random_state=config.random_seed,
    )
    target = (delta > 0.0).astype(np.int8)
    sample_weight = weights * np.minimum(np.abs(delta), 3.0)
    model.fit(fit_x, target, sample_weight=sample_weight)
    probability = np.asarray(model.predict_proba(predict_x), dtype=float)[:, 1]
    return ValueLogRegPrediction(
        score=probability,
        metadata={
            "model": "value_logreg",
            "config": asdict(config),
            "fit_rows": len(delta),
            "fit_median": imputer.statistics_.astype(float).tolist(),
            "scaled": True,
        },
    )


def fit_predict_delta_xgboost(
    *,
    train_x: np.ndarray,
    delta_r: np.ndarray,
    uniqueness: np.ndarray,
    score_x: np.ndarray,
    config: DirectionModelConfig = DirectionModelConfig(),
) -> DeltaXGBoostPrediction:
    """Fit the fixed continuous-delta XGBoost and score every activation."""
    from xgboost import XGBRegressor

    train, delta, weights, score = _non_tie_arrays(
        train_x, delta_r, uniqueness, score_x
    )
    fit_x, predict_x, imputer = _impute(train, score)
    model = XGBRegressor(
        objective="reg:squarederror",
        eval_metric="rmse",
        n_estimators=config.xgb_estimators,
        max_depth=config.xgb_depth,
        learning_rate=config.xgb_learning_rate,
        min_child_weight=config.xgb_min_child_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=config.xgb_reg_lambda,
        tree_method="hist",
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
        verbosity=0,
    )
    model.fit(fit_x, delta, sample_weight=weights)
    prediction = np.asarray(model.predict(predict_x), dtype=float)
    return DeltaXGBoostPrediction(
        predicted_delta_r=prediction,
        metadata={
            "model": "delta_xgboost",
            "config": asdict(config),
            "fit_rows": len(delta),
            "fit_median": imputer.statistics_.astype(float).tolist(),
            "scaled": False,
        },
    )


__all__ = [
    "DeltaXGBoostPrediction",
    "DirectionModelConfig",
    "ValueLogRegPrediction",
    "fit_predict_delta_xgboost",
    "fit_predict_value_logreg",
]
