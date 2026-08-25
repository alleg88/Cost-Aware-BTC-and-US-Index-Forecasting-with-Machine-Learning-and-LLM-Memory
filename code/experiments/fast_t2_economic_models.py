"""Robust net-R regressors for Fast-T2 enter-versus-skip decisions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ECONOMIC_MODEL_NAMES = ("ridge", "catboost", "xgboost")
ECONOMIC_MODEL_VARIANTS = ("pooled", "split_side")


@dataclass(frozen=True)
class EconomicPrediction:
    scores: np.ndarray
    target_low: float
    target_high: float
    fitted_models: int


def _validate(
    model_name: str,
    variant: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    train_side: np.ndarray,
    score_x: np.ndarray,
    score_side: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model_name not in ECONOMIC_MODEL_NAMES:
        raise ValueError(f"unsupported economic model: {model_name}")
    if variant not in ECONOMIC_MODEL_VARIANTS:
        raise ValueError(f"unsupported economic model variant: {variant}")
    train_x = np.asarray(train_x, dtype=float)
    score_x = np.asarray(score_x, dtype=float)
    train_y = np.asarray(train_y, dtype=float)
    train_weight = np.asarray(train_weight, dtype=float)
    train_side = np.asarray(train_side, dtype=float)
    score_side = np.asarray(score_side, dtype=float)
    if train_x.ndim != 2 or score_x.ndim != 2:
        raise ValueError("economic inputs must be two-dimensional")
    if train_x.shape[1] != score_x.shape[1]:
        raise ValueError("economic feature counts must match")
    if not (len(train_x) == len(train_y) == len(train_weight) == len(train_side)):
        raise ValueError("economic training arrays must align")
    if len(score_x) != len(score_side):
        raise ValueError("economic score arrays must align")
    if not np.isfinite(train_y).all():
        raise ValueError("economic target must be finite")
    if not np.isfinite(train_weight).all() or (train_weight <= 0).any():
        raise ValueError("economic weights must be positive and finite")
    if not set(np.unique(train_side)).issubset({-1.0, 1.0}):
        raise ValueError("train_side must contain only -1 and +1")
    if not set(np.unique(score_side)).issubset({-1.0, 1.0}):
        raise ValueError("score_side must contain only -1 and +1")
    return train_x, train_y, train_weight, train_side, score_x, score_side


def _fit_one(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    score_x: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    from sklearn.impute import SimpleImputer

    low, high = np.quantile(train_y, [0.01, 0.99])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("economic training target has no usable spread")
    robust_y = np.clip(train_y, low, high)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train_x)
    predict_x = imputer.transform(score_x)
    if model_name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        model = Ridge(alpha=10.0)
        model.fit(
            scaler.fit_transform(fit_x), robust_y, sample_weight=train_weight
        )
        scores = model.predict(scaler.transform(predict_x))
    elif model_name == "catboost":
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
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
        model.fit(fit_x, robust_y, sample_weight=train_weight)
        scores = model.predict(predict_x)
    else:
        from xgboost import XGBRegressor

        model = XGBRegressor(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            reg_lambda=10.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        model.fit(fit_x, robust_y, sample_weight=train_weight)
        scores = model.predict(predict_x)
    return np.asarray(scores, dtype=float), float(low), float(high)


def fit_predict_economic_model(
    model_name: str,
    variant: str,
    train_static: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    train_side: np.ndarray,
    score_static: np.ndarray,
    score_side: np.ndarray,
) -> EconomicPrediction:
    """Predict robust expected net R; skipping a decision has payoff zero R."""
    train_x, target, weights, train_side, score_x, score_side = _validate(
        model_name,
        variant,
        train_static,
        train_y,
        train_weight,
        train_side,
        score_static,
        score_side,
    )
    if variant == "pooled":
        scores, low, high = _fit_one(
            model_name, train_x, target, weights, score_x
        )
        return EconomicPrediction(scores, low, high, 1)

    predictions = np.empty(len(score_x), dtype=float)
    lows: list[float] = []
    highs: list[float] = []
    fitted = 0
    for side in (-1.0, 1.0):
        fit_mask = train_side == side
        score_mask = score_side == side
        if fit_mask.sum() < 10:
            raise ValueError(f"split_side needs at least ten training rows for {side:+g}")
        if not score_mask.any():
            continue
        side_scores, low, high = _fit_one(
            model_name,
            train_x[fit_mask],
            target[fit_mask],
            weights[fit_mask],
            score_x[score_mask],
        )
        predictions[score_mask] = side_scores
        lows.append(low)
        highs.append(high)
        fitted += 1
    if fitted == 0:
        raise ValueError("split_side has no scoring rows")
    return EconomicPrediction(predictions, min(lows), max(highs), fitted)
