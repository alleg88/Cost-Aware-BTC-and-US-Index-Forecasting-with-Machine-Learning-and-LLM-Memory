"""Fold-local LogReg/XGBoost bundles for Notebook L."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class CostAwareModelConfig:
    xgb_estimators: int = 300
    xgb_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_min_child_weight: float = 20.0
    xgb_reg_lambda: float = 10.0
    ridge_alpha: float = 10.0
    random_seed: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class CostAwareRawPrediction:
    logits: np.ndarray
    timeout_gross_r: np.ndarray
    enter_advantage: np.ndarray
    metadata: dict[str, object]


def _xgb_classifier(config: CostAwareModelConfig):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        num_class=4,
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
        eval_metric="mlogloss",
        verbosity=0,
    )


def _xgb_regressor(config: CostAwareModelConfig):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
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


def fit_predict_cost_aware_model(
    model_name: str,
    *,
    train_x: np.ndarray,
    outcome: np.ndarray,
    timeout_gross_target: np.ndarray,
    advantage_target: np.ndarray,
    advantage_valid: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    config: CostAwareModelConfig = CostAwareModelConfig(),
) -> CostAwareRawPrediction:
    """Fit one outcome, timeout-size and enter-vs-wait bundle on fit rows only."""
    if model_name not in {"logreg", "xgboost"}:
        raise ValueError("model_name must be logreg or xgboost")
    train = np.asarray(train_x, dtype=float)
    score = np.asarray(score_x, dtype=float)
    labels = np.asarray(outcome, dtype=np.int64)
    timeout = np.asarray(timeout_gross_target, dtype=float)
    advantage = np.asarray(advantage_target, dtype=float)
    advantage_mask = np.asarray(advantage_valid, dtype=bool).copy()
    weights = np.asarray(sample_weight, dtype=float)
    if train.ndim != 2 or score.ndim != 2 or train.shape[1] != score.shape[1]:
        raise ValueError("train_x and score_x must have aligned two-dimensional features")
    if not (
        len(train)
        == len(labels)
        == len(timeout)
        == len(advantage)
        == len(advantage_mask)
        == len(weights)
    ):
        raise ValueError("training arrays must align")
    if set(np.unique(labels)) != {0, 1, 2, 3}:
        raise ValueError("fit rows need unfilled, SL, TP, and timeout outcomes")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample weights must be positive and finite")
    timeout_mask = labels == 3
    if not timeout_mask.any() or not np.isfinite(timeout[timeout_mask]).all():
        raise ValueError("timeout targets must be observed and finite")
    advantage_mask &= np.isfinite(advantage)
    if not advantage_mask.any():
        raise ValueError("fit rows need valid enter-vs-wait targets")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(np.where(np.isfinite(train), train, np.nan))
    predict_x = imputer.transform(np.where(np.isfinite(score), score, np.nan))
    scaler = None
    if model_name == "logreg":
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(fit_x)
        predict_x = scaler.transform(predict_x)
        classifier = LogisticRegression(max_iter=2000, solver="lbfgs")
        classifier.fit(fit_x, labels, sample_weight=weights)
        timeout_prediction = np.full(
            len(predict_x),
            np.average(timeout[timeout_mask], weights=weights[timeout_mask]),
            dtype=float,
        )
        advantage_model = Ridge(alpha=config.ridge_alpha)
        advantage_model.fit(
            fit_x[advantage_mask],
            advantage[advantage_mask],
            sample_weight=weights[advantage_mask],
        )
        advantage_prediction = advantage_model.predict(predict_x)
    else:
        classifier = _xgb_classifier(config)
        classifier.fit(fit_x, labels, sample_weight=weights)
        timeout_model = _xgb_regressor(config)
        timeout_model.fit(
            fit_x[timeout_mask],
            timeout[timeout_mask],
            sample_weight=weights[timeout_mask],
        )
        timeout_prediction = timeout_model.predict(predict_x)
        advantage_model = _xgb_regressor(config)
        advantage_model.fit(
            fit_x[advantage_mask],
            advantage[advantage_mask],
            sample_weight=weights[advantage_mask],
        )
        advantage_prediction = advantage_model.predict(predict_x)
    probabilities = np.asarray(classifier.predict_proba(predict_x), dtype=float)
    if probabilities.shape != (len(predict_x), 4):
        raise AssertionError("outcome classifier must return four probabilities")
    if not np.array_equal(np.asarray(classifier.classes_), np.arange(4)):
        raise AssertionError("outcome classes must be ordered 0..3")
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    if not (
        np.isfinite(probabilities).all()
        and np.isfinite(timeout_prediction).all()
        and np.isfinite(advantage_prediction).all()
    ):
        raise AssertionError("model returned non-finite predictions")
    metadata: dict[str, object] = {
        "model": model_name,
        "config": asdict(config),
        "classes": classifier.classes_.astype(int).tolist(),
        "imputer_features": int(len(imputer.statistics_)),
        "scaled": scaler is not None,
        "timeout_fit_rows": int(timeout_mask.sum()),
        "advantage_fit_rows": int(advantage_mask.sum()),
    }
    return CostAwareRawPrediction(
        logits=np.log(probabilities),
        timeout_gross_r=np.asarray(timeout_prediction, dtype=float),
        enter_advantage=np.asarray(advantage_prediction, dtype=float),
        metadata=metadata,
    )


__all__ = [
    "CostAwareModelConfig",
    "CostAwareRawPrediction",
    "fit_predict_cost_aware_model",
]
