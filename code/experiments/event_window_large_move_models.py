"""Fold-local models for adaptive large-move direction experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LargeMoveModelConfig:
    xgb_estimators: int = 300
    xgb_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_min_child_weight: float = 20.0
    xgb_reg_lambda: float = 10.0
    logreg_max_iter: int = 2_000
    random_seed: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class LargeMoveRawPrediction:
    logits: np.ndarray
    metadata: dict[str, object]
    opportunity_logit: np.ndarray | None = None
    direction_logit: np.ndarray | None = None


def _validate(
    train_x: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=float)
    score = np.asarray(score_x, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=float)
    if train.ndim != 2 or score.ndim != 2 or train.shape[1] != score.shape[1]:
        raise ValueError("training and score features must be aligned matrices")
    if not len(train) or not len(score) or len(train) != len(y) or len(y) != len(weights):
        raise ValueError("training arrays must be non-empty and aligned")
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError("fit rows require no-big, up-big, and down-big classes")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample weights must be positive and finite")
    return (
        np.where(np.isfinite(train), train, np.nan),
        y,
        weights,
        np.where(np.isfinite(score), score, np.nan),
    )


def _validate_binary(
    train_x: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = np.asarray(train_x, dtype=float)
    score = np.asarray(score_x, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=float)
    if train.ndim != 2 or score.ndim != 2 or train.shape[1] != score.shape[1]:
        raise ValueError("training and score features must be aligned matrices")
    if not len(train) or not len(score) or len(train) != len(y) or len(y) != len(weights):
        raise ValueError("training arrays must be non-empty and aligned")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("opportunity fit rows require both binary classes")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("sample weights must be positive and finite")
    return (
        np.where(np.isfinite(train), train, np.nan),
        y,
        weights,
        np.where(np.isfinite(score), score, np.nan),
    )


def _xgb_classifier(config: LargeMoveModelConfig, *, classes: int):
    from xgboost import XGBClassifier

    parameters: dict[str, object] = {
        "n_estimators": config.xgb_estimators,
        "max_depth": config.xgb_depth,
        "learning_rate": config.xgb_learning_rate,
        "min_child_weight": config.xgb_min_child_weight,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": config.xgb_reg_lambda,
        "tree_method": "hist",
        "random_state": config.random_seed,
        "n_jobs": config.n_jobs,
        "verbosity": 0,
    }
    if classes == 2:
        parameters.update(objective="binary:logistic", eval_metric="logloss")
    else:
        parameters.update(
            objective="multi:softprob", num_class=classes, eval_metric="mlogloss"
        )
    return XGBClassifier(**parameters)


def _ordered_three_class_probabilities(model, score_x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(score_x), dtype=float)
    if probabilities.shape != (len(score_x), 3):
        raise AssertionError("multiclass model must return three probabilities")
    if not np.array_equal(np.asarray(model.classes_), np.arange(3)):
        raise AssertionError("classes must be ordered no-big, up-big, down-big")
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    return probabilities / probabilities.sum(axis=1, keepdims=True)


def fit_predict_multiclass(
    model_name: str,
    *,
    train_x: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    config: LargeMoveModelConfig = LargeMoveModelConfig(),
) -> LargeMoveRawPrediction:
    """Fit the LogReg baseline or one three-class XGBoost."""
    if model_name not in {"logreg", "xgboost"}:
        raise ValueError("model_name must be logreg or xgboost")
    train, y, weights, score = _validate(train_x, labels, sample_weight, score_x)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train)
    predict_x = imputer.transform(score)
    scaler = None
    if model_name == "logreg":
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(fit_x)
        predict_x = scaler.transform(predict_x)
        model = LogisticRegression(
            max_iter=config.logreg_max_iter,
            solver="lbfgs",
            random_state=config.random_seed,
        )
    else:
        model = _xgb_classifier(config, classes=3)
    model.fit(fit_x, y, sample_weight=weights)
    probabilities = _ordered_three_class_probabilities(model, predict_x)
    return LargeMoveRawPrediction(
        logits=np.log(probabilities),
        metadata={
            "model": model_name,
            "architecture": "multiclass",
            "config": asdict(config),
            "classes": [0, 1, 2],
            "fit_median": imputer.statistics_.astype(float).tolist(),
            "scaled": scaler is not None,
        },
    )


def fit_predict_opportunity(
    model_name: str,
    *,
    train_x: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    config: LargeMoveModelConfig = LargeMoveModelConfig(),
) -> LargeMoveRawPrediction:
    """Fit one binary BIG-versus-NO-BIG model without a direction target."""
    if model_name not in {"logreg", "xgboost"}:
        raise ValueError("model_name must be logreg or xgboost")
    train, y, weights, score = _validate_binary(
        train_x, labels, sample_weight, score_x
    )
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train)
    predict_x = imputer.transform(score)
    scaler = None
    if model_name == "logreg":
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(fit_x)
        predict_x = scaler.transform(predict_x)
        model = LogisticRegression(
            max_iter=config.logreg_max_iter,
            solver="lbfgs",
            random_state=config.random_seed,
        )
    else:
        model = _xgb_classifier(config, classes=2)
    model.fit(fit_x, y, sample_weight=weights)
    p_hit = np.asarray(model.predict_proba(predict_x), dtype=float)[:, 1]
    p_hit = np.clip(p_hit, 1e-8, 1.0 - 1e-8)
    opportunity_logit = np.log(p_hit / (1.0 - p_hit))
    return LargeMoveRawPrediction(
        logits=np.column_stack([np.log1p(-p_hit), np.log(p_hit)]),
        metadata={
            "model": model_name,
            "architecture": "binary_opportunity",
            "config": asdict(config),
            "classes": [0, 1],
            "fit_median": imputer.statistics_.astype(float).tolist(),
            "scaled": scaler is not None,
        },
        opportunity_logit=opportunity_logit,
    )


def fit_predict_two_stage_xgboost(
    *,
    train_x: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    config: LargeMoveModelConfig = LargeMoveModelConfig(),
) -> LargeMoveRawPrediction:
    """Fit XGBoost for opportunity, then a second XGBoost for direction."""
    train, y, weights, score = _validate(train_x, labels, sample_weight, score_x)
    big = (y != 0).astype(np.int8)
    big_mask = big.astype(bool)
    direction = (y[big_mask] == 1).astype(np.int8)
    if set(np.unique(big)) != {0, 1} or set(np.unique(direction)) != {0, 1}:
        raise ValueError("two-stage fit needs both opportunity and direction classes")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train)
    predict_x = imputer.transform(score)
    opportunity_model = _xgb_classifier(config, classes=2)
    direction_model = _xgb_classifier(config, classes=2)
    opportunity_model.fit(fit_x, big, sample_weight=weights)
    direction_model.fit(
        fit_x[big_mask], direction, sample_weight=weights[big_mask]
    )
    p_big = np.asarray(opportunity_model.predict_proba(predict_x), dtype=float)[:, 1]
    p_up_given_big = np.asarray(
        direction_model.predict_proba(predict_x), dtype=float
    )[:, 1]
    p_big = np.clip(p_big, 1e-8, 1.0 - 1e-8)
    p_up_given_big = np.clip(p_up_given_big, 1e-8, 1.0 - 1e-8)
    probabilities = np.column_stack(
        [1.0 - p_big, p_big * p_up_given_big, p_big * (1.0 - p_up_given_big)]
    )
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    if not np.isfinite(probabilities).all():
        raise AssertionError("two-stage XGBoost returned non-finite probabilities")
    return LargeMoveRawPrediction(
        logits=np.log(probabilities),
        metadata={
            "model": "xgboost_two_stage",
            "architecture": "opportunity_then_direction",
            "config": asdict(config),
            "classes": [0, 1, 2],
            "fit_median": imputer.statistics_.astype(float).tolist(),
            "opportunity_fit_rows": int(len(y)),
            "direction_fit_rows": int(big_mask.sum()),
        },
        opportunity_logit=np.log(p_big / (1.0 - p_big)),
        direction_logit=np.log(p_up_given_big / (1.0 - p_up_given_big)),
    )


__all__ = [
    "LargeMoveModelConfig",
    "LargeMoveRawPrediction",
    "fit_predict_opportunity",
    "fit_predict_multiclass",
    "fit_predict_two_stage_xgboost",
]
