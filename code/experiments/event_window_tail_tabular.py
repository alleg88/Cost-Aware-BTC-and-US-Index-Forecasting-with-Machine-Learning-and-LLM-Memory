"""Fold-local tabular outcome and timeout models for event-window tails."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class LogRegTailConfig:
    c: float = 1.0
    max_iter: int = 2000


@dataclass(frozen=True)
class XGBoostTailConfig:
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.03
    min_child_weight: float = 20.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_lambda: float = 10.0
    random_seed: int = 42
    n_jobs: int = 1


@dataclass(frozen=True)
class TabularTailPrediction:
    logits: np.ndarray
    timeout_net_r: np.ndarray
    model_metadata: dict[str, object]


def _validate_inputs(
    train_x: np.ndarray,
    outcome: np.ndarray,
    timeout_target: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    episode_ids: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    train = np.asarray(train_x, dtype=float)
    score = np.asarray(score_x, dtype=float)
    labels = np.asarray(outcome, dtype=np.int64)
    timeout = np.asarray(timeout_target, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    episodes = None if episode_ids is None else np.asarray(episode_ids)
    if train.ndim != 2 or score.ndim != 2:
        raise ValueError("tabular features must be two-dimensional")
    if train.shape[1] != score.shape[1]:
        raise ValueError("training and scoring feature counts must match")
    if not (len(train) == len(labels) == len(timeout) == len(weights)):
        raise ValueError("tabular training arrays must align")
    if episodes is not None and len(episodes) != len(train):
        raise ValueError("episode IDs must align with training rows")
    if not len(train) or not len(score):
        raise ValueError("tabular inputs cannot be empty")
    if set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError("tabular training needs SL, TP, and timeout classes")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("sample weights must be positive and finite")
    timeout_mask = labels == 2
    if not np.isfinite(timeout[timeout_mask]).all():
        raise ValueError("observed timeout targets must be finite")
    train = np.where(np.isfinite(train), train, np.nan)
    score = np.where(np.isfinite(score), score, np.nan)
    return train, labels, timeout, weights, score, episodes


def _impute_train_only(
    train_x: np.ndarray, score_x: np.ndarray
) -> tuple[np.ndarray, np.ndarray, SimpleImputer]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train_x)
    predict_x = imputer.transform(score_x)
    return fit_x, predict_x, imputer


def _weighted_timeout_mean(
    outcome: np.ndarray, timeout_target: np.ndarray, sample_weight: np.ndarray
) -> float:
    timeout_mask = outcome == 2
    return float(
        np.average(timeout_target[timeout_mask], weights=sample_weight[timeout_mask])
    )


def _ordered_probabilities(model, predict_x: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(predict_x), dtype=float)
    if probabilities.shape != (len(predict_x), 3):
        raise AssertionError("classifier did not return all three outcome classes")
    if not np.array_equal(np.asarray(model.classes_), np.array([0, 1, 2])):
        raise AssertionError("classifier outcome classes are not ordered as SL, TP, timeout")
    return np.clip(probabilities, 1e-8, 1.0)


def fit_predict_logreg(
    train_x: np.ndarray,
    outcome: np.ndarray,
    timeout_target: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    *,
    episode_ids: np.ndarray | None = None,
    config: LogRegTailConfig = LogRegTailConfig(),
) -> TabularTailPrediction:
    """Fit the weighted L2 baseline using fit-row preprocessing only."""
    train, labels, timeout, weights, score, _ = _validate_inputs(
        train_x, outcome, timeout_target, sample_weight, score_x, episode_ids
    )
    fit_x, predict_x, imputer = _impute_train_only(train, score)
    scaler = StandardScaler()
    fit_x = scaler.fit_transform(fit_x)
    predict_x = scaler.transform(predict_x)
    model = LogisticRegression(
        C=config.c,
        solver="lbfgs",
        max_iter=config.max_iter,
        class_weight=None,
    )
    model.fit(fit_x, labels, sample_weight=weights)
    probabilities = _ordered_probabilities(model, predict_x)
    timeout_mean = _weighted_timeout_mean(labels, timeout, weights)
    metadata: dict[str, object] = {
        "config": asdict(config),
        "fit_median": imputer.statistics_.astype(float).tolist(),
        "fit_mean": scaler.mean_.astype(float).tolist(),
        "fit_scale": scaler.scale_.astype(float).tolist(),
        "classes": model.classes_.astype(int).tolist(),
        "penalty": "l2",
        "solver": model.solver,
        "timeout_fallback": False,
    }
    return TabularTailPrediction(
        logits=np.log(probabilities),
        timeout_net_r=np.full(len(score), timeout_mean, dtype=float),
        model_metadata=metadata,
    )


def _xgb_classifier(config: XGBoostTailConfig):
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        tree_method="hist",
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
        eval_metric="mlogloss",
        verbosity=0,
    )


def _xgb_regressor(config: XGBoostTailConfig):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_child_weight=config.min_child_weight,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_lambda=config.reg_lambda,
        tree_method="hist",
        random_state=config.random_seed,
        n_jobs=config.n_jobs,
        verbosity=0,
    )


def _fit_timeout_xgb_or_mean(
    fit_x: np.ndarray,
    outcome: np.ndarray,
    timeout_target: np.ndarray,
    sample_weight: np.ndarray,
    predict_x: np.ndarray,
    episode_ids: np.ndarray | None,
    config: XGBoostTailConfig,
) -> tuple[np.ndarray, bool, int, int]:
    timeout_mask = outcome == 2
    timeout_rows = int(timeout_mask.sum())
    timeout_episodes = (
        0 if episode_ids is None else int(np.unique(episode_ids[timeout_mask]).size)
    )
    fallback = timeout_rows < 200 or timeout_episodes < 20
    if fallback:
        mean = _weighted_timeout_mean(outcome, timeout_target, sample_weight)
        return np.full(len(predict_x), mean, dtype=float), True, timeout_rows, timeout_episodes
    regressor = _xgb_regressor(config)
    regressor.fit(
        fit_x[timeout_mask],
        timeout_target[timeout_mask],
        sample_weight=sample_weight[timeout_mask],
    )
    prediction = np.asarray(regressor.predict(predict_x), dtype=float)
    if prediction.shape != (len(predict_x),) or not np.isfinite(prediction).all():
        raise AssertionError("timeout regressor returned invalid predictions")
    return prediction, False, timeout_rows, timeout_episodes


def fit_predict_xgboost(
    train_x: np.ndarray,
    outcome: np.ndarray,
    timeout_target: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    *,
    episode_ids: np.ndarray | None = None,
    config: XGBoostTailConfig = XGBoostTailConfig(),
) -> TabularTailPrediction:
    """Fit the frozen weighted XGBoost outcome and timeout models."""
    train, labels, timeout, weights, score, episodes = _validate_inputs(
        train_x, outcome, timeout_target, sample_weight, score_x, episode_ids
    )
    fit_x, predict_x, imputer = _impute_train_only(train, score)
    classifier = _xgb_classifier(config)
    classifier.fit(fit_x, labels, sample_weight=weights)
    probabilities = _ordered_probabilities(classifier, predict_x)
    timeout_prediction, fallback, timeout_rows, timeout_episodes = (
        _fit_timeout_xgb_or_mean(
            fit_x,
            labels,
            timeout,
            weights,
            predict_x,
            episodes,
            config,
        )
    )
    metadata: dict[str, object] = {
        "config": asdict(config),
        "fit_median": imputer.statistics_.astype(float).tolist(),
        "classes": classifier.classes_.astype(int).tolist(),
        "timeout_fallback": fallback,
        "timeout_rows": timeout_rows,
        "timeout_episodes": timeout_episodes,
    }
    return TabularTailPrediction(
        logits=np.log(probabilities),
        timeout_net_r=timeout_prediction,
        model_metadata=metadata,
    )


def fit_predict_tail_model(model_name: str, **kwargs) -> TabularTailPrediction:
    """Dispatch one of the registered tabular tail models."""
    if model_name == "logreg":
        return fit_predict_logreg(**kwargs)
    if model_name == "xgboost":
        return fit_predict_xgboost(**kwargs)
    raise ValueError(f"unsupported tabular model: {model_name}")
