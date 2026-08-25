"""Fold-local ordinal-magnitude model for Notebook O."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.impute import SimpleImputer

from experiments.event_window_large_move_models import (
    LargeMoveModelConfig,
    _xgb_classifier,
)
from experiments.event_window_magnitude_dataset import MAGNITUDE_BIN_LABELS


@dataclass(frozen=True)
class MagnitudeRawPrediction:
    logits: np.ndarray
    metadata: dict[str, object]


def fit_predict_magnitude_xgboost(
    *,
    train_x: np.ndarray,
    labels: np.ndarray,
    sample_weight: np.ndarray,
    score_x: np.ndarray,
    config: LargeMoveModelConfig = LargeMoveModelConfig(),
) -> MagnitudeRawPrediction:
    """Fit one five-bin XGBoost model and expose ordered class logits."""
    fit = np.asarray(train_x, dtype=float)
    score = np.asarray(score_x, dtype=float)
    y = np.asarray(labels, dtype=np.int64)
    weights = np.asarray(sample_weight, dtype=float)
    class_count = len(MAGNITUDE_BIN_LABELS)
    if fit.ndim != 2 or score.ndim != 2 or fit.shape[1] != score.shape[1]:
        raise ValueError("training and score feature matrices must align")
    if not len(fit) or not len(score) or len(fit) != len(y) or len(y) != len(weights):
        raise ValueError("training arrays must be non-empty and aligned")
    if set(np.unique(y)) != set(range(class_count)):
        raise ValueError("fit rows must contain all five registered magnitude bins")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("sample weights must be positive and finite")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_values = imputer.fit_transform(np.where(np.isfinite(fit), fit, np.nan))
    score_values = imputer.transform(np.where(np.isfinite(score), score, np.nan))
    model = _xgb_classifier(config, classes=class_count)
    model.fit(fit_values, y, sample_weight=weights)
    probabilities = np.asarray(model.predict_proba(score_values), dtype=float)
    if probabilities.shape != (len(score), class_count):
        raise AssertionError("magnitude model returned the wrong probability shape")
    if not np.array_equal(np.asarray(model.classes_), np.arange(class_count)):
        raise AssertionError("magnitude model classes are not in ordinal bin order")
    probabilities = np.clip(probabilities, 1e-8, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return MagnitudeRawPrediction(
        logits=np.log(probabilities),
        metadata={
            "model": "xgboost",
            "architecture": "one_five_bin_multiclass_head",
            "classes": list(range(class_count)),
            "bins": list(MAGNITUDE_BIN_LABELS),
            "config": asdict(config),
            "fit_median": imputer.statistics_.astype(float).tolist(),
        },
    )


__all__ = ["MagnitudeRawPrediction", "fit_predict_magnitude_xgboost"]
