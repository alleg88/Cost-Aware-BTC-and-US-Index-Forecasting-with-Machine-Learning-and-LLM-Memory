"""Fold-local calibration and expected-value helpers for event-window tails."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar


@dataclass(frozen=True)
class TemperatureCalibration:
    temperature: float
    rows: int


@dataclass(frozen=True)
class TimeoutCalibration:
    bias: float
    used_fallback: bool
    rows: int
    training_bounds: tuple[float, float]


def _calibration_arrays(
    logits: np.ndarray,
    outcome: np.ndarray,
    sample_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(outcome)
    weights = np.asarray(sample_weight, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("logits must have shape (rows, 3)")
    if labels.ndim != 1 or weights.ndim != 1:
        raise ValueError("outcome and sample_weight must be one-dimensional")
    if not (len(values) == len(labels) == len(weights)):
        raise ValueError("calibration arrays must have the same row count")
    if len(labels) == 0:
        raise ValueError("calibration data must not be empty")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("calibration data must be finite")
    if not np.issubdtype(labels.dtype, np.number) or not np.isfinite(labels).all():
        raise ValueError("calibration data must be finite")
    if not np.equal(labels, np.floor(labels)).all() or not np.isin(labels, (0, 1, 2)).all():
        raise ValueError("outcome must contain only class codes 0, 1, and 2")
    if (weights < 0.0).any() or weights.sum() <= 0.0:
        raise ValueError("sample_weight must be non-negative with positive total weight")
    return values, labels.astype(np.int64, copy=False), weights


def _weighted_multiclass_nll(
    logits: np.ndarray,
    outcome: np.ndarray,
    sample_weight: np.ndarray,
) -> float:
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_normaliser = np.log(np.exp(shifted).sum(axis=1))
    losses = log_normaliser - shifted[np.arange(len(outcome)), outcome]
    return float(np.average(losses, weights=sample_weight))


def fit_temperature(
    logits: np.ndarray,
    outcome: np.ndarray,
    sample_weight: np.ndarray,
) -> TemperatureCalibration:
    """Fit one temperature by weighted multiclass negative log-likelihood."""
    values, labels, weights = _calibration_arrays(logits, outcome, sample_weight)
    result = minimize_scalar(
        lambda log_t: _weighted_multiclass_nll(
            values / np.exp(log_t), labels, weights
        ),
        bounds=(-4.0, 4.0),
        method="bounded",
    )
    if not result.success or not np.isfinite(result.x):
        raise RuntimeError("temperature optimisation failed")
    return TemperatureCalibration(float(np.exp(result.x)), len(labels))


def apply_temperature(
    logits: np.ndarray,
    calibration: TemperatureCalibration,
) -> np.ndarray:
    """Convert three-class logits to temperature-scaled probabilities."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("logits must have shape (rows, 3)")
    if not np.isfinite(values).all():
        raise ValueError("logits must be finite")
    if not np.isfinite(calibration.temperature) or calibration.temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    if len(values) == 0:
        return np.empty_like(values)
    scaled = values / calibration.temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _timeout_bounds(training_bounds: tuple[float, float]) -> tuple[float, float]:
    bounds = np.asarray(training_bounds, dtype=np.float64)
    if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
        raise ValueError("training_bounds must be two ordered finite values")
    return float(bounds[0]), float(bounds[1])


def fit_timeout_bias(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_weight: np.ndarray,
    *,
    fallback_mean: float,
    training_bounds: tuple[float, float],
) -> TimeoutCalibration:
    """Fit the weighted timeout residual mean, or retain a training-only fallback."""
    predictions = np.asarray(prediction, dtype=np.float64)
    targets = np.asarray(target, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    bounds = _timeout_bounds(training_bounds)
    if predictions.ndim != 1 or targets.ndim != 1 or weights.ndim != 1:
        raise ValueError("timeout calibration arrays must be one-dimensional")
    if not (len(predictions) == len(targets) == len(weights)):
        raise ValueError("timeout calibration arrays must have the same row count")
    if not np.isfinite(fallback_mean):
        raise ValueError("fallback_mean must be finite")
    if len(targets) == 0:
        return TimeoutCalibration(float(fallback_mean), True, 0, bounds)
    if not (
        np.isfinite(predictions).all()
        and np.isfinite(targets).all()
        and np.isfinite(weights).all()
    ):
        raise ValueError("timeout calibration data must be finite")
    if (weights < 0.0).any() or weights.sum() <= 0.0:
        raise ValueError("sample_weight must be non-negative with positive total weight")
    bias = float(np.average(targets - predictions, weights=weights))
    return TimeoutCalibration(bias, False, len(targets), bounds)


def apply_timeout_bias(
    prediction: np.ndarray,
    calibration: TimeoutCalibration,
) -> np.ndarray:
    """Apply timeout correction within training-fold 1st/99th percentile bounds."""
    predictions = np.asarray(prediction, dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise ValueError("timeout predictions must be finite")
    if not np.isfinite(calibration.bias):
        raise ValueError("timeout calibration bias must be finite")
    low, high = _timeout_bounds(calibration.training_bounds)
    if calibration.used_fallback:
        corrected = np.full(predictions.shape, calibration.bias, dtype=np.float64)
    else:
        corrected = predictions + calibration.bias
    return np.clip(corrected, low, high)


def expected_net_r(
    probabilities: np.ndarray,
    *,
    tp_net_r: np.ndarray,
    sl_net_r: np.ndarray,
    timeout_net_r: np.ndarray,
) -> np.ndarray:
    """Calculate EV in the fixed class order SL, TP, timeout."""
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (rows, 3)")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    tp = np.asarray(tp_net_r, dtype=np.float64)
    sl = np.asarray(sl_net_r, dtype=np.float64)
    timeout = np.asarray(timeout_net_r, dtype=np.float64)
    try:
        tp = np.broadcast_to(tp, (len(values),))
        sl = np.broadcast_to(sl, (len(values),))
        timeout = np.broadcast_to(timeout, (len(values),))
    except ValueError as exc:
        raise ValueError("net R arrays must broadcast to probability rows") from exc
    if not (np.isfinite(tp).all() and np.isfinite(sl).all() and np.isfinite(timeout).all()):
        raise ValueError("net R inputs must be finite")
    return values[:, 0] * sl + values[:, 1] * tp + values[:, 2] * timeout
