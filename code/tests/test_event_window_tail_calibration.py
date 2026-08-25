from __future__ import annotations

import numpy as np
import pytest

from experiments.event_window_tail_calibration import (
    apply_temperature,
    apply_timeout_bias,
    expected_net_r,
    fit_temperature,
    fit_timeout_bias,
)


def _logits() -> np.ndarray:
    return np.array(
        [
            [2.0, 0.5, -1.0],
            [-0.5, 1.5, 0.0],
            [0.0, -0.5, 1.0],
            [1.0, 0.0, 0.5],
        ]
    )


def _outcomes() -> np.ndarray:
    return np.array([0, 1, 2, 0])


def _weights() -> np.ndarray:
    return np.array([1.0, 2.0, 1.0, 0.5])


def test_temperature_is_positive_and_probabilities_sum_to_one():
    fit = fit_temperature(_logits(), _outcomes(), _weights())
    probabilities = apply_temperature(_logits(), fit)
    assert fit.temperature > 0.0
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_expected_net_r_uses_known_tp_sl_and_timeout_prediction():
    probabilities = np.array([[0.2, 0.3, 0.5]])  # SL, TP, timeout
    ev = expected_net_r(
        probabilities,
        tp_net_r=np.array([1.8]),
        sl_net_r=np.array([-1.2]),
        timeout_net_r=np.array([0.1]),
    )
    assert ev[0] == pytest.approx(0.2 * -1.2 + 0.3 * 1.8 + 0.5 * 0.1)


def test_class_order_is_sl_tp_timeout_and_cost_is_not_subtracted_twice():
    probabilities = np.eye(3)
    ev = expected_net_r(
        probabilities,
        tp_net_r=np.full(3, 1.8),
        sl_net_r=np.full(3, -1.2),
        timeout_net_r=np.full(3, 0.4),
    )
    np.testing.assert_allclose(ev, [-1.2, 1.8, 0.4])


def test_timeout_bias_uses_only_observed_timeout_calibration_rows():
    fit = fit_timeout_bias(
        np.array([0.0, 0.5]),
        np.array([0.2, 0.9]),
        np.ones(2),
        fallback_mean=-0.1,
        training_bounds=(-1.0, 1.0),
    )
    assert fit.bias == pytest.approx(0.3)
    np.testing.assert_allclose(apply_timeout_bias(np.array([0.0]), fit), [0.3])


def test_temperature_rejects_empty_or_non_finite_calibration_data():
    with pytest.raises(ValueError, match="empty"):
        fit_temperature(np.empty((0, 3)), np.array([], dtype=int), np.array([]))
    with pytest.raises(ValueError, match="finite"):
        fit_temperature(
            np.array([[np.nan, 0.0, 1.0]]), np.array([2]), np.array([1.0])
        )


def test_timeout_fallback_is_clipped_training_mean_when_no_rows_are_available():
    fit = fit_timeout_bias(
        np.array([]),
        np.array([]),
        np.array([]),
        fallback_mean=1.5,
        training_bounds=(-1.0, 1.0),
    )
    assert fit.used_fallback
    assert fit.rows == 0
    np.testing.assert_allclose(
        apply_timeout_bias(np.array([-10.0, 10.0]), fit), [1.0, 1.0]
    )
