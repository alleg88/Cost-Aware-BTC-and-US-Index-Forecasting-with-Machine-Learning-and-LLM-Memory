"""Economic net-R scorers for the Fast-T2 entry continuation."""

from importlib import import_module

import numpy as np
import pytest


def _module():
    try:
        return import_module("experiments.fast_t2_economic_models")
    except ModuleNotFoundError:
        pytest.fail("economic model module is not implemented")


def test_ridge_predicts_enter_advantage_in_r_units():
    module = _module()
    x = np.column_stack([np.linspace(-2.0, 2.0, 40), np.tile([-1.0, 1.0], 20)])
    y = 0.6 * x[:, 0] + 0.4 * x[:, 1]
    weights = np.ones(len(y))

    result = module.fit_predict_economic_model(
        "ridge",
        "pooled",
        x,
        y,
        weights,
        x[:, 1],
        x[[0, -1]],
        x[[0, -1], 1],
    )

    assert result.scores.shape == (2,)
    assert np.isfinite(result.scores).all()
    assert result.scores[0] < 0.0 < result.scores[1]
    assert result.target_low < result.target_high


def test_split_side_fits_two_directional_economic_models():
    module = _module()
    side = np.repeat([-1.0, 1.0], 20)
    x = np.column_stack([np.linspace(-1.0, 1.0, 40), side])
    y = np.where(side > 0, 1.0 + x[:, 0], -1.0 + x[:, 0])

    result = module.fit_predict_economic_model(
        "ridge",
        "split_side",
        x,
        y,
        np.ones(len(y)),
        side,
        np.array([[0.0, 1.0], [0.0, -1.0]]),
        np.array([1.0, -1.0]),
    )

    assert result.scores[0] > 0.5
    assert result.scores[1] < -0.5
    assert result.fitted_models == 2


def test_xgboost_predicts_robust_economic_scores():
    pytest.importorskip("xgboost")
    module = _module()
    x = np.column_stack([
        np.linspace(-2.0, 2.0, 48),
        np.tile([-1.0, 1.0], 24),
    ])
    y = 0.5 * x[:, 0] ** 2 * x[:, 1] + 0.2 * x[:, 0]

    result = module.fit_predict_economic_model(
        "xgboost",
        "pooled",
        x,
        y,
        np.linspace(0.5, 1.5, len(y)),
        x[:, 1],
        x[[5, 42]],
        x[[5, 42], 1],
    )

    assert result.scores.shape == (2,)
    assert np.isfinite(result.scores).all()
    assert result.target_low < result.target_high
    assert result.fitted_models == 1


def test_economic_model_registry_includes_tree_regressors_only():
    module = _module()

    assert module.ECONOMIC_MODEL_NAMES == ("ridge", "catboost", "xgboost")
    assert module.ECONOMIC_MODEL_VARIANTS == ("pooled", "split_side")
    assert "logreg" not in module.ECONOMIC_MODEL_NAMES
    assert "gru" not in module.ECONOMIC_MODEL_NAMES
    assert "lstm" not in module.ECONOMIC_MODEL_NAMES
