"""Matched pooled CatBoost/XGBoost outcome-EV models."""

from importlib import import_module

import numpy as np
import pytest


def _module():
    try:
        return import_module("experiments.fast_t2_causal_ev_models")
    except ModuleNotFoundError:
        pytest.fail("causal EV model module is not implemented")


def test_temperature_calibration_keeps_multiclass_probabilities_valid():
    module = _module()
    probabilities = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 4
    )
    labels = np.tile([0, 1, 2], 4)
    temperature = module.select_temperature(
        probabilities, labels, np.ones(len(labels))
    )
    calibrated = module.apply_temperature(probabilities, temperature)

    assert temperature in module.TEMPERATURE_GRID
    assert calibrated.shape == probabilities.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert (calibrated > 0).all()


@pytest.mark.parametrize("model_name", ["catboost", "xgboost"])
def test_pooled_ev_models_predict_three_outcomes_and_timeout_magnitude(model_name):
    module = _module()
    if model_name == "catboost":
        pytest.importorskip("catboost")
    else:
        pytest.importorskip("xgboost")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(90, 5))
    labels = np.tile([0, 1, 2], 30)
    gross_r = np.where(labels == 2, 0.3 * x[:, 0], np.where(labels == 1, 2.0, -1.0))

    prediction = module.fit_predict_causal_ev_model(
        model_name,
        x,
        labels,
        gross_r,
        np.ones(len(labels)),
        x[:7],
    )

    assert prediction.outcome_probabilities.shape == (7, 3)
    assert np.allclose(prediction.outcome_probabilities.sum(axis=1), 1.0)
    assert prediction.timeout_gross_r.shape == (7,)
    assert np.isfinite(prediction.timeout_gross_r).all()
    assert prediction.timeout_target_low < prediction.timeout_target_high


def test_ev_model_registry_contains_only_two_pooled_tree_families():
    module = _module()

    assert module.EV_MODEL_NAMES == ("catboost", "xgboost")
    assert "logreg" not in module.EV_MODEL_NAMES
    assert "gru" not in module.EV_MODEL_NAMES
    assert "lstm" not in module.EV_MODEL_NAMES
