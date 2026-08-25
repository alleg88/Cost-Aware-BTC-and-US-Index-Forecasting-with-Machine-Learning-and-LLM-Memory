"""Matched XGBoost/GRU outcome-EV models and fixed ensemble."""

from importlib import import_module

import numpy as np
import pytest


def _module():
    try:
        return import_module("experiments.pre_t2_ev_models")
    except ModuleNotFoundError:
        pytest.fail("pre-T2 EV model module is not implemented")


@pytest.mark.parametrize("model_name", ["xgboost", "gru"])
def test_models_predict_matched_outcome_probabilities_and_timeout_r(model_name):
    module = _module()
    if model_name == "xgboost":
        pytest.importorskip("xgboost")
    else:
        pytest.importorskip("torch")
    rng = np.random.default_rng(42)
    static = rng.normal(size=(90, 7))
    sequence = rng.normal(size=(90, 30, 5)).astype(np.float32)
    labels = np.tile([0, 1, 2], 30)
    gross_r = np.where(labels == 2, 0.25 * static[:, 0], np.where(labels == 1, 2.0, -1.0))
    prediction = module.fit_predict_pre_t2_model(
        model_name,
        static,
        sequence,
        labels,
        gross_r,
        np.ones(len(labels)),
        static[:9],
        sequence[:9],
        epochs=1,
    )

    assert prediction.outcome_probabilities.shape == (9, 3)
    assert np.allclose(prediction.outcome_probabilities.sum(axis=1), 1.0)
    assert prediction.timeout_gross_r.shape == (9,)
    assert np.isfinite(prediction.timeout_gross_r).all()
    assert prediction.timeout_target_low < prediction.timeout_target_high


def test_fixed_ensemble_is_exact_arithmetic_mean():
    module = _module()
    xgb = module.PreT2Prediction(
        outcome_probabilities=np.array([[0.6, 0.3, 0.1]]),
        timeout_gross_r=np.array([0.4]),
        timeout_target_low=-1.0,
        timeout_target_high=1.0,
    )
    gru = module.PreT2Prediction(
        outcome_probabilities=np.array([[0.2, 0.5, 0.3]]),
        timeout_gross_r=np.array([-0.2]),
        timeout_target_low=-2.0,
        timeout_target_high=2.0,
    )
    ensemble = module.combine_predictions(xgb, gru)

    assert np.allclose(ensemble.outcome_probabilities, [[0.4, 0.4, 0.2]])
    assert np.allclose(ensemble.timeout_gross_r, [0.1])
    assert ensemble.timeout_target_low == -2.0
    assert ensemble.timeout_target_high == 2.0


def test_registry_excludes_catboost_lstm_and_40bps_variants():
    module = _module()

    assert module.MODEL_NAMES == ("xgboost", "gru")
    assert module.ENSEMBLE_NAME == "xgboost_gru_50_50"

