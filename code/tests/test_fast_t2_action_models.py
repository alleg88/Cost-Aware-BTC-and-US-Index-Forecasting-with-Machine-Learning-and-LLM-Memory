"""Common scorer contract for the five Fast-T2 action models."""

import numpy as np
import pytest

from experiments.fast_t2_action_models import (
    MODEL_NAMES,
    fit_predict_action_model,
)


def _arrays():
    rng = np.random.default_rng(11)
    train_static = rng.normal(size=(48, 6))
    train_sequence = rng.normal(size=(48, 30, 5)).astype(np.float32)
    train_y = np.arange(48) % 2
    train_weight = np.linspace(0.7, 1.3, 48)
    score_static = rng.normal(size=(8, 6))
    score_sequence = rng.normal(size=(8, 30, 5)).astype(np.float32)
    return (
        train_static,
        train_sequence,
        train_y,
        train_weight,
        score_static,
        score_sequence,
    )


def test_exact_five_model_registry():
    assert MODEL_NAMES == ("logreg", "catboost", "xgboost", "gru", "lstm")


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_each_model_returns_finite_probabilities(name):
    arrays = _arrays()

    score = fit_predict_action_model(name, *arrays, epochs=1)

    assert score.shape == (8,)
    assert np.isfinite(score).all()
    assert ((score >= 0.0) & (score <= 1.0)).all()


def test_lstm_is_repeatable_on_cpu():
    arrays = _arrays()

    first = fit_predict_action_model("lstm", *arrays, epochs=2)
    second = fit_predict_action_model("lstm", *arrays, epochs=2)

    np.testing.assert_allclose(first, second, atol=1e-7)


def test_recurrent_model_requires_aligned_sequences():
    arrays = list(_arrays())
    arrays[1] = arrays[1][:-1]

    with pytest.raises(ValueError, match="training sequence"):
        fit_predict_action_model("gru", *arrays, epochs=1)
