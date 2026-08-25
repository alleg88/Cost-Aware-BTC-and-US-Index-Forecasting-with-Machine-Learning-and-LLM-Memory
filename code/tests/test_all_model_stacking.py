from __future__ import annotations

import numpy as np

from experiments.all_model_stacking import MODEL_NAMES, WIDTHS
from experiments.correlation_ensemble import _feature_matrix, _fit_stack


def test_stack_uses_all_nine_models_and_eighteen_independent_features():
    probabilities = {
        model: np.tile(np.array([[0.2, 0.5, 0.3]]), (12, 1))
        for model in MODEL_NAMES
    }
    matrix = _feature_matrix(probabilities, MODEL_NAMES)
    assert len(MODEL_NAMES) == 9
    assert matrix.shape == (12, 18)


def test_meta_learner_is_fixed_balanced_l2_logistic_regression():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(90, 18))
    y = np.tile(np.arange(3), 30)
    meta = _fit_stack(X, y)
    logistic = meta.named_steps["logisticregression"]
    assert logistic.C == 0.1
    assert logistic.class_weight == "balanced"
    assert logistic.l1_ratio == 0.0


def test_stack_keeps_dead_zones_separate():
    assert WIDTHS == (55, 65, 75)
