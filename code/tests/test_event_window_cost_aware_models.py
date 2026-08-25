from __future__ import annotations

import numpy as np
import pytest

from experiments.event_window_cost_aware_models import (
    CostAwareModelConfig,
    fit_predict_cost_aware_model,
)


def _arrays(rows: int = 80):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(rows, 6))
    outcome = np.tile(np.arange(4), rows // 4)
    timeout = np.where(outcome == 3, rng.normal(0.2, 0.1, rows), np.nan)
    advantage = rng.normal(0.0, 0.5, rows)
    valid = np.ones(rows, dtype=bool)
    weights = np.linspace(0.5, 1.5, rows)
    return x, outcome, timeout, advantage, valid, weights


@pytest.mark.parametrize("model_name", ["logreg", "xgboost"])
def test_cost_aware_bundle_returns_aligned_finite_predictions(model_name):
    x, outcome, timeout, advantage, valid, weights = _arrays()
    prediction = fit_predict_cost_aware_model(
        model_name,
        train_x=x,
        outcome=outcome,
        timeout_gross_target=timeout,
        advantage_target=advantage,
        advantage_valid=valid,
        sample_weight=weights,
        score_x=x[:11],
        config=CostAwareModelConfig(xgb_estimators=5, xgb_min_child_weight=1.0),
    )
    assert prediction.logits.shape == (11, 4)
    assert prediction.timeout_gross_r.shape == (11,)
    assert prediction.enter_advantage.shape == (11,)
    assert np.isfinite(prediction.logits).all()
    assert np.isfinite(prediction.timeout_gross_r).all()
    assert np.isfinite(prediction.enter_advantage).all()


def test_bundle_rejects_missing_outcome_class():
    x, outcome, timeout, advantage, valid, weights = _arrays()
    with pytest.raises(ValueError, match="unfilled"):
        fit_predict_cost_aware_model(
            "logreg",
            train_x=x[outcome != 0],
            outcome=outcome[outcome != 0],
            timeout_gross_target=timeout[outcome != 0],
            advantage_target=advantage[outcome != 0],
            advantage_valid=valid[outcome != 0],
            sample_weight=weights[outcome != 0],
            score_x=x[:4],
        )
