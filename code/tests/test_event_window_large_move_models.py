import numpy as np
import pytest

from experiments.event_window_large_move_models import (
    LargeMoveModelConfig,
    fit_predict_multiclass,
    fit_predict_opportunity,
    fit_predict_two_stage_xgboost,
)


def _arrays(seed: int = 4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(90, 6))
    y = np.tile(np.arange(3), 30)
    x[y == 1, 0] += 1.5
    x[y == 2, 0] -= 1.5
    x[0, 2] = np.nan
    weights = np.linspace(0.5, 1.5, len(x))
    score = rng.normal(size=(11, 6))
    return x, y, weights, score


@pytest.mark.parametrize("model_name", ["logreg", "xgboost"])
def test_multiclass_models_return_ordered_probabilities(model_name):
    x, y, weights, score = _arrays()
    config = LargeMoveModelConfig(xgb_estimators=12, n_jobs=1)
    result = fit_predict_multiclass(
        model_name,
        train_x=x,
        labels=y,
        sample_weight=weights,
        score_x=score,
        config=config,
    )
    probabilities = np.exp(result.logits)
    assert probabilities.shape == (len(score), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.metadata["classes"] == [0, 1, 2]


def test_two_stage_xgboost_returns_joint_three_class_distribution():
    x, y, weights, score = _arrays()
    result = fit_predict_two_stage_xgboost(
        train_x=x,
        labels=y,
        sample_weight=weights,
        score_x=score,
        config=LargeMoveModelConfig(xgb_estimators=12, n_jobs=1),
    )
    probabilities = np.exp(result.logits)
    assert probabilities.shape == (len(score), 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.metadata["architecture"] == "opportunity_then_direction"
    assert result.metadata["direction_fit_rows"] == 60
    assert result.opportunity_logit.shape == (len(score),)
    assert result.direction_logit.shape == (len(score),)
    p_big = 1.0 / (1.0 + np.exp(-result.opportunity_logit))
    p_up = 1.0 / (1.0 + np.exp(-result.direction_logit))
    rebuilt = np.column_stack([1.0 - p_big, p_big * p_up, p_big * (1.0 - p_up)])
    assert np.allclose(probabilities, rebuilt)


@pytest.mark.parametrize("model_name", ["logreg", "xgboost"])
def test_opportunity_models_return_one_binary_probability(model_name):
    x, y, weights, score = _arrays()
    binary = (y != 0).astype(int)
    result = fit_predict_opportunity(
        model_name,
        train_x=x,
        labels=binary,
        sample_weight=weights,
        score_x=score,
        config=LargeMoveModelConfig(xgb_estimators=12, n_jobs=1),
    )
    probabilities = np.exp(result.logits)
    assert probabilities.shape == (len(score), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert result.opportunity_logit.shape == (len(score),)
    assert np.allclose(
        probabilities[:, 1],
        1.0 / (1.0 + np.exp(-result.opportunity_logit)),
    )
    assert result.metadata["architecture"] == "binary_opportunity"


def test_xgboost_predictions_are_deterministic():
    x, y, weights, score = _arrays()
    kwargs = dict(
        train_x=x,
        labels=y,
        sample_weight=weights,
        score_x=score,
        config=LargeMoveModelConfig(xgb_estimators=10, n_jobs=1),
    )
    first = fit_predict_two_stage_xgboost(**kwargs)
    second = fit_predict_two_stage_xgboost(**kwargs)
    assert np.array_equal(first.logits, second.logits)


def test_models_reject_missing_class():
    x, y, weights, score = _arrays()
    with pytest.raises(ValueError, match="require"):
        fit_predict_multiclass(
            "logreg",
            train_x=x[y != 2],
            labels=y[y != 2],
            sample_weight=weights[y != 2],
            score_x=score,
        )
    with pytest.raises(ValueError, match="both binary classes"):
        fit_predict_opportunity(
            "logreg",
            train_x=x[y == 0],
            labels=np.zeros((y == 0).sum(), dtype=int),
            sample_weight=weights[y == 0],
            score_x=score,
        )
