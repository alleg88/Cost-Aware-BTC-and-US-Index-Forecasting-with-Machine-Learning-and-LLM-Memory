import numpy as np
import pytest

from experiments.event_window_direction_models import (
    DirectionModelConfig,
    fit_predict_delta_xgboost,
    fit_predict_value_logreg,
)


def _arrays(seed: int = 14):
    rng = np.random.default_rng(seed)
    train_x = rng.normal(size=(48, 5))
    delta_r = np.tile(np.array([-1.5, -0.4, 0.6, 2.2]), 12)
    train_x[:, 0] += np.sign(delta_r)
    train_x[0, 2] = np.nan
    uniqueness = np.linspace(0.6, 1.4, len(train_x))
    score_x = rng.normal(size=(9, 5))
    score_x[0, 1] = np.nan
    return train_x, delta_r, uniqueness, score_x


def test_logreg_uses_value_weighted_binary_target():
    train_x, delta_r, uniqueness, score_x = _arrays()
    prediction = fit_predict_value_logreg(
        train_x=train_x,
        delta_r=delta_r,
        uniqueness=uniqueness,
        score_x=score_x,
    )
    assert prediction.score.shape == (len(score_x),)
    assert np.isfinite(prediction.score).all()
    assert ((prediction.score >= 0.0) & (prediction.score <= 1.0)).all()


def test_logreg_economic_value_can_outweigh_raw_class_count():
    prediction = fit_predict_value_logreg(
        train_x=np.zeros((4, 1)),
        delta_r=np.array([0.1, 0.1, 0.1, -3.0]),
        uniqueness=np.ones(4),
        score_x=np.zeros((1, 1)),
    )
    assert prediction.score[0] < 0.5


def test_xgboost_predicts_continuous_delta():
    train_x, delta_r, uniqueness, score_x = _arrays()
    prediction = fit_predict_delta_xgboost(
        train_x=train_x,
        delta_r=delta_r,
        uniqueness=uniqueness,
        score_x=score_x,
        config=DirectionModelConfig(xgb_estimators=12),
    )
    assert prediction.predicted_delta_r.shape == (len(score_x),)
    assert np.isfinite(prediction.predicted_delta_r).all()


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_models_exclude_ties_before_fitting(model):
    train_x, delta_r, uniqueness, score_x = _arrays()
    tied_x = np.vstack([train_x, np.full((3, train_x.shape[1]), 1_000_000.0)])
    tied_delta = np.concatenate([delta_r, np.array([0.0, 5e-13, -5e-13])])
    tied_uniqueness = np.concatenate([uniqueness, np.full(3, 10_000.0)])
    config = DirectionModelConfig(xgb_estimators=10)

    if model == "logreg":
        baseline = fit_predict_value_logreg(
            train_x=train_x,
            delta_r=delta_r,
            uniqueness=uniqueness,
            score_x=score_x,
            config=config,
        ).score
        with_ties = fit_predict_value_logreg(
            train_x=tied_x,
            delta_r=tied_delta,
            uniqueness=tied_uniqueness,
            score_x=score_x,
            config=config,
        ).score
    else:
        baseline = fit_predict_delta_xgboost(
            train_x=train_x,
            delta_r=delta_r,
            uniqueness=uniqueness,
            score_x=score_x,
            config=config,
        ).predicted_delta_r
        with_ties = fit_predict_delta_xgboost(
            train_x=tied_x,
            delta_r=tied_delta,
            uniqueness=tied_uniqueness,
            score_x=score_x,
            config=config,
        ).predicted_delta_r

    assert np.array_equal(baseline, with_ties)


@pytest.mark.parametrize("model", ["logreg", "xgboost"])
def test_models_require_both_non_tie_direction_classes(model):
    train_x, delta_r, uniqueness, score_x = _arrays()
    delta_r = np.abs(delta_r)
    function = (
        fit_predict_value_logreg if model == "logreg" else fit_predict_delta_xgboost
    )
    with pytest.raises(ValueError, match="both direction classes"):
        function(
            train_x=train_x,
            delta_r=delta_r,
            uniqueness=uniqueness,
            score_x=score_x,
            config=DirectionModelConfig(xgb_estimators=8),
        )
