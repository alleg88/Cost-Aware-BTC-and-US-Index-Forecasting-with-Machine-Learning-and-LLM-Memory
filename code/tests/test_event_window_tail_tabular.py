from __future__ import annotations

import inspect

import numpy as np
import pytest

from experiments.event_window_tail_tabular import (
    XGBoostTailConfig,
    fit_predict_logreg,
    fit_predict_tail_model,
    fit_predict_xgboost,
)


def _score_x(fill: float = 0.0) -> np.ndarray:
    return np.array(
        [
            [fill, -0.5, 0.0, 1.0],
            [fill, 0.5, 1.0, -1.0],
            [fill, 1.5, -1.0, 0.5],
        ],
        dtype=float,
    )


def _small_training_problem(
    *,
    score_fill: float = 0.0,
    weights: np.ndarray | None = None,
    timeout_rows: int | None = None,
    timeout_episodes: int | None = None,
) -> dict[str, np.ndarray]:
    if timeout_rows is None:
        outcome = np.tile(np.array([0, 1, 2], dtype=np.int64), 4)
    else:
        outcome = np.concatenate(
            [
                np.zeros(timeout_rows, dtype=np.int64),
                np.ones(timeout_rows, dtype=np.int64),
                np.full(timeout_rows, 2, dtype=np.int64),
            ]
        )
    rows = len(outcome)
    index = np.arange(rows, dtype=float)
    train_x = np.column_stack(
        [
            np.where(index % 4 == 0, np.nan, index - rows / 2),
            outcome + (index % 3) * 0.1,
            np.sin(index),
            np.cos(index / 2),
        ]
    )
    train_x[0] = np.array([-8.0, 3.0, 2.0, -2.0])
    timeout_target = np.full(rows, np.nan, dtype=float)
    timeout_mask = outcome == 2
    timeout_target[timeout_mask] = np.linspace(-0.5, 0.5, timeout_mask.sum())
    if weights is None:
        weights = np.ones(rows, dtype=float)
    episode_ids = np.arange(rows, dtype=np.int64)
    if timeout_episodes is not None:
        episode_ids[timeout_mask] = np.arange(timeout_mask.sum()) % timeout_episodes
    return {
        "train_x": train_x,
        "outcome": outcome,
        "timeout_target": timeout_target,
        "sample_weight": np.asarray(weights, dtype=float),
        "score_x": _score_x(score_fill),
        "episode_ids": episode_ids,
    }


@pytest.mark.parametrize("model_name", ["logreg", "xgboost"])
def test_tabular_model_returns_three_logits_and_timeout_prediction(model_name):
    prediction = fit_predict_tail_model(model_name, **_small_training_problem())
    assert prediction.logits.shape == (len(_score_x()), 3)
    assert prediction.timeout_net_r.shape == (len(_score_x()),)
    assert np.isfinite(prediction.logits).all()


def test_imputer_and_scaler_ignore_score_rows():
    first = fit_predict_logreg(**_small_training_problem(score_fill=0.0))
    second = fit_predict_logreg(**_small_training_problem(score_fill=9999.0))
    assert first.model_metadata["fit_median"] == second.model_metadata["fit_median"]


def test_sample_weight_changes_logreg_fit():
    light = fit_predict_logreg(**_small_training_problem(weights=np.ones(12)))
    heavy = fit_predict_logreg(
        **_small_training_problem(weights=np.array([20.0] + [1.0] * 11))
    )
    assert not np.allclose(light.logits, heavy.logits)


def test_xgboost_configuration_is_frozen():
    config = XGBoostTailConfig()
    assert (config.n_estimators, config.max_depth, config.learning_rate) == (
        300,
        3,
        0.03,
    )
    assert (config.min_child_weight, config.reg_lambda, config.n_jobs) == (
        20.0,
        10.0,
        1,
    )


def test_tabular_predictions_repeat_with_seed_42():
    first = fit_predict_xgboost(**_small_training_problem())
    second = fit_predict_xgboost(**_small_training_problem())
    np.testing.assert_allclose(first.logits, second.logits)


def test_future_score_labels_cannot_change_tabular_prediction():
    assert "score_outcome" not in inspect.signature(fit_predict_logreg).parameters
    assert "score_outcome" not in inspect.signature(fit_predict_xgboost).parameters


def test_logreg_timeout_prediction_is_fit_fold_mean():
    prediction = fit_predict_logreg(**_small_training_problem())
    assert np.unique(prediction.timeout_net_r).size == 1


def test_xgboost_timeout_fallback_is_explicit():
    prediction = fit_predict_xgboost(
        **_small_training_problem(timeout_rows=10, timeout_episodes=2)
    )
    assert prediction.model_metadata["timeout_fallback"] is True


def test_xgboost_timeout_head_requires_registered_row_and_episode_support():
    prediction = fit_predict_xgboost(
        **_small_training_problem(timeout_rows=200, timeout_episodes=20),
        config=XGBoostTailConfig(n_estimators=2),
    )
    assert prediction.model_metadata["timeout_fallback"] is False


def test_unknown_tabular_model_is_rejected():
    with pytest.raises(ValueError, match="unsupported tabular model"):
        fit_predict_tail_model("catboost", **_small_training_problem())
