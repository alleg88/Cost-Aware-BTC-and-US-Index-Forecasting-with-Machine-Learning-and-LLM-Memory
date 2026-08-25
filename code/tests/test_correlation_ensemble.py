from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.correlation_ensemble import (
    ERROR_CORRELATION_THRESHOLD,
    SCORE_CORRELATION_THRESHOLD,
    correlation_order,
    probabilities_to_frame,
    redundancy_components,
    silhouette_components,
)


def test_redundancy_requires_both_high_score_and_error_correlation():
    names = ["a", "b", "c", "d"]
    score = pd.DataFrame(np.eye(4), index=names, columns=names)
    error = score.copy()
    score.loc["a", "b"] = score.loc["b", "a"] = SCORE_CORRELATION_THRESHOLD
    error.loc["a", "b"] = error.loc["b", "a"] = ERROR_CORRELATION_THRESHOLD
    score.loc["b", "c"] = score.loc["c", "b"] = 0.95
    error.loc["b", "c"] = error.loc["c", "b"] = 0.20
    score.loc["a", "d"] = score.loc["d", "a"] = -0.95
    error.loc["a", "d"] = error.loc["d", "a"] = 0.95

    assert redundancy_components(score, error) == (("a", "b"), ("c",), ("d",))


def test_heatmap_order_contains_each_model_once():
    matrix = pd.DataFrame(
        [[1.0, 0.9, -0.2], [0.9, 1.0, -0.1], [-0.2, -0.1, 1.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    assert set(correlation_order(matrix)) == {"a", "b", "c"}


def test_silhouette_selects_natural_correlation_groups_without_fixed_count():
    names = ["a", "b", "c", "d"]
    matrix = pd.DataFrame(
        [[1.0, 0.95, 0.05, 0.00], [0.95, 1.0, 0.00, 0.05], [0.05, 0.00, 1.0, 0.94], [0.00, 0.05, 0.94, 1.0]],
        index=names,
        columns=names,
    )
    groups, score = silhouette_components(matrix)
    assert groups == (("a", "b"), ("c", "d"))
    assert score > 0.8


def test_probability_frame_normalises_and_derives_prediction():
    frame = probabilities_to_frame(
        timestamp=pd.Series(pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")),
        y_true=pd.Series([0, 2]),
        probabilities=np.array([[2.0, 1.0, 1.0], [0.1, 0.2, 0.7]]),
        refit_id="test",
    )
    assert np.allclose(frame[["p_short", "p_flat", "p_long"]].sum(axis=1), 1.0)
    assert frame["pred"].tolist() == [0, 2]
    assert frame["refit_id"].eq("test").all()
