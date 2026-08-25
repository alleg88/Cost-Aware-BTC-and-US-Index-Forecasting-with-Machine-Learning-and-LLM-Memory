from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ensemble.stacking import StackingEnsemble


class RecordingBase:
    """Fake base model that records every fit's row count."""

    fits: list[int] = []

    def __init__(self, params=None):
        pass

    def fit(self, X, y):
        RecordingBase.fits.append(len(X))
        return self

    def predict_proba(self, X):
        return np.tile([0.2, 0.5, 0.3], (len(X), 1))


class RecordingMeta:
    def __init__(self):
        self.n_rows = None

    def fit(self, Z, y):
        self.n_rows = len(Z)
        self.n_cols = Z.shape[1]
        return self

    def predict_proba(self, Z):
        return np.tile([0.1, 0.8, 0.1], (len(Z), 1))


def _data(n=100):
    X = pd.DataFrame({"f0": np.arange(n, dtype=float)})
    y = pd.Series(np.tile([0, 1, 2], n)[:n])
    return X, y


def test_meta_trains_only_on_embargoed_chronological_holdout():
    RecordingBase.fits = []
    meta = RecordingMeta()
    stack = StackingEnsemble(
        {"a": RecordingBase, "b": RecordingBase},
        meta_factory=lambda: meta,
        holdout_frac=0.2,
        embargo=4,
    )
    X, y = _data(100)
    stack.fit(X, y)

    # holdout pass: both bases fit on the first 80 rows; meta sees rows 84..99 only
    assert RecordingBase.fits[:2] == [80, 80]
    assert meta.n_rows == 16
    assert meta.n_cols == 2 * 5        # per base: 3 class probas + confidence + margin
    # final pass: bases refit on all rows for test-time predictions
    assert RecordingBase.fits[2:] == [100, 100]


def test_meta_features_include_confidence_and_margin():
    stack = StackingEnsemble({"a": RecordingBase})
    Z = stack._meta_features(pd.DataFrame({"f0": [0.0, 1.0]}), {"a": RecordingBase()})

    np.testing.assert_allclose(Z[0, :3], [0.2, 0.5, 0.3])   # class probabilities
    np.testing.assert_allclose(Z[0, 3], 0.5)                # confidence = top-1
    np.testing.assert_allclose(Z[0, 4], 0.2)                # margin = top1 - top2


def test_stack_predictions_have_ensemble_shape_and_classes():
    stack = StackingEnsemble(
        {"a": RecordingBase}, meta_factory=RecordingMeta, holdout_frac=0.2, embargo=4
    )
    X, y = _data(60)
    stack.fit(X, y)

    proba = stack.predict_proba(X.iloc[:10])
    assert proba.shape == (10, 3)
    assert set(stack.predict(X.iloc[:10])) <= {0, 1, 2}


def test_stack_rejects_degenerate_holdout():
    with pytest.raises(ValueError):
        StackingEnsemble({"a": RecordingBase}, holdout_frac=0.0)
    stack = StackingEnsemble({"a": RecordingBase}, holdout_frac=0.2, embargo=4)
    X, y = _data(5)                    # embargo swallows the whole holdout
    with pytest.raises(ValueError):
        stack.fit(X, y)
