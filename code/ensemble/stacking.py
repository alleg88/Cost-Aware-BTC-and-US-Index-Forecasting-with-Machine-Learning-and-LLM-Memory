"""Chronological holdout stacking for the RQ1/RQ2 ensemble.

Fit protocol (leak-safe by construction):
  1. Split the training data chronologically: first (1 - holdout_frac) -> base segment,
     last holdout_frac (after an embargo gap) -> meta segment.
  2. Fit every base model on the base segment only, predict class probabilities on the
     meta segment — so the meta-learner trains on genuinely out-of-sample base outputs.
  3. Fit the meta-learner (class-balanced CatBoost, per the project plan) on those
     probabilities.
  4. Refit the bases on the FULL training data so test-time base inputs come from the
     strongest available fit (standard stacking practice).

The embargo mirrors BlockingTimeSeriesSplit: it stops next-bar labels at the segment
boundary leaking base-model information into the meta segment.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _rows(obj, sl: slice):
    """Positional row slice for DataFrames/Series and plain arrays alike."""
    return obj.iloc[sl] if hasattr(obj, "iloc") else obj[sl]


class StackingEnsemble:
    """Stack base classifiers with a meta-learner trained on a chronological holdout."""

    def __init__(
        self,
        base_factories: dict[str, Callable],
        *,
        meta_factory: Callable | None = None,
        base_params: dict | None = None,
        holdout_frac: float = 0.2,
        embargo: int = 4,
    ):
        if not base_factories:
            raise ValueError("base_factories must not be empty")
        if not 0.0 < holdout_frac < 1.0:
            raise ValueError("holdout_frac must be in (0, 1)")
        self.base_factories = dict(base_factories)
        self.meta_factory = meta_factory
        self.base_params = base_params or {}
        self.holdout_frac = holdout_frac
        self.embargo = embargo
        self.classes_ = np.array([0, 1, 2])
        self._bases: dict[str, object] = {}
        self._meta = None

    def _make_base(self, name: str):
        return self.base_factories[name](self.base_params.get(name))

    def _make_meta(self):
        if self.meta_factory is not None:
            return self.meta_factory()
        from models.zoo import make_catboost

        return make_catboost({"iterations": 200, "depth": 4})

    def _meta_features(self, X, bases: dict[str, object]) -> np.ndarray:
        """Per base: 3 class probabilities + confidence (top-1) + top1-top2 margin.

        The spread features tell the meta-learner HOW SURE each base is, not just
        which class it prefers — a base that is barely picking up over flat carries
        different information than one that is certain.
        """
        blocks = []
        for name in self.base_factories:          # fixed registry order
            proba = np.asarray(bases[name].predict_proba(X), dtype=float)
            top2 = np.sort(proba, axis=1)[:, -2:]
            conf = top2[:, 1:2]
            margin = (top2[:, 1] - top2[:, 0])[:, None]
            blocks.append(np.hstack([proba, conf, margin]))
        return np.hstack(blocks)

    def fit(self, X, y):
        n = len(X)
        cut = int(n * (1.0 - self.holdout_frac))
        meta_start = cut + self.embargo
        if cut < 1 or meta_start >= n:
            raise ValueError("not enough rows for the stacking holdout split")
        X_base, y_base = _rows(X, slice(None, cut)), _rows(y, slice(None, cut))
        X_meta, y_meta = _rows(X, slice(meta_start, None)), _rows(y, slice(meta_start, None))

        holdout_bases = {name: self._make_base(name).fit(X_base, y_base)
                         for name in self.base_factories}
        Z = self._meta_features(X_meta, holdout_bases)
        self._meta = self._make_meta().fit(Z, y_meta)

        # refit on everything the ensemble is allowed to see for test-time predictions
        self._bases = {name: self._make_base(name).fit(X, y)
                       for name in self.base_factories}
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self._meta is None:
            raise RuntimeError("fit must be called before predict")
        proba = np.asarray(self._meta.predict_proba(self._meta_features(X, self._bases)))
        return proba

    def predict(self, X) -> np.ndarray:
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
