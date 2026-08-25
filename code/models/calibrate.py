"""Leak-free per-window probability calibration for the walk-forward.

Wraps any registry model so that, inside each walk-forward training window, the
base model is fit on an earlier "fit" slice and a per-class isotonic calibrator
is fit on a later "calibration" slice. Both slices are strictly in the past
relative to the validation week the wrapper will score, so the calibration adds
no leakage (same principle as BlockingTimeSeriesSplit).

Isotonic is monotonic, so this does not re-rank a single model's confidence — it
rescales it to an honest frequency. Its value is at the ENSEMBLE stage: putting
every base on a common, reliable probability scale before the soft-vote, so an
over-confident model can't swamp an under-confident one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from models.zoo import _aligned_proba

CLASSES = (0, 1, 2)


class ChronoIsotonicCalibrated:
    """Base model + per-class isotonic calibrator from a chronological holdout."""

    def __init__(self, base_factory, params=None, calib_frac: float = 0.2,
                 min_calib_rows: int = 500):
        self.base_factory = base_factory
        self.params = params
        self.calib_frac = calib_frac
        self.min_calib_rows = min_calib_rows

    @staticmethod
    def _fit_base(model, X, y, sample_weight):
        if sample_weight is None:
            return model.fit(X, y)
        return model.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=float))

    def fit(self, X: pd.DataFrame, y: pd.Series, sample_weight=None):
        n = len(X)
        cut = int(n * (1.0 - self.calib_frac))
        # fall back to no calibration if the holdout would be too thin
        if n - cut < self.min_calib_rows:
            self.base_ = self._fit_base(
                self.base_factory(self.params), X, y, sample_weight
            )
            self.calibrators_ = None
            self.classes_ = np.array(CLASSES)
            return self

        X_fit, y_fit = X.iloc[:cut], y.iloc[:cut]
        X_cal, y_cal = X.iloc[cut:], y.iloc[cut:]
        fit_weight = None if sample_weight is None else np.asarray(sample_weight)[:cut]
        cal_weight = None if sample_weight is None else np.asarray(sample_weight)[cut:]
        self.base_ = self._fit_base(
            self.base_factory(self.params), X_fit, y_fit, fit_weight
        )

        proba_cal = _aligned_proba(self.base_, X_cal)
        y_cal_arr = y_cal.to_numpy().astype(int)
        self.calibrators_ = {}
        for k in CLASSES:                       # one-vs-rest isotonic per class
            ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            ir.fit(
                proba_cal[:, k],
                (y_cal_arr == k).astype(float),
                sample_weight=cal_weight,
            )
            self.calibrators_[k] = ir
        self.classes_ = np.array(CLASSES)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = _aligned_proba(self.base_, X)
        if self.calibrators_ is None:
            return raw
        cal = np.column_stack([self.calibrators_[k].predict(raw[:, k]) for k in CLASSES])
        row = cal.sum(axis=1, keepdims=True)
        return np.divide(cal, row, out=np.full_like(cal, 1 / len(CLASSES)), where=row > 0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(CLASSES)[self.predict_proba(X).argmax(axis=1)]


def make_calibrated(base_factory, calib_frac: float = 0.2):
    """Return a model_factory(params) that yields a calibrated wrapper."""
    def factory(params=None):
        return ChronoIsotonicCalibrated(
            base_factory,
            params,
            calib_frac=calib_frac,
        )
    return factory
