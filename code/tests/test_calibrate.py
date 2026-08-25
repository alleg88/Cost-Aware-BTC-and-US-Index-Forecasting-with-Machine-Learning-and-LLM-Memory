from __future__ import annotations

import numpy as np
import pandas as pd

from models.calibrate import ChronoIsotonicCalibrated
from models.zoo import make_xgboost


def _xy(n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"f{i}" for i in range(4)])
    score = X["f0"].to_numpy() + rng.normal(0, 0.5, n)
    y = pd.Series(np.digitize(score, [-0.5, 0.5]), index=X.index)   # 0/1/2
    return X, y


def test_calibrated_proba_is_normalised_and_three_class():
    X, y = _xy(2500)
    m = ChronoIsotonicCalibrated(make_xgboost, {"n_estimators": 40, "max_depth": 3},
                                 calib_frac=0.2, min_calib_rows=100).fit(X, y)
    proba = m.predict_proba(X.tail(300))
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert m.calibrators_ is not None                 # calibration actually ran


def test_falls_back_to_uncalibrated_when_holdout_too_thin():
    X, y = _xy(400)
    m = ChronoIsotonicCalibrated(make_xgboost, {"n_estimators": 30, "max_depth": 2},
                                 calib_frac=0.2, min_calib_rows=500).fit(X, y)
    assert m.calibrators_ is None                      # too few calib rows -> skip
    proba = m.predict_proba(X.tail(50))
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_calibration_uses_only_the_fit_slice_for_the_base():
    # perturbing the calibration slice's FEATURES must not change the base model
    # (it is trained only on the earlier fit slice) — a structural leak guard.
    X, y = _xy(2500)
    m1 = ChronoIsotonicCalibrated(make_xgboost, {"n_estimators": 40, "max_depth": 3},
                                  calib_frac=0.2, min_calib_rows=100).fit(X, y)
    X2 = X.copy()
    cut = int(len(X) * 0.8)
    X2.iloc[cut:, :] = X2.iloc[cut:, :] * 5.0          # scramble only the calib slice
    m2 = ChronoIsotonicCalibrated(make_xgboost, {"n_estimators": 40, "max_depth": 3},
                                  calib_frac=0.2, min_calib_rows=100).fit(X2, y)
    base_proba_1 = m1.base_.predict_proba(X.head(10))
    base_proba_2 = m2.base_.predict_proba(X.head(10))
    assert np.allclose(base_proba_1, base_proba_2)     # base identical -> fit slice only
