from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import make_barrier_label


def _flat_bars(n: int, px: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    return pd.DataFrame(
        {"open": px, "high": px, "low": px, "close": px}, index=idx, dtype=float)


def test_up_barrier_first():
    bars = _flat_bars(10)
    bars.loc[bars.index[3], "high"] = 101.1        # bar 0's up barrier (101) via bar 3
    y = make_barrier_label(bars, up_bps=100, down_bps=100, max_hold=5)
    assert y.iloc[0] == 2


def test_down_barrier_first():
    bars = _flat_bars(10)
    bars.loc[bars.index[2], "low"] = 98.9
    y = make_barrier_label(bars, up_bps=100, down_bps=100, max_hold=5)
    assert y.iloc[0] == 0


def test_first_touch_wins_across_bars():
    bars = _flat_bars(10)
    bars.loc[bars.index[1], "high"] = 101.0        # up touched at h=1
    bars.loc[bars.index[3], "low"] = 99.0          # down touched later at h=3
    y = make_barrier_label(bars, up_bps=100, down_bps=100, max_hold=5)
    assert y.iloc[0] == 2                          # earlier touch decides


def test_same_bar_tie_goes_down():
    bars = _flat_bars(10)
    bars.loc[bars.index[2], "high"] = 101.5        # both barriers inside bar 2
    bars.loc[bars.index[2], "low"] = 98.5
    y = make_barrier_label(bars, up_bps=100, down_bps=100, max_hold=5)
    assert y.iloc[0] == 0                          # stop-first convention


def test_timeout_is_flat_and_truncation_is_dropped():
    bars = _flat_bars(12)                          # nothing ever moves
    y = make_barrier_label(bars, up_bps=100, down_bps=100, max_hold=4)
    assert (y.iloc[: 12 - 4] == 1).all()           # full windows: time-out flat
    assert (y.iloc[12 - 4:] == -1).all()           # truncated windows: dropped


def test_truncated_row_with_decided_touch_is_kept():
    bars = _flat_bars(6)
    bars.loc[bars.index[5], "high"] = 101.2        # bar 4's window is truncated but decided
    y = make_barrier_label(bars, up_bps=100, down_bps=100, max_hold=4)
    assert y.iloc[4] == 2


def test_asymmetric_thresholds():
    bars = _flat_bars(10)
    bars.loc[bars.index[1], "high"] = 100.4        # +40 bps
    bars.loc[bars.index[1], "low"] = 99.7          # -30 bps
    # tight down barrier (25 bps) is touched, wide up barrier (50 bps) is not
    y = make_barrier_label(bars, up_bps=50, down_bps=25, max_hold=5)
    assert y.iloc[0] == 0
    # flip the asymmetry: up barrier (25) touched, down barrier (50) not
    y2 = make_barrier_label(bars, up_bps=25, down_bps=50, max_hold=5)
    assert y2.iloc[0] == 2


class _ToyModel:
    """Records the training index; predicts flat."""

    seen_train_ends: list[pd.Timestamp] = []

    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        _ToyModel.seen_train_ends.append(X.index.max())
        return self

    def predict_proba(self, X):
        return np.tile([0.1, 0.8, 0.1], (len(X), 1))


def test_train_tail_trim_keeps_lookahead_out_of_training():
    idx = pd.date_range("2024-12-01", "2025-01-07 23:45", freq="15min", tz="UTC")
    X = pd.DataFrame({"signal": np.zeros(len(idx))}, index=idx)
    y = pd.Series(np.resize([0, 1, 2], len(idx)), index=idx, name="label")
    windows = weekly_walkforward_windows(idx, "2025-01-01", "2025-01-07", train_lookback="14D")

    _ToyModel.seen_train_ends = []
    run_walkforward_predictions(
        X, y, windows=windows, model_factory=_ToyModel, train_tail_trim=16)
    trimmed_end = _ToyModel.seen_train_ends[0]

    _ToyModel.seen_train_ends = []
    run_walkforward_predictions(
        X, y, windows=windows, model_factory=_ToyModel, train_tail_trim=0)
    untrimmed_end = _ToyModel.seen_train_ends[0]

    assert untrimmed_end - trimmed_end == pd.Timedelta("15min") * 16
