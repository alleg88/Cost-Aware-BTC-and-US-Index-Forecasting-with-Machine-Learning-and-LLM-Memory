"""Correctness tests for the 1-minute intrabar exit engine."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.trades import simulate_bracket_trades
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar


def _m15(n: int, start="2025-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="15min", tz="UTC")


def _minutes(m15_start, n_bars: int) -> pd.DatetimeIndex:
    return pd.date_range(m15_start, periods=n_bars * 15, freq="1min", tz="UTC")


def test_reconciles_per_bar_with_ledger():
    idx = _m15(40)
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.3, 40))
    bars = pd.DataFrame({"open": close, "high": close + 0.6,
                         "low": close - 0.6, "close": close}, index=idx)
    # build 1m bars: 15 minutes per M15 bar interpolating open->close, around its OHLC
    rows = []
    for t, r in bars.iterrows():
        m = pd.date_range(t, periods=15, freq="1min", tz="UTC")
        c = np.linspace(r["open"], r["close"], 15)
        rows.append(pd.DataFrame({"open": c, "high": np.maximum(c, r["high"]),
                                  "low": np.minimum(c, r["low"]), "close": c}, index=m))
    minute = pd.concat(rows)
    pred = pd.Series(2, index=idx)  # always "up"
    ledger, per_bar = simulate_bracket_trades_intrabar(
        bars, minute, pred, tp_bps=50, sl_bps=50, max_hold=8, fee_bps=5)
    assert np.isclose(per_bar.sum(), ledger["net_return"].sum(), atol=1e-9)


def test_matches_m15_engine_when_minutes_equal_m15():
    """Feeding the M15 bars as their own 'minute' bars must reproduce the M15 engine."""
    idx = _m15(60)
    rng = np.random.default_rng(1)
    close = 100 + np.cumsum(rng.normal(0, 0.4, 60))
    bars = pd.DataFrame({"open": close + rng.normal(0, 0.1, 60),
                         "high": close + np.abs(rng.normal(0, 0.5, 60)),
                         "low": close - np.abs(rng.normal(0, 0.5, 60)),
                         "close": close}, index=idx)
    bars["high"] = bars[["open", "high", "close"]].max(axis=1)
    bars["low"] = bars[["open", "low", "close"]].min(axis=1)
    pred = pd.Series(rng.choice([0, 1, 2], size=60), index=idx)
    conf = pd.Series(rng.uniform(0.4, 0.9, 60), index=idx)

    kw = dict(tau=0.5, tp_bps=40, sl_bps=30, max_hold=6, fee_bps=5, slippage_bps=2.0)
    led_m15, pb_m15 = simulate_bracket_trades(bars, pred, conf, **kw)
    led_1m, pb_1m = simulate_bracket_trades_intrabar(bars, bars, pred, conf, **kw)

    pd.testing.assert_series_equal(pb_m15, pb_1m, check_names=False)
    assert led_m15["exit_reason"].tolist() == led_1m["exit_reason"].tolist()
    assert np.allclose(led_m15["net_return"], led_1m["net_return"])


def test_intrabar_order_beats_stop_first():
    """TP touched in an early minute, SL in a later minute of the SAME M15 bar:
    the M15 engine mis-charges a stop; the 1m engine correctly books the TP."""
    idx = _m15(4)
    bars = pd.DataFrame({"open": [100, 100, 100, 100],
                         "high": [100, 103, 100, 100],   # entry bar (idx1) spans TP and SL
                         "low": [100, 97, 100, 100],
                         "close": [100, 100, 100, 100]}, index=idx)
    # 1m detail for the entry M15 bar (idx1): price rises to TP (+2%) first, THEN falls to SL
    m = _minutes(idx[0], 4)  # 60 one-minute bars covering 4 M15 bars
    price = np.full(60, 100.0)
    # minutes 15..29 belong to M15 bar idx1; make it hit +2% at min 17, then -2% at min 25
    price[15:30] = 100.0
    hi = np.full(60, 100.0); lo = np.full(60, 100.0)
    hi[17] = 102.5   # take-profit (tp=200bps) reached first
    lo[25] = 97.5    # stop-loss (sl=200bps) only later
    minute = pd.DataFrame({"open": price, "high": np.maximum(price, hi),
                           "low": np.minimum(price, lo), "close": price}, index=m)
    pred = pd.Series({idx[0]: 2})  # signal at bar0 -> enter at bar1 open

    led_m15, _ = simulate_bracket_trades(bars, pred, tp_bps=200, sl_bps=200,
                                         max_hold=1, fee_bps=0)
    led_1m, _ = simulate_bracket_trades_intrabar(bars, minute, pred, tp_bps=200,
                                                 sl_bps=200, max_hold=1, fee_bps=0)
    assert led_m15.iloc[0]["exit_reason"] == "stop_loss"     # conservative M15
    assert led_1m.iloc[0]["exit_reason"] == "take_profit"    # true order on 1m
    assert led_1m.iloc[0]["gross_return"] > 0


def test_trailing_stop_locks_in_profit():
    """A trade that runs up then reverses exits at the trailed stop, not at timeout."""
    idx = _m15(3)
    bars = pd.DataFrame({"open": [100, 100, 100], "high": [100, 100, 100],
                         "low": [100, 100, 100], "close": [100, 100, 100]}, index=idx)
    m = _minutes(idx[0], 3)
    price = np.full(45, 100.0)
    hi = price.copy(); lo = price.copy()
    # entry bar = idx1 (minutes 15..29): climb to +3%, then fall back
    for k, mi in enumerate(range(15, 30)):
        price[mi] = 100.0 + (3.0 if k < 8 else 3.0 - (k - 8) * 1.0)  # up to 103 then down
    hi = price.copy(); lo = price.copy()
    minute = pd.DataFrame({"open": price, "high": hi, "low": lo, "close": price}, index=m)
    pred = pd.Series({idx[0]: 2})
    # no fixed TP within reach (tp huge), trailing 100bps behind the high
    led, _ = simulate_bracket_trades_intrabar(bars, minute, pred, tp_bps=10_000,
                                              sl_bps=10_000, max_hold=1, fee_bps=0,
                                              trail_bps=100)
    assert len(led) == 1
    assert led.iloc[0]["exit_reason"] == "stop_loss"       # trailing stop fires
    assert led.iloc[0]["gross_return"] > 0                 # locked in a profit


def test_trailing_stop_accepts_a_causal_per_signal_series():
    idx = _m15(3)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=idx,
    )
    m = _minutes(idx[0], 3)
    price = np.full(45, 100.0)
    price[15:22] = np.linspace(100.0, 102.0, 7)
    price[22:30] = np.linspace(102.0, 100.0, 8)
    minute = pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price}, index=m
    )
    pred = pd.Series({idx[0]: 2})
    trail = pd.Series([50.0, 100.0, 150.0], index=idx)

    ledger, _ = simulate_bracket_trades_intrabar(
        bars,
        minute,
        pred,
        tp_bps=10_000,
        sl_bps=10_000,
        max_hold=1,
        fee_bps=0,
        trail_bps=trail,
    )

    assert len(ledger) == 1
    assert ledger.iloc[0]["exit_reason"] == "stop_loss"


def test_breakeven_stop_caps_the_loss():
    """Once +be is reached the stop moves to entry; a reversal exits near breakeven."""
    idx = _m15(3)
    bars = pd.DataFrame({"open": [100, 100, 100], "high": [100, 100, 100],
                         "low": [100, 100, 100], "close": [100, 100, 100]}, index=idx)
    m = _minutes(idx[0], 3)
    price = np.full(45, 100.0)
    for k, mi in enumerate(range(15, 30)):
        price[mi] = 100.0 + (1.5 if k < 6 else 1.5 - (k - 6) * 0.5)  # +1.5% then down through entry
    minute = pd.DataFrame({"open": price, "high": price, "low": price, "close": price}, index=m)
    pred = pd.Series({idx[0]: 2})
    led, _ = simulate_bracket_trades_intrabar(bars, minute, pred, tp_bps=10_000,
                                              sl_bps=300, max_hold=1, fee_bps=0,
                                              be_trigger_bps=100)
    assert len(led) == 1
    # breakeven stop (at entry) fires instead of the -3% hard stop
    assert led.iloc[0]["gross_return"] >= -1e-9
    assert led.iloc[0]["gross_return"] > -0.03


def test_microsecond_parquet_timestamps_respect_one_bar_hold():
    """Datetime resolution must not stretch a 15-minute hold into future bars."""
    idx = _m15(4).as_unit("us")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        index=idx,
    )
    minute_index = _minutes(idx[0], 4).as_unit("us")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        index=minute_index,
    )
    minute.loc[pd.Timestamp("2025-01-01 00:30", tz="UTC"), "high"] = 102.0
    pred = pd.Series({idx[0]: 2})

    ledger, _ = simulate_bracket_trades_intrabar(
        bars, minute, pred, tp_bps=100, sl_bps=100, max_hold=1, fee_bps=0
    )

    assert ledger.iloc[0]["exit_reason"] == "timeout"
    assert ledger.iloc[0]["exit_time"] == idx[1]
