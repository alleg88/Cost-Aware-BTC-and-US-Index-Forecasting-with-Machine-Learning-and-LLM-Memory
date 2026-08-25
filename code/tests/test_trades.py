from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.trades import simulate_bracket_trades, trade_stats


def _bars(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """Build an OHLC frame from (open, high, low, close) rows on a UTC M15 grid."""
    idx = pd.date_range("2025-01-01", periods=len(rows), freq="15min", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def _signal(bars: pd.DataFrame, at: int, cls: int) -> pd.Series:
    pred = pd.Series(1, index=bars.index, dtype="int64")
    pred.iloc[at] = cls
    return pred


def test_long_take_profit_first():
    bars = _bars([
        (100, 100, 100, 100),   # signal bar
        (100, 100.4, 99.9, 100.2),
        (100.2, 101.2, 100.0, 101.0),   # high touches TP=101
        (101.0, 103.0, 100.9, 102.9),   # never reached
    ])
    ledger, per_bar = simulate_bracket_trades(
        bars, _signal(bars, 0, 2), tp_bps=100, sl_bps=100, max_hold=10, fee_bps=0)
    assert len(ledger) == 1
    t = ledger.iloc[0]
    assert t.exit_reason == "take_profit"
    assert t.entry_price == pytest.approx(100.0)     # next-bar open
    assert t.exit_price == pytest.approx(101.0)      # exact TP price
    assert t.gross_return == pytest.approx(0.01)
    assert t.bars_held == 2
    # ledger and per-bar series reconcile exactly
    assert per_bar.sum() == pytest.approx(ledger.net_return.sum())
    assert per_bar.iloc[3] == 0.0                    # nothing after the exit


def test_long_stop_loss_first():
    bars = _bars([
        (100, 100, 100, 100),
        (100, 100.4, 99.05, 99.1),      # low touches SL=99
        (99.1, 105, 99, 105),
    ])
    ledger, _ = simulate_bracket_trades(
        bars, _signal(bars, 0, 2), tp_bps=100, sl_bps=100, max_hold=10, fee_bps=0)
    t = ledger.iloc[0]
    assert t.exit_reason == "stop_loss"
    assert t.exit_price == pytest.approx(99.0)
    assert t.gross_return == pytest.approx(-0.01)


def test_both_barriers_in_one_bar_assumes_stop_first():
    bars = _bars([
        (100, 100, 100, 100),
        (100, 101.5, 98.5, 100.0),      # wide bar: touches both TP=101 and SL=99
    ])
    ledger, _ = simulate_bracket_trades(
        bars, _signal(bars, 0, 2), tp_bps=100, sl_bps=100, max_hold=10, fee_bps=0)
    assert ledger.iloc[0].exit_reason == "stop_loss"


def test_timeout_exits_at_close():
    bars = _bars([(100, 100.1, 99.9, 100)] * 5)
    ledger, _ = simulate_bracket_trades(
        bars, _signal(bars, 0, 2), tp_bps=100, sl_bps=100, max_hold=3, fee_bps=0)
    t = ledger.iloc[0]
    assert t.exit_reason == "timeout"
    assert t.bars_held == 3
    assert t.exit_price == pytest.approx(100.0)


def test_short_side_mirrors_and_uses_own_brackets():
    bars = _bars([
        (100, 100, 100, 100),
        (100, 100.2, 99.45, 99.5),      # short TP at 99.5 (50 bps) touched, SL 100.25 not
    ])
    ledger, _ = simulate_bracket_trades(
        bars, _signal(bars, 0, 0), tp_bps=100, sl_bps=100, max_hold=10,
        fee_bps=0, tp_bps_short=50, sl_bps_short=25)
    t = ledger.iloc[0]
    assert t.side == -1
    assert t.exit_reason == "take_profit"
    assert t.exit_price == pytest.approx(99.5)
    assert t.gross_return == pytest.approx(0.005)


def test_short_stop_loss_on_high():
    bars = _bars([
        (100, 100, 100, 100),
        (100, 100.30, 99.8, 100.1),     # high touches short SL=100.25 (25 bps)
    ])
    ledger, _ = simulate_bracket_trades(
        bars, _signal(bars, 0, 0), tp_bps=100, sl_bps=25, max_hold=10, fee_bps=0)
    t = ledger.iloc[0]
    assert t.exit_reason == "stop_loss"
    assert t.gross_return == pytest.approx(-0.0025)


def test_fees_and_slippage_charged_per_side():
    bars = _bars([
        (100, 100, 100, 100),
        (100, 101.5, 99.9, 101.2),      # TP=101 hit
    ])
    ledger, per_bar = simulate_bracket_trades(
        bars, _signal(bars, 0, 2), tp_bps=100, sl_bps=100, max_hold=10,
        fee_bps=5, slippage_bps=2)
    t = ledger.iloc[0]
    assert t.net_return == pytest.approx(0.01 - 2 * 0.0007)
    assert per_bar.sum() == pytest.approx(t.net_return)


def test_confidence_gate_blocks_entry():
    bars = _bars([(100, 100.1, 99.9, 100)] * 4)
    pred = _signal(bars, 0, 2)
    conf = pd.Series(0.5, index=bars.index)
    ledger, per_bar = simulate_bracket_trades(
        bars, pred, conf, tau=0.7, tp_bps=100, sl_bps=100, max_hold=3, fee_bps=0)
    assert len(ledger) == 0
    assert (per_bar == 0.0).all()


def test_signal_on_last_bar_opens_nothing():
    bars = _bars([(100, 100.1, 99.9, 100)] * 3)
    ledger, _ = simulate_bracket_trades(
        bars, _signal(bars, 2, 2), tp_bps=100, sl_bps=100, max_hold=3, fee_bps=0)
    assert len(ledger) == 0


def test_one_trade_at_a_time():
    # signals on every bar, but the first trade holds 3 bars: overlapping ones ignored
    bars = _bars([(100, 100.1, 99.9, 100)] * 8)
    pred = pd.Series(2, index=bars.index, dtype="int64")
    ledger, _ = simulate_bracket_trades(
        bars, pred, tp_bps=100, sl_bps=100, max_hold=3, fee_bps=0)
    entries = ledger.entry_time.tolist()
    exits = ledger.exit_time.tolist()
    for k in range(1, len(ledger)):
        assert entries[k] > exits[k - 1]


def test_per_bar_series_reconciles_over_many_trades():
    rng = np.random.default_rng(7)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    open_ = np.roll(close, 1); open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.002, n))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.002, n))
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    bars = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)
    pred = pd.Series(rng.choice([0, 1, 2], size=n), index=idx)

    ledger, per_bar = simulate_bracket_trades(
        bars, pred, tp_bps=40, sl_bps=30, max_hold=8, fee_bps=5, slippage_bps=2)
    assert len(ledger) > 10
    assert per_bar.sum() == pytest.approx(ledger.net_return.sum(), abs=1e-12)


def test_vol_scaled_barriers():
    bars = _bars([
        (100, 100, 100, 100),
        (100, 100.6, 99.8, 100.5),      # TP = 2 * 0.25% = 0.5% -> 100.5 touched
    ])
    bars["vol_20"] = 0.0025
    ledger, _ = simulate_bracket_trades(
        bars, _signal(bars, 0, 2), tp_bps=2.0, sl_bps=4.0, max_hold=10,
        fee_bps=0, vol_scale_col="vol_20")
    t = ledger.iloc[0]
    assert t.exit_reason == "take_profit"
    assert t.exit_price == pytest.approx(100.5)


def test_trade_stats_summary():
    bars = _bars([
        (100, 100, 100, 100),
        (100, 101.5, 99.9, 101.2),      # trade 1: TP +1%
        (101.2, 101.3, 101.1, 101.2),   # signal bar for trade 2
        (101.2, 101.3, 100.1, 100.3),   # trade 2: SL -1%
        (100.3, 100.4, 100.2, 100.3),
    ])
    pred = pd.Series([2, 1, 2, 1, 1], index=bars.index, dtype="int64")
    ledger, _ = simulate_bracket_trades(
        bars, pred, tp_bps=100, sl_bps=100, max_hold=4, fee_bps=0)
    stats = trade_stats(ledger)
    assert stats["n_trades"] == 2
    assert stats["win_rate"] == pytest.approx(0.5)
    assert stats["tp_rate"] == pytest.approx(0.5)
    assert stats["sl_rate"] == pytest.approx(0.5)
    assert stats["profit_factor"] == pytest.approx(1.0, abs=0.05)


def test_empty_ledger_stats():
    stats = trade_stats(simulate_bracket_trades(
        _bars([(100, 100, 100, 100)] * 3),
        pd.Series(1, index=_bars([(100, 100, 100, 100)] * 3).index),
        tp_bps=100, sl_bps=100, max_hold=3, fee_bps=0)[0])
    assert stats["n_trades"] == 0
    assert stats["win_rate"] == 0.0
