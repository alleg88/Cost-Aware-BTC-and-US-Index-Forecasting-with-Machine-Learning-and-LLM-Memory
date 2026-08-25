"""
Unit tests for Linear Regression Channel Strategy components:
- Rolling channel calculation
- Dataset regime slicing
- Two-stage signal arming & reversal trigger ("faucet")
- Strategy backtesting & TP/SL execution
"""

import numpy as np
import pandas as pd
import pytest

from features.linear_channels import (
    channel_confluence,
    compute_rsi,
    compute_linear_regression_channels,
    gate_channel_signals,
    slice_dataset_by_channel_regime,
)
from features.channel_faucet import generate_channel_faucet_signals
from evaluation.channel_backtest import backtest_channel_strategy


def create_synthetic_channel_data(n_bars: int = 150) -> pd.DataFrame:
    """Create synthetic price data with predictable upward and downward trends."""
    np.random.seed(42)
    timestamps = pd.date_range("2026-01-01", periods=n_bars, freq="5min")

    # Upward trend for first 75 bars, downward trend for next 75 bars
    t = np.arange(n_bars, dtype=float)
    trend = np.where(t < 75, 100.0 + 0.5 * t, 137.5 - 0.5 * (t - 75))
    noise = np.sin(t * 0.4) * 2.0 + np.random.normal(0, 0.2, size=n_bars)
    close = trend + noise

    high = close + np.abs(np.random.normal(0.5, 0.2, size=n_bars))
    low = close - np.abs(np.random.normal(0.5, 0.2, size=n_bars))
    open_price = close - np.random.normal(0, 0.3, size=n_bars)

    df = pd.DataFrame({
        "timestamp": timestamps,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
    })
    df.set_index("timestamp", inplace=True)
    return df


def test_compute_linear_regression_channels():
    df = create_synthetic_channel_data(100)
    df_ch = compute_linear_regression_channels(df, window=30, num_std=2.0)

    assert "channel_slope" in df_ch.columns
    assert "channel_mid" in df_ch.columns
    assert "channel_upper" in df_ch.columns
    assert "channel_lower" in df_ch.columns
    assert "channel_pos" in df_ch.columns
    assert "rsi" in df_ch.columns

    # Check slope sign during known upward phase (bars 35-65)
    upward_slopes = df_ch["channel_slope"].iloc[35:65].dropna()
    assert (upward_slopes > 0).all()

    # Check upper > lower channel bounds
    valid = df_ch.dropna(subset=["channel_upper", "channel_lower"])
    assert (valid["channel_upper"] >= valid["channel_lower"]).all()


def test_channel_confluence_counts_only_windows_matching_the_primary_direction():
    regimes = pd.DataFrame(
        {
            "60": ["up", "up", "down", "none"],
            "90": ["up", "down", "down", "up"],
            "120": ["none", "down", "down", "up"],
        },
        index=pd.RangeIndex(4),
    )

    got = channel_confluence(regimes, primary="60", min_agree=2)

    assert got["channel_confluence_count"].tolist() == [2, 1, 3, 0]
    assert got["channel_confluence"].tolist() == [1, 0, 1, 0]


def test_confluence_gate_removes_only_non_confluent_policy_signals():
    candidates = pd.DataFrame(
        {"signal": [1, -1, 1, 0], "channel_confluence": [1, 0, 1, 0]}
    )

    policy = gate_channel_signals(candidates)

    assert policy["signal"].tolist() == [1, 0, 1, 0]
    assert candidates["signal"].tolist() == [1, -1, 1, 0]


def test_slice_dataset_by_channel_regime():
    df = create_synthetic_channel_data(120)
    df_ch = compute_linear_regression_channels(df, window=30)
    df_long, df_short = slice_dataset_by_channel_regime(df_ch, min_slope=0.1, channel_pos_entry_bound=0.40)

    # Check sliced dataset lengths are smaller than original
    assert len(df_long) < len(df)
    assert len(df_short) < len(df)

    # Check long regime has positive slope
    if not df_long.empty:
        assert (df_long["channel_slope"] > 0).all()

    # Check short regime has negative slope
    if not df_short.empty:
        assert (df_short["channel_slope"] < 0).all()


def test_generate_channel_faucet_signals():
    df = create_synthetic_channel_data(120)
    df_ch = compute_linear_regression_channels(df, window=25)
    df_sig = generate_channel_faucet_signals(
        df_ch,
        long_pos_threshold=0.40,
        short_pos_threshold=0.60,
        rsi_oversold=45.0,
        rsi_overbought=55.0,
        arm_max_bars=5,
    )

    assert "signal_1_long" in df_sig.columns
    assert "signal_1_short" in df_sig.columns
    assert "signal" in df_sig.columns

    # Valid signals are +1, -1, or 0
    unique_signals = df_sig["signal"].unique()
    assert set(unique_signals).issubset({-1, 0, 1})


def test_backtest_channel_strategy():
    df = create_synthetic_channel_data(150)
    df_ch = compute_linear_regression_channels(df, window=25)
    df_sig = generate_channel_faucet_signals(
        df_ch,
        long_pos_threshold=0.45,
        short_pos_threshold=0.55,
        rsi_oversold=50.0,
        rsi_overbought=50.0,
    )

    res = backtest_channel_strategy(
        df_sig,
        tp_pct=0.015,
        sl_pct=0.008,
        max_trades_per_day=5,
    )

    assert "num_trades" in res
    assert "win_rate" in res
    assert "total_net_return" in res
    assert "trades_df" in res
    assert isinstance(res["trades_df"], pd.DataFrame)
