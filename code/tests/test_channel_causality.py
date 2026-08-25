"""Guards that matter for the channel study.

The synthetic-data tests in test_linear_channels.py check that the modules produce
the right shapes. These check the properties a wrong result would still satisfy:
that nothing looks into the future, that the execution rules are the pessimistic
ones they claim to be, and that the sealed quarter stays sealed.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from features.linear_channels import (
    channel_episode_id,
    compute_linear_regression_channels,
    label_channel_regime,
)
from features.channel_faucet import generate_channel_faucet_signals
from evaluation.channel_backtest import backtest_channel_strategy

DATA = Path(__file__).resolve().parents[1] / "data"


def _bars(n=400, seed=7):
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    close = 100 + 0.05 * t + np.sin(t * 0.3) * 1.5 + rng.normal(0, 0.4, n)
    high = close + np.abs(rng.normal(0.4, 0.15, n))
    low = close - np.abs(rng.normal(0.4, 0.15, n))
    opn = close - rng.normal(0, 0.25, n)
    vol = rng.uniform(50, 150, n)
    return pd.DataFrame(
        {"open": opn, "high": high, "low": low, "close": close,
         "volume": vol, "taker_buy_base": vol * rng.uniform(0.3, 0.7, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
    )


def _pipeline(df):
    ch = compute_linear_regression_channels(df, window=30, num_std=2.0)
    ch["channel_regime"] = label_channel_regime(ch, min_slope=0.0, min_r2=0.3)
    return generate_channel_faucet_signals(
        ch, long_pos_threshold=0.45, short_pos_threshold=0.55,
        rsi_oversold=50.0, rsi_overbought=50.0, regime_col="channel_regime",
    )


CUT = 300


def test_channel_geometry_does_not_look_ahead():
    """Replacing every bar after CUT must leave the geometry before CUT untouched."""
    base = _bars()
    tampered = base.copy()
    tampered.iloc[CUT:] *= 1.5

    a = compute_linear_regression_channels(base, window=30)
    b = compute_linear_regression_channels(tampered, window=30)

    for col in ("channel_slope", "channel_mid", "channel_upper",
                "channel_lower", "channel_pos", "channel_r2", "rsi"):
        np.testing.assert_allclose(
            a[col].to_numpy()[:CUT], b[col].to_numpy()[:CUT],
            equal_nan=True, err_msg=f"{col} changed when only future bars moved",
        )


def test_faucet_signals_do_not_look_ahead():
    base = _bars()
    tampered = base.copy()
    tampered.iloc[CUT:] *= 1.5
    a, b = _pipeline(base), _pipeline(tampered)
    for col in ("signal_1_long", "signal_1_short", "signal_2", "signal_3", "signal"):
        np.testing.assert_array_equal(
            a[col].to_numpy()[:CUT], b[col].to_numpy()[:CUT],
            err_msg=f"{col} changed when only future bars moved",
        )


def _sequenced_faucet_fixture(*, confirming_taker_buy: float = 80.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    return pd.DataFrame(
        {"open": [100, 100, 100.8, 101.0, 102.0],
         "high": [100.5, 101.2, 101.3, 102.5, 102.5],
         "low": [99.5, 98.0, 98.5, 100.5, 101.5],
         "close": [100, 101.0, 101.1, 102.0, 102.0],
         "volume": [100] * 5,
         "taker_buy_base": [50, 50, 50, confirming_taker_buy, 50],
         "channel_pos": [0.5, 0.2, 0.5, 0.5, 0.5],
         "channel_slope": [10.0] * 5,
         "channel_regime": ["up"] * 5,
         "rsi": [50, 30, 31, 34, 50]},
        index=idx,
    )


def test_faucet_stages_must_occur_on_strictly_later_bars():
    """A reversal candle on the arming bar cannot count as the second tap."""
    sig = generate_channel_faucet_signals(
        _sequenced_faucet_fixture(), require_flow=True,
        regime_col="channel_regime",
    )
    assert sig["signal_1_long"].iloc[1] == 1
    assert sig["signal_2"].iloc[1] == 0
    assert sig["signal_2"].iloc[2] == 1
    assert sig["signal_3"].iloc[2] == 0
    assert sig["signal_3"].iloc[3] == 1


def test_required_flow_cannot_be_bypassed_by_rsi_recovery():
    """Wrong-signed aggressive flow must reject a flow-required confirmation."""
    sig = generate_channel_faucet_signals(
        _sequenced_faucet_fixture(confirming_taker_buy=10.0),
        require_flow=True, regime_col="channel_regime",
    )
    assert sig["taker_imbalance"].iloc[3] == pytest.approx(-0.8)
    assert sig["signal_3"].sum() == 0
    assert (sig["signal"] != 0).sum() == 0


def test_faucet_can_rearm_after_an_expired_window_in_the_same_channel():
    """One missed reversal must not disable the rest of a long channel episode."""
    idx = pd.date_range("2026-01-01", periods=8, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": [100, 100, 100, 100, 100, 100, 100.8, 101],
         "high": [100.2, 100.2, 100.2, 100.2, 100.2, 100.2, 101.3, 102.5],
         "low": [99.8, 99.8, 99.8, 99.8, 99.8, 99.8, 98.5, 100.5],
         "close": [100, 100, 100, 100, 100, 100, 101.1, 102],
         "volume": [100] * 8, "taker_buy_base": [50] * 7 + [80],
         "channel_pos": [0.5, 0.2, 0.5, 0.5, 0.5, 0.2, 0.5, 0.5],
         "channel_slope": [10.0] * 8, "channel_regime": ["up"] * 8,
         "rsi": [50, 30, 50, 50, 50, 30, 31, 34]},
        index=idx,
    )
    sig = generate_channel_faucet_signals(
        df, arm_max_bars=3, require_flow=True, regime_col="channel_regime",
    )
    assert sig["signal_2"].iloc[6] == 1
    assert sig["signal_3"].iloc[7] == 1


def test_entry_price_is_the_next_bar_open():
    sig = _pipeline(_bars())
    res = backtest_channel_strategy(sig, target_mode="rr", stop_mode="swing",
                                    rr_multiple=2.0, max_hold_bars=24)
    if res["num_trades"] == 0:
        pytest.skip("no trades on this synthetic series")
    trades = res["trades_df"]
    for _, tr in trades.iterrows():
        assert tr["entry"] == pytest.approx(sig.loc[tr["entry_time"], "open"]), (
            "entry must fill at the next bar's open, not the signal bar's close"
        )


@pytest.mark.parametrize(
    ("signal", "later_high", "later_low", "expected_target"),
    [(1, 105.0, 99.0, 104.0), (-1, 101.0, 95.0, 96.0)],
)
def test_measured_target_freezes_the_signal_time_swing_range_at_entry(
    signal, later_high, later_low, expected_target,
):
    idx = pd.date_range("2026-01-01", periods=6, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [101.0, 102.0, 101.0, later_high, 130.0, 130.0],
            "low": [99.0, 98.0, 99.0, later_low, 70.0, 70.0],
            "close": [100.0] * 6,
            "signal": [0, signal, 0, 0, 0, 0],
        },
        index=idx,
    )

    result = backtest_channel_strategy(
        df,
        target_mode="measured",
        stop_mode="pct",
        sl_pct=0.10,
        swing_lookback=2,
        max_hold_bars=4,
    )

    trade = result["trades_df"].iloc[0]
    assert trade["target"] == pytest.approx(expected_target)
    assert trade["outcome"] == "tp"


def test_none_capacity_limit_allows_overlapping_trades():
    idx = pd.date_range("2026-01-01", periods=15, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {
            "open": [100.0] * 15,
            "high": [100.1] * 15,
            "low": [99.9] * 15,
            "close": [100.0] * 15,
            "signal": [1] * 6 + [0] * 9,
        },
        index=idx,
    )

    result = backtest_channel_strategy(
        df,
        target_mode="pct",
        stop_mode="pct",
        tp_pct=0.20,
        sl_pct=0.20,
        max_hold_bars=5,
        max_concurrent=None,
    )

    assert result["num_trades"] == 6
    assert result["skipped"]["capacity"] == 0


def test_a_bar_touching_both_levels_is_scored_as_the_stop():
    """The pessimistic convention: bar data cannot order the two touches."""
    idx = pd.date_range("2026-01-01", periods=6, freq="15min", tz="UTC")
    df = pd.DataFrame(
        {"open": [100, 100, 100, 100, 100, 100],
         "high": [100, 100, 110, 110, 110, 110],     # target reachable
         "low": [100, 100, 90, 90, 90, 90],          # stop reachable, same bar
         "close": [100, 100, 100, 100, 100, 100],
         "signal": [0, 1, 0, 0, 0, 0],
         "channel_upper": 110.0, "channel_lower": 90.0},
        index=idx,
    )
    res = backtest_channel_strategy(df, target_mode="rr", stop_mode="pct",
                                    sl_pct=0.05, rr_multiple=2.0, max_hold_bars=4)
    assert res["num_trades"] == 1
    assert res["trades_df"].iloc[0]["outcome"] == "sl"


def test_entry_bar_is_part_of_the_stop_first_path():
    """A fill bar cannot disappear from the realised price path."""
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": [100, 100, 100],
         "high": [100, 102, 100.2],
         "low": [100, 98, 99.8],
         "close": [100, 100, 100],
         "signal": [1, 0, 0]},
        index=idx,
    )
    res = backtest_channel_strategy(
        df, target_mode="pct", stop_mode="pct", tp_pct=0.01, sl_pct=0.01,
        max_hold_bars=2,
    )
    assert res["trades_df"].iloc[0]["outcome"] == "sl"


def test_one_minute_path_orders_hits_inside_the_entry_bar():
    """The observed 1m path overrides an ambiguous stop-first 5m candle."""
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {"open": [100, 100, 100],
         "high": [100, 102, 100.2],
         "low": [100, 98, 99.8],
         "close": [100, 100, 100],
         "signal": [1, 0, 0]},
        index=idx,
    )
    minute_idx = pd.date_range(idx[1], periods=5, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {"open": [100, 101, 100, 100, 100],
         "high": [101.2, 101.1, 100.2, 100.2, 100.2],
         "low": [99.8, 98.5, 99.8, 99.8, 99.8],
         "close": [101, 100, 100, 100, 100]},
        index=minute_idx,
    )
    res = backtest_channel_strategy(
        bars, target_mode="pct", stop_mode="pct", tp_pct=0.01, sl_pct=0.01,
        max_hold_bars=1, execution_1m=minute,
    )
    trade = res["trades_df"].iloc[0]
    assert trade["outcome"] == "tp"
    assert trade["exit_time"] == minute_idx[0]


def test_missing_minute_in_execution_path_censors_the_label():
    """A missing minute before timeout makes the path unknowable, not flat."""
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {"open": [100] * 3, "high": [100.1] * 3, "low": [99.9] * 3,
         "close": [100] * 3, "signal": [1, 0, 0]}, index=idx,
    )
    minute_idx = pd.date_range(idx[1], periods=5, freq="1min", tz="UTC").delete(2)
    minute = pd.DataFrame(
        {"open": [100] * 4, "high": [100.1] * 4,
         "low": [99.9] * 4, "close": [100] * 4},
        index=minute_idx,
    )
    res = backtest_channel_strategy(
        bars, target_mode="pct", stop_mode="pct", tp_pct=0.01, sl_pct=0.01,
        max_hold_bars=1, execution_1m=minute,
    )
    assert res["num_trades"] == 0
    assert res["skipped"]["censored"] == 1
    assert res["orders_df"].iloc[0]["status"] == "censored"
    assert pd.isna(res["orders_df"].iloc[0]["r_net"])


def test_maker_fill_and_exit_follow_the_one_minute_path():
    """A resting order fills on its first minute touch and exits on later minutes."""
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {"open": [100] * 3, "high": [100.2, 100.5, 100.2],
         "low": [99.8, 98.4, 99.8], "close": [100] * 3,
         "signal": [1, 0, 0]}, index=idx,
    )
    minute_idx = pd.date_range(idx[1], periods=10, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {"open": [100, 99.5] + [100] * 8,
         "high": [100.2, 99.8] + [100.2] * 8,
         "low": [99.4, 98.4] + [99.8] * 8,
         "close": [99.5, 99.0] + [100] * 8},
        index=minute_idx,
    )
    res = backtest_channel_strategy(
        bars, target_mode="pct", stop_mode="pct", tp_pct=0.01, sl_pct=0.01,
        entry_mode="maker_limit", limit_offset_bps=50, fill_window_bars=1,
        max_hold_bars=1, execution_1m=minute,
    )
    trade = res["trades_df"].iloc[0]
    assert trade["entry"] == pytest.approx(99.5)
    assert trade["entry_time"] == minute_idx[0]
    assert trade["outcome"] == "sl"
    assert trade["exit_time"] == minute_idx[1]
    assert len(res["orders_df"]) == 1
    order = res["orders_df"].iloc[0]
    assert order["status"] == "filled"
    assert order["filled"]
    assert order["decision_time"] == idx[1]
    assert order["r_net"] == pytest.approx(trade["r_net"])


def test_unfilled_maker_order_is_kept_for_event_labels():
    """Dropping unfilled orders would train the model only on adverse-selection fills."""
    idx = pd.date_range("2026-01-01", periods=4, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {"open": [100] * 4, "high": [100.2] * 4,
         "low": [99.8] * 4, "close": [100] * 4,
         "signal": [1, 0, 0, 0]}, index=idx,
    )
    res = backtest_channel_strategy(
        bars, target_mode="pct", stop_mode="pct", tp_pct=0.01, sl_pct=0.01,
        entry_mode="maker_limit", limit_offset_bps=100, fill_window_bars=2,
    )
    assert res["num_trades"] == 0
    assert len(res["orders_df"]) == 1
    order = res["orders_df"].iloc[0]
    assert order["decision_time"] == idx[1]
    assert order["status"] == "unfilled"
    assert not order["filled"]
    assert order["r_net"] == pytest.approx(0.0)


def test_max_hold_counts_exactly_the_requested_number_of_bars():
    """A target on bar three is outside a two-bar holding horizon."""
    idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": [100] * 5,
         "high": [100, 100.2, 100.2, 102, 102],
         "low": [100, 99.8, 99.8, 99.8, 99.8],
         "close": [100] * 5,
         "signal": [1, 0, 0, 0, 0]},
        index=idx,
    )
    res = backtest_channel_strategy(
        df, target_mode="pct", stop_mode="pct", tp_pct=0.01, sl_pct=0.01,
        max_hold_bars=2,
    )
    trade = res["trades_df"].iloc[0]
    assert trade["outcome"] == "timeout"
    assert trade["exit_time"] == idx[2]


def test_explicit_minute_horizons_are_grid_invariant():
    """A wall-clock horizon must not change when the decision grid changes."""
    start = pd.Timestamp("2026-01-01 00:05", tz="UTC")
    minute_idx = pd.date_range(start, periods=60, freq="1min")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
        index=minute_idx,
    )
    five = pd.DataFrame(
        {"open": [100.0, 100.0], "high": [100.1, 100.1],
         "low": [99.9, 99.9], "close": [100.0, 100.0],
         "signal": [1, 0]},
        index=pd.date_range("2026-01-01", periods=2, freq="5min", tz="UTC"),
    )
    one = pd.DataFrame(
        {"open": [100.0, 100.0], "high": [100.1, 100.1],
         "low": [99.9, 99.9], "close": [100.0, 100.0],
         "signal": [1, 0]},
        index=pd.date_range("2026-01-01 00:04", periods=2, freq="1min", tz="UTC"),
    )
    common = dict(
        target_mode="pct", stop_mode="pct", tp_pct=0.10, sl_pct=0.10,
        execution_1m=minute, max_hold_minutes=60, fill_window_minutes=20,
    )

    five_result = backtest_channel_strategy(five, **common)
    one_result = backtest_channel_strategy(one, **common)

    five_trade = five_result["trades_df"].iloc[0]
    one_trade = one_result["trades_df"].iloc[0]
    assert five_trade["entry_time"] == one_trade["entry_time"] == start
    assert five_trade["exit_time"] == one_trade["exit_time"] == minute_idx[-1]
    assert five_trade["outcome"] == one_trade["outcome"] == "timeout"


def test_precomputed_swing_geometry_and_measured_move_are_frozen_at_signal():
    """The 15m geometry can be reused by either decision grid without recomputing it."""
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    bars = pd.DataFrame(
        {"open": [100.0, 100.0, 100.0],
         "high": [100.1, 102.1, 100.1],
         "low": [99.9, 99.5, 99.9],
         "close": [100.0, 102.0, 100.0],
         "signal": [1, 0, 0],
         "frozen_swing_low": [99.0, 1.0, 1.0],
         "frozen_swing_high": [101.0, 500.0, 500.0],
         "frozen_measured_move": [2.0, 499.0, 499.0]},
        index=idx,
    )

    result = backtest_channel_strategy(
        bars, target_mode="measured", stop_mode="swing", swing_lookback=12,
        swing_low_col="frozen_swing_low", swing_high_col="frozen_swing_high",
        measured_move_col="frozen_measured_move", stop_buffer_bps=0.0,
        max_hold_bars=1,
    )

    trade = result["trades_df"].iloc[0]
    assert trade["stop"] == pytest.approx(99.0)
    assert trade["target"] == pytest.approx(102.0)
    assert trade["outcome"] == "tp"


def test_truncated_horizon_is_censored_instead_of_labelled_timeout():
    """A split boundary is not evidence that the economic timeout occurred."""
    idx = pd.date_range("2026-01-01", periods=3, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": [100] * 3,
         "high": [100.1] * 3,
         "low": [99.9] * 3,
         "close": [100] * 3,
         "signal": [1, 0, 0]},
        index=idx,
    )
    res = backtest_channel_strategy(
        df, target_mode="pct", stop_mode="pct", tp_pct=0.02, sl_pct=0.02,
        max_hold_bars=10,
    )
    assert res["num_trades"] == 0
    assert res["skipped"]["censored"] == 1


def test_maker_order_is_cancelled_when_channel_ends_before_fill():
    """A resting order must not enter a channel that has already disappeared."""
    idx = pd.date_range("2026-01-01", periods=5, freq="5min", tz="UTC")
    df = pd.DataFrame(
        {"open": [100] * 5,
         "high": [100.2, 101, 101, 101, 101],
         "low": [99.8, 100.1, 99, 99, 99],
         "close": [100] * 5,
         "signal": [1, 0, 0, 0, 0],
         "channel_regime": ["up", "none", "none", "none", "none"]},
        index=idx,
    )
    res = backtest_channel_strategy(
        df, target_mode="pct", stop_mode="pct", tp_pct=0.01, sl_pct=0.01,
        entry_mode="maker_limit", fill_window_bars=4,
        regime_col="channel_regime", max_concurrent=10,
    )
    assert res["num_trades"] == 0
    assert res["skipped"]["channel_cancelled"] == 1
    assert res["orders_df"].iloc[0]["status"] == "channel_cancelled"
    assert res["orders_df"].iloc[0]["r_net"] == pytest.approx(0.0)


def test_incomplete_bars_invalidate_their_channel_window():
    df = _bars()
    df["minute_count"] = 15
    df.iloc[200, df.columns.get_loc("minute_count")] = 9      # one short bar
    ch = compute_linear_regression_channels(df, window=30, require_complete_bars=True)
    # the short bar poisons every window that contains it
    assert ch["channel_slope"].iloc[200:230].isna().all()
    assert ch["channel_slope"].iloc[150:199].notna().any()


def test_missing_timestamp_invalidates_the_channel_window():
    """Ten observations spanning eleven hours are not a complete ten-hour window."""
    idx = pd.date_range("2026-01-01", periods=30, freq="1h", tz="UTC").delete(15)
    df = pd.DataFrame(
        {"close": 100 + np.arange(len(idx), dtype=float) * 0.1,
         "minute_count": 60},
        index=idx,
    )
    ch = compute_linear_regression_channels(df, window=10, require_complete_bars=True)
    assert ch.loc[idx[15:24], "channel_slope"].isna().all()


def test_episode_ids_group_contiguous_regime_runs():
    regime = pd.Series(["none", "up", "up", "up", "none", "down", "down"])
    eid = channel_episode_id(regime)
    assert eid.tolist() == [1, 2, 2, 2, 3, 4, 4]
    assert regime.groupby(eid).nunique().eq(1).all()


def test_regime_persistence_suppresses_one_bar_flickers():
    df = _bars()
    ch = compute_linear_regression_channels(df, window=30)
    loose = label_channel_regime(ch, min_slope=0.0, min_r2=0.0, persist_bars=1)
    strict = label_channel_regime(ch, min_slope=0.0, min_r2=0.0, persist_bars=6)
    assert (strict == "none").sum() >= (loose == "none").sum()
    runs = strict[strict != "none"].groupby(
        (strict != strict.shift()).cumsum()[strict != "none"]).size()
    assert runs.empty or runs.min() >= 1


@pytest.mark.skipif(not (DATA / "btcusdt_m15_lockbox_2026Q2.parquet").exists(),
                    reason="lockbox snapshot not present")
def test_working_and_lockbox_snapshots_do_not_overlap():
    work = pd.read_parquet(DATA / "btcusdt_m15_2024_2025.parquet")
    lock = pd.read_parquet(DATA / "btcusdt_m15_lockbox_2026Q2.parquet")
    shared = work.index.intersection(lock.index)
    assert len(shared) == 0, (
        f"{len(shared)} bar(s) sit in both snapshots, first {shared[0] if len(shared) else None}"
    )


def test_existing_unseal_manifest_blocks_a_second_lockbox_run(tmp_path):
    from experiments.channel_study import authorise_stage

    marker = tmp_path / "LOCKBOX_UNSEALED.json"
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(PermissionError, match="already unsealed"):
        authorise_stage("lockbox", unsealing=True, marker=marker)


@pytest.mark.skipif(not (DATA / "btcusdt_15min_2021_2026.parquet").exists(),
                    reason="extended grid not built")
def test_derived_grid_matches_the_native_m15_snapshot():
    """The derived grids stand in for a native download, so they must reproduce it."""
    native = pd.read_parquet(DATA / "btcusdt_m15_2024_2025.parquet")
    derived = pd.read_parquet(DATA / "btcusdt_15min_2021_2026.parquet")
    common = native.index.intersection(derived.index)
    assert len(common) > 50_000
    for col in ("open", "high", "low", "close"):
        np.testing.assert_array_equal(
            native.loc[common, col].to_numpy(), derived.loc[common, col].to_numpy(),
            err_msg=f"derived {col} differs from the native M15 snapshot",
        )
