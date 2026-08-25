"""Causal geometry and strict three-trigger tests for Notebook J."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.supervisor_channel import (
    SupervisorSignalConfig,
    generate_supervisor_signals,
    project_closed_hourly_channel,
)


def _hourly_context() -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=3, freq="1h", tz="UTC")
    slope = np.log(1.01)
    return pd.DataFrame(
        {
            "channel_mid": [100.0, 110.0, 120.0],
            "channel_upper": [110.0, 121.0, 132.0],
            "channel_lower": [90.0, 99.0, 108.0],
            "channel_slope": [slope, slope, slope],
            "channel_r2": [0.80, 0.85, 0.90],
            "channel_regime": ["up", "up", "up"],
            "channel_episode_id": [7, 7, 7],
            "channel_confluence": [1, 1, 1],
            "channel_confluence_count": [3, 3, 3],
            "rsi_channel": [45.0, 55.0, 65.0],
        },
        index=index,
    )


def _ltf_projection_fixture() -> pd.DataFrame:
    index = pd.date_range("2025-01-01 00:50", periods=18, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 10.0,
            "minute_count": 5,
        },
        index=index,
    )


def test_projection_uses_only_a_closed_hour_and_extends_its_slope():
    projected = project_closed_hourly_channel(
        _hourly_context(), _ltf_projection_fixture(), channel_bar="1h"
    )

    at_hour_close = pd.Timestamp("2025-01-01 00:55", tz="UTC")
    five_minutes_later = pd.Timestamp("2025-01-01 01:00", tz="UTC")
    assert projected.loc[at_hour_close, "channel_mid"] == pytest.approx(100.0)
    assert projected.loc[at_hour_close, "rsi_channel"] == pytest.approx(45.0)
    assert projected.loc[five_minutes_later, "channel_mid"] == pytest.approx(
        100.0 * np.exp(np.log(1.01) * 5.0 / 60.0)
    )
    assert projected.loc[at_hour_close, "channel_availability_time"] == pd.Timestamp(
        "2025-01-01 01:00", tz="UTC"
    )


def test_future_hour_changes_cannot_change_an_earlier_projection():
    hourly = _hourly_context()
    changed = hourly.copy()
    changed.loc[pd.Timestamp("2025-01-01 02:00", tz="UTC"), [
        "channel_mid", "channel_upper", "channel_lower", "channel_slope"
    ]] = [999.0, 1099.0, 899.0, 0.5]

    original = project_closed_hourly_channel(hourly, _ltf_projection_fixture())
    altered = project_closed_hourly_channel(changed, _ltf_projection_fixture())
    cutoff = pd.Timestamp("2025-01-01 03:00", tz="UTC")
    columns = ["channel_mid", "channel_upper", "channel_lower", "channel_slope"]
    pd.testing.assert_frame_equal(
        original.loc[original["availability_time"] < cutoff, columns],
        altered.loc[altered["availability_time"] < cutoff, columns],
    )


def test_incomplete_five_minute_bar_cannot_carry_a_tradeable_regime():
    ltf = _ltf_projection_fixture()
    bad = pd.Timestamp("2025-01-01 01:10", tz="UTC")
    ltf.loc[bad, "minute_count"] = 4

    projected = project_closed_hourly_channel(_hourly_context(), ltf)

    assert projected.loc[bad, "channel_regime"] == "none"
    assert np.isnan(projected.loc[bad, "channel_mid"])


def _signal_frame(side: str, setup: str) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=6, freq="5min", tz="UTC")
    common = {
        "volume": [100.0] * 6,
        "taker_buy_base": [50.0] * 6,
        "channel_lower": [90.0] * 6,
        "channel_mid": [100.0] * 6,
        "channel_upper": [110.0] * 6,
        "channel_slope": [0.01 if side == "long" else -0.01] * 6,
        "channel_r2": [0.80] * 6,
        "channel_regime": ["up" if side == "long" else "down"] * 6,
        "channel_episode_id": [4] * 6,
        "channel_confluence": [1] * 6,
        "channel_confluence_count": [2] * 6,
        "minute_count": [5] * 6,
    }
    if side == "long" and setup == "edge_rejection":
        ohlc = {
            "open": [104, 96, 96, 98.5, 100, 100],
            "high": [105, 97, 99, 100.5, 101, 101],
            "low": [103, 94, 92, 98, 99, 99],
            "close": [104, 96, 98.5, 100.2, 100, 100],
        }
    elif side == "long" and setup == "midline_retest":
        ohlc = {
            "open": [104, 101, 100.5, 101.5, 103, 103],
            "high": [105, 102, 102, 103.0, 104, 104],
            "low": [103, 99.5, 98, 101, 102, 102],
            "close": [104, 100.5, 101.5, 102.8, 103, 103],
        }
    elif side == "short" and setup == "edge_rejection":
        ohlc = {
            "open": [96, 104, 104, 101.5, 100, 100],
            "high": [97, 106, 108, 102, 101, 101],
            "low": [95, 103, 101, 99.5, 99, 99],
            "close": [96, 104, 102, 99.8, 100, 100],
        }
    else:
        raise ValueError((side, setup))
    return pd.DataFrame({**ohlc, **common}, index=index)


def test_edge_t1_t2_t3_require_strictly_later_bars():
    frame = _signal_frame("long", "edge_rejection")
    signals = generate_supervisor_signals(frame, SupervisorSignalConfig())

    assert signals["edge_stage_1"].iloc[1] == 1
    assert signals["edge_stage_2"].iloc[1] == 0
    assert signals["edge_stage_2"].iloc[2] == 1
    assert signals["edge_stage_3"].iloc[2] == 0
    assert signals["edge_stage_3"].iloc[3] == 1
    assert signals["edge_stage_1_side"].iloc[1] == "long"
    assert signals["edge_stage_2_side"].iloc[2] == "long"
    assert signals["edge_stage_3_side"].iloc[3] == "long"
    assert signals["signal"].iloc[3] == 1
    assert signals["setup_type"].iloc[3] == "edge_rejection"
    assert signals["signal_swing_low"].iloc[3] == pytest.approx(92.0)
    assert signals["t1_time"].iloc[3] == frame.index[1]
    assert signals["t2_time"].iloc[3] == frame.index[2]


def test_midline_retest_is_labelled_separately():
    frame = _signal_frame("long", "midline_retest")
    signals = generate_supervisor_signals(frame, SupervisorSignalConfig())

    assert signals["midline_stage_1"].iloc[1] == 1
    assert signals["midline_stage_2"].iloc[2] == 1
    assert signals["midline_stage_3"].iloc[3] == 1
    assert signals["signal"].iloc[3] == 1
    assert signals["setup_type"].iloc[3] == "midline_retest"


def test_short_signal_uses_the_rejection_high_as_its_structural_stop_reference():
    frame = _signal_frame("short", "edge_rejection")
    signals = generate_supervisor_signals(frame, SupervisorSignalConfig())

    assert signals["signal"].iloc[3] == -1
    assert signals["edge_stage_3_side"].iloc[3] == "short"
    assert signals["signal_swing_high"].iloc[3] == pytest.approx(108.0)


def test_regime_flip_cancels_an_armed_sequence():
    frame = _signal_frame("long", "edge_rejection")
    frame.loc[frame.index[2]:, "channel_regime"] = "down"

    signals = generate_supervisor_signals(frame, SupervisorSignalConfig())

    assert (signals["signal"] != 0).sum() == 0


def test_pre_entry_stop_touch_permanently_cancels_the_sequence():
    frame = _signal_frame("long", "edge_rejection")
    # T2 low is 92.0, so the frozen 5 bps buffered stop is about 91.954.
    # The would-be T3 bar closes above T2 high but first trades through that stop.
    frame.loc[frame.index[3], "low"] = 91.0

    signals = generate_supervisor_signals(frame, SupervisorSignalConfig())

    assert signals["edge_stage_2"].iloc[2] == 1
    assert signals["edge_stage_3"].iloc[3] == 0
    assert (signals["signal"] != 0).sum() == 0
