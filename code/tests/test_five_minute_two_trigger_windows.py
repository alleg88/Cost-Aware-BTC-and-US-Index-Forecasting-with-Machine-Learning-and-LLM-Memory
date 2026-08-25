"""Behavior tests for causal two-trigger 5-minute opportunity windows."""

import numpy as np
import pandas as pd

from experiments.five_minute_two_trigger_windows import (
    TwoTriggerConfig,
    detect_fast_two_trigger_windows,
    detect_two_trigger_windows,
)


def _channel_frame(*, regime: str = "up", rows: int = 18) -> pd.DataFrame:
    decision_time = pd.date_range("2024-01-01", periods=rows, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "decision_time": decision_time,
            "open": 105.0,
            "high": 106.0,
            "low": 104.0,
            "close": 105.0,
            "channel_lower": 100.0,
            "channel_upper": 110.0,
            "channel_r2": 0.60,
            "channel_regime": regime,
            "channel_confluence": 1,
            "channel_episode_id": 7,
            "minute_count": 5,
        }
    )
    return frame


def _long_touch(frame: pd.DataFrame, row: int) -> None:
    frame.loc[row, ["open", "high", "low", "close"]] = [104.0, 105.0, 101.0, 103.0]


def _minute_frame(*, periods: int = 80) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00", periods=periods, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": 104.8, "high": 104.95, "low": 104.7, "close": 104.9},
        index=index,
    )


def test_long_confirmation_opens_next_bar_and_cooldown_suppresses_reentry():
    frame = _channel_frame()
    _long_touch(frame, 1)
    frame.loc[2, ["open", "high", "low", "close"]] = [104.5, 106.0, 104.0, 105.5]
    _long_touch(frame, 4)  # Valid T1 geometry, but still inside the 60-minute cooldown.
    _long_touch(frame, 14)  # Exactly 60 minutes after the first T2: eligible again.
    frame.loc[15, ["open", "high", "low", "close"]] = [104.8, 106.2, 104.5, 105.6]

    result = detect_two_trigger_windows(frame, config=TwoTriggerConfig())

    assert list(result.windows["t1_time"]) == [
        pd.Timestamp("2024-01-01 00:05", tz="UTC"),
        pd.Timestamp("2024-01-01 01:10", tz="UTC"),
    ]
    assert list(result.windows["t2_time"]) == [
        pd.Timestamp("2024-01-01 00:10", tz="UTC"),
        pd.Timestamp("2024-01-01 01:15", tz="UTC"),
    ]
    assert list(result.windows["window_start"]) == [
        pd.Timestamp("2024-01-01 00:15", tz="UTC"),
        pd.Timestamp("2024-01-01 01:20", tz="UTC"),
    ]
    assert result.windows["side"].tolist() == ["long", "long"]
    assert result.audit["cooldown_touches"] == 1


def test_short_confirmation_is_the_causal_mirror_and_may_arrive_on_third_bar():
    frame = _channel_frame(regime="down", rows=8)
    frame.loc[1, ["open", "high", "low", "close"]] = [106.5, 109.0, 105.5, 107.0]
    frame.loc[2:3, "close"] = 106.0  # Neither interim bar breaks the T1 low.
    frame.loc[4, ["open", "high", "low", "close"]] = [106.0, 106.2, 104.8, 105.0]

    result = detect_two_trigger_windows(frame, config=TwoTriggerConfig())

    assert len(result.windows) == 1
    window = result.windows.iloc[0]
    assert window["side"] == "short"
    assert window["t1_time"] == pd.Timestamp("2024-01-01 00:05", tz="UTC")
    assert window["t2_time"] == pd.Timestamp("2024-01-01 00:20", tz="UTC")
    assert window["window_start"] == pd.Timestamp("2024-01-01 00:25", tz="UTC")


def test_confirmation_after_three_bars_is_expired_not_backfilled():
    frame = _channel_frame(rows=8)
    _long_touch(frame, 1)
    frame.loc[5, ["open", "high", "low", "close"]] = [104.5, 106.0, 104.0, 105.5]

    result = detect_two_trigger_windows(frame, config=TwoTriggerConfig())

    assert result.windows.empty
    assert result.audit["expired_t1"] == 1


def test_incomplete_bar_cannot_arm_or_confirm_a_window():
    frame = _channel_frame(rows=6)
    _long_touch(frame, 1)
    frame.loc[1, "minute_count"] = 4
    frame.loc[2, ["open", "high", "low", "close"]] = [104.5, 106.0, 104.0, 105.5]

    result = detect_two_trigger_windows(frame, config=TwoTriggerConfig())

    assert result.windows.empty
    assert result.audit["armed_t1"] == 0


def test_fast_t2_confirms_on_fourth_completed_minute_and_enters_next_minute():
    channels = _channel_frame(rows=6)
    _long_touch(channels, 1)  # T1 is known at 00:05 UTC.
    minute = _minute_frame(periods=30)
    # Bar 00:08--00:09 is the fourth completed minute after T1.
    minute.loc[pd.Timestamp("2024-01-01 00:08", tz="UTC"), ["high", "close"]] = [105.2, 105.1]

    result = detect_fast_two_trigger_windows(
        channels,
        minute,
        config=TwoTriggerConfig(),
        confirmation_buffer_bps=0.0,
    )

    assert len(result.windows) == 1
    window = result.windows.iloc[0]
    assert window["t1_time"] == pd.Timestamp("2024-01-01 00:05", tz="UTC")
    assert window["t2_time"] == pd.Timestamp("2024-01-01 00:09", tz="UTC")
    # The bar stamped 00:08 closes at 00:09; the next 1m Open is also 00:09.
    assert window["window_start"] == pd.Timestamp("2024-01-01 00:09", tz="UTC")
    assert window["confirmation_lag_minutes"] == 4


def test_fast_t2_requires_minute_close_not_intraminute_high():
    channels = _channel_frame(rows=6)
    _long_touch(channels, 1)
    minute = _minute_frame(periods=30)
    minute.loc[pd.Timestamp("2024-01-01 00:08", tz="UTC"), ["high", "close"]] = [106.0, 104.9]

    result = detect_fast_two_trigger_windows(
        channels,
        minute,
        config=TwoTriggerConfig(),
        confirmation_buffer_bps=0.0,
    )

    assert result.windows.empty
    assert result.audit["expired_t1"] == 1


def test_fast_t2_short_is_the_causal_minute_mirror():
    channels = _channel_frame(regime="down", rows=6)
    channels.loc[1, ["open", "high", "low", "close"]] = [106.5, 109.0, 105.5, 107.0]
    minute = _minute_frame(periods=30)
    minute.loc[:, ["open", "high", "low", "close"]] = [106.0, 106.1, 105.8, 106.0]
    minute.loc[pd.Timestamp("2024-01-01 00:08", tz="UTC"), ["low", "close"]] = [105.3, 105.4]

    result = detect_fast_two_trigger_windows(
        channels,
        minute,
        config=TwoTriggerConfig(),
        confirmation_buffer_bps=2.0,
    )

    assert len(result.windows) == 1
    window = result.windows.iloc[0]
    assert window["side"] == "short"
    assert window["t2_time"] == pd.Timestamp("2024-01-01 00:09", tz="UTC")
    assert window["confirmation_lag_minutes"] == 4


def test_fast_t2_buffer_and_cooldown_are_applied_at_minute_resolution():
    channels = _channel_frame(rows=18)
    _long_touch(channels, 1)
    _long_touch(channels, 4)   # 00:20, inside cooldown after the first Fast-T2.
    _long_touch(channels, 14)  # 01:10, first 5m decision after cooldown expiry.
    minute = _minute_frame(periods=90)
    # 105.01 is above T1 high but below the 2 bps confirmation buffer.
    minute.loc[pd.Timestamp("2024-01-01 00:05", tz="UTC"), ["high", "close"]] = [105.02, 105.01]
    minute.loc[pd.Timestamp("2024-01-01 00:06", tz="UTC"), ["high", "close"]] = [105.05, 105.03]
    minute.loc[pd.Timestamp("2024-01-01 01:10", tz="UTC"), ["high", "close"]] = [105.06, 105.04]

    result = detect_fast_two_trigger_windows(
        channels,
        minute,
        config=TwoTriggerConfig(),
        confirmation_buffer_bps=2.0,
    )

    assert list(result.windows["t2_time"]) == [
        pd.Timestamp("2024-01-01 00:07", tz="UTC"),
        pd.Timestamp("2024-01-01 01:11", tz="UTC"),
    ]
    assert result.audit["cooldown_touches"] == 1


def test_fast_t2_invalidates_setup_when_a_minute_is_missing():
    channels = _channel_frame(rows=6)
    _long_touch(channels, 1)
    minute = _minute_frame(periods=30).drop(pd.Timestamp("2024-01-01 00:06", tz="UTC"))
    minute.loc[pd.Timestamp("2024-01-01 00:08", tz="UTC"), ["high", "close"]] = [105.2, 105.1]

    result = detect_fast_two_trigger_windows(
        channels,
        minute,
        config=TwoTriggerConfig(),
        confirmation_buffer_bps=0.0,
    )

    assert result.windows.empty
    assert result.audit["missing_minute_invalidations"] == 1
