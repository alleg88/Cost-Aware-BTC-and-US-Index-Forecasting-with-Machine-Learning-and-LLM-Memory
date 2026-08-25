"""Causality and lifecycle contracts for broad 5-minute event windows."""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.event_windows import (
    EventWindowConfig,
    build_event_window_manifest,
    build_hourly_channel_context,
    causal_activity_ratio,
    project_hourly_channels,
)


def _hourly_fixture(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    x = np.arange(periods, dtype=float)
    log_close = np.log(100.0) + 0.00035 * x + 0.025 * np.sin(x / 2.7)
    close = np.exp(log_close)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.003,
            "low": close * 0.997,
            "close": close,
            "volume": 100.0 + x,
            "minute_count": 60,
        },
        index=index,
    )


def _raw_five_minute_fixture() -> pd.DataFrame:
    index = pd.date_range("2024-01-06", periods=288, freq="5min", tz="UTC")
    x = np.arange(len(index), dtype=float)
    close = 105.0 * np.exp(0.0001 * x)
    return pd.DataFrame(
        {
            "open": close * 0.9999,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
            "minute_count": 5,
        },
        index=index,
    )


def _manifest_fixture(periods: int = 720) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="5min", tz="UTC")
    steps = np.where(np.arange(periods) % 2 == 0, 0.001, -0.001)
    close = 100.0 * np.exp(np.cumsum(steps))
    split = 420
    slope = np.where(np.arange(periods) < split, 0.001, -0.001)
    position = np.where(slope > 0, 0.25, 0.75)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
            "minute_count": 5,
            "channel_slope_raw": slope,
            "channel_slope_bps": slope * 10_000.0,
            "channel_pos": position,
            "channel_r2": 0.01,
            "channel_confluence_count": 1,
        },
        index=index,
    )


def _half_exit_fixture() -> pd.DataFrame:
    frame = _manifest_fixture()
    frame["channel_pos"] = 0.75
    source = frame.index[320]
    frame.loc[source, "channel_pos"] = 0.25
    return frame


def _reversal_at_utc_0100_fixture() -> pd.DataFrame:
    index = pd.date_range("2023-12-30", "2024-01-02 03:00", freq="5min", tz="UTC")
    periods = len(index)
    steps = np.where(np.arange(periods) % 2 == 0, 0.001, -0.001)
    close = 100.0 * np.exp(np.cumsum(steps))
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": 10.0,
            "minute_count": 5,
            "channel_slope_raw": 0.001,
            "channel_slope_bps": 10.0,
            "channel_pos": 0.75,
            "channel_r2": 0.01,
            "channel_confluence_count": 1,
        },
        index=index,
    )
    armed = pd.Timestamp("2024-01-02 00:25", tz="UTC")
    reversal = pd.Timestamp("2024-01-02 00:55", tz="UTC")
    frame.loc[armed:reversal, "channel_pos"] = 0.25
    frame.loc[reversal:, "channel_slope_raw"] = -0.001
    frame.loc[reversal:, "channel_slope_bps"] = -10.0
    frame.loc[reversal:, "channel_pos"] = 0.75
    return frame


def _gap_fixture() -> pd.DataFrame:
    frame = _manifest_fixture(periods=960)
    frame["channel_slope_raw"] = 0.001
    frame["channel_slope_bps"] = 10.0
    frame["channel_pos"] = 0.75
    frame.loc[frame.index[300], "channel_pos"] = 0.25
    frame.loc[frame.index[700], "channel_pos"] = 0.25
    return frame.drop(frame.index[305])


def test_hourly_channel_context_fits_60_90_120_without_quality_gates():
    context = build_hourly_channel_context(_hourly_fixture(), EventWindowConfig())

    assert {
        "channel_slope_raw_60",
        "channel_slope_bps_60",
        "channel_r2",
        "channel_sign_60",
        "channel_sign_90",
        "channel_sign_120",
        "channel_confluence_count",
    } <= set(context.columns)
    assert context.loc[context["channel_r2"] < 0.10, "channel_sign_60"].ne(0).any()


def test_hourly_projection_obeys_completed_bar_availability():
    context = build_hourly_channel_context(_hourly_fixture())
    projected = project_hourly_channels(_raw_five_minute_fixture(), context)
    usable = projected["channel_availability_time"].notna()

    assert usable.any()
    assert {
        "channel_slope_raw",
        "channel_slope_bps",
        "channel_regime_age_hours",
        "channel_sign_90",
        "channel_sign_120",
        "channel_confluence",
        "channel_confluence_count",
        "channel_mid",
        "channel_upper",
        "channel_lower",
        "rsi_channel",
    } <= set(projected.columns)
    assert (
        projected.loc[usable, "channel_availability_time"]
        <= projected.loc[usable, "availability_time"]
    ).all()


def test_future_hourly_mutation_cannot_change_prior_projection():
    hourly = _hourly_fixture()
    five = _raw_five_minute_fixture()
    cutoff = pd.Timestamp("2024-01-06 12:00", tz="UTC")
    first = project_hourly_channels(five, build_hourly_channel_context(hourly))
    hourly.loc[hourly.index >= cutoff, "close"] *= 4.0
    second = project_hourly_channels(five, build_hourly_channel_context(hourly))

    pd.testing.assert_frame_equal(
        first[first.availability_time <= cutoff],
        second[second.availability_time <= cutoff],
    )


def test_activity_ratio_invalidates_a_history_that_bridges_a_gap():
    frame = _manifest_fixture(periods=620).drop(_manifest_fixture(periods=620).index[310])
    ratio = causal_activity_ratio(frame, EventWindowConfig())
    after_gap = frame.index.get_loc(pd.Timestamp("2024-01-02 01:55", tz="UTC"))

    assert ratio.iloc[after_gap : after_gap + 288].isna().all()
    assert np.isfinite(ratio.iloc[-1])


def test_window_uses_slope_sign_half_channel_and_activity_not_r2_gate():
    frame = _manifest_fixture()
    frame.loc[:, "channel_r2"] = 0.01
    manifest = build_event_window_manifest(
        frame, EventWindowConfig(activity_floor=0.8)
    )

    assert not manifest.empty
    assert set(manifest["side"]) == {"long", "short"}


def test_windows_are_non_overlapping_and_at_least_sixty_minutes_apart():
    manifest = build_event_window_manifest(_manifest_fixture(), EventWindowConfig())
    starts = pd.to_datetime(manifest["window_start"], utc=True)
    ends = pd.to_datetime(manifest["window_end"], utc=True)

    assert starts.diff().dropna().ge(pd.Timedelta("60min")).all()
    assert (
        ends.iloc[:-1].reset_index(drop=True)
        <= starts.iloc[1:].reset_index(drop=True)
    ).all()
    assert (
        starts
        == pd.to_datetime(manifest["source_bar_time"], utc=True)
        + pd.Timedelta("5min")
    ).all()
    assert (ends > starts).all()


def test_final_source_bar_cannot_create_a_zero_decision_window():
    frame = _manifest_fixture(periods=400)
    frame["channel_pos"] = 0.75
    frame.loc[frame.index[-1], "channel_pos"] = 0.25
    manifest = build_event_window_manifest(frame, EventWindowConfig())
    assert manifest.empty


def test_leaving_directional_half_does_not_close_open_window():
    manifest = build_event_window_manifest(_half_exit_fixture(), EventWindowConfig())

    assert manifest.iloc[0]["window_end"] - manifest.iloc[0]["window_start"] == pd.Timedelta(
        "60min"
    )


def test_future_changes_do_not_modify_prior_windows():
    frame = _manifest_fixture()
    cutoff = frame.index[500]
    before = build_event_window_manifest(frame, EventWindowConfig())
    changed = frame.copy()
    changed.loc[changed.index > cutoff, "close"] *= 7.0
    changed.loc[changed.index > cutoff, "channel_slope_raw"] *= -1.0
    after = build_event_window_manifest(changed, EventWindowConfig())

    pd.testing.assert_frame_equal(
        before[before.window_end <= cutoff].reset_index(drop=True),
        after[after.window_end <= cutoff].reset_index(drop=True),
    )


def test_gap_or_slope_reversal_closes_before_same_boundary_decision():
    manifest = build_event_window_manifest(
        _reversal_at_utc_0100_fixture(), EventWindowConfig()
    )
    row = manifest.iloc[0]

    assert row.window_end == pd.Timestamp("2024-01-02 01:00", tz="UTC")
    assert row.end_reason == "slope_reversal"


def test_five_minute_gap_closes_window_and_resets_episode():
    manifest = build_event_window_manifest(_gap_fixture(), EventWindowConfig())

    assert manifest.iloc[0]["end_reason"] == "data_gap"
    assert manifest.iloc[1]["channel_episode_id"] != manifest.iloc[0][
        "channel_episode_id"
    ]
