"""Causality and lifecycle tests for Notebook B trading windows."""

import pandas as pd

from features.channel_windows import build_channel_window_manifest


def _window_fixture(periods: int = 12) -> pd.DataFrame:
    if periods < 10:
        raise ValueError("fixture needs at least 10 bars")
    idx = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    frame = pd.DataFrame(
        {
            "channel_regime": ["none"] * periods,
            "channel_episode_id": [1] * periods,
            "channel_r2": [0.80] * periods,
            "channel_confluence": [1] * periods,
            "channel_pos": [0.50] * periods,
            "minute_count": [15] * periods,
            "availability_time": idx + pd.Timedelta("15min"),
        },
        index=idx,
    )
    frame.iloc[2:6, frame.columns.get_loc("channel_regime")] = "up"
    frame.iloc[2:5, frame.columns.get_loc("channel_pos")] = 0.20
    frame.iloc[2:6, frame.columns.get_loc("channel_episode_id")] = 2
    frame.iloc[6:9, frame.columns.get_loc("channel_regime")] = "down"
    frame.iloc[6:8, frame.columns.get_loc("channel_pos")] = 0.80
    frame.iloc[6:9, frame.columns.get_loc("channel_episode_id")] = 3
    frame.iloc[9:, frame.columns.get_loc("channel_episode_id")] = 4
    return frame


def _long_edge_run(periods: int = 40) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "channel_regime": "up",
            "channel_episode_id": 7,
            "channel_r2": 0.70,
            "channel_confluence": 1,
            "channel_pos": 0.20,
            "minute_count": 15,
            "availability_time": idx + pd.Timedelta("15min"),
        },
        index=idx,
    )


def test_long_and_short_windows_use_opposite_channel_edges():
    frame = _window_fixture()
    manifest = build_channel_window_manifest(frame, zone=0.30)

    assert manifest["side"].tolist() == ["long", "short"]
    assert manifest["window_start"].tolist() == [
        frame["availability_time"].iloc[2],
        frame["availability_time"].iloc[6],
    ]
    assert (manifest["eligible_end_time"] <=
            manifest["window_start"] + pd.Timedelta("120min")).all()


def test_natural_end_is_retained_when_time_cap_truncates_eligible_end():
    manifest = build_channel_window_manifest(_long_edge_run(), max_duration="120min")

    row = manifest.iloc[0]
    assert row["natural_end_time"] > row["eligible_end_time"]
    assert row["eligible_end_time"] == row["window_start"] + pd.Timedelta("120min")
    assert row["window_end_reason"] == "time_cap"


def test_incomplete_bar_and_gap_split_contiguous_windows():
    frame = _long_edge_run(periods=12)
    frame.iloc[4, frame.columns.get_loc("minute_count")] = 12
    frame = frame.drop(frame.index[8])

    manifest = build_channel_window_manifest(frame)

    assert len(manifest) == 3
    assert manifest["window_id"].is_unique
    assert {"incomplete_bar", "gap", "data_end"} == set(
        manifest["window_end_reason"]
    )


def test_future_changes_cannot_change_an_earlier_window():
    original_frame = _window_fixture(60)
    original = build_channel_window_manifest(original_frame)
    changed = original_frame.copy()
    changed.iloc[40:, changed.columns.get_loc("channel_regime")] = "up"
    changed.iloc[40:, changed.columns.get_loc("channel_pos")] = 0.10
    changed.iloc[40:, changed.columns.get_loc("channel_episode_id")] = 99

    cutoff = changed.index[40]
    expected = original[original["window_start"] < cutoff].reset_index(drop=True)
    actual = build_channel_window_manifest(changed)
    actual = actual[actual["window_start"] < cutoff].reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, actual)


def test_window_ids_are_stable_for_the_same_symbol_and_inputs():
    frame = _window_fixture()
    first = build_channel_window_manifest(frame, symbol="BTCUSDT")
    second = build_channel_window_manifest(frame.copy(), symbol="BTCUSDT")
    other = build_channel_window_manifest(frame, symbol="ETHUSDT")

    assert first["window_id"].tolist() == second["window_id"].tolist()
    assert first["window_id"].tolist() != other["window_id"].tolist()


def test_confluence_can_be_retained_as_a_feature_without_being_a_hard_gate():
    frame = _long_edge_run(periods=12)
    frame["channel_confluence"] = 0

    hard = build_channel_window_manifest(frame, require_confluence=True)
    soft = build_channel_window_manifest(frame, require_confluence=False)

    assert hard.empty
    assert len(soft) == 1
