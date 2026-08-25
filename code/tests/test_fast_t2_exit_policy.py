"""Causal state and hard-exit-first tests for Fast-T2 early exit."""

import numpy as np
import pandas as pd

from experiments.fast_t2_exit_policy import (
    EXIT_FEATURE_COLUMNS,
    build_exit_sequences,
    build_exit_states,
    replay_exit_policy,
)


ENTRY_TIME = pd.Timestamp("2024-01-01 01:00", tz="UTC")


def _minute_fixture() -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00", periods=240, freq="1min", tz="UTC")
    close = 100.0 + np.linspace(0.0, 0.8, len(index))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.04,
            "low": close - 0.04,
            "close": close + 0.01,
            "volume": 10.0 + np.arange(len(index)) % 5,
            "taker_buy_base": 5.2 + np.arange(len(index)) % 3 / 10.0,
            "count": 100 + np.arange(len(index)) % 7,
        },
        index=index,
    )


def _entries(*, count: int = 1) -> pd.DataFrame:
    times = pd.date_range(ENTRY_TIME, periods=count, freq="5min", tz="UTC")
    return pd.DataFrame(
        {
            "decision_id": [f"entry-{i}" for i in range(count)],
            "window_id": [f"window-{i}" for i in range(count)],
            "side": "long",
            "channel_episode_id": np.arange(count) + 10,
            "entry_time": times,
            "stop_price": 99.0,
            "target_price": 105.0,
            "channel_r2": 0.7,
            "channel_width_pct": 0.04,
            "score": 0.8,
            "filled": True,
        }
    )


def test_hard_stop_preempts_exit_model_on_same_bar():
    minute = _minute_fixture()
    stop_bar = ENTRY_TIME + pd.Timedelta(minutes=2)
    minute.loc[stop_bar, "low"] = 98.5
    states = build_exit_states(_entries(), minute)

    assert 3 not in states["minutes_held"].tolist()
    scores = pd.DataFrame(
        {
            "decision_id": ["entry-0"],
            "exit_decision_time": [stop_bar + pd.Timedelta(minutes=1)],
            "score": [0.0],
        }
    )
    replay = replay_exit_policy(
        _entries(), scores, hold_threshold=0.5, minute_bars=minute
    )

    assert replay.iloc[0].outcome == "sl"
    assert replay.iloc[0].exit_time == stop_bar


def test_exit_now_uses_next_open_and_keeps_frozen_levels():
    minute = _minute_fixture()
    decision_time = ENTRY_TIME + pd.Timedelta(minutes=5)
    scores = pd.DataFrame(
        {
            "decision_id": ["entry-0"],
            "exit_decision_time": [decision_time],
            "score": [0.1],
        }
    )

    replay = replay_exit_policy(
        _entries(), scores, hold_threshold=0.5, minute_bars=minute
    )

    assert replay.iloc[0].outcome == "model_exit"
    assert replay.iloc[0].exit_time == decision_time
    assert replay.iloc[0].exit_price == minute.loc[decision_time, "open"]
    assert replay.iloc[0].stop_price == 99.0
    assert replay.iloc[0].target_price == 105.0


def test_exit_replay_preserves_entry_signal_ids():
    entries = _entries(count=2)
    scores = pd.DataFrame(
        columns=["decision_id", "exit_decision_time", "score"]
    )

    result = replay_exit_policy(
        entries, scores, hold_threshold=0.5, minute_bars=_minute_fixture()
    )

    assert result["decision_id"].tolist() == entries["decision_id"].tolist()


def test_exit_features_are_past_only_but_labels_can_change_with_future():
    before = build_exit_states(_entries(), _minute_fixture())
    changed_minute = _minute_fixture()
    changed_minute.loc[ENTRY_TIME + pd.Timedelta(minutes=20), "low"] = 98.0
    after = build_exit_states(_entries(), changed_minute)
    boundary = ENTRY_TIME + pd.Timedelta(minutes=5)

    before_row = before.loc[before["exit_decision_time"].eq(boundary)].iloc[0]
    after_row = after.loc[after["exit_decision_time"].eq(boundary)].iloc[0]
    np.testing.assert_allclose(
        before_row[list(EXIT_FEATURE_COLUMNS)].to_numpy(dtype=float),
        after_row[list(EXIT_FEATURE_COLUMNS)].to_numpy(dtype=float),
    )
    assert before_row["baseline_r_net"] != after_row["baseline_r_net"]


def test_exit_sequences_end_at_the_last_completed_bar():
    minute = _minute_fixture()
    states = build_exit_states(_entries(), minute).iloc[[4]]

    sequences = build_exit_sequences(states, minute)

    assert sequences.shape == (1, 30, 5)
    assert np.isfinite(sequences).all()
