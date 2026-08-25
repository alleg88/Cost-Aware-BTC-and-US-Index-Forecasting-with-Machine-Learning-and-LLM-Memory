"""Sequential entry-policy and trade-count behavior."""

import numpy as np
import pandas as pd

from evaluation.fast_t2_action_policy import (
    first_crossing_entries,
    replay_entry_capacity,
    select_inner_quantile,
    summarise_trade_activity,
)


def _decision_rows(scores) -> pd.DataFrame:
    times = pd.date_range("2024-01-01 00:00", periods=len(scores), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "window_id": "window-1",
            "decision_id": [f"window-1:{i:02d}" for i in range(len(scores))],
            "decision_time": times,
            "entry_time": times,
            "active_end_time": times + pd.Timedelta(minutes=20),
            "label_end": times + pd.Timedelta(minutes=20),
            "side": "long",
            "channel_episode_id": 1,
            "score": scores,
            "r_net": np.linspace(-0.5, 0.8, len(scores)),
            "filled": True,
            "holding_minutes": 20.0,
            "minutes_since_t2": np.arange(len(scores)),
        }
    )


def _overlapping_entries() -> pd.DataFrame:
    times = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02", "2024-01-01 00:03"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "window_id": [f"w{i}" for i in range(4)],
            "decision_id": [f"d{i}" for i in range(4)],
            "decision_time": times,
            "entry_time": times,
            "active_end_time": times + pd.Timedelta(minutes=30),
            "label_end": times + pd.Timedelta(minutes=30),
            "side": ["long", "short", "long", "short"],
            "channel_episode_id": [1, 2, 3, 4],
            "score": [0.9, 0.8, 0.7, 0.6],
            "r_net": [1.0, -1.0, 0.5, 0.2],
            "filled": True,
            "holding_minutes": 30.0,
            "minutes_since_t2": 0,
        }
    )


def test_first_crossing_enters_once_and_waits_before_it():
    ledger, actions = first_crossing_entries(
        _decision_rows([0.2, 0.4, 0.8, 0.9]), threshold=0.7
    )

    assert ledger["minutes_since_t2"].tolist() == [2]
    assert actions["action"].tolist() == ["WAIT", "WAIT", "ENTER"]


def test_no_crossing_produces_one_skip():
    ledger, actions = first_crossing_entries(
        _decision_rows([0.1, 0.2, 0.3]), threshold=0.7
    )

    assert ledger.empty
    assert actions["action"].tolist() == ["WAIT", "WAIT", "SKIP"]


def test_capacity_views_share_the_same_signal_ledger():
    entries = _overlapping_entries()

    unlimited = replay_entry_capacity(entries, capacity=None, evaluation_days=1)
    cap3 = replay_entry_capacity(entries, capacity=3, evaluation_days=1)
    cap1 = replay_entry_capacity(entries, capacity=1, evaluation_days=1)

    assert set(cap1.orders.decision_id) <= set(cap3.orders.decision_id)
    assert set(cap3.orders.decision_id) <= set(unlimited.orders.decision_id)
    assert len(unlimited.orders) == 4
    assert len(cap3.orders) == 3
    assert len(cap1.orders) == 1
    pd.testing.assert_series_equal(unlimited.input_scores, entries["score"])


def test_trade_activity_includes_zero_days_and_parallel_positions():
    replay = replay_entry_capacity(
        _overlapping_entries(), capacity=None, evaluation_days=3
    )
    metrics = replay.metrics

    assert metrics["filled_trades"] == 4
    assert metrics["trades_per_day"] == 4 / 3
    assert metrics["zero_trade_days"] == 2
    assert metrics["days_three_plus"] == 1
    assert metrics["long_trades"] == 2
    assert metrics["short_trades"] == 2
    assert metrics["max_concurrent"] == 4


def test_inner_quantile_selection_returns_complete_frontier():
    scored = pd.concat(
        [
            _decision_rows([0.1, 0.2, 0.8]).assign(window_id="a", decision_id=["a0", "a1", "a2"]),
            _decision_rows([0.3, 0.7, 0.9]).assign(
                window_id="b", decision_id=["b0", "b1", "b2"], side="short",
                channel_episode_id=2,
            ),
        ],
        ignore_index=True,
    )

    quantile, table = select_inner_quantile(scored, evaluation_days=2)

    assert quantile in {0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95}
    assert len(table) == 7
    assert {"threshold", "filled_trades", "trades_per_day", "mean_net_r"} <= set(table)


def test_summary_distinguishes_cancelled_submission_from_filled_trade():
    orders = _overlapping_entries().iloc[[0]].copy()
    orders.loc[:, "filled"] = False
    orders.loc[:, "r_net"] = 0.0

    metrics = summarise_trade_activity(orders, evaluation_days=1)

    assert metrics["submitted_orders"] == 1
    assert metrics["filled_trades"] == 0
    assert metrics["trades_per_day"] == 0.0


def test_capacity_replay_replaces_existing_concurrency_annotation():
    entries = _overlapping_entries().assign(concurrent_at_entry=99)

    replay = replay_entry_capacity(entries, capacity=3, evaluation_days=1)

    assert replay.orders.columns.is_unique
    assert replay.orders["concurrent_at_entry"].max() == 3
    assert replay.metrics["max_concurrent"] == 3
