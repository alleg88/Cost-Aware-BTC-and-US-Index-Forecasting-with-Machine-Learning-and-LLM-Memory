"""Submission throttle, capacity and threshold tests for Notebook B policy."""

import numpy as np
import pandas as pd
import pytest

from evaluation.channel_window_policy import (
    choose_frequency_matched,
    choose_tune_threshold,
    hard_macro_sensitivity,
    replay_capacity,
    sweep_thresholds,
)


def _minute_scores_every_minute(periods: int = 20) -> pd.DataFrame:
    decision = pd.date_range("2025-08-01", periods=periods, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "candidate_id": [f"c-{i}" for i in range(periods)],
            "decision_time": decision,
            "active_end_time": decision + pd.Timedelta("3min"),
            "side": np.where(np.arange(periods) % 2, "long", "short"),
            "score": 0.60,
            "r_net": 0.10,
            "filled": True,
            "window_id": [f"w-{i // 5}" for i in range(periods)],
            "channel_episode_id": np.arange(periods) // 5,
            "macro_alignment": 1,
        }
    )


def _overlapping_orders(periods: int = 40) -> pd.DataFrame:
    scored = _minute_scores_every_minute(periods)
    scored["active_end_time"] = scored["decision_time"] + pd.Timedelta("30min")
    scored["score"] = np.linspace(0.55, 0.95, periods)
    return scored


def test_only_first_crossing_in_each_five_minute_bucket_submits():
    replay = replay_capacity(_minute_scores_every_minute(), threshold=0.5, capacity=None)

    bucket_sizes = replay.orders.groupby(
        replay.orders["decision_time"].dt.floor("5min")
    ).size()
    assert bucket_sizes.max() == 1
    assert len(replay.orders) == 4


@pytest.mark.parametrize("capacity", [3, 5])
def test_capacity_counts_live_orders_and_positions(capacity):
    replay = replay_capacity(_overlapping_orders(), threshold=0.5, capacity=capacity)

    assert replay.max_concurrent <= capacity
    assert replay.metrics["capacity_skips"] > 0


def test_capacity_replay_never_changes_scores():
    scored = _overlapping_orders()
    replay = replay_capacity(scored, threshold=0.5, capacity=3)

    pd.testing.assert_series_equal(scored["score"], replay.input_scores)


def test_unfilled_live_order_consumes_capacity_until_expiry():
    scored = _minute_scores_every_minute(6).iloc[[0, 5]].copy()
    scored.loc[scored.index[0], "filled"] = False
    scored.loc[scored.index[0], "r_net"] = 0.0
    scored.loc[scored.index[0], "active_end_time"] = (
        scored.loc[scored.index[0], "decision_time"] + pd.Timedelta("20min")
    )

    replay = replay_capacity(scored, threshold=0.5, capacity=1)

    assert len(replay.orders) == 1
    assert replay.metrics["capacity_skips"] == 1


def test_tune_choice_enforces_economics_and_side_support():
    table = pd.DataFrame(
        {
            "capacity": [3, 3, 3],
            "threshold": [0.2, 0.5, 0.8],
            "mean_r_net": [-0.01, 0.08, 0.10],
            "total_net_r": [-1.0, 8.0, 5.0],
            "filled_trades": [100, 100, 25],
            "long_fills": [50, 50, 15],
            "short_fills": [50, 50, 10],
        }
    )

    assert choose_tune_threshold(table) == pytest.approx(0.5)
    matched = choose_frequency_matched(table, target_fills=90, capacity=3)
    assert matched["threshold"] == pytest.approx(0.5)


def test_threshold_sweep_is_deterministic_and_reports_both_intervals():
    scored = _overlapping_orders(80)
    first = sweep_thresholds(scored, capacities=(3, None), bootstrap_reps=100)
    second = sweep_thresholds(scored, capacities=(3, None), bootstrap_reps=100)

    pd.testing.assert_frame_equal(first, second)
    assert {"bootstrap_low", "bootstrap_high", "bonferroni_low",
            "bonferroni_high", "total_trial_count"} <= set(first.columns)


def test_explicit_calendar_days_control_the_reported_trade_frequency():
    scored = _minute_scores_every_minute(20)

    table = sweep_thresholds(
        scored, capacities=(None,), quantiles=(0.0,), bootstrap_reps=10,
        evaluation_days=10,
    )

    assert table.iloc[0]["filled_trades"] == 4
    assert table.iloc[0]["trades_per_day"] == pytest.approx(0.4)


def test_hard_macro_sensitivity_masks_scores_without_refitting():
    scored = _minute_scores_every_minute(10)
    scored.loc[scored.index[5:], "macro_alignment"] = 0

    replay = hard_macro_sensitivity(scored, threshold=0.5, capacity=None)

    assert replay.orders["macro_alignment"].eq(1).all()
    pd.testing.assert_series_equal(scored["score"], replay.input_scores)
