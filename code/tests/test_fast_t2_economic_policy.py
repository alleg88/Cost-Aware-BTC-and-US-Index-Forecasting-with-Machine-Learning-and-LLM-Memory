"""Economic first-crossing, frequency, RR, and loss-streak behavior."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def _module():
    try:
        return import_module("evaluation.fast_t2_economic_policy")
    except ModuleNotFoundError:
        pytest.fail("economic policy module is not implemented")


def _window(window_id, side, scores, rr, rewards, start, episode):
    times = pd.date_range(start, periods=len(scores), freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "window_id": window_id,
            "decision_id": [f"{window_id}:{i}" for i in range(len(scores))],
            "decision_time": times,
            "entry_time": times,
            "active_end_time": times + pd.Timedelta(minutes=10),
            "side": side,
            "channel_episode_id": episode,
            "score": scores,
            "rr_proxy": rr,
            "r_net": rewards,
            "filled": True,
            "holding_minutes": 10.0,
            "minutes_since_t2": np.arange(len(scores)),
        }
    )


def test_rr_gate_waits_for_the_first_eligible_economic_crossing():
    module = _module()
    scored = _window(
        "w1", "long", [0.9, 0.8, 0.7], [0.8, 1.2, 1.8], [-1.0, 0.5, 1.0],
        "2024-01-01", 1,
    )

    entries, actions = module.economic_first_crossing_entries(
        scored, threshold=0.75, min_rr=1.0
    )

    assert entries["decision_id"].tolist() == ["w1:1"]
    assert actions["action"].tolist() == ["WAIT", "ENTER"]
    assert actions["rr_eligible"].tolist() == [False, True]


def test_threshold_selection_respects_average_frequency_and_both_sides():
    module = _module()
    scored = pd.concat(
        [
            _window("a", "long", [0.1, 0.8], [2.0, 2.0], [-1.0, 1.0], "2024-01-01 00:00", 1),
            _window("b", "short", [0.2, 0.7], [2.0, 2.0], [-1.0, 0.8], "2024-01-01 01:00", 2),
            _window("c", "long", [0.3, 0.6], [2.0, 2.0], [-1.0, 0.5], "2024-01-02 00:00", 3),
            _window("d", "short", [0.4, 0.5], [2.0, 2.0], [-1.0, 0.4], "2024-01-02 01:00", 4),
        ],
        ignore_index=True,
    )

    quantile, table = module.select_economic_threshold(
        scored, evaluation_days=2, min_rr=None, min_trades_per_day=1.0
    )
    chosen = table.loc[table["quantile"].eq(quantile)].iloc[0]

    assert chosen["trades_per_day"] >= 1.0
    assert chosen["long_trades"] > 0
    assert chosen["short_trades"] > 0
    assert bool(chosen["frequency_eligible"])
    assert "robust_mean_net_r" in table


def test_loss_streak_summary_uses_completed_trade_order():
    module = _module()
    times = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    trades = pd.DataFrame(
        {
            "active_end_time": times,
            "decision_id": [f"d{i}" for i in range(8)],
            "r_net": [-1.0, -0.2, 0.5, -0.1, -1.2, -0.4, 0.3, -0.2],
        }
    )

    summary = module.summarise_loss_streaks(trades)

    assert summary["loss_trades"] == 6
    assert summary["loss_streak_count"] == 3
    assert summary["max_loss_streak"] == 3
    assert summary["mean_loss_streak"] == 2.0


def test_frequency_eligibility_is_computed_for_each_policy_row():
    module = _module()
    policies = pd.DataFrame(
        {
            "trades_per_day": [1.1, 0.3, 1.2],
            "long_trades": [10, 10, 0],
            "short_trades": [10, 10, 20],
        }
    )

    marked = module.mark_frequency_eligibility(policies, minimum=1.0)

    assert marked["frequency_eligible"].tolist() == [True, False, False]


def test_primary_arm_selection_uses_raw_economics_not_robust_training_metric():
    module = _module()
    policies = pd.DataFrame(
        {
            "arm": ["catboost", "ridge"],
            "frequency_eligible": [True, True],
            "mean_net_r": [-0.30, -0.10],
            "robust_mean_net_r": [-0.05, -0.20],
            "total_net_r": [-30.0, -10.0],
        }
    )

    winner = module.select_primary_economic_arm(policies)

    assert winner["arm"] == "ridge"
