from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.event_window_cost_aware_policy import (
    CostAwarePolicyConfig,
    replay_cost_aware_policy,
)


START = pd.Timestamp("2022-01-01", tz="UTC")


def _labels(
    *,
    outcomes: tuple[str, ...] = ("tp", "tp", "tp"),
    observed: tuple[bool, ...] | None = None,
    geometry_valid: tuple[bool, ...] | None = None,
    risk_bps: float = 10.0,
    r_gross: float = 2.0,
) -> pd.DataFrame:
    count = len(outcomes)
    observed = observed or tuple(outcome != "censored" for outcome in outcomes)
    geometry_valid = geometry_valid or (True,) * count
    return pd.DataFrame(
        {
            "window_id": ["w1"] * count,
            "channel_episode_id": ["e1"] * count,
            "side": ["long"] * count,
            "step": np.arange(count),
            "decision_time": pd.date_range(START, periods=count, freq="5min"),
            "entry_time": pd.date_range(START, periods=count, freq="5min"),
            "geometry_valid": geometry_valid,
            "path_observed": observed,
            "outcome": outcomes,
            "r_gross": [r_gross if value else np.nan for value in observed],
            "risk_bps": risk_bps,
        }
    )


def _scores(ev: list[float], advantage: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1"] * len(ev),
            "step": np.arange(len(ev)),
            "conservative_net_ev": ev,
            "enter_advantage_vs_wait": advantage,
        }
    )


def test_waits_then_enters_at_first_joint_crossing_and_stops_window():
    replay = replay_cost_aware_policy(
        _scores([-0.1, 0.2, 0.4], [0.5, 0.0, 0.9]),
        _labels(),
        CostAwarePolicyConfig(threshold=0.0),
    )
    assert replay.actions[["step", "action"]].to_records(index=False).tolist() == [
        (0, "WAIT"),
        (1, "ENTER"),
    ]
    assert replay.trades.step.tolist() == [1]


def test_positive_ev_with_negative_enter_advantage_waits_then_skips():
    replay = replay_cost_aware_policy(
        _scores([0.2, 0.3], [-0.01, -0.02]),
        _labels(outcomes=("tp", "tp")),
    )
    assert replay.actions.action.tolist() == ["WAIT", "SKIP"]
    assert replay.attempted_trades == 0


def test_last_geometry_valid_decision_skips_even_if_later_row_is_invalid():
    replay = replay_cost_aware_policy(
        _scores([-0.1, -0.2, 1.0], [0.1, 0.1, 1.0]),
        _labels(geometry_valid=(True, True, False)),
    )
    assert replay.actions[["step", "action"]].to_records(index=False).tolist() == [
        (0, "WAIT"),
        (1, "SKIP"),
    ]
    assert replay.trades.empty


def test_maximum_one_trade_per_window_is_hard():
    replay = replay_cost_aware_policy(
        _scores([0.1, 0.2, 0.3], [0.0, 0.1, 0.2]),
        _labels(),
    )
    assert replay.trades.groupby("window_id").size().max() == 1
    assert replay.trades.step.tolist() == [0]


def test_censored_enter_is_preserved_and_consumes_window():
    replay = replay_cost_aware_policy(
        _scores([0.1, 0.5], [0.0, 0.5]),
        _labels(outcomes=("censored", "tp"), observed=(False, True)),
    )
    assert replay.attempted_trades == 1
    assert replay.observed_trades == 0
    assert replay.filled_trades == 0
    assert replay.trades[["step", "outcome"]].iloc[0].tolist() == [0, "censored"]
    assert pd.isna(replay.trades.iloc[0].realized_net_r)
    assert pd.isna(replay.trades.iloc[0].round_trip_cost_bps)


def test_unfilled_maker_order_consumes_window_without_fee_or_trade():
    replay = replay_cost_aware_policy(
        _scores([0.1, 0.5], [0.0, 0.5]),
        _labels(outcomes=("unfilled", "tp"), observed=(True, True)),
    )
    assert replay.attempted_trades == 1
    assert replay.observed_trades == 1
    assert replay.filled_trades == 0
    assert replay.trades.iloc[0].round_trip_cost_bps == 0.0
    assert replay.trades.iloc[0].realized_net_r == 0.0


@pytest.mark.parametrize(
    ("outcome", "exit_bps", "expected_net_r"),
    [("tp", 2.0, 1.7), ("sl", 4.0, 1.5), ("timeout", 6.0, 1.3)],
)
def test_realized_net_r_uses_outcome_specific_round_trip_fee(
    outcome: str, exit_bps: float, expected_net_r: float
):
    config = CostAwarePolicyConfig(
        maker_entry_bps=1.0,
        maker_tp_exit_bps=2.0,
        taker_sl_exit_bps=4.0,
        timeout_exit_bps=6.0,
    )
    replay = replay_cost_aware_policy(
        _scores([0.1], [0.0]),
        _labels(outcomes=(outcome,), risk_bps=10.0, r_gross=2.0),
        config,
    )
    trade = replay.trades.iloc[0]
    assert trade.round_trip_cost_bps == 1.0 + exit_bps
    assert trade.realized_net_r == pytest.approx(expected_net_r)


def test_fixed_threshold_never_forces_a_trade_for_frequency():
    replay = replay_cost_aware_policy(
        _scores([0.19, 0.19, 0.19], [1.0, 1.0, 1.0]),
        _labels(),
        CostAwarePolicyConfig(threshold=0.2),
    )
    assert replay.attempted_trades == 0
    assert replay.actions.action.tolist() == ["WAIT", "WAIT", "SKIP"]


def test_policy_orders_decisions_causally_not_by_input_row_order():
    labels = _labels().iloc[[2, 0, 1]].reset_index(drop=True)
    scores = _scores([-0.1, 0.1, 0.2], [0.0, 0.0, 0.0]).iloc[[2, 0, 1]]
    replay = replay_cost_aware_policy(scores, labels)
    assert replay.actions[["step", "action"]].to_records(index=False).tolist() == [
        (0, "WAIT"),
        (1, "ENTER"),
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"threshold": -0.1}, "threshold"),
        ({"maker_entry_bps": -1.0}, "maker_entry_bps"),
        ({"timeout_exit_bps": np.nan}, "timeout_exit_bps"),
    ],
)
def test_config_rejects_negative_ev_forcing_or_invalid_fees(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CostAwarePolicyConfig(**kwargs)
