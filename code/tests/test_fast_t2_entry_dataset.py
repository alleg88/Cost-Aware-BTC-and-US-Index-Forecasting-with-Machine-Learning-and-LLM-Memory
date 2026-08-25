"""Causal decision-row tests for Fast-T2 entry timing."""

import numpy as np
import pandas as pd

from experiments.fast_t2_entry_dataset import (
    ENTRY_FEATURE_COLUMNS,
    SEQUENCE_FEATURE_COLUMNS,
    EntryDecisionConfig,
    build_entry_decisions,
    build_entry_sequences,
)


T2 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def _minute_fixture() -> pd.DataFrame:
    index = pd.date_range("2023-12-31 23:00", periods=240, freq="1min", tz="UTC")
    close = 100.0 + np.linspace(0.0, 0.24, len(index))
    return pd.DataFrame(
        {
            "open": close - 0.01,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": 10.0 + np.linspace(0.0, 1.0, len(index)),
            "taker_buy_base": 5.2,
            "count": 100 + np.arange(len(index)) % 7,
        },
        index=index,
    )


def _one_fast_event() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["fast:long:one"],
            "side": ["long"],
            "channel_episode_id": [7],
            "decision_time": [T2],
            "stop_price": [99.0],
            "target_price": [104.0],
            "side_sign": [1.0],
            "channel_slope_bps_5m_side": [2.0],
            "channel_r2": [0.7],
            "channel_width_pct": [0.04],
            "channel_confluence_count": [2.0],
            "t1_edge_depth": [0.1],
            "t1_range_bps": [20.0],
            "t1_body_fraction": [0.3],
            "t1_wick_share_side": [0.5],
            "t1_close_location_side": [0.7],
            "confirmation_lag_minutes": [4.0],
            "breakout_margin_bps": [2.5],
        }
    )


def test_entry_window_has_fifteen_causal_decisions_and_frozen_levels():
    decisions, audit = build_entry_decisions(
        _one_fast_event(), _minute_fixture(), EntryDecisionConfig()
    )

    assert len(decisions) == 15
    assert decisions["minutes_since_t2"].tolist() == list(range(15))
    assert decisions["window_id"].nunique() == 1
    assert decisions["stop_price"].nunique() == 1
    assert decisions["target_price"].nunique() == 1
    assert (decisions["entry_time"] == decisions["decision_time"]).all()
    assert audit == {
        "raw_windows": 1,
        "rows": 15,
        "censored": 0,
        "cancelled": 0,
    }


def test_future_bar_change_does_not_change_earlier_decision_features():
    before, _ = build_entry_decisions(_one_fast_event(), _minute_fixture())
    changed = _minute_fixture()
    changed.loc[T2 + pd.Timedelta(minutes=8), ["high", "low", "close"]] *= 3.0
    after, _ = build_entry_decisions(_one_fast_event(), changed)

    pd.testing.assert_frame_equal(
        before.loc[:7, list(ENTRY_FEATURE_COLUMNS)],
        after.loc[:7, list(ENTRY_FEATURE_COLUMNS)],
    )


def test_sequence_uses_thirty_completed_minutes_before_decision():
    decisions, _ = build_entry_decisions(_one_fast_event(), _minute_fixture())
    sequence = build_entry_sequences(decisions.iloc[[0]], _minute_fixture())

    assert sequence.shape == (1, 30, len(SEQUENCE_FEATURE_COLUMNS))


def test_entry_features_exclude_outcome_fields():
    forbidden = {
        "r_net",
        "label_net_positive",
        "outcome",
        "exit_price",
        "label_end",
        "entry_price",
    }

    assert set(ENTRY_FEATURE_COLUMNS).isdisjoint(forbidden)
