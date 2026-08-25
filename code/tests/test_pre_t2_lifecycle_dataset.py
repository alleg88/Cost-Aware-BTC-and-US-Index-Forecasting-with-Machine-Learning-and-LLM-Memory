"""Causal T1 lifecycle rows for Notebook I."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from experiments.five_minute_two_trigger_windows import TwoTriggerConfig


def _module():
    try:
        return import_module("experiments.pre_t2_lifecycle_dataset")
    except ModuleNotFoundError:
        pytest.fail("pre-T2 lifecycle dataset module is not implemented")


def _channel() -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": times - pd.Timedelta(minutes=5),
            "decision_time": times,
            "open": 105.0,
            "high": 106.0,
            "low": 104.0,
            "close": 105.0,
            "minute_count": 5,
            "channel_lower": 100.0,
            "channel_upper": 110.0,
            "channel_r2": 0.8,
            "channel_regime": "up",
            "channel_confluence": 1,
            "channel_confluence_count": 2,
            "channel_episode_id": 7,
        }
    )
    for timestamp in (times[1], times[5]):
        index = frame.index[frame["decision_time"].eq(timestamp)][0]
        frame.loc[index, ["open", "high", "low", "close"]] = [
            101.0,
            102.0,
            99.5,
            101.5,
        ]
    return frame


def _minutes() -> pd.DataFrame:
    index = pd.date_range(
        "2023-12-31 23:00", periods=240, freq="1min", tz="UTC"
    )
    frame = pd.DataFrame(
        {
            "open": 101.5,
            "high": 101.8,
            "low": 101.2,
            "close": 101.5,
            "volume": 10.0,
            "taker_buy_base": 5.0,
            "count": 20.0,
        },
        index=index,
    )
    frame.loc[pd.Timestamp("2024-01-01 00:06", tz="UTC"), ["high", "close"]] = [
        102.2,
        102.1,
    ]
    return frame


def test_detector_retains_confirmed_and_expired_t1_episodes():
    module = _module()
    lifecycles, audit = module.detect_t1_lifecycles(
        _channel(),
        _minutes(),
        config=TwoTriggerConfig(cooldown_minutes=0),
        confirmation_buffer_bps=2.0,
    )

    assert lifecycles["status"].tolist() == ["confirmed", "expired"]
    assert lifecycles.loc[0, "t2_time"] == pd.Timestamp(
        "2024-01-01 00:07", tz="UTC"
    )
    assert pd.isna(lifecycles.loc[1, "t2_time"])
    assert audit["armed_t1"] == 2
    assert audit["confirmed_t2"] == 1
    assert audit["expired_t1"] == 1


def test_dataset_includes_honest_pre_and_post_states_with_34_features():
    module = _module()
    lifecycles, _ = module.detect_t1_lifecycles(
        _channel(),
        _minutes(),
        config=TwoTriggerConfig(cooldown_minutes=0),
        confirmation_buffer_bps=2.0,
    )
    decisions, sequences, audit = module.build_lifecycle_decisions(
        lifecycles, _channel(), _minutes()
    )

    confirmed = decisions[decisions["lifecycle_status"].eq("confirmed")]
    expired = decisions[decisions["lifecycle_status"].eq("expired")]
    assert set(confirmed["decision_phase"]) == {"pre_t2", "post_t2"}
    assert set(expired["decision_phase"]) == {"pre_t2"}
    assert not confirmed.loc[
        confirmed["decision_phase"].eq("pre_t2"), "t2_confirmed"
    ].any()
    assert expired["arm_id"].nunique() == 1
    assert sequences.shape == (len(decisions), 30, 5)
    assert len(module.BASE_FEATURE_COLUMNS) == 34
    assert tuple(module.SEQUENCE_FEATURE_COLUMNS) == (
        "return_1m_side",
        "range_bps",
        "close_location_in_range_side",
        "taker_imbalance_side",
        "log_volume",
    )
    assert decisions["decision_id"].is_unique
    assert audit["labelled_rows"] == len(decisions)


def test_future_minute_cannot_change_earlier_features():
    module = _module()
    minute = _minutes()
    lifecycles, _ = module.detect_t1_lifecycles(
        _channel(),
        minute,
        config=TwoTriggerConfig(cooldown_minutes=0),
        confirmation_buffer_bps=2.0,
    )
    before, _, _ = module.build_lifecycle_decisions(
        lifecycles, _channel(), minute
    )
    changed = minute.copy()
    changed.loc[pd.Timestamp("2024-01-01 01:50", tz="UTC"), ["high", "low", "close"]] = [
        150.0,
        50.0,
        120.0,
    ]
    after, _, _ = module.build_lifecycle_decisions(
        lifecycles, _channel(), changed
    )
    cutoff = pd.Timestamp("2024-01-01 00:20", tz="UTC")
    left = before[before["decision_time"].lt(cutoff)].set_index("decision_id")
    right = after[after["decision_time"].lt(cutoff)].set_index("decision_id")
    pd.testing.assert_frame_equal(
        left[list(module.BASE_FEATURE_COLUMNS)],
        right[list(module.BASE_FEATURE_COLUMNS)],
    )


def test_completed_stop_touch_permanently_closes_later_decisions():
    module = _module()
    minute = _minutes()
    lifecycles, _ = module.detect_t1_lifecycles(
        _channel(),
        minute,
        config=TwoTriggerConfig(cooldown_minutes=0),
        confirmation_buffer_bps=2.0,
    )
    minute.loc[pd.Timestamp("2024-01-01 00:07", tz="UTC"), "low"] = 90.0
    decisions, _, _ = module.build_lifecycle_decisions(
        lifecycles.iloc[:1], _channel(), minute
    )

    assert decisions["decision_time"].max() < pd.Timestamp(
        "2024-01-01 00:08", tz="UTC"
    )
    assert decisions["distance_to_stop_bps"].ge(25.0).all()


def test_outcome_fields_are_not_model_features():
    module = _module()
    forbidden = {
        "outcome",
        "r_net",
        "label_end",
        "exit_time",
        "t2_time",
        "lifecycle_status",
    }

    assert forbidden.isdisjoint(module.BASE_FEATURE_COLUMNS)
    assert "entry_delay_fraction" not in module.BASE_FEATURE_COLUMNS

