"""Strict positive-EV lifecycle policy."""

from importlib import import_module

import pandas as pd
import pytest


def _module():
    try:
        return import_module("evaluation.pre_t2_entry_policy")
    except ModuleNotFoundError:
        pytest.fail("pre-T2 entry policy module is not implemented")


def _scores() -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "arm_id": ["a", "a", "a", "b", "b"],
            "decision_id": ["a:0", "a:1", "a:2", "b:0", "b:1"],
            "decision_time": [
                start,
                start + pd.Timedelta(minutes=1),
                start + pd.Timedelta(minutes=2),
                start + pd.Timedelta(minutes=10),
                start + pd.Timedelta(minutes=11),
            ],
            "decision_phase": ["pre_t2", "post_t2", "post_t2", "pre_t2", "post_t2"],
            "ev_score": [0.1, 0.3, 0.4, -0.1, 0.0],
            "distance_to_stop_bps": [30.0, 35.0, 40.0, 50.0, 55.0],
            "stop_invalidated_before_decision": False,
            "filled": True,
            "entry_time": [
                start,
                start + pd.Timedelta(minutes=1),
                start + pd.Timedelta(minutes=2),
                start + pd.Timedelta(minutes=10),
                start + pd.Timedelta(minutes=11),
            ],
            "active_end_time": [start + pd.Timedelta(minutes=30)] * 5,
            "side": ["long"] * 5,
            "channel_episode_id": [1, 1, 1, 2, 2],
            "r_net": [0.4, 0.5, 0.6, -0.2, -0.1],
            "holding_minutes": [30.0] * 5,
            "score": [0.1, 0.3, 0.4, -0.1, 0.0],
        }
    )


def test_all_lifecycle_enters_before_t2_when_pre_score_is_positive():
    module = _module()
    entries, _ = module.first_positive_lifecycle_entries(_scores(), phase="all")

    assert entries["decision_id"].tolist() == ["a:0"]
    assert entries["ev_score"].gt(0.0).all()


def test_post_only_reuses_scores_and_waits_for_post_t2_row():
    module = _module()
    entries, _ = module.first_positive_lifecycle_entries(
        _scores(), phase="post_t2"
    )

    assert entries["decision_id"].tolist() == ["a:1"]


def test_zero_or_negative_ev_never_enters_to_force_frequency():
    module = _module()
    entries, actions = module.first_positive_lifecycle_entries(
        _scores(), phase="pre_t2"
    )

    assert entries["arm_id"].tolist() == ["a"]
    assert not actions.loc[actions["arm_id"].eq("b"), "action"].eq("ENTER").any()

