"""Causal geometry and outcome labels for Notebook H."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def _module():
    try:
        return import_module("experiments.fast_t2_causal_ev_dataset")
    except ModuleNotFoundError:
        pytest.fail("causal EV dataset module is not implemented")


def _minutes() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=6, freq="1min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * 6,
            "high": [100.5, 100.5, 102.5, 100.5, 100.5, 100.5],
            "low": [98.5, 99.5, 99.5, 99.5, 99.5, 99.5],
            "close": [100.0] * 6,
        },
        index=index,
    )


def _decisions(side: str = "long") -> pd.DataFrame:
    times = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC")
    sign = 1.0 if side == "long" else -1.0
    stop = 99.0 if side == "long" else 101.0
    target = 102.0 if side == "long" else 98.0
    return pd.DataFrame(
        {
            "window_id": ["w"] * 3,
            "decision_id": [f"w:{i}" for i in range(3)],
            "side": [side] * 3,
            "t2_time": [times[0]] * 3,
            "decision_time": times,
            "entry_price": [100.0] * 3,
            "stop_price": [stop] * 3,
            "target_price": [target] * 3,
            "distance_to_stop_bps": [100.0] * 3,
            "distance_to_target_bps": [200.0] * 3,
            "rr_proxy": [2.0] * 3,
            "outcome": ["tp", "sl", "timeout"],
            "r_net": [1.9, -1.1, 0.2],
            "filled": [True] * 3,
            "side_sign": [sign] * 3,
        }
    )


def test_stop_touch_closes_window_before_all_later_decisions():
    module = _module()
    annotated = module.annotate_causal_ev_rows(_decisions(), _minutes())

    assert annotated["stop_invalidated_before_decision"].tolist() == [False, True, True]
    assert annotated["primary_eligible"].tolist() == [True, False, False]
    assert annotated["sensitivity_40_eligible"].tolist() == [True, False, False]


def test_future_bar_cannot_change_an_earlier_window_state():
    module = _module()
    clean = _minutes()
    clean.loc[clean.index[0], "low"] = 99.5
    changed = clean.copy()
    changed.loc[changed.index[2], "low"] = 90.0

    before = module.annotate_causal_ev_rows(_decisions(), clean)
    after = module.annotate_causal_ev_rows(_decisions(), changed)

    assert before.loc[:1, "stop_invalidated_before_decision"].equals(
        after.loc[:1, "stop_invalidated_before_decision"]
    )


def test_ev_labels_are_in_r_units_and_catastrophic_artifacts_are_not_eligible():
    module = _module()
    decisions = _decisions()
    decisions.loc[0, "distance_to_stop_bps"] = 20.0
    decisions.loc[0, "r_net"] = -4.0
    annotated = module.annotate_causal_ev_rows(decisions, _minutes())

    assert np.isclose(annotated.loc[1, "planned_cost_r_10bps"], 0.1)
    assert np.isclose(annotated.loc[1, "planned_tp_gross_r"], 2.0)
    assert annotated["outcome_class"].tolist() == [1, 0, 2]
    assert bool(annotated.loc[0, "catastrophic_loss"])
    assert not bool(annotated.loc[0, "primary_eligible"])


def test_target_touch_is_recorded_separately_from_primary_stop_cancellation():
    module = _module()
    minutes = _minutes()
    minutes.loc[minutes.index[0], "low"] = 99.5
    minutes.loc[minutes.index[1], "high"] = 102.5
    annotated = module.annotate_causal_ev_rows(_decisions(), minutes)

    assert annotated["target_reached_before_decision"].tolist() == [False, False, True]
    assert bool(annotated.loc[2, "primary_eligible"])
    assert not bool(annotated.loc[2, "target_cancel_eligible"])
