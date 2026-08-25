"""Fixed-positive-EV policy rules for Notebook H."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def _module():
    try:
        return import_module("evaluation.fast_t2_causal_ev_policy")
    except ModuleNotFoundError:
        pytest.fail("causal EV policy module is not implemented")


def test_expected_value_subtracts_planned_cost_exactly_once():
    module = _module()
    probabilities = np.array([[0.5, 0.3, 0.2]])  # SL, TP, timeout
    score = module.expected_net_r(
        probabilities,
        planned_tp_gross_r=np.array([2.0]),
        timeout_gross_r=np.array([0.25]),
        planned_cost_r=np.array([0.1]),
    )

    assert np.isclose(score[0], 0.3 * 2.0 - 0.5 + 0.2 * 0.25 - 0.1)


def _scores() -> pd.DataFrame:
    rows = []
    for window, values in {
        "a": [(-0.1, 30.0), (0.0, 45.0), (0.2, 50.0)],
        "b": [(-0.2, 50.0), (-0.1, 60.0), (-0.05, 70.0)],
    }.items():
        for delay, (score, risk) in enumerate(values):
            timestamp = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(
                minutes=delay + (10 if window == "b" else 0)
            )
            rows.append(
                {
                    "window_id": window,
                    "decision_id": f"{window}:{delay}",
                    "decision_time": timestamp,
                    "entry_time": timestamp,
                    "active_end_time": timestamp + pd.Timedelta(minutes=30),
                    "side": "long",
                    "channel_episode_id": 1 if window == "a" else 2,
                    "ev_score": score,
                    "distance_to_stop_bps": risk,
                    "target_reached_before_decision": False,
                    "r_net": 0.5,
                    "filled": True,
                    "holding_minutes": 30.0,
                }
            )
    return pd.DataFrame(rows)


def test_policy_enters_only_strictly_positive_ev_without_forcing_frequency():
    module = _module()
    entries, actions = module.first_positive_ev_entries(
        _scores(), min_risk_bps=25.0, cancel_after_target=False
    )

    assert entries["decision_id"].tolist() == ["a:2"]
    assert entries["ev_score"].gt(0.0).all()
    assert not set(actions.loc[actions["action"].eq("ENTER"), "window_id"]).intersection({"b"})


def test_40bps_sensitivity_reuses_scores_and_never_adds_entries():
    module = _module()
    base, _ = module.first_positive_ev_entries(
        _scores(), min_risk_bps=25.0, cancel_after_target=False
    )
    strict, _ = module.first_positive_ev_entries(
        _scores(), min_risk_bps=40.0, cancel_after_target=False
    )

    assert set(strict["decision_id"]) <= set(base["decision_id"])
