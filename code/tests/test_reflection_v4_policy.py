from __future__ import annotations

import pandas as pd
import pytest

from reflection_agent.v4.policies import (
    POLICY_IDS,
    apply_router_policy,
    policy_mask,
    validate_router_opportunities,
)


def _opportunities() -> pd.DataFrame:
    rows = [
        {
            "opportunity_id": "u1",
            "route": "UNION_BASE",
            "side": "LONG",
            "decision_time": "2024-01-01T00:00:00Z",
            "entry_time": "2024-01-01T00:15:00Z",
            "outcome_available_time": "2024-01-01T00:29:00Z",
            "confidence_tier": None,
            "signal_run_bucket": None,
            "vol_regime": "NORMAL",
            "trend_regime": "FLAT",
            "funding_regime": "NEUTRAL",
            "oi_regime": "FLAT",
            "net_return": 0.01,
            "gross_return": 0.011,
            "round_trip_cost": 0.001,
            "xgb_available": False,
            "xgb_p_move_raw": None,
            "xgb_direction_confidence": None,
            "xgb_side": None,
        },
        {
            "opportunity_id": "c-overlap",
            "route": "COVERAGE_CANDIDATE",
            "side": "LONG",
            "decision_time": "2024-01-01T00:00:00Z",
            "entry_time": "2024-01-01T00:15:00Z",
            "outcome_available_time": "2024-01-01T00:29:00Z",
            "confidence_tier": "HIGH_EXTRA",
            "signal_run_bucket": "FIRST",
            "vol_regime": "HIGH",
            "trend_regime": "FLAT",
            "funding_regime": "NEUTRAL",
            "oi_regime": "FLAT",
            "net_return": -0.001,
            "gross_return": 0.0,
            "round_trip_cost": 0.001,
            "xgb_available": True,
            "xgb_p_move_raw": 0.99,
            "xgb_direction_confidence": 0.99,
            "xgb_side": "LONG",
        },
        {
            "opportunity_id": "c-context",
            "route": "COVERAGE_CANDIDATE",
            "side": "LONG",
            "decision_time": "2024-01-01T01:00:00Z",
            "entry_time": "2024-01-01T01:15:00Z",
            "outcome_available_time": "2024-01-01T01:29:00Z",
            "confidence_tier": "LOW_EXTRA",
            "signal_run_bucket": "THIRD_PLUS",
            "vol_regime": "NORMAL",
            "trend_regime": "UP",
            "funding_regime": "POSITIVE",
            "oi_regime": "RISING",
            "net_return": 0.002,
            "gross_return": 0.003,
            "round_trip_cost": 0.001,
            "xgb_available": True,
            "xgb_p_move_raw": 0.85,
            "xgb_direction_confidence": 0.85,
            "xgb_side": "LONG",
        },
        {
            "opportunity_id": "c-xgb-wrong-side",
            "route": "COVERAGE_CANDIDATE",
            "side": "SHORT",
            "decision_time": "2024-01-01T02:00:00Z",
            "entry_time": "2024-01-01T02:15:00Z",
            "outcome_available_time": "2024-01-01T02:29:00Z",
            "confidence_tier": "MID_EXTRA",
            "signal_run_bucket": "SECOND",
            "vol_regime": "LOW",
            "trend_regime": "DOWN",
            "funding_regime": "NEGATIVE",
            "oi_regime": "FALLING",
            "net_return": 0.001,
            "gross_return": 0.002,
            "round_trip_cost": 0.001,
            "xgb_available": True,
            "xgb_p_move_raw": 0.90,
            "xgb_direction_confidence": 0.90,
            "xgb_side": "LONG",
        },
    ]
    return pd.DataFrame(rows)


def test_policy_menu_is_frozen_and_host_owned() -> None:
    assert POLICY_IDS == (
        "UNION_ONLY",
        "LSTM_HIGH",
        "LSTM_ALL",
        "FUNDING_CONTINUATION",
        "VOLATILITY_RESET",
        "CONTEXT_COMBINED",
        "FIRST_ONLY",
        "THIRD_PLUS_ONLY",
        "XGB_STRONG",
    )


def test_context_and_xgb_masks_are_exact() -> None:
    frame = _opportunities()
    assert frame.loc[policy_mask(frame, "FUNDING_CONTINUATION"), "opportunity_id"].tolist() == [
        "c-context"
    ]
    assert frame.loc[policy_mask(frame, "VOLATILITY_RESET"), "opportunity_id"].tolist() == [
        "c-overlap"
    ]
    assert frame.loc[policy_mask(frame, "XGB_STRONG"), "opportunity_id"].tolist() == [
        "c-overlap",
        "c-context",
    ]


def test_union_is_immutable_and_overlap_wins() -> None:
    selected = apply_router_policy(_opportunities(), "CONTEXT_COMBINED")
    union = selected.loc[selected["route"].eq("UNION_BASE")]
    assert union["selected"].all()
    assert selected.set_index("opportunity_id").at["c-overlap", "skip_reason"] == "OVERLAP_UNION"
    assert selected.set_index("opportunity_id").at["c-context", "selected"]


def test_week_boundary_candidate_is_vetoed_for_every_policy() -> None:
    frame = _opportunities()
    frame["block_boundary_eligible"] = True
    frame.loc[frame["opportunity_id"].eq("c-context"), "block_boundary_eligible"] = False
    selected = apply_router_policy(frame, "LSTM_ALL").set_index("opportunity_id")
    assert not selected.at["c-context", "selected"]
    assert selected.at["c-context", "skip_reason"] == "BLOCK_BOUNDARY"


def test_unknown_policy_and_q2_timestamp_fail_closed() -> None:
    with pytest.raises(ValueError, match="unknown router policy"):
        policy_mask(_opportunities(), "INVENTED")
    frame = _opportunities()
    frame.loc[0, "outcome_available_time"] = "2026-04-01T00:00:00Z"
    with pytest.raises(ValueError, match="Q2"):
        validate_router_opportunities(frame)
