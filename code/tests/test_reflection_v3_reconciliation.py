from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.reconcile_reflection_agent_v3 import (
    paired_policy_delta,
    reconcile_final_experiment,
)


CODE_ROOT = Path(__file__).parents[1]
FINAL_ROOT = CODE_ROOT / "experiments" / "cache" / "reflection_agent_v3"


def paired_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, net in enumerate((0.01, -0.005, 0.02, -0.002)):
        rows.append(
            {
                "opportunity_id": f"opportunity-{index}",
                "route": "UNION_BASE" if index < 2 else "COVERAGE_CANDIDATE",
                "side": "LONG" if index % 2 == 0 else "SHORT",
                "entry_time": pd.Timestamp("2025-01-01", tz="UTC")
                + pd.DateOffset(months=index),
                "net_return": net,
                "selected": index < 2,
            }
        )
    union = pd.DataFrame(rows)
    variant = union.copy()
    variant.loc[variant["route"].eq("COVERAGE_CANDIDATE"), "selected"] = True
    return variant, union


def test_paired_policy_delta_is_aligned_and_deterministic() -> None:
    variant, union = paired_frames()

    first = paired_policy_delta(variant, union, bootstrap_samples=1000, seed=7)
    second = paired_policy_delta(variant, union, bootstrap_samples=1000, seed=7)

    assert first == second
    assert first["selected_trade_delta"] == 2
    assert first["selected_long_trade_delta"] == 1
    assert first["selected_short_trade_delta"] == 1
    assert first["net_return_delta"] == pytest.approx(0.018)
    assert first["ci95_low"] <= first["net_return_delta"] <= first["ci95_high"]

    drifted = union.copy()
    drifted.loc[0, "net_return"] += 0.1
    with pytest.raises(ValueError, match="opportunity ledger drift"):
        paired_policy_delta(variant, drifted, bootstrap_samples=100, seed=7)


def test_final_experiment_reconciles_registered_continuous_protocol() -> None:
    report = reconcile_final_experiment(FINAL_ROOT)

    json.dumps(report, allow_nan=False)
    assert report["status"] == "complete"
    assert report["artifact_hashes_verified"] is True
    assert report["all_prompt_audits_passed"] is True
    assert report["lockbox_2026_q2_used"] is False
    assert report["forward_evidence_role"] == "secondary_reused_forward"
    assert isinstance(report["coverage_success"], bool)
    assert isinstance(report["memory_benefit_established"], bool)
    assert report["stage_counts"] == {
        "development": {
            "opportunities": 3456,
            "union_base": 948,
            "coverage_candidates": 2508,
        },
        "h1": {
            "opportunities": 1024,
            "union_base": 88,
            "coverage_candidates": 936,
        },
        "forward": {
            "opportunities": 1002,
            "union_base": 74,
            "coverage_candidates": 928,
        },
    }
    assert report["tier_counts"]["development"] == {
        "HIGH_EXTRA": 524,
        "LOW_EXTRA": 1150,
        "MID_EXTRA": 834,
    }
    assert set(report["results"]["development"]) == {
        "reflection_real_memory",
        "reflection_no_memory",
        "reflection_shuffled_memory",
        "static_high_extra",
        "static_all_extra",
        "union_baseline",
    }
    assert report["controls_called_llm"] is False
    assert report["union_invariant_across_variants"] is True
    assert report["continuous_stage_state_verified"] is True
    assert report["transport_failure_gate_passed"] is True
    assert report["results"]["forward"]["union_baseline"]["selected_trades"] == 74
    assert report["comparisons"]["forward"]["static_all_extra"][
        "selected_trade_delta"
    ] == 928
    assert report["comparisons"]["forward"]["static_all_extra"][
        "net_return_delta"
    ] < 0.0
