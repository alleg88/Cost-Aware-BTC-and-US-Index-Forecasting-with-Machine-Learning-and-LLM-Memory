from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from experiments.reconcile_reflection_agent_v2 import (
    paired_policy_delta,
    reconcile_final_experiment,
)


CODE_ROOT = Path(__file__).parents[1]
FINAL_ROOT = CODE_ROOT / "experiments" / "cache" / "reflection_agent_final"


def paired_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for index, net in enumerate((0.01, -0.005, 0.02, -0.002)):
        rows.append(
            {
                "opportunity_id": f"opportunity-{index}",
                "route": "UNION_BASE" if index < 2 else "REENTRY",
                "side": "LONG" if index % 2 == 0 else "SHORT",
                "entry_time": pd.Timestamp("2025-01-01", tz="UTC")
                + pd.DateOffset(months=index),
                "net_return": net,
                "selected": index < 2,
            }
        )
    union = pd.DataFrame(rows)
    variant = union.copy()
    variant.loc[variant["route"].eq("REENTRY"), "selected"] = True
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


def test_final_experiment_reconciles_all_stages_variants_and_lockbox() -> None:
    report = reconcile_final_experiment(FINAL_ROOT)

    assert report["status"] == "complete"
    assert report["conclusion"] == "benefit_not_established"
    assert report["all_prompt_audits_passed"] is True
    assert report["lockbox_2026_q2_used"] is False
    assert report["stage_counts"] == {
        "development": {"opportunities": 1231, "union_base": 948, "reentries": 283},
        "h1": {"opportunities": 110, "union_base": 88, "reentries": 22},
        "forward": {"opportunities": 88, "union_base": 74, "reentries": 14},
    }
    assert report["development_funnel"]["reflection_real_memory"]["terminal"] == {
        "INCONCLUSIVE": 1,
        "REJECT": 1,
    }
    assert report["development_funnel"]["reflection_real_memory"]["promoted_rules"] == 0
    assert report["comparisons"]["forward"]["static_add_all"][
        "selected_trade_delta"
    ] == 14
    assert report["comparisons"]["forward"]["static_add_all"][
        "net_return_delta"
    ] < 0.0
