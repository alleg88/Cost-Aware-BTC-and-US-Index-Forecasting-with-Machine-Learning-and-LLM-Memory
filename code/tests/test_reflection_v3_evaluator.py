from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from reflection_agent.v3.contracts import AllowRule, Predicate
from reflection_agent.v3.evaluator import ShadowCandidate, evaluate_shadow
from reflection_agent.v3.policy import ActiveAllowRule


UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _rule(side: str = "LONG", tier: str = "HIGH_EXTRA") -> AllowRule:
    return AllowRule(
        action="ALLOW_CANDIDATE",
        predicates=[
            Predicate(field="side", operator="EQ", value=side),
            Predicate(field="confidence_tier", operator="EQ", value=tier),
        ],
    )


def _candidate(
    *,
    decision: str = "ADD_ALLOW_RULE",
    proposed_rule: AllowRule | None = None,
    target_rule_id: str | None = None,
    stage: str = "development",
    fold_id: int = 0,
) -> ShadowCandidate:
    return ShadowCandidate(
        candidate_id="candidate_a1",
        decision=decision,
        source_episode_id="episode_a1",
        source_stage=stage,
        source_fold_id=fold_id,
        source_episode_cutoff_utc=BASE,
        eligible_after_utc=BASE + timedelta(minutes=1),
        proposed_rule=proposed_rule if decision == "ADD_ALLOW_RULE" else None,
        target_rule_id=target_rule_id if decision == "REMOVE_ALLOW_RULE" else None,
    )


def _opportunities(
    *,
    matching: int = 12,
    total: int = 12,
    candidate_net: float = 0.002,
    stage: str = "development",
    fold_id: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index in range(total):
        decision = BASE + timedelta(minutes=30 * (index + 1))
        is_match = index < matching
        side = "LONG" if is_match else "SHORT"
        tier = "HIGH_EXTRA" if is_match else "MID_EXTRA"
        net = candidate_net if is_match else 0.0
        rows.append(
            {
                "opportunity_id": f"candidate_{index:03d}",
                "stage": stage,
                "fold_id": fold_id,
                "route": "COVERAGE_CANDIDATE",
                "side": side,
                "decision_time": decision,
                "entry_time": decision + timedelta(minutes=15),
                "exit_time": decision + timedelta(minutes=15),
                "outcome_available_time": decision + timedelta(minutes=25),
                "confidence_tier": tier,
                "signal_run_bucket": "FIRST",
                "vol_regime": "NORMAL",
                "trend_regime": "UP" if side == "LONG" else "DOWN",
                "funding_regime": "NEUTRAL",
                "oi_regime": "RISING",
                "gross_return": net + 0.001,
                "net_return": net,
                "round_trip_cost": 0.001,
            }
        )
    for index in range(2):
        decision = (
            BASE + timedelta(minutes=5)
            if index == 0
            else BASE + timedelta(minutes=30 * (total + 1))
        )
        rows.append(
            {
                "opportunity_id": f"union_{index}",
                "stage": stage,
                "fold_id": fold_id,
                "route": "UNION_BASE",
                "side": "LONG" if index == 0 else "SHORT",
                "decision_time": decision,
                "entry_time": decision + timedelta(minutes=15),
                "exit_time": decision + timedelta(minutes=15),
                "outcome_available_time": decision + timedelta(minutes=20),
                "confidence_tier": pd.NA,
                "signal_run_bucket": pd.NA,
                "vol_regime": "NORMAL",
                "trend_regime": "FLAT",
                "funding_regime": "NEUTRAL",
                "oi_regime": "FLAT",
                "gross_return": 0.002,
                "net_return": 0.001,
                "round_trip_cost": 0.001,
            }
        )
    return pd.DataFrame(rows)


def test_twelve_matching_candidates_promote_a_noninferior_rule() -> None:
    result = evaluate_shadow(
        _candidate(proposed_rule=_rule()),
        _opportunities(),
        active_rules=[],
    )
    assert result.decision == "PROMOTE"
    assert result.matching_candidates == 12
    assert result.total_coverage_candidates == 12
    assert result.triggered_trades == 12
    assert result.gate_results["total_net_noninferior"]
    assert result.gate_results["two_subblock_support"]
    assert result.gate_results["exact_reconciliation"]


def test_forty_candidate_cap_and_fold_boundary_are_inconclusive() -> None:
    capped = evaluate_shadow(
        _candidate(proposed_rule=_rule()),
        _opportunities(matching=11, total=41),
        active_rules=[],
    )
    assert capped.decision == "INCONCLUSIVE"
    assert capped.total_coverage_candidates == 40
    assert capped.matching_candidates == 11

    current = _opportunities(matching=8, total=8)
    next_fold = _opportunities(matching=12, total=12, fold_id=1)
    next_fold["opportunity_id"] = "next_" + next_fold["opportunity_id"]
    bounded = evaluate_shadow(
        _candidate(proposed_rule=_rule()),
        pd.concat([current, next_fold], ignore_index=True),
        active_rules=[],
    )
    assert bounded.decision == "INCONCLUSIVE"
    assert bounded.matching_candidates == 8


def test_absolute_economic_noninferiority_rejects_bad_extra_trades() -> None:
    result = evaluate_shadow(
        _candidate(proposed_rule=_rule()),
        _opportunities(candidate_net=-0.01),
        active_rules=[],
    )
    assert result.decision == "REJECT"
    assert not result.gate_results["total_net_noninferior"]
    assert not result.gate_results["target_side_net_noninferior"]
    assert result.candidate_metrics.net_return < result.union_metrics.net_return - 0.005


def test_exact_cost_reconciliation_is_fail_closed() -> None:
    frame = _opportunities()
    frame.loc[0, "round_trip_cost"] = 0.0005
    with pytest.raises(ValueError, match="gross/net/cost"):
        evaluate_shadow(_candidate(proposed_rule=_rule()), frame, active_rules=[])


def test_unknown_removal_target_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown removal target"):
        evaluate_shadow(
            _candidate(decision="REMOVE_ALLOW_RULE", target_rule_id="missing"),
            _opportunities(),
            active_rules=[],
        )


def test_removal_requires_strict_improvement() -> None:
    active = ActiveAllowRule(
        rule_id="bad_rule",
        source_candidate_id="old_candidate",
        source_stage="development",
        source_fold_id=0,
        activates_at_utc=BASE,
        rule=_rule(),
    )
    result = evaluate_shadow(
        _candidate(decision="REMOVE_ALLOW_RULE", target_rule_id="bad_rule"),
        _opportunities(candidate_net=-0.01),
        active_rules=[active],
    )
    assert result.decision == "PROMOTE"
    assert result.gate_results["removal_strictly_improves"]
