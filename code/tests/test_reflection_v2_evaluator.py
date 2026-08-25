from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd

from reflection_agent.v2.contracts import AllowRule
from reflection_agent.v2.evaluator import ShadowCandidate, evaluate_shadow
from reflection_agent.v2.policy import ActiveAllowRule


SOURCE_CUTOFF = datetime(2023, 12, 31, 23, 45, tzinfo=UTC)
ELIGIBLE_AFTER = datetime(2024, 1, 1, tzinfo=UTC)


def rule(*, side: str, vol: str) -> AllowRule:
    return AllowRule.model_validate(
        {
            "action": "ALLOW_REENTRY",
            "predicates": [
                {"field": "side", "operator": "EQ", "value": side},
                {"field": "vol_regime", "operator": "EQ", "value": vol},
            ],
        }
    )


def shadow_opportunities(reentries: int = 12) -> pd.DataFrame:
    rows = []
    for index in range(reentries):
        decision = datetime(2024, 1, 2, tzinfo=UTC) + timedelta(days=5 * index)
        base_net = 0.001 if index % 3 else -0.0002
        rows.append(
            {
                "opportunity_id": f"base-{index}",
                "stage": "development",
                "fold_id": 0,
                "route": "UNION_BASE",
                "side": "LONG" if index % 2 else "SHORT",
                "vol_regime": "NORMAL",
                "decision_time": decision,
                "entry_time": decision + timedelta(minutes=15),
                "outcome_available_time": decision + timedelta(minutes=29),
                "gross_return": base_net + 0.001,
                "net_return": base_net,
                "round_trip_cost": 0.001,
            }
        )
        is_short = index % 2 == 0
        reentry_net = 0.004 if is_short else -0.003
        rows.append(
            {
                "opportunity_id": f"reentry-{index}",
                "stage": "development",
                "fold_id": 0,
                "route": "REENTRY",
                "side": "SHORT" if is_short else "LONG",
                "vol_regime": "HIGH" if is_short else "NORMAL",
                "decision_time": decision + timedelta(minutes=30),
                "entry_time": decision + timedelta(minutes=45),
                "outcome_available_time": decision + timedelta(minutes=59),
                "gross_return": reentry_net + 0.001,
                "net_return": reentry_net,
                "round_trip_cost": 0.001,
            }
        )
    return pd.DataFrame(rows)


def add_candidate(add_rule: AllowRule) -> ShadowCandidate:
    return ShadowCandidate(
        candidate_id="candidate-add",
        decision="ADD_ALLOW_RULE",
        source_episode_id="episode-1",
        source_fold_id=0,
        source_episode_cutoff_utc=SOURCE_CUTOFF,
        eligible_after_utc=ELIGIBLE_AFTER,
        proposed_rule=add_rule,
        target_rule_id=None,
    )


def test_positive_short_rule_is_promoted_only_by_future_shadow() -> None:
    result = evaluate_shadow(
        add_candidate(rule(side="SHORT", vol="HIGH")),
        shadow_opportunities(),
        active_rules=[],
    )
    assert result.decision == "PROMOTE"
    assert result.delta_net_return > 0.0
    assert result.triggered_trades >= 3
    assert result.shadow_reentry_opportunities >= 10
    assert result.shadow_long_opportunities >= 3
    assert result.shadow_short_opportunities >= 3
    assert all(result.gate_results.values())
    assert min(result.evaluated_decision_times) >= ELIGIBLE_AFTER


def test_cost_losing_long_rule_is_rejected() -> None:
    result = evaluate_shadow(
        add_candidate(rule(side="LONG", vol="NORMAL")),
        shadow_opportunities(),
        active_rules=[],
    )
    assert result.decision == "REJECT"
    assert result.delta_net_return < 0.0
    assert not result.gate_results["incremental_net_positive"]


def test_insufficient_future_pool_is_inconclusive() -> None:
    result = evaluate_shadow(
        add_candidate(rule(side="SHORT", vol="HIGH")),
        shadow_opportunities(reentries=9),
        active_rules=[],
    )
    assert result.decision == "INCONCLUSIVE"
    assert result.shadow_reentry_opportunities == 9
    assert not result.gate_results["shadow_minimums"]


def test_empty_future_pool_serializes_only_finite_metrics() -> None:
    empty = shadow_opportunities(1).iloc[0:0]
    result = evaluate_shadow(
        add_candidate(rule(side="SHORT", vol="HIGH")),
        empty,
        active_rules=[],
    )

    assert result.decision == "INCONCLUSIVE"
    for metrics in (result.union_metrics, result.control_metrics, result.candidate_metrics):
        assert all(
            math.isfinite(value)
            for value in (
                metrics.gross_return,
                metrics.cost_return,
                metrics.net_return,
                metrics.sortino,
                metrics.sharpe,
                metrics.max_drawdown,
            )
        )


def test_removing_a_losing_active_rule_can_be_promoted() -> None:
    active = ActiveAllowRule(
        rule_id="rule-losing-long",
        source_candidate_id="candidate-old",
        source_fold_id=0,
        activates_at_utc=ELIGIBLE_AFTER - timedelta(days=1),
        deactivates_at_utc=None,
        rule=rule(side="LONG", vol="NORMAL"),
    )
    candidate = ShadowCandidate(
        candidate_id="candidate-remove",
        decision="REMOVE_ALLOW_RULE",
        source_episode_id="episode-1",
        source_fold_id=0,
        source_episode_cutoff_utc=SOURCE_CUTOFF,
        eligible_after_utc=ELIGIBLE_AFTER,
        proposed_rule=None,
        target_rule_id="rule-losing-long",
    )
    result = evaluate_shadow(candidate, shadow_opportunities(), active_rules=[active])
    assert result.decision == "PROMOTE"
    assert result.delta_net_return > 0.0
    assert result.candidate_metrics.trades < result.control_metrics.trades


def test_cross_fold_and_preeligible_rows_are_never_evaluated() -> None:
    frame = shadow_opportunities()
    future_fold = frame.iloc[[0]].copy()
    future_fold["opportunity_id"] = "wrong-fold"
    future_fold["fold_id"] = 1
    preeligible = frame.iloc[[1]].copy()
    preeligible["opportunity_id"] = "preeligible"
    preeligible["decision_time"] = SOURCE_CUTOFF
    preeligible["entry_time"] = SOURCE_CUTOFF + timedelta(minutes=15)
    preeligible["outcome_available_time"] = SOURCE_CUTOFF + timedelta(minutes=29)
    frame = pd.concat([future_fold, preeligible, frame], ignore_index=True)
    result = evaluate_shadow(
        add_candidate(rule(side="SHORT", vol="HIGH")), frame, active_rules=[]
    )
    assert "wrong-fold" not in result.evaluated_opportunity_ids
    assert "preeligible" not in result.evaluated_opportunity_ids
