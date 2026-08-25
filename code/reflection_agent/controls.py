"""Frozen non-LLM controls and candidate policy materialization."""
from __future__ import annotations

from reflection_agent.contracts import (
    Candidate,
    ConditionPredicate,
    ConditionTree,
    PolicyEdit,
    PolicyRule,
)
from reflection_agent.execution import CONSENSUS_RULE
from reflection_agent.policy import resolve_edits
from reflection_agent.search import candidate_rule


DETERMINISTIC_ROUTER_RULE = PolicyRule(
    rule_id="deterministic-high-disagreement-router",
    conditions=ConditionTree(all=[
        ConditionPredicate(field="model_disagreement", operator="gte", value=0.75),
    ]),
    edits=[
        PolicyEdit(edit_id="router-lstm", action="select_frozen_expert", target="lstm"),
        PolicyEdit(edit_id="router-agreement", action="require_minimum_agreement", value=0.55),
    ],
)


def deterministic_router_rules() -> tuple[PolicyRule, ...]:
    """Route high-disagreement rows to the frozen LSTM within the two-edit budget."""
    return CONSENSUS_RULE, DETERMINISTIC_ROUTER_RULE


def materialize_candidate_rules(candidate: Candidate) -> tuple[PolicyRule, ...]:
    """Apply a candidate to the frozen consensus state without ambiguous meta-edits."""
    rule = candidate_rule(candidate)
    has_meta = any(edit.action in {"remove_active_edit", "reduce_active_edit"} for edit in candidate.edits)
    if not has_meta:
        return CONSENSUS_RULE, rule
    resolved = resolve_edits(CONSENSUS_RULE.edits, candidate.edits)
    if not resolved:
        return ()
    return (PolicyRule(rule_id=candidate.candidate_id, edits=list(resolved)),)
