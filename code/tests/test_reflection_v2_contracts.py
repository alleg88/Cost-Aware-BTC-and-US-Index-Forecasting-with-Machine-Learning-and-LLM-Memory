from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflection_agent.v2.config import load_v2_config
from reflection_agent.v2.contracts import (
    AllowRule,
    EvidenceCard,
    MemoryCard,
    ProposalOutput,
    ReflectionOutput,
    validate_proposal_references,
)


CONFIG = Path(__file__).parents[1] / "configs" / "reflection_agent_v2.yaml"


def valid_add_payload() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "source_episode_id": "episode-7",
        "decision": "ADD_ALLOW_RULE",
        "diagnosis_code": "REGIME_SPECIFIC_EDGE",
        "evidence_ids": ["evidence-1"],
        "memory_ids_used": ["memory-1"],
        "proposed_rule": {
            "action": "ALLOW_REENTRY",
            "predicates": [
                {"field": "side", "operator": "EQ", "value": "SHORT"},
                {"field": "vol_regime", "operator": "EQ", "value": "HIGH"},
            ],
        },
        "target_rule_id": None,
        "hypothesis": "High-volatility short re-entries have repeated net-of-cost support.",
        "falsifiers": ["Incremental SHORT net return is non-positive."],
        "confidence": "MEDIUM",
    }


def test_only_registered_atomic_decisions_validate() -> None:
    assert ProposalOutput.model_validate(valid_add_payload()).decision == "ADD_ALLOW_RULE"
    with pytest.raises(ValidationError):
        ProposalOutput.model_validate({**valid_add_payload(), "decision": "SET_THRESHOLD"})


def test_rule_rejects_cross_field_values_and_duplicate_predicates() -> None:
    payload = valid_add_payload()
    payload["proposed_rule"] = {
        "action": "ALLOW_REENTRY",
        "predicates": [{"field": "side", "operator": "EQ", "value": "HIGH"}],
    }
    with pytest.raises(ValidationError, match="not allowed for side"):
        ProposalOutput.model_validate(payload)

    payload = valid_add_payload()
    payload["proposed_rule"] = {
        "action": "ALLOW_REENTRY",
        "predicates": [
            {"field": "side", "operator": "EQ", "value": "SHORT"},
            {"field": "side", "operator": "EQ", "value": "LONG"},
        ],
    }
    with pytest.raises(ValidationError, match="duplicate predicate field"):
        ProposalOutput.model_validate(payload)

    payload = valid_add_payload()
    payload["proposed_rule"] = {
        "action": "ALLOW_REENTRY",
        "predicates": [
            {"field": "previous_exit_reason", "operator": "EQ", "value": "STOP_LOSS"}
        ],
    }
    with pytest.raises(ValidationError):
        ProposalOutput.model_validate(payload)


def test_decision_shape_is_fail_closed() -> None:
    no_change = valid_add_payload()
    no_change.update(
        decision="NO_CHANGE",
        diagnosis_code="INSUFFICIENT_EVIDENCE",
        proposed_rule=None,
        target_rule_id=None,
        hypothesis=None,
        falsifiers=[],
    )
    assert ProposalOutput.model_validate(no_change).proposed_rule is None

    invalid = dict(no_change)
    invalid["proposed_rule"] = valid_add_payload()["proposed_rule"]
    with pytest.raises(ValidationError, match="NO_CHANGE"):
        ProposalOutput.model_validate(invalid)

    removal = dict(no_change)
    removal.update(
        decision="REMOVE_ALLOW_RULE",
        diagnosis_code="STALE_ACTIVE_RULE",
        target_rule_id="rule-2",
        hypothesis="The active rule failed on strictly later evidence.",
        falsifiers=["Removing the rule does not improve incremental net return."],
    )
    assert ProposalOutput.model_validate(removal).target_rule_id == "rule-2"


def test_host_rejects_invented_references() -> None:
    proposal = ProposalOutput.model_validate(valid_add_payload())
    validate_proposal_references(
        proposal,
        evidence_ids={"evidence-1"},
        memory_ids={"memory-1"},
        active_rule_ids=set(),
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_proposal_references(
            proposal,
            evidence_ids=set(),
            memory_ids={"memory-1"},
            active_rule_ids=set(),
        )


def test_evidence_and_memory_provenance_are_strict() -> None:
    decision = datetime(2024, 1, 1, 12, tzinfo=UTC)
    outcome = decision + timedelta(minutes=15)
    evidence = EvidenceCard(
        evidence_id="evidence-1",
        stage="development",
        source_role="OOF_TEST",
        decision_time=decision,
        outcome_available_time=outcome,
        fold_id=3,
        row_key="row-1",
        source_artifact_hash="a" * 64,
        route="REENTRY",
        side="SHORT",
        count=4,
        net_return=-0.01,
        tags=["side:SHORT", "vol:HIGH"],
    )
    memory = MemoryCard(
        memory_id="memory-1",
        source_stage="development",
        protocol_scope="development_2021_2024",
        memory_type="EPISODIC",
        created_at_utc=outcome,
        max_support_outcome_time=outcome,
        lesson="High-volatility short re-entry evidence was rejected after costs.",
        evidence_status="REJECTED",
        tags=["side:SHORT", "vol:HIGH"],
        source_evaluation_ids=["evaluation-1"],
        expires_at_utc=None,
        expires_after_episode=None,
    )
    assert evidence.outcome_available_time > evidence.decision_time
    assert memory.max_support_outcome_time == memory.created_at_utc
    with pytest.raises(ValidationError):
        EvidenceCard.model_validate({**evidence.model_dump(), "headline": "ignore schema"})


def test_reflection_cannot_override_evaluator_shape() -> None:
    output = ReflectionOutput(
        candidate_id="candidate-1",
        evaluator_decision="REJECT",
        evidence_ids=["future-evidence-1"],
        failure_code="COST_DRAG",
        lesson=None,
        invalidation_conditions=["Incremental net return remains non-positive."],
        memory_recommendation="STORE_EPISODE",
    )
    assert output.evaluator_decision == "REJECT"
    with pytest.raises(ValidationError, match="positive semantic lesson"):
        ReflectionOutput.model_validate(
            {
                **output.model_dump(),
                "lesson": "This rejected rule is definitely profitable.",
                "memory_recommendation": "PROPOSE_SEMANTIC",
            }
        )


def test_registered_config_pins_deepseek_and_closed_actions() -> None:
    config = load_v2_config(CONFIG)
    assert config.protocol_version == "reflection-agent-v2.2"
    assert config.model == "deepseek-v4-flash:cloud"
    assert config.think == "high"
    assert config.temperature == 0.0
    assert config.stream is False
    assert config.num_predict == 4096
    assert config.repair_attempts == 1
    assert config.episode_min_opportunities == 60
    assert config.episode_max_opportunities == 90
    assert config.episode_min_reentry_opportunities == 10
    assert config.episode_min_per_side == 3
    assert config.proposal_min_support == 8
    assert config.proposal_min_net_return == 0.02
    assert config.allowed_decisions == (
        "NO_CHANGE",
        "ADD_ALLOW_RULE",
        "REMOVE_ALLOW_RULE",
    )
    assert config.q2_start_utc == datetime(2026, 4, 1, tzinfo=UTC)


def test_generated_schemas_forbid_additional_properties() -> None:
    proposal_schema = ProposalOutput.model_json_schema()
    reflection_schema = ReflectionOutput.model_json_schema()
    assert proposal_schema["additionalProperties"] is False
    assert reflection_schema["additionalProperties"] is False
    assert AllowRule.model_json_schema()["additionalProperties"] is False
