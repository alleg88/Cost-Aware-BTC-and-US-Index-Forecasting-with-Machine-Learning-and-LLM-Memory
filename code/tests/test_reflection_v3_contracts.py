from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from reflection_agent.v3.config import load_v3_config
from reflection_agent.v3.contracts import (
    AllowRule,
    EvidenceCard,
    MemoryCard,
    ProposalChoiceOutput,
    ProposalOutput,
    ReflectionChoiceOutput,
    ReflectionOutput,
    validate_proposal_references,
)


CONFIG = Path(__file__).parents[1] / "configs" / "reflection_agent_v3.yaml"


def valid_add_payload(side: str = "SHORT") -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "source_episode_id": "episode-7",
        "decision": "ADD_ALLOW_RULE",
        "diagnosis_code": "REGIME_SPECIFIC_EDGE",
        "evidence_ids": ["evidence-1"],
        "memory_ids_used": ["memory-1"],
        "proposed_rule": {
            "action": "ALLOW_CANDIDATE",
            "predicates": [
                {"field": "side", "operator": "EQ", "value": side},
                {
                    "field": "confidence_tier",
                    "operator": "EQ",
                    "value": "HIGH_EXTRA",
                },
            ],
        },
        "target_rule_id": None,
        "hypothesis": "The supplied side and tier have repeated net-of-cost support.",
        "falsifiers": ["Strictly later target-side net return breaches its margin."],
        "confidence": "MEDIUM",
    }


def test_only_registered_atomic_decisions_validate() -> None:
    assert ProposalOutput.model_validate(valid_add_payload()).decision == "ADD_ALLOW_RULE"
    with pytest.raises(ValidationError):
        ProposalOutput.model_validate({**valid_add_payload(), "decision": "SET_THRESHOLD"})


def test_host_owned_choice_transport_is_small_closed_and_indexed() -> None:
    proposal = ProposalChoiceOutput(
        choice_index=3,
        evidence_indices=[0, 2],
        memory_indices=[1],
    )
    reflection = ReflectionChoiceOutput(
        evidence_indices=[0],
        memory_action_index=1,
    )
    assert proposal.model_dump(mode="json") == {
        "schema_version": "3.0",
        "choice_index": 3,
        "evidence_indices": [0, 2],
        "memory_indices": [1],
    }
    assert reflection.memory_action_index == 1
    with pytest.raises(ValidationError):
        ProposalChoiceOutput(
            choice_index=-1,
            evidence_indices=[0],
            memory_indices=[],
        )
    with pytest.raises(ValidationError):
        ProposalChoiceOutput.model_validate(
            {
                **proposal.model_dump(),
                "invented_rule": {"field": "price"},
            }
        )


def test_confidence_accepts_case_only_transport_variance() -> None:
    payload = valid_add_payload()
    payload["confidence"] = "medium"
    assert ProposalOutput.model_validate(payload).confidence == "MEDIUM"

    with pytest.raises(ValidationError):
        ProposalOutput.model_validate({**payload, "confidence": "fairly-high"})


def test_no_change_canonicalizes_only_non_actionable_transport_shape() -> None:
    payload = valid_add_payload()
    payload.update(
        {
            "decision": "no_change",
            "diagnosis_code": "insufficient_positive_evidence",
            "proposed_rule": payload["proposed_rule"],
            "target_rule_id": None,
            "hypothesis": "There is not enough repeated support to open a candidate.",
            "falsifiers": "Later repeated support would overturn this abstention.",
            "confidence": "low",
        }
    )

    result = ProposalOutput.model_validate(payload)

    assert result.decision == "NO_CHANGE"
    assert result.diagnosis_code == "INSUFFICIENT_EVIDENCE"
    assert result.proposed_rule is None
    assert result.target_rule_id is None
    assert result.hypothesis is None
    assert result.falsifiers == []
    assert result.confidence == "LOW"


def test_observed_flat_add_rule_is_canonicalized_without_changing_action() -> None:
    payload = valid_add_payload()
    payload.update(
        {
            "diagnosis_code": "supported_pattern",
            "proposed_rule": {
                "action": "ALLOW_CANDIDATE",
                "side": "short",
                "trend": "up",
                "oi": "flat",
            },
            "falsifiers": "Later target-side net support becomes negative.",
            "confidence": 0.7,
        }
    )

    result = ProposalOutput.model_validate(payload)

    assert result.decision == "ADD_ALLOW_RULE"
    assert result.diagnosis_code == "REGIME_SPECIFIC_EDGE"
    assert result.confidence == "HIGH"
    assert result.falsifiers == [
        "Later target-side net support becomes negative."
    ]
    assert [item.model_dump() for item in result.proposed_rule.predicates] == [
        {"field": "side", "operator": "EQ", "value": "SHORT"},
        {"field": "trend_regime", "operator": "EQ", "value": "UP"},
        {"field": "oi_regime", "operator": "EQ", "value": "FLAT"},
    ]


def test_observed_missing_eq_operators_are_representation_only() -> None:
    payload = valid_add_payload("LONG")
    payload["proposed_rule"] = {
        "action": "ALLOW_CANDIDATE",
        "predicates": [
            {"field": "side", "value": "long"},
            {"field": "vol", "value": "high"},
        ],
    }
    result = ProposalOutput.model_validate(payload)
    assert [item.model_dump() for item in result.proposed_rule.predicates] == [
        {"field": "side", "operator": "EQ", "value": "LONG"},
        {"field": "vol_regime", "operator": "EQ", "value": "HIGH"},
    ]


def test_closed_adapter_rejects_unknown_rule_semantics() -> None:
    payload = valid_add_payload()
    payload["proposed_rule"] = {
        "action": "ALLOW_CANDIDATE",
        "side": "SHORT",
        "momentum": "HIGH",
    }
    with pytest.raises(ValidationError):
        ProposalOutput.model_validate(payload)

    payload = valid_add_payload()
    payload["diagnosis_code"] = "MAGIC_EDGE"
    with pytest.raises(ValidationError):
        ProposalOutput.model_validate(payload)


def test_allow_rule_requires_one_side_and_at_most_two_extra_predicates() -> None:
    assert AllowRule.model_validate(valid_add_payload()["proposed_rule"]).action == (
        "ALLOW_CANDIDATE"
    )
    without_side = valid_add_payload()
    without_side["proposed_rule"] = {
        "action": "ALLOW_CANDIDATE",
        "predicates": [
            {"field": "vol_regime", "operator": "EQ", "value": "HIGH"}
        ],
    }
    with pytest.raises(ValidationError, match="side predicate"):
        ProposalOutput.model_validate(without_side)

    too_many = valid_add_payload()
    too_many["proposed_rule"] = {
        "action": "ALLOW_CANDIDATE",
        "predicates": [
            {"field": "side", "operator": "EQ", "value": "LONG"},
            {"field": "vol_regime", "operator": "EQ", "value": "HIGH"},
            {"field": "trend_regime", "operator": "EQ", "value": "UP"},
            {"field": "oi_regime", "operator": "EQ", "value": "RISING"},
        ],
    }
    with pytest.raises(ValidationError):
        ProposalOutput.model_validate(too_many)


def test_rule_values_are_field_specific_and_predicates_unique() -> None:
    bad_value = valid_add_payload()
    bad_value["proposed_rule"] = {
        "action": "ALLOW_CANDIDATE",
        "predicates": [{"field": "side", "operator": "EQ", "value": "HIGH"}],
    }
    with pytest.raises(ValidationError, match="not allowed for side"):
        ProposalOutput.model_validate(bad_value)

    duplicate = valid_add_payload()
    duplicate["proposed_rule"] = {
        "action": "ALLOW_CANDIDATE",
        "predicates": [
            {"field": "side", "operator": "EQ", "value": "SHORT"},
            {"field": "side", "operator": "EQ", "value": "LONG"},
        ],
    }
    with pytest.raises(ValidationError, match="side predicate"):
        ProposalOutput.model_validate(duplicate)


def test_host_rejects_invented_ids_and_side_reversal() -> None:
    proposal = ProposalOutput.model_validate(valid_add_payload("SHORT"))
    validate_proposal_references(
        proposal,
        evidence_ids={"evidence-1"},
        memory_ids={"memory-1"},
        active_rule_ids=set(),
        source_sides={"SHORT"},
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        validate_proposal_references(
            proposal,
            evidence_ids=set(),
            memory_ids={"memory-1"},
            active_rule_ids=set(),
            source_sides={"SHORT"},
        )
    with pytest.raises(ValueError, match="source side"):
        validate_proposal_references(
            proposal,
            evidence_ids={"evidence-1"},
            memory_ids={"memory-1"},
            active_rule_ids=set(),
            source_sides={"LONG"},
        )


def test_evidence_and_continuous_memory_provenance_are_strict() -> None:
    decision = datetime(2024, 1, 1, 12, tzinfo=UTC)
    outcome = decision + timedelta(minutes=15)
    evidence = EvidenceCard(
        evidence_id="evidence-1",
        stage="development",
        source_role="OOF_TEST",
        decision_time=decision,
        outcome_available_time=outcome,
        fold_id=3,
        row_key="opaque-row-1",
        source_artifact_hash="a" * 64,
        route="COVERAGE_CANDIDATE",
        side="SHORT",
        count=8,
        gross_return=0.001,
        cost_return=0.0008,
        net_return=0.0002,
        tags=["side:SHORT", "tier:HIGH_EXTRA"],
    )
    memory = MemoryCard(
        memory_id="memory-1",
        source_stage="development",
        protocol_scope="continuous_2021_2026",
        memory_type="EPISODIC",
        created_at_utc=outcome,
        max_support_outcome_time=outcome,
        lesson="The short high-extra condition passed its future shadow.",
        evidence_status="SUPPORTED",
        tags=["side:SHORT", "tier:HIGH_EXTRA"],
        source_evaluation_ids=["evaluation-1"],
    )
    assert evidence.gross_return - evidence.cost_return == pytest.approx(
        evidence.net_return
    )
    assert memory.protocol_scope == "continuous_2021_2026"
    with pytest.raises(ValidationError):
        EvidenceCard.model_validate({**evidence.model_dump(), "price": 60_000})


def test_reflection_cannot_override_deterministic_evaluation() -> None:
    output = ReflectionOutput(
        candidate_id="candidate-1",
        evaluator_decision="REJECT",
        evidence_ids=["future-evidence-1"],
        failure_code="COST_DRAG",
        lesson=None,
        invalidation_conditions=["The target-side margin remains breached."],
        memory_recommendation="STORE_EPISODE",
    )
    with pytest.raises(ValidationError, match="positive semantic lesson"):
        ReflectionOutput.model_validate(
            {
                **output.model_dump(),
                "lesson": "This rejected rule is profitable.",
                "memory_recommendation": "PROPOSE_SEMANTIC",
            }
        )


def test_registered_config_pins_continuous_protocol() -> None:
    config = load_v3_config(CONFIG)
    assert config.protocol_version == "reflection-agent-v3.0"
    assert config.protocol_scope == "continuous_2021_2026"
    assert config.required_model_digest == (
        "5166728b9358990e5f6c34f87cbe48716be2f2cd2d3b98527dff27ea755bf3ba"
    )
    assert config.model == "deepseek-v4-flash:cloud"
    assert config.think == "low"
    assert config.episode_min_candidates == 20
    assert config.episode_max_candidates == 30
    assert config.episode_min_per_side == 6
    assert config.shadow_min_matching_candidates == 12
    assert config.shadow_max_candidates == 40
    assert config.max_active_rules_per_side == 3
    assert config.net_noninferiority_margin == 0.005
    assert config.side_noninferiority_margin == 0.0025
    assert config.q2_start_utc == datetime(2026, 4, 1, tzinfo=UTC)


def test_generated_schemas_forbid_additional_properties() -> None:
    assert ProposalOutput.model_json_schema()["additionalProperties"] is False
    assert ReflectionOutput.model_json_schema()["additionalProperties"] is False
    assert AllowRule.model_json_schema()["additionalProperties"] is False
