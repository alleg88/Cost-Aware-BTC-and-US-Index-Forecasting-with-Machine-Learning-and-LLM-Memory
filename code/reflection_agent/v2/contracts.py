"""Strict host/LLM contracts for the bounded Reflection Agent v2."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "2.0"

CONDITION_VALUES: dict[str, tuple[str, ...]] = {
    "side": ("LONG", "SHORT"),
    "member_pattern": ("LSTM_ONLY", "SVM_ONLY", "BOTH_AGREE"),
    "episode_bar_bucket": ("SECOND", "THIRD_PLUS"),
    "vol_regime": ("LOW", "NORMAL", "HIGH"),
    "trend_regime": ("DOWN", "FLAT", "UP"),
    "funding_regime": ("NEGATIVE", "NEUTRAL", "POSITIVE", "MISSING"),
    "oi_regime": ("FALLING", "FLAT", "RISING", "MISSING"),
}

ConditionField = Literal[
    "side",
    "member_pattern",
    "episode_bar_bucket",
    "vol_regime",
    "trend_regime",
    "funding_regime",
    "oi_regime",
]
Decision = Literal["NO_CHANGE", "ADD_ALLOW_RULE", "REMOVE_ALLOW_RULE"]
DiagnosisCode = Literal[
    "NO_STABLE_PATTERN",
    "INSUFFICIENT_EVIDENCE",
    "COST_DRAG",
    "LONG_WEAKNESS",
    "SHORT_WEAKNESS",
    "REGIME_SPECIFIC_EDGE",
    "STALE_ACTIVE_RULE",
]


class StrictModel(BaseModel):
    """Reject extra fields and prevent in-process mutation after validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Predicate(StrictModel):
    field: ConditionField
    operator: Literal["EQ"]
    value: str

    @model_validator(mode="after")
    def value_matches_field(self) -> "Predicate":
        if self.value not in CONDITION_VALUES[self.field]:
            raise ValueError(f"value {self.value!r} is not allowed for {self.field}")
        return self


class AllowRule(StrictModel):
    action: Literal["ALLOW_REENTRY"]
    predicates: Annotated[list[Predicate], Field(min_length=1, max_length=2)]

    @field_validator("predicates")
    @classmethod
    def predicate_fields_are_unique(cls, predicates: list[Predicate]) -> list[Predicate]:
        fields = [predicate.field for predicate in predicates]
        if len(fields) != len(set(fields)):
            raise ValueError("duplicate predicate field")
        return predicates


class EvidenceCard(StrictModel):
    evidence_id: Annotated[str, Field(min_length=1, max_length=128)]
    stage: Literal["development", "h1", "forward"]
    source_role: Literal["OOF_TEST", "FROZEN_EXACT", "FIT", "CALIBRATION"]
    decision_time: datetime
    outcome_available_time: datetime
    fold_id: Annotated[int, Field(ge=0)]
    row_key: Annotated[str, Field(min_length=1, max_length=256)]
    source_artifact_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    route: Literal["UNION_BASE", "REENTRY"]
    side: Literal["LONG", "SHORT"]
    count: Annotated[int, Field(ge=0)]
    net_return: float
    tags: Annotated[list[str], Field(max_length=16)] = Field(default_factory=list)

    @model_validator(mode="after")
    def outcome_is_not_before_decision(self) -> "EvidenceCard":
        if self.decision_time.tzinfo is None or self.outcome_available_time.tzinfo is None:
            raise ValueError("evidence timestamps must be timezone-aware")
        if self.outcome_available_time < self.decision_time:
            raise ValueError("outcome cannot be available before its decision")
        return self


class MemoryCard(StrictModel):
    memory_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_stage: Literal["development", "h1", "forward"]
    protocol_scope: Annotated[str, Field(min_length=1, max_length=128)]
    memory_type: Literal["WORKING", "EPISODIC", "SEMANTIC", "PROCEDURAL"]
    created_at_utc: datetime
    max_support_outcome_time: datetime
    lesson: Annotated[str, Field(min_length=10, max_length=400)]
    evidence_status: Literal[
        "SUPPORTED", "REJECTED", "INCONCLUSIVE", "CONTRADICTED", "PROCEDURAL"
    ]
    tags: Annotated[list[str], Field(max_length=16)] = Field(default_factory=list)
    source_evaluation_ids: Annotated[list[str], Field(max_length=8)] = Field(default_factory=list)
    expires_at_utc: datetime | None = None
    expires_after_episode: Annotated[int | None, Field(ge=0)] = None

    @model_validator(mode="after")
    def support_precedes_creation_and_expiry(self) -> "MemoryCard":
        if self.created_at_utc.tzinfo is None or self.max_support_outcome_time.tzinfo is None:
            raise ValueError("memory timestamps must be timezone-aware")
        if self.max_support_outcome_time > self.created_at_utc:
            raise ValueError("memory cannot precede its supporting outcome")
        if self.expires_at_utc is not None:
            if self.expires_at_utc.tzinfo is None:
                raise ValueError("memory expiry must be timezone-aware")
            if self.expires_at_utc <= self.created_at_utc:
                raise ValueError("memory expiry must follow creation")
        return self


class ProposalOutput(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    source_episode_id: Annotated[str, Field(min_length=1, max_length=128)]
    decision: Decision
    diagnosis_code: DiagnosisCode
    evidence_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    memory_ids_used: Annotated[list[str], Field(max_length=4)] = Field(default_factory=list)
    proposed_rule: AllowRule | None
    target_rule_id: Annotated[str | None, Field(min_length=1, max_length=128)]
    hypothesis: Annotated[str | None, Field(min_length=10, max_length=400)]
    falsifiers: Annotated[list[str], Field(max_length=4)]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]

    @model_validator(mode="after")
    def decision_shape_is_closed(self) -> "ProposalOutput":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence ID")
        if len(self.memory_ids_used) != len(set(self.memory_ids_used)):
            raise ValueError("duplicate memory ID")
        if self.decision == "NO_CHANGE":
            if any((self.proposed_rule, self.target_rule_id, self.hypothesis)) or self.falsifiers:
                raise ValueError("NO_CHANGE requires null edit fields and no falsifiers")
        elif self.decision == "ADD_ALLOW_RULE":
            if self.proposed_rule is None or self.target_rule_id is not None:
                raise ValueError("ADD_ALLOW_RULE requires proposed_rule and no target_rule_id")
            if self.hypothesis is None or not self.falsifiers:
                raise ValueError("ADD_ALLOW_RULE requires a hypothesis and falsifiers")
        else:
            if self.proposed_rule is not None or self.target_rule_id is None:
                raise ValueError("REMOVE_ALLOW_RULE requires target_rule_id and no proposed_rule")
            if self.hypothesis is None or not self.falsifiers:
                raise ValueError("REMOVE_ALLOW_RULE requires a hypothesis and falsifiers")
        return self


class ReflectionOutput(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    evaluator_decision: Literal["PROMOTE", "REJECT", "INCONCLUSIVE"]
    evidence_ids: Annotated[list[str], Field(min_length=1, max_length=4)]
    failure_code: Literal[
        "NONE",
        "COST_DRAG",
        "SIDE_IMBALANCE",
        "REGIME_MISMATCH",
        "TOO_FEW_TRIGGERS",
        "OVERFIT_CONCENTRATION",
        "RULE_COLLISION",
        "SCHEMA_FAILURE",
    ]
    lesson: Annotated[str | None, Field(min_length=10, max_length=400)]
    invalidation_conditions: Annotated[list[str], Field(max_length=4)] = Field(default_factory=list)
    memory_recommendation: Literal["STORE_EPISODE", "PROPOSE_SEMANTIC", "DO_NOT_GENERALIZE"]

    @model_validator(mode="after")
    def reflection_cannot_rewrite_evaluation(self) -> "ReflectionOutput":
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("duplicate evidence ID")
        if self.evaluator_decision == "PROMOTE" and self.failure_code != "NONE":
            raise ValueError("PROMOTE requires failure_code NONE")
        if self.evaluator_decision != "PROMOTE" and self.failure_code == "NONE":
            raise ValueError("non-promoted evaluation requires a failure code")
        if self.evaluator_decision == "INCONCLUSIVE" and self.lesson is not None:
            raise ValueError("INCONCLUSIVE cannot generalize a lesson")
        if self.evaluator_decision != "PROMOTE" and self.memory_recommendation == "PROPOSE_SEMANTIC":
            raise ValueError("a rejected or inconclusive result cannot create a positive semantic lesson")
        return self


def validate_proposal_references(
    proposal: ProposalOutput,
    *,
    evidence_ids: set[str],
    memory_ids: set[str],
    active_rule_ids: set[str],
) -> None:
    """Fail closed when the model cites an ID the host did not supply."""

    unknown_evidence = set(proposal.evidence_ids) - evidence_ids
    if unknown_evidence:
        raise ValueError(f"unknown evidence IDs: {sorted(unknown_evidence)}")
    unknown_memory = set(proposal.memory_ids_used) - memory_ids
    if unknown_memory:
        raise ValueError(f"unknown memory IDs: {sorted(unknown_memory)}")
    if proposal.decision == "REMOVE_ALLOW_RULE" and proposal.target_rule_id not in active_rule_ids:
        raise ValueError(f"unknown active rule ID: {proposal.target_rule_id}")
