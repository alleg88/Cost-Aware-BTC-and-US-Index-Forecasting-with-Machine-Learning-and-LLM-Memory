"""Strict host/LLM contracts for the continuous coverage agent."""
from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "3.0"

CONDITION_VALUES: dict[str, tuple[str, ...]] = {
    "side": ("LONG", "SHORT"),
    "confidence_tier": ("HIGH_EXTRA", "MID_EXTRA", "LOW_EXTRA"),
    "signal_run_bucket": ("FIRST", "SECOND", "THIRD_PLUS"),
    "vol_regime": ("LOW", "NORMAL", "HIGH"),
    "trend_regime": ("DOWN", "FLAT", "UP"),
    "funding_regime": ("NEGATIVE", "NEUTRAL", "POSITIVE", "MISSING"),
    "oi_regime": ("FALLING", "FLAT", "RISING", "MISSING"),
}

ConditionField = Literal[
    "side",
    "confidence_tier",
    "signal_run_bucket",
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

_FIELD_ALIASES = {
    "side": "side",
    "tier": "confidence_tier",
    "confidence_tier": "confidence_tier",
    "run": "signal_run_bucket",
    "signal_run_bucket": "signal_run_bucket",
    "vol": "vol_regime",
    "vol_regime": "vol_regime",
    "trend": "trend_regime",
    "trend_regime": "trend_regime",
    "funding": "funding_regime",
    "funding_regime": "funding_regime",
    "oi": "oi_regime",
    "oi_regime": "oi_regime",
}
_DIAGNOSIS_ALIASES = {
    "INSUFFICIENT_POSITIVE_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    "NO_CHANGE": "INSUFFICIENT_EVIDENCE",
    "NO_CHANGE_SPARSE_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    "NO_POSITIVE_NET_PATTERN": "NO_STABLE_PATTERN",
    "NEGATIVE_NET_PATTERN": "NO_STABLE_PATTERN",
    "NEGATIVE_NET_CANDIDATE": "NO_STABLE_PATTERN",
    "NEGATIVE_NET_PATTERN_ALL_ROUTES": "NO_STABLE_PATTERN",
    "SPARSE_SUPPORT": "INSUFFICIENT_EVIDENCE",
    "SPARSE_EVIDENCE": "INSUFFICIENT_EVIDENCE",
    "INSUFFICIENT_REPEATED_NET_PATTERN": "INSUFFICIENT_EVIDENCE",
    "COST_ERODED_PATTERN": "COST_DRAG",
    "COSTS_ERASE_PATTERN": "COST_DRAG",
    "SUPPORTED_PATTERN": "REGIME_SPECIFIC_EDGE",
    "POSITIVE_NET_PATTERN": "REGIME_SPECIFIC_EDGE",
}


def _normalized_predicate(value: object) -> object:
    if not isinstance(value, dict):
        return value
    if not set(value).issubset({"field", "operator", "value"}):
        return value
    predicate = dict(value)
    field = predicate.get("field")
    if isinstance(field, str):
        predicate["field"] = _FIELD_ALIASES.get(field.lower(), field)
    operator = predicate.get("operator", "EQ")
    if isinstance(operator, str):
        operator = operator.upper()
    predicate["operator"] = operator
    if isinstance(predicate.get("value"), str):
        predicate["value"] = predicate["value"].upper()
    return predicate


def _normalized_rule(value: object) -> object:
    if not isinstance(value, dict):
        return value
    rule = dict(value)
    action = rule.get("action", "ALLOW_CANDIDATE")
    if isinstance(action, str):
        action = action.upper()
    predicates = rule.get("predicates")
    if isinstance(predicates, list):
        return {
            "action": action,
            "predicates": [_normalized_predicate(item) for item in predicates],
        }

    flat = predicates if isinstance(predicates, dict) else rule
    allowed_flat = {"action", *_FIELD_ALIASES}
    if not set(flat).issubset(allowed_flat):
        return value
    flat_action = flat.get("action", action)
    if isinstance(flat_action, str):
        flat_action = flat_action.upper()
    normalized_values: dict[str, object] = {}
    for field, raw in flat.items():
        if field == "action":
            continue
        canonical = _FIELD_ALIASES[field]
        normalized_values[canonical] = raw.upper() if isinstance(raw, str) else raw
    order = tuple(CONDITION_VALUES)
    return {
        "action": flat_action,
        "predicates": [
            {"field": field, "operator": "EQ", "value": normalized_values[field]}
            for field in order
            if field in normalized_values
        ],
    }


class StrictModel(BaseModel):
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
    action: Literal["ALLOW_CANDIDATE"]
    predicates: Annotated[list[Predicate], Field(min_length=1, max_length=3)]

    @field_validator("predicates")
    @classmethod
    def side_is_mandatory_and_fields_are_unique(
        cls, predicates: list[Predicate]
    ) -> list[Predicate]:
        fields = [predicate.field for predicate in predicates]
        if fields.count("side") != 1:
            raise ValueError("exactly one side predicate is required")
        if len(fields) != len(set(fields)):
            raise ValueError("side predicate and every condition field must be unique")
        return predicates

    @property
    def side(self) -> str:
        return next(item.value for item in self.predicates if item.field == "side")


class PolicyChoice(StrictModel):
    choice_index: Annotated[int, Field(ge=0, le=64)]
    decision: Decision
    proposed_rule: AllowRule | None
    target_rule_id: Annotated[str | None, Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def choice_shape_is_closed(self) -> "PolicyChoice":
        if self.decision == "NO_CHANGE":
            if self.proposed_rule is not None or self.target_rule_id is not None:
                raise ValueError("NO_CHANGE choice has no edit")
        elif self.decision == "ADD_ALLOW_RULE":
            if self.proposed_rule is None or self.target_rule_id is not None:
                raise ValueError("ADD choice requires only a proposed rule")
        elif self.proposed_rule is not None or self.target_rule_id is None:
            raise ValueError("REMOVE choice requires only a target rule ID")
        return self


IndexValue = Annotated[int, Field(ge=0, le=63)]


class ProposalChoiceOutput(StrictModel):
    """Small Cloud transport DTO; the host owns all executable policy JSON."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    choice_index: IndexValue
    evidence_indices: Annotated[list[IndexValue], Field(min_length=1, max_length=4)]
    memory_indices: Annotated[list[IndexValue], Field(max_length=4)] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def indices_are_unique(self) -> "ProposalChoiceOutput":
        if len(self.evidence_indices) != len(set(self.evidence_indices)):
            raise ValueError("duplicate evidence index")
        if len(self.memory_indices) != len(set(self.memory_indices)):
            raise ValueError("duplicate memory index")
        return self


class ReflectionChoiceOutput(StrictModel):
    """Small reflection DTO; evaluator verdict and lesson remain host-owned."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    evidence_indices: Annotated[list[IndexValue], Field(min_length=1, max_length=4)]
    memory_action_index: Annotated[int, Field(ge=0, le=2)]

    @field_validator("evidence_indices")
    @classmethod
    def evidence_indices_are_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate evidence index")
        return value


class EvidenceCard(StrictModel):
    evidence_id: Annotated[str, Field(min_length=1, max_length=128)]
    stage: Literal["development", "h1", "forward"]
    source_role: Literal["OOF_TEST", "FROZEN_EXACT", "FIT", "CALIBRATION"]
    decision_time: datetime
    outcome_available_time: datetime
    fold_id: Annotated[int, Field(ge=0)]
    row_key: Annotated[str, Field(min_length=1, max_length=256)]
    source_artifact_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    route: Literal["UNION_BASE", "COVERAGE_CANDIDATE"]
    side: Literal["LONG", "SHORT"]
    count: Annotated[int, Field(ge=0)]
    gross_return: float
    cost_return: float
    net_return: float
    tags: Annotated[list[str], Field(max_length=16)] = Field(default_factory=list)

    @model_validator(mode="after")
    def timestamps_and_economics_reconcile(self) -> "EvidenceCard":
        if self.decision_time.tzinfo is None or self.outcome_available_time.tzinfo is None:
            raise ValueError("evidence timestamps must be timezone-aware")
        if self.outcome_available_time <= self.decision_time:
            raise ValueError("outcome must be available strictly after its decision")
        values = (self.gross_return, self.cost_return, self.net_return)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("evidence economics must be finite")
        if not math.isclose(
            self.gross_return - self.cost_return,
            self.net_return,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("evidence gross, cost, and net returns do not reconcile")
        return self


class MemoryCard(StrictModel):
    memory_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_stage: Literal["development", "h1", "forward"]
    protocol_scope: Literal["continuous_2021_2026"]
    memory_type: Literal["WORKING", "EPISODIC", "SEMANTIC", "PROCEDURAL"]
    created_at_utc: datetime
    max_support_outcome_time: datetime
    lesson: Annotated[str, Field(min_length=10, max_length=400)]
    evidence_status: Literal[
        "SUPPORTED", "REJECTED", "INCONCLUSIVE", "CONTRADICTED", "PROCEDURAL"
    ]
    tags: Annotated[list[str], Field(max_length=16)] = Field(default_factory=list)
    source_evaluation_ids: Annotated[list[str], Field(max_length=8)] = Field(
        default_factory=list
    )
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
    memory_ids_used: Annotated[list[str], Field(max_length=4)] = Field(
        default_factory=list
    )
    proposed_rule: AllowRule | None
    target_rule_id: Annotated[str | None, Field(min_length=1, max_length=128)]
    hypothesis: Annotated[str | None, Field(min_length=10, max_length=400)]
    falsifiers: Annotated[list[str], Field(max_length=4)]
    confidence: Literal["LOW", "MEDIUM", "HIGH"]

    @model_validator(mode="before")
    @classmethod
    def canonicalize_transport_shape(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        for field in ("decision", "diagnosis_code", "confidence"):
            if isinstance(payload.get(field), str):
                payload[field] = payload[field].upper()
        diagnosis = payload.get("diagnosis_code")
        if isinstance(diagnosis, str):
            payload["diagnosis_code"] = _DIAGNOSIS_ALIASES.get(
                diagnosis, diagnosis
            )
        confidence = payload.get("confidence")
        if (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(float(confidence))
            and 0.0 <= float(confidence) <= 1.0
        ):
            payload["confidence"] = (
                "LOW"
                if float(confidence) <= 1.0 / 3.0
                else "MEDIUM"
                if float(confidence) <= 2.0 / 3.0
                else "HIGH"
            )
        if payload.get("decision") == "NO_CHANGE":
            payload.update(
                proposed_rule=None,
                target_rule_id=None,
                hypothesis=None,
                falsifiers=[],
            )
        else:
            payload["proposed_rule"] = _normalized_rule(
                payload.get("proposed_rule")
            )
            if isinstance(payload.get("falsifiers"), str):
                payload["falsifiers"] = [payload["falsifiers"]]
        return payload

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence_case(cls, value: object) -> object:
        if isinstance(value, str):
            return value.upper()
        return value

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
    invalidation_conditions: Annotated[list[str], Field(max_length=4)] = Field(
        default_factory=list
    )
    memory_recommendation: Literal[
        "STORE_EPISODE", "PROPOSE_SEMANTIC", "DO_NOT_GENERALIZE"
    ]

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
        if self.evaluator_decision != "PROMOTE" and (
            self.memory_recommendation == "PROPOSE_SEMANTIC" or self.lesson is not None
        ):
            raise ValueError(
                "a rejected or inconclusive result cannot create a positive semantic lesson"
            )
        return self


def validate_proposal_references(
    proposal: ProposalOutput,
    *,
    evidence_ids: set[str],
    memory_ids: set[str],
    active_rule_ids: set[str],
    source_sides: set[str],
) -> None:
    unknown_evidence = set(proposal.evidence_ids) - evidence_ids
    if unknown_evidence:
        raise ValueError(f"unknown evidence IDs: {sorted(unknown_evidence)}")
    unknown_memory = set(proposal.memory_ids_used) - memory_ids
    if unknown_memory:
        raise ValueError(f"unknown memory IDs: {sorted(unknown_memory)}")
    if proposal.decision == "REMOVE_ALLOW_RULE" and proposal.target_rule_id not in active_rule_ids:
        raise ValueError(f"unknown active rule ID: {proposal.target_rule_id}")
    if proposal.decision == "ADD_ALLOW_RULE":
        assert proposal.proposed_rule is not None
        if proposal.proposed_rule.side not in source_sides:
            raise ValueError("proposed rule side is not a supplied source side")


__all__ = [
    "AllowRule",
    "CONDITION_VALUES",
    "EvidenceCard",
    "MemoryCard",
    "PolicyChoice",
    "Predicate",
    "ProposalChoiceOutput",
    "ProposalOutput",
    "ReflectionChoiceOutput",
    "ReflectionOutput",
    "SCHEMA_VERSION",
    "StrictModel",
    "validate_proposal_references",
]
