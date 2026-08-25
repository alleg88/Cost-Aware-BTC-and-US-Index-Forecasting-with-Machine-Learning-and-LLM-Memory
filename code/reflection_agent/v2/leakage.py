"""Fail-closed leakage audits that run before cloud transport or policy replay."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from reflection_agent.v2.contracts import EvidenceCard, MemoryCard, StrictModel

Q2_START = datetime.fromisoformat("2026-04-01T00:00:00+00:00")

STAGE_SCOPES = {
    "development": "development_2021_2024",
    "h1": "exact_h1_2025",
    "forward": "exact_forward_2025_2026",
}
STAGE_BOUNDS = {
    "development": (
        datetime.fromisoformat("2021-01-01T00:00:00+00:00"),
        datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
    ),
    "h1": (
        datetime.fromisoformat("2025-01-01T00:00:00+00:00"),
        datetime.fromisoformat("2025-07-01T00:00:00+00:00"),
    ),
    "forward": (
        datetime.fromisoformat("2025-07-01T00:00:00+00:00"),
        Q2_START,
    ),
}


class LeakageError(RuntimeError):
    """Raised before any external call when provenance is not causal."""


class FeatureProvenance(StrictModel):
    feature_name: Annotated[str, Field(min_length=1, max_length=128)]
    available_at_utc: datetime
    source_row_key: Annotated[str, Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> "FeatureProvenance":
        if self.available_at_utc.tzinfo is None:
            raise ValueError("feature availability must be timezone-aware")
        return self


class SignalProvenance(StrictModel):
    row_key: Annotated[str, Field(min_length=1, max_length=256)]
    stage: Literal["development", "h1", "forward"]
    source_role: Literal["OOF_TEST", "FROZEN_EXACT", "FIT", "CALIBRATION"]
    decision_time: datetime
    fold_id: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def timestamp_is_aware(self) -> "SignalProvenance":
        if self.decision_time.tzinfo is None:
            raise ValueError("signal decision time must be timezone-aware")
        return self


class PromptAuditContext(StrictModel):
    call_id: Annotated[str, Field(min_length=1, max_length=128)]
    call_kind: Literal["PROPOSAL", "REFLECTION", "REPAIR"]
    stage: Literal["development", "h1", "forward"]
    protocol_scope: Annotated[str, Field(min_length=1, max_length=128)]
    cutoff_utc: datetime
    fold_id: Annotated[int, Field(ge=0)]
    source_episode_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_episode_cutoff_utc: datetime
    source_episode_outcome_max_utc: datetime
    candidate_eligible_after_utc: datetime | None = None
    opportunity_decision_time: datetime | None = None
    active_policy_activates_at_utc: datetime | None = None
    regime_mapping_id: Literal["fixed-v2"]
    features: list[FeatureProvenance] = Field(default_factory=list)
    signals: list[SignalProvenance] = Field(default_factory=list)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    memories: Annotated[list[MemoryCard], Field(max_length=4)] = Field(default_factory=list)
    eligible_memory_ids: set[str] = Field(default_factory=set)
    memory_variant: Literal["REAL", "NONE", "SHUFFLED", "STATIC"]
    prompt_hash: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def timestamps_are_aware(self) -> "PromptAuditContext":
        values = (
            self.cutoff_utc,
            self.source_episode_cutoff_utc,
            self.source_episode_outcome_max_utc,
            self.candidate_eligible_after_utc,
            self.opportunity_decision_time,
            self.active_policy_activates_at_utc,
        )
        if any(value is not None and value.tzinfo is None for value in values):
            raise ValueError("prompt audit timestamps must be timezone-aware")
        return self


class LeakageAudit(StrictModel):
    call_id: str
    call_kind: Literal["PROPOSAL", "REFLECTION", "REPAIR"]
    stage: Literal["development", "h1", "forward"]
    protocol_scope: str
    fold_id: int
    cutoff_utc: datetime
    prompt_hash: str
    max_feature_time: datetime | None
    max_decision_time: datetime | None
    max_outcome_available_time: datetime | None
    max_memory_created_at: datetime | None
    max_memory_support_outcome_time: datetime | None
    active_policy_activates_at_utc: datetime | None
    checks: dict[str, bool]
    passed: Literal[True] = True


def _latest(values: list[datetime]) -> datetime | None:
    return max(values) if values else None


def _assert_before_q2(values: list[datetime | None]) -> None:
    for value in values:
        if value is not None and value >= Q2_START:
            raise LeakageError("Q2 lockbox access is forbidden")


def assert_stage_scope(stage: str, protocol_scope: str, cutoff_utc: datetime) -> None:
    """Require the registered store and cutoff for one experimental stage."""

    if cutoff_utc.tzinfo is None:
        raise LeakageError("stage cutoff must be timezone-aware")
    if stage not in STAGE_SCOPES:
        raise LeakageError(f"unknown stage scope: {stage}")
    if protocol_scope != STAGE_SCOPES[stage]:
        raise LeakageError(
            f"protocol scope mismatch: {protocol_scope!r} is not valid for {stage!r}"
        )
    if cutoff_utc >= Q2_START:
        raise LeakageError("Q2 lockbox access is forbidden")
    start, end = STAGE_BOUNDS[stage]
    if not start <= cutoff_utc < end:
        raise LeakageError(f"cutoff is outside the registered {stage} stage scope")


class LeakageAuditor:
    """Validate all host provenance and append an audit row only on success."""

    def __init__(self, audit_path: str | Path | None = None) -> None:
        self.audit_path = Path(audit_path) if audit_path is not None else None

    def audit_activation(
        self,
        *,
        activates_at_utc: datetime,
        opportunity_decision_time: datetime,
        stage: str,
        protocol_scope: str,
    ) -> None:
        assert_stage_scope(stage, protocol_scope, opportunity_decision_time)
        if activates_at_utc.tzinfo is None or opportunity_decision_time.tzinfo is None:
            raise LeakageError("policy activation timestamps must be timezone-aware")
        _assert_before_q2([activates_at_utc, opportunity_decision_time])
        if activates_at_utc >= opportunity_decision_time:
            raise LeakageError("policy activation must be strictly prior to opportunity")

    def audit_prompt(self, context: PromptAuditContext) -> LeakageAudit:
        assert_stage_scope(context.stage, context.protocol_scope, context.cutoff_utc)

        all_times: list[datetime | None] = [
            context.cutoff_utc,
            context.source_episode_cutoff_utc,
            context.source_episode_outcome_max_utc,
            context.candidate_eligible_after_utc,
            context.opportunity_decision_time,
            context.active_policy_activates_at_utc,
            *(item.available_at_utc for item in context.features),
            *(item.decision_time for item in context.signals),
            *(item.decision_time for item in context.evidence_cards),
            *(item.outcome_available_time for item in context.evidence_cards),
            *(item.created_at_utc for item in context.memories),
            *(item.max_support_outcome_time for item in context.memories),
        ]
        _assert_before_q2(all_times)

        if any(item.available_at_utc > context.cutoff_utc for item in context.features):
            raise LeakageError("future feature entered prompt")
        if any(item.decision_time > context.cutoff_utc for item in context.signals):
            raise LeakageError("future model signal entered prompt")
        if any(
            item.outcome_available_time > context.cutoff_utc
            for item in context.evidence_cards
        ):
            raise LeakageError("future outcome entered prompt")
        if any(item.created_at_utc > context.cutoff_utc for item in context.memories):
            raise LeakageError("future memory entered prompt")
        if any(
            item.max_support_outcome_time > item.created_at_utc
            for item in context.memories
        ):
            raise LeakageError("memory predates its supporting outcome")

        expected_role = "OOF_TEST" if context.stage == "development" else "FROZEN_EXACT"
        if any(
            item.stage != context.stage or item.source_role != expected_role
            for item in context.signals
        ):
            raise LeakageError("signal came from calibration, fit, or the wrong stage")
        if any(
            item.stage != context.stage or item.source_role != expected_role
            for item in context.evidence_cards
        ):
            raise LeakageError("evidence came from calibration, fit, or the wrong stage")
        if any(item.fold_id != context.fold_id for item in context.signals):
            raise LeakageError("signal crossed the active fold")
        if any(item.fold_id != context.fold_id for item in context.evidence_cards):
            raise LeakageError("shadow evidence crossed fold")

        if context.source_episode_outcome_max_utc > context.source_episode_cutoff_utc:
            raise LeakageError("source episode closed before all outcomes were available")
        if context.source_episode_cutoff_utc > context.cutoff_utc:
            raise LeakageError("source episode cutoff is in the future")
        if context.call_kind == "REFLECTION":
            eligible = context.candidate_eligible_after_utc
            if eligible is None or eligible <= context.source_episode_cutoff_utc:
                raise LeakageError("candidate shadow must start strictly later than source episode")
            if any(item.decision_time < eligible for item in context.evidence_cards):
                raise LeakageError("shadow evidence predates candidate eligibility")
        elif context.candidate_eligible_after_utc is not None:
            if context.candidate_eligible_after_utc <= context.source_episode_cutoff_utc:
                raise LeakageError("candidate must become eligible strictly later than source episode")

        if (
            context.active_policy_activates_at_utc is not None
            or context.opportunity_decision_time is not None
        ):
            if (
                context.active_policy_activates_at_utc is None
                or context.opportunity_decision_time is None
            ):
                raise LeakageError("policy activation audit requires both timestamps")
            self.audit_activation(
                activates_at_utc=context.active_policy_activates_at_utc,
                opportunity_decision_time=context.opportunity_decision_time,
                stage=context.stage,
                protocol_scope=context.protocol_scope,
            )

        allowed_memory_stages = {
            "development": {"development"},
            "h1": {"h1"},
            "forward": {"h1", "forward"},
        }[context.stage]
        if any(item.protocol_scope != context.protocol_scope for item in context.memories):
            raise LeakageError("memory came from the wrong isolated protocol scope")
        if any(item.source_stage not in allowed_memory_stages for item in context.memories):
            raise LeakageError("memory crossed a forbidden stage boundary")
        retrieved_ids = {item.memory_id for item in context.memories}
        if not retrieved_ids.issubset(context.eligible_memory_ids):
            raise LeakageError("memory was not sampled from the eligible pre-cutoff pool")
        if context.memory_variant in {"NONE", "STATIC"} and context.memories:
            raise LeakageError("memory-free control received persistent memory")

        checks = {
            "stage_scope": True,
            "q2_sealed": True,
            "features_causal": True,
            "signals_causal_oof_or_frozen": True,
            "outcomes_resolved": True,
            "memory_causal_and_isolated": True,
            "source_episode_complete": True,
            "future_shadow_strict": True,
            "policy_activation_strict": True,
            "regime_mapping_fixed": context.regime_mapping_id == "fixed-v2",
        }
        audit = LeakageAudit(
            call_id=context.call_id,
            call_kind=context.call_kind,
            stage=context.stage,
            protocol_scope=context.protocol_scope,
            fold_id=context.fold_id,
            cutoff_utc=context.cutoff_utc,
            prompt_hash=context.prompt_hash,
            max_feature_time=_latest(
                [item.available_at_utc for item in context.features]
            ),
            max_decision_time=_latest(
                [item.decision_time for item in context.signals]
                + [item.decision_time for item in context.evidence_cards]
            ),
            max_outcome_available_time=_latest(
                [item.outcome_available_time for item in context.evidence_cards]
            ),
            max_memory_created_at=_latest(
                [item.created_at_utc for item in context.memories]
            ),
            max_memory_support_outcome_time=_latest(
                [item.max_support_outcome_time for item in context.memories]
            ),
            active_policy_activates_at_utc=context.active_policy_activates_at_utc,
            checks=checks,
        )
        self._persist(audit)
        return audit

    def _persist(self, audit: LeakageAudit) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            audit.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


__all__ = [
    "FeatureProvenance",
    "LeakageAudit",
    "LeakageAuditor",
    "LeakageError",
    "PromptAuditContext",
    "SignalProvenance",
    "assert_stage_scope",
]
