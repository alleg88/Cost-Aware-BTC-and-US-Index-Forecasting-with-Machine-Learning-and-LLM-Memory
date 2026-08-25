"""Fail-closed causal and latent-history audits for Reflection Agent v3."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Callable, Literal, TypeVar

from pydantic import Field, model_validator

from reflection_agent.v3.contracts import EvidenceCard, MemoryCard, StrictModel


Q2_START = datetime.fromisoformat("2026-04-01T00:00:00+00:00")
PROTOCOL_SCOPE = "continuous_2021_2026"
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
STAGE_ORDER = {"development": 0, "h1": 1, "forward": 2}

_HASH_PATTERN = r"^[0-9a-f]{64}$"
_ASSET_PATTERN = re.compile(
    r"\b(?:btc|bitcoin|eth|ethereum|binance|coinbase|bybit|kraken|okx|btcusdt|ethusdt)\b",
    re.IGNORECASE,
)
_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_DATE_PATTERN = re.compile(
    r"\b(?:19|20)\d{2}[-/]\d{2}[-/]\d{2}(?:[T ][0-9:.+\-Z]+)?\b",
    re.IGNORECASE,
)
_MONTH_PATTERN = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b"
)
_RAW_MARKET_KEY_PATTERN = re.compile(
    r'"(?:open|high|low|close|price)"\s*:', re.IGNORECASE
)
_FORBIDDEN_SUFFIX_KEY_PATTERN = re.compile(
    r'"[^"\\]*(?:_price|_timestamp|_path|_file)"\s*:', re.IGNORECASE
)
_WINDOWS_PATH_PATTERN = re.compile(r"\b[A-Za-z]:\\")
_UNIX_PATH_PATTERN = re.compile(r"(?:^|[\s\"'])/(?:[A-Za-z0-9._-]+/)+")

# Every runtime JSON key is host-owned. Values remain separately constrained by
# the Pydantic contracts and categorical allowlists.
CANONICAL_COMPACT_KEYS = frozenset(
    {
        "action",
        "active_rule_ids",
        "active_rules",
        "activates_after",
        "age_bucket",
        "candidate",
        "candidate_id",
        "candidate_metrics",
        "choice_index",
        "choice_menu",
        "confidence_tier",
        "control_metrics",
        "count",
        "cost_bps",
        "decision",
        "diagnosis_code",
        "evidence_cards",
        "evidence_id",
        "evidence_index",
        "evidence_ids",
        "evidence_indices",
        "evidence_status",
        "evaluation",
        "evaluator_decision",
        "failure_code",
        "failure_codes",
        "falsifiers",
        "field",
        "funding_regime",
        "gate_checks",
        "gross_bps",
        "hypothesis",
        "invalidation_conditions",
        "lesson",
        "long_net_bps",
        "matching_candidate_count",
        "matching_candidates",
        "max_drawdown",
        "memory_cards",
        "memory_id",
        "memory_index",
        "memory_action_index",
        "memory_ids_used",
        "memory_indices",
        "memory_recommendation",
        "memory_type",
        "metrics",
        "net_bps",
        "oi_regime",
        "operator",
        "outcome",
        "predicates",
        "proposal",
        "proposed_rule",
        "route",
        "rule_id",
        "schema_version",
        "side",
        "short_net_bps",
        "signal_run_bucket",
        "sl_count",
        "sortino",
        "source_episode_id",
        "source_stage",
        "subblock_count",
        "tags",
        "target_rule_id",
        "timeout_count",
        "tp_count",
        "trades",
        "triggered_trades",
        "total_candidate_count",
        "total_coverage_candidates",
        "trend_regime",
        "union_metrics",
        "value",
        "vol_regime",
    }
)


class LeakageError(RuntimeError):
    """Raised before external transport when an audit fails."""


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
    protocol_scope: Literal[PROTOCOL_SCOPE]
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
    memories: Annotated[list[MemoryCard], Field(max_length=4)] = Field(
        default_factory=list
    )
    eligible_memory_ids: set[str] = Field(default_factory=set)
    memory_variant: Literal["REAL", "NONE", "SHUFFLED", "STATIC"]
    prompt_text: Annotated[str, Field(min_length=1, max_length=200_000)]
    prompt_hash: Annotated[str, Field(pattern=_HASH_PATTERN)]
    schema_hash: Annotated[str, Field(pattern=_HASH_PATTERN)]
    request_hash: Annotated[str, Field(pattern=_HASH_PATTERN)]

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
    protocol_scope: Literal[PROTOCOL_SCOPE]
    fold_id: int
    cutoff_utc: datetime
    prompt_hash: str
    schema_hash: str
    request_hash: str
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
    if any(value is not None and value >= Q2_START for value in values):
        raise LeakageError("Q2 lockbox access is forbidden")


def assert_stage_scope(stage: str, protocol_scope: str, cutoff_utc: datetime) -> None:
    """Require one continuous store and a registered half-open stage cutoff."""

    if cutoff_utc.tzinfo is None:
        raise LeakageError("stage cutoff must be timezone-aware")
    if protocol_scope != PROTOCOL_SCOPE:
        raise LeakageError("protocol scope is not the registered continuous store")
    if stage not in STAGE_BOUNDS:
        raise LeakageError(f"unknown stage scope: {stage}")
    start, end = STAGE_BOUNDS[stage]
    if not start <= cutoff_utc < end:
        raise LeakageError(f"cutoff is outside the registered {stage} stage")
    _assert_before_q2([cutoff_utc])


def _runtime_json_objects(prompt: str) -> list[object]:
    decoder = json.JSONDecoder()
    objects: list[object] = []
    for marker in ("INPUT_JSON=", "CANDIDATE_JSON=", "EVALUATION_JSON="):
        start = 0
        while True:
            position = prompt.find(marker, start)
            if position < 0:
                break
            payload = prompt[position + len(marker) :].lstrip()
            try:
                value, consumed = decoder.raw_decode(payload)
            except json.JSONDecodeError as exc:
                raise LeakageError(f"runtime prompt JSON is invalid after {marker}") from exc
            objects.append(value)
            start = position + len(marker) + consumed
    if not objects:
        raise LeakageError("prompt contains no canonical runtime JSON")
    return objects


def _audit_json_keys(value: object) -> None:
    if isinstance(value, dict):
        unknown = set(value) - CANONICAL_COMPACT_KEYS
        if unknown:
            raise LeakageError("runtime key is outside the canonical compact schema")
        for child in value.values():
            _audit_json_keys(child)
    elif isinstance(value, list):
        for child in value:
            _audit_json_keys(child)


def assert_prompt_redacted(prompt: str) -> None:
    """Reject direct and structural latent-history hints before hashing/transport."""

    if _ASSET_PATTERN.search(prompt):
        raise LeakageError("prompt contains an asset or exchange token")
    if (
        _DATE_PATTERN.search(prompt)
        or _YEAR_PATTERN.search(prompt)
        or _MONTH_PATTERN.search(prompt)
    ):
        raise LeakageError("prompt contains an absolute calendar hint")
    if _RAW_MARKET_KEY_PATTERN.search(prompt):
        raise LeakageError("prompt contains a raw market key")
    if _FORBIDDEN_SUFFIX_KEY_PATTERN.search(prompt):
        raise LeakageError("prompt contains a forbidden provenance or price key")
    if _WINDOWS_PATH_PATTERN.search(prompt) or _UNIX_PATH_PATTERN.search(prompt):
        raise LeakageError("prompt contains a filesystem path")
    for payload in _runtime_json_objects(prompt):
        _audit_json_keys(payload)


class LeakageAuditor:
    """Validate provenance and persist content-free success/failure audits."""

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
        try:
            audit = self._audit_prompt(context)
        except LeakageError as exc:
            self._persist_failure(context, str(exc))
            raise
        self._persist(audit.model_dump(mode="json"))
        return audit

    def _audit_prompt(self, context: PromptAuditContext) -> LeakageAudit:
        assert_stage_scope(context.stage, context.protocol_scope, context.cutoff_utc)
        stage_start, stage_end = STAGE_BOUNDS[context.stage]
        if not (
            stage_start <= context.source_episode_outcome_max_utc < stage_end
            and stage_start <= context.source_episode_cutoff_utc < stage_end
        ):
            raise LeakageError("source episode crossed its registered stage")

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
        if any(
            item.created_at_utc >= context.cutoff_utc
            or item.max_support_outcome_time >= context.cutoff_utc
            for item in context.memories
        ):
            raise LeakageError("memory is not strictly earlier than prompt cutoff")

        expected_role = "OOF_TEST" if context.stage == "development" else "FROZEN_EXACT"
        if any(
            item.stage != context.stage or item.source_role != expected_role
            for item in context.signals
        ):
            raise LeakageError("signal came from fit, calibration, or the wrong stage")
        if any(
            item.stage != context.stage or item.source_role != expected_role
            for item in context.evidence_cards
        ):
            raise LeakageError("evidence came from fit, calibration, or the wrong stage")
        if context.stage == "development":
            if any(item.fold_id != context.fold_id for item in context.signals):
                raise LeakageError("development signal crossed the active OOF fold")
            if any(item.fold_id != context.fold_id for item in context.evidence_cards):
                raise LeakageError("development shadow evidence crossed an OOF fold")

        if context.source_episode_outcome_max_utc > context.source_episode_cutoff_utc:
            raise LeakageError("source episode closed before all outcomes resolved")
        if context.source_episode_cutoff_utc > context.cutoff_utc:
            raise LeakageError("source episode cutoff is in the future")
        if context.call_kind == "REFLECTION":
            eligible = context.candidate_eligible_after_utc
            if eligible is None or eligible <= context.source_episode_cutoff_utc:
                raise LeakageError("future shadow does not start strictly later")
            if any(item.decision_time < eligible for item in context.evidence_cards):
                raise LeakageError("shadow evidence predates candidate eligibility")
        elif (
            context.candidate_eligible_after_utc is not None
            and context.candidate_eligible_after_utc <= context.source_episode_cutoff_utc
        ):
            raise LeakageError("candidate eligibility is not strictly later")

        if (
            context.active_policy_activates_at_utc is not None
            or context.opportunity_decision_time is not None
        ):
            if (
                context.active_policy_activates_at_utc is None
                or context.opportunity_decision_time is None
            ):
                raise LeakageError("activation audit requires both timestamps")
            self.audit_activation(
                activates_at_utc=context.active_policy_activates_at_utc,
                opportunity_decision_time=context.opportunity_decision_time,
                stage=context.stage,
                protocol_scope=context.protocol_scope,
            )

        if any(item.protocol_scope != context.protocol_scope for item in context.memories):
            raise LeakageError("memory came from another protocol store")
        if any(
            STAGE_ORDER[item.source_stage] > STAGE_ORDER[context.stage]
            for item in context.memories
        ):
            raise LeakageError("memory came from a future stage")
        retrieved_ids = {item.memory_id for item in context.memories}
        if not retrieved_ids.issubset(context.eligible_memory_ids):
            raise LeakageError("memory was not sampled from the eligible causal pool")
        if context.memory_variant in {"NONE", "STATIC"} and context.memories:
            raise LeakageError("memory-free control received persistent memory")

        assert_prompt_redacted(context.prompt_text)
        actual_prompt_hash = hashlib.sha256(context.prompt_text.encode("utf-8")).hexdigest()
        if actual_prompt_hash != context.prompt_hash:
            raise LeakageError("prompt hash does not match audited prompt")

        checks = {
            "stage_scope": True,
            "q2_sealed": True,
            "features_causal": True,
            "signals_causal_oof_or_frozen": True,
            "outcomes_resolved": True,
            "memory_causal_continuous_and_isolated": True,
            "source_episode_complete": True,
            "future_shadow_strict": True,
            "policy_activation_strict": True,
            "regime_mapping_fixed": context.regime_mapping_id == "fixed-v2",
            "prompt_redacted": True,
            "prompt_hash_verified": True,
        }
        return LeakageAudit(
            call_id=context.call_id,
            call_kind=context.call_kind,
            stage=context.stage,
            protocol_scope=context.protocol_scope,
            fold_id=context.fold_id,
            cutoff_utc=context.cutoff_utc,
            prompt_hash=context.prompt_hash,
            schema_hash=context.schema_hash,
            request_hash=context.request_hash,
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

    def _persist_failure(self, context: PromptAuditContext, reason: str) -> None:
        self._persist(
            {
                "call_id": context.call_id,
                "call_kind": context.call_kind,
                "stage": context.stage,
                "protocol_scope": context.protocol_scope,
                "fold_id": context.fold_id,
                "cutoff_utc": context.cutoff_utc.isoformat(),
                "prompt_hash": context.prompt_hash,
                "schema_hash": context.schema_hash,
                "request_hash": context.request_hash,
                "checks": {},
                "passed": False,
                "failure_reason": reason,
            }
        )

    def _persist(self, record: dict[str, object]) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        with self.audit_path.open("a", encoding="utf-8") as handle:
            handle.write(encoded + "\n")


TransportResult = TypeVar("TransportResult")


def run_audited_call(
    context: PromptAuditContext,
    *,
    transport: Callable[[str], TransportResult],
    auditor: LeakageAuditor | None = None,
) -> TransportResult:
    """Prove audit-before-transport ordering for host integrations."""

    active_auditor = auditor or LeakageAuditor()
    active_auditor.audit_prompt(context)
    return transport(context.prompt_text)


__all__ = [
    "CANONICAL_COMPACT_KEYS",
    "FeatureProvenance",
    "LeakageAudit",
    "LeakageAuditor",
    "LeakageError",
    "PromptAuditContext",
    "SignalProvenance",
    "assert_prompt_redacted",
    "assert_stage_scope",
    "run_audited_call",
]
