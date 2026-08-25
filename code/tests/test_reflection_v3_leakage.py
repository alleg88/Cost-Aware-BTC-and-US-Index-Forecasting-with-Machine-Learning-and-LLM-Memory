from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone

import pytest

from reflection_agent.v3.contracts import EvidenceCard, MemoryCard
from reflection_agent.v3.leakage import (
    FeatureProvenance,
    LeakageAuditor,
    LeakageError,
    PromptAuditContext,
    SignalProvenance,
    assert_prompt_redacted,
    run_audited_call,
)
from reflection_agent.v3.prompts import proposal_messages


UTC = timezone.utc


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _payload() -> dict[str, object]:
    return {
        "active_rules": [],
        "evidence_cards": [
            {
                "confidence_tier": "HIGH_EXTRA",
                "cost_bps": 10.0,
                "evidence_id": "ev_a1",
                "funding_regime": "NEUTRAL",
                "gross_bps": 18.0,
                "net_bps": 8.0,
                "oi_regime": "RISING",
                "outcome": "TAKE_PROFIT",
                "side": "LONG",
                "signal_run_bucket": "FIRST",
                "trend_regime": "UP",
                "vol_regime": "NORMAL",
            }
        ],
        "memory_cards": [],
        "schema_version": "3.0",
        "source_episode_id": "ep_a1",
    }


def _prompt(payload: dict[str, object]) -> str:
    messages = proposal_messages(payload)
    return "\n\n".join(message["content"] for message in messages)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _evidence(
    *,
    stage: str = "development",
    source_role: str = "OOF_TEST",
    fold_id: int = 1,
    decision_time: datetime | None = None,
    outcome_time: datetime | None = None,
) -> EvidenceCard:
    decision = decision_time or _dt("2024-02-01T00:00:00+00:00")
    outcome = outcome_time or decision + timedelta(minutes=20)
    return EvidenceCard(
        evidence_id="ev_a1",
        stage=stage,
        source_role=source_role,
        decision_time=decision,
        outcome_available_time=outcome,
        fold_id=fold_id,
        row_key="row_a1",
        source_artifact_hash="a" * 64,
        route="COVERAGE_CANDIDATE",
        side="LONG",
        count=1,
        gross_return=0.0018,
        cost_return=0.001,
        net_return=0.0008,
        tags=["LONG", "HIGH_EXTRA"],
    )


def _memory(
    *,
    source_stage: str = "development",
    created_at: datetime | None = None,
    support_time: datetime | None = None,
) -> MemoryCard:
    created = created_at or _dt("2024-02-01T12:00:00+00:00")
    support = support_time or created - timedelta(minutes=1)
    return MemoryCard(
        memory_id="mem_a1",
        source_stage=source_stage,
        protocol_scope="continuous_2021_2026",
        memory_type="EPISODIC",
        created_at_utc=created,
        max_support_outcome_time=support,
        lesson="A supported categorical coverage lesson.",
        evidence_status="SUPPORTED",
        tags=["LONG"],
        source_evaluation_ids=["eval_a1"],
    )


def _context(
    *,
    stage: str = "development",
    cutoff: datetime | None = None,
    prompt_payload: dict[str, object] | None = None,
    evidence: list[EvidenceCard] | None = None,
    memories: list[MemoryCard] | None = None,
    signals: list[SignalProvenance] | None = None,
    features: list[FeatureProvenance] | None = None,
    fold_id: int = 1,
    call_kind: str = "PROPOSAL",
    candidate_eligible_after: datetime | None = None,
    opportunity_decision_time: datetime | None = None,
    active_policy_activates_at: datetime | None = None,
) -> PromptAuditContext:
    cutoff_value = cutoff or _dt("2024-02-02T00:00:00+00:00")
    prompt = _prompt(prompt_payload or _payload())
    evidence_cards = evidence if evidence is not None else [_evidence(fold_id=fold_id)]
    signal_cards = signals if signals is not None else [
        SignalProvenance(
            row_key="row_a1",
            stage=stage,
            source_role="OOF_TEST" if stage == "development" else "FROZEN_EXACT",
            decision_time=evidence_cards[0].decision_time,
            fold_id=fold_id,
        )
    ]
    feature_cards = features if features is not None else [
        FeatureProvenance(
            feature_name="vol_regime",
            available_at_utc=evidence_cards[0].decision_time,
            source_row_key="row_a1",
        )
    ]
    memory_cards = memories or []
    return PromptAuditContext(
        call_id="call_a1",
        call_kind=call_kind,
        stage=stage,
        protocol_scope="continuous_2021_2026",
        cutoff_utc=cutoff_value,
        fold_id=fold_id,
        source_episode_id="ep_a1",
        source_episode_cutoff_utc=cutoff_value - timedelta(hours=1),
        source_episode_outcome_max_utc=cutoff_value - timedelta(hours=1, minutes=1),
        candidate_eligible_after_utc=candidate_eligible_after,
        opportunity_decision_time=opportunity_decision_time,
        active_policy_activates_at_utc=active_policy_activates_at,
        regime_mapping_id="fixed-v2",
        features=feature_cards,
        signals=signal_cards,
        evidence_cards=evidence_cards,
        memories=memory_cards,
        eligible_memory_ids={item.memory_id for item in memory_cards},
        memory_variant="REAL",
        prompt_text=prompt,
        prompt_hash=_hash(prompt),
        schema_hash="b" * 64,
        request_hash="c" * 64,
    )


class SpyTransport:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        return prompt


def _mutated_context(mutation: str) -> PromptAuditContext:
    cutoff = _dt("2024-02-02T00:00:00+00:00")
    if mutation == "future_feature":
        return _context(
            features=[
                FeatureProvenance(
                    feature_name="vol_regime",
                    available_at_utc=cutoff + timedelta(minutes=1),
                    source_row_key="row_a1",
                )
            ]
        )
    if mutation == "future_outcome":
        return _context(
            evidence=[
                _evidence(outcome_time=cutoff + timedelta(minutes=1))
            ]
        )
    if mutation == "future_memory":
        memory = _memory(
            created_at=cutoff + timedelta(minutes=1), support_time=cutoff
        )
        return _context(memories=[memory])
    if mutation == "fit_role":
        return _context(
            signals=[
                SignalProvenance(
                    row_key="row_a1",
                    stage="development",
                    source_role="FIT",
                    decision_time=_dt("2024-02-01T00:00:00+00:00"),
                    fold_id=1,
                )
            ]
        )
    if mutation == "same_time_activation":
        return _context(
            opportunity_decision_time=cutoff,
            active_policy_activates_at=cutoff,
        )
    if mutation == "cross_fold_shadow":
        decision = _dt("2024-02-01T23:02:00+00:00")
        return _context(
            cutoff=_dt("2024-02-01T23:30:00+00:00"),
            call_kind="REFLECTION",
            candidate_eligible_after=_dt("2024-02-01T23:01:00+00:00"),
            evidence=[
                _evidence(
                    fold_id=2,
                    decision_time=decision,
                    outcome_time=decision + timedelta(minutes=15),
                )
            ],
        )
    if mutation == "future_stage_memory":
        return _context(memories=[_memory(source_stage="h1")])
    if mutation == "q2_timestamp":
        return _context(cutoff=_dt("2026-04-01T00:00:00+00:00"))

    payload = _payload()
    if mutation == "asset_name":
        payload["source_episode_id"] = "BTC"
    elif mutation == "absolute_date":
        payload["source_episode_id"] = "2024-02-01"
    elif mutation == "price_field":
        payload["close"] = 42.0
    elif mutation == "file_path":
        payload["source_episode_id"] = "C:\\temp\\secret.txt"
    else:
        raise AssertionError(mutation)
    return _context(prompt_payload=payload)


@pytest.mark.parametrize(
    "mutation",
    [
        "future_feature",
        "future_outcome",
        "future_memory",
        "fit_role",
        "same_time_activation",
        "cross_fold_shadow",
        "future_stage_memory",
        "q2_timestamp",
        "asset_name",
        "absolute_date",
        "price_field",
        "file_path",
    ],
)
def test_leakage_or_latent_history_hint_aborts_before_transport(
    mutation: str, tmp_path
) -> None:
    spy = SpyTransport()
    audit_path = tmp_path / "audit.jsonl"
    auditor = LeakageAuditor(audit_path)
    with pytest.raises(LeakageError):
        run_audited_call(_mutated_context(mutation), transport=spy, auditor=auditor)
    assert spy.calls == 0
    persisted = audit_path.read_text(encoding="utf-8")
    assert '"passed":false' in persisted
    assert "prompt_text" not in persisted
    assert "INPUT_JSON" not in persisted


def test_valid_prompt_and_continuous_earlier_stage_memory_pass(tmp_path) -> None:
    cutoff = _dt("2025-01-10T00:00:00+00:00")
    decision = _dt("2025-01-09T00:00:00+00:00")
    evidence = _evidence(
        stage="h1",
        source_role="FROZEN_EXACT",
        decision_time=decision,
        outcome_time=decision + timedelta(minutes=15),
    )
    memory = _memory(
        source_stage="development",
        created_at=_dt("2024-12-31T00:00:00+00:00"),
        support_time=_dt("2024-12-30T23:59:00+00:00"),
    )
    context = _context(
        stage="h1",
        cutoff=cutoff,
        evidence=[evidence],
        memories=[memory],
        features=[
            FeatureProvenance(
                feature_name="vol_regime",
                available_at_utc=decision,
                source_row_key="row_a1",
            )
        ],
    )
    audit = LeakageAuditor(tmp_path / "audit.jsonl").audit_prompt(context)
    assert audit.passed is True
    assert audit.max_memory_created_at == memory.created_at_utc
    assert all(audit.checks.values())


def test_prompt_redaction_allows_opportunity_but_rejects_unknown_key() -> None:
    assert_prompt_redacted(_prompt(_payload()))
    payload = _payload()
    payload["future_hint"] = "opportunity"
    with pytest.raises(LeakageError, match="canonical compact schema"):
        assert_prompt_redacted(_prompt(payload))
