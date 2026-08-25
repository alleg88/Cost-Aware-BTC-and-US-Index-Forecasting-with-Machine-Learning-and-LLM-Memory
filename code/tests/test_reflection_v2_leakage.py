from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from reflection_agent.v2.contracts import EvidenceCard, MemoryCard
from reflection_agent.v2.leakage import (
    FeatureProvenance,
    LeakageAuditor,
    LeakageError,
    PromptAuditContext,
    SignalProvenance,
    assert_stage_scope,
)


T0 = datetime(2024, 6, 1, tzinfo=UTC)


def valid_context() -> PromptAuditContext:
    source_cutoff = T0 - timedelta(days=10)
    eligible_after = source_cutoff + timedelta(minutes=1)
    evidence = EvidenceCard(
        evidence_id="evidence-shadow-1",
        stage="development",
        source_role="OOF_TEST",
        decision_time=T0 - timedelta(days=2),
        outcome_available_time=T0 - timedelta(days=1),
        fold_id=2,
        row_key="row-shadow-1",
        source_artifact_hash="b" * 64,
        route="REENTRY",
        side="SHORT",
        count=1,
        net_return=0.002,
        tags=["side:SHORT", "vol:HIGH"],
    )
    memory = MemoryCard(
        memory_id="memory-1",
        source_stage="development",
        protocol_scope="development_2021_2024",
        memory_type="EPISODIC",
        created_at_utc=T0 - timedelta(days=3),
        max_support_outcome_time=T0 - timedelta(days=4),
        lesson="A prior short re-entry rule failed after transaction costs.",
        evidence_status="REJECTED",
        tags=["side:SHORT"],
        source_evaluation_ids=["evaluation-1"],
        expires_at_utc=None,
        expires_after_episode=None,
    )
    return PromptAuditContext(
        call_id="call-1",
        call_kind="REFLECTION",
        stage="development",
        protocol_scope="development_2021_2024",
        cutoff_utc=T0,
        fold_id=2,
        source_episode_id="episode-1",
        source_episode_cutoff_utc=source_cutoff,
        source_episode_outcome_max_utc=source_cutoff - timedelta(minutes=1),
        candidate_eligible_after_utc=eligible_after,
        opportunity_decision_time=T0 - timedelta(days=2),
        active_policy_activates_at_utc=T0 - timedelta(days=3),
        regime_mapping_id="fixed-v2",
        features=[
            FeatureProvenance(
                feature_name="vol_regime",
                available_at_utc=T0 - timedelta(days=2),
                source_row_key="row-shadow-1",
            )
        ],
        signals=[
            SignalProvenance(
                row_key="row-shadow-1",
                stage="development",
                source_role="OOF_TEST",
                decision_time=T0 - timedelta(days=2),
                fold_id=2,
            )
        ],
        evidence_cards=[evidence],
        memories=[memory],
        eligible_memory_ids={"memory-1"},
        memory_variant="REAL",
        prompt_hash="a" * 64,
    )


def inject_leak(context: PromptAuditContext, mutation: str) -> PromptAuditContext:
    payload = context.model_dump()
    if mutation == "future_feature":
        payload["features"][0]["available_at_utc"] = T0 + timedelta(minutes=1)
    elif mutation == "future_outcome":
        payload["evidence_cards"][0]["outcome_available_time"] = T0 + timedelta(minutes=1)
    elif mutation == "future_memory":
        payload["memories"][0]["created_at_utc"] = T0 + timedelta(minutes=2)
        payload["memories"][0]["max_support_outcome_time"] = T0 + timedelta(minutes=1)
    elif mutation == "calibration_role":
        payload["signals"][0]["source_role"] = "CALIBRATION"
    elif mutation == "same_time_activation":
        payload["active_policy_activates_at_utc"] = payload["opportunity_decision_time"]
    elif mutation == "cross_fold_shadow":
        payload["evidence_cards"][0]["fold_id"] = 3
    elif mutation == "wrong_stage":
        payload["evidence_cards"][0]["stage"] = "h1"
        payload["evidence_cards"][0]["source_role"] = "FROZEN_EXACT"
    elif mutation == "q2_timestamp":
        payload["cutoff_utc"] = datetime(2026, 4, 1, tzinfo=UTC)
    else:
        raise AssertionError(mutation)
    return PromptAuditContext.model_validate(payload)


class SpyTransport:
    def __init__(self) -> None:
        self.calls = 0

    def call(self) -> None:
        self.calls += 1


@pytest.mark.parametrize(
    "mutation",
    [
        "future_feature",
        "future_outcome",
        "future_memory",
        "calibration_role",
        "same_time_activation",
        "cross_fold_shadow",
        "wrong_stage",
        "q2_timestamp",
    ],
)
def test_future_or_cross_scope_input_fails_before_transport(mutation: str) -> None:
    auditor = LeakageAuditor()
    transport = SpyTransport()
    with pytest.raises(LeakageError):
        auditor.audit_prompt(inject_leak(valid_context(), mutation))
        transport.call()
    assert transport.calls == 0


def test_valid_prompt_persists_exact_maxima_and_hash(tmp_path) -> None:
    audit_path = tmp_path / "prompt_audit.jsonl"
    audit = LeakageAuditor(audit_path).audit_prompt(valid_context())
    assert audit.passed is True
    assert audit.max_feature_time == T0 - timedelta(days=2)
    assert audit.max_decision_time == T0 - timedelta(days=2)
    assert audit.max_outcome_available_time == T0 - timedelta(days=1)
    assert audit.max_memory_created_at == T0 - timedelta(days=3)
    assert audit.prompt_hash == "a" * 64
    rows = [json.loads(line) for line in audit_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["call_id"] == "call-1"
    assert all(rows[0]["checks"].values())


def test_shuffled_memory_must_come_from_same_pre_cutoff_pool() -> None:
    context = valid_context().model_copy(
        update={"memory_variant": "SHUFFLED", "eligible_memory_ids": {"other-memory"}}
    )
    with pytest.raises(LeakageError, match="eligible pre-cutoff pool"):
        LeakageAuditor().audit_prompt(context)


def test_source_episode_and_shadow_order_are_strict() -> None:
    context = valid_context().model_copy(
        update={"candidate_eligible_after_utc": valid_context().source_episode_cutoff_utc}
    )
    with pytest.raises(LeakageError, match="strictly later"):
        LeakageAuditor().audit_prompt(context)


def test_activation_and_stage_helpers_fail_closed() -> None:
    auditor = LeakageAuditor()
    with pytest.raises(LeakageError, match="strictly prior"):
        auditor.audit_activation(
            activates_at_utc=T0,
            opportunity_decision_time=T0,
            stage="development",
            protocol_scope="development_2021_2024",
        )
    with pytest.raises(LeakageError, match="scope"):
        assert_stage_scope("development", "exact_h1_2025", T0)
