from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from reflection_agent.v2.contracts import ProposalOutput, ReflectionOutput
from reflection_agent.v2.leakage import LeakageAuditor, LeakageError
from reflection_agent.v2.memory import RealMemory
from reflection_agent.v2.orchestrator import ReflectionOrchestrator
from reflection_agent.v2.transport import SchemaCallResult


T0 = datetime(2024, 1, 2, tzinfo=UTC)
SCOPE = "development_2021_2024"


def source_hash() -> str:
    return hashlib.sha256(b"source").hexdigest()


def episode_frame() -> pd.DataFrame:
    rows = []
    for index in range(20):
        decision = T0 + timedelta(minutes=30 * index)
        route = "UNION_BASE" if index % 2 == 0 else "REENTRY"
        rows.append(
            {
                "opportunity_id": f"source-{index}",
                "stage": "development",
                "source_role": "OOF_TEST",
                "fold_id": 0,
                "row_key": f"source-row-{index}",
                "source_artifact_hash": source_hash(),
                "route": route,
                "side": "SHORT" if index % 4 < 2 else "LONG",
                "member_pattern": "LSTM_ONLY",
                "episode_bar_bucket": "SECOND" if route == "REENTRY" else None,
                "vol_regime": "HIGH" if index % 4 < 2 else "NORMAL",
                "trend_regime": "DOWN" if index % 4 < 2 else "UP",
                "funding_regime": "NEUTRAL",
                "oi_regime": "FLAT",
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": decision + timedelta(minutes=15),
                "outcome_available_time": decision + timedelta(minutes=29),
                "gross_return": 0.002,
                "net_return": 0.001,
                "round_trip_cost": 0.001,
            }
        )
    return pd.DataFrame(rows)


def future_frame(start: datetime) -> pd.DataFrame:
    rows = []
    for index in range(12):
        decision = start + timedelta(days=5 * index)
        base_net = 0.001 if index % 3 else -0.0002
        common = {
            "stage": "development",
            "source_role": "OOF_TEST",
            "fold_id": 0,
            "source_artifact_hash": source_hash(),
            "member_pattern": "LSTM_ONLY",
            "trend_regime": "DOWN",
            "funding_regime": "NEUTRAL",
            "oi_regime": "FLAT",
        }
        rows.append(
            {
                **common,
                "opportunity_id": f"future-base-{index}",
                "row_key": f"future-base-row-{index}",
                "route": "UNION_BASE",
                "side": "SHORT" if index % 2 == 0 else "LONG",
                "episode_bar_bucket": None,
                "vol_regime": "NORMAL",
                "decision_time": decision,
                "feature_available_time": decision,
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
                **common,
                "opportunity_id": f"future-reentry-{index}",
                "row_key": f"future-reentry-row-{index}",
                "route": "REENTRY",
                "side": "SHORT" if is_short else "LONG",
                "episode_bar_bucket": "THIRD_PLUS",
                "vol_regime": "HIGH" if is_short else "NORMAL",
                "decision_time": decision + timedelta(minutes=30),
                "feature_available_time": decision + timedelta(minutes=30),
                "entry_time": decision + timedelta(minutes=45),
                "outcome_available_time": decision + timedelta(minutes=59),
                "gross_return": reentry_net + 0.001,
                "net_return": reentry_net,
                "round_trip_cost": 0.001,
            }
        )
    return pd.DataFrame(rows)


class StubCaller:
    def __init__(self, *, invented_evidence: bool = False) -> None:
        self.calls: list[str] = []
        self.invented_evidence = invented_evidence

    def call(self, *, role, messages, response_model, allowed_ids):
        self.calls.append(role)
        if role == "proposal":
            evidence_id = (
                "invented-evidence"
                if self.invented_evidence
                else allowed_ids["evidence_ids"][0]
            )
            value = ProposalOutput(
                source_episode_id=allowed_ids["source_episode_id"],
                decision="ADD_ALLOW_RULE",
                diagnosis_code="REGIME_SPECIFIC_EDGE",
                evidence_ids=[evidence_id],
                memory_ids_used=[],
                proposed_rule={
                    "action": "ALLOW_REENTRY",
                    "predicates": [
                        {"field": "side", "operator": "EQ", "value": "SHORT"},
                        {"field": "vol_regime", "operator": "EQ", "value": "HIGH"},
                    ],
                },
                target_rule_id=None,
                hypothesis="High-volatility short re-entries retain future net support.",
                falsifiers=["Future incremental SHORT net return is non-positive."],
                confidence="MEDIUM",
            )
        else:
            value = ReflectionOutput(
                candidate_id=allowed_ids["candidate_id"],
                evaluator_decision=allowed_ids["evaluator_decision"],
                evidence_ids=allowed_ids["evidence_ids"],
                failure_code="NONE",
                lesson="High-volatility short re-entries retained positive future net returns.",
                invalidation_conditions=["Incremental SHORT net return becomes non-positive."],
                memory_recommendation="PROPOSE_SEMANTIC",
            )
        return SchemaCallResult(
            status="success",
            value=value,
            raw_content=value.model_dump_json(),
            request_hash="a" * 64,
            response_hash="b" * 64,
            schema_hash="c" * 64,
            attempts=1,
            latency_seconds=0.01,
            metadata={},
            errors=(),
        )


def orchestrator(tmp_path, caller: StubCaller):
    memory = RealMemory(
        tmp_path / "agent_state.sqlite",
        protocol_hash="protocol-hash",
        protocol_scope=SCOPE,
    )
    return (
        ReflectionOrchestrator(
            caller=caller,
            auditor=LeakageAuditor(tmp_path / "prompt_audit.jsonl"),
            memory=memory,
            protocol_hash="protocol-hash",
        ),
        memory,
    )


def test_proposal_uses_host_ids_and_opens_only_future_candidate(tmp_path) -> None:
    caller = StubCaller()
    agent, _ = orchestrator(tmp_path, caller)
    result = agent.propose_episode(
        episode_frame(), episode_id="episode-1", episode_number=1, active_rules=[]
    )
    assert result.candidate is not None
    assert result.candidate.candidate_id.startswith("candidate-")
    assert result.candidate.eligible_after_utc > result.candidate.source_episode_cutoff_utc
    assert result.status == "candidate_opened"
    assert caller.calls == ["proposal"]


def test_existing_open_candidate_blocks_a_second_cloud_proposal(tmp_path) -> None:
    caller = StubCaller()
    agent, _ = orchestrator(tmp_path, caller)
    first = agent.propose_episode(
        episode_frame(), episode_id="episode-1", episode_number=1, active_rules=[]
    )
    second = agent.propose_episode(
        episode_frame(),
        episode_id="episode-2",
        episode_number=2,
        active_rules=[],
        open_candidate=first.candidate,
    )
    assert second.status == "open_candidate_exists"
    assert second.candidate == first.candidate
    assert caller.calls == ["proposal"]


def test_invented_reference_is_rejected_by_host(tmp_path) -> None:
    caller = StubCaller(invented_evidence=True)
    agent, _ = orchestrator(tmp_path, caller)
    result = agent.propose_episode(
        episode_frame(), episode_id="episode-1", episode_number=1, active_rules=[]
    )
    assert result.candidate is None
    assert result.status == "invalid_reference"


def test_leakage_aborts_before_caller(tmp_path) -> None:
    caller = StubCaller()
    agent, _ = orchestrator(tmp_path, caller)
    frame = episode_frame()
    frame.loc[0, "feature_available_time"] = frame["outcome_available_time"].max() + timedelta(
        minutes=1
    )
    with pytest.raises(LeakageError):
        agent.propose_episode(
            frame, episode_id="episode-1", episode_number=1, active_rules=[]
        )
    assert caller.calls == []


def test_future_evaluator_promotes_then_activates_strictly_after_shadow(tmp_path) -> None:
    caller = StubCaller()
    agent, memory = orchestrator(tmp_path, caller)
    opened = agent.propose_episode(
        episode_frame(), episode_id="episode-1", episode_number=1, active_rules=[]
    )
    assert opened.candidate is not None
    future_start = (
        pd.Timestamp(opened.candidate.eligible_after_utc)
        .ceil("15min")
        .to_pydatetime()
        + timedelta(days=1)
    )
    future = future_frame(future_start)
    combined = pd.concat([episode_frame(), future], ignore_index=True)
    closure = agent.close_candidate(
        opened.candidate,
        combined,
        active_rules=[],
        episode_number=2,
    )
    assert closure.evaluation.decision == "PROMOTE"
    assert closure.new_active_rule is not None
    assert closure.new_active_rule.rule_id.startswith("rule-")
    assert closure.new_active_rule.activates_at_utc > closure.evaluation.shadow_cutoff_utc
    assert not any(
        opportunity_id.startswith("source-")
        for opportunity_id in closure.evaluation.evaluated_opportunity_ids
    )
    assert caller.calls == ["proposal", "reflection"]
    assert len(memory.all_cards()) == 1
