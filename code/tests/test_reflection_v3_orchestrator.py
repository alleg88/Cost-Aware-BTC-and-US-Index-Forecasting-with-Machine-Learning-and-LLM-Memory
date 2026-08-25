from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from reflection_agent.v3.contracts import ProposalChoiceOutput, ReflectionChoiceOutput
from reflection_agent.v3.leakage import LeakageAuditor, assert_prompt_redacted
from reflection_agent.v3.memory import RealMemory
from reflection_agent.v3.orchestrator import (
    ReflectionOrchestrator,
    build_evidence_cards,
    build_policy_choices,
    compact_observation,
)


UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _frame(
    *,
    start_index: int,
    count: int,
    episode_id: str,
    net_return: float = 0.002,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for offset in range(count):
        index = start_index + offset
        decision = BASE + timedelta(minutes=30 * (index + 1))
        side = "LONG" if offset % 2 == 0 else "SHORT"
        rows.append(
            {
                "opportunity_id": f"opp_{index:04d}",
                "stage": "development",
                "source_role": "OOF_TEST",
                "fold_id": 0,
                "row_key": f"row_{index:04d}",
                "source_artifact_hash": "a" * 64,
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": decision + timedelta(minutes=15),
                "exit_time": decision + timedelta(minutes=15),
                "outcome_available_time": decision + timedelta(minutes=25),
                "route": "COVERAGE_CANDIDATE",
                "side": side,
                "confidence_tier": "HIGH_EXTRA",
                "signal_run_bucket": "FIRST",
                "gross_return": net_return + 0.001,
                "net_return": net_return,
                "round_trip_cost": 0.001,
                "exit_reason": "TAKE_PROFIT" if net_return > 0 else "STOP_LOSS",
                "vol_regime": "NORMAL",
                "trend_regime": "UP" if side == "LONG" else "DOWN",
                "funding_regime": "NEUTRAL",
                "oi_regime": "RISING",
                "path_complete": True,
                "observation_episode_id": episode_id,
                "episode_can_propose": True,
            }
        )
    return pd.DataFrame(rows)


class AdaptiveCaller:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model="deepseek-v4-flash:cloud",
            think="low",
            stream=False,
            temperature=0.0,
            num_predict=4096,
        )
        self.roles: list[str] = []

    def call(self, *, role, messages, response_model, allowed_ids):
        self.roles.append(role)
        if role == "reflection":
            value = ReflectionChoiceOutput(
                evidence_indices=[allowed_ids["evidence_indices"][0]],
                memory_action_index=1,
            )
        elif len([item for item in self.roles if item == "proposal"]) == 1:
            value = ProposalChoiceOutput(
                choice_index=allowed_ids["choice_indices"][1],
                evidence_indices=[allowed_ids["evidence_indices"][0]],
                memory_indices=[],
            )
        else:
            value = ProposalChoiceOutput(
                choice_index=0,
                evidence_indices=[allowed_ids["evidence_indices"][0]],
                memory_indices=[],
            )
        schema_hash, request_hash = ReflectionOrchestrator.transport_hashes(
            self.config,
            role=role,
            messages=messages,
            response_model=response_model,
            allowed_ids=allowed_ids,
        )
        return SimpleNamespace(
            status="success",
            value=value,
            schema_hash=schema_hash,
            request_hash=request_hash,
            response_hash="d" * 64,
            attempts=1,
            errors=(),
        )


def _orchestrator(tmp_path: Path) -> tuple[ReflectionOrchestrator, RealMemory, AdaptiveCaller]:
    memory = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    caller = AdaptiveCaller()
    orchestrator = ReflectionOrchestrator(
        caller=caller,
        auditor=LeakageAuditor(tmp_path / "leakage.jsonl"),
        memory=memory,
        protocol_hash="p",
    )
    return orchestrator, memory, caller


def test_evidence_prompt_is_compact_resolved_and_anonymous() -> None:
    episode = _frame(start_index=0, count=20, episode_id="ep_a1")
    cards = build_evidence_cards(episode, namespace="ep_a1")
    payload = compact_observation(
        source_episode_id="ep_a1", cards=cards, memories=[], active_rules=[]
    )
    assert 1 <= len(cards) <= 4
    assert sum(card.count for card in cards) == 20
    prompt_text = "INPUT_JSON=" + __import__("json").dumps(
        payload, sort_keys=True, separators=(",", ":")
    )
    assert_prompt_redacted(prompt_text)
    lowered = prompt_text.lower()
    assert "decision_time" not in lowered
    assert "entry_time" not in lowered
    assert '"price"' not in lowered
    assert "btc" not in lowered
    assert all("gross_bps" in item and "tp_count" in item for item in payload["evidence_cards"])


def test_host_policy_choice_menu_is_bounded_deterministic_and_side_preserving() -> None:
    episode = _frame(start_index=0, count=20, episode_id="ep_a1")
    cards = build_evidence_cards(episode, namespace="ep_a1")
    first = build_policy_choices(cards, [])
    second = build_policy_choices(cards, [])
    assert first == second
    assert first[0].choice_index == 0
    assert first[0].decision == "NO_CHANGE"
    assert len(first) <= 45
    assert all(
        choice.proposed_rule.side in {"LONG", "SHORT"}
        for choice in first[1:]
        if choice.decision == "ADD_ALLOW_RULE"
    )


def test_run_episode_closes_prior_shadow_before_retrieval_and_new_proposal(tmp_path) -> None:
    orchestrator, memory, caller = _orchestrator(tmp_path)
    first_episode = _frame(start_index=0, count=20, episode_id="ep_first")
    first = orchestrator.run_episode(
        first_episode,
        episode_id="ep_first",
        episode_number=0,
        active_rules=[],
        open_candidates=[],
        available_opportunities=first_episode,
    )
    assert first.status == "candidate_opened"
    assert first.candidate is not None
    assert first.candidate.eligible_after_utc == (
        first_episode["outcome_available_time"].max().to_pydatetime()
        + timedelta(microseconds=1)
    )
    assert first.closures == []
    assert first.open_candidates == [first.candidate]

    future = _frame(start_index=20, count=24, episode_id="ep_second")
    future["side"] = "LONG"
    available = pd.concat([first_episode, future], ignore_index=True)
    second = orchestrator.run_episode(
        future,
        episode_id="ep_second",
        episode_number=1,
        active_rules=first.active_rules,
        open_candidates=first.open_candidates,
        available_opportunities=available,
    )
    assert second.closures[0].status == "closed"
    assert second.closures[0].evaluation.decision == "PROMOTE"
    assert len(second.active_rules) == 1
    assert second.open_candidates == []
    assert len(memory.all_cards()) == 1
    assert caller.roles == ["proposal", "reflection", "proposal"]
    events = orchestrator.transition_log
    close_index = events.index("close_future_shadow")
    assert events.index("reflect_future_shadow", close_index) < events.index(
        "apply_deterministic_policy", close_index
    ) < events.index("store_resolved_memory", close_index)
    retrieve_index = events.index("retrieve_pre_cutoff_memory", close_index)
    assert events.index("store_resolved_memory", close_index) < retrieve_index
    assert retrieve_index < events.index("audit_proposal_prompt", retrieve_index)
    assert events.index("audit_proposal_prompt", retrieve_index) < events.index(
        "call_proposal_model", retrieve_index
    )
