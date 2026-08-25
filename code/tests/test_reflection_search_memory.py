from datetime import UTC, datetime, timedelta

from reflection_agent.contracts import (
    Candidate,
    EvaluationRecord,
    ExpectedEffect,
    MetricBundle,
    PolicyEdit,
    RefinerBatch,
    ReflectionRecord,
)
from reflection_agent.memory import MemoryManager
from reflection_agent.search import select_beam, validate_refinements
from reflection_agent.store import AgentStore
import pytest


def _candidate(candidate_id, value=0.75):
    return Candidate(
        candidate_id=candidate_id,
        hypothesis="Require stronger agreement in the supplied regime.",
        edits=[PolicyEdit(edit_id=f"{candidate_id}-e", action="require_minimum_agreement", value=value)],
        mechanism="Reduce entries that lack support across frozen experts.",
        expected_effect=ExpectedEffect(net_return="increase", turnover="decrease"),
        falsifiers=["delta_net_return_lte_0"],
        confidence=0.5,
    )


def _evaluation(candidate_id, delta, *, decision=None):
    base = MetricBundle(
        trades=10, long_trades=5, short_trades=5, net_return=0.01, sortino=1.0, sharpe=0.5,
        max_drawdown=0.02, turnover=10, monthly_gain_concentration=0.5,
    )
    candidate = base.model_copy(update={"net_return": 0.01 + delta, "sortino": 1.0 + delta})
    return EvaluationRecord(
        evaluation_id=f"ev-{candidate_id}", candidate_id=candidate_id, window_ids=["w1"],
        cutoff_utc=datetime(2025, 7, 1, tzinfo=UTC), baseline=base, candidate=candidate,
        delta_net_return=delta, paired_weekly_deltas=[delta], guard_results={"positive": delta > 0},
        decision=decision or ("historical_keep" if delta > 0 else "historical_prune"),
    )


def test_beam_is_deterministic_and_refiner_cannot_invent_ids():
    candidates = [_candidate(f"c{i}") for i in range(4)]
    evaluations = [_evaluation(f"c{i}", i / 100) for i in range(4)]
    assert [candidate.candidate_id for candidate in select_beam(candidates, evaluations)] == ["c3", "c2", "c1"]
    with pytest.raises(ValueError, match="invented"):
        validate_refinements(candidates[:3], RefinerBatch(candidates=[_candidate("new")]))


def test_one_episode_cannot_create_semantic_memory(tmp_path):
    store = AgentStore(tmp_path / "state.sqlite")
    memory = MemoryManager(store, protocol_hash="p")
    now = datetime(2025, 7, 20, tzinfo=UTC)
    reflection = ReflectionRecord(
        reflection_id="r1", candidate_id="c1", evaluation_id="e1", evidence=["positive delta"],
        speculation=[], generalized_lesson="Higher agreement helped in high volatility.",
        memory_recommendation="propose_semantic",
    )
    memory.store_episode(
        reflection=reflection, evaluation=_evaluation("c1", 0.01, decision="promote"),
        cutoff_utc=now, tags=["vol:high"],
    )
    assert memory.consolidate(now_utc=now) == []


def test_two_independent_episodes_consolidate_and_future_memory_is_not_retrieved(tmp_path):
    store = AgentStore(tmp_path / "state.sqlite")
    memory = MemoryManager(store, protocol_hash="p")
    now = datetime(2025, 7, 20, tzinfo=UTC)
    for index in (1, 2):
        reflection = ReflectionRecord(
            reflection_id=f"r{index}", candidate_id=f"c{index}", evaluation_id=f"e{index}",
            evidence=["positive delta"], speculation=[],
            generalized_lesson="Higher agreement helped in high volatility.",
            memory_recommendation="propose_semantic",
        )
        memory.store_episode(
            reflection=reflection, evaluation=_evaluation(f"c{index}", 0.01, decision="promote"),
            cutoff_utc=now + timedelta(days=index), tags=["vol:high"],
        )
    beliefs = memory.consolidate(now_utc=now + timedelta(days=3))
    assert len(beliefs) == 1
    before = memory.retrieve(cutoff_utc=now, tags=["vol:high"])
    after = memory.retrieve(cutoff_utc=now + timedelta(days=4), tags=["vol:high"])
    assert before == []
    assert any(item.memory_type == "semantic" for item in after)
    assert memory.retrieve(cutoff_utc=now + timedelta(days=4), tags=["trend:down"]) == []
