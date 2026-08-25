from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from reflection_agent.v2.contracts import MemoryCard
from reflection_agent.v2.memory import (
    NoMemory,
    RealMemory,
    SemanticSupport,
    ShuffledMemory,
    StaticPolicyControl,
    shuffled_memory,
)


T0 = datetime(2024, 1, 1, tzinfo=UTC)
SCOPE = "development_2021_2024"


def card(
    memory_id: str,
    *,
    created_at_utc: datetime,
    memory_type: str = "EPISODIC",
    tags: list[str] | None = None,
    expires_at_utc: datetime | None = None,
    expires_after_episode: int | None = None,
) -> MemoryCard:
    return MemoryCard(
        memory_id=memory_id,
        source_stage="development",
        protocol_scope=SCOPE,
        memory_type=memory_type,
        created_at_utc=created_at_utc,
        max_support_outcome_time=created_at_utc - timedelta(minutes=1),
        lesson=f"Causal lesson for {memory_id} from completed future-shadow evidence.",
        evidence_status="SUPPORTED" if memory_type == "SEMANTIC" else "REJECTED",
        tags=tags or [],
        source_evaluation_ids=[f"evaluation-{memory_id}"],
        expires_at_utc=expires_at_utc,
        expires_after_episode=expires_after_episode,
    )


def memory(tmp_path) -> RealMemory:
    return RealMemory(
        tmp_path / "agent_state.sqlite",
        protocol_hash="protocol-hash",
        protocol_scope=SCOPE,
    )


def test_real_memory_never_retrieves_future_or_same_cutoff(tmp_path) -> None:
    store = memory(tmp_path)
    store.store(card("past", created_at_utc=T0 - timedelta(minutes=1), tags=["side:SHORT"]))
    store.store(card("same", created_at_utc=T0, tags=["side:SHORT"]))
    store.store(card("future", created_at_utc=T0 + timedelta(minutes=1), tags=["side:SHORT"]))
    result = store.retrieve(
        cutoff_utc=T0,
        tags=("side:SHORT",),
        episode_number=1,
        episode_id="episode-1",
    )
    assert [item.memory_id for item in result] == ["past"]


def test_retrieval_ranks_overlap_then_recency_and_honours_both_expiries(tmp_path) -> None:
    store = memory(tmp_path)
    store.store(card("older-match", created_at_utc=T0 - timedelta(days=5), tags=["side:SHORT"]))
    store.store(card("newer-nonmatch", created_at_utc=T0 - timedelta(days=1), tags=["side:LONG"]))
    store.store(
        card(
            "expired-time",
            created_at_utc=T0 - timedelta(days=3),
            tags=["side:SHORT"],
            expires_at_utc=T0 - timedelta(hours=1),
        )
    )
    store.store(
        card(
            "expired-episode",
            created_at_utc=T0 - timedelta(days=2),
            tags=["side:SHORT"],
            expires_after_episode=4,
        )
    )
    result = store.retrieve(
        cutoff_utc=T0,
        tags=("side:SHORT",),
        episode_number=4,
        episode_id="episode-4",
        maximum=4,
    )
    assert [item.memory_id for item in result] == ["older-match", "newer-nonmatch"]


def support(
    evaluation_id: str,
    *,
    start: datetime,
    end: datetime,
) -> SemanticSupport:
    return SemanticSupport(
        evaluation_id=evaluation_id,
        candidate_id=f"candidate-{evaluation_id}",
        normalized_policy_key="ALLOW_REENTRY|side=SHORT|vol_regime=HIGH",
        source_stage="development",
        protocol_scope=SCOPE,
        fold_id=1,
        shadow_start_utc=start,
        shadow_end_utc=end,
        max_support_outcome_time=end,
        lesson="High-volatility SHORT re-entries retained positive net returns after costs.",
        tags=["side:SHORT", "vol:HIGH"],
        evaluator_decision="PROMOTE",
    )


def test_semantic_memory_requires_two_non_overlapping_promotions(tmp_path) -> None:
    store = memory(tmp_path)
    store.store_semantic_support(
        support("one", start=T0 - timedelta(days=30), end=T0 - timedelta(days=25))
    )
    assert store.consolidate(now_utc=T0, current_episode_number=10) == []
    store.store_semantic_support(
        support("overlap", start=T0 - timedelta(days=28), end=T0 - timedelta(days=24))
    )
    assert store.consolidate(now_utc=T0, current_episode_number=10) == []
    store.store_semantic_support(
        support("two", start=T0 - timedelta(days=20), end=T0 - timedelta(days=15))
    )
    semantic = store.consolidate(now_utc=T0, current_episode_number=10)
    assert len(semantic) == 1
    assert semantic[0].memory_type == "SEMANTIC"
    assert semantic[0].expires_at_utc == T0 + timedelta(days=180)
    assert semantic[0].expires_after_episode == 16
    assert set(semantic[0].source_evaluation_ids) >= {"one", "two"}


def test_contradiction_is_append_only_and_removes_semantic_from_retrieval(tmp_path) -> None:
    store = memory(tmp_path)
    semantic = card(
        "semantic-1",
        created_at_utc=T0 - timedelta(days=2),
        memory_type="SEMANTIC",
        tags=["side:SHORT"],
        expires_at_utc=T0 + timedelta(days=10),
        expires_after_episode=10,
    )
    store.store(semantic)
    store.contradict(
        memory_id="semantic-1",
        event_id="contradiction-1",
        contradicted_at_utc=T0 - timedelta(days=1),
        evaluation_id="evaluation-counterexample",
    )
    assert [item.memory_id for item in store.all_cards()] == ["semantic-1"]
    assert store.retrieve(
        cutoff_utc=T0,
        tags=("side:SHORT",),
        episode_number=3,
        episode_id="episode-3",
    ) == []
    assert store.event_count("semantic-1") == 1


def test_shuffle_is_deterministic_causal_and_preserves_card_multiset() -> None:
    cards = [
        card(f"memory-{index}", created_at_utc=T0 - timedelta(days=index + 1), tags=[f"tag:{index}"])
        for index in range(4)
    ]
    first = shuffled_memory(cards, episode_id="episode-4", protocol_hash="p")
    second = shuffled_memory(cards, episode_id="episode-4", protocol_hash="p")
    assert first == second
    assert sorted(item.memory_id for item in first) == sorted(item.memory_id for item in cards)
    assert sorted(item.created_at_utc for item in first) == sorted(
        item.created_at_utc for item in cards
    )
    assert [item.tags for item in first] != [item.tags for item in cards]
    assert max(item.created_at_utc for item in first) < T0


def test_registered_controls_retrieve_without_cross_contamination(tmp_path) -> None:
    real = memory(tmp_path)
    for index in range(4):
        real.store(
            card(
                f"memory-{index}",
                created_at_utc=T0 - timedelta(days=index + 1),
                tags=[f"tag:{index}"],
            )
        )
    no_memory = NoMemory(real)
    shuffled = ShuffledMemory(real)
    static = StaticPolicyControl(real)
    kwargs = {
        "cutoff_utc": T0,
        "tags": ("tag:0",),
        "episode_number": 5,
        "episode_id": "episode-5",
        "maximum": 4,
    }
    real_cards = real.retrieve(**kwargs)
    assert no_memory.retrieve(**kwargs) == []
    shuffled_cards = shuffled.retrieve(**kwargs)
    assert sorted(item.memory_id for item in shuffled_cards) == sorted(
        item.memory_id for item in real_cards
    )
    assert [item.memory_id for item in shuffled_cards] != [
        item.memory_id for item in real_cards
    ]
    assert static.retrieve(**kwargs) == []
    assert static.static_add_all is True
    assert static.uses_llm is False
    assert real.retrieval_audit_count() == 4


def test_store_is_idempotent_but_conflicting_rewrite_fails(tmp_path) -> None:
    store = memory(tmp_path)
    original = card("memory-1", created_at_utc=T0 - timedelta(days=1))
    store.store(original)
    store.store(original)
    with pytest.raises(ValueError, match="conflicting append-only write"):
        store.store(
            card(
                "memory-1",
                created_at_utc=T0 - timedelta(days=1),
                tags=["changed"],
            )
        )
