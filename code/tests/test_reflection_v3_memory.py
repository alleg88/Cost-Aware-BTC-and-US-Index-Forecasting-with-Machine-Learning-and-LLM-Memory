from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reflection_agent.v3.contracts import MemoryCard
from reflection_agent.v3.memory import (
    NoMemory,
    RealMemory,
    SemanticSupport,
    ShuffledMemory,
    StaticPolicyControl,
    continuous_stage_rank,
)


UTC = timezone.utc
DEV_T1 = datetime(2024, 12, 20, tzinfo=UTC)
DEV_T2 = datetime(2024, 12, 30, tzinfo=UTC)
H1_T1 = datetime(2025, 1, 10, tzinfo=UTC)
FORWARD_T2 = datetime(2025, 8, 1, tzinfo=UTC)


def _card(
    memory_id: str = "memory_a1",
    *,
    source_stage: str = "development",
    created_at_utc: datetime = DEV_T2,
    tags: tuple[str, ...] = ("side:SHORT", "tier:HIGH_EXTRA"),
    expires_at_utc: datetime | None = None,
    expires_after_episode: int | None = None,
) -> MemoryCard:
    return MemoryCard(
        memory_id=memory_id,
        source_stage=source_stage,
        protocol_scope="continuous_2021_2026",
        memory_type="EPISODIC",
        created_at_utc=created_at_utc,
        max_support_outcome_time=created_at_utc - timedelta(minutes=1),
        lesson="A resolved categorical policy lesson.",
        evidence_status="SUPPORTED",
        tags=list(tags),
        source_evaluation_ids=[f"eval_{memory_id}"],
        expires_at_utc=expires_at_utc,
        expires_after_episode=expires_after_episode,
    )


def _retrieve(
    store,
    *,
    cutoff: datetime,
    stage: str,
    tags: tuple[str, ...] = ("side:SHORT",),
    episode_number: int = 1,
    episode_id: str = "episode_a1",
):
    return store.retrieve(
        cutoff_utc=cutoff,
        stage=stage,
        tags=tags,
        episode_number=episode_number,
        episode_id=episode_id,
    )


def test_development_memory_is_retrievable_in_h1_only_after_support(tmp_path) -> None:
    store = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    store.store(_card())
    assert _retrieve(store, cutoff=DEV_T1, stage="development") == []
    retrieved = _retrieve(store, cutoff=H1_T1, stage="h1")
    assert retrieved[0].source_stage == "development"


def test_forward_memory_can_never_enter_h1_replay(tmp_path) -> None:
    store = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    store.store(
        _card(
            "forward_memory",
            source_stage="forward",
            created_at_utc=FORWARD_T2,
        )
    )
    assert _retrieve(store, cutoff=H1_T1, stage="h1", tags=()) == []
    assert [continuous_stage_rank(stage) for stage in ("development", "h1", "forward")] == [
        0,
        1,
        2,
    ]


def test_memory_store_is_append_only_and_idempotent(tmp_path) -> None:
    store = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    card = _card()
    store.store(card)
    store.store(card)
    assert len(store.all_cards()) == 1
    with pytest.raises(ValueError, match="conflicting append-only"):
        store.store(_card(tags=("side:LONG",)))


def test_calendar_and_episode_expiry_are_both_enforced(tmp_path) -> None:
    store = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    created = datetime(2024, 1, 1, tzinfo=UTC)
    store.store(
        _card(
            created_at_utc=created,
            expires_at_utc=created + timedelta(days=180),
            expires_after_episode=7,
        )
    )
    assert _retrieve(
        store,
        cutoff=created + timedelta(days=100),
        stage="development",
        episode_number=6,
    )
    assert _retrieve(
        store,
        cutoff=created + timedelta(days=100),
        stage="development",
        episode_number=7,
    ) == []
    assert _retrieve(
        store,
        cutoff=created + timedelta(days=180),
        stage="development",
        episode_number=6,
    ) == []


def _support(
    evaluation_id: str,
    *,
    start: datetime,
    end: datetime,
) -> SemanticSupport:
    return SemanticSupport(
        evaluation_id=evaluation_id,
        candidate_id=f"candidate_{evaluation_id}",
        normalized_policy_key="side=SHORT|confidence_tier=HIGH_EXTRA",
        source_stage="development",
        protocol_scope="continuous_2021_2026",
        fold_id=0,
        shadow_start_utc=start,
        shadow_end_utc=end,
        max_support_outcome_time=end,
        lesson="High confidence short candidates retained support.",
        tags=["side:SHORT", "tier:HIGH_EXTRA"],
        evaluator_decision="PROMOTE",
    )


def test_semantic_memory_requires_two_non_overlapping_future_shadows(tmp_path) -> None:
    store = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    store.store_semantic_support(
        _support("eval_1", start=start, end=start + timedelta(days=2))
    )
    store.store_semantic_support(
        _support(
            "eval_2",
            start=start + timedelta(days=1),
            end=start + timedelta(days=3),
        )
    )
    assert store.consolidate(
        now_utc=start + timedelta(days=4), current_episode_number=3
    ) == []
    store.store_semantic_support(
        _support(
            "eval_3",
            start=start + timedelta(days=3),
            end=start + timedelta(days=4),
        )
    )
    semantic = store.consolidate(
        now_utc=start + timedelta(days=5), current_episode_number=4
    )
    assert len(semantic) == 1
    assert semantic[0].memory_type == "SEMANTIC"
    assert semantic[0].expires_at_utc == start + timedelta(days=185)
    assert semantic[0].expires_after_episode == 10


def test_contradiction_is_retained_but_not_retrieved(tmp_path) -> None:
    store = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    card = _card()
    store.store(card)
    store.contradict(
        memory_id=card.memory_id,
        event_id="event_a1",
        contradicted_at_utc=DEV_T2 + timedelta(days=1),
        evaluation_id="eval_later",
    )
    assert len(store.all_cards()) == 1
    assert store.event_count(card.memory_id) == 1
    assert _retrieve(store, cutoff=H1_T1, stage="h1") == []


def test_no_memory_logs_the_causal_eligible_count(tmp_path) -> None:
    real = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    real.store(_card())
    control = NoMemory(real)
    assert _retrieve(control, cutoff=H1_T1, stage="h1") == []
    audit = real.retrieval_audits()[-1]
    assert audit["variant"] == "NONE"
    assert audit["eligible_count"] == 1
    assert audit["returned_memory_ids"] == []


def test_shuffled_memory_is_causal_deterministic_and_age_matched(tmp_path) -> None:
    real = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    for index, tags in enumerate(
        [
            ("side:SHORT", "tier:HIGH_EXTRA"),
            ("side:LONG", "tier:LOW_EXTRA"),
            ("side:LONG", "tier:MID_EXTRA"),
            ("side:SHORT", "tier:MID_EXTRA"),
        ]
    ):
        real.store(
            _card(
                f"memory_{index}",
                created_at_utc=DEV_T2 - timedelta(days=index),
                tags=tags,
            )
        )
    control = ShuffledMemory(real)
    first = _retrieve(
        control,
        cutoff=H1_T1,
        stage="h1",
        tags=("side:SHORT", "tier:HIGH_EXTRA"),
        episode_id="episode_shuffle",
    )
    second = _retrieve(
        control,
        cutoff=H1_T1,
        stage="h1",
        tags=("side:SHORT", "tier:HIGH_EXTRA"),
        episode_id="episode_shuffle",
    )
    assert [card.memory_id for card in first] == [card.memory_id for card in second]
    assert len(first) == 4
    assert all(card.created_at_utc < H1_T1 for card in first)
    audit = real.retrieval_audits()[-1]
    assert audit["returned_age_buckets"] == audit["real_age_buckets"]


def test_static_control_has_no_proposal_interface(tmp_path) -> None:
    real = RealMemory(tmp_path / "state.sqlite", protocol_hash="p")
    control = StaticPolicyControl(real, static_variant="static_high_extra")
    assert control.uses_llm is False
    assert _retrieve(control, cutoff=H1_T1, stage="h1") == []
    with pytest.raises(RuntimeError, match="no proposal interface"):
        control.propose()

