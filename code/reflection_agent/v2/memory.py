"""Append-only causal memory and registered memory-policy controls."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Iterable, Literal

from pydantic import Field, model_validator

from reflection_agent.v2.contracts import MemoryCard, StrictModel


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


class SemanticSupport(StrictModel):
    evaluation_id: Annotated[str, Field(min_length=1, max_length=128)]
    candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    normalized_policy_key: Annotated[str, Field(min_length=1, max_length=400)]
    source_stage: Literal["development", "h1", "forward"]
    protocol_scope: Annotated[str, Field(min_length=1, max_length=128)]
    fold_id: Annotated[int, Field(ge=0)]
    shadow_start_utc: datetime
    shadow_end_utc: datetime
    max_support_outcome_time: datetime
    lesson: Annotated[str, Field(min_length=10, max_length=400)]
    tags: Annotated[list[str], Field(max_length=16)] = Field(default_factory=list)
    evaluator_decision: Literal["PROMOTE"]

    @model_validator(mode="after")
    def interval_is_causal(self) -> "SemanticSupport":
        values = (
            self.shadow_start_utc,
            self.shadow_end_utc,
            self.max_support_outcome_time,
        )
        if any(value.tzinfo is None for value in values):
            raise ValueError("semantic support timestamps must be timezone-aware")
        if self.shadow_start_utc >= self.shadow_end_utc:
            raise ValueError("semantic support interval must be increasing")
        if self.max_support_outcome_time > self.shadow_end_utc:
            raise ValueError("semantic support outcome exceeds its shadow interval")
        return self


class RealMemory:
    """SQLite-backed source of truth for one variant and protocol scope."""

    memory_variant = "REAL"
    static_add_all = False
    uses_llm = True

    def __init__(
        self,
        path: str | Path,
        *,
        protocol_hash: str,
        protocol_scope: str,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.protocol_hash = protocol_hash
        self.protocol_scope = protocol_scope
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_cards (
                    memory_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    protocol_scope TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS semantic_supports (
                    evaluation_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    protocol_scope TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    memory_id TEXT NOT NULL,
                    protocol_hash TEXT NOT NULL,
                    protocol_scope TEXT NOT NULL,
                    event_time_utc TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS retrieval_audits (
                    retrieval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    protocol_hash TEXT NOT NULL,
                    protocol_scope TEXT NOT NULL,
                    cutoff_utc TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )

    def store(self, card: MemoryCard) -> None:
        if card.protocol_scope != self.protocol_scope:
            raise ValueError("memory card belongs to a different protocol scope")
        payload = _canonical_json(card.model_dump(mode="json"))
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT protocol_hash, protocol_scope, payload_json FROM memory_cards "
                "WHERE memory_id = ?",
                (card.memory_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["protocol_hash"] != self.protocol_hash
                    or existing["protocol_scope"] != self.protocol_scope
                    or existing["payload_json"] != payload
                ):
                    raise ValueError(
                        f"conflicting append-only write for memory {card.memory_id}"
                    )
                return
            connection.execute(
                "INSERT INTO memory_cards "
                "(memory_id, protocol_hash, protocol_scope, created_at_utc, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    card.memory_id,
                    self.protocol_hash,
                    self.protocol_scope,
                    card.created_at_utc.isoformat(),
                    payload,
                ),
            )

    def all_cards(self) -> list[MemoryCard]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM memory_cards "
                "WHERE protocol_hash = ? AND protocol_scope = ? ORDER BY memory_id",
                (self.protocol_hash, self.protocol_scope),
            ).fetchall()
        return [MemoryCard.model_validate_json(row["payload_json"]) for row in rows]

    def _contradicted_ids(self, cutoff_utc: datetime) -> set[str]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT memory_id, payload_json FROM memory_events "
                "WHERE protocol_hash = ? AND protocol_scope = ? AND event_time_utc <= ?",
                (self.protocol_hash, self.protocol_scope, cutoff_utc.isoformat()),
            ).fetchall()
        contradicted: set[str] = set()
        for row in rows:
            if json.loads(row["payload_json"]).get("event_type") == "CONTRADICTED":
                contradicted.add(row["memory_id"])
        return contradicted

    def _eligible_cards(
        self,
        *,
        cutoff_utc: datetime,
        episode_number: int,
    ) -> list[MemoryCard]:
        if cutoff_utc.tzinfo is None:
            raise ValueError("memory cutoff must be timezone-aware")
        contradicted = self._contradicted_ids(cutoff_utc)
        eligible: list[MemoryCard] = []
        for card in self.all_cards():
            if card.created_at_utc >= cutoff_utc:
                continue
            if card.memory_type == "WORKING":
                continue
            if card.memory_id in contradicted:
                continue
            if card.expires_at_utc is not None and card.expires_at_utc <= cutoff_utc:
                continue
            if (
                card.expires_after_episode is not None
                and episode_number >= card.expires_after_episode
            ):
                continue
            eligible.append(card)
        return eligible

    @staticmethod
    def _rank(
        cards: Iterable[MemoryCard], tags: Iterable[str]
    ) -> list[MemoryCard]:
        requested = set(tags)
        scored = [
            (len(requested.intersection(card.tags)), card.created_at_utc, card.memory_id, card)
            for card in cards
        ]
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[-1] for item in scored]

    def _log_retrieval(
        self,
        *,
        cutoff_utc: datetime,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int,
        variant: str,
        eligible: list[MemoryCard],
        returned: list[MemoryCard],
    ) -> None:
        payload = {
            "episode_id": episode_id,
            "episode_number": episode_number,
            "tags": sorted(set(tags)),
            "maximum": maximum,
            "eligible_memory_ids": sorted(card.memory_id for card in eligible),
            "returned_memory_ids": [card.memory_id for card in returned],
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO retrieval_audits "
                "(protocol_hash, protocol_scope, cutoff_utc, variant, payload_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    self.protocol_hash,
                    self.protocol_scope,
                    cutoff_utc.isoformat(),
                    variant,
                    _canonical_json(payload),
                ),
            )

    def retrieve(
        self,
        *,
        cutoff_utc: datetime,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int = 4,
    ) -> list[MemoryCard]:
        eligible = self._eligible_cards(
            cutoff_utc=cutoff_utc, episode_number=episode_number
        )
        returned = self._rank(eligible, tags)[:maximum]
        self._log_retrieval(
            cutoff_utc=cutoff_utc,
            tags=tags,
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=maximum,
            variant=self.memory_variant,
            eligible=eligible,
            returned=returned,
        )
        return returned

    def store_semantic_support(self, support: SemanticSupport) -> None:
        if support.protocol_scope != self.protocol_scope:
            raise ValueError("semantic support belongs to a different protocol scope")
        payload = _canonical_json(support.model_dump(mode="json"))
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT protocol_hash, protocol_scope, payload_json FROM semantic_supports "
                "WHERE evaluation_id = ?",
                (support.evaluation_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["protocol_hash"] != self.protocol_hash
                    or existing["protocol_scope"] != self.protocol_scope
                    or existing["payload_json"] != payload
                ):
                    raise ValueError(
                        f"conflicting semantic support {support.evaluation_id}"
                    )
                return
            connection.execute(
                "INSERT INTO semantic_supports "
                "(evaluation_id, protocol_hash, protocol_scope, payload_json) "
                "VALUES (?, ?, ?, ?)",
                (
                    support.evaluation_id,
                    self.protocol_hash,
                    self.protocol_scope,
                    payload,
                ),
            )

    def _semantic_supports(self) -> list[SemanticSupport]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT payload_json FROM semantic_supports "
                "WHERE protocol_hash = ? AND protocol_scope = ? ORDER BY evaluation_id",
                (self.protocol_hash, self.protocol_scope),
            ).fetchall()
        return [SemanticSupport.model_validate_json(row["payload_json"]) for row in rows]

    @staticmethod
    def _has_non_overlapping_pair(supports: list[SemanticSupport]) -> bool:
        for left_index, left in enumerate(supports):
            for right in supports[left_index + 1 :]:
                if (
                    left.shadow_end_utc <= right.shadow_start_utc
                    or right.shadow_end_utc <= left.shadow_start_utc
                ):
                    return True
        return False

    def consolidate(
        self,
        *,
        now_utc: datetime,
        current_episode_number: int,
    ) -> list[MemoryCard]:
        if now_utc.tzinfo is None:
            raise ValueError("semantic consolidation time must be timezone-aware")
        grouped: dict[tuple[str, str], list[SemanticSupport]] = {}
        for support in self._semantic_supports():
            normalized_lesson = " ".join(support.lesson.lower().split())
            grouped.setdefault(
                (support.normalized_policy_key, normalized_lesson), []
            ).append(support)
        consolidated: list[MemoryCard] = []
        for (policy_key, normalized_lesson), supports in sorted(grouped.items()):
            if len({support.evaluation_id for support in supports}) < 2:
                continue
            if not self._has_non_overlapping_pair(supports):
                continue
            maximum_outcome = max(
                support.max_support_outcome_time for support in supports
            )
            if maximum_outcome > now_utc:
                raise ValueError("semantic memory would precede supporting outcomes")
            identity = hashlib.sha256(
                _canonical_json(
                    {"policy_key": policy_key, "lesson": normalized_lesson}
                ).encode("utf-8")
            ).hexdigest()[:24]
            evaluation_ids = sorted(
                {support.evaluation_id for support in supports}
            )[:8]
            semantic = MemoryCard(
                memory_id=f"semantic-{identity}",
                source_stage=max(
                    supports, key=lambda support: support.shadow_end_utc
                ).source_stage,
                protocol_scope=self.protocol_scope,
                memory_type="SEMANTIC",
                created_at_utc=now_utc,
                max_support_outcome_time=maximum_outcome,
                lesson=supports[0].lesson,
                evidence_status="SUPPORTED",
                tags=sorted({tag for support in supports for tag in support.tags})[:16],
                source_evaluation_ids=evaluation_ids,
                expires_at_utc=now_utc + timedelta(days=180),
                expires_after_episode=current_episode_number + 6,
            )
            self.store(semantic)
            consolidated.append(semantic)
        return consolidated

    def contradict(
        self,
        *,
        memory_id: str,
        event_id: str,
        contradicted_at_utc: datetime,
        evaluation_id: str,
    ) -> None:
        cards = {card.memory_id: card for card in self.all_cards()}
        if memory_id not in cards:
            raise KeyError(f"unknown memory: {memory_id}")
        if contradicted_at_utc.tzinfo is None:
            raise ValueError("contradiction time must be timezone-aware")
        if contradicted_at_utc < cards[memory_id].created_at_utc:
            raise ValueError("contradiction predates memory creation")
        payload = _canonical_json(
            {
                "event_type": "CONTRADICTED",
                "memory_id": memory_id,
                "evaluation_id": evaluation_id,
                "contradicted_at_utc": contradicted_at_utc.isoformat(),
            }
        )
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT payload_json FROM memory_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_json"] != payload:
                    raise ValueError(f"conflicting memory event {event_id}")
                return
            connection.execute(
                "INSERT INTO memory_events "
                "(event_id, memory_id, protocol_hash, protocol_scope, event_time_utc, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    memory_id,
                    self.protocol_hash,
                    self.protocol_scope,
                    contradicted_at_utc.isoformat(),
                    payload,
                ),
            )

    def event_count(self, memory_id: str) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM memory_events "
                    "WHERE memory_id = ? AND protocol_hash = ? AND protocol_scope = ?",
                    (memory_id, self.protocol_hash, self.protocol_scope),
                ).fetchone()[0]
            )

    def retrieval_audit_count(self) -> int:
        with self._connect() as connection:
            return int(
                connection.execute(
                    "SELECT COUNT(*) FROM retrieval_audits "
                    "WHERE protocol_hash = ? AND protocol_scope = ?",
                    (self.protocol_hash, self.protocol_scope),
                ).fetchone()[0]
            )


def shuffled_memory(
    cards: list[MemoryCard],
    *,
    episode_id: str,
    protocol_hash: str,
) -> list[MemoryCard]:
    """Deterministically reorder only the already causal eligible card multiset."""

    if len(cards) <= 1:
        return list(cards)
    digest = hashlib.sha256(f"{protocol_hash}|{episode_id}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % (len(cards) - 1) + 1
    return list(cards[offset:]) + list(cards[:offset])


class _MemoryControl:
    memory_variant = "NONE"
    static_add_all = False
    uses_llm = True

    def __init__(self, real: RealMemory) -> None:
        self.real = real

    def __getattr__(self, name: str):
        return getattr(self.real, name)


class NoMemory(_MemoryControl):
    memory_variant = "NONE"

    def retrieve(
        self,
        *,
        cutoff_utc: datetime,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int = 4,
    ) -> list[MemoryCard]:
        eligible = self.real._eligible_cards(
            cutoff_utc=cutoff_utc, episode_number=episode_number
        )
        self.real._log_retrieval(
            cutoff_utc=cutoff_utc,
            tags=tags,
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=maximum,
            variant=self.memory_variant,
            eligible=eligible,
            returned=[],
        )
        return []


class ShuffledMemory(_MemoryControl):
    memory_variant = "SHUFFLED"

    def retrieve(
        self,
        *,
        cutoff_utc: datetime,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int = 4,
    ) -> list[MemoryCard]:
        eligible = self.real._eligible_cards(
            cutoff_utc=cutoff_utc, episode_number=episode_number
        )
        shuffled = shuffled_memory(
            eligible, episode_id=episode_id, protocol_hash=self.real.protocol_hash
        )
        returned = shuffled[: min(maximum, len(eligible))]
        self.real._log_retrieval(
            cutoff_utc=cutoff_utc,
            tags=tags,
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=maximum,
            variant=self.memory_variant,
            eligible=eligible,
            returned=returned,
        )
        return returned


class StaticPolicyControl(NoMemory):
    memory_variant = "STATIC"
    static_add_all = True
    uses_llm = False


__all__ = [
    "NoMemory",
    "RealMemory",
    "SemanticSupport",
    "ShuffledMemory",
    "StaticPolicyControl",
    "shuffled_memory",
]
