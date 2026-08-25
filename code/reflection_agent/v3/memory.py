"""Continuous append-only causal memory and registered v3 controls."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Iterable, Literal

from pydantic import Field, model_validator

from reflection_agent.v3.contracts import MemoryCard, StrictModel


PROTOCOL_SCOPE = "continuous_2021_2026"
_STAGE_RANK = {"development": 0, "h1": 1, "forward": 2}


def continuous_stage_rank(stage: str) -> int:
    try:
        return _STAGE_RANK[stage]
    except KeyError as exc:
        raise ValueError(f"unknown continuous stage: {stage}") from exc


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _age_bucket(card: MemoryCard, cutoff_utc: datetime) -> int:
    age = max(cutoff_utc - card.created_at_utc, timedelta(0))
    return int(age.days // 30)


class SemanticSupport(StrictModel):
    evaluation_id: Annotated[str, Field(min_length=1, max_length=128)]
    candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    normalized_policy_key: Annotated[str, Field(min_length=1, max_length=400)]
    source_stage: Literal["development", "h1", "forward"]
    protocol_scope: Literal[PROTOCOL_SCOPE]
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
    """One SQLite source of truth spanning development, H1, and forward."""

    memory_variant = "REAL"
    uses_llm = True

    def __init__(
        self,
        path: str | Path,
        *,
        protocol_hash: str,
        protocol_scope: str = PROTOCOL_SCOPE,
    ) -> None:
        if not protocol_hash:
            raise ValueError("protocol_hash must not be empty")
        if protocol_scope != PROTOCOL_SCOPE:
            raise ValueError("v3 memory requires the continuous protocol scope")
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
                CREATE TABLE IF NOT EXISTS state_meta (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    protocol_hash TEXT NOT NULL,
                    protocol_scope TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_cards (
                    memory_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    protocol_scope TEXT NOT NULL,
                    source_stage TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    max_support_outcome_time TEXT NOT NULL,
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
                    stage TEXT NOT NULL,
                    variant TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS policy_history (
                    event_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS episode_audit (
                    event_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS call_audit (
                    event_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stage_checkpoints (
                    stage TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                """
            )
            existing = connection.execute(
                "SELECT protocol_hash, protocol_scope FROM state_meta WHERE singleton = 1"
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO state_meta (singleton, protocol_hash, protocol_scope) "
                    "VALUES (1, ?, ?)",
                    (self.protocol_hash, self.protocol_scope),
                )
            elif (
                existing["protocol_hash"] != self.protocol_hash
                or existing["protocol_scope"] != self.protocol_scope
            ):
                raise ValueError("state database belongs to a different protocol")

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
                "(memory_id, protocol_hash, protocol_scope, source_stage, "
                "created_at_utc, max_support_outcome_time, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    card.memory_id,
                    self.protocol_hash,
                    self.protocol_scope,
                    card.source_stage,
                    card.created_at_utc.isoformat(),
                    card.max_support_outcome_time.isoformat(),
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
                "WHERE protocol_hash = ? AND protocol_scope = ? AND event_time_utc < ?",
                (self.protocol_hash, self.protocol_scope, cutoff_utc.isoformat()),
            ).fetchall()
        return {
            row["memory_id"]
            for row in rows
            if json.loads(row["payload_json"]).get("event_type") == "CONTRADICTED"
        }

    def _eligible_cards(
        self,
        *,
        cutoff_utc: datetime,
        stage: str,
        episode_number: int,
    ) -> list[MemoryCard]:
        if cutoff_utc.tzinfo is None:
            raise ValueError("memory cutoff must be timezone-aware")
        stage_rank = continuous_stage_rank(stage)
        contradicted = self._contradicted_ids(cutoff_utc)
        eligible: list[MemoryCard] = []
        for card in self.all_cards():
            if continuous_stage_rank(card.source_stage) > stage_rank:
                continue
            if (
                card.created_at_utc >= cutoff_utc
                or card.max_support_outcome_time >= cutoff_utc
                or card.max_support_outcome_time > card.created_at_utc
            ):
                continue
            if card.memory_type == "WORKING" or card.memory_id in contradicted:
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
    def _rank(cards: Iterable[MemoryCard], tags: Iterable[str]) -> list[MemoryCard]:
        requested = set(tags)
        scored = [
            (
                len(requested.intersection(card.tags)),
                card.created_at_utc,
                card.memory_id,
                card,
            )
            for card in cards
        ]
        scored.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[-1] for item in scored]

    def _log_retrieval(
        self,
        *,
        cutoff_utc: datetime,
        stage: str,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int,
        variant: str,
        eligible: list[MemoryCard],
        real_returned: list[MemoryCard],
        returned: list[MemoryCard],
    ) -> None:
        payload = {
            "episode_id": episode_id,
            "episode_number": episode_number,
            "tags": sorted(set(tags)),
            "maximum": maximum,
            "eligible_count": len(eligible),
            "eligible_memory_ids": sorted(card.memory_id for card in eligible),
            "real_memory_ids": [card.memory_id for card in real_returned],
            "returned_memory_ids": [card.memory_id for card in returned],
            "real_age_buckets": sorted(
                _age_bucket(card, cutoff_utc) for card in real_returned
            ),
            "returned_age_buckets": sorted(
                _age_bucket(card, cutoff_utc) for card in returned
            ),
        }
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO retrieval_audits "
                "(protocol_hash, protocol_scope, cutoff_utc, stage, variant, payload_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    self.protocol_hash,
                    self.protocol_scope,
                    cutoff_utc.isoformat(),
                    stage,
                    variant,
                    _canonical_json(payload),
                ),
            )

    def retrieve(
        self,
        *,
        cutoff_utc: datetime,
        stage: str,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int = 4,
    ) -> list[MemoryCard]:
        if not 0 <= maximum <= 4:
            raise ValueError("at most four memories may be retrieved")
        eligible = self._eligible_cards(
            cutoff_utc=cutoff_utc, stage=stage, episode_number=episode_number
        )
        returned = self._rank(eligible, tags)[:maximum]
        self._log_retrieval(
            cutoff_utc=cutoff_utc,
            stage=stage,
            tags=tags,
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=maximum,
            variant=self.memory_variant,
            eligible=eligible,
            real_returned=returned,
            returned=returned,
        )
        return returned

    def retrieval_audits(self) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT cutoff_utc, stage, variant, payload_json FROM retrieval_audits "
                "WHERE protocol_hash = ? AND protocol_scope = ? ORDER BY retrieval_id",
                (self.protocol_hash, self.protocol_scope),
            ).fetchall()
        output: list[dict[str, object]] = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload.update(
                {
                    "cutoff_utc": row["cutoff_utc"],
                    "stage": row["stage"],
                    "variant": row["variant"],
                }
            )
            output.append(payload)
        return output

    def store_semantic_support(self, support: SemanticSupport) -> None:
        if support.protocol_scope != self.protocol_scope:
            raise ValueError("semantic support belongs to a different scope")
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
        self, *, now_utc: datetime, current_episode_number: int
    ) -> list[MemoryCard]:
        if now_utc.tzinfo is None:
            raise ValueError("semantic consolidation time must be timezone-aware")
        grouped: dict[tuple[str, str], list[SemanticSupport]] = {}
        for support in self._semantic_supports():
            normalized_lesson = " ".join(support.lesson.lower().split())
            grouped.setdefault(
                (support.normalized_policy_key, normalized_lesson), []
            ).append(support)
        existing_ids = {card.memory_id for card in self.all_cards()}
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
            memory_id = f"semantic-{identity}"
            if memory_id in existing_ids:
                continue
            semantic = MemoryCard(
                memory_id=memory_id,
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
                source_evaluation_ids=sorted(
                    {support.evaluation_id for support in supports}
                )[:8],
                expires_at_utc=now_utc + timedelta(days=180),
                expires_after_episode=current_episode_number + 6,
            )
            self.store(semantic)
            existing_ids.add(memory_id)
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
                    "SELECT COUNT(*) FROM memory_events WHERE memory_id = ? "
                    "AND protocol_hash = ? AND protocol_scope = ?",
                    (memory_id, self.protocol_hash, self.protocol_scope),
                ).fetchone()[0]
            )

    def _append_record(self, table: str, event_id: str, payload: object) -> None:
        if table not in {"policy_history", "episode_audit", "call_audit"}:
            raise ValueError("unknown append-only audit table")
        encoded = _canonical_json(payload)
        with self._connect() as connection:
            existing = connection.execute(
                f"SELECT protocol_hash, payload_json FROM {table} WHERE event_id = ?",
                (event_id,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["protocol_hash"] != self.protocol_hash
                    or existing["payload_json"] != encoded
                ):
                    raise ValueError(f"conflicting append-only {table} record {event_id}")
                return
            connection.execute(
                f"INSERT INTO {table} (event_id, protocol_hash, payload_json) "
                "VALUES (?, ?, ?)",
                (event_id, self.protocol_hash, encoded),
            )

    def append_policy_history(self, event_id: str, payload: object) -> None:
        self._append_record("policy_history", event_id, payload)

    def append_episode_audit(self, event_id: str, payload: object) -> None:
        self._append_record("episode_audit", event_id, payload)

    def append_call_audit(self, event_id: str, payload: object) -> None:
        self._append_record("call_audit", event_id, payload)

    def checkpoint_stage(self, stage: str, payload: object) -> None:
        continuous_stage_rank(stage)
        encoded = _canonical_json(payload)
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT protocol_hash, payload_json FROM stage_checkpoints WHERE stage = ?",
                (stage,),
            ).fetchone()
            if existing is not None:
                if (
                    existing["protocol_hash"] != self.protocol_hash
                    or existing["payload_json"] != encoded
                ):
                    raise ValueError(f"conflicting stage checkpoint: {stage}")
                return
            connection.execute(
                "INSERT INTO stage_checkpoints (stage, protocol_hash, payload_json) "
                "VALUES (?, ?, ?)",
                (stage, self.protocol_hash, encoded),
            )


class _MemoryControl:
    memory_variant = "NONE"
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
        stage: str,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int = 4,
    ) -> list[MemoryCard]:
        eligible = self.real._eligible_cards(
            cutoff_utc=cutoff_utc, stage=stage, episode_number=episode_number
        )
        real_returned = self.real._rank(eligible, tags)[:maximum]
        self.real._log_retrieval(
            cutoff_utc=cutoff_utc,
            stage=stage,
            tags=tags,
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=maximum,
            variant=self.memory_variant,
            eligible=eligible,
            real_returned=real_returned,
            returned=[],
        )
        return []


class ShuffledMemory(_MemoryControl):
    memory_variant = "SHUFFLED"

    def retrieve(
        self,
        *,
        cutoff_utc: datetime,
        stage: str,
        tags: Iterable[str],
        episode_number: int,
        episode_id: str,
        maximum: int = 4,
    ) -> list[MemoryCard]:
        eligible = self.real._eligible_cards(
            cutoff_utc=cutoff_utc, stage=stage, episode_number=episode_number
        )
        real_returned = self.real._rank(eligible, tags)[:maximum]
        requested = set(tags)
        needed = Counter(_age_bucket(card, cutoff_utc) for card in real_returned)
        selected: list[MemoryCard] = []
        for bucket, count in sorted(needed.items()):
            candidates = [
                card
                for card in eligible
                if _age_bucket(card, cutoff_utc) == bucket
                and card.memory_id not in {item.memory_id for item in selected}
            ]
            candidates.sort(
                key=lambda card: (
                    len(requested.intersection(card.tags)),
                    hashlib.sha256(
                        f"{self.real.protocol_hash}|{episode_id}|{card.memory_id}".encode(
                            "utf-8"
                        )
                    ).hexdigest(),
                )
            )
            selected.extend(candidates[:count])
        returned = selected[:maximum]
        self.real._log_retrieval(
            cutoff_utc=cutoff_utc,
            stage=stage,
            tags=tags,
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=maximum,
            variant=self.memory_variant,
            eligible=eligible,
            real_returned=real_returned,
            returned=returned,
        )
        return returned


class StaticPolicyControl(NoMemory):
    memory_variant = "STATIC"
    uses_llm = False

    def __init__(
        self,
        real: RealMemory,
        *,
        static_variant: Literal[
            "static_high_extra", "static_all_extra", "union_baseline"
        ],
    ) -> None:
        super().__init__(real)
        self.static_variant = static_variant

    def propose(self, *args, **kwargs):
        raise RuntimeError("static control has no proposal interface")


__all__ = [
    "NoMemory",
    "PROTOCOL_SCOPE",
    "RealMemory",
    "SemanticSupport",
    "ShuffledMemory",
    "StaticPolicyControl",
    "continuous_stage_rank",
]

