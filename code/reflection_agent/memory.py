"""CoALA-style episodic and evidence-gated semantic memory."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Iterable, Sequence

from reflection_agent.contracts import EvaluationRecord, MemorySnippet, ReflectionRecord, SemanticBelief
from reflection_agent.manifest import sha256_payload
from reflection_agent.store import AgentStore


class MemoryManager:
    def __init__(self, store: AgentStore, *, protocol_hash: str):
        self.store = store
        self.protocol_hash = protocol_hash

    def store_episode(
        self,
        *,
        reflection: ReflectionRecord,
        evaluation: EvaluationRecord,
        cutoff_utc: datetime,
        tags: Sequence[str],
    ) -> str:
        episode_id = f"episode-{sha256_payload({'reflection': reflection.reflection_id, 'evaluation': evaluation.evaluation_id})[:20]}"
        lesson = reflection.generalized_lesson or reflection.failure_cause or "No generalizable lesson was supported."
        status = "supported" if evaluation.decision == "promote" else "rejected"
        snippet = MemorySnippet(
            memory_id=episode_id,
            memory_type="episodic",
            cutoff_utc=cutoff_utc,
            lesson=lesson,
            evidence_status=status,
            tags=sorted(set(tags)),
        )
        self.store.save_record(
            "memories",
            episode_id,
            self.protocol_hash,
            {
                "memory_type": "episodic",
                "cutoff_utc": cutoff_utc.isoformat(),
                "snippet": snippet.model_dump(mode="json"),
                "reflection": reflection.model_dump(mode="json"),
                "evaluation_id": evaluation.evaluation_id,
                "candidate_id": reflection.candidate_id,
                "tags": sorted(set(tags)),
            },
            window_id=evaluation.window_ids[-1] if evaluation.window_ids else None,
        )
        return episode_id

    def consolidate(self, *, now_utc: datetime, expiry_days: int = 90) -> list[SemanticBelief]:
        episodes = [
            row for row in self.store.list_records("memories", protocol_hash=self.protocol_hash)
            if row["payload"]["memory_type"] == "episodic"
        ]
        grouped: dict[str, list[dict]] = {}
        for row in episodes:
            if row["payload"]["snippet"]["evidence_status"] != "supported":
                continue
            reflection = row["payload"]["reflection"]
            lesson = reflection.get("generalized_lesson")
            if reflection.get("memory_recommendation") != "propose_semantic" or not lesson:
                continue
            grouped.setdefault(" ".join(lesson.lower().split()), []).append(row)
        beliefs = []
        for normalized_lesson, rows in sorted(grouped.items()):
            independent = {row["payload"]["evaluation_id"] for row in rows}
            if len(independent) < 2:
                continue
            belief_id = f"semantic-{sha256_payload(normalized_lesson)[:20]}"
            supporting = sorted(row["record_id"] for row in rows)
            tags = sorted({tag for row in rows for tag in row["payload"].get("tags", [])})
            belief = SemanticBelief(
                belief_id=belief_id,
                lesson=rows[0]["payload"]["reflection"]["generalized_lesson"],
                supporting_episode_ids=supporting,
                valid_tags=tags,
                confidence=min(0.95, 0.50 + 0.10 * len(independent)),
                created_at_utc=now_utc,
                expires_at_utc=now_utc + timedelta(days=expiry_days),
            )
            existing = self.store.load_record(
                "memories", belief_id, protocol_hash=self.protocol_hash
            )
            if existing:
                beliefs.append(SemanticBelief.model_validate(existing["belief"]))
                continue
            snippet = MemorySnippet(
                memory_id=belief_id,
                memory_type="semantic",
                cutoff_utc=now_utc,
                lesson=belief.lesson,
                evidence_status="supported",
                tags=tags,
            )
            self.store.save_record(
                "memories",
                belief_id,
                self.protocol_hash,
                {
                    "memory_type": "semantic",
                    "cutoff_utc": now_utc.isoformat(),
                    "expires_at_utc": belief.expires_at_utc.isoformat(),
                    "snippet": snippet.model_dump(mode="json"),
                    "belief": belief.model_dump(mode="json"),
                    "tags": tags,
                },
            )
            beliefs.append(belief)
        return beliefs

    def retrieve(
        self,
        *,
        cutoff_utc: datetime,
        tags: Iterable[str],
        maximum: int = 8,
    ) -> list[MemorySnippet]:
        requested = set(tags)
        eligible = []
        for row in self.store.list_records("memories", protocol_hash=self.protocol_hash):
            payload = row["payload"]
            memory_cutoff = datetime.fromisoformat(payload["cutoff_utc"])
            if memory_cutoff.tzinfo is None:
                memory_cutoff = memory_cutoff.replace(tzinfo=UTC)
            if memory_cutoff > cutoff_utc:
                continue
            expires = payload.get("expires_at_utc")
            if expires and datetime.fromisoformat(expires) <= cutoff_utc:
                continue
            snippet = MemorySnippet.model_validate(payload["snippet"])
            overlap = len(requested.intersection(snippet.tags))
            if requested and overlap == 0 and snippet.memory_type != "procedural":
                continue
            eligible.append((overlap, memory_cutoff, snippet.memory_id, snippet))
        eligible.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [item[-1] for item in eligible[:maximum]]
