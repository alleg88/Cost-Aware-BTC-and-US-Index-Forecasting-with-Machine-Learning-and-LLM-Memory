"""Pure helpers for registered multi-window reflection replay variants."""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from reflection_agent.contracts import MemorySnippet, ObservationReport, ShadowState
from reflection_agent.manifest import sha256_payload


@dataclass(frozen=True)
class ReplayVariant:
    variant_id: str
    memory_mode: str
    news_mode: str


REGISTERED_VARIANTS = {
    variant.variant_id: variant for variant in (
        ReplayVariant("reflection_no_memory", "none", "bounded_text"),
        ReplayVariant("reflection_real_memory", "real", "bounded_text"),
        ReplayVariant("reflection_shuffled_memory", "shuffled", "bounded_text"),
        ReplayVariant("news_none", "real", "none"),
        ReplayVariant("news_aggregate", "real", "aggregate"),
    )
}


def observation_tags(observation: ObservationReport) -> tuple[str, ...]:
    return (
        f"vol:{observation.market.vol_regime}",
        f"trend:{observation.market.trend_regime}",
    )


def shuffle_time_eligible_memories(
    memories: list[MemorySnippet], *, window_id: str
) -> list[MemorySnippet]:
    shuffled = list(memories)
    seed = int(sha256_payload({"window_id": window_id, "memory_ids": [item.memory_id for item in shuffled]})[:16], 16)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def shadow_window_ids(shadow: ShadowState, *, close_end_exclusive: datetime) -> list[str]:
    start = pd.Timestamp(shadow.eligible_after_utc).ceil("1us")
    end = pd.Timestamp(close_end_exclusive)
    if end <= start:
        return []
    weekly_starts = pd.date_range(start=start.normalize(), end=end - pd.Timedelta(microseconds=1), freq="7D")
    return [f"{timestamp.isocalendar().year}-W{timestamp.isocalendar().week:02d}" for timestamp in weekly_starts]


class DisabledMemoryManager:
    """No-op memory store for the registered no-memory control."""

    def store_episode(self, **kwargs):
        return "disabled"

    def consolidate(self, **kwargs):
        return []

    def retrieve(self, **kwargs):
        return []
