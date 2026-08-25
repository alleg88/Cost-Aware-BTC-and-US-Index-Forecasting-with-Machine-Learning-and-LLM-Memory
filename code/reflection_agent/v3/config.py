"""Frozen configuration loader for Reflection Agent v3."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from reflection_agent.v3.contracts import CONDITION_VALUES, StrictModel

ALLOWED_DECISIONS = ("NO_CHANGE", "ADD_ALLOW_RULE", "REMOVE_ALLOW_RULE")
REGISTERED_VARIANTS = (
    "reflection_real_memory",
    "reflection_no_memory",
    "reflection_shuffled_memory",
    "static_high_extra",
    "static_all_extra",
    "union_baseline",
)
CONFIDENCE_TIERS = {
    "HIGH_EXTRA": (0.70, 0.75),
    "MID_EXTRA": (0.65, 0.70),
    "LOW_EXTRA": (0.60, 0.65),
}


class ProtocolConfigV3(StrictModel):
    protocol_version: Literal["reflection-agent-v3.0"]
    schema_version: Literal["3.0"]
    protocol_scope: Literal["continuous_2021_2026"]
    model: Literal["deepseek-v4-flash:cloud"]
    required_model_digest: AnnotatedDigest
    think: Literal["low"]
    stream: Literal[False]
    temperature: Literal[0.0]
    num_predict: Literal[4096]
    timeout_seconds: Literal[300]
    repair_attempts: Literal[1]
    allowed_decisions: tuple[
        Literal["NO_CHANGE"], Literal["ADD_ALLOW_RULE"], Literal["REMOVE_ALLOW_RULE"]
    ]
    registered_variants: tuple[
        Literal["reflection_real_memory"],
        Literal["reflection_no_memory"],
        Literal["reflection_shuffled_memory"],
        Literal["static_high_extra"],
        Literal["static_all_extra"],
        Literal["union_baseline"],
    ]
    development_start_utc: datetime
    development_end_utc: datetime
    h1_start_utc: datetime
    h1_end_utc: datetime
    forward_start_utc: datetime
    forward_end_utc: datetime
    q2_start_utc: datetime
    q2_end_utc: datetime
    confidence_tiers: dict[str, tuple[float, float]]
    episode_min_candidates: Literal[20]
    episode_max_candidates: Literal[30]
    episode_min_per_side: Literal[6]
    shadow_min_matching_candidates: Literal[12]
    shadow_max_candidates: Literal[40]
    max_active_rules_per_side: Literal[3]
    max_retrieved_memories: Literal[4]
    semantic_lifetime_days: Literal[180]
    semantic_lifetime_episodes: Literal[6]
    minimum_trade_gain_fraction: Literal[0.25]
    minimum_side_gain_fraction: Literal[0.10]
    net_noninferiority_margin: Literal[0.005]
    side_noninferiority_margin: Literal[0.0025]
    sortino_noninferiority_margin: Literal[0.10]
    max_drawdown_absolute_margin: Literal[0.01]
    max_extra_trade_concentration: Literal[0.60]
    max_transport_failure_fraction: Literal[0.05]
    condition_values: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def frozen_protocol_is_consistent(self) -> "ProtocolConfigV3":
        timestamps = (
            self.development_start_utc,
            self.development_end_utc,
            self.h1_start_utc,
            self.h1_end_utc,
            self.forward_start_utc,
            self.forward_end_utc,
            self.q2_start_utc,
            self.q2_end_utc,
        )
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("all protocol timestamps must be timezone-aware")
        if not (
            self.development_start_utc < self.development_end_utc
            == self.h1_start_utc < self.h1_end_utc
            == self.forward_start_utc < self.forward_end_utc
            == self.q2_start_utc < self.q2_end_utc
        ):
            raise ValueError("registered stage intervals changed or overlap")
        if self.allowed_decisions != ALLOWED_DECISIONS:
            raise ValueError("allowed decision order changed")
        if self.registered_variants != REGISTERED_VARIANTS:
            raise ValueError("registered variants changed")
        if {key: tuple(value) for key, value in self.confidence_tiers.items()} != (
            CONFIDENCE_TIERS
        ):
            raise ValueError("confidence tiers changed")
        if {key: tuple(value) for key, value in self.condition_values.items()} != (
            CONDITION_VALUES
        ):
            raise ValueError("condition allowlist changed")
        return self


AnnotatedDigest = str


def load_v3_config(path: str | Path) -> ProtocolConfigV3:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ProtocolConfigV3.model_validate(yaml.safe_load(handle))


__all__ = [
    "ALLOWED_DECISIONS",
    "CONFIDENCE_TIERS",
    "ProtocolConfigV3",
    "REGISTERED_VARIANTS",
    "load_v3_config",
]
