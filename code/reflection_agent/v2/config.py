"""Frozen configuration loader for Reflection Agent v2."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from reflection_agent.v2.contracts import CONDITION_VALUES, StrictModel

ALLOWED_DECISIONS = ("NO_CHANGE", "ADD_ALLOW_RULE", "REMOVE_ALLOW_RULE")
REGISTERED_VARIANTS = (
    "reflection_real_memory",
    "reflection_no_memory",
    "reflection_shuffled_memory",
    "static_add_all",
    "union_baseline",
)


class ProtocolConfigV2(StrictModel):
    protocol_version: Literal["reflection-agent-v2.2"]
    schema_version: Literal["2.0"]
    model: Literal["deepseek-v4-flash:cloud"]
    think: Literal["high"]
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
        Literal["static_add_all"],
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
    episode_min_opportunities: Literal[60]
    episode_max_opportunities: Literal[90]
    episode_min_reentry_opportunities: Literal[10]
    episode_min_per_side: Literal[3]
    proposal_min_support: Literal[8]
    proposal_min_net_return: Literal[0.02]
    shadow_min_reentry_opportunities: Literal[10]
    shadow_min_per_side: Literal[3]
    shadow_max_opportunities: Literal[60]
    max_active_rules: Literal[3]
    max_retrieved_memories: Literal[4]
    semantic_lifetime_days: Literal[180]
    semantic_lifetime_episodes: Literal[6]
    minimum_trade_gain_fraction: Literal[0.10]
    sortino_noninferiority_margin: Literal[0.10]
    max_drawdown_ratio: Literal[1.10]
    max_positive_return_concentration: Literal[0.60]
    condition_values: dict[str, tuple[str, ...]]

    @model_validator(mode="after")
    def frozen_protocol_is_consistent(self) -> "ProtocolConfigV2":
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
            self.development_start_utc < self.development_end_utc <= self.h1_start_utc
            < self.h1_end_utc == self.forward_start_utc
            < self.forward_end_utc == self.q2_start_utc
            < self.q2_end_utc
        ):
            raise ValueError("registered stage intervals changed or overlap")
        if self.allowed_decisions != ALLOWED_DECISIONS:
            raise ValueError("allowed decision order changed")
        if self.registered_variants != REGISTERED_VARIANTS:
            raise ValueError("registered variants changed")
        normalized = {key: tuple(values) for key, values in self.condition_values.items()}
        if normalized != CONDITION_VALUES:
            raise ValueError("condition allowlist changed")
        return self


def load_v2_config(path: str | Path) -> ProtocolConfigV2:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ProtocolConfigV2.model_validate(yaml.safe_load(handle))
