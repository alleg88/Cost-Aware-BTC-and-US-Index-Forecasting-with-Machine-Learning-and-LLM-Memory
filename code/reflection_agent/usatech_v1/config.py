"""Frozen configuration for the USATECH reflection-weight replication."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ConfigDict, model_validator

from reflection_agent.index_v1.config import (
    MODEL_DIGEST,
    MODEL_NAMES,
    REGISTERED_VARIANTS,
)
from reflection_agent.index_v1.contracts import StrictModel


_DATETIME_FIELDS = (
    "h1_start_utc",
    "h1_end_utc",
    "forward_start_utc",
    "forward_end_utc",
    "q2_start_utc",
    "q2_end_utc",
)


class USATechAgentConfig(StrictModel):
    model_config = ConfigDict(
        extra="forbid", frozen=True, strict=True, str_strip_whitespace=True
    )

    protocol_version: Literal["usatech-reflection-weight-agent-v1.0"]
    schema_version: Literal["1.0"]
    protocol_scope: Literal["frozen_usatech_ensemble_direction_overlay"]
    stream_name: Literal["usatech"]
    source_candidate_id: Literal["deepseek_full__soft_vote"]
    source_arm: Literal["deepseek_full"]
    source_variant: Literal["soft_vote"]
    source_width_bps: Literal[10]
    source_tau: Literal[0.55]
    round_trip_cost_bps: Literal[3.0]
    model: Literal["deepseek-v4-flash:0731-cloud"]
    required_model_digest: Literal[MODEL_DIGEST]
    think: Literal["low"]
    ollama_stream: Literal[False]
    temperature: Literal[0.0]
    seed: Literal[42]
    num_predict: Literal[4096]
    timeout_seconds: Literal[300]
    repair_attempts: Literal[1]
    direct_batch_size: Literal[10]
    h1_start_utc: datetime
    h1_end_utc: datetime
    forward_start_utc: datetime
    forward_end_utc: datetime
    q2_start_utc: datetime
    q2_end_utc: datetime
    model_names: tuple[str, ...]
    registered_variants: tuple[str, ...]
    max_memory_cards: Literal[12]
    weight_grid: Literal[0.01]
    max_model_weight: Literal[0.80]
    minimum_material_models: Literal[3]
    material_weight: Literal[0.05]
    hedge_eta: Literal[0.5]
    max_transport_failure_fraction: Literal[0.05]
    max_drawdown_absolute_margin: Literal[0.01]
    minimum_side_trades: Literal[15]
    bootstrap_samples: Literal[2000]

    @model_validator(mode="after")
    def frozen_protocol_is_consistent(self) -> "USATechAgentConfig":
        timestamps = tuple(getattr(self, field) for field in _DATETIME_FIELDS)
        if any(value.tzinfo is None for value in timestamps):
            raise ValueError("all protocol timestamps must be timezone-aware")
        if not (
            self.h1_start_utc
            < self.h1_end_utc
            == self.forward_start_utc
            < self.forward_end_utc
            == self.q2_start_utc
            < self.q2_end_utc
        ):
            raise ValueError("H1, Forward and Q2 boundaries changed or overlap")
        if self.model_names != MODEL_NAMES:
            raise ValueError("frozen nine-model order changed")
        if self.registered_variants != REGISTERED_VARIANTS:
            raise ValueError("registered arm order changed")
        return self


def _strict_payload(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    for field in _DATETIME_FIELDS:
        payload[field] = datetime.fromisoformat(str(payload[field]).replace("Z", "+00:00"))
    payload["model_names"] = tuple(payload["model_names"])
    payload["registered_variants"] = tuple(payload["registered_variants"])
    return payload


def load_usatech_agent_config(path: str | Path) -> USATechAgentConfig:
    return USATechAgentConfig.model_validate(_strict_payload(path))


__all__ = [
    "MODEL_DIGEST",
    "MODEL_NAMES",
    "REGISTERED_VARIANTS",
    "USATechAgentConfig",
    "load_usatech_agent_config",
]
