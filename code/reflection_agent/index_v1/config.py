"""Frozen configuration for the USA500 reflection-weight experiment."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator

from reflection_agent.index_v1.contracts import StrictModel


MODEL_NAMES = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)
REGISTERED_VARIANTS = (
    "original_frozen",
    "direct_real_memory",
    "direct_no_memory",
    "direct_shuffled_memory",
    "weekly_real_memory",
    "weekly_no_memory",
    "weekly_shuffled_memory",
    "hedge_weekly",
    "static_h1",
)
MODEL_DIGEST = "d3f1c87447216481a8001f48c517a51e13bfb141853a8df5e52f81bf765dabc3"


class IndexAgentConfig(StrictModel):
    protocol_version: Literal["usa500-reflection-weight-agent-v1.0"]
    schema_version: Literal["1.0"]
    protocol_scope: Literal["frozen_usa500_ensemble_direction_overlay"]
    stream_name: Literal["usa500"]
    source_candidate_id: Literal["selected_base__directional_majority"]
    source_arm: Literal["selected_base"]
    source_variant: Literal["directional_majority"]
    source_width_bps: Literal[15]
    source_tau: Literal[0.8]
    round_trip_cost_bps: Literal[2.0]
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
    def frozen_protocol_is_consistent(self) -> "IndexAgentConfig":
        timestamps = (
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
            self.h1_start_utc
            < self.h1_end_utc
            == self.forward_start_utc
            < self.forward_end_utc
            == self.q2_start_utc
            < self.q2_end_utc
        ):
            raise ValueError("H1, Forward and Q2 boundaries changed or overlap")
        if tuple(self.model_names) != MODEL_NAMES:
            raise ValueError("frozen nine-model order changed")
        if tuple(self.registered_variants) != REGISTERED_VARIANTS:
            raise ValueError("registered arm order changed")
        return self


def load_index_agent_config(path: str | Path) -> IndexAgentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return IndexAgentConfig.model_validate(yaml.safe_load(handle))


__all__ = [
    "IndexAgentConfig",
    "MODEL_DIGEST",
    "MODEL_NAMES",
    "REGISTERED_VARIANTS",
    "load_index_agent_config",
]
