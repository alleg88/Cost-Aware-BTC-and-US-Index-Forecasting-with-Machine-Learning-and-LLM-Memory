"""Frozen configuration for the USA500 budgeted reversal experiment."""
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
    "frozen_parent",
    "budgeted_real_memory",
    "budgeted_no_memory",
    "budgeted_shuffled_memory",
    "uncertainty_control",
    "seeded_hash_control",
)
MODEL_DIGEST = "d3f1c87447216481a8001f48c517a51e13bfb141853a8df5e52f81bf765dabc3"
TARGET_CHANGE_RATES = (0.15, 0.20, 0.25)


class BudgetedReversalConfig(StrictModel):
    protocol_version: Literal["usa500-budgeted-reversal-agent-v1.2"]
    schema_version: Literal["1.0"]
    protocol_scope: Literal["frozen_usa500_ensemble_budgeted_reversal_overlay"]
    stream_name: Literal["usa500"]
    source_candidate_id: Literal["selected_base__directional_majority"]
    source_arm: Literal["selected_base"]
    source_variant: Literal["directional_majority"]
    source_width_bps: Literal[15]
    source_tau: Literal[0.8]
    round_trip_cost_bps: Literal[2.0]
    model: Literal["deepseek-v4-flash:0731-cloud"]
    required_model_digest: Literal[MODEL_DIGEST]
    think: Literal[False]
    ollama_stream: Literal[False]
    temperature: Literal[0.0]
    seed: Literal[42]
    num_predict: Literal[4096]
    timeout_seconds: Literal[300]
    repair_attempts: Literal[1]
    score_batch_size: Literal[10]
    h1_start_utc: datetime
    h1_end_utc: datetime
    forward_start_utc: datetime
    forward_end_utc: datetime
    q2_start_utc: datetime
    q2_end_utc: datetime
    model_names: tuple[str, ...]
    registered_variants: tuple[str, ...]
    max_memory_cards: Literal[12]
    target_change_rates: tuple[float, ...]
    minimum_forward_change_rate: Literal[0.15]
    maximum_forward_change_rate: Literal[0.30]
    minimum_distinct_h1_scores: Literal[20]
    maximum_h1_score_concentration: Literal[0.50]
    minimum_llm_attributed_change_fraction: Literal[0.75]
    max_transport_failure_fraction: Literal[0.05]
    max_drawdown_absolute_margin: Literal[0.01]
    bootstrap_samples: Literal[2000]

    @model_validator(mode="after")
    def frozen_protocol_is_consistent(self) -> "BudgetedReversalConfig":
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
        if tuple(self.target_change_rates) != TARGET_CHANGE_RATES:
            raise ValueError("registered H1 target rates changed")
        return self


def load_budgeted_reversal_config(path: str | Path) -> BudgetedReversalConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return BudgetedReversalConfig.model_validate(yaml.safe_load(handle))


__all__ = [
    "BudgetedReversalConfig",
    "MODEL_DIGEST",
    "MODEL_NAMES",
    "REGISTERED_VARIANTS",
    "TARGET_CHANGE_RATES",
    "load_budgeted_reversal_config",
]
