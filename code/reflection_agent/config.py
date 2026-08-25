"""Validated immutable configuration for the reflection-agent protocol."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import ConfigDict, Field, model_validator

from reflection_agent.contracts import MODEL_IDS, StrictModel
from reflection_agent.policy import AGREEMENT_GRID, CONFIDENCE_GRID, WEIGHT_GRID


class NewsConfig(StrictModel):
    max_items: int = Field(ge=1, le=12)
    max_items_per_source_family: int = Field(ge=1, le=3)
    max_summary_characters: int = Field(ge=50, le=500)


class MemoryConfig(StrictModel):
    max_retrieved: int = Field(ge=0, le=8)
    semantic_min_independent_episodes: int = Field(ge=2)


class MarketFeatureConfig(StrictModel):
    recent_return_bars: int = Field(ge=1)
    realized_volatility_bars: int = Field(ge=2)
    volatility_reference_bars: int = Field(ge=3)
    trend_threshold: float = Field(gt=0.0)
    news_lookback_bars: int = Field(ge=1)


class ProtocolConfig(StrictModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    protocol_version: str
    schema_version: Literal["1.0"]
    model: Literal["glm-5.2:cloud"]
    think: Literal["high"]
    output_mode: Literal["probe", "schema", "json"]
    stream: Literal[False]
    timeout_seconds: int = Field(ge=1)
    retry_delays_seconds: list[int] = Field(min_length=1, max_length=3)
    repair_attempts: Literal[2]
    temperatures: dict[Literal["actor", "refiner", "reflector"], float]
    models: list[Literal[*MODEL_IDS]]
    primary_baseline: Literal["lstm"]
    ensemble_control: Literal["unanimity_consensus"]
    fee_bps_per_side: float = Field(ge=0.0)
    development_start_utc: datetime
    development_end_utc: datetime
    sealed_start_utc: datetime
    sealed_end_utc: datetime
    candidate_budget: int = Field(ge=1, le=6)
    beam_width: int = Field(ge=1, le=3)
    tree_depth: Literal[2]
    max_open_shadows: int = Field(ge=1, le=2)
    shadow_min_weeks: int = Field(ge=2)
    shadow_max_weeks: int = Field(le=4)
    shadow_min_trades: int = Field(ge=10)
    shadow_min_trades_per_side: int = Field(ge=3)
    final_min_trades: int = Field(ge=30)
    final_min_trades_per_side: int = Field(ge=10)
    max_monthly_gain_concentration: float = Field(ge=0.0, le=0.60)
    sortino_noninferiority_margin: float = Field(ge=0.0, le=0.10)
    max_drawdown_ratio: float = Field(ge=1.0, le=1.10)
    weight_grid: list[float]
    confidence_grid: list[float]
    agreement_grid: list[float]
    news: NewsConfig
    memory: MemoryConfig
    market: MarketFeatureConfig

    @model_validator(mode="after")
    def fixed_protocol_is_consistent(self) -> "ProtocolConfig":
        if self.models != list(MODEL_IDS):
            raise ValueError("models must match the registered order")
        if tuple(self.weight_grid) != WEIGHT_GRID:
            raise ValueError("weight grid changed")
        if tuple(self.confidence_grid) != CONFIDENCE_GRID:
            raise ValueError("confidence grid changed")
        if tuple(self.agreement_grid) != AGREEMENT_GRID:
            raise ValueError("agreement grid changed")
        if self.development_start_utc >= self.development_end_utc:
            raise ValueError("development interval is empty")
        if self.development_end_utc > self.sealed_start_utc:
            raise ValueError("development interval enters the sealed lockbox")
        if self.sealed_start_utc >= self.sealed_end_utc:
            raise ValueError("sealed interval is empty")
        if self.shadow_min_weeks > self.shadow_max_weeks:
            raise ValueError("shadow minimum exceeds maximum")
        if self.beam_width > self.candidate_budget:
            raise ValueError("beam width exceeds candidate budget")
        if set(self.temperatures) != {"actor", "refiner", "reflector"}:
            raise ValueError("all three role temperatures are required")
        return self


def load_config(path: str | Path) -> ProtocolConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ProtocolConfig.model_validate(yaml.safe_load(handle))
