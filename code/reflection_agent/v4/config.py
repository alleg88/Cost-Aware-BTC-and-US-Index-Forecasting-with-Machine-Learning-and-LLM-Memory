"""Frozen configuration loader for the v4 policy router."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import model_validator

from reflection_agent.v4.contracts import StrictModel
from reflection_agent.v4.policies import POLICY_IDS


AGENT_VARIANTS = (
    "reflection_real_memory",
    "reflection_no_memory",
    "reflection_shuffled_memory",
)


class ProtocolConfigV4(StrictModel):
    protocol_version: Literal["reflection-agent-v4.0"]
    schema_version: Literal["4.0"]
    protocol_scope: Literal["causal_full_information_policy_router"]
    model: Literal["deepseek-v4-flash:cloud"]
    required_model_digest: str
    think: Literal["low"]
    stream: Literal[False]
    temperature: Literal[0.0]
    num_predict: Literal[4096]
    timeout_seconds: Literal[300]
    repair_attempts: Literal[1]
    development_start_utc: datetime
    development_end_utc: datetime
    h1_start_utc: datetime
    h1_end_utc: datetime
    forward_start_utc: datetime
    forward_end_utc: datetime
    q2_start_utc: datetime
    q2_end_utc: datetime
    policy_ids: tuple[str, ...]
    agent_variants: tuple[str, ...]
    max_memory_cards: Literal[12]
    hedge_eta: Literal[0.5]
    random_seed: Literal[42]
    minimum_trade_gain_fraction: Literal[0.25]
    minimum_side_gain_fraction: Literal[0.10]
    net_noninferiority_margin: Literal[0.005]
    side_noninferiority_margin: Literal[0.0025]
    sortino_noninferiority_margin: Literal[0.10]
    max_drawdown_absolute_margin: Literal[0.01]
    max_extra_trade_concentration: Literal[0.60]
    max_transport_failure_fraction: Literal[0.05]

    @model_validator(mode="after")
    def frozen_contract_is_consistent(self) -> "ProtocolConfigV4":
        times = (
            self.development_start_utc,
            self.development_end_utc,
            self.h1_start_utc,
            self.h1_end_utc,
            self.forward_start_utc,
            self.forward_end_utc,
            self.q2_start_utc,
            self.q2_end_utc,
        )
        if any(value.tzinfo is None for value in times):
            raise ValueError("all v4 timestamps must be timezone-aware")
        if not (
            self.development_start_utc
            < self.development_end_utc
            == self.h1_start_utc
            < self.h1_end_utc
            == self.forward_start_utc
            < self.forward_end_utc
            == self.q2_start_utc
            < self.q2_end_utc
        ):
            raise ValueError("v4 stage intervals changed or overlap")
        if tuple(self.policy_ids) != POLICY_IDS:
            raise ValueError("v4 policy menu changed")
        if tuple(self.agent_variants) != AGENT_VARIANTS:
            raise ValueError("v4 agent variants changed")
        if len(self.required_model_digest) != 64:
            raise ValueError("v4 model digest must be a 64-hex snapshot")
        int(self.required_model_digest, 16)
        return self


def load_v4_config(path: str | Path) -> ProtocolConfigV4:
    with Path(path).open("r", encoding="utf-8") as handle:
        return ProtocolConfigV4.model_validate(yaml.safe_load(handle))


__all__ = ["AGENT_VARIANTS", "ProtocolConfigV4", "load_v4_config"]
