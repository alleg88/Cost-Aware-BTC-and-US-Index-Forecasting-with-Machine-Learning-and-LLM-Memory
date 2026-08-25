"""Causal full-information weekly router and deterministic controls."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from reflection_agent.v4.contracts import RouterChoice
from reflection_agent.v4.policies import POLICY_DESCRIPTIONS, POLICY_IDS, Q2_START
from reflection_agent.v4.prompts import router_messages


MemoryMode = Literal["real", "no_memory", "shuffled", "static", "hedge", "random"]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WeeklyBlock:
    block_id: str
    stage: str
    starts_at: pd.Timestamp
    available_at: pd.Timestamp


def build_weekly_blocks(payoffs: pd.DataFrame) -> list[WeeklyBlock]:
    required = {
        "block_id",
        "block_start",
        "block_available_at",
        "stage",
        "choice_index",
        "policy_id",
        "incremental_net",
        "additional_trades",
        "additional_long_trades",
        "additional_short_trades",
    }
    missing = sorted(required.difference(payoffs.columns))
    if missing:
        raise ValueError(f"router payoff table lacks columns: {missing}")
    table = payoffs.copy()
    table["block_start"] = pd.to_datetime(table["block_start"], utc=True)
    table["block_available_at"] = pd.to_datetime(table["block_available_at"], utc=True)
    blocks: list[WeeklyBlock] = []
    for block_id, rows in table.groupby("block_id", sort=False):
        if rows["block_start"].nunique() != 1 or rows["block_available_at"].nunique() != 1:
            raise ValueError("router block timestamps disagree across policies")
        if rows["stage"].nunique() != 1:
            raise ValueError("router block crossed a stage")
        if set(rows["policy_id"]) != set(POLICY_IDS[: len(rows)]):
            # Toy tests may deliberately carry a prefix of the frozen menu.
            indices = sorted(rows["choice_index"].astype(int).tolist())
            if indices != list(range(len(rows))):
                raise ValueError("router block policy menu is incomplete")
        starts_at = pd.Timestamp(rows["block_start"].iloc[0])
        available_at = pd.Timestamp(rows["block_available_at"].iloc[0])
        if starts_at >= Q2_START or available_at >= Q2_START:
            raise ValueError("Q2 block entered the router")
        if available_at <= starts_at:
            raise ValueError("router feedback must follow block commitment")
        blocks.append(
            WeeklyBlock(
                block_id=str(block_id),
                stage=str(rows["stage"].iloc[0]),
                starts_at=starts_at,
                available_at=available_at,
            )
        )
    blocks.sort(key=lambda item: (item.starts_at, item.block_id))
    for previous, current in zip(blocks, blocks[1:], strict=False):
        if previous.starts_at >= current.starts_at:
            raise ValueError("router block starts are not strictly chronological")
    return blocks


class FullInformationRouter:
    def __init__(
        self,
        *,
        caller: Any | None,
        memory_mode: MemoryMode,
        static_choice: int | None = None,
        seed: int = 42,
        hedge_eta: float = 0.5,
    ) -> None:
        self.caller = caller
        self.memory_mode = memory_mode
        self.static_choice = static_choice
        self.seed = int(seed)
        self.hedge_eta = float(hedge_eta)
        if memory_mode in {"real", "no_memory", "shuffled"} and caller is None:
            raise ValueError("agent memory modes require a schema caller")
        if memory_mode == "static" and static_choice is None:
            raise ValueError("static router requires a choice index")

    @staticmethod
    def _policy_statistics(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not history:
            return rows
        flattened = [item for card in history for item in card["policy_payoffs"]]
        rolling = [item for card in history[-12:] for item in card["policy_payoffs"]]
        frame = pd.DataFrame(flattened)
        rolling_frame = pd.DataFrame(rolling)
        for index, (policy_id, group) in enumerate(frame.groupby("policy_id", sort=False)):
            rolling_group = rolling_frame.loc[rolling_frame["policy_id"].eq(policy_id)]
            rows.append(
                {
                    "evidence_index": index,
                    "policy_id": str(policy_id),
                    "blocks": int(len(group)),
                    "incremental_net_sum": round(float(group["incremental_net"].sum()), 8),
                    "positive_block_share": round(float(group["incremental_net"].gt(0).mean()), 6),
                    "additional_trades_sum": int(group["additional_trades"].sum()),
                    "additional_long_sum": int(group["additional_long_trades"].sum()),
                    "additional_short_sum": int(group["additional_short_trades"].sum()),
                    "rolling_blocks": int(len(rolling_group)),
                    "rolling_incremental_net_sum": round(
                        float(rolling_group["incremental_net"].sum()), 8
                    ),
                    "rolling_positive_block_share": round(
                        float(rolling_group["incremental_net"].gt(0).mean()), 6
                    ),
                }
            )
        return rows

    def _shuffled_history(self, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        shuffled: list[dict[str, Any]] = []
        for card_number, card in enumerate(history):
            values = [dict(item) for item in card["policy_payoffs"]]
            labels = [item["policy_id"] for item in values]
            if len(labels) > 1:
                shift = 1 + ((self.seed + card_number) % (len(labels) - 1))
                labels = labels[shift:] + labels[:shift]
            for item, policy_id in zip(values, labels, strict=True):
                item["policy_id"] = policy_id
            shuffled.append({**card, "policy_payoffs": values})
        return shuffled

    @staticmethod
    def _coverage_status(chosen_history: list[dict[str, Any]]) -> dict[str, float | int]:
        union = sum(int(item.get("union_trades", 0)) for item in chosen_history)
        extra = sum(int(item.get("additional_trades", 0)) for item in chosen_history)
        union_long = sum(int(item.get("union_long_trades", 0)) for item in chosen_history)
        union_short = sum(int(item.get("union_short_trades", 0)) for item in chosen_history)
        extra_long = sum(int(item.get("additional_long_trades", 0)) for item in chosen_history)
        extra_short = sum(int(item.get("additional_short_trades", 0)) for item in chosen_history)
        return {
            "union_trades": union,
            "additional_trades": extra,
            "trade_ratio": round((union + extra) / union, 6) if union else 1.0,
            "long_growth": round(extra_long / union_long, 6) if union_long else 0.0,
            "short_growth": round(extra_short / union_short, 6) if union_short else 0.0,
        }

    def run(self, payoffs: pd.DataFrame) -> pd.DataFrame:
        table = payoffs.copy()
        table["block_start"] = pd.to_datetime(table["block_start"], utc=True)
        table["block_available_at"] = pd.to_datetime(table["block_available_at"], utc=True)
        blocks = build_weekly_blocks(table)
        history: list[dict[str, Any]] = []
        chosen_history: list[dict[str, Any]] = []
        records: list[dict[str, Any]] = []
        policy_count = int(table["choice_index"].max()) + 1
        hedge_weights = np.ones(policy_count, dtype=float)
        rng = np.random.default_rng(self.seed)

        for block_number, block in enumerate(blocks):
            rows = table.loc[table["block_id"].eq(block.block_id)].sort_values("choice_index")
            eligible_history = [
                item
                for item in history
                if pd.Timestamp(item["available_at"]) < block.starts_at
            ]
            call_status = "not_called"
            request_hash = ""
            response_hash = ""
            evidence_indices: list[int] = []
            memory_indices: list[int] = []
            if self.memory_mode == "static":
                choice_index = int(self.static_choice)
            elif self.memory_mode == "random":
                choice_index = int(rng.integers(0, policy_count))
            elif self.memory_mode == "hedge":
                choice_index = int(np.flatnonzero(hedge_weights == hedge_weights.max())[0])
            else:
                visible_all = eligible_history if self.memory_mode == "real" else []
                if self.memory_mode == "shuffled":
                    visible_all = self._shuffled_history(eligible_history)
                visible = visible_all[-12:]
                prompt_cards = [
                    {
                        key: value
                        for key, value in card.items()
                        if key != "available_at"
                    }
                    for card in visible
                ]
                payload = {
                    "schema_version": "4.0",
                    "coverage_status": self._coverage_status(
                        [
                            item
                            for item in chosen_history
                            if str(item.get("stage")) == block.stage
                        ]
                    ),
                    "policy_menu": [
                        {
                            "choice_index": int(row.choice_index),
                            "policy_id": str(row.policy_id),
                            "description": POLICY_DESCRIPTIONS.get(
                                str(row.policy_id), "Host-owned frozen policy."
                            ),
                        }
                        for row in rows.itertuples(index=False)
                    ],
                    "policy_statistics": self._policy_statistics(visible_all),
                    "memory_cards": [
                        {"memory_index": index, **card}
                        for index, card in enumerate(prompt_cards)
                    ],
                }
                allowed_ids = {
                    "choice_indices": rows["choice_index"].astype(int).tolist(),
                    "evidence_indices": list(range(len(payload["policy_statistics"]))),
                    "memory_indices": list(range(len(payload["memory_cards"]))),
                }
                result = self.caller.call(
                    role="router",
                    messages=router_messages(payload),
                    response_model=RouterChoice,
                    allowed_ids=allowed_ids,
                )
                call_status = str(result.status)
                request_hash = str(result.request_hash)
                response_hash = str(result.response_hash)
                proposed = int(result.value.choice_index) if result.value is not None else 0
                if result.value is not None:
                    evidence_indices = list(result.value.evidence_indices)
                    memory_indices = list(result.value.memory_indices)
                references_valid = (
                    proposed in set(allowed_ids["choice_indices"])
                    and set(evidence_indices).issubset(allowed_ids["evidence_indices"])
                    and set(memory_indices).issubset(allowed_ids["memory_indices"])
                )
                if references_valid:
                    choice_index = proposed
                else:
                    choice_index = 0
                    evidence_indices = []
                    memory_indices = []
                    call_status = "invalid_reference"

            chosen = rows.loc[rows["choice_index"].eq(choice_index)].iloc[0]
            records.append(
                {
                    "block_id": block.block_id,
                    "stage": block.stage,
                    "decision_time": block.starts_at,
                    "choice_index": choice_index,
                    "policy_id": str(chosen["policy_id"]),
                    "call_status": call_status,
                    "request_hash": request_hash,
                    "response_hash": response_hash,
                    "evidence_indices": _canonical_json(evidence_indices),
                    "memory_indices": _canonical_json(memory_indices),
                    "memory_cards_visible": (
                        len(eligible_history[-12:])
                        if self.memory_mode in {"real", "shuffled"}
                        else 0
                    ),
                }
            )
            chosen_history.append(chosen.to_dict())
            card = {
                "block_id": block.block_id,
                "available_at": block.available_at.isoformat(),
                "policy_payoffs": [
                    {
                        "policy_id": str(row.policy_id),
                        "incremental_net": round(float(row.incremental_net), 8),
                        "additional_trades": int(row.additional_trades),
                        "additional_long_trades": int(row.additional_long_trades),
                        "additional_short_trades": int(row.additional_short_trades),
                    }
                    for row in rows.itertuples(index=False)
                ],
            }
            history.append(card)
            if self.memory_mode == "hedge":
                reward = np.tanh(rows["incremental_net"].to_numpy(float) / 0.01)
                hedge_weights *= np.exp(self.hedge_eta * reward)
                hedge_weights /= hedge_weights.sum()

        result = pd.DataFrame.from_records(records)
        result.attrs["choice_hash"] = _hash(result.to_dict(orient="records"))
        return result


__all__ = ["FullInformationRouter", "WeeklyBlock", "build_weekly_blocks"]
