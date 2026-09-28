"""Causal weekly weights and exact one-bar execution for the RQ4 comparison."""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Annotated, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from experiments.catboost_execution_scoring import simulate_policy
from experiments.raw_hold_control import MODEL_NAMES

BAR = pd.Timedelta(minutes=15)
WEEK = pd.Timedelta(days=7)
Weight = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class ModelWeights(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    logreg: Weight
    decision_tree: Weight
    random_forest: Weight
    svm_linear: Weight
    xgboost_balanced: Weight
    catboost_balanced: Weight
    mlp: Weight
    lstm: Weight
    gru: Weight

    @model_validator(mode="after")
    def unit_sum(self) -> Self:
        total = sum(Decimal(str(value)) for value in self.model_dump().values())
        if abs(total - Decimal("1")) > Decimal("0.000001"):
            raise ValueError("all nine weights must sum to one within 0.000001")
        return self


class WeightChoice(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    weights: ModelWeights
    reason: str = Field(min_length=1, max_length=180)


def array_weights(choice: WeightChoice) -> np.ndarray:
    return np.array([getattr(choice.weights, name) for name in MODEL_NAMES], dtype=float)


def weight_signal(probabilities: np.ndarray, weights: np.ndarray, tau: float = 0.55) -> np.ndarray:
    """Rows x models x classes; retain ordinary Flat and confidence abstentions."""
    mixed = np.einsum("nmc,nm->nc", probabilities, weights)
    prediction = mixed.argmax(axis=1)
    return np.where(mixed.max(axis=1) >= tau, prediction - 1, 0).astype(int)


def candidate_outcomes(bars: pd.DataFrame, minute: pd.DataFrame,
                       start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Obtain every possible one-bar path from the existing execution engine.

    Alternating signal rows avoid the engine's occupied-entry-bar restriction.
    The final replay restores that restriction across all weekly weight changes.
    """
    grid = bars.index[(bars.index >= start) & (bars.index < end)]
    chunks = []
    for tp in (150, 200):
        for side in (-1, 1):
            for parity in (0, 1):
                frame = pd.DataFrame({"timestamp": grid,
                                      "pred": np.where(np.arange(len(grid)) % 2 == parity, side + 1, 1),
                                      "confidence": 1.0})
                ledger, _ = simulate_policy(
                    bars=bars, execution=minute, prediction_frame=frame,
                    start=start, end=end, resolution="1m", tau=0, tp_bps=tp,
                    sl_bps=100, max_hold=1, fee_bps=5,
                )
                ledger["signal_time"] = ledger["entry_time"] - BAR
                ledger["policy_tp_bps"] = tp
                ledger["available_time"] = ledger["intrabar_exit_time"] + pd.Timedelta(minutes=1)
                chunks.append(ledger)
    result = pd.concat(chunks, ignore_index=True).sort_values(["signal_time", "policy_tp_bps", "side"])
    if result.duplicated(["signal_time", "policy_tp_bps", "side"]).any():
        raise ValueError("duplicate candidate execution paths")
    return result.reset_index(drop=True)


def replay_signals(grid: pd.DatetimeIndex, signal: pd.Series, outcomes: pd.DataFrame,
                   tp_bps: int | pd.Series = 150) -> tuple[pd.DataFrame, pd.Series]:
    """One position and no new signal on its entry bar, including week boundaries."""
    if not np.all(np.diff(grid.as_unit("ns").asi8) == BAR.value):
        raise ValueError("RQ4 replay requires a complete M15 grid")
    side = signal.reindex(grid, fill_value=0).to_numpy(int)
    tp = np.full(len(grid), tp_bps, dtype=int) if np.isscalar(tp_bps) else tp_bps.reindex(grid).to_numpy(int)
    lookup = {(row.signal_time, int(row.policy_tp_bps), int(row.side)): i
              for i, row in enumerate(outcomes.itertuples(index=False))}
    selected = []
    i = 0
    while i < len(grid) - 1:
        if side[i] == 0:
            i += 1
            continue
        if side[i] not in (-1, 1):
            raise ValueError("invalid signal")
        key = (grid[i], int(tp[i]), int(side[i]))
        if key not in lookup:
            raise ValueError(f"missing execution path: {key}")
        selected.append(lookup[key])
        i += 2
    ledger = outcomes.iloc[selected].copy().reset_index(drop=True)
    per_bar = pd.Series(0.0, index=grid, name="bracket_return")
    if len(ledger):
        if not ledger["bars_held"].eq(1).all():
            raise ValueError("only the frozen one-bar holding policy is supported")
        per_bar.loc[pd.DatetimeIndex(ledger["entry_time"])] = ledger["net_return"].to_numpy(float)
    if not np.isclose(per_bar.sum(), ledger["net_return"].sum(), atol=1e-12):
        raise AssertionError("replay does not reconcile")
    return ledger, per_bar


def memory_cards(expert_ledgers: dict[str, pd.DataFrame], decision: pd.Timestamp,
                 observed_start: pd.Timestamp, horizon: int = 4) -> list[dict]:
    """Completed seven-day windows only; use actual minute-close availability."""
    cards = []
    for age in range(horizon, 0, -1):
        left, right = decision - age * WEEK, decision - (age - 1) * WEEK
        if left < observed_start:
            continue
        values = []
        for name in MODEL_NAMES:
            ledger = expert_ledgers[name]
            closed = ledger.loc[(ledger["available_time"] >= left)
                                & (ledger["available_time"] < right)
                                & (ledger["available_time"] < decision)]
            returns = closed.sort_values("available_time")["net_return"].to_numpy(float)
            curve = np.r_[0.0, returns.cumsum()]
            drawdown = float(np.max(np.maximum.accumulate(curve) - curve))
            values.append([round(float(returns.sum()) * 10000, 6), len(closed),
                           round(drawdown * 10000, 6)])
        cards.append({"age_weeks": age, "values": values})
    return cards


def shuffle_cards(cards: list[dict], seed: int) -> list[dict]:
    """Permute expert identities, preserving each entire numeric performance row."""
    shuffled = []
    for card in cards:
        rng = np.random.default_rng(seed + card["age_weeks"])
        order = rng.permutation(len(MODEL_NAMES))
        while np.any(order == np.arange(len(MODEL_NAMES))):
            order = rng.permutation(len(MODEL_NAMES))
        shuffled.append({"age_weeks": card["age_weeks"],
                         "values": [list(card["values"][i]) for i in order]})
    return shuffled


def hedge_weights(cards: list[dict], eta: float) -> np.ndarray:
    scores = np.zeros(len(MODEL_NAMES))
    for card in cards:
        scores += np.tanh(np.array(card["values"], dtype=float)[:, 0] / 100.0)
    log_weights = eta * scores
    weights = np.exp(log_weights - log_weights.max())
    return weights / weights.sum()


SYSTEM_PROMPT = """You allocate nonnegative weights to nine BTC forecasting models for the next seven days.
Return exactly one JSON object with two top-level keys: "weights" and "reason". "weights" must be an object containing all nine model names; never put model names at the top level. Its values must be in [0,1] and sum to 1 within 0.000001. "reason" is a string of at most 180 characters.
Make one concise allocation from the supplied evidence. Do not perform hypothetical backtests or search a grid of allocations. Two decimal places are sufficient for weights if they sum to 1.00.
The weighted probabilities represent Short, Flat and Long with the same 65 bps label dead zone. Trade only if the top probability is at least 0.55; Flat means no trade. Position size is one unit. Execution is next M15 open, TP 150 bps, SL 100 bps, at most one M15 bar, and 5 bps fees per side. No new signal is admitted on an occupied entry bar.
Use only the supplied information. Do not infer calendar dates or outside historical events. The current state and current model probabilities are available now. If memory is supplied, it contains completed weekly outcomes of each individual model under that same execution policy. Memory values follow model_order; each row is [net_return_bps, trades, maximum_additive_drawdown_bps]. Empty memory means that historical performance is unavailable. Choose weights to improve net return while considering losses and limited evidence. Do not invent observations."""


def messages_for(state: dict, cards: list[dict]) -> list[dict[str, str]]:
    # No arm label, previous weights, previous response, or hidden chat history.
    payload = {"model_order": list(MODEL_NAMES), "current_state": state, "memory": cards}
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, separators=(",", ":"), allow_nan=False)}]
