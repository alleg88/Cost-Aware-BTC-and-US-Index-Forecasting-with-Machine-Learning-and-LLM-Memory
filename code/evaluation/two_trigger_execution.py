"""Deterministic one-minute execution for frozen two-trigger geometry."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TradeResolution:
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    exit_time: pd.Timestamp
    exit_price: float
    outcome: str
    r_net: float


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def resolve_fixed_trade(
    minute: pd.DataFrame,
    *,
    entry_time: pd.Timestamp,
    side: str,
    stop: float,
    target: float,
    max_hold_minutes: int = 120,
    round_trip_cost_bps: float = 10.0,
) -> TradeResolution | None:
    """Resolve a frozen SL/TP trade, conservatively checking SL before TP."""
    if side not in {"long", "short"}:
        raise ValueError("side must be long or short")
    if max_hold_minutes < 1:
        raise ValueError("max_hold_minutes must be positive")
    if round_trip_cost_bps < 0:
        raise ValueError("round_trip_cost_bps cannot be negative")

    entry_time = _utc(entry_time)
    position = minute.index.get_indexer([entry_time])[0]
    if position < 0:
        return None
    path = minute.iloc[position:position + max_hold_minutes]
    expected = pd.date_range(
        entry_time, periods=max_hold_minutes, freq="1min", tz="UTC"
    )
    required = ["open", "high", "low", "close"]
    if not path.index.equals(expected) or path[required].isna().any().any():
        return None

    entry = float(path.iloc[0].open)
    stop = float(stop)
    target = float(target)
    side_sign = 1.0 if side == "long" else -1.0
    risk = (entry - stop) if side_sign > 0 else (stop - entry)
    reward = (target - entry) if side_sign > 0 else (entry - target)
    if risk <= 0 or reward <= 0:
        return TradeResolution(
            entry_time=entry_time,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            exit_time=entry_time,
            exit_price=entry,
            outcome="entry_cancelled",
            r_net=0.0,
        )

    exit_time = expected[-1]
    exit_price = float(path.iloc[-1].close)
    outcome = "timeout"
    for timestamp, row in path.iterrows():
        stop_hit = (
            float(row.low) <= stop if side_sign > 0 else float(row.high) >= stop
        )
        target_hit = (
            float(row.high) >= target if side_sign > 0 else float(row.low) <= target
        )
        if stop_hit:
            exit_time, exit_price, outcome = timestamp, stop, "sl"
            break
        if target_hit:
            exit_time, exit_price, outcome = timestamp, target, "tp"
            break

    move = (exit_price - entry) * side_sign
    cost = (round_trip_cost_bps / 10_000.0) * entry
    return TradeResolution(
        entry_time=entry_time,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        exit_time=exit_time,
        exit_price=float(exit_price),
        outcome=outcome,
        r_net=float(move / risk - cost / risk),
    )
