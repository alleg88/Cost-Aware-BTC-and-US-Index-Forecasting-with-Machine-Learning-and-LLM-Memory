"""Causal one-minute entry decisions after a Fast-T2 +2 bps confirmation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.two_trigger_execution import resolve_fixed_trade


BASE_STATIC_FEATURE_COLUMNS = (
    "side_sign",
    "channel_slope_bps_5m_side",
    "channel_r2",
    "channel_width_pct",
    "channel_confluence_count",
    "t1_edge_depth",
    "t1_range_bps",
    "t1_body_fraction",
    "t1_wick_share_side",
    "t1_close_location_side",
    "confirmation_lag_minutes",
    "breakout_margin_bps",
)

ENTRY_FEATURE_COLUMNS = (
    *BASE_STATIC_FEATURE_COLUMNS,
    "minutes_since_t2",
    "entry_delay_fraction",
    "price_from_t2_bps_side",
    "distance_to_stop_bps",
    "distance_to_target_bps",
    "rr_proxy",
    "post_t2_mfe_bps",
    "post_t2_mae_bps",
    "realized_vol_30m_bps",
    "return_1m_side",
    "return_5m_side",
    "volume_ratio_5_20",
    "trade_count_ratio_5_20",
    "taker_imbalance_5_side",
    "hour_sin",
    "hour_cos",
)

SEQUENCE_FEATURE_COLUMNS = (
    "return_1m_side",
    "range_bps",
    "body_bps_side",
    "taker_imbalance_side",
    "log_volume",
)

_MINUTE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_base",
    "count",
)


@dataclass(frozen=True)
class EntryDecisionConfig:
    decision_minutes: int = 15
    sequence_minutes: int = 30
    max_hold_minutes: int = 120
    round_trip_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.decision_minutes < 1:
            raise ValueError("decision_minutes must be positive")
        if self.sequence_minutes < 6:
            raise ValueError("sequence_minutes must be at least six")
        if self.max_hold_minutes < 1:
            raise ValueError("max_hold_minutes must be positive")
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps cannot be negative")


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _completed_history(
    minute: pd.DataFrame, decision_time: pd.Timestamp, length: int
) -> pd.DataFrame | None:
    decision_time = _utc(decision_time)
    position = minute.index.get_indexer([decision_time])[0]
    if position < length:
        return None
    observed = minute.iloc[position - length:position]
    expected = pd.date_range(
        end=decision_time - pd.Timedelta(minutes=1),
        periods=length,
        freq="1min",
        tz="UTC",
    )
    if not observed.index.equals(expected):
        return None
    if observed[list(_MINUTE_COLUMNS)].isna().any().any():
        return None
    return observed


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _decision_features(
    event: object,
    history: pd.DataFrame,
    minute: pd.DataFrame,
    *,
    t2_time: pd.Timestamp,
    decision_time: pd.Timestamp,
    delay: int,
    config: EntryDecisionConfig,
) -> dict[str, float]:
    side_sign = float(event.side_sign)
    close = history["close"].to_numpy(dtype=float)
    high = history["high"].to_numpy(dtype=float)
    low = history["low"].to_numpy(dtype=float)
    volume = history["volume"].to_numpy(dtype=float)
    count = history["count"].to_numpy(dtype=float)
    returns = np.diff(np.log(close))
    current = float(close[-1])
    t2_position = minute.index.get_indexer([t2_time])[0]
    if t2_position < 1:
        raise ValueError("T2 has no completed reference minute")
    t2_price = float(minute.iloc[t2_position - 1].close)
    stop = float(event.stop_price)
    target = float(event.target_price)
    risk_distance = (current - stop) if side_sign > 0 else (stop - current)
    target_distance = (target - current) if side_sign > 0 else (current - target)

    if delay:
        recent = minute.iloc[t2_position:t2_position + delay]
        recent_high = float(recent.high.max())
        recent_low = float(recent.low.min())
        mfe = (
            recent_high / t2_price - 1.0
            if side_sign > 0
            else t2_price / recent_low - 1.0
        )
        mae = (
            recent_low / t2_price - 1.0
            if side_sign > 0
            else t2_price / recent_high - 1.0
        )
    else:
        mfe = mae = 0.0

    taker = np.divide(
        2.0 * history["taker_buy_base"].to_numpy(dtype=float),
        volume,
        out=np.ones_like(volume),
        where=volume > 0,
    ) - 1.0
    hour = decision_time.hour + decision_time.minute / 60.0
    values = {name: float(getattr(event, name)) for name in BASE_STATIC_FEATURE_COLUMNS}
    values.update(
        {
            "minutes_since_t2": float(delay),
            "entry_delay_fraction": float(
                delay / max(1, config.decision_minutes - 1)
            ),
            "price_from_t2_bps_side": float(
                (current / t2_price - 1.0) * 10_000.0 * side_sign
            ),
            "distance_to_stop_bps": float(risk_distance / current * 10_000.0),
            "distance_to_target_bps": float(
                target_distance / current * 10_000.0
            ),
            "rr_proxy": float(
                target_distance / risk_distance if risk_distance > 0 else 0.0
            ),
            "post_t2_mfe_bps": float(mfe * 10_000.0),
            "post_t2_mae_bps": float(mae * 10_000.0),
            "realized_vol_30m_bps": float(np.std(returns, ddof=1) * 10_000.0),
            "return_1m_side": float(returns[-1] * 10_000.0 * side_sign),
            "return_5m_side": float(
                np.log(close[-1] / close[-6]) * 10_000.0 * side_sign
            ),
            "volume_ratio_5_20": _safe_ratio(
                float(volume[-5:].mean()), float(volume[-25:-5].mean())
            ),
            "trade_count_ratio_5_20": _safe_ratio(
                float(count[-5:].mean()), float(count[-25:-5].mean())
            ),
            "taker_imbalance_5_side": float(taker[-5:].mean() * side_sign),
            "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
            "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
        }
    )
    return values


def build_entry_decisions(
    events: pd.DataFrame,
    minute_bars: pd.DataFrame,
    config: EntryDecisionConfig = EntryDecisionConfig(),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Expand each Fast-T2 event into causal counterfactual entry minutes."""
    required = {
        "candidate_id",
        "side",
        "channel_episode_id",
        "decision_time",
        "stop_price",
        "target_price",
        *BASE_STATIC_FEATURE_COLUMNS,
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"events missing entry columns: {missing}")
    minute = minute_bars.sort_index(kind="stable")
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.tz is None:
        raise ValueError("minute bars need a timezone-aware DatetimeIndex")

    audit = {
        "raw_windows": int(len(events)),
        "rows": 0,
        "censored": 0,
        "cancelled": 0,
    }
    rows: list[dict[str, object]] = []
    ordered = events.sort_values("decision_time", kind="stable")
    for event in ordered.itertuples(index=False):
        t2_time = _utc(event.decision_time)
        for delay in range(config.decision_minutes):
            decision_time = t2_time + pd.Timedelta(minutes=delay)
            history = _completed_history(
                minute, decision_time, config.sequence_minutes
            )
            if history is None or decision_time not in minute.index:
                audit["censored"] += 1
                continue
            resolution = resolve_fixed_trade(
                minute,
                entry_time=decision_time,
                side=str(event.side),
                stop=float(event.stop_price),
                target=float(event.target_price),
                max_hold_minutes=config.max_hold_minutes,
                round_trip_cost_bps=config.round_trip_cost_bps,
            )
            if resolution is None:
                audit["censored"] += 1
                continue
            features = _decision_features(
                event,
                history,
                minute,
                t2_time=t2_time,
                decision_time=decision_time,
                delay=delay,
                config=config,
            )
            window_id = str(event.candidate_id)
            rows.append(
                {
                    "window_id": window_id,
                    "decision_id": f"{window_id}:{delay:02d}",
                    "side": str(event.side),
                    "channel_episode_id": int(event.channel_episode_id),
                    "t2_time": t2_time,
                    "decision_time": decision_time,
                    "entry_time": resolution.entry_time,
                    "label_start": resolution.entry_time,
                    "label_end": resolution.exit_time,
                    "active_end_time": resolution.exit_time,
                    "entry_price": resolution.entry_price,
                    "stop_price": resolution.stop_price,
                    "target_price": resolution.target_price,
                    "exit_time": resolution.exit_time,
                    "exit_price": resolution.exit_price,
                    "outcome": resolution.outcome,
                    "r_net": resolution.r_net,
                    "label_net_positive": int(resolution.r_net > 0.0),
                    "filled": resolution.outcome != "entry_cancelled",
                    "holding_minutes": float(
                        (resolution.exit_time - resolution.entry_time)
                        / pd.Timedelta(minutes=1)
                    ),
                    **features,
                }
            )
            audit["cancelled"] += int(
                resolution.outcome == "entry_cancelled"
            )

    decisions = pd.DataFrame(rows)
    if not decisions.empty:
        decisions = decisions.sort_values(
            ["decision_time", "decision_id"], kind="stable"
        ).reset_index(drop=True)
        if not decisions["decision_id"].is_unique:
            raise AssertionError("entry decision IDs must be unique")
    audit["rows"] = int(len(decisions))
    return decisions, audit


def build_entry_sequences(
    decisions: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    sequence_minutes: int = 30,
) -> np.ndarray:
    """Return past-only one-minute sequences aligned to decision boundaries."""
    minute = minute_bars.sort_index(kind="stable")
    sequences: list[np.ndarray] = []
    for row in decisions.itertuples(index=False):
        history = _completed_history(minute, _utc(row.decision_time), sequence_minutes)
        if history is None:
            raise ValueError(f"missing sequence history for {row.decision_id}")
        side_sign = float(row.side_sign)
        close = history["close"].to_numpy(dtype=float)
        open_ = history["open"].to_numpy(dtype=float)
        volume = history["volume"].to_numpy(dtype=float)
        returns = np.diff(np.log(close), prepend=np.log(close[0])) * 10_000.0
        taker = np.divide(
            2.0 * history["taker_buy_base"].to_numpy(dtype=float),
            volume,
            out=np.ones_like(volume),
            where=volume > 0,
        ) - 1.0
        values = np.column_stack(
            [
                returns * side_sign,
                (history["high"].to_numpy(dtype=float)
                 - history["low"].to_numpy(dtype=float)) / close * 10_000.0,
                (close - open_) / open_ * 10_000.0 * side_sign,
                taker * side_sign,
                np.log1p(volume),
            ]
        )
        if not np.isfinite(values).all():
            raise ValueError(f"non-finite sequence for {row.decision_id}")
        sequences.append(values.astype(np.float32))
    if not sequences:
        return np.empty(
            (0, sequence_minutes, len(SEQUENCE_FEATURE_COLUMNS)), dtype=np.float32
        )
    return np.stack(sequences)
