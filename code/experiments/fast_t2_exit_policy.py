"""Causal early-exit states and hard-exit-first replay for Fast-T2 trades."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.two_trigger_execution import TradeResolution, resolve_fixed_trade
from experiments.fast_t2_entry_dataset import SEQUENCE_FEATURE_COLUMNS


EXIT_FEATURE_COLUMNS = (
    "side_sign",
    "minutes_held",
    "unrealized_r",
    "distance_to_stop_r",
    "distance_to_target_r",
    "return_since_entry_bps_side",
    "mfe_since_entry_bps",
    "mae_since_entry_bps",
    "realized_vol_30m_bps",
    "return_1m_side",
    "return_5m_side",
    "volume_ratio_5_20",
    "trade_count_ratio_5_20",
    "taker_imbalance_5_side",
    "channel_r2",
    "channel_width_pct",
    "hour_sin",
    "hour_cos",
)
EXIT_SEQUENCE_FEATURE_COLUMNS = SEQUENCE_FEATURE_COLUMNS

_MINUTE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "taker_buy_base",
    "count",
)
_ENTRY_REQUIRED = frozenset(
    {
        "decision_id",
        "window_id",
        "side",
        "channel_episode_id",
        "entry_time",
        "stop_price",
        "target_price",
        "channel_r2",
        "channel_width_pct",
    }
)


@dataclass(frozen=True)
class ExitPolicyConfig:
    sequence_minutes: int = 30
    max_hold_minutes: int = 120
    round_trip_cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if self.sequence_minutes < 6:
            raise ValueError("sequence_minutes must be at least six")
        if self.max_hold_minutes < 2:
            raise ValueError("max_hold_minutes must be at least two")
        if self.round_trip_cost_bps < 0:
            raise ValueError("round_trip_cost_bps cannot be negative")


def _utc(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _validate_minute(minute_bars: pd.DataFrame) -> pd.DataFrame:
    minute = minute_bars.sort_index(kind="stable")
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.tz is None:
        raise ValueError("minute bars need a timezone-aware DatetimeIndex")
    missing = sorted(set(_MINUTE_COLUMNS).difference(minute.columns))
    if missing:
        raise ValueError(f"minute bars missing exit columns: {missing}")
    if minute.index.has_duplicates:
        raise ValueError("minute bars cannot contain duplicate timestamps")
    return minute


def _validate_entries(entries: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_ENTRY_REQUIRED.difference(entries.columns))
    if missing:
        raise ValueError(f"entries missing exit-policy columns: {missing}")
    work = entries.copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="raise")
    if not work["decision_id"].is_unique:
        raise ValueError("entry decision_id must be unique")
    if "filled" in work:
        work = work[work["filled"].astype(bool)].copy()
    return work.reset_index(drop=True)


def _history(
    minute: pd.DataFrame, boundary: pd.Timestamp, length: int
) -> pd.DataFrame | None:
    boundary = _utc(boundary)
    position = minute.index.get_indexer([boundary])[0]
    if position < length:
        return None
    observed = minute.iloc[position - length:position]
    expected = pd.date_range(
        end=boundary - pd.Timedelta(minutes=1),
        periods=length,
        freq="1min",
        tz="UTC",
    )
    if not observed.index.equals(expected):
        return None
    if observed[list(_MINUTE_COLUMNS)].isna().any().any():
        return None
    return observed


def _trade_path(
    minute: pd.DataFrame, entry_time: pd.Timestamp, max_hold_minutes: int
) -> pd.DataFrame | None:
    entry_time = _utc(entry_time)
    position = minute.index.get_indexer([entry_time])[0]
    if position < 0:
        return None
    path = minute.iloc[position:position + max_hold_minutes]
    expected = pd.date_range(
        entry_time, periods=max_hold_minutes, freq="1min", tz="UTC"
    )
    if not path.index.equals(expected):
        return None
    if path[list(_MINUTE_COLUMNS)].isna().any().any():
        return None
    return path


def _hard_hit(row: object, side_sign: float, stop: float, target: float) -> str | None:
    stop_hit = float(row.low) <= stop if side_sign > 0 else float(row.high) >= stop
    target_hit = (
        float(row.high) >= target if side_sign > 0 else float(row.low) <= target
    )
    if stop_hit:
        return "sl"
    if target_hit:
        return "tp"
    return None


def _net_r(
    *,
    entry_price: float,
    exit_price: float,
    side_sign: float,
    risk: float,
    round_trip_cost_bps: float,
) -> float:
    cost = round_trip_cost_bps / 10_000.0 * entry_price
    move = (exit_price - entry_price) * side_sign
    return float(move / risk - cost / risk)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _state_features(
    entry: object,
    baseline: TradeResolution,
    history: pd.DataFrame,
    path_so_far: pd.DataFrame,
    *,
    boundary: pd.Timestamp,
    config: ExitPolicyConfig,
) -> dict[str, float]:
    side_sign = 1.0 if str(entry.side) == "long" else -1.0
    entry_price = float(baseline.entry_price)
    stop = float(baseline.stop_price)
    target = float(baseline.target_price)
    risk = (entry_price - stop) if side_sign > 0 else (stop - entry_price)
    current = float(history.iloc[-1].close)
    close = history["close"].to_numpy(dtype=float)
    volume = history["volume"].to_numpy(dtype=float)
    count = history["count"].to_numpy(dtype=float)
    returns = np.diff(np.log(close))
    taker = np.divide(
        2.0 * history["taker_buy_base"].to_numpy(dtype=float),
        volume,
        out=np.ones_like(volume),
        where=volume > 0,
    ) - 1.0
    high = path_so_far["high"].to_numpy(dtype=float)
    low = path_so_far["low"].to_numpy(dtype=float)
    if side_sign > 0:
        mfe = high.max() / entry_price - 1.0
        mae = low.min() / entry_price - 1.0
    else:
        mfe = entry_price / low.min() - 1.0
        mae = entry_price / high.max() - 1.0
    hour = boundary.hour + boundary.minute / 60.0
    values = {
        "side_sign": side_sign,
        "minutes_held": float(len(path_so_far)),
        "unrealized_r": _net_r(
            entry_price=entry_price,
            exit_price=current,
            side_sign=side_sign,
            risk=risk,
            round_trip_cost_bps=config.round_trip_cost_bps,
        ),
        "distance_to_stop_r": float((current - stop) * side_sign / risk),
        "distance_to_target_r": float((target - current) * side_sign / risk),
        "return_since_entry_bps_side": float(
            np.log(current / entry_price) * 10_000.0 * side_sign
        ),
        "mfe_since_entry_bps": float(mfe * 10_000.0),
        "mae_since_entry_bps": float(mae * 10_000.0),
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
        "channel_r2": float(entry.channel_r2),
        "channel_width_pct": float(entry.channel_width_pct),
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
    }
    if not np.isfinite(np.fromiter(values.values(), dtype=float)).all():
        raise ValueError(f"non-finite exit features for {entry.decision_id}")
    return values


def build_exit_states(
    entries: pd.DataFrame,
    minute_bars: pd.DataFrame,
    config: ExitPolicyConfig = ExitPolicyConfig(),
) -> pd.DataFrame:
    """Build one past-only HOLD/EXIT state after each eligible completed bar."""
    work = _validate_entries(entries)
    minute = _validate_minute(minute_bars)
    rows: list[dict[str, object]] = []
    for entry in work.itertuples(index=False):
        baseline = resolve_fixed_trade(
            minute,
            entry_time=entry.entry_time,
            side=str(entry.side),
            stop=float(entry.stop_price),
            target=float(entry.target_price),
            max_hold_minutes=config.max_hold_minutes,
            round_trip_cost_bps=config.round_trip_cost_bps,
        )
        path = _trade_path(minute, entry.entry_time, config.max_hold_minutes)
        if baseline is None or path is None or baseline.outcome == "entry_cancelled":
            continue
        side_sign = 1.0 if str(entry.side) == "long" else -1.0
        risk = (
            baseline.entry_price - baseline.stop_price
            if side_sign > 0
            else baseline.stop_price - baseline.entry_price
        )
        for offset, (timestamp, bar) in enumerate(path.iterrows()):
            if _hard_hit(bar, side_sign, baseline.stop_price, baseline.target_price):
                break
            if offset == config.max_hold_minutes - 1:
                break
            boundary = timestamp + pd.Timedelta(minutes=1)
            history = _history(minute, boundary, config.sequence_minutes)
            if history is None or boundary not in minute.index:
                break
            exit_now_price = float(minute.loc[boundary, "open"])
            exit_now_r = _net_r(
                entry_price=baseline.entry_price,
                exit_price=exit_now_price,
                side_sign=side_sign,
                risk=risk,
                round_trip_cost_bps=config.round_trip_cost_bps,
            )
            minutes_held = offset + 1
            record: dict[str, object] = {
                "state_id": f"{entry.decision_id}:exit:{minutes_held:03d}",
                "decision_id": str(entry.decision_id),
                "window_id": str(entry.window_id),
                "side": str(entry.side),
                "channel_episode_id": int(entry.channel_episode_id),
                "entry_time": baseline.entry_time,
                "entry_price": baseline.entry_price,
                "stop_price": baseline.stop_price,
                "target_price": baseline.target_price,
                "decision_time": boundary,
                "exit_decision_time": boundary,
                "exit_execute_time": boundary,
                "label_start": boundary,
                "label_end": baseline.exit_time,
                "active_end_time": baseline.exit_time,
                "exit_now_price": exit_now_price,
                "exit_now_r_net": exit_now_r,
                "baseline_exit_time": baseline.exit_time,
                "baseline_exit_price": baseline.exit_price,
                "baseline_outcome": baseline.outcome,
                "baseline_r_net": baseline.r_net,
                "continue_positive": int(baseline.r_net > exit_now_r),
                **_state_features(
                    entry,
                    baseline,
                    history,
                    path.iloc[: offset + 1],
                    boundary=boundary,
                    config=config,
                ),
            }
            if hasattr(entry, "fold_id"):
                record["entry_fold_id"] = str(entry.fold_id)
            if hasattr(entry, "score"):
                record["entry_score"] = float(entry.score)
            rows.append(record)
    states = pd.DataFrame(rows)
    if not states.empty:
        states = states.sort_values(
            ["decision_time", "state_id"], kind="stable"
        ).reset_index(drop=True)
        if not states["state_id"].is_unique:
            raise AssertionError("exit state IDs must be unique")
    return states


def build_exit_sequences(
    states: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    sequence_minutes: int = 30,
) -> np.ndarray:
    """Return the completed one-minute path immediately before each exit action."""
    minute = _validate_minute(minute_bars)
    sequences: list[np.ndarray] = []
    for state in states.itertuples(index=False):
        history = _history(minute, state.exit_decision_time, sequence_minutes)
        if history is None:
            raise ValueError(f"missing exit sequence for {state.state_id}")
        side_sign = float(state.side_sign)
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
            raise ValueError(f"non-finite exit sequence for {state.state_id}")
        sequences.append(values.astype(np.float32))
    if not sequences:
        return np.empty(
            (0, sequence_minutes, len(EXIT_SEQUENCE_FEATURE_COLUMNS)),
            dtype=np.float32,
        )
    return np.stack(sequences)


def replay_exit_policy(
    entries: pd.DataFrame,
    exit_scores: pd.DataFrame,
    *,
    hold_threshold: float,
    minute_bars: pd.DataFrame,
    config: ExitPolicyConfig = ExitPolicyConfig(),
    return_actions: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Replay HOLD/EXIT NOW after checking SL, TP, and timeout first."""
    if not np.isfinite(hold_threshold) or not 0.0 <= hold_threshold <= 1.0:
        raise ValueError("hold_threshold must be finite and inside [0, 1]")
    work = _validate_entries(entries)
    minute = _validate_minute(minute_bars)
    required = {"decision_id", "exit_decision_time", "score"}
    missing = sorted(required.difference(exit_scores.columns))
    if missing:
        raise ValueError(f"exit scores missing columns: {missing}")
    scores = exit_scores.copy()
    scores["exit_decision_time"] = pd.to_datetime(
        scores["exit_decision_time"], utc=True, errors="raise"
    )
    if scores.duplicated(["decision_id", "exit_decision_time"]).any():
        raise ValueError("exit scores must be unique per entry and boundary")
    lookup = {
        (str(row.decision_id), row.exit_decision_time): float(row.score)
        for row in scores.itertuples(index=False)
    }

    records: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    for entry in work.itertuples(index=False):
        baseline = resolve_fixed_trade(
            minute,
            entry_time=entry.entry_time,
            side=str(entry.side),
            stop=float(entry.stop_price),
            target=float(entry.target_price),
            max_hold_minutes=config.max_hold_minutes,
            round_trip_cost_bps=config.round_trip_cost_bps,
        )
        path = _trade_path(minute, entry.entry_time, config.max_hold_minutes)
        if baseline is None or path is None:
            raise ValueError(f"incomplete replay path for {entry.decision_id}")
        side_sign = 1.0 if str(entry.side) == "long" else -1.0
        risk = (
            baseline.entry_price - baseline.stop_price
            if side_sign > 0
            else baseline.stop_price - baseline.entry_price
        )
        exit_time = baseline.exit_time
        exit_price = baseline.exit_price
        outcome = baseline.outcome
        applied_score = np.nan
        if baseline.outcome != "entry_cancelled":
            for offset, (timestamp, bar) in enumerate(path.iterrows()):
                hard = _hard_hit(
                    bar, side_sign, baseline.stop_price, baseline.target_price
                )
                if hard is not None:
                    exit_time = timestamp
                    exit_price = (
                        baseline.stop_price if hard == "sl" else baseline.target_price
                    )
                    outcome = hard
                    break
                if offset == config.max_hold_minutes - 1:
                    exit_time = timestamp
                    exit_price = float(bar.close)
                    outcome = "timeout"
                    break
                boundary = timestamp + pd.Timedelta(minutes=1)
                key = (str(entry.decision_id), boundary)
                score = lookup.get(key, np.nan)
                action = "HOLD"
                if np.isfinite(score) and score < hold_threshold:
                    action = "EXIT NOW"
                actions.append(
                    {
                        "state_id": f"{entry.decision_id}:exit:{offset + 1:03d}",
                        "decision_id": str(entry.decision_id),
                        "exit_decision_time": boundary,
                        "score": score,
                        "score_available": bool(np.isfinite(score)),
                        "action": action,
                    }
                )
                if action == "EXIT NOW":
                    exit_time = boundary
                    exit_price = float(minute.loc[boundary, "open"])
                    outcome = "model_exit"
                    applied_score = score
                    break
        r_net = (
            0.0
            if baseline.outcome == "entry_cancelled"
            else _net_r(
                entry_price=baseline.entry_price,
                exit_price=float(exit_price),
                side_sign=side_sign,
                risk=risk,
                round_trip_cost_bps=config.round_trip_cost_bps,
            )
        )
        record: dict[str, object] = {
            "decision_id": str(entry.decision_id),
            "window_id": str(entry.window_id),
            "side": str(entry.side),
            "channel_episode_id": int(entry.channel_episode_id),
            "decision_time": (
                _utc(entry.decision_time)
                if hasattr(entry, "decision_time")
                else baseline.entry_time
            ),
            "entry_time": baseline.entry_time,
            "entry_price": baseline.entry_price,
            "stop_price": baseline.stop_price,
            "target_price": baseline.target_price,
            "exit_time": exit_time,
            "active_end_time": exit_time,
            "exit_price": float(exit_price),
            "outcome": outcome,
            "r_net": r_net,
            "baseline_r_net": baseline.r_net,
            "delta_r": float(r_net - baseline.r_net),
            "filled": baseline.outcome != "entry_cancelled",
            "holding_minutes": float(
                (exit_time - baseline.entry_time) / pd.Timedelta(minutes=1)
            ),
            "exit_score": applied_score,
        }
        if hasattr(entry, "fold_id"):
            record["fold_id"] = str(entry.fold_id)
        if hasattr(entry, "score"):
            record["score"] = float(entry.score)
            record["entry_score"] = float(entry.score)
        if hasattr(entry, "t2_time"):
            record["t2_time"] = _utc(entry.t2_time)
        if hasattr(entry, "minutes_since_t2"):
            record["minutes_since_t2"] = float(entry.minutes_since_t2)
        if hasattr(entry, "model"):
            record["model"] = str(entry.model)
        records.append(record)
    trades = pd.DataFrame(records)
    action_frame = pd.DataFrame(
        actions,
        columns=[
            "state_id",
            "decision_id",
            "exit_decision_time",
            "score",
            "score_available",
            "action",
        ],
    )
    return (trades, action_frame) if return_actions else trades
