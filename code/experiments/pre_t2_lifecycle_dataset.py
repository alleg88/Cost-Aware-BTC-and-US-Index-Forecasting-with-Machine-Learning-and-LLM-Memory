"""Causal T1 lifecycle decisions before and after Fast-T2 confirmation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.five_minute_two_trigger_windows import (
    TwoTriggerConfig,
    _touch_side,
)


BASE_FEATURE_COLUMNS = (
    "side_sign",
    "channel_slope_bps_5m_side",
    "channel_r2",
    "channel_width_pct",
    "channel_confluence_count",
    "channel_position_side",
    "channel_age_minutes",
    "t1_edge_depth",
    "t1_range_bps",
    "t1_body_fraction",
    "t1_wick_share_side",
    "t1_close_location_side",
    "trigger_margin_bps_side",
    "trigger_velocity_1m_bps_side",
    "trigger_velocity_3m_bps_side",
    "t2_confirmed",
    "minutes_since_t1",
    "minutes_since_t2",
    "consecutive_closes_toward_trigger",
    "price_from_t1_bps_side",
    "since_t1_mfe_bps",
    "since_t1_mae_bps",
    "realized_vol_30m_bps",
    "distance_to_stop_bps",
    "distance_to_target_bps",
    "rr_proxy",
    "atr_15m_bps",
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
    "close_location_in_range_side",
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
class LifecycleDecisionConfig:
    sequence_minutes: int = 30
    post_t2_decision_minutes: int = 15
    max_hold_minutes: int = 120
    round_trip_cost_bps: float = 10.0
    min_risk_bps: float = 25.0


def _utc(value: object) -> pd.Timestamp:
    value = pd.Timestamp(value)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def _channel_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.sort_values("decision_time", kind="stable").reset_index(drop=True).copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True)
    midpoint = (work["channel_lower"].astype(float) + work["channel_upper"].astype(float)) / 2.0
    previous = midpoint.groupby(work["channel_episode_id"], sort=False).shift(1)
    inferred = (midpoint / previous - 1.0) * 10_000.0
    if "channel_slope_bps_5m" in work:
        supplied = pd.to_numeric(work["channel_slope_bps_5m"], errors="coerce")
        work["channel_slope_bps_5m_internal"] = supplied.fillna(inferred).fillna(0.0)
    else:
        work["channel_slope_bps_5m_internal"] = inferred.fillna(0.0)
    episode_start = work.groupby("channel_episode_id", sort=False)["decision_time"].transform("min")
    work["channel_age_minutes_internal"] = (
        (work["decision_time"] - episode_start) / pd.Timedelta(minutes=1)
    ).astype(float)
    return work


def _lifecycle_record(
    armed: dict[str, object], *, status: str, end_time: pd.Timestamp,
    t2_time: pd.Timestamp | pd.NaT = pd.NaT, confirmation_price: float = np.nan,
) -> dict[str, object]:
    side_sign = 1.0 if armed["side"] == "long" else -1.0
    row = armed["channel_row"]
    span = float(row.channel_upper) - float(row.channel_lower)
    candle_range = float(row.high) - float(row.low)
    body_low = min(float(row.open), float(row.close))
    body_high = max(float(row.open), float(row.close))
    wick = body_low - float(row.low) if side_sign > 0 else float(row.high) - body_high
    close_location = (
        (float(row.close) - float(row.low)) / candle_range
        if side_sign > 0
        else (float(row.high) - float(row.close)) / candle_range
    )
    edge_depth = (
        (float(row.channel_lower) - float(row.low)) / span
        if side_sign > 0
        else (float(row.high) - float(row.channel_upper)) / span
    )
    return {
        "arm_id": str(armed["arm_id"]),
        "side": str(armed["side"]),
        "status": status,
        "channel_episode_id": row.channel_episode_id,
        "t1_time": armed["time"],
        "t2_time": t2_time,
        "expiry_time": armed["expiry"],
        "lifecycle_end_time": end_time,
        "confirmation_price": float(confirmation_price),
        "confirmation_threshold": float(armed["threshold"]),
        "confirmation_lag_minutes": (
            float((t2_time - armed["time"]) / pd.Timedelta(minutes=1))
            if not pd.isna(t2_time) else np.nan
        ),
        "t1_open": float(row.open),
        "t1_high": float(row.high),
        "t1_low": float(row.low),
        "t1_close": float(row.close),
        "stop_price": float(row.low if side_sign > 0 else row.high),
        "target_price": float(row.channel_upper if side_sign > 0 else row.channel_lower),
        "channel_lower": float(row.channel_lower),
        "channel_upper": float(row.channel_upper),
        "channel_r2": float(row.channel_r2),
        "channel_confluence_count": float(getattr(row, "channel_confluence_count", 1.0)),
        "channel_slope_bps_5m_side": float(row.channel_slope_bps_5m_internal) * side_sign,
        "channel_age_minutes": float(row.channel_age_minutes_internal),
        "t1_edge_depth": float(edge_depth),
        "t1_range_bps": float(candle_range / float(row.close) * 10_000.0),
        "t1_body_fraction": float(abs(float(row.close) - float(row.open)) / candle_range),
        "t1_wick_share_side": float(wick / candle_range),
        "t1_close_location_side": float(close_location),
    }


def detect_t1_lifecycles(
    channel_frame: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    config: TwoTriggerConfig,
    confirmation_buffer_bps: float = 2.0,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Retain every armed T1, including setups that expire without T2."""
    if confirmation_buffer_bps < 0:
        raise ValueError("confirmation_buffer_bps must be non-negative")
    channels = _channel_enrichment(channel_frame)
    channel_times = pd.DatetimeIndex(channels["decision_time"])
    if channel_times.has_duplicates:
        raise ValueError("channel decision_time must be unique")
    touches: dict[pd.Timestamp, tuple[object, str]] = {}
    for row in channels.itertuples(index=False, name="ChannelRow"):
        side = _touch_side(row, config)
        if side is not None:
            touches[_utc(row.decision_time)] = (row, side)

    minute = minute_bars.sort_index(kind="stable")
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.tz is None:
        raise ValueError("minute bars need a timezone-aware DatetimeIndex")
    if minute.index.has_duplicates:
        raise ValueError("minute bars must be unique")
    decision_times = minute.index.tz_convert("UTC") + pd.Timedelta(minutes=1)
    horizon = pd.Timedelta(minutes=config.confirmation_bars * config.cadence_minutes)
    cooldown = pd.Timedelta(minutes=config.cooldown_minutes)
    buffer = confirmation_buffer_bps / 10_000.0
    audit = {
        "candidate_touches": int(len(touches)),
        "armed_t1": 0,
        "confirmed_t2": 0,
        "expired_t1": 0,
        "cooldown_touches": 0,
        "overlapping_touches": 0,
        "missing_minute_invalidations": 0,
        "right_censored_t1": 0,
    }
    records: list[dict[str, object]] = []
    armed: dict[str, object] | None = None
    cooldown_until: pd.Timestamp | None = None
    last_decision: pd.Timestamp | None = None

    for minute_row, now in zip(minute.itertuples(index=False), decision_times, strict=True):
        now = _utc(now)
        if armed is not None and last_decision is not None and now > last_decision + pd.Timedelta(minutes=1):
            records.append(_lifecycle_record(armed, status="invalidated", end_time=last_decision))
            audit["missing_minute_invalidations"] += 1
            armed = None

        if armed is not None:
            if now <= armed["expiry"]:
                confirmed = (
                    armed["side"] == "long" and float(minute_row.close) > float(armed["threshold"])
                ) or (
                    armed["side"] == "short" and float(minute_row.close) < float(armed["threshold"])
                )
                if confirmed:
                    records.append(
                        _lifecycle_record(
                            armed, status="confirmed", end_time=now,
                            t2_time=now, confirmation_price=float(minute_row.close),
                        )
                    )
                    audit["confirmed_t2"] += 1
                    cooldown_until = now + cooldown
                    armed = None
            if armed is not None and now >= armed["expiry"]:
                records.append(_lifecycle_record(armed, status="expired", end_time=now))
                audit["expired_t1"] += 1
                armed = None

        candidate = touches.get(now)
        if candidate is not None:
            channel_row, side = candidate
            if cooldown_until is not None and now < cooldown_until:
                audit["cooldown_touches"] += 1
            elif armed is not None:
                audit["overlapping_touches"] += 1
            else:
                threshold = (
                    float(channel_row.high) * (1.0 + buffer)
                    if side == "long"
                    else float(channel_row.low) * (1.0 - buffer)
                )
                armed = {
                    "arm_id": f"arm_{audit['armed_t1']:08d}",
                    "side": side,
                    "time": now,
                    "expiry": now + horizon,
                    "threshold": threshold,
                    "channel_row": channel_row,
                }
                audit["armed_t1"] += 1
        last_decision = now

    if armed is not None:
        if last_decision is not None and last_decision >= armed["expiry"]:
            records.append(_lifecycle_record(armed, status="expired", end_time=armed["expiry"]))
            audit["expired_t1"] += 1
        else:
            audit["right_censored_t1"] += 1
    lifecycles = pd.DataFrame(records)
    if not lifecycles.empty:
        lifecycles = lifecycles.sort_values("t1_time", kind="stable").reset_index(drop=True)
    return lifecycles, audit


def _completed_history(
    minute: pd.DataFrame, decision_time: pd.Timestamp, length: int
) -> pd.DataFrame | None:
    position = minute.index.get_indexer([decision_time])[0]
    if position < length:
        return None
    observed = minute.iloc[position - length:position]
    expected = pd.date_range(
        end=decision_time - pd.Timedelta(minutes=1), periods=length,
        freq="1min", tz="UTC",
    )
    if not observed.index.equals(expected):
        return None
    if observed[list(_MINUTE_COLUMNS)].isna().any().any():
        return None
    return observed


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _consecutive_toward(close: np.ndarray, side_sign: float) -> int:
    changes = np.diff(close) * side_sign
    count = 0
    for value in changes[::-1]:
        if value <= 0:
            break
        count += 1
    return count


def _features_and_sequence(
    lifecycle: object,
    history: pd.DataFrame,
    since_t1: pd.DataFrame,
    *,
    decision_time: pd.Timestamp,
) -> tuple[dict[str, float], np.ndarray]:
    side_sign = 1.0 if lifecycle.side == "long" else -1.0
    close = history["close"].to_numpy(float)
    high = history["high"].to_numpy(float)
    low = history["low"].to_numpy(float)
    volume = history["volume"].to_numpy(float)
    count = history["count"].to_numpy(float)
    open_ = history["open"].to_numpy(float)
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    current = float(close[-1])
    span = float(lifecycle.channel_upper - lifecycle.channel_lower)
    position = (
        (current - lifecycle.channel_lower) / span
        if side_sign > 0 else (lifecycle.channel_upper - current) / span
    )
    stop_distance = (current - lifecycle.stop_price) * side_sign
    target_distance = (lifecycle.target_price - current) * side_sign
    trigger_margin = (current / lifecycle.confirmation_threshold - 1.0) * 10_000.0 * side_sign
    t2_confirmed = not pd.isna(lifecycle.t2_time) and decision_time >= lifecycle.t2_time
    since_t2 = (
        float((decision_time - lifecycle.t2_time) / pd.Timedelta(minutes=1))
        if t2_confirmed else -1.0
    )
    observed_high = since_t1["high"].to_numpy(float) if len(since_t1) else np.array([lifecycle.t1_high])
    observed_low = since_t1["low"].to_numpy(float) if len(since_t1) else np.array([lifecycle.t1_low])
    mfe = (
        observed_high.max() / lifecycle.t1_close - 1.0
        if side_sign > 0 else lifecycle.t1_close / observed_low.min() - 1.0
    )
    mae = (
        observed_low.min() / lifecycle.t1_close - 1.0
        if side_sign > 0 else lifecycle.t1_close / observed_high.max() - 1.0
    )
    taker = np.divide(
        2.0 * history["taker_buy_base"].to_numpy(float), volume,
        out=np.ones_like(volume), where=volume > 0,
    ) - 1.0
    previous_close = np.r_[open_[0], close[:-1]]
    true_range = np.maximum.reduce(
        [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    candle_range = high - low
    close_location = np.divide(
        close - low, candle_range, out=np.full_like(close, 0.5), where=candle_range > 0,
    )
    if side_sign < 0:
        close_location = 1.0 - close_location
    hour = decision_time.hour + decision_time.minute / 60.0
    values = {
        "side_sign": side_sign,
        "channel_slope_bps_5m_side": float(lifecycle.channel_slope_bps_5m_side),
        "channel_r2": float(lifecycle.channel_r2),
        "channel_width_pct": span / ((lifecycle.channel_upper + lifecycle.channel_lower) / 2.0),
        "channel_confluence_count": float(lifecycle.channel_confluence_count),
        "channel_position_side": float(position),
        "channel_age_minutes": float(lifecycle.channel_age_minutes),
        "t1_edge_depth": float(lifecycle.t1_edge_depth),
        "t1_range_bps": float(lifecycle.t1_range_bps),
        "t1_body_fraction": float(lifecycle.t1_body_fraction),
        "t1_wick_share_side": float(lifecycle.t1_wick_share_side),
        "t1_close_location_side": float(lifecycle.t1_close_location_side),
        "trigger_margin_bps_side": float(trigger_margin),
        "trigger_velocity_1m_bps_side": float(returns[-1] * 10_000.0 * side_sign),
        "trigger_velocity_3m_bps_side": float(np.log(close[-1] / close[-4]) * 10_000.0 * side_sign),
        "t2_confirmed": float(t2_confirmed),
        "minutes_since_t1": float((decision_time - lifecycle.t1_time) / pd.Timedelta(minutes=1)),
        "minutes_since_t2": since_t2,
        "consecutive_closes_toward_trigger": float(_consecutive_toward(close[-6:], side_sign)),
        "price_from_t1_bps_side": float((current / lifecycle.t1_close - 1.0) * 10_000.0 * side_sign),
        "since_t1_mfe_bps": float(mfe * 10_000.0),
        "since_t1_mae_bps": float(mae * 10_000.0),
        "realized_vol_30m_bps": float(np.std(returns[1:], ddof=1) * 10_000.0),
        "distance_to_stop_bps": float(stop_distance / current * 10_000.0),
        "distance_to_target_bps": float(target_distance / current * 10_000.0),
        "rr_proxy": float(target_distance / stop_distance if stop_distance > 0 else 0.0),
        "atr_15m_bps": float(true_range[-15:].mean() / current * 10_000.0),
        "return_1m_side": float(returns[-1] * 10_000.0 * side_sign),
        "return_5m_side": float(np.log(close[-1] / close[-6]) * 10_000.0 * side_sign),
        "volume_ratio_5_20": _safe_ratio(float(volume[-5:].mean()), float(volume[-25:-5].mean())),
        "trade_count_ratio_5_20": _safe_ratio(float(count[-5:].mean()), float(count[-25:-5].mean())),
        "taker_imbalance_5_side": float(taker[-5:].mean() * side_sign),
        "hour_sin": float(np.sin(2.0 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2.0 * np.pi * hour / 24.0)),
    }
    sequence = np.column_stack(
        [
            returns * 10_000.0 * side_sign,
            candle_range / close * 10_000.0,
            close_location,
            taker * side_sign,
            np.log1p(volume),
        ]
    ).astype(np.float32)
    return values, sequence


def build_lifecycle_decisions(
    lifecycles: pd.DataFrame,
    channel_frame: pd.DataFrame,
    minute_bars: pd.DataFrame,
    config: LifecycleDecisionConfig = LifecycleDecisionConfig(),
) -> tuple[pd.DataFrame, np.ndarray, dict[str, int]]:
    """Expand T1 arms into causal decision rows and fixed-geometry labels."""
    del channel_frame  # Geometry is frozen and stored on each lifecycle row.
    minute = minute_bars.sort_index(kind="stable").copy()
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.tz is None:
        raise ValueError("minute bars need a timezone-aware DatetimeIndex")
    minute.index = minute.index.tz_convert("UTC")
    minute_index = minute.index
    open_values = minute["open"].to_numpy(float)
    high_values = minute["high"].to_numpy(float)
    low_values = minute["low"].to_numpy(float)
    close_values = minute["close"].to_numpy(float)
    rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    audit = {
        "lifecycles": int(len(lifecycles)),
        "labelled_rows": 0,
        "history_censored": 0,
        "future_censored": 0,
        "stop_closed_arms": 0,
        "risk_rejected_rows": 0,
    }
    for lifecycle in lifecycles.sort_values("t1_time", kind="stable").itertuples(index=False):
        t1_time = _utc(lifecycle.t1_time)
        t1_position = int(minute_index.get_indexer([t1_time])[0])
        if t1_position < 0:
            audit["history_censored"] += 1
            continue
        t2_time = pd.NaT if pd.isna(lifecycle.t2_time) else _utc(lifecycle.t2_time)
        expiry = _utc(lifecycle.expiry_time)
        if lifecycle.status == "confirmed":
            pre_times = pd.date_range(t1_time, t2_time - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
            post_times = pd.date_range(t2_time, periods=config.post_t2_decision_minutes, freq="1min", tz="UTC")
            decisions = pre_times.append(post_times)
        elif lifecycle.status == "expired":
            decisions = pd.date_range(t1_time, expiry - pd.Timedelta(minutes=1), freq="1min", tz="UTC")
        else:
            continue
        arm_stopped = False
        for decision_time in decisions:
            decision_position = int(minute_index.get_indexer([decision_time])[0])
            if decision_position < 0:
                audit["history_censored"] += 1
                continue
            prior = minute.iloc[t1_position:decision_position]
            if decision_position > t1_position:
                stop_hit = (
                    prior["low"].le(lifecycle.stop_price).any()
                    if lifecycle.side == "long"
                    else prior["high"].ge(lifecycle.stop_price).any()
                )
                if stop_hit:
                    arm_stopped = True
                    audit["stop_closed_arms"] += 1
                    break
            history = _completed_history(minute, decision_time, config.sequence_minutes)
            if history is None:
                audit["history_censored"] += 1
                continue
            since_t1 = prior
            features, sequence = _features_and_sequence(
                lifecycle, history, since_t1, decision_time=decision_time
            )
            if (
                not np.isfinite(list(features.values())).all()
                or features["distance_to_stop_bps"] < config.min_risk_bps
                or features["distance_to_target_bps"] <= 0.0
                or features["rr_proxy"] <= 0.0
            ):
                audit["risk_rejected_rows"] += 1
                continue
            path_end = decision_position + config.max_hold_minutes
            if (
                path_end > len(minute)
                or minute_index[path_end - 1]
                != decision_time + pd.Timedelta(minutes=config.max_hold_minutes - 1)
            ):
                audit["future_censored"] += 1
                continue
            entry_price = float(open_values[decision_position])
            stop_price = float(lifecycle.stop_price)
            target_price = float(lifecycle.target_price)
            risk = (
                entry_price - stop_price
                if lifecycle.side == "long" else stop_price - entry_price
            )
            reward = (
                target_price - entry_price
                if lifecycle.side == "long" else entry_price - target_price
            )
            if risk <= 0.0 or reward <= 0.0:
                audit["risk_rejected_rows"] += 1
                continue
            path_high = high_values[decision_position:path_end]
            path_low = low_values[decision_position:path_end]
            if lifecycle.side == "long":
                stop_hits = np.flatnonzero(path_low <= stop_price)
                target_hits = np.flatnonzero(path_high >= target_price)
            else:
                stop_hits = np.flatnonzero(path_high >= stop_price)
                target_hits = np.flatnonzero(path_low <= target_price)
            stop_offset = int(stop_hits[0]) if len(stop_hits) else config.max_hold_minutes
            target_offset = int(target_hits[0]) if len(target_hits) else config.max_hold_minutes
            if stop_offset <= target_offset and stop_offset < config.max_hold_minutes:
                exit_offset, exit_price, outcome = stop_offset, stop_price, "sl"
            elif target_offset < config.max_hold_minutes:
                exit_offset, exit_price, outcome = target_offset, target_price, "tp"
            else:
                exit_offset = config.max_hold_minutes - 1
                exit_price = float(close_values[decision_position + exit_offset])
                outcome = "timeout"
            exit_time = minute_index[decision_position + exit_offset]
            side_sign = 1.0 if lifecycle.side == "long" else -1.0
            cost = config.round_trip_cost_bps / 10_000.0 * entry_price
            r_net = float((exit_price - entry_price) * side_sign / risk - cost / risk)
            phase = "post_t2" if not pd.isna(t2_time) and decision_time >= t2_time else "pre_t2"
            decision_number = int((decision_time - t1_time) / pd.Timedelta(minutes=1))
            planned_cost_r = config.round_trip_cost_bps / features["distance_to_stop_bps"]
            outcome_class = {"sl": 0, "tp": 1, "timeout": 2}[outcome]
            rows.append(
                {
                    "arm_id": str(lifecycle.arm_id),
                    "window_id": str(lifecycle.arm_id),
                    "decision_id": f"{lifecycle.arm_id}:{decision_number:02d}",
                    "side": str(lifecycle.side),
                    "channel_episode_id": lifecycle.channel_episode_id,
                    "lifecycle_status": str(lifecycle.status),
                    "decision_phase": phase,
                    "t1_time": t1_time,
                    "t2_time": t2_time,
                    "decision_time": decision_time,
                    "entry_time": decision_time,
                    "label_start": decision_time,
                    "label_end": exit_time,
                    "active_end_time": exit_time,
                    "entry_price": entry_price,
                    "stop_price": stop_price,
                    "target_price": target_price,
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "outcome": outcome,
                    "outcome_class": outcome_class,
                    "r_net": r_net,
                    "observed_gross_r": r_net + planned_cost_r,
                    "planned_cost_r_10bps": planned_cost_r,
                    "planned_tp_gross_r": features["rr_proxy"],
                    "filled": True,
                    "holding_minutes": float((exit_time - decision_time) / pd.Timedelta(minutes=1)),
                    "stop_invalidated_before_decision": False,
                    **features,
                }
            )
            sequences.append(sequence)
        if arm_stopped:
            continue
    decisions_frame = pd.DataFrame(rows)
    if not decisions_frame.empty:
        decisions_frame = decisions_frame.sort_values(
            ["decision_time", "decision_id"], kind="stable"
        ).reset_index(drop=True)
        order = {row["decision_id"]: pos for pos, row in enumerate(rows)}
        sequence_order = [order[value] for value in decisions_frame["decision_id"]]
        sequence_array = np.stack(sequences)[sequence_order]
        if not decisions_frame["decision_id"].is_unique:
            raise AssertionError("decision IDs must be unique")
    else:
        sequence_array = np.empty(
            (0, config.sequence_minutes, len(SEQUENCE_FEATURE_COLUMNS)), dtype=np.float32
        )
    audit["labelled_rows"] = int(len(decisions_frame))
    return decisions_frame, sequence_array, audit
