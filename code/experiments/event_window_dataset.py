"""Causal pooled tensors for broad event-window entry decisions."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from features.event_window_inputs import merge_positioning_asof, known_structural_stop
from features.event_windows import EventWindowConfig


SEQUENCE_FEATURES = (
    "log_return_side", "cumulative_return_side", "range_bps",
    "body_bps_side", "lower_wick_fraction", "upper_wick_fraction",
    "volume_log_ratio_24", "quote_volume_log_ratio_24",
    "trade_count_log_ratio_24", "taker_imbalance_side",
    "taker_imbalance_mean_3_side", "taker_imbalance_delta_side",
    "realized_vol_12", "activity_ratio", "channel_position_side",
    "distance_adverse_rail_bps", "distance_mid_bps_side",
    "distance_favourable_rail_bps", "pre_window_mask", "active_window_mask",
    "bar_complete", "cadence_gap",
)

CONTEXT_FEATURES = (
    "side_sign", "window_reason_code", "window_age_fraction",
    "time_remaining_fraction", "channel_regime_age_hours",
    "channel_slope_side", "channel_r2", "channel_width_pct",
    "channel_confluence", "channel_confluence_count",
    "channel_sign_agree_90", "channel_sign_agree_120", "rsi_channel_side",
    "volatility_regime_percentile", "running_favourable_excursion_bps",
    "running_recovery_bps", "bars_since_adverse_extreme", "retest_count",
    "oi_chg_15m_side", "oi_chg_1h_side", "oi_chg_4h_side",
    "oi_accel_1h_side", "oi_z_7d", "price_oi_interaction",
    "funding_rate_side", "funding_z_side", "toptrader_log_ratio_side",
    "taker_log_ratio_side", "oi_missing", "funding_missing",
    "toptrader_missing", "taker_ratio_missing", "positioning_missing",
    "positioning_stale", "positioning_age_log",
    "structural_risk_bps_known", "rail_room_r_known",
)

_RAW_SEQUENCE = {
    "range_bps": "range_bps",
    "lower_wick_fraction": "lower_wick_fraction",
    "upper_wick_fraction": "upper_wick_fraction",
    "volume_log_ratio_24": "volume_log_ratio_24",
    "quote_volume_log_ratio_24": "quote_volume_log_ratio_24",
    "trade_count_log_ratio_24": "trade_count_log_ratio_24",
    "realized_vol_12": "realized_vol_12",
    "activity_ratio": "activity_ratio",
    "bar_complete": "bar_complete",
    "cadence_gap": "cadence_gap",
}


@dataclass(frozen=True)
class EventWindowSequences:
    metadata: pd.DataFrame
    sequence: np.ndarray
    context: np.ndarray
    source_bar_times: np.ndarray
    decision_times: np.ndarray
    sequence_valid: np.ndarray
    decision_valid: np.ndarray
    sequence_features: tuple[str, ...] = SEQUENCE_FEATURES
    context_features: tuple[str, ...] = CONTEXT_FEATURES


def _utc_index(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="raise")
    if not out.index.is_monotonic_increasing or out.index.has_duplicates:
        raise ValueError(f"{name} index must be sorted and unique")
    return out


def _empty(metadata_columns: pd.Index | list[str]) -> EventWindowSequences:
    return EventWindowSequences(
        metadata=pd.DataFrame(columns=metadata_columns),
        sequence=np.empty((0, 36, len(SEQUENCE_FEATURES)), dtype=np.float32),
        context=np.empty((0, 12, len(CONTEXT_FEATURES)), dtype=np.float32),
        source_bar_times=np.empty((0, 12), dtype="datetime64[ns]"),
        decision_times=np.empty((0, 12), dtype="datetime64[ns]"),
        sequence_valid=np.empty((0, 36), dtype=bool),
        decision_valid=np.empty((0, 12), dtype=bool),
    )


def _number(row: pd.Series, name: str, default: float = np.nan) -> float:
    value = row.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _rolling_percentile(values: pd.Series, window: int = 288) -> pd.Series:
    return values.astype(float).rolling(window, min_periods=window).rank(pct=True)


def _sequence_row(
    raw: pd.Series,
    *,
    side_sign: float,
    cumulative: float,
    pre_window: bool,
) -> np.ndarray:
    values = {name: _number(raw, source) for name, source in _RAW_SEQUENCE.items()}
    values.update(
        {
            "log_return_side": side_sign * _number(raw, "log_return"),
            "cumulative_return_side": cumulative,
            "body_bps_side": side_sign * _number(raw, "body_bps"),
            "taker_imbalance_side": side_sign * _number(raw, "taker_imbalance"),
            "taker_imbalance_mean_3_side": side_sign * _number(raw, "taker_imbalance_mean_3"),
            "taker_imbalance_delta_side": side_sign * _number(raw, "taker_imbalance_delta"),
            "channel_position_side": (
                _number(raw, "channel_pos")
                if side_sign > 0
                else 1.0 - _number(raw, "channel_pos")
            ),
            "distance_adverse_rail_bps": (
                _number(raw, "distance_lower_bps")
                if side_sign > 0
                else -_number(raw, "distance_upper_bps")
            ),
            "distance_mid_bps_side": side_sign * _number(raw, "distance_mid_bps"),
            "distance_favourable_rail_bps": (
                -_number(raw, "distance_upper_bps")
                if side_sign > 0
                else _number(raw, "distance_lower_bps")
            ),
            "pre_window_mask": float(pre_window),
            "active_window_mask": float(not pre_window),
        }
    )
    return np.asarray([values[name] for name in SEQUENCE_FEATURES], dtype=np.float32)


def _context_row(
    raw: pd.Series,
    positioning: pd.Series,
    *,
    side_sign: float,
    step: int,
    volatility_percentile: float,
    favourable_excursion: float,
    recovery: float,
    bars_since_adverse: int,
    retest_count: int,
    price_return_1h_side: float,
    risk_bps: float,
    rail_room_r: float,
) -> np.ndarray:
    primary_sign = np.sign(_number(raw, "channel_slope_raw", _number(raw, "channel_slope_bps")))
    rsi = _number(raw, "rsi_channel")
    age = _number(positioning, "positioning_age_min")
    oi_1h = _number(positioning, "oi_chg_1h")
    values = {
        "side_sign": side_sign,
        "window_reason_code": 0.0 if side_sign > 0 else 1.0,
        "window_age_fraction": (step + 1.0) / 12.0,
        "time_remaining_fraction": (11.0 - step) / 12.0,
        "channel_regime_age_hours": _number(raw, "channel_regime_age_hours"),
        "channel_slope_side": side_sign * _number(raw, "channel_slope_bps"),
        "channel_r2": _number(raw, "channel_r2"),
        "channel_width_pct": _number(raw, "channel_width") / _number(raw, "close") * 100.0,
        "channel_confluence": _number(raw, "channel_confluence"),
        "channel_confluence_count": _number(raw, "channel_confluence_count"),
        "channel_sign_agree_90": float(np.sign(_number(raw, "channel_sign_90")) == primary_sign),
        "channel_sign_agree_120": float(np.sign(_number(raw, "channel_sign_120")) == primary_sign),
        "rsi_channel_side": rsi if side_sign > 0 else 100.0 - rsi,
        "volatility_regime_percentile": volatility_percentile,
        "running_favourable_excursion_bps": favourable_excursion,
        "running_recovery_bps": recovery,
        "bars_since_adverse_extreme": float(bars_since_adverse),
        "retest_count": float(retest_count),
        "oi_chg_15m_side": side_sign * _number(positioning, "oi_chg_15m"),
        "oi_chg_1h_side": side_sign * oi_1h,
        "oi_chg_4h_side": side_sign * _number(positioning, "oi_chg_4h"),
        "oi_accel_1h_side": side_sign * _number(positioning, "oi_accel_1h"),
        "oi_z_7d": _number(positioning, "oi_z_7d"),
        "price_oi_interaction": price_return_1h_side * oi_1h,
        "funding_rate_side": side_sign * _number(positioning, "funding_rate"),
        "funding_z_side": side_sign * _number(positioning, "funding_z"),
        "toptrader_log_ratio_side": side_sign * _number(positioning, "toptrader_log_ratio"),
        "taker_log_ratio_side": side_sign * _number(positioning, "taker_log_ratio"),
        "oi_missing": _number(positioning, "oi_missing", 1.0),
        "funding_missing": _number(positioning, "funding_missing", 1.0),
        "toptrader_missing": _number(positioning, "toptrader_missing", 1.0),
        "taker_ratio_missing": _number(positioning, "taker_ratio_missing", 1.0),
        "positioning_missing": _number(positioning, "positioning_missing", 1.0),
        "positioning_stale": _number(positioning, "positioning_stale", 1.0),
        "positioning_age_log": np.log1p(max(age, 0.0)) if np.isfinite(age) else np.nan,
        "structural_risk_bps_known": risk_bps,
        "rail_room_r_known": rail_room_r,
    }
    return np.asarray([values[name] for name in CONTEXT_FEATURES], dtype=np.float32)


def build_event_window_sequences(
    manifest: pd.DataFrame,
    five_minute_features: pd.DataFrame,
    positioning_features: pd.DataFrame,
    config: EventWindowConfig = EventWindowConfig(),
) -> EventWindowSequences:
    """Return one pooled, causally masked tensor sample per opportunity window."""
    if config.pre_context_bars != 24 or config.active_bars != 12:
        raise ValueError("the frozen tensor contract requires 24 pre-window and 12 active bars")
    if manifest.empty:
        return _empty(manifest.columns)

    five = _utc_index(five_minute_features, name="five_minute_features")
    positioning = _utc_index(positioning_features, name="positioning_features")
    volatility_percentile = _rolling_percentile(five["realized_vol_12"])
    price_return_1h = np.log(five["close"].astype(float)).diff(12)
    cadence = pd.Timedelta(config.bar)
    positioning_at_decision = merge_positioning_asof(
        pd.DataFrame({"decision_time": five.index + cadence}),
        positioning,
    ).set_index("decision_time")
    if not positioning_at_decision.index.is_unique:
        raise ValueError("decision grid must be unique for positioning carry")

    metadata_rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    source_times_rows: list[np.ndarray] = []
    decision_times_rows: list[np.ndarray] = []
    sequence_masks: list[np.ndarray] = []
    decision_masks: list[np.ndarray] = []

    for record in manifest.to_dict("records"):
        if record.get("side") not in {"long", "short"}:
            raise ValueError(f"unknown window side: {record.get('side')!r}")
        start = pd.Timestamp(record["window_start"])
        end = pd.Timestamp(record["window_end"])
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        trigger_source = start - cadence
        recorded_source = pd.Timestamp(record["source_bar_time"])
        recorded_source = (
            recorded_source.tz_localize("UTC")
            if recorded_source.tzinfo is None
            else recorded_source.tz_convert("UTC")
        )
        if recorded_source != trigger_source:
            raise ValueError("window_start must equal source_bar_time plus one 5m cadence")
        expected = pd.date_range(
            trigger_source - config.pre_context_bars * cadence,
            periods=config.pre_context_bars + config.active_bars,
            freq=cadence,
        )
        pre = expected[: config.pre_context_bars]
        if not pre.isin(five.index).all():
            continue
        pre_frame = five.loc[pre]
        if not pre_frame.get("bar_complete", pd.Series(1, index=pre)).astype(bool).all():
            continue
        if pre_frame.get("cadence_gap", pd.Series(0, index=pre)).astype(bool).any():
            continue

        active_sources = expected[config.pre_context_bars :]
        decisions = active_sources + cadence
        decision_valid = np.zeros(config.active_bars, dtype=bool)
        for step, (source_time, decision_time) in enumerate(zip(active_sources, decisions, strict=True)):
            if decision_time >= end or source_time not in five.index:
                break
            raw = five.loc[source_time]
            if not bool(raw.get("bar_complete", 1)) or bool(raw.get("cadence_gap", 0)):
                break
            decision_valid[step] = True

        sequence_valid = np.zeros(config.pre_context_bars + config.active_bars, dtype=bool)
        sequence_valid[: config.pre_context_bars] = True
        sequence_valid[config.pre_context_bars : config.pre_context_bars + decision_valid.sum()] = True
        tensor = np.full((len(expected), len(SEQUENCE_FEATURES)), np.nan, dtype=np.float32)
        first_close = float(five.loc[expected[0], "close"])
        for position in np.flatnonzero(sequence_valid):
            raw = five.loc[expected[position]]
            cumulative = float(record["side"] == "long") * 2.0 - 1.0
            cumulative *= np.log(_number(raw, "close") / first_close)
            tensor[position] = _sequence_row(
                raw,
                side_sign=1.0 if record["side"] == "long" else -1.0,
                cumulative=cumulative,
                pre_window=position < config.pre_context_bars,
            )

        valid_sources = active_sources[decision_valid]
        valid_decisions = decisions[decision_valid]
        joined = positioning_at_decision.reindex(valid_decisions).reset_index()
        context = np.full((config.active_bars, len(CONTEXT_FEATURES)), np.nan, dtype=np.float32)
        side_sign = 1.0 if record["side"] == "long" else -1.0
        trigger_close = float(five.loc[trigger_source, "close"])
        signed_path: list[float] = []
        retests = 0
        for step, (source_time, decision_time) in enumerate(
            zip(valid_sources, valid_decisions, strict=True)
        ):
            raw = five.loc[source_time]
            signed_path.append(side_sign * np.log(_number(raw, "close") / trigger_close) * 1e4)
            path = np.asarray(signed_path, dtype=float)
            adverse_at = int(np.nanargmin(path))
            favourable = max(0.0, float(np.nanmax(path)))
            recovery = float(path[-1] - path[adverse_at])
            if side_sign > 0:
                touched = _number(raw, "low") <= _number(raw, "channel_lower") * 1.0005
            else:
                touched = _number(raw, "high") >= _number(raw, "channel_upper") * 0.9995
            retests += int(touched)
            stop_history = five.loc[
                source_time - 11 * cadence : source_time
            ]
            stop = known_structural_stop(
                stop_history,
                side=record["side"],
                source_bar_time=source_time,
                lookback=12,
                buffer_bps=5.0,
            )
            source_close = _number(raw, "close")
            risk_price = source_close - stop if side_sign > 0 else stop - source_close
            risk_bps = risk_price / source_close * 1e4 if risk_price > 0 else np.nan
            room = (
                _number(raw, "channel_upper") - source_close
                if side_sign > 0
                else source_close - _number(raw, "channel_lower")
            )
            rail_room_r = room / risk_price if risk_price > 0 else np.nan
            context[step] = _context_row(
                raw,
                joined.iloc[step],
                side_sign=side_sign,
                step=step,
                volatility_percentile=float(volatility_percentile.loc[source_time]),
                favourable_excursion=favourable,
                recovery=recovery,
                bars_since_adverse=step - adverse_at,
                retest_count=retests,
                price_return_1h_side=side_sign * float(price_return_1h.loc[source_time]),
                risk_bps=risk_bps,
                rail_room_r=rail_room_r,
            )

        metadata_rows.append(record)
        sequences.append(tensor)
        contexts.append(context)
        source_times_rows.append(active_sources.tz_localize(None).to_numpy(dtype="datetime64[ns]"))
        decision_times_rows.append(decisions.tz_localize(None).to_numpy(dtype="datetime64[ns]"))
        sequence_masks.append(sequence_valid)
        decision_masks.append(decision_valid)

    if not metadata_rows:
        return _empty(manifest.columns)
    return EventWindowSequences(
        metadata=pd.DataFrame(metadata_rows).reset_index(drop=True),
        sequence=np.stack(sequences).astype(np.float32, copy=False),
        context=np.stack(contexts).astype(np.float32, copy=False),
        source_bar_times=np.stack(source_times_rows),
        decision_times=np.stack(decision_times_rows),
        sequence_valid=np.stack(sequence_masks),
        decision_valid=np.stack(decision_masks),
    )
