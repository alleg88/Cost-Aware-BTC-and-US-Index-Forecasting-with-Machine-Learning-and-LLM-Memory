"""Causal 1h-channel projection and supervisor-style 5m entry triggers.

The hourly fit is usable only after its source bar closes. Its already-known log
slope is then projected through the next hour, so the lower-timeframe rails remain
sloped without reading the still-open hourly candle. Signals require three strictly
later completed 5m bars: location, rejection, then follow-through confirmation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SupervisorSignalConfig:
    """Frozen candle-sequence parameters for the Notebook J feasibility test."""

    edge_zone: float = 0.30
    midline_tolerance: float = 0.05
    arm_max_bars: int = 3
    confirm_max_bars: int = 2
    wick_frac: float = 0.30
    close_frac: float = 0.35
    stop_buffer_bps: float = 5.0


_CHANNEL_COLUMNS = (
    "channel_mid",
    "channel_upper",
    "channel_lower",
    "channel_slope",
    "channel_r2",
    "channel_regime",
    "channel_episode_id",
    "channel_confluence",
    "channel_confluence_count",
)


def project_closed_hourly_channel(
    hourly: pd.DataFrame,
    ltf: pd.DataFrame,
    *,
    channel_bar: str = "1h",
    ltf_bar: str = "5min",
) -> pd.DataFrame:
    """Project the latest completed hourly channel onto completed LTF decisions.

    Both inputs use bar-open timestamps. An hourly row at 00:00 is first usable at
    01:00; a 5m row at 00:55 is decided at 01:00. The fitted log slope and residual
    offsets are frozen at that availability time and projected only until a newer
    completed hourly fit becomes available.
    """
    missing_hourly = [name for name in _CHANNEL_COLUMNS if name not in hourly]
    missing_ltf = [name for name in ("open", "high", "low", "close") if name not in ltf]
    if missing_hourly or missing_ltf:
        raise KeyError(
            f"missing hourly={missing_hourly or None}; missing ltf={missing_ltf or None}"
        )
    if not isinstance(hourly.index, pd.DatetimeIndex) or not isinstance(
        ltf.index, pd.DatetimeIndex
    ):
        raise TypeError("hourly and ltf require DatetimeIndex bar-open timestamps")
    if hourly.index.has_duplicates or ltf.index.has_duplicates:
        raise ValueError("hourly and ltf timestamps must be unique")

    channel_delta = pd.Timedelta(channel_bar)
    ltf_delta = pd.Timedelta(ltf_bar)
    source_columns = list(_CHANNEL_COLUMNS)
    if "rsi_channel" in hourly:
        source_columns.append("rsi_channel")
    source = hourly.sort_index().loc[:, source_columns].copy()
    source["channel_source_time"] = source.index
    source["channel_availability_time"] = source.index + channel_delta
    source.index = pd.DatetimeIndex(source["channel_availability_time"])

    out = ltf.sort_index().copy()
    decision_index = out.index + ltf_delta
    aligned = source.reindex(decision_index, method="ffill")
    aligned.index = out.index
    elapsed_hours = (
        pd.Series(decision_index, index=out.index)
        - pd.to_datetime(aligned["channel_availability_time"], utc=True)
    ) / pd.Timedelta(hours=1)
    factor = np.exp(aligned["channel_slope"].astype(float) * elapsed_hours.astype(float))
    for name in ("channel_mid", "channel_upper", "channel_lower"):
        aligned[name] = aligned[name].astype(float) * factor

    out["availability_time"] = decision_index
    for name in aligned.columns:
        out[name] = aligned[name]

    complete = pd.Series(True, index=out.index)
    if "minute_count" in out:
        expected_minutes = int(ltf_delta / pd.Timedelta(minutes=1))
        complete &= out["minute_count"].eq(expected_minutes)
    if len(out) > 1:
        cadence_ok = out.index.to_series().diff().eq(ltf_delta)
        cadence_ok.iloc[0] = True
        complete &= cadence_ok

    geometry = ["channel_mid", "channel_upper", "channel_lower", "channel_slope", "channel_r2"]
    out.loc[~complete, geometry] = np.nan
    if "rsi_channel" in out:
        out.loc[~complete, "rsi_channel"] = np.nan
    out.loc[~complete, "channel_regime"] = "none"
    out.loc[~complete, "channel_confluence"] = 0
    out.loc[~complete, "channel_confluence_count"] = 0
    span = out["channel_upper"] - out["channel_lower"]
    out["channel_width"] = span
    out["channel_pos"] = np.where(
        span > 0, (out["close"] - out["channel_lower"]) / span, np.nan
    )
    return out


def _reversal_candles(
    frame: pd.DataFrame, *, wick_frac: float, close_frac: float
) -> tuple[np.ndarray, np.ndarray]:
    opn = frame["open"].to_numpy(dtype=float)
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    rng = high - low
    valid = np.isfinite(rng) & (rng > 0)
    lower_wick = np.divide(
        np.minimum(opn, close) - low,
        rng,
        out=np.zeros_like(rng),
        where=valid,
    )
    upper_wick = np.divide(
        high - np.maximum(opn, close),
        rng,
        out=np.zeros_like(rng),
        where=valid,
    )
    close_high = np.divide(close - low, rng, out=np.zeros_like(rng), where=valid)
    close_low = np.divide(high - close, rng, out=np.zeros_like(rng), where=valid)
    body_lo = np.minimum(opn, close)
    body_hi = np.maximum(opn, close)
    prev_lo = np.r_[np.nan, body_lo[:-1]]
    prev_hi = np.r_[np.nan, body_hi[:-1]]
    bull_engulf = (close > opn) & (body_lo <= prev_lo) & (body_hi >= prev_hi)
    bear_engulf = (close < opn) & (body_lo <= prev_lo) & (body_hi >= prev_hi)
    bull = valid & (
        ((lower_wick >= wick_frac) & (close_high >= 1.0 - close_frac)) | bull_engulf
    )
    bear = valid & (
        ((upper_wick >= wick_frac) & (close_low >= 1.0 - close_frac)) | bear_engulf
    )
    return bull, bear


def generate_supervisor_signals(
    frame: pd.DataFrame,
    config: SupervisorSignalConfig = SupervisorSignalConfig(),
) -> pd.DataFrame:
    """Attach edge/midline T1-T2-T3 signals to a projected 5m channel frame."""
    required = (
        "open",
        "high",
        "low",
        "close",
        "channel_lower",
        "channel_mid",
        "channel_upper",
        "channel_regime",
    )
    missing = [name for name in required if name not in frame]
    if missing:
        raise KeyError(f"missing signal columns: {missing}")
    if config.arm_max_bars < 1 or config.confirm_max_bars < 1:
        raise ValueError("arm_max_bars and confirm_max_bars must be >= 1")

    out = frame.copy()
    n = len(out)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    lower = out["channel_lower"].to_numpy(dtype=float)
    mid = out["channel_mid"].to_numpy(dtype=float)
    upper = out["channel_upper"].to_numpy(dtype=float)
    span = upper - lower
    valid = np.isfinite(span) & (span > 0)
    regime = out["channel_regime"].astype(str).to_numpy()
    allow = {
        "long": valid & (regime == "up"),
        "short": valid & (regime == "down"),
    }
    bull, bear = _reversal_candles(
        out, wick_frac=config.wick_frac, close_frac=config.close_frac
    )

    low_pos = np.divide(low - lower, span, out=np.full(n, np.nan), where=valid)
    high_pos = np.divide(high - lower, span, out=np.full(n, np.nan), where=valid)
    touch_mid = valid & (low <= mid + config.midline_tolerance * span) & (
        high >= mid - config.midline_tolerance * span
    )
    arms = {
        ("edge_rejection", "long"): allow["long"] & (low_pos <= config.edge_zone),
        ("edge_rejection", "short"): allow["short"] & (
            high_pos >= 1.0 - config.edge_zone
        ),
        ("midline_retest", "long"): allow["long"] & touch_mid & (close >= mid),
        ("midline_retest", "short"): allow["short"] & touch_mid & (close <= mid),
    }
    reversals = {"long": bull, "short": bear}

    stage_columns: dict[tuple[str, int], np.ndarray] = {}
    stage_sides: dict[tuple[str, int], np.ndarray] = {}
    for setup in ("edge", "midline"):
        for stage in (1, 2, 3):
            stage_columns[(setup, stage)] = np.zeros(n, dtype=np.int8)
            stage_sides[(setup, stage)] = np.full(n, None, dtype=object)
    signal = np.zeros(n, dtype=np.int8)
    setup_type = np.full(n, None, dtype=object)
    signal_swing_low = np.full(n, np.nan)
    signal_swing_high = np.full(n, np.nan)
    t1_times = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    t2_times = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")

    states: dict[tuple[str, str], dict[str, int | None]] = {
        key: {"armed_at": None, "reversal_at": None} for key in arms
    }

    for i in range(n):
        candidates: list[tuple[str, str, int, int]] = []
        for setup in ("edge_rejection", "midline_retest"):
            short_name = "edge" if setup == "edge_rejection" else "midline"
            for side in ("long", "short"):
                state = states[(setup, side)]
                if not allow[side][i]:
                    state["armed_at"] = None
                    state["reversal_at"] = None
                    continue

                armed_at = state["armed_at"]
                reversal_at = state["reversal_at"]
                if reversal_at is not None:
                    if i - reversal_at > config.confirm_max_bars:
                        state["armed_at"] = None
                        state["reversal_at"] = None
                    else:
                        if side == "long":
                            frozen_stop = low[reversal_at] * (
                                1.0 - config.stop_buffer_bps / 10_000.0
                            )
                            stop_hit = low[i] <= frozen_stop
                        else:
                            frozen_stop = high[reversal_at] * (
                                1.0 + config.stop_buffer_bps / 10_000.0
                            )
                            stop_hit = high[i] >= frozen_stop
                        if stop_hit:
                            state["armed_at"] = None
                            state["reversal_at"] = None
                            continue
                        broke = close[i] > high[reversal_at] if side == "long" else close[i] < low[reversal_at]
                        if broke:
                            stage_columns[(short_name, 3)][i] = 1
                            stage_sides[(short_name, 3)][i] = side
                            candidates.append((setup, side, int(armed_at), int(reversal_at)))
                            continue

                armed_at = state["armed_at"]
                if armed_at is not None and state["reversal_at"] is None:
                    if i - armed_at > config.arm_max_bars:
                        state["armed_at"] = None
                    elif 0 < i - armed_at and reversals[side][i]:
                        stage_columns[(short_name, 2)][i] = 1
                        stage_sides[(short_name, 2)][i] = side
                        state["reversal_at"] = i
                        continue

                if state["armed_at"] is None and arms[(setup, side)][i]:
                    state["armed_at"] = i
                    stage_columns[(short_name, 1)][i] = 1
                    stage_sides[(short_name, 1)][i] = side

        if candidates:
            candidates.sort(key=lambda item: 0 if item[0] == "edge_rejection" else 1)
            chosen_setup, chosen_side, t1_at, t2_at = candidates[0]
            signal[i] = 1 if chosen_side == "long" else -1
            setup_type[i] = chosen_setup
            signal_swing_low[i] = low[t2_at]
            signal_swing_high[i] = high[t2_at]
            t1_times[i] = out.index[t1_at].to_datetime64()
            t2_times[i] = out.index[t2_at].to_datetime64()
            for setup in ("edge_rejection", "midline_retest"):
                states[(setup, chosen_side)]["armed_at"] = None
                states[(setup, chosen_side)]["reversal_at"] = None

    out["edge_stage_1"] = stage_columns[("edge", 1)]
    out["edge_stage_2"] = stage_columns[("edge", 2)]
    out["edge_stage_3"] = stage_columns[("edge", 3)]
    out["edge_stage_1_side"] = stage_sides[("edge", 1)]
    out["edge_stage_2_side"] = stage_sides[("edge", 2)]
    out["edge_stage_3_side"] = stage_sides[("edge", 3)]
    out["midline_stage_1"] = stage_columns[("midline", 1)]
    out["midline_stage_2"] = stage_columns[("midline", 2)]
    out["midline_stage_3"] = stage_columns[("midline", 3)]
    out["midline_stage_1_side"] = stage_sides[("midline", 1)]
    out["midline_stage_2_side"] = stage_sides[("midline", 2)]
    out["midline_stage_3_side"] = stage_sides[("midline", 3)]
    out["stage_1"] = ((out["edge_stage_1"] + out["midline_stage_1"]) > 0).astype(int)
    out["stage_2"] = ((out["edge_stage_2"] + out["midline_stage_2"]) > 0).astype(int)
    out["stage_3"] = ((out["edge_stage_3"] + out["midline_stage_3"]) > 0).astype(int)
    out["signal"] = signal
    out["setup_type"] = setup_type
    out["signal_swing_low"] = signal_swing_low
    out["signal_swing_high"] = signal_swing_high
    out["t1_time"] = pd.to_datetime(t1_times, utc=True)
    out["t2_time"] = pd.to_datetime(t2_times, utc=True)
    return out


__all__ = [
    "SupervisorSignalConfig",
    "generate_supervisor_signals",
    "project_closed_hourly_channel",
]
