"""Broad, causal opportunity windows for the replacement Notebook J.

Hourly regression channels provide direction and geometry.  Completed five-minute
bars provide the trigger location and activity state.  The resulting manifest is
an auditable schedule of one-hour intervals in which a later model may decide
whether to trade; it contains no outcome information.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from features.linear_channels import compute_linear_regression_channels


MANIFEST_COLUMNS = (
    "window_id",
    "channel_episode_id",
    "side",
    "window_start",
    "window_end",
    "source_bar_time",
    "activity_ratio_open",
    "slope_raw_open",
    "slope_bps_open",
    "r2_open",
    "confluence_open",
    "end_reason",
)


@dataclass(frozen=True)
class EventWindowConfig:
    """Frozen geometry and activity settings for broad event windows."""

    bar: str = "5min"
    channel_windows: tuple[int, ...] = (60, 90, 120)
    primary_channel_window: int = 60
    channel_quantile: float = 0.10
    pre_context_bars: int = 24
    active_bars: int = 12
    activity_short_bars: int = 12
    activity_long_bars: int = 288
    activity_floor: float = 0.80
    channel_half: float = 0.50

    def __post_init__(self) -> None:
        if not self.channel_windows or any(window < 2 for window in self.channel_windows):
            raise ValueError("channel_windows must contain values >= 2")
        if self.primary_channel_window not in self.channel_windows:
            raise ValueError("primary_channel_window must be in channel_windows")
        if not 0.0 < self.channel_quantile < 0.5:
            raise ValueError("channel_quantile must be between 0 and 0.5")
        if self.pre_context_bars < 0 or self.active_bars < 1:
            raise ValueError("pre_context_bars must be >= 0 and active_bars >= 1")
        if self.activity_short_bars < 2:
            raise ValueError("activity_short_bars must be >= 2")
        if self.activity_long_bars < self.activity_short_bars:
            raise ValueError("activity_long_bars must be >= activity_short_bars")
        if self.activity_floor < 0.0:
            raise ValueError("activity_floor must be non-negative")
        if not 0.0 < self.channel_half < 1.0:
            raise ValueError("channel_half must be between 0 and 1")
        if pd.Timedelta(self.bar) <= pd.Timedelta(0):
            raise ValueError("bar must be a positive duration")


def _utc_index(frame: pd.DataFrame, name: str) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} requires DatetimeIndex bar-open timestamps")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be unique and increasing")
    index = pd.DatetimeIndex(frame.index)
    return index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")


def _stable_id(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _raw_sign(values: pd.Series) -> pd.Series:
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    sign = np.where(np.isfinite(raw), np.sign(raw), 0).astype(np.int8)
    return pd.Series(sign, index=values.index, dtype="int8")


def build_hourly_channel_context(
    hourly: pd.DataFrame,
    config: EventWindowConfig = EventWindowConfig(),
) -> pd.DataFrame:
    """Fit log-price channels on trailing completed 60/90/120-hour histories.

    Input timestamps are bar opens.  A row remains unavailable to downstream code
    until one hour after its timestamp; ``project_hourly_channels`` enforces that
    boundary.
    """
    missing = [name for name in ("close",) if name not in hourly]
    if missing:
        raise KeyError(f"hourly frame missing columns: {missing}")
    index = _utc_index(hourly, "hourly frame")
    source = hourly.copy()
    source.index = index
    out = source.copy()

    for window in config.channel_windows:
        fitted = compute_linear_regression_channels(
            source,
            window=window,
            log_price=True,
            method="quantile",
            quantile=config.channel_quantile,
        )
        valid = fitted["channel_slope"].notna()
        raw_slope = fitted["channel_slope"].where(valid)
        out[f"channel_slope_raw_{window}"] = raw_slope
        out[f"channel_slope_bps_{window}"] = 10_000.0 * raw_slope
        out[f"channel_mid_{window}"] = fitted["channel_mid"].where(valid)
        out[f"channel_upper_{window}"] = fitted["channel_upper"].where(valid)
        out[f"channel_lower_{window}"] = fitted["channel_lower"].where(valid)
        out[f"channel_r2_{window}"] = fitted["channel_r2"].where(valid)
        out[f"channel_width_{window}"] = fitted["channel_width"].where(valid)
        out[f"channel_sign_{window}"] = _raw_sign(raw_slope)
        if window == config.primary_channel_window:
            out["rsi_channel"] = fitted["rsi"]

    primary = config.primary_channel_window
    out["channel_mid_at_fit"] = out[f"channel_mid_{primary}"]
    out["channel_upper_at_fit"] = out[f"channel_upper_{primary}"]
    out["channel_lower_at_fit"] = out[f"channel_lower_{primary}"]
    out["channel_r2"] = out[f"channel_r2_{primary}"]
    out["channel_width_at_fit"] = out[f"channel_width_{primary}"]

    primary_sign = out[f"channel_sign_{primary}"]
    agreement = pd.DataFrame(
        {
            window: out[f"channel_sign_{window}"].eq(primary_sign)
            & primary_sign.ne(0)
            for window in config.channel_windows
        },
        index=out.index,
    )
    out["channel_confluence_count"] = agreement.sum(axis=1).astype("int8")
    out["channel_confluence"] = out["channel_confluence_count"].ge(2).astype("int8")
    cadence_gap = out.index.to_series().diff().ne(pd.Timedelta("1h"))
    cadence_gap.iloc[0] = False
    episode_start = cadence_gap | primary_sign.ne(primary_sign.shift())
    age = primary_sign.groupby(episode_start.cumsum()).cumcount() + 1
    out["channel_regime_age_hours"] = age.where(primary_sign.ne(0), 0).astype("int32")
    return out


def project_hourly_channels(
    five_minute: pd.DataFrame,
    hourly_context: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only the latest completed hourly fit to each completed 5m bar."""
    missing_five = [
        name for name in ("open", "high", "low", "close") if name not in five_minute
    ]
    required_context = (
        "channel_slope_raw_60",
        "channel_slope_bps_60",
        "channel_mid_at_fit",
        "channel_upper_at_fit",
        "channel_lower_at_fit",
        "channel_r2",
        "channel_confluence_count",
    )
    missing_context = [name for name in required_context if name not in hourly_context]
    if missing_five or missing_context:
        raise KeyError(
            f"missing five_minute={missing_five or None}; "
            f"missing hourly_context={missing_context or None}"
        )

    five_index = _utc_index(five_minute, "five-minute frame")
    hourly_index = _utc_index(hourly_context, "hourly context")
    source = hourly_context.copy()
    source.index = hourly_index
    source["channel_source_time"] = source.index
    source["channel_availability_time"] = source.index + pd.Timedelta("1h")
    source.index = pd.DatetimeIndex(source["channel_availability_time"])

    out = five_minute.copy()
    out.index = five_index
    availability = five_index + pd.Timedelta("5min")
    aligned = source.reindex(availability, method="ffill")
    aligned.index = out.index
    out["availability_time"] = availability
    for name in aligned.columns:
        if name not in out.columns:
            out[name] = aligned[name]

    out["channel_slope_raw"] = aligned["channel_slope_raw_60"]
    out["channel_slope_bps"] = aligned["channel_slope_bps_60"]
    out["channel_sign"] = _raw_sign(out["channel_slope_raw"])
    elapsed_hours = (
        pd.Series(availability, index=out.index)
        - pd.to_datetime(aligned["channel_availability_time"], utc=True)
    ) / pd.Timedelta(hours=1)
    factor = np.exp(
        out["channel_slope_raw"].astype(float) * elapsed_hours.astype(float)
    )
    out["channel_mid"] = aligned["channel_mid_at_fit"].astype(float) * factor
    out["channel_upper"] = aligned["channel_upper_at_fit"].astype(float) * factor
    out["channel_lower"] = aligned["channel_lower_at_fit"].astype(float) * factor

    complete = pd.Series(True, index=out.index)
    if "minute_count" in out:
        complete &= pd.to_numeric(out["minute_count"], errors="coerce").eq(5)
    if "bar_complete" in out:
        complete &= out["bar_complete"].fillna(False).astype(bool)
    if len(out) > 1:
        cadence_ok = out.index.to_series().diff().eq(pd.Timedelta("5min"))
        cadence_ok.iloc[0] = True
        complete &= cadence_ok
    geometry = (
        "channel_slope_raw",
        "channel_slope_bps",
        "channel_mid",
        "channel_upper",
        "channel_lower",
        "channel_r2",
    )
    out.loc[~complete, list(geometry)] = np.nan
    out.loc[~complete, "channel_sign"] = 0

    span = out["channel_upper"] - out["channel_lower"]
    out["channel_width"] = span
    out["channel_pos"] = np.where(
        span > 0,
        (pd.to_numeric(out["close"], errors="coerce") - out["channel_lower"]) / span,
        np.nan,
    )
    return out


def causal_activity_ratio(
    frame: pd.DataFrame,
    config: EventWindowConfig = EventWindowConfig(),
) -> pd.Series:
    """Short/long realised-volatility ratio without crossing bad 5m histories."""
    if "close" not in frame:
        raise KeyError("activity frame missing close")
    index = _utc_index(frame, "activity frame")
    close = pd.to_numeric(frame["close"], errors="coerce")
    complete = close.gt(0.0) & close.notna()
    expected_minutes = pd.Timedelta(config.bar) / pd.Timedelta(minutes=1)
    if "minute_count" in frame:
        complete &= pd.to_numeric(frame["minute_count"], errors="coerce").eq(
            expected_minutes
        )
    if "bar_complete" in frame:
        complete &= frame["bar_complete"].fillna(False).astype(bool)
    cadence = pd.Series(index, index=frame.index).diff().eq(pd.Timedelta(config.bar))
    valid_return = complete & complete.shift(fill_value=False) & cadence
    returns = np.log(close).diff().where(valid_return)
    short = returns.rolling(
        config.activity_short_bars,
        min_periods=config.activity_short_bars,
    ).std()
    long = returns.rolling(
        config.activity_long_bars,
        min_periods=config.activity_long_bars,
    ).std()
    return (short / long.replace(0.0, np.nan)).rename("activity_ratio")


def _episode_ids(
    index: pd.DatetimeIndex,
    signs: np.ndarray,
    gaps: np.ndarray,
    symbol: str,
) -> np.ndarray:
    identifiers = np.empty(len(index), dtype=object)
    episode_id = ""
    previous_sign: int | None = None
    for row, (timestamp, sign) in enumerate(zip(index, signs, strict=True)):
        if row == 0 or gaps[row] or int(sign) != previous_sign:
            episode_id = _stable_id(
                symbol.upper(), "slope_episode", int(sign), timestamp.isoformat()
            )
        identifiers[row] = episode_id
        previous_sign = int(sign)
    return identifiers


def build_event_window_manifest(
    frame: pd.DataFrame,
    config: EventWindowConfig = EventWindowConfig(),
    *,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Create causal, non-overlapping one-hour windows from completed 5m bars."""
    required = ("close", "channel_slope_raw", "channel_pos")
    missing = [name for name in required if name not in frame]
    if missing:
        raise KeyError(f"event-window frame missing columns: {missing}")
    source_index = _utc_index(frame, "event-window frame")
    if frame.empty:
        return pd.DataFrame(columns=list(MANIFEST_COLUMNS))

    bar = pd.Timedelta(config.bar)
    decisions = source_index + bar
    cadence_gap = np.zeros(len(frame), dtype=bool)
    if len(frame) > 1:
        cadence_gap[1:] = np.asarray(source_index[1:] - source_index[:-1] != bar)
    complete = np.ones(len(frame), dtype=bool)
    expected_minutes = bar / pd.Timedelta(minutes=1)
    if "minute_count" in frame:
        complete &= (
            pd.to_numeric(frame["minute_count"], errors="coerce")
            .eq(expected_minutes)
            .to_numpy(dtype=bool)
        )
    if "bar_complete" in frame:
        complete &= frame["bar_complete"].fillna(False).to_numpy(dtype=bool)
    complete &= ~cadence_gap

    slope = pd.to_numeric(frame["channel_slope_raw"], errors="coerce")
    signs = _raw_sign(slope).to_numpy(dtype=np.int8)
    episodes = _episode_ids(source_index, signs, cadence_gap, symbol)
    activity = causal_activity_ratio(frame, config).to_numpy(dtype=float)
    position = pd.to_numeric(frame["channel_pos"], errors="coerce").to_numpy(
        dtype=float
    )
    r2 = (
        pd.to_numeric(frame["channel_r2"], errors="coerce").to_numpy(dtype=float)
        if "channel_r2" in frame
        else np.full(len(frame), np.nan)
    )
    slope_bps = (
        pd.to_numeric(frame["channel_slope_bps"], errors="coerce").to_numpy(
            dtype=float
        )
        if "channel_slope_bps" in frame
        else slope.to_numpy(dtype=float) * 10_000.0
    )
    confluence = (
        pd.to_numeric(frame["channel_confluence_count"], errors="coerce")
        .fillna(0)
        .to_numpy(dtype=int)
        if "channel_confluence_count" in frame
        else np.zeros(len(frame), dtype=int)
    )

    rows: list[dict[str, object]] = []
    active: dict[str, object] | None = None
    last_start: pd.Timestamp | None = None
    duration = config.active_bars * bar

    def close_active(end: pd.Timestamp, reason: str) -> None:
        nonlocal active
        if active is None:
            return
        start = pd.Timestamp(active["window_start"])
        if end < start:
            raise AssertionError("event window cannot end before it starts")
        if end == start:
            active = None
            return
        active["window_end"] = end
        active["end_reason"] = reason
        rows.append(active)
        active = None

    for row in range(len(frame)):
        decision = decisions[row]
        if active is not None:
            planned_end = pd.Timestamp(active["window_start"]) + duration
            if cadence_gap[row] or not complete[row]:
                gap_boundary = decisions[row - 1] + bar if row else decision
                if planned_end <= gap_boundary:
                    close_active(planned_end, "time_cap")
                else:
                    close_active(gap_boundary, "data_gap")
            elif signs[row] != int(active["slope_sign"]):
                close_active(decision, "slope_reversal")
            elif decision >= planned_end:
                close_active(planned_end, "time_cap")

        long_on = signs[row] > 0 and position[row] <= config.channel_half
        short_on = signs[row] < 0 and position[row] >= 1.0 - config.channel_half
        eligible = (
            complete[row]
            and np.isfinite(activity[row])
            and activity[row] >= config.activity_floor
            and (long_on or short_on)
        )
        cooldown_ok = last_start is None or decision - last_start >= duration
        if active is None and eligible and cooldown_ok:
            side = "long" if long_on else "short"
            start = decision
            episode = episodes[row]
            active = {
                "window_id": _stable_id(
                    symbol.upper(), "event_window", episode, start.isoformat()
                ),
                "channel_episode_id": episode,
                "side": side,
                "window_start": start,
                "window_end": pd.NaT,
                "source_bar_time": source_index[row],
                "activity_ratio_open": float(activity[row]),
                "slope_raw_open": float(slope.iloc[row]),
                "slope_bps_open": float(slope_bps[row]),
                "r2_open": float(r2[row]),
                "confluence_open": int(confluence[row]),
                "end_reason": None,
                "slope_sign": int(signs[row]),
            }
            last_start = start

    if active is not None:
        close_active(min(pd.Timestamp(active["window_start"]) + duration, decisions[-1]), "data_end")

    if not rows:
        return pd.DataFrame(columns=list(MANIFEST_COLUMNS))
    manifest = pd.DataFrame(rows)
    manifest = manifest.drop(columns=["slope_sign"])
    if not (manifest["window_end"] > manifest["window_start"]).all():
        raise AssertionError("event windows must contain at least one decision")
    return manifest.loc[:, list(MANIFEST_COLUMNS)].sort_values(
        "window_start", kind="stable"
    ).reset_index(drop=True)


__all__ = [
    "EventWindowConfig",
    "MANIFEST_COLUMNS",
    "build_event_window_manifest",
    "build_hourly_channel_context",
    "causal_activity_ratio",
    "project_hourly_channels",
]
