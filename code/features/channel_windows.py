"""Causal LONG/SHORT channel-window manifests for Notebook B.

Each manifest row describes when a completed 15-minute edge observation made a
trading opportunity available.  Lifecycle timestamps are audit metadata, not
model features: the model may use only information known at its decision time.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


WINDOW_COLUMNS = (
    "window_id",
    "channel_episode_id",
    "side",
    "window_start",
    "natural_end_time",
    "eligible_end_time",
    "window_end_reason",
    "channel_r2_open",
    "channel_regime_open",
)

_REQUIRED_COLUMNS = frozenset(
    {
        "channel_regime",
        "channel_episode_id",
        "channel_r2",
        "channel_confluence",
        "channel_pos",
    }
)


def _as_utc(value: pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _infer_cadence(frame: pd.DataFrame) -> pd.Timedelta:
    if "minute_count" in frame and frame["minute_count"].notna().any():
        expected_minutes = float(frame["minute_count"].dropna().max())
        if expected_minutes > 0:
            return pd.Timedelta(minutes=expected_minutes)
    if len(frame.index) < 2:
        raise ValueError("at least two rows or minute_count are needed to infer cadence")
    deltas = frame.index.to_series().diff().dropna()
    positive = deltas[deltas > pd.Timedelta(0)]
    if positive.empty:
        raise ValueError("cannot infer a positive source cadence")
    return pd.Timedelta(positive.mode().iloc[0])


def _stable_window_id(
    symbol: str, side: str, episode: object, start: pd.Timestamp
) -> str:
    payload = f"{symbol.upper()}|{side}|{episode}|{_as_utc(start).isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _end_reason(
    frame: pd.DataFrame,
    row: int,
    side: str,
    complete: np.ndarray,
    require_confluence: bool,
) -> str:
    if not complete[row]:
        return "incomplete_bar"
    expected_regime = "up" if side == "long" else "down"
    if frame["channel_regime"].iloc[row] != expected_regime:
        return "regime_change"
    if require_confluence and not bool(
        frame["channel_confluence"].fillna(0).iloc[row]
    ):
        return "confluence_end"
    if not np.isfinite(frame["channel_r2"].iloc[row]) or frame["channel_r2"].iloc[row] < 0.20:
        return "r2_end"
    return "edge_end"


def _side_windows(
    frame: pd.DataFrame,
    *,
    side: str,
    symbol: str,
    on: np.ndarray,
    available: pd.DatetimeIndex,
    complete: np.ndarray,
    cadence: pd.Timedelta,
    max_duration: pd.Timedelta,
    require_confluence: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    run_start: int | None = None
    run_episode: object = None

    def close_run(end: pd.Timestamp, reason: str) -> None:
        nonlocal run_start, run_episode
        if run_start is None:
            return
        start = available[run_start]
        cap = start + max_duration
        capped = cap < end
        rows.append(
            {
                "window_id": _stable_window_id(symbol, side, run_episode, start),
                "channel_episode_id": run_episode,
                "side": side,
                "window_start": start,
                "natural_end_time": end,
                "eligible_end_time": cap if capped else end,
                "window_end_reason": "time_cap" if capped else reason,
                "channel_r2_open": float(frame["channel_r2"].iloc[run_start]),
                "channel_regime_open": frame["channel_regime"].iloc[run_start],
            }
        )
        run_start = None
        run_episode = None

    for row in range(len(frame)):
        if run_start is not None:
            expected = available[row - 1] + cadence
            if available[row] != expected:
                close_run(expected, "gap")
            elif frame["channel_episode_id"].iloc[row] != run_episode:
                close_run(available[row], "episode_change")
            elif not on[row]:
                close_run(
                    available[row],
                    _end_reason(frame, row, side, complete, require_confluence),
                )

        if run_start is None and on[row]:
            run_start = row
            run_episode = frame["channel_episode_id"].iloc[row]

    if run_start is not None:
        close_run(available[-1] + cadence, "data_end")
    return rows


def build_channel_window_manifest(
    frame: pd.DataFrame,
    *,
    symbol: str = "BTCUSDT",
    zone: float = 0.30,
    max_duration: str = "120min",
    require_confluence: bool = True,
) -> pd.DataFrame:
    """Convert completed 15-minute edge states into auditable opportunity windows.

    LONG windows require a rising, confluent channel near its lower edge; SHORT
    windows mirror the rule in a falling channel.  The broad ML eligibility floor
    is deliberately R-squared 0.20 and is separate from the 0.40 mechanical control.
    """
    missing = sorted(_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"channel-window frame missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("channel-window frame needs a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("channel-window frame index must be unique and increasing")
    if not 0.0 < zone < 0.5:
        raise ValueError("zone must be between 0 and 0.5")
    duration = pd.Timedelta(max_duration)
    if duration <= pd.Timedelta(0):
        raise ValueError("max_duration must be positive")
    if frame.empty:
        return pd.DataFrame(columns=list(WINDOW_COLUMNS))

    cadence = _infer_cadence(frame)
    if "availability_time" in frame:
        available = pd.DatetimeIndex(pd.to_datetime(frame["availability_time"], utc=True))
    else:
        source_index = pd.DatetimeIndex(frame.index)
        if source_index.tz is None:
            source_index = source_index.tz_localize("UTC")
        else:
            source_index = source_index.tz_convert("UTC")
        available = source_index + cadence
    if not available.is_monotonic_increasing or not available.is_unique:
        raise ValueError("availability_time must be unique and increasing")

    expected_minutes = cadence / pd.Timedelta(minutes=1)
    complete = np.ones(len(frame), dtype=bool)
    if "minute_count" in frame:
        complete &= frame["minute_count"].to_numpy(dtype=float) == expected_minutes
    if "bar_complete" in frame:
        complete &= frame["bar_complete"].fillna(False).to_numpy(dtype=bool)

    regime = frame["channel_regime"]
    confluence = frame["channel_confluence"].fillna(0).astype(bool)
    confluence_gate = confluence if require_confluence else pd.Series(
        True, index=frame.index
    )
    r2_ok = frame["channel_r2"].ge(0.20).fillna(False)
    position = frame["channel_pos"]
    long_on = (
        regime.eq("up") & confluence_gate & r2_ok & position.le(zone) & complete
    ).to_numpy(dtype=bool)
    short_on = (
        regime.eq("down") & confluence_gate & r2_ok & position.ge(1.0 - zone) & complete
    ).to_numpy(dtype=bool)

    rows = _side_windows(
        frame,
        side="long",
        symbol=symbol,
        on=long_on,
        available=available,
        complete=complete,
        cadence=cadence,
        max_duration=duration,
        require_confluence=require_confluence,
    )
    rows.extend(
        _side_windows(
            frame,
            side="short",
            symbol=symbol,
            on=short_on,
            available=available,
            complete=complete,
            cadence=cadence,
            max_duration=duration,
            require_confluence=require_confluence,
        )
    )
    manifest = pd.DataFrame(rows, columns=list(WINDOW_COLUMNS))
    if manifest.empty:
        return manifest
    return manifest.sort_values(["window_start", "side"], kind="stable").reset_index(drop=True)
