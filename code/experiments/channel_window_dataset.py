"""Matched 5-minute and 1-minute decision rows for channel-window ML.

The two cadences see the same causal 15-minute geometry and higher-timeframe
context.  Every source is timestamped when its bar is complete and is joined with
a backward as-of merge; lifecycle and later execution fields are metadata only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from evaluation.channel_backtest import backtest_channel_strategy


@dataclass(frozen=True)
class DecisionFrames:
    minute: pd.DataFrame
    five_minute: pd.DataFrame
    fifteen_minute: pd.DataFrame
    hourly: pd.DataFrame
    daily: pd.DataFrame
    positioning: pd.DataFrame


@dataclass(frozen=True)
class WindowLabelConfig:
    min_risk_bps: float = 40.0
    max_risk_bps: float = 250.0
    min_rr: float = 1.2
    max_hold_minutes: int = 1440
    limit_offset_bps: float = 5.0
    fill_window_minutes: int = 20
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0


LABEL_EXECUTION = {
    "target_mode": "measured",
    "stop_mode": "swing",
    "swing_low_col": "swing_low_15m",
    "swing_high_col": "swing_high_15m",
    "measured_move_col": "measured_move_15m",
    "entry_mode": "maker_limit",
    "max_trades_per_day": None,
    "max_concurrent": None,
}


CHANNEL_WINDOW_FEATURES = (
    "channel_slope_side",
    "channel_r2",
    "channel_width_pct",
    "channel_position_side",
    "channel_confluence",
    "macro_sma_distance_side",
    "macro_alignment",
    "window_age_fraction",
    "pullback_depth",
    "swing_reversal_score",
    "candle_rejection_side",
    "taker_imbalance_side",
    "return_1m_side",
    "return_5m_side",
    "vol_ratio_15m_1h",
    "funding_z_side",
    "oi_chg_4h_side",
    "positioning_stale",
    "positioning_age_log",
    "minute_count_ratio",
    "risk_bps_decision",
    "rr_planned_decision",
    "edge_recovery_bps_side",
    "bars_since_window_extreme",
    "body_direction_side",
    "reversal_count_3",
    "rejection_mean_3_side",
    "taker_imbalance_mean_3_side",
    "taker_imbalance_delta_side",
    "volume_ratio_12",
)

FORBIDDEN_WINDOW_FEATURES = frozenset(
    {
        "natural_end_time",
        "eligible_end_time",
        "window_end_reason",
        "label_end",
        "entry_time",
        "exit_time",
        "order_status",
        "filled",
        "outcome",
        "r_net",
        "bars_held",
        "layer_no",
    }
)

_MANIFEST_REQUIRED = frozenset(
    {
        "window_id",
        "channel_episode_id",
        "side",
        "window_start",
        "natural_end_time",
        "eligible_end_time",
        "window_end_reason",
    }
)
_OHLC = ("open", "high", "low", "close")
_WINDOW_MAX_MINUTES = 120.0
_LIMIT_OFFSET_BPS = 5.0
_STOP_BUFFER_BPS = 5.0


def _utc_index(index: pd.Index) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(index)
    return values.tz_localize("UTC") if values.tz is None else values.tz_convert("UTC")


def _available_frame(
    frame: pd.DataFrame,
    cadence: str,
    *,
    index_is_available: bool = False,
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{cadence} context needs a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{cadence} context index must be unique and increasing")
    out = frame.copy()
    source_index = _utc_index(out.index)
    if "availability_time" in out:
        available = pd.DatetimeIndex(pd.to_datetime(out.pop("availability_time"), utc=True))
    elif index_is_available:
        available = source_index
    else:
        available = source_index + pd.Timedelta(cadence)
    out.insert(0, "availability_time", available)
    out.insert(0, "source_bar_time", source_index)
    return out.reset_index(drop=True)


def _valid_completed_bars(frame: pd.DataFrame, cadence: str) -> np.ndarray:
    valid = np.ones(len(frame), dtype=bool)
    for column in _OHLC:
        if column not in frame:
            raise ValueError(f"{cadence} decision frame missing {column!r}")
        valid &= np.isfinite(frame[column].to_numpy(dtype=float))
    if "minute_count" in frame:
        expected = pd.Timedelta(cadence) / pd.Timedelta(minutes=1)
        valid &= frame["minute_count"].to_numpy(dtype=float) == expected
    return valid


def _candidate_id(window_id: str, cadence: str, decision_time: pd.Timestamp) -> str:
    payload = f"{window_id}|{cadence}|{decision_time.isoformat()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _decision_rows(
    manifest: pd.DataFrame, source: pd.DataFrame, cadence: str
) -> pd.DataFrame:
    available_source = _available_frame(source, cadence)
    available = pd.DatetimeIndex(available_source["availability_time"])
    complete = _valid_completed_bars(source, cadence)
    expected_count = pd.Timedelta(cadence) / pd.Timedelta(minutes=1)
    rows: list[dict[str, object]] = []

    for window in manifest.sort_values(["window_start", "side"]).itertuples(index=False):
        if window.side not in ("long", "short"):
            raise ValueError(f"unknown window side: {window.side!r}")
        start = pd.Timestamp(window.window_start)
        end = pd.Timestamp(window.eligible_end_time)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        left = int(available.searchsorted(start, side="left"))
        right = int(available.searchsorted(end, side="left"))
        for pos in range(left, right):
            if not complete[pos]:
                continue
            decision_time = available[pos]
            bar = source.iloc[pos]
            minute_count = (
                float(bar["minute_count"])
                if "minute_count" in source
                else float(expected_count)
            )
            rows.append(
                {
                    "candidate_id": _candidate_id(window.window_id, cadence, decision_time),
                    "window_id": window.window_id,
                    "channel_episode_id": window.channel_episode_id,
                    "side": window.side,
                    "cadence": cadence,
                    "source_bar_time": available_source["source_bar_time"].iloc[pos],
                    "decision_time": decision_time,
                    "next_entry_time": decision_time,
                    "window_start": start,
                    "natural_end_time": window.natural_end_time,
                    "eligible_end_time": end,
                    "window_end_reason": window.window_end_reason,
                    "decision_open": float(bar["open"]),
                    "decision_high": float(bar["high"]),
                    "decision_low": float(bar["low"]),
                    "decision_close": float(bar["close"]),
                    "decision_volume": float(bar.get("volume", np.nan)),
                    "decision_taker_buy_base": float(bar.get("taker_buy_base", np.nan)),
                    "minute_count_ratio": minute_count / float(expected_count),
                }
            )
    return pd.DataFrame(rows)


def _invalidate_incomplete_rolls(frame: pd.DataFrame, cadence: str, window: int) -> np.ndarray:
    valid = _valid_completed_bars(frame, cadence)
    idx = _utc_index(frame.index)
    gaps = np.zeros(len(frame), dtype=bool)
    if len(frame) > 1:
        gaps[1:] = idx[1:] - idx[:-1] != pd.Timedelta(cadence)
    bad = pd.Series(~valid | gaps, index=frame.index)
    return bad.rolling(window, min_periods=1).max().to_numpy(dtype=bool)


def _swing_context(frame: pd.DataFrame) -> pd.DataFrame:
    context = frame.copy()
    contaminated = _invalidate_incomplete_rolls(context, "15min", 12)
    context["swing_low_15m"] = context["low"].rolling(12, min_periods=12).min()
    context["swing_high_15m"] = context["high"].rolling(12, min_periods=12).max()
    context.loc[contaminated, ["swing_low_15m", "swing_high_15m"]] = np.nan
    context["measured_move_15m"] = (
        context["swing_high_15m"] - context["swing_low_15m"]
    )
    return _available_frame(
        context[["swing_low_15m", "swing_high_15m", "measured_move_15m"]],
        "15min",
    )


def _return_context(frame: pd.DataFrame, cadence: str, name: str) -> pd.DataFrame:
    close = frame["close"].astype(float)
    returns = close.pct_change(fill_method=None)
    idx = _utc_index(frame.index)
    if len(frame) > 1:
        broken = np.r_[False, idx[1:] - idx[:-1] != pd.Timedelta(cadence)]
        returns.iloc[np.flatnonzero(broken)] = np.nan
    context = pd.DataFrame({name: returns}, index=frame.index)
    return _available_frame(context, cadence)


def _sequence_context(frame: pd.DataFrame, cadence: str) -> pd.DataFrame:
    """Completed-bar sequence used by the candle and flow faucets."""
    open_ = frame["open"].astype(float)
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    price_range = (high - low).replace(0.0, np.nan)
    volume = (
        frame["volume"].astype(float)
        if "volume" in frame
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    taker = (
        frame["taker_buy_base"].astype(float)
        if "taker_buy_base" in frame
        else pd.Series(np.nan, index=frame.index, dtype=float)
    )
    imbalance = 2.0 * taker / volume.replace(0.0, np.nan) - 1.0
    move = close.diff()
    up = move.gt(0).astype(float).where(move.notna())
    down = move.lt(0).astype(float).where(move.notna())
    lower_rejection = (np.minimum(open_, close) - low) / price_range
    upper_rejection = (high - np.maximum(open_, close)) / price_range

    context = pd.DataFrame(index=frame.index)
    context["body_direction_raw"] = (close - open_) / price_range
    context["up_count_3"] = up.rolling(3, min_periods=3).sum()
    context["down_count_3"] = down.rolling(3, min_periods=3).sum()
    context["lower_rejection_mean_3"] = lower_rejection.rolling(
        3, min_periods=3
    ).mean()
    context["upper_rejection_mean_3"] = upper_rejection.rolling(
        3, min_periods=3
    ).mean()
    context["taker_imbalance_mean_3"] = imbalance.rolling(3, min_periods=3).mean()
    context["taker_imbalance_delta"] = imbalance.diff()
    rolling_volume = volume.rolling(12, min_periods=12).median().replace(0.0, np.nan)
    context["volume_ratio_12"] = volume / rolling_volume

    contaminated_3 = _invalidate_incomplete_rolls(frame, cadence, 3)
    context.loc[
        contaminated_3,
        [
            "up_count_3",
            "down_count_3",
            "lower_rejection_mean_3",
            "upper_rejection_mean_3",
            "taker_imbalance_mean_3",
            "taker_imbalance_delta",
        ],
    ] = np.nan
    contaminated_12 = _invalidate_incomplete_rolls(frame, cadence, 12)
    context.loc[contaminated_12, "volume_ratio_12"] = np.nan
    return _available_frame(context, cadence)


def _attach_window_extreme_features(candidates: pd.DataFrame) -> pd.DataFrame:
    """Measure recovery from the running extreme inside each window only."""
    out = candidates.copy()
    recovery = pd.Series(np.nan, index=out.index, dtype=float)
    bars_since = pd.Series(np.nan, index=out.index, dtype=float)
    for _, group in out.groupby("window_id", sort=False):
        group = group.sort_values("decision_time", kind="stable")
        side_values = group["side"].dropna().unique()
        if len(side_values) != 1:
            raise ValueError("one window cannot contain both trade sides")
        close = group["decision_close"].to_numpy(dtype=float)
        positions = np.arange(len(group), dtype=float)
        if side_values[0] == "long":
            points = group["decision_low"].to_numpy(dtype=float)
            extreme = np.minimum.accumulate(points)
            previous = np.r_[np.inf, extreme[:-1]]
            is_new = points <= previous
            aligned_recovery = (close / extreme - 1.0) * 1e4
        elif side_values[0] == "short":
            points = group["decision_high"].to_numpy(dtype=float)
            extreme = np.maximum.accumulate(points)
            previous = np.r_[-np.inf, extreme[:-1]]
            is_new = points >= previous
            aligned_recovery = (extreme / close - 1.0) * 1e4
        else:
            raise ValueError(f"unknown window side: {side_values[0]!r}")
        last_extreme = np.maximum.accumulate(np.where(is_new, positions, -1.0))
        recovery.loc[group.index] = aligned_recovery
        bars_since.loc[group.index] = positions - last_extreme
    out["edge_recovery_bps_side"] = recovery
    out["bars_since_window_extreme"] = bars_since
    return out


def _minute_context(frame: pd.DataFrame) -> pd.DataFrame:
    returns = frame["close"].astype(float).pct_change(fill_method=None)
    idx = _utc_index(frame.index)
    if len(frame) > 1:
        broken = np.r_[False, idx[1:] - idx[:-1] != pd.Timedelta("1min")]
        returns.iloc[np.flatnonzero(broken)] = np.nan
    vol_15 = returns.rolling(15, min_periods=15).std()
    vol_60 = returns.rolling(60, min_periods=60).std().replace(0.0, np.nan)
    context = pd.DataFrame(
        {"return_1m": returns, "vol_ratio_15m_1h": vol_15 / vol_60},
        index=frame.index,
    )
    return _available_frame(context, "1min")


def _hourly_context(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "channel_slope", "channel_r2", "channel_lower", "channel_mid",
        "channel_upper", "channel_confluence",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"hourly context missing channel columns: {missing}")
    context = frame[list(required)].copy()
    context["channel_width"] = context["channel_upper"] - context["channel_lower"]
    return _available_frame(context, "1h")


def _daily_context(frame: pd.DataFrame) -> pd.DataFrame:
    if "close" not in frame:
        raise ValueError("daily context missing 'close'")
    close = frame["close"].astype(float)
    context = pd.DataFrame(
        {"macro_close": close, "macro_sma200": close.rolling(200, min_periods=200).mean()},
        index=frame.index,
    )
    return _available_frame(context, "1D")


def _positioning_context(frame: pd.DataFrame) -> pd.DataFrame:
    context = pd.DataFrame(index=frame.index)
    context["funding_z"] = frame.get("funding_z", np.nan)
    context["oi_chg_4h"] = frame.get("oi_chg_4h", np.nan)
    context["positioning_stale_raw"] = frame.get("positioning_stale", True)
    context["positioning_age_min"] = frame.get("positioning_age_min", np.nan)
    if "availability_time" in frame:
        context["availability_time"] = frame["availability_time"]
    return _available_frame(context, "15min", index_is_available=True)


def _merge_context(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    left = left.sort_values("decision_time", kind="stable")
    right = right.sort_values("availability_time", kind="stable").drop(
        columns="source_bar_time"
    )
    merged = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="availability_time",
        direction="backward",
        allow_exact_matches=True,
    )
    return merged.drop(columns="availability_time")


def _attach_features(
    candidates: pd.DataFrame, frames: DecisionFrames, cadence: str
) -> pd.DataFrame:
    out = _attach_window_extreme_features(candidates)
    out = _merge_context(out, _swing_context(frames.fifteen_minute))
    out = _merge_context(out, _hourly_context(frames.hourly))
    out = _merge_context(out, _minute_context(frames.minute))
    out = _merge_context(out, _return_context(frames.five_minute, "5min", "return_5m"))
    out = _merge_context(out, _daily_context(frames.daily))
    out = _merge_context(out, _positioning_context(frames.positioning))
    source = frames.minute if cadence == "1min" else frames.five_minute
    out = _merge_context(out, _sequence_context(source, cadence))

    side_sign = out["side"].map({"long": 1.0, "short": -1.0}).to_numpy()
    price = out["decision_close"].to_numpy(dtype=float)
    lower = out["channel_lower"].to_numpy(dtype=float)
    mid = out["channel_mid"].to_numpy(dtype=float)
    upper = out["channel_upper"].to_numpy(dtype=float)
    width = upper - lower
    with np.errstate(divide="ignore", invalid="ignore"):
        raw_position = (price - lower) / width
        out["channel_slope_side"] = out["channel_slope"].to_numpy(dtype=float) * side_sign
        out["channel_width_pct"] = width / price
        out["channel_position_side"] = np.where(
            side_sign > 0, 1.0 - raw_position, raw_position
        )
        macro_distance = (
            out["macro_close"].to_numpy(dtype=float)
            / out["macro_sma200"].to_numpy(dtype=float)
            - 1.0
        ) * side_sign
        out["macro_sma_distance_side"] = macro_distance
        out["macro_alignment"] = np.where(
            np.isfinite(macro_distance), (macro_distance > 0).astype(float), np.nan
        )
        age_minutes = (
            out["decision_time"] - pd.to_datetime(out["window_start"], utc=True)
        ).dt.total_seconds() / 60.0
        out["window_age_fraction"] = (age_minutes / _WINDOW_MAX_MINUTES).clip(0.0, 1.0)
        pullback_scale = np.where(side_sign > 0, mid - lower, upper - mid)
        out["pullback_depth"] = side_sign * (mid - price) / pullback_scale
        swing_ref = np.where(
            side_sign > 0,
            out["swing_low_15m"].to_numpy(dtype=float),
            out["swing_high_15m"].to_numpy(dtype=float),
        )
        measured = out["measured_move_15m"].to_numpy(dtype=float)
        out["swing_reversal_score"] = side_sign * (price - swing_ref) / measured
        candle_range = (
            out["decision_high"].to_numpy(dtype=float)
            - out["decision_low"].to_numpy(dtype=float)
        )
        lower_wick = (
            np.minimum(
                out["decision_open"].to_numpy(dtype=float),
                out["decision_close"].to_numpy(dtype=float),
            )
            - out["decision_low"].to_numpy(dtype=float)
        )
        upper_wick = (
            out["decision_high"].to_numpy(dtype=float)
            - np.maximum(
                out["decision_open"].to_numpy(dtype=float),
                out["decision_close"].to_numpy(dtype=float),
            )
        )
        out["candle_rejection_side"] = np.where(
            side_sign > 0, lower_wick / candle_range, upper_wick / candle_range
        )
        volume = out["decision_volume"].to_numpy(dtype=float)
        taker = out["decision_taker_buy_base"].to_numpy(dtype=float)
        out["taker_imbalance_side"] = (2.0 * taker / volume - 1.0) * side_sign
        out["body_direction_side"] = (
            out["body_direction_raw"].to_numpy(dtype=float) * side_sign
        )
        out["reversal_count_3"] = np.where(
            side_sign > 0,
            out["up_count_3"].to_numpy(dtype=float),
            out["down_count_3"].to_numpy(dtype=float),
        )
        out["rejection_mean_3_side"] = np.where(
            side_sign > 0,
            out["lower_rejection_mean_3"].to_numpy(dtype=float),
            out["upper_rejection_mean_3"].to_numpy(dtype=float),
        )
        out["taker_imbalance_mean_3_side"] = (
            out["taker_imbalance_mean_3"].to_numpy(dtype=float) * side_sign
        )
        out["taker_imbalance_delta_side"] = (
            out["taker_imbalance_delta"].to_numpy(dtype=float) * side_sign
        )
        out["return_1m_side"] = out["return_1m"].to_numpy(dtype=float) * side_sign
        out["return_5m_side"] = out["return_5m"].to_numpy(dtype=float) * side_sign
        out["funding_z_side"] = out["funding_z"].to_numpy(dtype=float) * side_sign
        out["oi_chg_4h_side"] = out["oi_chg_4h"].to_numpy(dtype=float) * side_sign
        out["positioning_age_log"] = np.log1p(
            out["positioning_age_min"].astype(float).clip(lower=0.0)
        )
        entry = price * (1.0 - side_sign * _LIMIT_OFFSET_BPS / 1e4)
        stop = swing_ref * (1.0 - side_sign * _STOP_BUFFER_BPS / 1e4)
        risk = side_sign * (entry - stop)
        out["risk_bps_decision"] = risk / entry * 1e4
        out["rr_planned_decision"] = np.where(risk > 0, measured / risk, np.nan)

    out["channel_r2"] = out["channel_r2"].astype(float)
    out["channel_confluence"] = out["channel_confluence"].astype(float)
    out["positioning_stale"] = out["positioning_stale_raw"].fillna(True).astype("int8")
    missing_features = sorted(set(CHANNEL_WINDOW_FEATURES).difference(out.columns))
    if missing_features:
        raise AssertionError(f"feature construction missed: {missing_features}")
    if set(CHANNEL_WINDOW_FEATURES).intersection(FORBIDDEN_WINDOW_FEATURES):
        raise AssertionError("outcome metadata leaked into CHANNEL_WINDOW_FEATURES")
    return out.sort_values(["decision_time", "side", "window_id"], kind="stable").reset_index(
        drop=True
    )


def build_decision_candidates(
    manifest: pd.DataFrame,
    frames: DecisionFrames,
    *,
    cadence: str,
) -> pd.DataFrame:
    """Build every completed decision row inside each eligible channel window."""
    missing = sorted(_MANIFEST_REQUIRED.difference(manifest.columns))
    if missing:
        raise ValueError(f"window manifest missing columns: {missing}")
    if cadence not in ("1min", "5min"):
        raise ValueError("cadence must be '1min' or '5min'")
    if manifest.empty:
        return pd.DataFrame(columns=[*CHANNEL_WINDOW_FEATURES])
    source = frames.minute if cadence == "1min" else frames.five_minute
    candidates = _decision_rows(manifest, source, cadence)
    if candidates.empty:
        return pd.DataFrame(columns=[*candidates.columns, *CHANNEL_WINDOW_FEATURES])
    if not candidates["candidate_id"].is_unique:
        raise ValueError("overlapping manifest rows created duplicate candidates")
    return _attach_features(candidates, frames, cadence)


def _execution_grid(execution_1m: pd.DataFrame, cadence: str) -> pd.DataFrame:
    required = set(_OHLC)
    missing = sorted(required.difference(execution_1m.columns))
    if missing:
        raise ValueError(f"execution_1m missing columns: {missing}")
    if not isinstance(execution_1m.index, pd.DatetimeIndex):
        raise TypeError("execution_1m needs a DatetimeIndex")
    minute = execution_1m.sort_index()
    if cadence == "1min":
        grid = minute[list(_OHLC)].copy()
        grid["minute_count"] = 1
        return grid
    if cadence != "5min":
        raise ValueError(f"unsupported candidate cadence: {cadence!r}")
    grid = minute.resample("5min", label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
    )
    grid["minute_count"] = minute["close"].resample(
        "5min", label="left", closed="left"
    ).count()
    return grid.dropna(subset=list(_OHLC))


def _window_regimes(grid: pd.DataFrame, candidates: pd.DataFrame, cadence: str) -> pd.Series:
    regime = pd.Series("none", index=grid.index, dtype=object)
    delta = pd.Timedelta(cadence)
    lifecycle = candidates[
        ["window_id", "window_start", "eligible_end_time"]
    ].drop_duplicates("window_id")
    for window in lifecycle.itertuples(index=False):
        start = pd.Timestamp(window.window_start)
        end = pd.Timestamp(window.eligible_end_time)
        start = start.tz_localize("UTC") if start.tzinfo is None else start.tz_convert("UTC")
        end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
        mask = (regime.index >= start - delta) & (regime.index < end)
        conflict = mask & regime.ne("none") & regime.ne(window.window_id)
        if conflict.any():
            raise ValueError("overlapping windows cannot share an execution-grid bar")
        regime.loc[mask] = window.window_id
    return regime


def _label_one_cadence(
    candidates: pd.DataFrame,
    execution_1m: pd.DataFrame,
    config: WindowLabelConfig,
) -> pd.DataFrame:
    cadence_values = candidates["cadence"].dropna().unique()
    if len(cadence_values) != 1:
        raise ValueError("label group must contain exactly one cadence")
    cadence = str(cadence_values[0])
    grid = _execution_grid(execution_1m, cadence)
    grid["signal"] = 0
    grid["window_regime"] = _window_regimes(grid, candidates, cadence)
    grid["channel_episode_id"] = 0
    for column in ("swing_low_15m", "swing_high_15m", "measured_move_15m"):
        grid[column] = np.nan

    finite_geometry = (
        candidates[["swing_low_15m", "swing_high_15m", "measured_move_15m",
                    "risk_bps_decision", "rr_planned_decision"]]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .notna()
        .all(axis=1)
    )
    eligible = candidates.loc[
        finite_geometry
        & candidates["risk_bps_decision"].between(
            config.min_risk_bps, config.max_risk_bps, inclusive="both"
        )
        & candidates["rr_planned_decision"].ge(config.min_rr)
        & candidates["measured_move_15m"].gt(0)
    ].copy()
    if eligible.empty:
        return pd.DataFrame(columns=[*candidates.columns, "order_status", "r_net"])
    if eligible["source_bar_time"].duplicated().any():
        raise ValueError("more than one candidate occupies the same signal bar")

    side_value = eligible["side"].map({"long": 1, "short": -1})
    if side_value.isna().any():
        raise ValueError("candidate side must be 'long' or 'short'")
    for row, signal_value in zip(eligible.itertuples(index=False), side_value, strict=True):
        signal_time = pd.Timestamp(row.source_bar_time)
        if signal_time not in grid.index:
            continue
        grid.loc[signal_time, "signal"] = int(signal_value)
        grid.loc[signal_time, "window_regime"] = row.window_id
        grid.loc[signal_time, "channel_episode_id"] = row.channel_episode_id
        grid.loc[signal_time, "swing_low_15m"] = row.swing_low_15m
        grid.loc[signal_time, "swing_high_15m"] = row.swing_high_15m
        grid.loc[signal_time, "measured_move_15m"] = row.measured_move_15m

    result = backtest_channel_strategy(
        grid,
        **LABEL_EXECUTION,
        stop_buffer_bps=_STOP_BUFFER_BPS,
        min_risk_bps=config.min_risk_bps,
        max_risk_bps=config.max_risk_bps,
        min_rr=config.min_rr,
        max_hold_minutes=config.max_hold_minutes,
        limit_offset_bps=config.limit_offset_bps,
        fill_window_minutes=config.fill_window_minutes,
        maker_fee_bps=config.maker_fee_bps,
        taker_fee_bps=config.taker_fee_bps,
        regime_col="window_regime",
        episode_col="channel_episode_id",
        execution_1m=execution_1m,
    )
    orders = result["orders_df"].rename(
        columns={"signal_time": "source_bar_time", "status": "order_status"}
    )
    order_columns = [
        "source_bar_time", "order_status", "filled", "entry_time", "exit_time",
        "active_end_time", "entry", "stop", "target", "outcome", "r_net",
    ]
    labelled = eligible.merge(orders[order_columns], on="source_bar_time", how="inner")
    labelled = labelled[labelled["order_status"].ne("censored") & labelled["r_net"].notna()]
    labelled["label_start"] = pd.to_datetime(labelled["next_entry_time"], utc=True)
    labelled["active_end_time"] = pd.to_datetime(labelled["active_end_time"], utc=True)
    labelled["label_end"] = labelled["active_end_time"]
    if (labelled["label_end"] < labelled["label_start"]).any():
        raise AssertionError("label interval ends before the order became active")
    labelled["label_net_positive"] = labelled["r_net"].gt(0).astype("int8")
    return labelled


def label_decision_candidates(
    candidates: pd.DataFrame,
    execution_1m: pd.DataFrame,
    *,
    config: WindowLabelConfig,
) -> pd.DataFrame:
    """Label each economically eligible decision without portfolio constraints."""
    required = {
        "candidate_id", "window_id", "channel_episode_id", "side", "cadence",
        "source_bar_time", "decision_time", "next_entry_time", "window_start",
        "eligible_end_time", "swing_low_15m", "swing_high_15m",
        "measured_move_15m", "risk_bps_decision", "rr_planned_decision",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"candidates missing label columns: {missing}")
    if candidates.empty:
        return candidates.copy()
    groups = [
        _label_one_cadence(group.copy(), execution_1m, config)
        for _, group in candidates.groupby("cadence", sort=False)
    ]
    labelled = pd.concat(groups, ignore_index=True)
    if not labelled.empty and not labelled["candidate_id"].is_unique:
        raise AssertionError("a candidate received more than one economic label")
    return labelled.sort_values(["decision_time", "candidate_id"], kind="stable").reset_index(
        drop=True
    )
