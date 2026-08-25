"""Causal raw inputs shared by the event-window sequence and label layers.

The source index is the bar-open timestamp.  Five-minute values become known at
``index + 5 minutes``; positioning values become known at their explicit
availability timestamp or, conservatively, at ``index + 15 minutes``.  Every
rolling transform is backward-looking and gaps reset its usable history.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.event_windows import causal_activity_ratio


RAW_FIVE_MINUTE_FEATURES = (
    "log_return",
    "range_bps",
    "body_bps",
    "lower_wick_fraction",
    "upper_wick_fraction",
    "volume_log_ratio_24",
    "quote_volume_log_ratio_24",
    "trade_count_log_ratio_24",
    "taker_imbalance",
    "taker_imbalance_mean_3",
    "taker_imbalance_delta",
    "realized_vol_12",
    "activity_ratio",
    "channel_pos",
    "distance_lower_bps",
    "distance_mid_bps",
    "distance_upper_bps",
    "bar_complete",
    "cadence_gap",
)

POSITIONING_FEATURES = (
    "oi_chg_15m",
    "oi_chg_1h",
    "oi_chg_4h",
    "oi_accel_1h",
    "oi_z_7d",
    "funding_rate",
    "funding_z",
    "toptrader_log_ratio",
    "taker_log_ratio",
    "oi_missing",
    "funding_missing",
    "toptrader_missing",
    "taker_ratio_missing",
    "positioning_missing",
    "positioning_stale",
    "positioning_age_min",
)

_OHLC = ("open", "high", "low", "close")
_FIVE_MINUTES = pd.Timedelta("5min")
_FIFTEEN_MINUTES = pd.Timedelta("15min")
_POSITIONING_Z_ROWS = 672


def _utc_index(frame: pd.DataFrame, *, name: str) -> pd.DatetimeIndex:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} needs a DatetimeIndex")
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"{name} index must be unique and increasing")
    return index


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _completed_five_minute_bars(frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(_OHLC).difference(frame.columns))
    if missing:
        raise ValueError(f"five-minute frame missing OHLC columns: {missing}")
    complete = np.ones(len(frame), dtype=bool)
    for column in _OHLC:
        complete &= np.isfinite(_numeric(frame, column).to_numpy(dtype=float))
    if "minute_count" in frame:
        complete &= _numeric(frame, "minute_count").to_numpy(dtype=float) == 5.0
    if "bar_complete" in frame:
        complete &= frame["bar_complete"].fillna(False).astype(bool).to_numpy()
    return complete


def _contiguous_run_lengths(
    index: pd.DatetimeIndex, complete: np.ndarray, cadence: pd.Timedelta
) -> np.ndarray:
    runs = np.zeros(len(index), dtype=np.int32)
    for row in range(len(index)):
        if not complete[row]:
            continue
        if (
            row > 0
            and complete[row - 1]
            and index[row] - index[row - 1] == cadence
        ):
            runs[row] = runs[row - 1] + 1
        else:
            runs[row] = 1
    return runs


def _previous_24_log_surprise(values: pd.Series, runs: np.ndarray) -> pd.Series:
    logged = np.log1p(values.where(values >= 0.0))
    prior_median = logged.shift(1).rolling(24, min_periods=24).median()
    surprise = logged - prior_median
    return surprise.where(runs >= 25)


def _rail(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame:
        return _numeric(frame, name)
    primary_name = f"{name}_60"
    return _numeric(frame, primary_name)


def build_five_minute_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach raw causal transforms while retaining source and channel columns."""
    index = _utc_index(frame, name="five-minute frame")
    out = frame.copy()
    out.index = index
    complete = _completed_five_minute_bars(out)
    gaps = np.zeros(len(out), dtype=bool)
    if len(out) > 1:
        gaps[1:] = index[1:] - index[:-1] != _FIVE_MINUTES
    runs = _contiguous_run_lengths(index, complete, _FIVE_MINUTES)

    open_ = _numeric(out, "open")
    high = _numeric(out, "high")
    low = _numeric(out, "low")
    close = _numeric(out, "close")
    positive_close = close.where(close > 0.0)
    price_range = (high - low).where((high - low) > 0.0)
    positive_open = open_.where(open_ > 0.0)

    out["log_return"] = np.log(positive_close).diff().where(runs >= 2)
    out["range_bps"] = ((high - low) / positive_open * 1e4).where(complete)
    out["body_bps"] = ((close - open_) / positive_open * 1e4).where(complete)
    out["lower_wick_fraction"] = (
        (np.minimum(open_, close) - low) / price_range
    ).where(complete)
    out["upper_wick_fraction"] = (
        (high - np.maximum(open_, close)) / price_range
    ).where(complete)

    out["volume_log_ratio_24"] = _previous_24_log_surprise(
        _numeric(out, "volume"), runs
    )
    out["quote_volume_log_ratio_24"] = _previous_24_log_surprise(
        _numeric(out, "quote_volume"), runs
    )
    trade_count_column = "trade_count" if "trade_count" in out else "count"
    out["trade_count_log_ratio_24"] = _previous_24_log_surprise(
        _numeric(out, trade_count_column), runs
    )

    volume = _numeric(out, "volume").where(lambda values: values > 0.0)
    taker_buy = _numeric(out, "taker_buy_base")
    imbalance = (2.0 * taker_buy / volume - 1.0).where(complete)
    out["taker_imbalance"] = imbalance
    out["taker_imbalance_mean_3"] = imbalance.rolling(3, min_periods=3).mean().where(
        runs >= 3
    )
    out["taker_imbalance_delta"] = imbalance.diff().where(runs >= 2)
    out["realized_vol_12"] = (
        out["log_return"].rolling(12, min_periods=12).std().where(runs >= 13)
    )

    activity = (
        _numeric(out, "activity_ratio")
        if "activity_ratio" in out
        else causal_activity_ratio(out)
    )
    out["activity_ratio"] = activity.where(complete & (runs >= 289))
    out["channel_pos"] = _numeric(out, "channel_pos").where(complete)
    denominator = positive_close
    out["distance_lower_bps"] = ((close - _rail(out, "channel_lower")) / denominator * 1e4).where(
        complete
    )
    out["distance_mid_bps"] = ((close - _rail(out, "channel_mid")) / denominator * 1e4).where(
        complete
    )
    out["distance_upper_bps"] = ((close - _rail(out, "channel_upper")) / denominator * 1e4).where(
        complete
    )
    out["bar_complete"] = complete.astype("int8")
    out["cadence_gap"] = gaps.astype("int8")

    if "availability_time" in out:
        out["availability_time"] = pd.to_datetime(out["availability_time"], utc=True)
    else:
        out["availability_time"] = index + _FIVE_MINUTES
    return out


def _positioning_run_lengths(index: pd.DatetimeIndex) -> np.ndarray:
    complete = np.ones(len(index), dtype=bool)
    return _contiguous_run_lengths(index, complete, _FIFTEEN_MINUTES)


def _positive_log(values: pd.Series) -> pd.Series:
    return np.log(values.where(values > 0.0))


def _causal_zscore(values: pd.Series, runs: np.ndarray) -> pd.Series:
    mean = values.rolling(_POSITIONING_Z_ROWS, min_periods=_POSITIONING_Z_ROWS).mean()
    std = values.rolling(_POSITIONING_Z_ROWS, min_periods=_POSITIONING_Z_ROWS).std()
    return ((values - mean) / std.replace(0.0, np.nan)).where(
        runs >= _POSITIONING_Z_ROWS
    )


def build_positioning_feature_frame(positioning: pd.DataFrame) -> pd.DataFrame:
    """Build features once per completed 15m observation, before any 5m carry.

    OI changes and its seven-day z-score use positive log OI, matching the
    project's existing scale-stable positioning convention.
    """
    index = _utc_index(positioning, name="positioning frame")
    out = positioning.copy()
    out.index = index

    if "availability_time" in out:
        availability = pd.to_datetime(out.pop("availability_time"), utc=True)
    elif "positioning_availability_time" in out:
        availability = pd.to_datetime(
            out.pop("positioning_availability_time"), utc=True
        )
    else:
        availability = pd.Series(index + _FIFTEEN_MINUTES, index=index)
    availability_index = pd.DatetimeIndex(availability)
    if availability_index.has_duplicates or not availability_index.is_monotonic_increasing:
        raise ValueError(
            "positioning availability must be unique and monotone in source order"
        )
    if (availability_index < index).any():
        raise ValueError("positioning cannot be available before its source timestamp")
    out["positioning_source_time"] = index
    out["positioning_availability_time"] = availability_index

    runs = _positioning_run_lengths(index)
    oi = _numeric(out, "sum_open_interest")
    funding = _numeric(out, "funding_rate")
    toptrader = _numeric(out, "toptrader_ls")
    taker = _numeric(out, "taker_ls")
    log_oi = _positive_log(oi)

    for name, rows in (("oi_chg_15m", 1), ("oi_chg_1h", 4), ("oi_chg_4h", 16)):
        out[name] = log_oi.diff(rows).where(runs >= rows + 1)
    out["oi_accel_1h"] = (out["oi_chg_1h"] - out["oi_chg_1h"].shift(4)).where(
        runs >= 9
    )
    out["oi_z_7d"] = _causal_zscore(log_oi, runs)
    out["funding_rate"] = funding
    out["funding_z"] = _causal_zscore(funding, runs)
    out["toptrader_log_ratio"] = _positive_log(toptrader)
    out["taker_log_ratio"] = _positive_log(taker)

    out["oi_missing"] = (~np.isfinite(oi) | oi.le(0.0)).astype("int8")
    out["funding_missing"] = (~np.isfinite(funding)).astype("int8")
    out["toptrader_missing"] = (
        ~np.isfinite(toptrader) | toptrader.le(0.0)
    ).astype("int8")
    out["taker_ratio_missing"] = (~np.isfinite(taker) | taker.le(0.0)).astype(
        "int8"
    )
    out["positioning_missing"] = np.zeros(len(out), dtype="int8")

    if "positioning_stale" in positioning:
        source_stale = positioning["positioning_stale"].fillna(True).astype(bool)
    else:
        source_stale = pd.Series(False, index=positioning.index)
    if "positioning_age_min" in positioning:
        source_age = pd.to_numeric(
            positioning["positioning_age_min"], errors="coerce"
        ).astype(float)
    else:
        source_age = pd.Series(0.0, index=positioning.index)
    source_stale.index = index
    source_age.index = index
    out["positioning_age_min"] = source_age.clip(lower=0.0)
    out["positioning_stale"] = (
        source_stale
        | out["positioning_age_min"].isna()
        | out["positioning_age_min"].gt(60.0)
    ).astype("int8")
    return out


def merge_positioning_asof(
    decisions: pd.DataFrame, positioning_features: pd.DataFrame
) -> pd.DataFrame:
    """Backward-asof positioning onto decisions and advance carried age."""
    if "decision_time" not in decisions:
        raise ValueError("decisions missing 'decision_time'")
    overlap = sorted(set(POSITIONING_FEATURES).intersection(decisions.columns))
    if overlap:
        raise ValueError(f"decisions already contain positioning features: {overlap}")

    left = decisions.copy()
    left["decision_time"] = pd.to_datetime(left["decision_time"], utc=True)
    left["_event_input_order"] = np.arange(len(left), dtype=np.int64)
    left = left.sort_values("decision_time", kind="stable")

    required = {"positioning_availability_time", *POSITIONING_FEATURES}
    missing = sorted(required.difference(positioning_features.columns))
    if missing:
        raise ValueError(f"positioning feature frame missing columns: {missing}")
    right_columns = [
        "positioning_source_time",
        "positioning_availability_time",
        *POSITIONING_FEATURES,
    ]
    right_columns = [column for column in right_columns if column in positioning_features]
    right = positioning_features[right_columns].copy()
    right["positioning_availability_time"] = pd.to_datetime(
        right["positioning_availability_time"], utc=True
    )
    right = right.loc[right["positioning_availability_time"].notna()].sort_values(
        "positioning_availability_time", kind="stable"
    )
    if right["positioning_availability_time"].duplicated().any():
        raise ValueError("positioning availability times must be unique")

    merged = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="positioning_availability_time",
        direction="backward",
        allow_exact_matches=True,
    )
    matched = merged["positioning_availability_time"].notna()
    elapsed = (
        merged["decision_time"] - merged["positioning_availability_time"]
    ).dt.total_seconds() / 60.0
    base_age = pd.to_numeric(merged["positioning_age_min"], errors="coerce")
    merged["positioning_age_min"] = (base_age + elapsed).where(matched)
    source_stale = merged["positioning_stale"].fillna(True).astype(bool)
    merged["positioning_stale"] = (
        source_stale
        | merged["positioning_age_min"].isna()
        | merged["positioning_age_min"].gt(60.0)
    ).astype("int8")
    merged["positioning_missing"] = (~matched).astype("int8")
    for column in (
        "oi_missing",
        "funding_missing",
        "toptrader_missing",
        "taker_ratio_missing",
    ):
        merged[column] = merged[column].fillna(1).astype("int8")

    return merged.sort_values("_event_input_order", kind="stable").drop(
        columns="_event_input_order"
    ).reset_index(drop=True)


def known_structural_stop(
    frame: pd.DataFrame,
    *,
    side: str,
    source_bar_time: pd.Timestamp,
    lookback: int = 12,
    buffer_bps: float = 5.0,
) -> float:
    """Return the trailing completed-bar adverse extreme plus a fixed buffer."""
    if lookback < 1:
        raise ValueError("lookback must be positive")
    if buffer_bps < 0.0:
        raise ValueError("buffer_bps must be non-negative")
    normalized_side = str(side).lower()
    if normalized_side not in {"long", "short"}:
        raise ValueError("side must be 'long' or 'short'")

    index = _utc_index(frame, name="structural-stop frame")
    source = pd.Timestamp(source_bar_time)
    source = source.tz_localize("UTC") if source.tzinfo is None else source.tz_convert("UTC")
    position = index.get_indexer([source])[0]
    if position < 0 or position + 1 < lookback:
        return float("nan")
    start = position + 1 - lookback
    history_index = index[start : position + 1]
    expected = pd.date_range(source - (lookback - 1) * _FIVE_MINUTES, source, freq="5min")
    if not history_index.equals(expected):
        return float("nan")

    normalized = frame.copy()
    normalized.index = index
    history = normalized.iloc[start : position + 1]
    if not _completed_five_minute_bars(history).all():
        return float("nan")
    if normalized_side == "long":
        extreme = _numeric(history, "low").min()
        return float(extreme * (1.0 - buffer_bps / 1e4))
    extreme = _numeric(history, "high").max()
    return float(extreme * (1.0 + buffer_bps / 1e4))
