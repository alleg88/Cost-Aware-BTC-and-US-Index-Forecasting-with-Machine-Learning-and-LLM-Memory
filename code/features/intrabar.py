"""Causal one-minute aggregates and market-stage features on the M15 clock."""
from __future__ import annotations

import numpy as np
import pandas as pd

INTRABAR_FEATURES = (
    "m1_realized_vol",
    "m1_range",
    "m1_path_efficiency",
    "m1_close_location",
    "m1_final5_return",
    "m1_ofi_mean",
    "m1_ofi_std",
    "m1_late_volume_share",
)
STAGE_FEATURES = (
    "stage_return_1h",
    "stage_return_4h",
    "stage_return_1d",
    "stage_efficiency_4h",
    "stage_vol_percentile_1d",
)
REGIME_STAGE_FEATURES = (
    "stage_return_7d",
    "stage_drawdown_7d",
    "stage_rebound_7d",
)


def validate_minute_bars(minute: pd.DataFrame) -> None:
    required = {"open", "high", "low", "close", "volume", "taker_buy_base"}
    missing = required.difference(minute.columns)
    if missing:
        raise ValueError(f"missing 1m columns: {sorted(missing)}")
    if not isinstance(minute.index, pd.DatetimeIndex) or minute.index.tz is None:
        raise ValueError("1m index must be a timezone-aware DatetimeIndex")
    if not minute.index.is_unique or not minute.index.is_monotonic_increasing:
        raise ValueError("1m index must be unique and sorted")
    expected = pd.date_range(minute.index[0], minute.index[-1], freq="1min", tz=minute.index.tz)
    aligned = minute.index[0] == minute.index[0].floor("15min")
    complete = len(minute) % 15 == 0 and minute.index.equals(expected)
    if not aligned or not complete:
        raise ValueError("1m data must contain complete 15-minute groups")


def build_intrabar_features(minute: pd.DataFrame) -> pd.DataFrame:
    """Aggregate completed 1m bars into one causal feature row per M15 bar."""
    validate_minute_bars(minute)
    work = minute.copy()
    key = work.index.floor("15min")
    row = work.groupby(key).cumcount()
    minute_return = np.log(work["close"].astype(float) / work["open"].astype(float))
    volume = work["volume"].astype(float)
    ofi = 2.0 * work["taker_buy_base"].astype(float) / volume.replace(0.0, np.nan) - 1.0

    grouped = work.groupby(key)
    first_open = grouped["open"].first().astype(float)
    last_close = grouped["close"].last().astype(float)
    high = grouped["high"].max().astype(float)
    low = grouped["low"].min().astype(float)
    total_volume = volume.groupby(key).sum()
    path = minute_return.abs().groupby(key).sum().replace(0.0, np.nan)
    late_open = work.loc[row == 10, "open"].astype(float)
    late_open.index = key[row == 10]

    result = pd.DataFrame(index=first_open.index)
    result["m1_realized_vol"] = np.sqrt((minute_return ** 2).groupby(key).sum())
    result["m1_range"] = high / low - 1.0
    result["m1_path_efficiency"] = (np.log(last_close / first_open).abs() / path).clip(0.0, 1.0)
    result["m1_close_location"] = (last_close - low) / (high - low).replace(0.0, np.nan)
    result["m1_final5_return"] = last_close / late_open - 1.0
    result["m1_ofi_mean"] = ofi.groupby(key).mean()
    result["m1_ofi_std"] = ofi.groupby(key).std(ddof=0)
    result["m1_late_volume_share"] = volume.where(row >= 10, 0.0).groupby(key).sum() / total_volume
    result.index.name = minute.index.name
    return result.loc[:, INTRABAR_FEATURES]


def build_market_stage_features(
    m15: pd.DataFrame, *, include_regime_stage: bool = False
) -> pd.DataFrame:
    """Past-only trend and volatility state on native M15 closes."""
    if not isinstance(m15.index, pd.DatetimeIndex) or m15.index.tz is None:
        raise ValueError("M15 index must be timezone-aware")
    if not m15.index.is_unique or not m15.index.is_monotonic_increasing:
        raise ValueError("M15 index must be unique and sorted")
    close = m15["close"].astype(float)
    log_close = np.log(close)
    log_return = log_close.diff()
    movement_4h = log_return.abs().rolling(16).sum().replace(0.0, np.nan)
    vol_4h = log_return.rolling(16).std()

    result = pd.DataFrame(index=m15.index)
    result["stage_return_1h"] = close.pct_change(4)
    result["stage_return_4h"] = close.pct_change(16)
    result["stage_return_1d"] = close.pct_change(96)
    result["stage_efficiency_4h"] = (log_close.diff(16).abs() / movement_4h).clip(0.0, 1.0)
    result["stage_vol_percentile_1d"] = vol_4h.rolling(96).rank(pct=True)
    if include_regime_stage:
        window_7d = close.rolling(673)
        result["stage_return_7d"] = close.pct_change(672)
        result["stage_drawdown_7d"] = close / window_7d.max() - 1.0
        result["stage_rebound_7d"] = close / window_7d.min() - 1.0
        return result.loc[:, (*STAGE_FEATURES, *REGIME_STAGE_FEATURES)]
    return result.loc[:, STAGE_FEATURES]
