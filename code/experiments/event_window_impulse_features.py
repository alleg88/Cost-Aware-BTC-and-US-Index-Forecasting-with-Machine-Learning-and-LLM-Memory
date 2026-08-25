"""Causal impulse-arrival features for the Notebook T timing ablation."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


IMPULSE_FEATURE_COLUMNS = (
    "impulse_count_15m",
    "impulse_count_30m",
    "impulse_count_60m",
    "impulse_count_120m",
    "minutes_since_last_impulse",
    "median_interarrival_last_5",
    "impulse_excess_energy_60m",
)


@dataclass(frozen=True)
class ImpulseFeatureConfig:
    bar_minutes: int = 5
    lookback_bars: int = 2_016
    quantile: float = 0.90
    count_windows_minutes: tuple[int, ...] = (15, 30, 60, 120)
    energy_window_minutes: int = 60


def _validated_config(config: ImpulseFeatureConfig) -> None:
    if config.bar_minutes != 5:
        raise ValueError("Notebook T requires completed five-minute bars")
    if config.lookback_bars < 2:
        raise ValueError("impulse lookback requires at least two returns")
    if not 0.0 < config.quantile < 1.0:
        raise ValueError("impulse quantile must be between zero and one")
    windows = (*config.count_windows_minutes, config.energy_window_minutes)
    if any(window <= 0 or window % config.bar_minutes for window in windows):
        raise ValueError("impulse windows must be positive multiples of five minutes")


def _regular_five_minute_frame(
    five_minute: pd.DataFrame,
    *,
    bar_minutes: int,
) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(five_minute.columns))
    if missing:
        raise ValueError(f"five-minute source missing columns: {missing}")
    if five_minute.empty:
        raise ValueError("five-minute source is empty")
    source = five_minute.copy()
    source.index = pd.DatetimeIndex(
        pd.to_datetime(source.index, utc=True, errors="raise"),
        name="source_bar_time",
    )
    source = source.sort_index(kind="stable")
    if not source.index.is_unique:
        raise ValueError("five-minute source times must be unique")
    cadence = pd.Timedelta(minutes=bar_minutes)
    regular_index = pd.date_range(
        source.index.min(), source.index.max(), freq=cadence, name="source_bar_time"
    )
    return source.reindex(regular_index)


def _complete_bar_mask(source: pd.DataFrame) -> pd.Series:
    ohlc = source.loc[:, ["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    )
    complete = pd.Series(
        np.isfinite(ohlc.to_numpy(float)).all(axis=1), index=source.index
    )
    complete &= ohlc["close"].gt(0.0)
    if "minute_count" in source:
        complete &= pd.to_numeric(source["minute_count"], errors="coerce").eq(5)
    if "bar_complete" in source:
        complete &= source["bar_complete"].fillna(False).astype(bool)
    return complete.astype(bool)


def build_impulse_feature_frame(
    five_minute: pd.DataFrame,
    *,
    config: ImpulseFeatureConfig = ImpulseFeatureConfig(),
) -> pd.DataFrame:
    """Build seven features known immediately after each completed five-minute bar."""
    _validated_config(config)
    source = _regular_five_minute_frame(
        five_minute, bar_minutes=config.bar_minutes
    )
    complete = _complete_bar_mask(source)
    close = pd.to_numeric(source["close"], errors="coerce")
    log_return = np.log(close).diff()
    valid_return = complete & complete.shift(1, fill_value=False)
    log_return = log_return.where(valid_return)
    absolute_return = log_return.abs()

    # The latest return is not in its own threshold. With a regularised grid,
    # requiring every shifted value also rebuilds the full history after a gap.
    threshold = (
        absolute_return.shift(1)
        .rolling(config.lookback_bars, min_periods=config.lookback_bars)
        .quantile(config.quantile, interpolation="linear")
    )
    event_valid = log_return.notna() & threshold.notna()
    event = absolute_return.gt(threshold) & event_valid
    event_value = event.astype(float).where(event_valid)

    decision_index = pd.DatetimeIndex(
        source.index + pd.Timedelta(minutes=config.bar_minutes),
        name="decision_time",
    )
    features: dict[str, np.ndarray] = {}
    for window in config.count_windows_minutes:
        bars = window // config.bar_minutes
        features[f"impulse_count_{window}m"] = (
            event_value.rolling(bars, min_periods=bars).sum().to_numpy(float)
        )

    minutes_since = np.full(len(source), np.nan, dtype=float)
    median_interarrival = np.full(len(source), np.nan, dtype=float)
    recent_events: deque[pd.Timestamp] = deque(maxlen=5)
    previous_valid = False
    for position, valid in enumerate(event_valid.to_numpy(bool)):
        if not valid:
            recent_events.clear()
            previous_valid = False
            continue
        if not previous_valid:
            recent_events.clear()
        decision_time = decision_index[position]
        if bool(event.iloc[position]):
            recent_events.append(decision_time)
        if recent_events:
            minutes_since[position] = (
                decision_time - recent_events[-1]
            ).total_seconds() / 60.0
        if len(recent_events) == 5:
            gaps = np.diff(pd.DatetimeIndex(recent_events).asi8) / 60_000_000_000.0
            median_interarrival[position] = float(np.median(gaps))
        previous_valid = True
    features["minutes_since_last_impulse"] = minutes_since
    features["median_interarrival_last_5"] = median_interarrival

    excess_bps = (absolute_return - threshold).clip(lower=0.0) * 10_000.0
    excess_energy = excess_bps.pow(2).where(event_valid)
    energy_bars = config.energy_window_minutes // config.bar_minutes
    features["impulse_excess_energy_60m"] = (
        excess_energy.rolling(energy_bars, min_periods=energy_bars)
        .sum()
        .to_numpy(float)
    )
    result = pd.DataFrame(features, index=decision_index)
    return result.loc[:, IMPULSE_FEATURE_COLUMNS]


def append_impulse_features(
    dataset: LargeMoveDecisionDataset,
    impulse_features: pd.DataFrame,
) -> LargeMoveDecisionDataset:
    """Append a unique decision-time feature table to row-repeated window decisions."""
    required_keys = {"window_id", "step", "decision_time"}
    missing = sorted(required_keys.difference(dataset.decisions.columns))
    if missing:
        raise ValueError(f"impulse join keys are missing: {missing}")
    if dataset.decisions.duplicated(["window_id", "step"]).any():
        raise ValueError("impulse decisions require unique window_id, step keys")
    if dataset.tabular.ndim != 2 or len(dataset.tabular) != len(dataset.decisions):
        raise ValueError("impulse feature rows must align with the decision matrix")
    if dataset.tabular.shape[1] != len(dataset.tabular_features):
        raise ValueError("impulse feature names must align with the decision matrix")
    duplicate_names = sorted(
        set(IMPULSE_FEATURE_COLUMNS).intersection(dataset.tabular_features)
    )
    if duplicate_names:
        raise ValueError(f"impulse features are already present: {duplicate_names}")

    source = impulse_features.copy()
    missing_features = sorted(set(IMPULSE_FEATURE_COLUMNS).difference(source.columns))
    if missing_features:
        raise ValueError(f"impulse source missing columns: {missing_features}")
    source.index = pd.DatetimeIndex(
        pd.to_datetime(source.index, utc=True, errors="raise"), name="decision_time"
    )
    if not source.index.is_unique:
        raise ValueError("impulse source requires unique decision_time values")
    decision_times = pd.DatetimeIndex(
        pd.to_datetime(dataset.decisions["decision_time"], utc=True, errors="raise")
    )
    aligned = source.loc[:, IMPULSE_FEATURE_COLUMNS].reindex(decision_times)
    return LargeMoveDecisionDataset(
        decisions=dataset.decisions.copy(),
        tabular=np.column_stack(
            [
                np.asarray(dataset.tabular, dtype=np.float32),
                aligned.to_numpy(dtype=np.float32),
            ]
        ).astype(np.float32, copy=False),
        tabular_features=(*dataset.tabular_features, *IMPULSE_FEATURE_COLUMNS),
        dropped_features=dataset.dropped_features,
        feature_set=f"{dataset.feature_set}+impulse_v1",
    )


__all__ = [
    "IMPULSE_FEATURE_COLUMNS",
    "ImpulseFeatureConfig",
    "append_impulse_features",
    "build_impulse_feature_frame",
]
