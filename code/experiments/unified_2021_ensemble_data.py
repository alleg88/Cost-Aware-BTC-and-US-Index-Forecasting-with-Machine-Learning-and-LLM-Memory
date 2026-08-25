"""Causal shared data contract for Notebook 04d's three-model ensemble."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.splits import BlockingTimeSeriesSplit
from experiments.event_window_direction_dataset import pair_direction_paths
from experiments.run_event_window_economic_feasibility import replay_brackets
from features.build import add_features
from features.event_window_inputs import (
    build_positioning_feature_frame,
    merge_positioning_asof,
)


UNIFIED_FEATURES = (
    "r1",
    "r5",
    "r20",
    "vol_10",
    "vol_20",
    "vol_60",
    "hl_range",
    "co_range",
    "rsi_14",
    "volume",
    "vol_z",
    "ofi",
    "ofi_z20",
    "ofi_mom5",
    "trade_intensity_z",
    "funding_rate",
    "funding_z",
    "oi_chg_15m",
    "oi_chg_1h",
    "oi_chg_4h",
    "oi_accel_1h",
    "oi_z",
    "toptrader_ls_z",
    "taker_ls_z",
    "price_oi_interaction",
    "positioning_stale",
    "positioning_age_log",
    "funding_missing",
    "open_interest_missing",
    "toptrader_ls_missing",
    "taker_ls_missing",
    "adaptive_barrier_bps",
    "past_rv_15_bps",
    "past_rv_30_bps",
    "past_rv_60_bps",
    "past_rv_120_bps",
    "rv_ratio_15_60",
    "rv_ratio_60_120",
    "past_range_15_bps",
    "past_range_60_bps",
    "past_range_120_bps",
    "range_ratio_15_120",
    "past_abs_return_15_bps",
    "past_abs_return_60_bps",
    "volume_ratio_15_120",
    "trade_count_ratio_15_120",
    "taker_imbalance_abs_delta_15_120",
    "abs_oi_chg_1h",
    "abs_oi_chg_4h",
    "abs_funding_z",
    "upside_semivar_60",
    "downside_semivar_60",
    "upside_semivar_240",
    "downside_semivar_240",
    "drawdown_60",
    "drawdown_240",
    "channel_position_20",
    "channel_slope_20",
    "body_bps",
    "lower_wick_fraction",
    "upper_wick_fraction",
    "taker_imbalance",
    "taker_imbalance_delta_15_120",
    "m15_incomplete",
)


@dataclass(frozen=True)
class UnifiedDataConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-01-01"
    h1_end_exclusive: str = "2025-07-01"
    lockbox_start: str = "2026-04-01"
    sequence_length: int = 32
    hold_minutes: int = 120
    target_multiple_b: float = 2.0
    stop_multiple_b: float = 1.0
    entry_cost_bps: float = 5.0
    exit_cost_bps: float = 5.0
    minimum_barrier_bps: float = 75.0
    maximum_barrier_bps: float = 250.0
    volatility_addon_weight: float = 0.5
    n_splits: int = 5
    train_fraction: float = 0.8
    calibration_fraction: float = 0.15
    embargo_bars: int = 8


@dataclass(frozen=True)
class UnifiedDataset:
    decisions: pd.DataFrame
    tabular: np.ndarray
    sequences: np.ndarray
    feature_names: tuple[str, ...]
    economic_paths: pd.DataFrame


def _utc_frame(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} needs a DatetimeIndex")
    output = frame.copy()
    index = pd.DatetimeIndex(output.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(f"{name} index must be unique and increasing")
    output.index = index
    return output


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").astype(float)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    top = pd.to_numeric(numerator, errors="coerce").to_numpy(float)
    bottom = pd.to_numeric(denominator, errors="coerce").to_numpy(float)
    values = np.divide(
        top,
        bottom,
        out=np.full(len(top), np.nan, dtype=float),
        where=np.isfinite(bottom) & (np.abs(bottom) > 1e-12),
    )
    return pd.Series(values, index=numerator.index, dtype=float)


def _rolling_zscore(values: pd.Series, window: int) -> pd.Series:
    mean = values.rolling(window, min_periods=window).mean()
    std = values.rolling(window, min_periods=window).std()
    return (values - mean) / std.replace(0.0, np.nan)


def _rolling_slope(values: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    centered_x = x - x.mean()
    denominator = float(np.square(centered_x).sum())

    def slope(raw: np.ndarray) -> float:
        if not np.isfinite(raw).all():
            return np.nan
        return float(np.dot(centered_x, raw - raw.mean()) / denominator * 1e4)

    return values.rolling(window, min_periods=window).apply(slope, raw=True)


def _validate_config(config: UnifiedDataConfig) -> None:
    if config.sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if config.hold_minutes != 120:
        raise ValueError("Notebook 04d hold_minutes must remain 120")
    if config.target_multiple_b != 2.0 or config.stop_multiple_b != 1.0:
        raise ValueError("Notebook 04d must retain RR2 geometry")
    if config.entry_cost_bps != 5.0 or config.exit_cost_bps != 5.0:
        raise ValueError("Notebook 04d must retain 5+5 bps costs")
    if config.embargo_bars != 8:
        raise ValueError("Notebook 04d embargo must remain eight M15 bars")
    if not 0.0 < config.calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")


def build_unified_decisions(
    m15: pd.DataFrame,
    positioning: pd.DataFrame,
    config: UnifiedDataConfig = UnifiedDataConfig(),
) -> pd.DataFrame:
    """Build one causal feature row for every supplied completed M15 bar."""
    _validate_config(config)
    bars = _utc_frame(m15, name="M15 frame")
    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise ValueError(f"M15 frame missing columns: {missing}")
    lockbox = pd.Timestamp(config.lockbox_start, tz="UTC")
    decision_index = bars.index + pd.Timedelta(minutes=15)
    if len(decision_index) and decision_index.max() >= lockbox:
        raise ValueError("M15 decisions must remain before the Q2-2026 lockbox")

    engineered = add_features(bars)
    decisions = engineered.reset_index(drop=True)
    decisions.insert(0, "bar_time", bars.index)
    decisions.insert(1, "decision_time", decision_index)
    decisions.insert(
        0,
        "row_key",
        pd.Series(decision_index.strftime("m15-%Y%m%dT%H%M%S%z"), dtype="string"),
    )
    decisions["bar_availability_time"] = decisions["decision_time"]
    decisions["feature_available_time"] = decisions["decision_time"]

    positioning_features = build_positioning_feature_frame(
        _utc_frame(positioning, name="positioning frame")
    )
    decisions = merge_positioning_asof(decisions, positioning_features)
    if (
        decisions["positioning_availability_time"].notna()
        & (
            decisions["positioning_availability_time"]
            > decisions["decision_time"]
        )
    ).any():
        raise AssertionError("future positioning value reached a decision")

    close = _numeric(bars, "close")
    high = _numeric(bars, "high")
    low = _numeric(bars, "low")
    open_price = _numeric(bars, "open")
    volume = _numeric(bars, "volume")
    trade_count = _numeric(bars, "count")
    taker_buy = _numeric(bars, "taker_buy_base")
    log_return = np.log(close.where(close > 0.0)).diff()
    imbalance = 2.0 * taker_buy / volume.replace(0.0, np.nan) - 1.0

    sigma = log_return.rolling(4, min_periods=4).std() * 1e4
    decisions["adaptive_barrier_bps"] = np.clip(
        config.minimum_barrier_bps
        + config.volatility_addon_weight
        * sigma.to_numpy(float)
        * np.sqrt(config.hold_minutes / 15.0),
        config.minimum_barrier_bps,
        config.maximum_barrier_bps,
    )

    for minutes, rows in ((15, 1), (30, 2), (60, 4), (120, 8)):
        decisions[f"past_rv_{minutes}_bps"] = (
            np.sqrt(log_return.pow(2).rolling(rows, min_periods=rows).sum()) * 1e4
        ).to_numpy(float)
        decisions[f"past_range_{minutes}_bps"] = (
            (
                high.rolling(rows, min_periods=rows).max()
                / low.rolling(rows, min_periods=rows).min()
                - 1.0
            )
            * 1e4
        ).to_numpy(float)
    for minutes, rows in ((15, 1), (60, 4)):
        decisions[f"past_abs_return_{minutes}_bps"] = (
            (close / close.shift(rows) - 1.0).abs() * 1e4
        ).to_numpy(float)

    decisions["rv_ratio_15_60"] = _safe_ratio(
        decisions["past_rv_15_bps"], decisions["past_rv_60_bps"]
    )
    decisions["rv_ratio_60_120"] = _safe_ratio(
        decisions["past_rv_60_bps"], decisions["past_rv_120_bps"]
    )
    decisions["range_ratio_15_120"] = _safe_ratio(
        decisions["past_range_15_bps"], decisions["past_range_120_bps"]
    )
    decisions["volume_ratio_15_120"] = _safe_ratio(
        volume.rolling(1, min_periods=1).mean().reset_index(drop=True),
        volume.rolling(8, min_periods=8).mean().reset_index(drop=True),
    )
    decisions["trade_count_ratio_15_120"] = _safe_ratio(
        trade_count.rolling(1, min_periods=1).mean().reset_index(drop=True),
        trade_count.rolling(8, min_periods=8).mean().reset_index(drop=True),
    )
    imbalance_recent = imbalance.rolling(1, min_periods=1).mean()
    imbalance_prior = imbalance.rolling(8, min_periods=8).mean()
    decisions["taker_imbalance"] = imbalance.to_numpy(float)
    decisions["taker_imbalance_delta_15_120"] = (
        imbalance_recent - imbalance_prior
    ).to_numpy(float)
    decisions["taker_imbalance_abs_delta_15_120"] = decisions[
        "taker_imbalance_delta_15_120"
    ].abs()

    decisions["oi_z"] = pd.to_numeric(decisions["oi_z_7d"], errors="coerce")
    decisions["toptrader_ls_z"] = _rolling_zscore(
        pd.to_numeric(decisions["toptrader_log_ratio"], errors="coerce"), 96
    )
    decisions["taker_ls_z"] = _rolling_zscore(
        pd.to_numeric(decisions["taker_log_ratio"], errors="coerce"), 96
    )
    decisions["positioning_age_log"] = np.log1p(
        pd.to_numeric(decisions["positioning_age_min"], errors="coerce").clip(lower=0.0)
    )
    decisions["open_interest_missing"] = decisions["oi_missing"].astype("int8")
    decisions["toptrader_ls_missing"] = decisions["toptrader_missing"].astype("int8")
    decisions["taker_ls_missing"] = decisions["taker_ratio_missing"].astype("int8")
    decisions["price_oi_interaction"] = (
        pd.to_numeric(decisions["r1"], errors="coerce")
        * pd.to_numeric(decisions["oi_chg_15m"], errors="coerce")
    )
    decisions["abs_oi_chg_1h"] = pd.to_numeric(
        decisions["oi_chg_1h"], errors="coerce"
    ).abs()
    decisions["abs_oi_chg_4h"] = pd.to_numeric(
        decisions["oi_chg_4h"], errors="coerce"
    ).abs()
    decisions["abs_funding_z"] = pd.to_numeric(
        decisions["funding_z"], errors="coerce"
    ).abs()

    positive_square = log_return.clip(lower=0.0).pow(2)
    negative_square = log_return.clip(upper=0.0).pow(2)
    for minutes, rows in ((60, 4), (240, 16)):
        decisions[f"upside_semivar_{minutes}"] = positive_square.rolling(
            rows, min_periods=rows
        ).mean().to_numpy(float)
        decisions[f"downside_semivar_{minutes}"] = negative_square.rolling(
            rows, min_periods=rows
        ).mean().to_numpy(float)
        decisions[f"drawdown_{minutes}"] = (
            close / close.rolling(rows, min_periods=rows).max() - 1.0
        ).to_numpy(float)

    channel_low = low.rolling(20, min_periods=20).min()
    channel_high = high.rolling(20, min_periods=20).max()
    decisions["channel_position_20"] = _safe_ratio(
        (close - channel_low).reset_index(drop=True),
        (channel_high - channel_low).reset_index(drop=True),
    )
    decisions["channel_slope_20"] = _rolling_slope(
        np.log(close.where(close > 0.0)), 20
    ).to_numpy(float)
    decisions["body_bps"] = np.log(close / open_price).to_numpy(float) * 1e4
    full_range = (high - low).replace(0.0, np.nan)
    decisions["lower_wick_fraction"] = (
        (np.minimum(open_price, close) - low) / full_range
    ).to_numpy(float)
    decisions["upper_wick_fraction"] = (
        (high - np.maximum(open_price, close)) / full_range
    ).to_numpy(float)
    minute_count = _numeric(bars, "minute_count")
    decisions["m15_incomplete"] = (
        minute_count.ne(15.0) | ~np.isfinite(minute_count)
    ).astype("int8").to_numpy()

    missing_features = sorted(set(UNIFIED_FEATURES).difference(decisions.columns))
    if missing_features:
        raise AssertionError(f"unified feature construction missed: {missing_features}")
    if decisions["row_key"].duplicated().any():
        raise AssertionError("unified decision keys must be unique")
    return decisions.reset_index(drop=True)


def build_causal_sequences(tabular: np.ndarray, sequence_length: int) -> np.ndarray:
    """Return padded past-only windows whose final row is the tabular decision."""
    values = np.asarray(tabular)
    if values.ndim != 2:
        raise ValueError("tabular features must be two-dimensional")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if not len(values):
        return np.empty((0, sequence_length, values.shape[1]), dtype=values.dtype)
    padding = np.repeat(values[:1], sequence_length - 1, axis=0)
    padded = np.concatenate([padding, values], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, (sequence_length, values.shape[1])
    )
    return np.ascontiguousarray(windows[:, 0])


def _entry_details(
    decision_times: pd.Series, minute: pd.DataFrame
) -> tuple[pd.Series, np.ndarray]:
    index = pd.DatetimeIndex(minute.index).as_unit("ns")
    decisions = pd.DatetimeIndex(
        pd.to_datetime(decision_times, utc=True, errors="raise")
    ).as_unit("ns")
    positions = index.searchsorted(decisions, side="right")
    entry_time = pd.Series(pd.NaT, index=decision_times.index, dtype="datetime64[ns, UTC]")
    entry_price = np.full(len(decisions), np.nan, dtype=float)
    valid = positions < len(index)
    if valid.any():
        entry_time.iloc[np.flatnonzero(valid)] = index[positions[valid]]
        opens = pd.to_numeric(minute["open"], errors="coerce").to_numpy(float)
        entry_price[valid] = opens[positions[valid]]
    return entry_time, entry_price


def _path_signature(
    minute_index_ns: np.ndarray,
    minute_ohlc: np.ndarray,
    entry_time: pd.Timestamp | pd.NaT,
    hold_minutes: int,
) -> str | None:
    if pd.isna(entry_time):
        return None
    entry = pd.Timestamp(entry_time)
    entry = entry.tz_localize("UTC") if entry.tzinfo is None else entry.tz_convert("UTC")
    entry_ns = int(entry.as_unit("ns").value)
    position = int(np.searchsorted(minute_index_ns, entry_ns, side="left"))
    end = position + hold_minutes
    if end > len(minute_index_ns) or position >= len(minute_index_ns):
        return None
    timestamps = minute_index_ns[position:end]
    minute_ns = 60 * 1_000_000_000
    if (
        timestamps[0] != entry_ns
        or timestamps[-1] != entry_ns + (hold_minutes - 1) * minute_ns
        or (len(timestamps) > 1 and not np.all(np.diff(timestamps) == minute_ns))
    ):
        return None
    values = minute_ohlc[position:end]
    if not np.isfinite(values).all():
        return None
    digest = hashlib.sha256()
    digest.update(np.asarray(timestamps, dtype="<i8").tobytes())
    digest.update(np.asarray(values, dtype="<f8").tobytes())
    return digest.hexdigest()


def _manual_censored_paths(attempts: pd.DataFrame, config: UnifiedDataConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for attempt in attempts.to_dict(orient="records"):
        for direction in ("long", "short"):
            rows.append(
                {
                    **attempt,
                    "direction": direction,
                    "target_multiple_b": config.target_multiple_b,
                    "hold_minutes": config.hold_minutes,
                    "path_complete": False,
                    "censored": True,
                    "entry_price": attempt.get("reference_price", np.nan),
                    "exit_price": np.nan,
                    "bars_held": np.nan,
                    "outcome": "censored",
                    "gross_bps": np.nan,
                    "net_bps": np.nan,
                    "gross_r": np.nan,
                    "net_r": np.nan,
                    "cost_bps": np.nan,
                    "cost_r": np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_economic_labels(
    decisions: pd.DataFrame,
    minute: pd.DataFrame,
    config: UnifiedDataConfig = UnifiedDataConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay paired LONG/SHORT RR2 paths from the first M1 Open after cutoff."""
    _validate_config(config)
    required = {"row_key", "decision_time", "adaptive_barrier_bps"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"unified decisions missing label inputs: {missing}")
    one = _utc_frame(minute, name="minute frame")
    missing_minute = sorted({"open", "high", "low", "close"}.difference(one.columns))
    if missing_minute:
        raise ValueError(f"minute frame missing columns: {missing_minute}")

    base = decisions[["row_key", "decision_time", "adaptive_barrier_bps"]].copy()
    base["decision_time"] = pd.to_datetime(base["decision_time"], utc=True, errors="raise")
    entry_time, entry_price = _entry_details(base["decision_time"], one)
    base["entry_time"] = entry_time
    base["reference_price"] = entry_price
    base["activation_key"] = base["row_key"]
    base["window_id"] = base["row_key"]
    base["step"] = 0
    base["channel_episode_id"] = base["row_key"]
    base["channel_side"] = "long"
    base["signal_decision_time"] = base["decision_time"]
    base["decision_time"] = base["entry_time"]

    barrier = pd.to_numeric(base["adaptive_barrier_bps"], errors="coerce")
    valid_attempt = (
        base["entry_time"].notna()
        & np.isfinite(base["reference_price"])
        & base["reference_price"].gt(0.0)
        & np.isfinite(barrier)
        & barrier.gt(0.0)
    )
    pieces: list[pd.DataFrame] = []
    if valid_attempt.any():
        pieces.append(
            replay_brackets(
                base.loc[valid_attempt],
                one,
                target_multiples=(config.target_multiple_b,),
                hold_minutes=(config.hold_minutes,),
                entry_cost_bps=config.entry_cost_bps,
                target_exit_cost_bps=config.exit_cost_bps,
                other_exit_cost_bps=config.exit_cost_bps,
            )
        )
    if (~valid_attempt).any():
        pieces.append(_manual_censored_paths(base.loc[~valid_attempt], config))
    paths = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pd.DataFrame()
    if len(paths) != 2 * len(base):
        raise AssertionError("each decision must produce exactly two hypothetical paths")

    paths["entry_time"] = pd.to_datetime(paths["decision_time"], utc=True, errors="coerce")
    paths["decision_time"] = pd.to_datetime(
        paths["signal_decision_time"], utc=True, errors="raise"
    )
    held = pd.to_numeric(paths["bars_held"], errors="coerce")
    paths["actual_exit_time"] = paths["entry_time"] + pd.to_timedelta(
        held - 1.0, unit="min"
    )
    sign = np.where(paths["direction"].eq("long"), 1.0, -1.0)
    paths["gross_return"] = sign * (
        pd.to_numeric(paths["exit_price"], errors="coerce")
        / pd.to_numeric(paths["entry_price"], errors="coerce")
        - 1.0
    )
    paths["net_return"] = paths["gross_return"] - (
        config.entry_cost_bps + config.exit_cost_bps
    ) / 10_000.0
    signature_index_ns = np.asarray(
        pd.DatetimeIndex(one.index).as_unit("ns").asi8, dtype=np.int64
    )
    signature_ohlc = one[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    signature_by_key = {
        row.row_key: _path_signature(
            signature_index_ns,
            signature_ohlc,
            row.entry_time,
            config.hold_minutes,
        )
        for row in base.itertuples(index=False)
    }
    paths["path_signature"] = paths["row_key"].map(signature_by_key)
    signature_complete = paths["path_signature"].notna()
    paths["path_complete"] = paths["path_complete"].astype(bool) & signature_complete
    paths["censored"] = ~paths["path_complete"]

    labels = decisions[["row_key", "decision_time"]].copy()
    entry_by_key = base.set_index("row_key")["entry_time"]
    labels["entry_time"] = labels["row_key"].map(entry_by_key)
    labels["label_end"] = labels["entry_time"] + pd.Timedelta(
        minutes=config.hold_minutes
    )
    complete_paths = paths.loc[
        paths["path_complete"]
        & np.isfinite(pd.to_numeric(paths["net_r"], errors="coerce"))
    ].copy()
    complete_counts = complete_paths.groupby("row_key")["direction"].agg(
        lambda values: set(values)
    )
    complete_keys = complete_counts.loc[
        complete_counts.map(lambda values: values == {"long", "short"})
    ].index
    complete_paths = complete_paths.loc[complete_paths["row_key"].isin(complete_keys)]
    labels["path_complete"] = labels["row_key"].isin(complete_keys)

    if len(complete_paths):
        pair_input = complete_paths[
            [
                "row_key",
                "direction",
                "net_r",
                "target_multiple_b",
                "hold_minutes",
                "cost_bps",
            ]
        ].rename(columns={"row_key": "activation_key"})
        paired = pair_direction_paths(
            pair_input,
            tie_tolerance=0.0,
        )
        paired = paired.rename(columns={"activation_key": "row_key"})
        labels = labels.merge(paired, on="row_key", how="left", validate="one_to_one")
        for metric in (
            "gross_r",
            "gross_return",
            "net_return",
            "outcome",
            "exit_price",
            "actual_exit_time",
        ):
            wide = complete_paths.pivot(index="row_key", columns="direction", values=metric)
            wide.columns = [f"{metric}_{direction}" for direction in wide.columns]
            labels = labels.merge(wide.reset_index(), on="row_key", how="left")
    else:
        for column in (
            "net_r_long",
            "net_r_short",
            "delta_r",
            "best_side",
            "economic_value",
        ):
            labels[column] = np.nan

    complete = labels["path_complete"].astype(bool)
    opportunity = pd.Series(pd.NA, index=labels.index, dtype="Int8")
    opportunity.loc[complete] = (
        labels.loc[complete, ["net_r_long", "net_r_short"]].max(axis=1).gt(0.0)
    ).astype("int8")
    labels["opportunity"] = opportunity
    labels["side"] = "censored"
    labels.loc[complete, "side"] = labels.loc[complete, "best_side"].astype(str)
    labels["side_eligible"] = (
        labels["opportunity"].eq(1).fillna(False) & labels["side"].ne("tie")
    )
    return labels.sort_values("decision_time", kind="stable").reset_index(drop=True), paths.sort_values(
        ["decision_time", "direction"], kind="stable"
    ).reset_index(drop=True)


def make_blocking_fold_manifest(
    decisions: pd.DataFrame,
    labels: pd.DataFrame,
    config: UnifiedDataConfig = UnifiedDataConfig(),
) -> pd.DataFrame:
    """Assign non-overlapping outer blocks and purged fit/calibration/test roles."""
    _validate_config(config)
    decision_required = {"row_key", "decision_time", "adaptive_barrier_bps"}
    label_required = {"row_key", "path_complete", "label_end"}
    missing_decision = sorted(decision_required.difference(decisions.columns))
    missing_label = sorted(label_required.difference(labels.columns))
    if missing_decision or missing_label:
        raise ValueError(
            f"fold inputs missing decisions={missing_decision}, labels={missing_label}"
        )
    work = decisions[list(decision_required)].merge(
        labels[list(label_required)], on="row_key", how="left", validate="one_to_one"
    )
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    work["label_end"] = pd.to_datetime(work["label_end"], utc=True, errors="coerce")
    work = work.sort_values("decision_time", kind="stable").reset_index(drop=True)
    splitter = BlockingTimeSeriesSplit(
        n_splits=config.n_splits,
        train_frac=config.train_fraction,
        embargo=config.embargo_bars,
    )
    block_size = len(work) // config.n_splits
    rows: list[pd.DataFrame] = []
    for fold_id, (outer_train, outer_test) in enumerate(splitter.split(work)):
        block_start = fold_id * block_size
        block_stop = block_start + block_size
        outer_cut = int(outer_train[-1]) + 1
        test_start = int(outer_test[0])
        calibration_rows = max(
            1, int(np.ceil(len(outer_train) * config.calibration_fraction))
        )
        calibration_start = outer_cut - calibration_rows
        inner_embargo_start = max(
            block_start, calibration_start - config.embargo_bars
        )
        block = work.iloc[block_start:block_stop].copy()
        absolute_position = np.arange(block_start, block_stop)
        role = np.full(len(block), "unassigned", dtype=object)
        role[
            (absolute_position >= outer_cut) & (absolute_position < test_start)
        ] = "outer_embargo"
        role[
            (absolute_position >= inner_embargo_start)
            & (absolute_position < calibration_start)
        ] = "inner_embargo"

        barrier = pd.to_numeric(block["adaptive_barrier_bps"], errors="coerce")
        warm = np.isfinite(barrier) & barrier.gt(0.0)
        complete = block["path_complete"].fillna(False).astype(bool)
        label_present = block["label_end"].notna()
        eligible = warm & complete & label_present
        first_calibration_decision = work.iloc[calibration_start]["decision_time"]
        first_test_decision = work.iloc[test_start]["decision_time"]

        fit_candidate = absolute_position < inner_embargo_start
        calibration_candidate = (
            (absolute_position >= calibration_start) & (absolute_position < outer_cut)
        )
        test_candidate = absolute_position >= test_start
        fit_valid = (
            fit_candidate
            & eligible.to_numpy()
            & block["label_end"].lt(first_calibration_decision).to_numpy()
        )
        calibration_valid = (
            calibration_candidate
            & eligible.to_numpy()
            & block["label_end"].lt(first_test_decision).to_numpy()
        )
        test_valid = test_candidate & eligible.to_numpy()
        role[fit_valid] = "fit"
        role[calibration_valid] = "calibration"
        role[test_valid] = "test"

        candidate = fit_candidate | calibration_candidate | test_candidate
        role[candidate & ~warm.to_numpy()] = "warmup"
        role[candidate & warm.to_numpy() & (~complete.to_numpy() | ~label_present.to_numpy())] = (
            "censored"
        )
        purged = (
            (fit_candidate & eligible.to_numpy() & ~block["label_end"].lt(first_calibration_decision).to_numpy())
            | (
                calibration_candidate
                & eligible.to_numpy()
                & ~block["label_end"].lt(first_test_decision).to_numpy()
            )
        )
        role[purged] = "label_purged"
        if (role == "unassigned").any():
            raise AssertionError(f"fold {fold_id} has unassigned rows")

        block["fold_id"] = fold_id
        block["position"] = absolute_position
        block["role"] = role
        block["block_start_position"] = block_start
        block["block_stop_exclusive"] = block_stop
        block["outer_cut_position"] = outer_cut
        block["test_start_position"] = test_start
        block["calibration_start_position"] = calibration_start
        block["inner_embargo_start_position"] = inner_embargo_start
        block["outer_train_fraction"] = config.train_fraction
        rows.append(block)

    manifest = pd.concat(rows, ignore_index=True)
    test_keys = manifest.loc[manifest["role"].eq("test"), "row_key"]
    if test_keys.duplicated().any():
        raise AssertionError("outer test keys overlap across folds")
    for fold_id, fold in manifest.groupby("fold_id", sort=True):
        for role_name in ("fit", "calibration", "test"):
            if not fold["role"].eq(role_name).any():
                raise ValueError(f"fold {fold_id} has no {role_name} rows after purge")
    return manifest


def assert_information_contract(dataset: UnifiedDataset) -> pd.DataFrame:
    """Audit the shared decision keys and information cutoff."""
    decisions = dataset.decisions
    checks = [
        {
            "check": "completed M15 availability",
            "passed": bool(
                (
                    decisions["decision_time"]
                    == decisions["bar_time"] + pd.Timedelta(minutes=15)
                ).all()
            ),
        },
        {
            "check": "positioning availability",
            "passed": bool(
                (
                    decisions["positioning_availability_time"].isna()
                    | (
                        decisions["positioning_availability_time"]
                        <= decisions["decision_time"]
                    )
                ).all()
            ),
        },
        {
            "check": "entry strictly after decision",
            "passed": bool((decisions["entry_time"] > decisions["decision_time"]).all()),
        },
        {
            "check": "LSTM final row matches tabular row",
            "passed": bool(
                np.allclose(
                    dataset.sequences[:, -1, :],
                    dataset.tabular,
                    rtol=0.0,
                    atol=0.0,
                    equal_nan=True,
                )
            ),
        },
        {
            "check": "identical feature contract",
            "passed": tuple(dataset.feature_names) == UNIFIED_FEATURES,
        },
    ]
    audit = pd.DataFrame(checks)
    if not audit["passed"].astype(bool).all():
        failed = audit.loc[~audit["passed"].astype(bool), "check"].tolist()
        raise AssertionError(f"unified information contract failed: {failed}")
    return audit


def build_unified_dataset(
    m15: pd.DataFrame,
    minute: pd.DataFrame,
    positioning: pd.DataFrame,
    config: UnifiedDataConfig = UnifiedDataConfig(),
) -> UnifiedDataset:
    """Build aligned decisions, common economic labels, and model matrices."""
    decisions = build_unified_decisions(m15, positioning, config)
    labels, economic_paths = build_economic_labels(decisions, minute, config)
    decisions = decisions.merge(
        labels.drop(columns="decision_time"), on="row_key", how="left", validate="one_to_one"
    )
    tabular = decisions.loc[:, UNIFIED_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float32)
    sequences = build_causal_sequences(tabular, config.sequence_length)
    dataset = UnifiedDataset(
        decisions=decisions,
        tabular=tabular,
        sequences=sequences,
        feature_names=UNIFIED_FEATURES,
        economic_paths=economic_paths,
    )
    assert_information_contract(dataset)
    return dataset


__all__ = [
    "UNIFIED_FEATURES",
    "UnifiedDataConfig",
    "UnifiedDataset",
    "assert_information_contract",
    "build_causal_sequences",
    "build_economic_labels",
    "build_unified_dataset",
    "build_unified_decisions",
    "make_blocking_fold_manifest",
]
