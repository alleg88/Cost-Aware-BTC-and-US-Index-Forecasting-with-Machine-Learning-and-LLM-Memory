"""Notebook W: final matched-frequency channel-window ablation.

The primary comparison changes only the candidate universe.  One pair of
channel-free opportunity/direction models is trained on all completed 5-minute
decisions.  The channel arm may rank only decisions that belong to the frozen
Notebook J channel windows; the channel-blind arm may rank every decision.
Both arms receive the same timing activation count in every scored half-year
before identical native one-minute economic replay. The reference counts remain
fixed by default; compact Rebuild obtains them from the recomputed U timing ledger.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left, insort
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from experiments.channel_rebuild_contract import recomputed_handoffs
from experiments.event_window_direction_oof import SCORED_FOLDS
from experiments.run_event_window_cost_aware_entry import _Store, _sha256, _sha_payload
from experiments.run_event_window_direction_head import (
    FROZEN_U_ROOT,
    load_frozen_u_artifacts,
)
from experiments.run_event_window_economic_feasibility import replay_brackets
from experiments.run_event_window_tail_models import (
    FROZEN_J_ROOT,
    load_frozen_j_artifacts,
)
from experiments.run_event_window_tcn import _load_bounded_parquet
from features.event_window_inputs import (
    build_positioning_feature_frame,
    merge_positioning_asof,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "channel_vs_volatility_ablation"

FOLD_BOUNDS = {
    "2022H2": ("2022-07-01", "2023-01-01"),
    "2023H1": ("2023-01-01", "2023-07-01"),
    "2023H2": ("2023-07-01", "2024-01-01"),
    "2024H1": ("2024-01-01", "2024-07-01"),
    "2024H2": ("2024-07-01", "2025-01-01"),
    "2025H1": ("2025-01-01", "2025-07-01"),
}
MATCHED_FOLD_COUNTS = {
    "2022H2": 236,
    "2023H1": 541,
    "2023H2": 305,
    "2024H1": 910,
    "2024H2": 532,
    "2025H1": 415,
}
SCORED_CALENDAR_DAYS = 1_096
EXPECTED_MATCHED_ACTIVATIONS = sum(MATCHED_FOLD_COUNTS.values())
TARGET_ACTIVATIONS_PER_DAY = EXPECTED_MATCHED_ACTIVATIONS / SCORED_CALENDAR_DAYS
MODELS = ("logreg", "xgboost")
WINDOW_SOURCES = ("channel", "channel_blind")


def _validated_matched_fold_counts(
    matched_fold_counts: Mapping[str, int] | None,
) -> dict[str, int]:
    """Return ordered, positive per-fold targets without changing the constants."""
    if matched_fold_counts is None:
        return dict(MATCHED_FOLD_COUNTS)
    if not isinstance(matched_fold_counts, Mapping):
        raise TypeError("matched fold counts must be a mapping")
    expected_folds = set(SCORED_FOLDS)
    actual_folds = set(matched_fold_counts)
    if actual_folds != expected_folds:
        missing = sorted(expected_folds.difference(actual_folds))
        unexpected = sorted(actual_folds.difference(expected_folds))
        raise ValueError(
            f"matched fold counts must cover SCORED_FOLDS exactly; "
            f"missing={missing}, unexpected={unexpected}"
        )
    result: dict[str, int] = {}
    for fold_id in SCORED_FOLDS:
        value = matched_fold_counts[fold_id]
        if isinstance(value, bool):
            raise ValueError(f"matched fold count is not a positive integer: {fold_id}")
        try:
            integer = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"matched fold count is not a positive integer: {fold_id}"
            ) from error
        if integer <= 0 or integer != value:
            raise ValueError(f"matched fold count is not a positive integer: {fold_id}")
        result[fold_id] = integer
    return result


def _matched_frequency_source(
    matched_fold_counts: Mapping[str, int] | None,
    target_counts: Mapping[str, int],
) -> str:
    if recomputed_handoffs() and matched_fold_counts is not None:
        return "recomputed_u_scored_timing_ledger"
    if matched_fold_counts is None or dict(target_counts) == MATCHED_FOLD_COUNTS:
        return "canonical_frozen_v_fold_counts"
    return "explicit_matched_fold_counts"


def derive_matched_fold_counts(ledger: pd.DataFrame) -> dict[str, int]:
    """Derive positive scored-fold targets from a validated U timing ledger."""
    if "fold_id" not in ledger.columns:
        raise ValueError("U timing ledger must contain fold_id")
    if ledger["fold_id"].isna().any():
        raise ValueError("U timing ledger contains missing fold_id")
    unexpected = set(ledger["fold_id"]) - set(SCORED_FOLDS) - {"2022H1"}
    if unexpected:
        raise ValueError(f"U timing ledger contains unexpected folds: {sorted(unexpected)}")
    counts = ledger.loc[ledger["fold_id"].isin(SCORED_FOLDS)].groupby("fold_id").size()
    result = {
        fold_id: int(counts.get(fold_id, 0))
        for fold_id in SCORED_FOLDS
    }
    if any(value <= 0 for value in result.values()):
        raise ValueError(f"U timing ledger has an empty scored fold: {result}")
    return result


def effective_matched_fold_counts(ledger: pd.DataFrame) -> dict[str, int]:
    """Use U's runtime fold counts only for an explicit compact-Rebuild opt-in."""
    if recomputed_handoffs():
        return derive_matched_fold_counts(ledger)
    return dict(MATCHED_FOLD_COUNTS)


OPPORTUNITY_FEATURES = (
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
    "positioning_age_log",
)

DIRECTION_FEATURES = (
    "adaptive_barrier_bps",
    "return_5m_bps",
    "past_return_15_bps",
    "past_return_60_bps",
    "body_bps",
    "lower_wick_fraction",
    "upper_wick_fraction",
    "taker_imbalance",
    "taker_imbalance_delta_15_120",
    "oi_chg_15m",
    "oi_chg_1h",
    "oi_chg_4h",
    "oi_accel_1h",
    "funding_z",
    "price_oi_interaction",
    "positioning_stale",
    "positioning_age_log",
)

READER_ARTIFACTS = (
    "oof_predictions.parquet",
    "selected_activations.parquet",
    "economic_paths.parquet",
    "feature_audit.csv",
    "fold_audit.csv",
    "predictive_metrics.csv",
    "opportunity_metrics.csv",
    "economic_metrics.csv",
    "matched_comparisons.csv",
    "frequency_audit.csv",
    "leakage_audit.csv",
    "protocol.json",
    "frozen_protocol.json",
    "summary.json",
)

_SOURCE_DEPENDENCIES = (
    Path(__file__),
    CODE_ROOT / "features" / "event_window_inputs.py",
    CODE_ROOT / "experiments" / "event_window_direction_oof.py",
    CODE_ROOT / "experiments" / "run_event_window_direction_head.py",
    CODE_ROOT / "experiments" / "run_event_window_economic_feasibility.py",
    CODE_ROOT / "experiments" / "run_event_window_tail_models.py",
    CODE_ROOT / "experiments" / "run_event_window_tcn.py",
)


@dataclass(frozen=True)
class ChannelAblationConfig:
    development_start: str = "2021-01-01"
    development_end_exclusive: str = "2025-07-01"
    hold_minutes: int = 120
    target_multiple_b: float = 2.0
    minimum_barrier_bps: float = 75.0
    maximum_barrier_bps: float = 250.0
    volatility_addon_weight: float = 0.5
    cooldown_minutes: int = 60
    entry_cost_bps: float = 5.0
    exit_cost_bps: float = 5.0
    xgb_estimators: int = 300
    xgb_depth: int = 3
    xgb_learning_rate: float = 0.03
    xgb_min_child_weight: float = 20.0
    xgb_reg_lambda: float = 10.0
    logreg_max_iter: int = 2_000
    random_seed: int = 42
    n_jobs: int = 1
    bootstrap_draws: int = 2_000


@dataclass(frozen=True)
class ChannelAblationResult:
    run_dir: Path
    summary: dict[str, object]


def protocol_dict(
    config: ChannelAblationConfig = ChannelAblationConfig(),
    *,
    matched_fold_counts: Mapping[str, int] | None = None,
) -> dict[str, object]:
    target_counts = _validated_matched_fold_counts(matched_fold_counts)
    target_activations = sum(target_counts.values())
    return {
        "study": "notebook_w_channel_vs_volatility_ablation",
        "stage": "dev",
        **asdict(config),
        "models": list(MODELS),
        "window_sources": list(WINDOW_SOURCES),
        "opportunity_features": list(OPPORTUNITY_FEATURES),
        "direction_features": list(DIRECTION_FEATURES),
        "matched_fold_counts": target_counts,
        "matched_total_activations": target_activations,
        "matching_target_source": _matched_frequency_source(
            matched_fold_counts, target_counts
        ),
        "scored_calendar_days": SCORED_CALENDAR_DAYS,
        "target_activations_per_day": target_activations / SCORED_CALENDAR_DAYS,
        "matching_policy": "foldwise label-blind matched top-k with global 60m refractory period",
        "matching_is_online_policy": False,
        "channel_blind_definition": "all completed 5m decisions; no channel feature or gate",
        "channel_definition": "same scores restricted to frozen Notebook J channel decision times",
        "model_training": "one expanding channel-free model per family/head/fold, shared by both window sources",
        "round_trip_cost_bps": config.entry_cost_bps + config.exit_cost_bps,
        "entry": "native one-minute Open at decision time",
        "same_minute_ambiguity": "stop_first",
        "bootstrap_unit": "calendar_week",
        "primary_model": "xgboost",
        "benchmark_model": "logreg",
        "primary_success_rule": (
            "xgboost channel absolute net-R CI low > 0, channel-minus-blind "
            "net-R CI low > 0, and channel-minus-blind opportunity-rate CI low > 0"
        ),
        "forward_or_lockbox_loaded": False,
    }


def channel_retention_decision(
    *,
    channel_ci_low: float,
    channel_minus_blind_ci_low: float,
    opportunity_delta_ci_low: float,
    leakage_passed: bool,
    frequency_matched: bool,
) -> bool:
    """Return the pre-registered strong-evidence gate for retaining channels."""
    return bool(
        leakage_passed
        and frequency_matched
        and channel_ci_low > 0.0
        and channel_minus_blind_ci_low > 0.0
        and opportunity_delta_ci_low > 0.0
    )


def matched_topk_refractory(
    frame: pd.DataFrame,
    *,
    count: int,
    cooldown_minutes: int = 60,
    score_column: str = "score",
) -> pd.DataFrame:
    """Select an exact label-blind top-k subject to a global time refractory period."""
    required = {"decision_time", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"matched top-k rows missing columns: {missing}")
    if count < 0 or cooldown_minutes <= 0:
        raise ValueError("count must be non-negative and cooldown must be positive")
    work = frame.copy().reset_index(drop=True)
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    work[score_column] = pd.to_numeric(work[score_column], errors="raise")
    if work["decision_time"].duplicated().any():
        raise ValueError("matched top-k needs unique decision timestamps")
    if not np.isfinite(work[score_column].to_numpy(float)).all():
        raise ValueError("matched top-k scores must be finite")
    if count == 0:
        return work.iloc[:0].copy()

    ranked = work.sort_values(
        [score_column, "decision_time"],
        ascending=[False, True],
        kind="stable",
    )
    cooldown_ns = int(pd.Timedelta(minutes=cooldown_minutes).value)
    selected_ns: list[int] = []
    selected_rows: list[int] = []
    for row in ranked.itertuples():
        stamp = int(pd.Timestamp(row.decision_time).value)
        position = bisect_left(selected_ns, stamp)
        previous_ok = position == 0 or stamp - selected_ns[position - 1] >= cooldown_ns
        next_ok = position == len(selected_ns) or selected_ns[position] - stamp >= cooldown_ns
        if previous_ok and next_ok:
            insort(selected_ns, stamp)
            selected_rows.append(int(row.Index))
            if len(selected_rows) == count:
                break
    if len(selected_rows) != count:
        raise ValueError(
            f"candidate universe supports only {len(selected_rows)} of {count} "
            f"refractory activations"
        )
    return work.loc[selected_rows].sort_values("decision_time", kind="stable").reset_index(drop=True)


def first_touch_labels(
    decisions: pd.DataFrame,
    minute: pd.DataFrame,
    *,
    horizon_minutes: int = 120,
    chunk_rows: int = 5_000,
) -> pd.DataFrame:
    """Label first symmetric barrier touch on half-open native 1m paths."""
    required = {"decision_time", "reference_price", "adaptive_barrier_bps"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"first-touch decisions missing columns: {missing}")
    if horizon_minutes <= 0 or chunk_rows <= 0:
        raise ValueError("horizon and chunk size must be positive")
    required_minute = {"open", "high", "low", "close"}
    missing_minute = sorted(required_minute.difference(minute.columns))
    if missing_minute:
        raise ValueError(f"minute frame missing columns: {missing_minute}")
    one = minute.copy()
    one.index = pd.to_datetime(one.index, utc=True, errors="raise")
    one = one.sort_index(kind="stable")
    if one.index.has_duplicates:
        raise ValueError("minute timestamps must be unique")
    work = decisions.reset_index(drop=True).copy()
    times = pd.DatetimeIndex(
        pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    ).as_unit("ns")
    references = pd.to_numeric(work["reference_price"], errors="coerce").to_numpy(float)
    barriers = pd.to_numeric(work["adaptive_barrier_bps"], errors="coerce").to_numpy(float)
    values = one[["open", "high", "low", "close"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    positions = one.index.searchsorted(times, side="left")
    path_complete = np.zeros(len(work), dtype=bool)
    labels = np.full(len(work), "censored", dtype=object)
    opportunity = np.full(len(work), -1, dtype=np.int8)
    direction_valid = np.zeros(len(work), dtype=bool)
    direction_up = np.zeros(len(work), dtype=np.int8)
    minute_ns = int(pd.Timedelta(minutes=1).value)
    index_ns = one.index.as_unit("ns").asi8

    for left in range(0, len(work), chunk_rows):
        right = min(left + chunk_rows, len(work))
        local_positions = positions[left:right]
        local_times = times[left:right].asi8
        valid = (
            (local_positions >= 0)
            & (local_positions + horizon_minutes <= len(one))
            & np.isfinite(references[left:right])
            & (references[left:right] > 0.0)
            & np.isfinite(barriers[left:right])
            & (barriers[left:right] > 0.0)
        )
        candidates = np.flatnonzero(valid)
        if len(candidates):
            starts = local_positions[candidates]
            valid[candidates] &= index_ns[starts] == local_times[candidates]
            valid[candidates] &= (
                index_ns[starts + horizon_minutes - 1]
                == local_times[candidates] + (horizon_minutes - 1) * minute_ns
            )
        candidates = np.flatnonzero(valid)
        if not len(candidates):
            continue
        starts = local_positions[candidates]
        offsets = np.arange(horizon_minutes, dtype=np.int64)
        path_positions = starts[:, None] + offsets[None, :]
        paths = values[path_positions]
        finite = np.isfinite(paths).all(axis=(1, 2))
        candidates = candidates[finite]
        paths = paths[finite]
        if not len(candidates):
            continue
        global_rows = left + candidates
        upper = references[global_rows] * np.exp(barriers[global_rows] / 1e4)
        lower = references[global_rows] * np.exp(-barriers[global_rows] / 1e4)
        up = paths[:, :, 1] >= upper[:, None]
        down = paths[:, :, 2] <= lower[:, None]
        up_any = up.any(axis=1)
        down_any = down.any(axis=1)
        first_up = np.where(up_any, up.argmax(axis=1), horizon_minutes + 1)
        first_down = np.where(down_any, down.argmax(axis=1), horizon_minutes + 1)
        ambiguous = up_any & down_any & (first_up == first_down)
        up_first = up_any & (first_up < first_down)
        down_first = down_any & (first_down < first_up)
        no_hit = ~up_any & ~down_any
        path_complete[global_rows] = True
        labels[global_rows[up_first]] = "up_big"
        labels[global_rows[down_first]] = "down_big"
        labels[global_rows[ambiguous]] = "ambiguous"
        labels[global_rows[no_hit]] = "no_big_move"
        opportunity[global_rows] = (~no_hit).astype(np.int8)
        direction_valid[global_rows] = up_first | down_first
        direction_up[global_rows[up_first]] = 1

    output = work.copy()
    output["move_label"] = labels
    output["opportunity"] = opportunity
    output["direction_valid"] = direction_valid
    output["direction_up"] = direction_up
    output["path_complete"] = path_complete
    output["label_end"] = times + pd.Timedelta(minutes=horizon_minutes)
    return output


def _sample_at(series: pd.Series, timestamps: pd.DatetimeIndex) -> np.ndarray:
    return pd.to_numeric(series.reindex(timestamps), errors="coerce").to_numpy(float)


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=np.isfinite(denominator) & (denominator > 0.0),
    )


def build_channel_free_decisions(
    minute: pd.DataFrame,
    five_minute: pd.DataFrame,
    positioning: pd.DataFrame,
    *,
    channel_times: pd.DatetimeIndex,
    config: ChannelAblationConfig = ChannelAblationConfig(),
) -> pd.DataFrame:
    """Build the common causal five-minute row universe and both targets."""
    one = minute.copy()
    one.index = pd.to_datetime(one.index, utc=True, errors="raise")
    one = one.sort_index(kind="stable")
    five = five_minute.copy()
    five.index = pd.to_datetime(five.index, utc=True, errors="raise")
    five = five.sort_index(kind="stable")
    development_start = pd.Timestamp(config.development_start, tz="UTC")
    development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
    decision_index = five.index + pd.Timedelta(minutes=5)
    five = five.loc[
        (decision_index >= development_start) & (decision_index < development_end)
    ].copy()
    if one.index.has_duplicates or five.index.has_duplicates:
        raise ValueError("raw timestamp indices must be unique")
    for name, frame in (("minute", one), ("five_minute", five)):
        missing = sorted({"open", "high", "low", "close"}.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} frame missing OHLC: {missing}")

    decision_times = pd.DatetimeIndex(five.index + pd.Timedelta(minutes=5))
    source_close_times = decision_times - pd.Timedelta(minutes=1)
    decisions = pd.DataFrame(
        {
            "source_bar_time": five.index,
            "decision_time": decision_times,
        }
    )
    minute_open = pd.to_numeric(one["open"], errors="coerce")
    decisions["reference_price"] = _sample_at(minute_open, decision_times)

    close = pd.to_numeric(one["close"], errors="coerce")
    high = pd.to_numeric(one["high"], errors="coerce")
    low = pd.to_numeric(one["low"], errors="coerce")
    log_return = np.log(close.where(close > 0.0)).diff()
    for horizon in (15, 30, 60, 120):
        rv = np.sqrt(log_return.pow(2).rolling(horizon, min_periods=horizon).sum()) * 1e4
        price_range = (
            high.rolling(horizon, min_periods=horizon).max()
            / low.rolling(horizon, min_periods=horizon).min()
            - 1.0
        ) * 1e4
        decisions[f"past_rv_{horizon}_bps"] = _sample_at(rv, source_close_times)
        decisions[f"past_range_{horizon}_bps"] = _sample_at(
            price_range, source_close_times
        )
    for horizon in (15, 60):
        past_return = (close / close.shift(horizon) - 1.0) * 1e4
        decisions[f"past_return_{horizon}_bps"] = _sample_at(
            past_return, source_close_times
        )
        decisions[f"past_abs_return_{horizon}_bps"] = np.abs(
            decisions[f"past_return_{horizon}_bps"]
        )
    decisions["rv_ratio_15_60"] = _safe_ratio(
        decisions["past_rv_15_bps"].to_numpy(float),
        decisions["past_rv_60_bps"].to_numpy(float),
    )
    decisions["rv_ratio_60_120"] = _safe_ratio(
        decisions["past_rv_60_bps"].to_numpy(float),
        decisions["past_rv_120_bps"].to_numpy(float),
    )
    decisions["range_ratio_15_120"] = _safe_ratio(
        decisions["past_range_15_bps"].to_numpy(float),
        decisions["past_range_120_bps"].to_numpy(float),
    )

    five_close = pd.to_numeric(five["close"], errors="coerce")
    sigma = np.log(five_close.where(five_close > 0.0)).diff().rolling(
        12, min_periods=12
    ).std() * 1e4
    sigma_values = sigma.to_numpy(float)
    decisions["past_sigma_5m_bps"] = sigma_values
    decisions["adaptive_barrier_bps"] = np.clip(
        config.minimum_barrier_bps
        + config.volatility_addon_weight
        * sigma_values
        * np.sqrt(config.hold_minutes / 5.0),
        config.minimum_barrier_bps,
        config.maximum_barrier_bps,
    )

    volume = pd.to_numeric(one.get("volume"), errors="coerce")
    count_name = "trade_count" if "trade_count" in one else "count"
    trade_count = pd.to_numeric(one.get(count_name), errors="coerce")
    for values, output in (
        (volume, "volume_ratio_15_120"),
        (trade_count, "trade_count_ratio_15_120"),
    ):
        recent = values.rolling(15, min_periods=15).mean()
        prior = values.rolling(120, min_periods=120).mean()
        decisions[output] = _safe_ratio(
            _sample_at(recent, source_close_times),
            _sample_at(prior, source_close_times),
        )
    taker_buy = pd.to_numeric(one.get("taker_buy_base"), errors="coerce")
    imbalance = (2.0 * taker_buy - volume) / volume.where(volume > 0.0)
    imbalance_15 = imbalance.rolling(15, min_periods=15).mean()
    imbalance_120 = imbalance.rolling(120, min_periods=120).mean()
    imbalance_delta = _sample_at(imbalance_15 - imbalance_120, source_close_times)
    decisions["taker_imbalance_delta_15_120"] = imbalance_delta
    decisions["taker_imbalance_abs_delta_15_120"] = np.abs(imbalance_delta)

    five_open = pd.to_numeric(five["open"], errors="coerce")
    five_high = pd.to_numeric(five["high"], errors="coerce")
    five_low = pd.to_numeric(five["low"], errors="coerce")
    five_range = (five_high - five_low).where((five_high - five_low) > 0.0)
    decisions["return_5m_bps"] = (
        np.log(five_close.where(five_close > 0.0)).diff() * 1e4
    ).to_numpy(float)
    decisions["body_bps"] = ((five_close - five_open) / five_open * 1e4).to_numpy(float)
    decisions["lower_wick_fraction"] = (
        (np.minimum(five_open, five_close) - five_low) / five_range
    ).to_numpy(float)
    decisions["upper_wick_fraction"] = (
        (five_high - np.maximum(five_open, five_close)) / five_range
    ).to_numpy(float)
    five_volume = pd.to_numeric(five.get("volume"), errors="coerce")
    five_taker = pd.to_numeric(five.get("taker_buy_base"), errors="coerce")
    decisions["taker_imbalance"] = (
        (2.0 * five_taker - five_volume) / five_volume.where(five_volume > 0.0)
    ).to_numpy(float)

    positioning_features = build_positioning_feature_frame(positioning)
    decisions = merge_positioning_asof(decisions, positioning_features)
    decisions["abs_oi_chg_1h"] = np.abs(decisions["oi_chg_1h"])
    decisions["abs_oi_chg_4h"] = np.abs(decisions["oi_chg_4h"])
    decisions["abs_funding_z"] = np.abs(decisions["funding_z"])
    decisions["positioning_age_log"] = np.log1p(
        pd.to_numeric(decisions["positioning_age_min"], errors="coerce").clip(lower=0.0)
    )
    decisions["price_oi_interaction"] = (
        decisions["past_return_15_bps"] * decisions["oi_chg_1h"]
    )
    decisions["in_channel_window"] = pd.DatetimeIndex(
        decisions["decision_time"]
    ).isin(pd.DatetimeIndex(channel_times))

    labels = first_touch_labels(
        decisions[["decision_time", "reference_price", "adaptive_barrier_bps"]],
        one,
        horizon_minutes=config.hold_minutes,
    )
    for column in (
        "move_label",
        "opportunity",
        "direction_valid",
        "direction_up",
        "path_complete",
        "label_end",
    ):
        decisions[column] = labels[column].to_numpy()
    decisions["row_key"] = decisions["decision_time"].astype(str)
    decisions["fold_id"] = ""
    for fold_id, (start, end) in FOLD_BOUNDS.items():
        mask = decisions["decision_time"].ge(pd.Timestamp(start, tz="UTC")) & decisions[
            "decision_time"
        ].lt(pd.Timestamp(end, tz="UTC"))
        decisions.loc[mask, "fold_id"] = fold_id
    return decisions.reset_index(drop=True)


def _interval_uniqueness(
    decision_times: pd.Series,
    *,
    horizon_minutes: int,
) -> np.ndarray:
    """Mean reciprocal concurrency on a dense minute axis, normalised to mean one."""
    times = pd.DatetimeIndex(
        pd.to_datetime(decision_times, utc=True, errors="raise")
    ).as_unit("ns")
    if not len(times):
        return np.empty(0, dtype=float)
    minute_ns = int(pd.Timedelta(minutes=1).value)
    starts = times.asi8 // minute_ns
    origin = int(starts.min())
    relative = (starts - origin).astype(np.int64)
    length = int(relative.max() + horizon_minutes + 1)
    delta = np.zeros(length + 1, dtype=np.int32)
    np.add.at(delta, relative, 1)
    np.add.at(delta, relative + horizon_minutes, -1)
    concurrency = np.cumsum(delta[:-1])
    reciprocal = np.divide(
        1.0,
        concurrency,
        out=np.zeros_like(concurrency, dtype=float),
        where=concurrency > 0,
    )
    prefix = np.concatenate(([0.0], np.cumsum(reciprocal)))
    weights = (
        prefix[relative + horizon_minutes] - prefix[relative]
    ) / horizon_minutes
    weights /= weights.mean()
    return weights


def _fit_predict_head(
    model_name: str,
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_weight: np.ndarray,
    score_x: np.ndarray,
    *,
    config: ChannelAblationConfig,
) -> np.ndarray:
    if model_name not in MODELS:
        raise ValueError(f"unknown model: {model_name}")
    if set(np.unique(train_y)) != {0, 1}:
        raise ValueError("binary head training requires both classes")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train_x)
    predict_x = imputer.transform(score_x)
    if model_name == "logreg":
        scaler = StandardScaler()
        fit_x = scaler.fit_transform(fit_x)
        predict_x = scaler.transform(predict_x)
        model = LogisticRegression(
            solver="lbfgs",
            max_iter=config.logreg_max_iter,
            random_state=config.random_seed,
        )
    else:
        from xgboost import XGBClassifier

        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=config.xgb_estimators,
            max_depth=config.xgb_depth,
            learning_rate=config.xgb_learning_rate,
            min_child_weight=config.xgb_min_child_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=config.xgb_reg_lambda,
            tree_method="hist",
            random_state=config.random_seed,
            n_jobs=config.n_jobs,
            verbosity=0,
        )
    model.fit(fit_x, train_y, sample_weight=train_weight)
    return np.asarray(model.predict_proba(predict_x), dtype=float)[:, 1]


def run_expanding_oof(
    decisions: pd.DataFrame,
    *,
    config: ChannelAblationConfig = ChannelAblationConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit shared channel-free opportunity/direction heads on strictly prior rows."""
    opportunity_x = decisions.loc[:, OPPORTUNITY_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float32)
    direction_x = decisions.loc[:, DIRECTION_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(np.float32)
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for fold_id in SCORED_FOLDS:
        start = pd.Timestamp(FOLD_BOUNDS[fold_id][0], tz="UTC")
        validation_mask = decisions["fold_id"].eq(fold_id).to_numpy()
        validation_positions = np.flatnonzero(validation_mask)
        train_mask = decisions["path_complete"].astype(bool)
        train_mask &= decisions["label_end"].le(start)
        train_positions = np.flatnonzero(train_mask.to_numpy())
        direction_positions = train_positions[
            decisions.iloc[train_positions]["direction_valid"].to_numpy(bool)
        ]
        if not len(train_positions) or not len(validation_positions) or not len(direction_positions):
            raise ValueError(f"fold {fold_id} lacks purged train/validation rows")
        opportunity_weight = _interval_uniqueness(
            decisions.iloc[train_positions]["decision_time"],
            horizon_minutes=config.hold_minutes,
        )
        direction_weight = _interval_uniqueness(
            decisions.iloc[direction_positions]["decision_time"],
            horizon_minutes=config.hold_minutes,
        )
        for model_name in MODELS:
            opportunity_score = _fit_predict_head(
                model_name,
                opportunity_x[train_positions],
                decisions.iloc[train_positions]["opportunity"].to_numpy(np.int8),
                opportunity_weight,
                opportunity_x[validation_positions],
                config=config,
            )
            direction_score = _fit_predict_head(
                model_name,
                direction_x[direction_positions],
                decisions.iloc[direction_positions]["direction_up"].to_numpy(np.int8),
                direction_weight,
                direction_x[validation_positions],
                config=config,
            )
            frame = decisions.iloc[validation_positions][
                [
                    "row_key",
                    "fold_id",
                    "decision_time",
                    "reference_price",
                    "adaptive_barrier_bps",
                    "in_channel_window",
                    "move_label",
                    "opportunity",
                    "direction_valid",
                    "direction_up",
                    "path_complete",
                ]
            ].copy()
            frame.insert(0, "model", model_name)
            frame["opportunity_score"] = opportunity_score
            frame["direction_score"] = direction_score
            frame["chosen_direction"] = np.where(
                direction_score >= 0.5, "long", "short"
            )
            predictions.append(frame)
            audits.append(
                {
                    "model": model_name,
                    "fold_id": fold_id,
                    "train_rows": len(train_positions),
                    "direction_train_rows": len(direction_positions),
                    "validation_rows": len(validation_positions),
                    "validation_start": start,
                    "train_label_end_max": decisions.iloc[train_positions][
                        "label_end"
                    ].max(),
                    "opportunity_uniqueness_mean": float(opportunity_weight.mean()),
                    "direction_uniqueness_mean": float(direction_weight.mean()),
                    "shared_between_window_sources": True,
                }
            )
    result = pd.concat(predictions, ignore_index=True)
    if result.duplicated(["model", "decision_time"]).any():
        raise AssertionError("OOF predictions duplicated model-time rows")
    return result, pd.DataFrame(audits)


def select_matched_activations(
    predictions: pd.DataFrame,
    *,
    config: ChannelAblationConfig = ChannelAblationConfig(),
    matched_fold_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Apply the selected per-fold targets to both universes and model families."""
    target_counts = _validated_matched_fold_counts(matched_fold_counts)
    target_activations = sum(target_counts.values())
    rows: list[pd.DataFrame] = []
    for model_name in MODELS:
        model_rows = predictions.loc[predictions["model"].eq(model_name)]
        for fold_id in SCORED_FOLDS:
            target = target_counts[fold_id]
            fold_rows = model_rows.loc[
                model_rows["fold_id"].eq(fold_id)
                & model_rows["path_complete"].astype(bool)
                & np.isfinite(model_rows["reference_price"])
                & np.isfinite(model_rows["adaptive_barrier_bps"])
            ].copy()
            for source in WINDOW_SOURCES:
                candidates = (
                    fold_rows.loc[fold_rows["in_channel_window"].astype(bool)]
                    if source == "channel"
                    else fold_rows
                )
                selected = matched_topk_refractory(
                    candidates,
                    count=target,
                    cooldown_minutes=config.cooldown_minutes,
                    score_column="opportunity_score",
                )
                selected.insert(1, "window_source", source)
                selected["candidate_rows"] = len(candidates)
                selected["matched_fold_count"] = target
                selected["activation_key"] = (
                    model_name
                    + "::"
                    + source
                    + "::"
                    + selected["decision_time"].astype(str)
                )
                selected["calendar_week"] = selected["decision_time"].dt.strftime(
                    "%G-W%V"
                )
                rows.append(selected)
    result = pd.concat(rows, ignore_index=True)
    if result.duplicated(["model", "window_source", "decision_time"]).any():
        raise AssertionError("matched activation ledger contains duplicate times")
    counts = result.groupby(["model", "window_source"]).size()
    if not counts.eq(target_activations).all():
        raise AssertionError(f"matched activation totals changed: {counts.to_dict()}")
    return result


def _predictive_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model_name, group in predictions.groupby("model", sort=False):
        valid = group.loc[group["path_complete"].astype(bool)].copy()
        y = valid["opportunity"].to_numpy(np.int8)
        score = np.clip(valid["opportunity_score"].to_numpy(float), 1e-8, 1 - 1e-8)
        direction = valid.loc[valid["direction_valid"].astype(bool)]
        direction_accuracy = float(
            (
                (direction["direction_score"].to_numpy(float) >= 0.5)
                == direction["direction_up"].to_numpy(bool)
            ).mean()
        )
        rows.append(
            {
                "model": model_name,
                "rows": len(valid),
                "positive_rate": float(y.mean()),
                "roc_auc": float(roc_auc_score(y, score)),
                "pr_auc": float(average_precision_score(y, score)),
                "brier": float(brier_score_loss(y, score)),
                "log_loss": float(log_loss(y, score)),
                "direction_rows": len(direction),
                "direction_accuracy_on_resolved_hits": direction_accuracy,
            }
        )
    return pd.DataFrame(rows)


def _weekly_mean_interval(
    frame: pd.DataFrame,
    *,
    value_column: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    grouped = frame.groupby("calendar_week")[value_column].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan, np.nan
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    point = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(grouped), size=(draws, len(grouped)))
    values = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def _weekly_difference_interval(
    channel: pd.DataFrame,
    blind: pd.DataFrame,
    *,
    value_column: str,
    draws: int,
    seed: int,
) -> tuple[float, float, float]:
    weeks = pd.Index(
        sorted(set(channel["calendar_week"]) | set(blind["calendar_week"]))
    )
    left = channel.groupby("calendar_week")[value_column].agg(["sum", "count"])
    right = blind.groupby("calendar_week")[value_column].agg(["sum", "count"])
    left_sum = left["sum"].reindex(weeks, fill_value=0.0).to_numpy(float)
    left_count = left["count"].reindex(weeks, fill_value=0.0).to_numpy(float)
    right_sum = right["sum"].reindex(weeks, fill_value=0.0).to_numpy(float)
    right_count = right["count"].reindex(weeks, fill_value=0.0).to_numpy(float)
    point = float(channel[value_column].mean() - blind[value_column].mean())
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(weeks), size=(draws, len(weeks)))
    left_denominator = left_count[sampled].sum(axis=1)
    right_denominator = right_count[sampled].sum(axis=1)
    usable = (left_denominator > 0.0) & (right_denominator > 0.0)
    values = (
        left_sum[sampled].sum(axis=1)[usable] / left_denominator[usable]
        - right_sum[sampled].sum(axis=1)[usable] / right_denominator[usable]
    )
    low, high = np.quantile(values, [0.025, 0.975])
    return point, float(low), float(high)


def _opportunity_tables(
    selected: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for position, ((model_name, source), group) in enumerate(
        selected.groupby(["model", "window_source"], sort=False)
    ):
        point, low, high = _weekly_mean_interval(
            group,
            value_column="opportunity",
            draws=draws,
            seed=seed + position,
        )
        direction = group.loc[group["direction_valid"].astype(bool)]
        metric_rows.append(
            {
                "model": model_name,
                "window_source": source,
                "activations": len(group),
                "opportunity_rate": point,
                "opportunity_rate_ci_low": low,
                "opportunity_rate_ci_high": high,
                "mean_opportunity_score": float(group["opportunity_score"].mean()),
                "direction_accuracy_on_resolved_hits": (
                    float(
                        (
                            (direction["direction_score"].to_numpy(float) >= 0.5)
                            == direction["direction_up"].to_numpy(bool)
                        ).mean()
                    )
                    if len(direction)
                    else np.nan
                ),
                "resolved_direction_rows": len(direction),
                "mean_barrier_bps": float(group["adaptive_barrier_bps"].mean()),
                "selected_inside_channel_fraction": float(
                    group["in_channel_window"].astype(bool).mean()
                ),
            }
        )
    for position, model_name in enumerate(MODELS, start=30):
        channel = selected.loc[
            selected["model"].eq(model_name)
            & selected["window_source"].eq("channel")
        ]
        blind = selected.loc[
            selected["model"].eq(model_name)
            & selected["window_source"].eq("channel_blind")
        ]
        point, low, high = _weekly_difference_interval(
            channel,
            blind,
            value_column="opportunity",
            draws=draws,
            seed=seed + position,
        )
        comparison_rows.append(
            {
                "metric": "opportunity_rate",
                "model": model_name,
                "candidate": "channel",
                "baseline": "channel_blind",
                "point_delta": point,
                "ci_low": low,
                "ci_high": high,
                "draws": draws,
                "bootstrap_unit": "calendar_week",
            }
        )
    return pd.DataFrame(metric_rows), pd.DataFrame(comparison_rows)


def _replay_selected(
    selected: pd.DataFrame,
    minute: pd.DataFrame,
    *,
    config: ChannelAblationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    attempts = selected.copy()
    attempts["window_id"] = attempts["activation_key"]
    attempts["step"] = 0
    attempts["channel_episode_id"] = attempts["calendar_week"]
    attempts["channel_side"] = attempts["chosen_direction"]
    paths = replay_brackets(
        attempts,
        minute,
        target_multiples=(config.target_multiple_b,),
        hold_minutes=(config.hold_minutes,),
        entry_cost_bps=config.entry_cost_bps,
        target_exit_cost_bps=config.exit_cost_bps,
        other_exit_cost_bps=config.exit_cost_bps,
    )
    chosen = paths.loc[paths["direction"].eq(paths["chosen_direction"])].copy()
    if len(chosen) != len(selected) or not chosen["path_complete"].astype(bool).all():
        raise AssertionError("native replay did not preserve every selected activation")
    if set(chosen["cost_bps"]) != {config.entry_cost_bps + config.exit_cost_bps}:
        raise AssertionError("economic replay did not apply uniform 5+5 costs")
    return paths, chosen


def _economic_tables(
    chosen: pd.DataFrame,
    comparisons: pd.DataFrame,
    *,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    comparison_rows = [comparisons]
    for position, ((model_name, source), group) in enumerate(
        chosen.groupby(["model", "window_source"], sort=False)
    ):
        point, low, high = _weekly_mean_interval(
            group,
            value_column="net_r",
            draws=draws,
            seed=seed + 100 + position,
        )
        rows.append(
            {
                "model": model_name,
                "window_source": source,
                "activations": len(group),
                "mean_gross_r": float(group["gross_r"].mean()),
                "total_gross_r": float(group["gross_r"].sum()),
                "mean_net_r": point,
                "total_net_r": float(group["net_r"].sum()),
                "mean_net_r_ci_low": low,
                "mean_net_r_ci_high": high,
                "mean_net_bps": float(group["net_bps"].mean()),
                "tp_fraction": float(group["outcome"].eq("tp").mean()),
                "sl_fraction": float(group["outcome"].eq("sl").mean()),
                "timeout_fraction": float(group["outcome"].eq("timeout").mean()),
                "path_completeness": float(group["path_complete"].astype(bool).mean()),
            }
        )
    economic = pd.DataFrame(rows)
    net_comparisons: list[dict[str, object]] = []
    for position, model_name in enumerate(MODELS, start=60):
        channel = chosen.loc[
            chosen["model"].eq(model_name)
            & chosen["window_source"].eq("channel")
        ]
        blind = chosen.loc[
            chosen["model"].eq(model_name)
            & chosen["window_source"].eq("channel_blind")
        ]
        point, low, high = _weekly_difference_interval(
            channel,
            blind,
            value_column="net_r",
            draws=draws,
            seed=seed + position,
        )
        net_comparisons.append(
            {
                "metric": "mean_net_r",
                "model": model_name,
                "candidate": "channel",
                "baseline": "channel_blind",
                "point_delta": point,
                "ci_low": low,
                "ci_high": high,
                "draws": draws,
                "bootstrap_unit": "calendar_week",
            }
        )
    comparison_rows.append(pd.DataFrame(net_comparisons))
    return economic, pd.concat(comparison_rows, ignore_index=True)


def _frequency_audit(
    selected: pd.DataFrame,
    *,
    matched_fold_counts: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    target_counts = _validated_matched_fold_counts(matched_fold_counts)
    target_activations = sum(target_counts.values())
    target_source = _matched_frequency_source(matched_fold_counts, target_counts)
    rows: list[dict[str, object]] = []
    for (model_name, source), group in selected.groupby(
        ["model", "window_source"], sort=False
    ):
        ordered = group.sort_values("decision_time")
        minimum_gap = ordered["decision_time"].diff().dropna().min()
        observed_series = (
            ordered.groupby("fold_id")
            .size()
            .reindex(SCORED_FOLDS, fill_value=0)
            .astype(int)
        )
        observed_counts = {
            fold_id: int(observed_series[fold_id]) for fold_id in SCORED_FOLDS
        }
        rows.append(
            {
                "model": model_name,
                "window_source": source,
                "activations": len(group),
                "matched_total_activations": target_activations,
                "calendar_days": SCORED_CALENDAR_DAYS,
                "activations_per_day": len(group) / SCORED_CALENDAR_DAYS,
                "target_activations_per_day": target_activations / SCORED_CALENDAR_DAYS,
                "matching_target_source": target_source,
                "matched_fold_counts": json.dumps(
                    target_counts, separators=(",", ":")
                ),
                "observed_fold_counts": json.dumps(
                    observed_counts, separators=(",", ":")
                ),
                "fold_counts_exact": observed_counts == target_counts,
                "minimum_global_gap_minutes": (
                    float(minimum_gap.total_seconds() / 60.0)
                    if minimum_gap is not pd.NaT
                    else np.nan
                ),
                "global_60m_refractory_passed": bool(
                    minimum_gap is pd.NaT
                    or minimum_gap >= pd.Timedelta(minutes=60)
                ),
                "matching_used_outcome_labels": False,
            }
        )
    return pd.DataFrame(rows)


def _feature_audit() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for head, features in (
        ("opportunity", OPPORTUNITY_FEATURES),
        ("direction", DIRECTION_FEATURES),
    ):
        for position, feature in enumerate(features):
            rows.append(
                {
                    "head": head,
                    "position": position,
                    "feature": feature,
                    "channel_feature": "channel" in feature.lower(),
                    "future_feature": any(
                        token in feature.lower()
                        for token in ("future", "outcome", "label", "target", "net_r")
                    ),
                    "known_at_decision_time": True,
                }
            )
    return pd.DataFrame(rows)


def _source_hash() -> str:
    digest = hashlib.sha256()
    for path in _SOURCE_DEPENDENCIES:
        digest.update(str(path.relative_to(CODE_ROOT)).encode("utf-8"))
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
        digest.update(text.encode("utf-8"))
    return digest.hexdigest()


def _bounded_identity(
    frames: dict[str, pd.DataFrame],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    from experiments.run_event_window_direction_head import (
        _canonical_bounded_frame_sha256,
    )

    fingerprints: dict[str, dict[str, object]] = {}
    for role, frame in frames.items():
        maximum = frame.index.max() if len(frame) else None
        fingerprints[role] = {
            "content_sha256": _canonical_bounded_frame_sha256(
                frame,
                source_role=role,
                start=start,
                end=end,
            ),
            "rows": len(frame),
            "max_timestamp": maximum.isoformat() if maximum is not None else None,
        }
    payload = {
        "method": "pyarrow_filtered_canonical_ipc_sha256_v1",
        "development_start": start.isoformat(),
        "development_end_exclusive": end.isoformat(),
        "source_fingerprints": fingerprints,
    }
    return {**payload, "aggregate_sha256": _sha_payload(payload)}


def _load_inputs(
    data_root: Path,
    *,
    config: ChannelAblationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    start = pd.Timestamp(config.development_start, tz="UTC")
    end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
    paths = {
        "minute": Path(data_root) / "btcusdt_1m_2021_2026.parquet",
        "five_minute": Path(data_root) / "btcusdt_5min_2021_2026.parquet",
        "positioning": Path(data_root) / "btcusdt_positioning_15min_2021_2026.parquet",
    }
    frames = {
        role: _load_bounded_parquet(path, start=start, end=end)
        for role, path in paths.items()
    }
    identity = _bounded_identity(frames, start=start, end=end)
    return (
        frames["minute"],
        frames["five_minute"],
        frames["positioning"],
        identity,
    )


def _validated_cached_summary(
    run_dir: Path,
    identity: dict[str, str],
) -> dict[str, object] | None:
    try:
        state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "complete" or any(
            state.get(name) != value for name, value in identity.items()
        ):
            return None
        records = state.get("artifacts", {})
        for name in READER_ARTIFACTS:
            path = run_dir / name
            record = records.get(name, {})
            if (
                not path.is_file()
                or int(record.get("size", -1)) != path.stat().st_size
                or str(record.get("sha256", "")) != _sha256(path)
            ):
                return None
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        if state.get("summary") != summary:
            return None
        return summary
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _publish_latest(run_root: Path, identity: dict[str, str]) -> None:
    path = Path(run_root) / "latest_dev.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            {
                "run_hash": identity["run_hash"],
                "protocol_hash": identity["protocol_hash"],
                "relative_path": f"{identity['run_hash']}/full",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def run_channel_vs_volatility_ablation(
    *,
    stage: str = "dev",
    data_root: Path = DEFAULT_DATA_ROOT,
    run_root: Path = RUN_ROOT,
    config: ChannelAblationConfig = ChannelAblationConfig(),
) -> ChannelAblationResult:
    """Run the single registered full-development Notebook W experiment."""
    if stage != "dev":
        raise ValueError("Notebook W permits development only; forward and Q2 are sealed")
    if config != ChannelAblationConfig():
        raise ValueError("Notebook W complete registered configuration is frozen")
    frozen_u = load_frozen_u_artifacts(FROZEN_U_ROOT)
    frozen_j = load_frozen_j_artifacts(FROZEN_J_ROOT)
    scored_u = frozen_u.ledger.loc[frozen_u.ledger["fold_id"].isin(SCORED_FOLDS)]
    frozen_counts = derive_matched_fold_counts(scored_u)
    matched_fold_counts = effective_matched_fold_counts(scored_u)
    if not recomputed_handoffs() and frozen_counts != MATCHED_FOLD_COUNTS:
        raise AssertionError(f"frozen V fold counts changed: {frozen_counts}")
    target_activations = sum(matched_fold_counts.values())
    target_activations_per_day = target_activations / SCORED_CALENDAR_DAYS
    protocol = protocol_dict(
        config,
        matched_fold_counts=matched_fold_counts,
    )
    protocol["frozen_u_scored_fold_counts"] = frozen_counts

    minute, five, positioning, bounded_identity = _load_inputs(
        Path(data_root), config=config
    )
    protocol["bounded_development_input_identity"] = bounded_identity
    input_payload = {
        "bounded_development_input_identity": bounded_identity,
        "frozen_u_run_hash": frozen_u.run_hash,
        "frozen_u_manifest_sha256": frozen_u.manifest_sha256,
        "frozen_u_activation_ledger_sha256": frozen_u.activation_ledger_sha256,
        "frozen_j_run_hash": frozen_j.run_hash,
        "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
        "frozen_j_labels_rr2_sha256": _sha256(frozen_j.run_dir / "labels_rr2.parquet"),
    }
    protocol_hash = _sha_payload(protocol)
    source_hash = _source_hash()
    input_hash = _sha_payload(input_payload)
    identity = {
        "run_hash": _sha_payload(
            {
                "protocol_hash": protocol_hash,
                "source_hash": source_hash,
                "input_hash": input_hash,
            }
        )[:20],
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
    }
    run_dir = Path(run_root) / identity["run_hash"] / "full"
    cached = _validated_cached_summary(run_dir, identity)
    if cached is not None:
        _publish_latest(Path(run_root), identity)
        return ChannelAblationResult(run_dir=run_dir, summary=cached)

    store = _Store(run_dir, identity)
    try:
        store.json("protocol.json", {**protocol, **identity})
        channel_times = pd.DatetimeIndex(
            pd.to_datetime(
                frozen_j.labels_rr2["decision_time"].drop_duplicates(),
                utc=True,
                errors="raise",
            )
        )
        decisions = build_channel_free_decisions(
            minute,
            five,
            positioning,
            channel_times=channel_times,
            config=config,
        )
        predictions, fold_audit = run_expanding_oof(decisions, config=config)
        selected = select_matched_activations(
            predictions,
            config=config,
            matched_fold_counts=matched_fold_counts,
        )
        predictive = _predictive_metrics(predictions)
        opportunity, opportunity_comparisons = _opportunity_tables(
            selected,
            draws=config.bootstrap_draws,
            seed=config.random_seed,
        )
        economic_paths, chosen_paths = _replay_selected(
            selected, minute, config=config
        )
        economics, comparisons = _economic_tables(
            chosen_paths,
            opportunity_comparisons,
            draws=config.bootstrap_draws,
            seed=config.random_seed,
        )
        frequency = _frequency_audit(
            selected,
            matched_fold_counts=matched_fold_counts,
        )
        feature_audit = _feature_audit()

        development_end = pd.Timestamp(config.development_end_exclusive, tz="UTC")
        frequency_matched = bool(
            frequency["activations"].eq(target_activations).all()
            and frequency["fold_counts_exact"].astype(bool).all()
            and frequency["global_60m_refractory_passed"].astype(bool).all()
        )
        leakage = pd.DataFrame(
            [
                {
                    "check": "development-only bounded raw reads",
                    "passed": all(
                        pd.Timestamp(record["max_timestamp"]) < development_end
                        for record in bounded_identity["source_fingerprints"].values()
                    ),
                    "detail": bounded_identity["aggregate_sha256"],
                },
                {
                    "check": "recomputed U foldwise activation counts" if recomputed_handoffs() else "frozen V foldwise activation counts",
                    "passed": frozen_counts == matched_fold_counts,
                    "detail": str(frozen_counts),
                },
                {
                    "check": "shared models and OOF scores",
                    "passed": fold_audit["shared_between_window_sources"].astype(bool).all(),
                    "detail": "one model/head/fold scored both candidate universes",
                },
                {
                    "check": "strict prior-label purge",
                    "passed": (
                        pd.to_datetime(fold_audit["train_label_end_max"], utc=True)
                        <= pd.to_datetime(fold_audit["validation_start"], utc=True)
                    ).all(),
                    "detail": "120m label interval ends before validation",
                },
                {
                    "check": "channel-free feature allow-list",
                    "passed": not feature_audit["channel_feature"].astype(bool).any()
                    and not feature_audit["future_feature"].astype(bool).any(),
                    "detail": f"{len(OPPORTUNITY_FEATURES)} opportunity + {len(DIRECTION_FEATURES)} direction",
                },
                {
                    "check": "channel used only as candidate-universe restriction",
                    "passed": True,
                    "detail": "in_channel_window excluded from both model matrices",
                },
                {
                    "check": "exact matched frequency and global cooldown",
                    "passed": frequency_matched,
                    "detail": f"{target_activations} per arm/model; {target_activations_per_day:.6f}/day",
                },
                {
                    "check": "label-blind matched top-k",
                    "passed": not frequency["matching_used_outcome_labels"].astype(bool).any(),
                    "detail": "ranking uses only OOF opportunity_score and decision_time",
                },
                {
                    "check": "native RR2/120m stop-first replay",
                    "passed": set(economic_paths["target_multiple_b"]) == {2.0}
                    and set(economic_paths["hold_minutes"]) == {120},
                    "detail": "both directions replayed at every selected native 1m Open",
                },
                {
                    "check": "uniform 5+5 costs",
                    "passed": set(economic_paths["cost_bps"]) == {10.0},
                    "detail": "10 bps on TP, SL and timeout",
                },
                {
                    "check": "complete selected paths",
                    "passed": chosen_paths["path_complete"].astype(bool).all(),
                    "detail": f"{len(chosen_paths)} chosen paths",
                },
                {
                    "check": "forward and Q2 remain sealed",
                    "passed": decisions["decision_time"].max() < development_end
                    and minute.index.max() < development_end,
                    "detail": str(minute.index.max()),
                },
            ]
        )
        if not leakage["passed"].astype(bool).all():
            failed = leakage.loc[~leakage["passed"].astype(bool), "check"].tolist()
            raise AssertionError(f"Notebook W leakage audit failed: {failed}")

        xgb_channel = economics.loc[
            economics["model"].eq("xgboost")
            & economics["window_source"].eq("channel")
        ].iloc[0]
        xgb_net_delta = comparisons.loc[
            comparisons["model"].eq("xgboost")
            & comparisons["metric"].eq("mean_net_r")
        ].iloc[0]
        xgb_opportunity_delta = comparisons.loc[
            comparisons["model"].eq("xgboost")
            & comparisons["metric"].eq("opportunity_rate")
        ].iloc[0]
        retain_channels = channel_retention_decision(
            channel_ci_low=float(xgb_channel["mean_net_r_ci_low"]),
            channel_minus_blind_ci_low=float(xgb_net_delta["ci_low"]),
            opportunity_delta_ci_low=float(xgb_opportunity_delta["ci_low"]),
            leakage_passed=bool(leakage["passed"].astype(bool).all()),
            frequency_matched=frequency_matched,
        )
        decision = (
            "retain channels only as a candidate-window generator"
            if retain_channels
            else "close the channel branch; matched-frequency evidence did not pass"
        )

        store.parquet("oof_predictions.parquet", predictions)
        store.parquet("selected_activations.parquet", selected)
        store.parquet("economic_paths.parquet", economic_paths)
        store.csv("feature_audit.csv", feature_audit)
        store.csv("fold_audit.csv", fold_audit)
        store.csv("predictive_metrics.csv", predictive)
        store.csv("opportunity_metrics.csv", opportunity)
        store.csv("economic_metrics.csv", economics)
        store.csv("matched_comparisons.csv", comparisons)
        store.csv("frequency_audit.csv", frequency)
        store.csv("leakage_audit.csv", leakage)
        store.json(
            "frozen_protocol.json",
            {
                "frozen_u_run_hash": frozen_u.run_hash,
                "frozen_u_manifest_sha256": frozen_u.manifest_sha256,
                "frozen_u_activation_ledger_sha256": frozen_u.activation_ledger_sha256,
                "frozen_j_run_hash": frozen_j.run_hash,
                "frozen_j_manifest_sha256": frozen_j.manifest_sha256,
                "frozen_j_labels_rr2_sha256": input_payload[
                    "frozen_j_labels_rr2_sha256"
                ],
                "matched_fold_counts": matched_fold_counts,
                "matched_total_activations": target_activations,
                "target_activations_per_day": target_activations_per_day,
                "frozen_u_scored_fold_counts": frozen_counts,
                "matching_target_source": protocol["matching_target_source"],
                "bounded_development_input_identity": bounded_identity,
                "forward_or_lockbox_loaded": False,
            },
        )
        summary = {
            **identity,
            "research_claim": "development_matched_frequency_ablation",
            "decision_rows": len(decisions),
            "oof_rows": len(predictions),
            "matched_fold_counts": matched_fold_counts,
            "matched_total_activations": target_activations,
            "selected_activations_per_arm_model": target_activations,
            "target_activations_per_day": target_activations_per_day,
            "activations_per_day": target_activations_per_day,
            "frozen_u_scored_fold_counts": frozen_counts,
            "matching_target_source": protocol["matching_target_source"],
            "models": list(MODELS),
            "window_sources": list(WINDOW_SOURCES),
            "opportunity_feature_count": len(OPPORTUNITY_FEATURES),
            "direction_feature_count": len(DIRECTION_FEATURES),
            "round_trip_cost_bps": 10.0,
            "hold_minutes": config.hold_minutes,
            "target_multiple_b": config.target_multiple_b,
            "frequency_matched": frequency_matched,
            "xgboost_channel_mean_net_r": float(xgb_channel["mean_net_r"]),
            "xgboost_channel_mean_net_r_ci_low": float(
                xgb_channel["mean_net_r_ci_low"]
            ),
            "xgboost_channel_mean_net_r_ci_high": float(
                xgb_channel["mean_net_r_ci_high"]
            ),
            "xgboost_channel_minus_blind_mean_net_r": float(
                xgb_net_delta["point_delta"]
            ),
            "xgboost_channel_minus_blind_net_ci_low": float(xgb_net_delta["ci_low"]),
            "xgboost_channel_minus_blind_net_ci_high": float(xgb_net_delta["ci_high"]),
            "xgboost_channel_minus_blind_opportunity_rate": float(
                xgb_opportunity_delta["point_delta"]
            ),
            "xgboost_channel_minus_blind_opportunity_ci_low": float(
                xgb_opportunity_delta["ci_low"]
            ),
            "xgboost_channel_minus_blind_opportunity_ci_high": float(
                xgb_opportunity_delta["ci_high"]
            ),
            "retain_channels": retain_channels,
            "decision": decision,
            "leakage_checks": len(leakage),
            "leakage_passed": bool(leakage["passed"].astype(bool).all()),
            "max_loaded_timestamp": str(minute.index.max()),
            "forward_or_lockbox_loaded": False,
        }
        store.json("summary.json", summary)
        for name in READER_ARTIFACTS:
            if not (run_dir / name).is_file():
                raise RuntimeError(f"Notebook W artifact missing before completion: {name}")
        store.complete(summary)
        _publish_latest(Path(run_root), identity)
        return ChannelAblationResult(run_dir=run_dir, summary=summary)
    except BaseException as error:
        store.fail(error)
        raise


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="dev", choices=("dev",))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    args = parser.parse_args()
    result = run_channel_vs_volatility_ablation(
        stage=args.stage,
        data_root=args.data_root,
        run_root=args.run_root,
    )
    print(json.dumps(result.summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()


__all__ = [
    "ChannelAblationConfig",
    "ChannelAblationResult",
    "DIRECTION_FEATURES",
    "MATCHED_FOLD_COUNTS",
    "OPPORTUNITY_FEATURES",
    "READER_ARTIFACTS",
    "RUN_ROOT",
    "build_channel_free_decisions",
    "channel_retention_decision",
    "derive_matched_fold_counts",
    "effective_matched_fold_counts",
    "first_touch_labels",
    "matched_topk_refractory",
    "protocol_dict",
    "run_channel_vs_volatility_ablation",
]
