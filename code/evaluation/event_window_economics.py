"""Causal counterfactual labels and chronological event-window policy replay.

Each active five-minute decision is labelled from the native one-minute path.
The policy then takes the first geometry-valid score crossing in each window;
an attempted entry consumes that window even when its future path is censored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from evaluation.channel_backtest import _resolve_1m
from features.event_window_inputs import known_structural_stop


LABEL_COLUMNS = (
    "window_id",
    "channel_episode_id",
    "side",
    "step",
    "source_bar_time",
    "decision_time",
    "entry_time",
    "exit_time",
    "entry",
    "stop",
    "target",
    "risk_bps",
    "outcome",
    "r_gross",
    "r_net",
    "label_start",
    "label_end",
    "geometry_valid",
    "path_observed",
    "model_target_valid",
)

DEFAULT_THRESHOLD_QUANTILES = (0.00, 0.25, 0.50, 0.75, 0.90, 0.95, 1.00)


@dataclass(frozen=True)
class EventLabelConfig:
    rr_multiple: float = 2.0
    stop_buffer_bps: float = 5.0
    trailing_stop_bars: int = 12
    max_hold_minutes: int = 120
    cost_bps: float = 10.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.rr_multiple) or self.rr_multiple <= 0.0:
            raise ValueError("rr_multiple must be positive and finite")
        if not np.isfinite(self.stop_buffer_bps) or self.stop_buffer_bps < 0.0:
            raise ValueError("stop_buffer_bps must be non-negative and finite")
        if not isinstance(self.trailing_stop_bars, int) or self.trailing_stop_bars < 1:
            raise ValueError("trailing_stop_bars must be a positive integer")
        if not isinstance(self.max_hold_minutes, int) or self.max_hold_minutes < 1:
            raise ValueError("max_hold_minutes must be a positive integer")
        if not np.isfinite(self.cost_bps) or self.cost_bps < 0.0:
            raise ValueError("cost_bps must be non-negative and finite")


@dataclass(frozen=True)
class PolicyReplay:
    trades: pd.DataFrame
    daily_frequency: pd.DataFrame
    summary: dict[str, float | int | bool]


def _utc_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _utc_frame(frame: pd.DataFrame, *, required: set[str], name: str) -> pd.DataFrame:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{name} must use a DatetimeIndex")
    out = frame.copy()
    index = pd.to_datetime(out.index, utc=True, errors="raise")
    if index.has_duplicates:
        raise ValueError(f"{name} index must be unique")
    if not index.is_monotonic_increasing:
        raise ValueError(f"{name} index must be sorted")
    out.index = index
    return out


def _sequence_contract(
    sequences: Any,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    required = ("metadata", "source_bar_times", "decision_times", "decision_valid")
    missing = [name for name in required if not hasattr(sequences, name)]
    if missing:
        raise ValueError(f"EventWindowSequences missing attributes: {missing}")
    metadata = sequences.metadata.copy()
    missing_columns = sorted(
        {"window_id", "channel_episode_id", "side"}.difference(metadata.columns)
    )
    if missing_columns:
        raise ValueError(f"sequence metadata missing columns: {missing_columns}")
    source = np.asarray(sequences.source_bar_times)
    decisions = np.asarray(sequences.decision_times)
    valid = np.asarray(sequences.decision_valid, dtype=bool)
    if source.ndim != 2 or source.shape != decisions.shape or source.shape != valid.shape:
        raise ValueError("source_bar_times, decision_times and decision_valid must align")
    if source.shape[0] != len(metadata):
        raise ValueError("sequence timestamp rows must align with metadata")
    return metadata.reset_index(drop=True), source, decisions, valid


def _gross_r(side: str, entry: float, exit_price: float, risk: float) -> float:
    move = exit_price - entry if side == "long" else entry - exit_price
    return float(move / risk)


def label_window_steps(
    sequences: Any,
    five_minute: pd.DataFrame,
    minute: pd.DataFrame,
    config: EventLabelConfig = EventLabelConfig(),
) -> pd.DataFrame:
    """Label every valid active decision from its native next-open 1m path.

    ``sequences`` is intentionally duck-typed so the economic layer can be
    tested independently from the tensor-builder implementation.
    """
    metadata, source_times, decision_times, decision_valid = _sequence_contract(sequences)
    five = _utc_frame(
        five_minute,
        required={"open", "high", "low", "close"},
        name="five_minute",
    )
    one = _utc_frame(minute, required={"open", "high", "low", "close"}, name="minute")

    rows: list[dict[str, object]] = []
    for sample_no, meta in metadata.iterrows():
        side = str(meta["side"]).lower()
        if side not in {"long", "short"}:
            raise ValueError(f"invalid side in sequence metadata: {meta['side']!r}")
        for step in np.flatnonzero(decision_valid[sample_no]):
            source_time = _utc_timestamp(source_times[sample_no, step], name="source_bar_time")
            decision_time = _utc_timestamp(decision_times[sample_no, step], name="decision_time")
            if pd.isna(source_time) or pd.isna(decision_time):
                raise ValueError("valid decisions require finite source and decision times")
            if decision_time != source_time + pd.Timedelta(minutes=5):
                raise ValueError("decision_time must be the next 5m open after source_bar_time")

            entry_time = decision_time
            entry = np.nan
            if entry_time in one.index:
                entry = float(one.at[entry_time, "open"])
            # The shared helper needs only the causal trailing slice. Passing the
            # multi-year frame here would make its defensive copy quadratic in
            # the number of labelled decisions.
            stop_history = five.loc[:source_time].tail(config.trailing_stop_bars)
            stop = known_structural_stop(
                stop_history,
                side=side,
                source_bar_time=source_time,
                lookback=config.trailing_stop_bars,
                buffer_bps=config.stop_buffer_bps,
            )
            risk = (entry - stop) if side == "long" else (stop - entry)
            risk_bps = (
                float(risk / entry * 1e4)
                if np.isfinite(entry) and entry > 0.0 and np.isfinite(risk)
                else np.nan
            )
            target = (
                entry + config.rr_multiple * risk
                if side == "long"
                else entry - config.rr_multiple * risk
            )
            geometry_valid = bool(
                np.isfinite(entry)
                and entry > 0.0
                and np.isfinite(stop)
                and stop > 0.0
                and np.isfinite(risk)
                and risk > 0.0
                and np.isfinite(target)
                and target > 0.0
            )

            exit_time: pd.Timestamp | pd.NaT = pd.NaT
            label_end = entry_time
            outcome = "invalid_geometry"
            r_gross = np.nan
            r_net = np.nan
            path_observed = False
            if geometry_valid:
                exit_time, exit_price, outcome = _resolve_1m(
                    side,
                    float(entry),
                    float(stop),
                    float(target),
                    one,
                    entry_time,
                    config.max_hold_minutes,
                    None,
                    None,
                )
                exit_time = _utc_timestamp(exit_time, name="exit_time")
                label_end = exit_time + pd.Timedelta(minutes=1)
                path_observed = outcome in {"tp", "sl", "timeout"}
                if path_observed and np.isfinite(exit_price):
                    r_gross = _gross_r(side, float(entry), float(exit_price), float(risk))
                    r_net = float(r_gross - config.cost_bps / risk_bps)

            model_target_valid = bool(geometry_valid and path_observed)
            rows.append(
                {
                    "window_id": meta["window_id"],
                    "channel_episode_id": meta["channel_episode_id"],
                    "side": side,
                    "step": int(step),
                    "source_bar_time": source_time,
                    "decision_time": decision_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "risk_bps": risk_bps,
                    "outcome": outcome,
                    "r_gross": r_gross,
                    "r_net": r_net,
                    "label_start": entry_time,
                    "label_end": label_end,
                    "geometry_valid": geometry_valid,
                    "path_observed": path_observed,
                    "model_target_valid": model_target_valid,
                }
            )

    return pd.DataFrame(rows, columns=LABEL_COLUMNS)


def _evaluation_calendar(
    start: Any, end: Any
) -> tuple[pd.Timestamp, pd.Timestamp, pd.DatetimeIndex]:
    start_time = _utc_timestamp(start, name="start")
    end_time = _utc_timestamp(end, name="end")
    if pd.isna(start_time) or pd.isna(end_time) or end_time <= start_time:
        raise ValueError("end must be later than start")
    final_day = (end_time - pd.Timedelta(nanoseconds=1)).normalize()
    calendar = pd.date_range(start_time.normalize(), final_day, freq="D", tz="UTC")
    calendar.name = "date"
    return start_time, end_time, calendar


def _attach_scores(scores: Any, labels: pd.DataFrame) -> pd.DataFrame:
    if isinstance(scores, pd.DataFrame):
        if "score" not in scores:
            raise ValueError("scores must contain a score column")
        if {"window_id", "step"} <= set(scores.columns):
            lookup = scores[["window_id", "step", "score"]].copy()
            if lookup.duplicated(["window_id", "step"]).any():
                raise ValueError("scores must be unique by window_id and step")
            work = labels.merge(
                lookup,
                on=["window_id", "step"],
                how="left",
                validate="one_to_one",
            )
        elif len(scores) == len(labels):
            work = labels.copy()
            work["score"] = scores["score"].to_numpy()
        else:
            raise ValueError("unkeyed scores must align row-for-row with labels")
    else:
        values = np.asarray(scores)
        if values.ndim != 1 or len(values) != len(labels):
            raise ValueError("scores must be one-dimensional and align with labels")
        work = labels.copy()
        work["score"] = values
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    return work


def replay_first_crossing(
    scores: Any,
    labels: pd.DataFrame,
    threshold: float,
    start: Any,
    end: Any,
) -> PolicyReplay:
    """Take the first geometry-valid score crossing in each causal window."""
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be non-negative and finite")
    required = {
        "window_id",
        "channel_episode_id",
        "side",
        "step",
        "decision_time",
        "entry_time",
        "geometry_valid",
        "path_observed",
        "model_target_valid",
        "outcome",
        "r_net",
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"labels missing policy columns: {missing}")
    start_time, end_time, calendar = _evaluation_calendar(start, end)
    work = _attach_scores(scores, labels)
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="raise")
    work = work[
        work["entry_time"].ge(start_time) & work["entry_time"].lt(end_time)
    ].copy()
    work["_input_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["window_id", "decision_time", "step", "_input_order"], kind="stable"
    )
    crossing = (
        work["geometry_valid"].astype(bool)
        & np.isfinite(work["score"])
        & work["score"].ge(float(threshold))
    )
    trades = (
        work.loc[crossing]
        .groupby("window_id", sort=False, as_index=False)
        .head(1)
        .drop(columns="_input_order")
        .sort_values(["entry_time", "window_id"], kind="stable")
        .reset_index(drop=True)
    )

    daily = pd.DataFrame(index=calendar)
    if trades.empty:
        daily["attempted_trades"] = 0
        daily["observed_trades"] = 0
        daily["censored_trades"] = 0
        daily["net_r"] = 0.0
    else:
        trade_day = trades["entry_time"].dt.normalize()
        observed = trades["path_observed"].astype(bool)
        daily["attempted_trades"] = trade_day.value_counts().reindex(calendar, fill_value=0)
        daily["observed_trades"] = trade_day[observed].value_counts().reindex(
            calendar, fill_value=0
        )
        daily["censored_trades"] = trade_day[~observed].value_counts().reindex(
            calendar, fill_value=0
        )
        daily["net_r"] = (
            trades.loc[observed]
            .groupby(trade_day[observed])["r_net"]
            .sum()
            .reindex(calendar, fill_value=0.0)
        )
    for column in ("attempted_trades", "observed_trades", "censored_trades"):
        daily[column] = daily[column].astype(int)
    daily["net_r"] = daily["net_r"].astype(float)

    observed = (
        trades["path_observed"].astype(bool)
        if len(trades)
        else pd.Series(dtype=bool)
    )
    observed_r = pd.to_numeric(trades.loc[observed, "r_net"], errors="coerce").dropna()
    attempted = int(len(trades))
    trades_per_day = float(attempted / len(calendar))
    total_net_r = float(observed_r.sum()) if len(observed_r) else 0.0
    mean_net_r = float(observed_r.mean()) if len(observed_r) else np.nan
    frequency_admissible = bool(2.0 <= trades_per_day <= 5.0)
    economic_admissible = bool(mean_net_r > 0.0 and total_net_r > 0.0)
    summary: dict[str, float | int | bool] = {
        "threshold": float(threshold),
        "calendar_days": int(len(calendar)),
        "attempted_trades": attempted,
        "observed_trades": int(observed.sum()),
        "censored_trades": int((~observed).sum()),
        "long_trades": int(trades["side"].eq("long").sum()) if attempted else 0,
        "short_trades": int(trades["side"].eq("short").sum()) if attempted else 0,
        "trades_per_day": trades_per_day,
        "total_net_r": total_net_r,
        "mean_net_r": mean_net_r,
        "frequency_admissible": frequency_admissible,
        "economic_admissible": economic_admissible,
        "policy_admissible": bool(frequency_admissible and economic_admissible),
    }
    return PolicyReplay(trades=trades, daily_frequency=daily, summary=summary)


def _threshold_grid(scores: Any, labels: pd.DataFrame) -> list[float]:
    work = _attach_scores(scores, labels)
    finite = pd.to_numeric(work["score"], errors="coerce")
    non_negative = finite[np.isfinite(finite) & finite.ge(0.0)]
    if non_negative.empty:
        return [0.0]
    values = [0.0]
    values.extend(float(non_negative.quantile(q)) for q in DEFAULT_THRESHOLD_QUANTILES)
    return sorted({value for value in values if np.isfinite(value) and value >= 0.0})


def sweep_event_thresholds(
    scores: Any,
    labels: pd.DataFrame,
    *,
    start: Any,
    end: Any,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """Replay a non-negative score frontier with separate frequency/economic flags."""
    grid = (
        _threshold_grid(scores, labels)
        if thresholds is None
        else [float(x) for x in thresholds]
    )
    if not grid:
        raise ValueError("threshold grid cannot be empty")
    if any(not np.isfinite(value) or value < 0.0 for value in grid):
        raise ValueError("thresholds must be non-negative and finite")

    rows: list[dict[str, object]] = []
    for threshold in sorted(set(grid)):
        replay = replay_first_crossing(
            scores,
            labels,
            threshold=threshold,
            start=start,
            end=end,
        )
        summary = replay.summary
        frequency = bool(2.0 <= float(summary["trades_per_day"]) <= 5.0)
        economic = bool(
            threshold >= 0.0
            and float(summary["mean_net_r"]) > 0.0
            and float(summary["total_net_r"]) > 0.0
        )
        rows.append(
            {
                **summary,
                "frequency_admissible": frequency,
                "economic_admissible": economic,
                "policy_admissible": bool(frequency and economic),
            }
        )
    return pd.DataFrame(rows).sort_values("threshold", kind="stable").reset_index(drop=True)


__all__ = [
    "DEFAULT_THRESHOLD_QUANTILES",
    "EventLabelConfig",
    "LABEL_COLUMNS",
    "PolicyReplay",
    "label_window_steps",
    "replay_first_crossing",
    "sweep_event_thresholds",
]
