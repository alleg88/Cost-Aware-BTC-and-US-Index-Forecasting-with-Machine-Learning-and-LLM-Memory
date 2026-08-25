"""Causal activation policy for direction-free opportunity scores."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    target_activations_per_day: float
    actual_activations_per_day: float
    activations: int
    calendar_days: int
    calibration_rows: int


def collapse_episode_time(
    frame: pd.DataFrame,
    *,
    score_column: str = "score",
) -> pd.DataFrame:
    """Keep the maximum score for duplicate views of one episode-time decision."""
    required = {"channel_episode_id", "decision_time", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"opportunity rows missing columns: {missing}")
    work = frame.copy().reset_index(drop=True)
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    work[score_column] = pd.to_numeric(work[score_column], errors="raise")
    values = work[score_column].to_numpy(float)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("opportunity scores must be finite probabilities")
    work["_source_order"] = np.arange(len(work))
    work = work.sort_values(
        ["channel_episode_id", "decision_time", score_column, "_source_order"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop_duplicates(["channel_episode_id", "decision_time"], keep="first")
    return work.drop(columns="_source_order").sort_values(
        ["decision_time", "channel_episode_id"], kind="stable"
    ).reset_index(drop=True)


def causal_crossing_alerts(
    frame: pd.DataFrame,
    *,
    threshold: float,
    score_column: str = "score",
    cooldown_minutes: int = 60,
) -> pd.DataFrame:
    """Emit only fresh upward crossings with a forward-only episode cooldown."""
    required = {"channel_episode_id", "decision_time", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"opportunity policy missing columns: {missing}")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability")
    if cooldown_minutes <= 0:
        raise ValueError("cooldown_minutes must be positive")
    work = frame.copy().reset_index(drop=True)
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    score = pd.to_numeric(work[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(score).all() or ((score < 0.0) | (score > 1.0)).any():
        raise ValueError("opportunity scores must be finite probabilities")
    if work.duplicated(["channel_episode_id", "decision_time"]).any():
        raise ValueError("collapse duplicate episode-time rows before applying policy")
    work["_source_order"] = np.arange(len(work))
    work = work.sort_values(
        ["channel_episode_id", "decision_time", "_source_order"], kind="stable"
    ).reset_index(drop=True)
    work["alert"] = False
    cooldown = pd.Timedelta(minutes=int(cooldown_minutes))
    for _, positions in work.groupby("channel_episode_id", sort=False).groups.items():
        previous_above = False
        last_alert: pd.Timestamp | None = None
        for position in positions:
            above = bool(work.at[position, score_column] >= threshold)
            crossing = above and not previous_above
            time = pd.Timestamp(work.at[position, "decision_time"])
            outside_cooldown = last_alert is None or time - last_alert >= cooldown
            if crossing and outside_cooldown:
                work.at[position, "alert"] = True
                last_alert = time
            previous_above = above
    return work.drop(columns="_source_order").sort_values(
        ["decision_time", "channel_episode_id"], kind="stable"
    ).reset_index(drop=True)


def causal_level_rearm_alerts(
    frame: pd.DataFrame,
    *,
    threshold: float,
    score_column: str = "score",
    cooldown_minutes: int = 60,
) -> pd.DataFrame:
    """Emit an alert whenever an above-threshold level is re-armed by cooldown."""
    required = {"channel_episode_id", "decision_time", score_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"opportunity policy missing columns: {missing}")
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite probability")
    if cooldown_minutes <= 0:
        raise ValueError("cooldown_minutes must be positive")
    work = frame.copy().reset_index(drop=True)
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    score = pd.to_numeric(work[score_column], errors="raise").to_numpy(float)
    if not np.isfinite(score).all() or ((score < 0.0) | (score > 1.0)).any():
        raise ValueError("opportunity scores must be finite probabilities")
    if work.duplicated(["channel_episode_id", "decision_time"]).any():
        raise ValueError("collapse duplicate episode-time rows before applying policy")
    work["_source_order"] = np.arange(len(work))
    work = work.sort_values(
        ["channel_episode_id", "decision_time", "_source_order"], kind="stable"
    ).reset_index(drop=True)
    work["alert"] = False
    cooldown = pd.Timedelta(minutes=int(cooldown_minutes))
    for _, positions in work.groupby("channel_episode_id", sort=False).groups.items():
        last_alert: pd.Timestamp | None = None
        for position in positions:
            above = bool(work.at[position, score_column] >= threshold)
            time = pd.Timestamp(work.at[position, "decision_time"])
            outside_cooldown = last_alert is None or time - last_alert >= cooldown
            if above and outside_cooldown:
                work.at[position, "alert"] = True
                last_alert = time
    return work.drop(columns="_source_order").sort_values(
        ["decision_time", "channel_episode_id"], kind="stable"
    ).reset_index(drop=True)


def _level_rearm_counts_by_threshold(
    frame: pd.DataFrame,
    *,
    thresholds: np.ndarray,
    score_column: str,
    cooldown_minutes: int,
) -> np.ndarray:
    """Count re-arm alerts without rebuilding a DataFrame for every threshold."""
    ordered = frame.sort_values(
        ["channel_episode_id", "decision_time"], kind="stable"
    )
    episodes = ordered["channel_episode_id"].to_numpy()
    times = ordered["decision_time"].dt.as_unit("ns").astype("int64").to_numpy()
    scores = ordered[score_column].to_numpy(float)
    boundaries = np.concatenate(
        ([0], np.flatnonzero(episodes[1:] != episodes[:-1]) + 1, [len(ordered)])
    )
    cooldown_ns = int(pd.Timedelta(minutes=int(cooldown_minutes)).value)
    counts = np.zeros(len(thresholds), dtype=np.int64)
    for threshold_index, threshold in enumerate(thresholds):
        count = 0
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            candidates = times[start:end][scores[start:end] >= threshold]
            if not len(candidates):
                continue
            count += 1
            last_alert = int(candidates[0])
            for candidate in candidates[1:]:
                candidate = int(candidate)
                if candidate - last_alert >= cooldown_ns:
                    count += 1
                    last_alert = candidate
        counts[threshold_index] = count
    return counts


def select_causal_threshold(
    calibration: pd.DataFrame,
    *,
    target_activations_per_day: float,
    score_column: str = "score",
    cooldown_minutes: int = 60,
    grid_size: int = 51,
    alert_policy: str = "crossing",
) -> ThresholdSelection:
    """Choose a fixed threshold on a reserved past-only calibration block."""
    if target_activations_per_day <= 0.0:
        raise ValueError("target_activations_per_day must be positive")
    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    if alert_policy not in {"crossing", "level_rearm"}:
        raise ValueError("alert_policy must be crossing or level_rearm")
    work = collapse_episode_time(calibration, score_column=score_column)
    if work.empty:
        raise ValueError("threshold calibration rows must be non-empty")
    start = work["decision_time"].min().normalize()
    end = work["decision_time"].max().normalize()
    calendar_days = int((end - start).days + 1)
    values = work[score_column].to_numpy(float)
    candidates = np.unique(
        np.quantile(values, np.linspace(0.0, 1.0, min(grid_size, len(values))))
    )
    best: tuple[float, float, int, float] | None = None
    alert_function = (
        causal_crossing_alerts
        if alert_policy == "crossing"
        else causal_level_rearm_alerts
    )
    rearm_counts = (
        _level_rearm_counts_by_threshold(
            work,
            thresholds=candidates,
            score_column=score_column,
            cooldown_minutes=cooldown_minutes,
        )
        if alert_policy == "level_rearm"
        else None
    )
    for index, threshold in enumerate(candidates):
        if rearm_counts is None:
            replay = alert_function(
                work,
                threshold=float(threshold),
                score_column=score_column,
                cooldown_minutes=cooldown_minutes,
            )
            activations = int(replay["alert"].sum())
        else:
            activations = int(rearm_counts[index])
        rate = activations / calendar_days
        candidate = (
            abs(rate - target_activations_per_day),
            -float(threshold),
            activations,
            rate,
        )
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    if best is None:
        raise AssertionError("threshold grid unexpectedly empty")
    return ThresholdSelection(
        threshold=-best[1],
        target_activations_per_day=float(target_activations_per_day),
        actual_activations_per_day=float(best[3]),
        activations=int(best[2]),
        calendar_days=calendar_days,
        calibration_rows=len(work),
    )


__all__ = [
    "ThresholdSelection",
    "causal_crossing_alerts",
    "causal_level_rearm_alerts",
    "collapse_episode_time",
    "select_causal_threshold",
]
