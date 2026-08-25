"""Episode-purged walk-forward folds and interval uniqueness for Notebook B."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PurgedFold:
    fold_id: str
    train: np.ndarray
    valid: np.ndarray
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


VALID_BLOCKS = tuple(
    (
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(start, tz="UTC") + pd.DateOffset(months=6),
    )
    for start in (
        "2022-01-01",
        "2022-07-01",
        "2023-01-01",
        "2023-07-01",
        "2024-01-01",
        "2024-07-01",
        "2025-01-01",
    )
)

_FOLD_REQUIRED = frozenset(
    {"decision_time", "label_start", "label_end", "channel_episode_id"}
)


def _utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def expanding_purged_folds(
    events: pd.DataFrame,
    *,
    valid_blocks: tuple[tuple[pd.Timestamp, pd.Timestamp], ...] = VALID_BLOCKS,
) -> list[PurgedFold]:
    """Return anchored folds without episode sharing or live-label overlap."""
    missing = sorted(_FOLD_REQUIRED.difference(events.columns))
    if missing:
        raise ValueError(f"events missing validation columns: {missing}")
    work = events.copy()
    for column in ("decision_time", "label_start", "label_end"):
        work[column] = _utc(work[column])
    if work[list(_FOLD_REQUIRED)].isna().any().any():
        raise ValueError("validation fields cannot contain missing values")
    if (work["label_end"] < work["label_start"]).any():
        raise ValueError("label_end cannot precede label_start")

    groups = work.groupby("channel_episode_id", sort=False).agg(
        group_start=("decision_time", "min"),
        group_decision_end=("decision_time", "max"),
        group_label_end=("label_end", "max"),
    )
    folds: list[PurgedFold] = []
    for valid_start, valid_end in valid_blocks:
        valid_start = pd.Timestamp(valid_start)
        valid_end = pd.Timestamp(valid_end)
        valid_start = (
            valid_start.tz_localize("UTC")
            if valid_start.tzinfo is None
            else valid_start.tz_convert("UTC")
        )
        valid_end = (
            valid_end.tz_localize("UTC")
            if valid_end.tzinfo is None
            else valid_end.tz_convert("UTC")
        )
        train_episodes = groups.index[groups["group_label_end"] < valid_start]
        valid_episodes = groups.index[
            groups["group_start"].ge(valid_start)
            & groups["group_decision_end"].lt(valid_end)
            & groups["group_label_end"].lt(valid_end)
        ]
        train_mask = (
            work["channel_episode_id"].isin(train_episodes)
            & work["label_end"].lt(valid_start)
        )
        valid_mask = (
            work["channel_episode_id"].isin(valid_episodes)
            & work["decision_time"].ge(valid_start)
            & work["decision_time"].lt(valid_end)
            & work["label_end"].lt(valid_end)
        )
        train = np.flatnonzero(train_mask.to_numpy())
        valid = np.flatnonzero(valid_mask.to_numpy())
        overlap = set(work.iloc[train]["channel_episode_id"]).intersection(
            work.iloc[valid]["channel_episode_id"]
        )
        if overlap:
            raise AssertionError(f"fold split channel episodes: {sorted(overlap)}")
        fold_id = f"{valid_start.year}H{1 if valid_start.month == 1 else 2}"
        folds.append(
            PurgedFold(
                fold_id=fold_id,
                train=train,
                valid=valid,
                train_end=valid_start,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return folds


def interval_uniqueness(
    events: pd.DataFrame,
    train_index: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Average inverse label concurrency using only the requested training rows."""
    required = {"label_start", "label_end"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise ValueError(f"events missing interval columns: {missing}")
    positions = np.asarray(train_index, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("train_index must be one-dimensional")
    if positions.size == 0:
        return np.array([], dtype=float)
    if positions.min() < 0 or positions.max() >= len(events):
        raise IndexError("train_index contains an out-of-range row")

    selected = events.iloc[positions]
    start = _utc(selected["label_start"])
    end = _utc(selected["label_end"])
    if start.isna().any() or end.isna().any():
        raise ValueError("label intervals cannot contain missing timestamps")
    if (end < start).any():
        raise ValueError("label_end cannot precede label_start")

    origin = start.min().floor("min")
    start_minutes = (start - origin).dt.total_seconds().to_numpy() / 60.0
    end_minutes = (end - origin).dt.total_seconds().to_numpy() / 60.0
    start_offset = np.floor(start_minutes).astype(np.int64)
    end_offset = np.ceil(end_minutes).astype(np.int64)
    difference = np.zeros(int(end_offset.max()) + 2, dtype=np.int64)
    np.add.at(difference, start_offset, 1)
    np.add.at(difference, end_offset + 1, -1)
    concurrency = np.cumsum(difference[:-1])

    weights = np.empty(len(selected), dtype=float)
    for row, (left, right) in enumerate(zip(start_offset, end_offset, strict=True)):
        active = concurrency[left:right + 1]
        if active.size == 0 or (active <= 0).any():
            raise AssertionError("invalid label concurrency")
        weights[row] = np.mean(1.0 / active)
    if normalize:
        mean_weight = float(weights.mean())
        if not np.isfinite(mean_weight) or mean_weight <= 0:
            raise ValueError("uniqueness weights are not positive and finite")
        weights = weights / mean_weight
    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish effective sample size for positive sample weights."""
    values = np.asarray(weights, dtype=float)
    if values.ndim != 1:
        raise ValueError("weights must be one-dimensional")
    if values.size == 0:
        return 0.0
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("weights must be positive and finite")
    return float(values.sum() ** 2 / np.square(values).sum())
