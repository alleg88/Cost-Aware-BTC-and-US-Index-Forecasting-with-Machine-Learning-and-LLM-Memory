"""Causal timestamp-only feature block for Notebook S."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


CALENDAR_FEATURE_COLUMNS = (
    "utc_hour_sin",
    "utc_hour_cos",
    "utc_weekday_sin",
    "utc_weekday_cos",
    "weekend_flag",
    "us_cash_session_flag",
)


def append_calendar_features(
    dataset: LargeMoveDecisionDataset,
) -> LargeMoveDecisionDataset:
    """Append six deterministic features known at the decision timestamp."""
    if "decision_time" not in dataset.decisions:
        raise ValueError("calendar features require decision_time")
    if dataset.tabular.ndim != 2 or len(dataset.tabular) != len(dataset.decisions):
        raise ValueError("calendar feature rows must align with the decision matrix")
    if dataset.tabular.shape[1] != len(dataset.tabular_features):
        raise ValueError("calendar feature names must align with the decision matrix")
    duplicate = sorted(set(CALENDAR_FEATURE_COLUMNS).intersection(dataset.tabular_features))
    if duplicate:
        raise ValueError(f"calendar features are already present: {duplicate}")

    utc = pd.to_datetime(dataset.decisions["decision_time"], utc=True, errors="raise")
    fractional_hour = utc.dt.hour.to_numpy(float) + utc.dt.minute.to_numpy(float) / 60.0
    weekday = utc.dt.dayofweek.to_numpy(float)
    new_york = utc.dt.tz_convert("America/New_York")
    local_minute = (
        new_york.dt.hour.to_numpy(int) * 60 + new_york.dt.minute.to_numpy(int)
    )
    local_weekday = new_york.dt.dayofweek.to_numpy(int)

    block = np.column_stack(
        [
            np.sin(2.0 * np.pi * fractional_hour / 24.0),
            np.cos(2.0 * np.pi * fractional_hour / 24.0),
            np.sin(2.0 * np.pi * weekday / 7.0),
            np.cos(2.0 * np.pi * weekday / 7.0),
            (weekday >= 5.0).astype(float),
            (
                (local_weekday < 5)
                & (local_minute >= 9 * 60 + 30)
                & (local_minute < 16 * 60)
            ).astype(float),
        ]
    ).astype(np.float32)
    if not np.isfinite(block).all():
        raise ValueError("calendar features must be finite")

    return LargeMoveDecisionDataset(
        decisions=dataset.decisions.copy(),
        tabular=np.column_stack(
            [np.asarray(dataset.tabular, dtype=np.float32), block]
        ).astype(np.float32, copy=False),
        tabular_features=(*dataset.tabular_features, *CALENDAR_FEATURE_COLUMNS),
        dropped_features=dataset.dropped_features,
        feature_set=f"{dataset.feature_set}+calendar_v1",
    )


__all__ = ["CALENDAR_FEATURE_COLUMNS", "append_calendar_features"]
