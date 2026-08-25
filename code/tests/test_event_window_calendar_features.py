from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_calendar_features import (
    CALENDAR_FEATURE_COLUMNS,
    append_calendar_features,
)
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


def _dataset_at(timestamps: list[str]) -> LargeMoveDecisionDataset:
    rows = len(timestamps)
    decisions = pd.DataFrame(
        {
            "window_id": [f"w{index}" for index in range(rows)],
            "step": np.arange(rows, dtype=int),
            "decision_time": pd.to_datetime(timestamps, utc=True),
            "outcome": ["tp" if index % 2 == 0 else "sl" for index in range(rows)],
            "magnitude_class": np.arange(rows, dtype=int) % 5,
        }
    )
    return LargeMoveDecisionDataset(
        decisions=decisions,
        tabular=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        tabular_features=("past_a", "past_b"),
        dropped_features=("future_field",),
        feature_set="opportunity_side_neutral_volatility",
    )


def _calendar_matrix(dataset: LargeMoveDecisionDataset) -> np.ndarray:
    return dataset.tabular[:, -len(CALENDAR_FEATURE_COLUMNS) :]


def _calendar_column(dataset: LargeMoveDecisionDataset, name: str) -> np.ndarray:
    index = dataset.tabular_features.index(name)
    return dataset.tabular[:, index]


def test_calendar_block_appends_exact_six_finite_columns_without_changing_base():
    source = _dataset_at(["2024-01-02 00:00Z", "2024-01-02 12:05Z"])

    result = append_calendar_features(source)

    assert CALENDAR_FEATURE_COLUMNS == (
        "utc_hour_sin",
        "utc_hour_cos",
        "utc_weekday_sin",
        "utc_weekday_cos",
        "weekend_flag",
        "us_cash_session_flag",
    )
    assert result.tabular_features == (*source.tabular_features, *CALENDAR_FEATURE_COLUMNS)
    assert np.array_equal(result.tabular[:, :2], source.tabular)
    assert np.isfinite(_calendar_matrix(result)).all()
    assert result.tabular.dtype == np.float32
    assert result.feature_set == "opportunity_side_neutral_volatility+calendar_v1"


def test_us_cash_session_proxy_handles_est_edt_and_exclusive_close():
    source = _dataset_at(
        [
            "2024-01-02 14:30Z",  # 09:30 EST
            "2024-07-02 13:30Z",  # 09:30 EDT
            "2024-01-02 21:00Z",  # 16:00 EST, exclusive
            "2024-01-06 15:00Z",  # Saturday
        ]
    )

    result = append_calendar_features(source)

    assert _calendar_column(result, "us_cash_session_flag").tolist() == [
        1.0,
        1.0,
        0.0,
        0.0,
    ]
    assert _calendar_column(result, "weekend_flag").tolist() == [0.0, 0.0, 0.0, 1.0]


def test_calendar_values_depend_only_on_decision_time_not_outcomes():
    source = _dataset_at(["2024-03-11 13:35Z", "2024-11-04 14:35Z"])
    changed_decisions = source.decisions.copy()
    changed_decisions["outcome"] = ["sl", "tp"]
    changed_decisions["magnitude_class"] = [4, 0]
    changed = LargeMoveDecisionDataset(
        decisions=changed_decisions,
        tabular=source.tabular.copy(),
        tabular_features=source.tabular_features,
        dropped_features=source.dropped_features,
        feature_set=source.feature_set,
    )

    original_features = _calendar_matrix(append_calendar_features(source))
    changed_features = _calendar_matrix(append_calendar_features(changed))

    assert np.array_equal(original_features, changed_features)


def test_calendar_transform_rejects_missing_invalid_or_duplicate_contract():
    source = _dataset_at(["2024-01-02 12:00Z"])
    missing = LargeMoveDecisionDataset(
        decisions=source.decisions.drop(columns="decision_time"),
        tabular=source.tabular,
        tabular_features=source.tabular_features,
        dropped_features=source.dropped_features,
        feature_set=source.feature_set,
    )
    duplicate = LargeMoveDecisionDataset(
        decisions=source.decisions,
        tabular=np.column_stack([source.tabular, np.ones((1, 1), dtype=np.float32)]),
        tabular_features=(*source.tabular_features, "utc_hour_sin"),
        dropped_features=source.dropped_features,
        feature_set=source.feature_set,
    )

    with pytest.raises(ValueError, match="decision_time"):
        append_calendar_features(missing)
    with pytest.raises(ValueError, match="already present"):
        append_calendar_features(duplicate)

