from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_compact_features import (
    COMPACT_FEATURES,
    DERIVED_COMPACT_FEATURES,
    NATIVE_COMPACT_FEATURES,
    build_compact_dataset,
)
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


EXPECTED_NATIVE = (
    "adaptive_barrier_bps",
    "past_sigma_5m_bps",
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
    "range_bps",
    "realized_vol_12",
    "activity_ratio",
    "range_bps_mean_3",
    "realized_vol_12_mean_3",
    "window_age_fraction",
    "channel_regime_age_hours",
    "channel_width_pct",
    "volatility_regime_percentile",
)
EXPECTED_DERIVED = (
    "channel_center_distance",
    "nearest_rail_distance_bps",
    "rail_approach_15m",
)
DERIVATION_SOURCES = (
    "raw_channel_position",
    "raw_channel_position_mean_3",
    "raw_distance_lower_bps",
    "raw_distance_upper_bps",
)


def _dataset(rows: list[dict[str, float]] | None = None) -> LargeMoveDecisionDataset:
    raw_rows = rows or [{}]
    feature_names = (*EXPECTED_NATIVE, *DERIVATION_SOURCES, "oi_z_7d")
    matrix_rows = []
    defaults = {name: float(index + 1) for index, name in enumerate(feature_names)}
    defaults.update(
        {
            "raw_channel_position": 0.2,
            "raw_channel_position_mean_3": 0.3,
            "raw_distance_lower_bps": 20.0,
            "raw_distance_upper_bps": 80.0,
        }
    )
    for row in raw_rows:
        values = {**defaults, **row}
        matrix_rows.append([values[name] for name in feature_names])
    decisions = pd.DataFrame(
        {
            "window_id": np.arange(len(raw_rows), dtype=int) + 1,
            "step": np.zeros(len(raw_rows), dtype=int),
            "decision_time": pd.date_range(
                "2024-01-01 00:05:00", periods=len(raw_rows), freq="5min", tz="UTC"
            ),
        }
    )
    return LargeMoveDecisionDataset(
        decisions=decisions,
        tabular=np.asarray(matrix_rows, dtype=np.float32),
        tabular_features=feature_names,
        dropped_features=(),
        feature_set="frozen_N3_side_neutral_volatility",
    )


def _row_lookup(dataset: LargeMoveDecisionDataset, row: int = 0) -> dict[str, float]:
    return dict(zip(dataset.tabular_features, dataset.tabular[row], strict=True))


def test_compact_contract_is_exact_ordered_and_small():
    assert NATIVE_COMPACT_FEATURES == EXPECTED_NATIVE
    assert DERIVED_COMPACT_FEATURES == EXPECTED_DERIVED
    assert COMPACT_FEATURES == (*EXPECTED_NATIVE, *EXPECTED_DERIVED)
    assert len(COMPACT_FEATURES) == 28
    assert not any(
        token in feature.lower()
        for feature in COMPACT_FEATURES
        for token in (
            "funding",
            "positioning",
            "sentiment",
            "calendar",
            "impulse",
            "rsi",
            "oi_",
            "side",
        )
    )


def test_build_compact_dataset_preserves_rows_and_derives_timing_fields():
    base = _dataset()
    compact = build_compact_dataset(base)
    values = _row_lookup(compact)

    assert compact.feature_set == "compact_volatility_timing_v1"
    assert compact.tabular_features == COMPACT_FEATURES
    assert compact.tabular.shape == (1, 28)
    assert compact.tabular.dtype == np.float32
    pd.testing.assert_frame_equal(compact.decisions, base.decisions)
    assert values["channel_center_distance"] == pytest.approx(0.3)
    assert values["nearest_rail_distance_bps"] == pytest.approx(20.0)
    assert values["rail_approach_15m"] == pytest.approx(0.1)
    assert "oi_z_7d" not in compact.tabular_features


def test_channel_timing_features_are_reflection_and_rail_swap_invariant():
    left = build_compact_dataset(
        _dataset(
            [
                {
                    "raw_channel_position": 0.2,
                    "raw_channel_position_mean_3": 0.3,
                    "raw_distance_lower_bps": 20.0,
                    "raw_distance_upper_bps": 80.0,
                }
            ]
        )
    )
    right = build_compact_dataset(
        _dataset(
            [
                {
                    "raw_channel_position": 0.8,
                    "raw_channel_position_mean_3": 0.7,
                    "raw_distance_lower_bps": 80.0,
                    "raw_distance_upper_bps": 20.0,
                }
            ]
        )
    )

    np.testing.assert_allclose(left.tabular, right.tabular, rtol=0.0, atol=1e-7)


@pytest.mark.parametrize(
    "missing_source,missing_outputs",
    [
        ("raw_channel_position", {"channel_center_distance", "rail_approach_15m"}),
        ("raw_channel_position_mean_3", {"rail_approach_15m"}),
        ("raw_distance_lower_bps", {"nearest_rail_distance_bps"}),
    ],
)
def test_missing_channel_source_stays_missing(missing_source, missing_outputs):
    compact = build_compact_dataset(_dataset([{missing_source: np.nan}]))
    values = _row_lookup(compact)
    assert all(np.isnan(values[name]) for name in missing_outputs)


def test_future_row_perturbation_cannot_change_prior_compact_row():
    base = _dataset([{}, {"past_rv_120_bps": 999.0, "raw_channel_position": 0.9}])
    changed = _dataset([{}, {"past_rv_120_bps": -999.0, "raw_channel_position": 0.1}])

    before = build_compact_dataset(base).tabular[0]
    after = build_compact_dataset(changed).tabular[0]
    np.testing.assert_array_equal(before, after)


def test_missing_required_source_is_rejected():
    base = _dataset()
    keep = [name != "past_rv_30_bps" for name in base.tabular_features]
    broken = LargeMoveDecisionDataset(
        decisions=base.decisions,
        tabular=base.tabular[:, keep],
        tabular_features=tuple(
            name for name, retained in zip(base.tabular_features, keep, strict=True) if retained
        ),
        dropped_features=(),
        feature_set=base.feature_set,
    )

    with pytest.raises(ValueError, match="past_rv_30_bps"):
        build_compact_dataset(broken)
