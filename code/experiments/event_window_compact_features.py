"""Pre-registered direction-invariant volatility and timing feature matrix."""
from __future__ import annotations

import numpy as np

from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


NATIVE_COMPACT_FEATURES = (
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

DERIVED_COMPACT_FEATURES = (
    "channel_center_distance",
    "nearest_rail_distance_bps",
    "rail_approach_15m",
)

COMPACT_FEATURES = (*NATIVE_COMPACT_FEATURES, *DERIVED_COMPACT_FEATURES)

_DERIVATION_SOURCES = (
    "raw_channel_position",
    "raw_channel_position_mean_3",
    "raw_distance_lower_bps",
    "raw_distance_upper_bps",
)


def _validated_lookup(base: LargeMoveDecisionDataset) -> dict[str, int]:
    if base.tabular.ndim != 2:
        raise ValueError("frozen base matrix must be two-dimensional")
    if base.tabular.shape[0] != len(base.decisions):
        raise ValueError("frozen base rows do not align with decisions")
    if base.tabular.shape[1] != len(base.tabular_features):
        raise ValueError("frozen base columns do not align with feature names")
    if len(base.tabular_features) != len(set(base.tabular_features)):
        raise ValueError("frozen base feature names must be unique")
    lookup = {name: index for index, name in enumerate(base.tabular_features)}
    required = (*NATIVE_COMPACT_FEATURES, *_DERIVATION_SOURCES)
    missing = sorted(set(required).difference(lookup))
    if missing:
        raise ValueError(f"frozen base is missing compact sources: {missing}")
    return lookup


def build_compact_dataset(base: LargeMoveDecisionDataset) -> LargeMoveDecisionDataset:
    """Select the fixed native fields and derive three row-causal timing fields."""
    lookup = _validated_lookup(base)
    matrix = np.asarray(base.tabular, dtype=np.float32)
    native = matrix[:, [lookup[name] for name in NATIVE_COMPACT_FEATURES]]

    position = matrix[:, lookup["raw_channel_position"]]
    position_mean_3 = matrix[:, lookup["raw_channel_position_mean_3"]]
    lower = matrix[:, lookup["raw_distance_lower_bps"]]
    upper = matrix[:, lookup["raw_distance_upper_bps"]]

    center_distance = np.abs(position - np.float32(0.5))
    nearest_rail = np.minimum(np.abs(lower), np.abs(upper))
    rail_approach = center_distance - np.abs(position_mean_3 - np.float32(0.5))
    compact = np.column_stack(
        (native, center_distance, nearest_rail, rail_approach)
    ).astype(np.float32, copy=False)
    if compact.shape[1] != len(COMPACT_FEATURES):
        raise AssertionError("compact feature matrix does not match its contract")

    dropped = tuple(
        dict.fromkeys(
            (
                *base.dropped_features,
                *(
                    name
                    for name in base.tabular_features
                    if name not in NATIVE_COMPACT_FEATURES
                ),
            )
        )
    )
    return LargeMoveDecisionDataset(
        decisions=base.decisions.copy(),
        tabular=compact,
        tabular_features=COMPACT_FEATURES,
        dropped_features=dropped,
        feature_set="compact_volatility_timing_v1",
    )


__all__ = [
    "COMPACT_FEATURES",
    "DERIVED_COMPACT_FEATURES",
    "NATIVE_COMPACT_FEATURES",
    "build_compact_dataset",
]
