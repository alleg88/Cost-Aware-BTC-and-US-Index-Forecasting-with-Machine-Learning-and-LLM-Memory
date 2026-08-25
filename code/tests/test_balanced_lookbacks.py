import pandas as pd

from experiments.run_balanced_lookbacks import (
    LOOKBACK_DAYS,
    evaluation_quarter_masks,
    lookback_cache_path,
)
from experiments.run_walkforward_arms import cache_path as arm_cache_path


def test_lookback_cache_paths_reuse_frozen_180_day_baseline():
    assert LOOKBACK_DAYS == (60, 90, 180)
    assert lookback_cache_path(180) == arm_cache_path("econ", 40)
    assert lookback_cache_path(60).name == (
        "btc_bothofpos_cb-econ_dz40_lb60d_to2026.parquet"
    )
    assert lookback_cache_path(90).name == (
        "btc_bothofpos_cb-econ_dz40_lb90d_to2026.parquet"
    )


def test_evaluation_quarters_stop_before_q2_lockbox():
    index = pd.DatetimeIndex(
        [
            "2025-07-01T00:00:00Z",
            "2025-10-01T00:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-04-01T00:00:00Z",
        ]
    )

    masks = evaluation_quarter_masks(index)

    assert list(masks) == ["2025Q3", "2025Q4", "2026Q1"]
    assert [int(mask.sum()) for mask in masks.values()] == [1, 1, 1]
    assert not any(mask[-1] for mask in masks.values())
