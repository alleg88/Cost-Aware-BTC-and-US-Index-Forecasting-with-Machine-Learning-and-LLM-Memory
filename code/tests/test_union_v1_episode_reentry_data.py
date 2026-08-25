from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.union_v1_episode_reentry_data import (
    attach_union_dead_zone_targets,
    load_frozen_union_reentry_dataset,
    make_union_reentry_manifest,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
FROZEN_04D = CODE_ROOT / "experiments" / "cache" / "unified_2021_ensemble"


def _dataset(decisions: pd.DataFrame) -> UnifiedDataset:
    rows = len(decisions)
    return UnifiedDataset(
        decisions=decisions,
        tabular=np.zeros((rows, 1), dtype=np.float32),
        sequences=np.empty((0, 0, 0), dtype=np.float32),
        feature_names=("feature",),
        economic_paths=pd.DataFrame(),
    )


def test_dead_zone_targets_use_exact_next_close_and_censor_gaps() -> None:
    decisions = pd.DataFrame(
        {
            "row_key": ["a", "b", "c", "d"],
            "decision_time": pd.to_datetime(
                [
                    "2021-01-01 00:15:00+00:00",
                    "2021-01-01 00:30:00+00:00",
                    "2021-01-01 00:45:00+00:00",
                    "2021-01-01 01:15:00+00:00",
                ],
                utc=True,
            ),
            "close": [100.0, 100.6, 99.8, 101.0],
            "outcome_long": ["timeout", "stop_loss", "take_profit", "timeout"],
        }
    )

    output = attach_union_dead_zone_targets(_dataset(decisions)).decisions

    assert output["target_dz55"].tolist() == [2, 0, -1, -1]
    assert output["target_dz75"].tolist() == [1, 0, -1, -1]
    assert output.loc[0, "union_target_time"] == pd.Timestamp(
        "2021-01-01 00:30:00", tz="UTC"
    )
    assert output.loc[0, "target_dz55"] != output.loc[0, "outcome_long"]


def test_manifest_has_five_non_overlapping_80_20_blocks_with_eight_bar_embargo() -> None:
    rows = 500
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{index:04d}" for index in range(rows)],
            "decision_time": pd.date_range(
                "2021-01-01 00:15:00", periods=rows, freq="15min", tz="UTC"
            ),
            "close": 100.0 + np.sin(np.arange(rows) / 7.0),
        }
    )
    dataset = attach_union_dead_zone_targets(_dataset(decisions))

    manifest = make_union_reentry_manifest(dataset)

    assert sorted(manifest["fold_id"].unique()) == [0, 1, 2, 3, 4]
    for _, fold in manifest.groupby("fold_id", sort=True):
        fit = fold.loc[fold["role"].eq("fit")]
        test = fold.loc[fold["role"].eq("test")]
        assert len(fit) > 0
        assert len(test) > 0
        assert int(fit["position"].max()) + 8 < int(test["position"].min())
        assert fit["union_target_time"].max() < test["decision_time"].min()
    test_keys = manifest.loc[manifest["role"].eq("test"), "row_key"]
    assert test_keys.is_unique


def test_manifest_censors_the_tail_and_non_contiguous_target() -> None:
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{index:03d}" for index in range(500)],
            "decision_time": pd.date_range(
                "2021-01-01 00:15:00", periods=500, freq="15min", tz="UTC"
            ),
            "close": np.linspace(100.0, 105.0, 500),
        }
    )
    decisions.loc[90:, "decision_time"] += pd.Timedelta(minutes=15)
    dataset = attach_union_dead_zone_targets(_dataset(decisions))

    manifest = make_union_reentry_manifest(dataset)

    gap_key = decisions.loc[89, "row_key"]
    gap_rows = manifest.loc[manifest["row_key"].eq(gap_key)]
    assert len(gap_rows) == 1
    assert gap_rows.iloc[0]["role"] == "target_censored"


def test_frozen_loader_hash_verifies_04d_and_stops_before_2025() -> None:
    dataset, manifest, audit = load_frozen_union_reentry_dataset(FROZEN_04D)

    assert dataset.decisions["decision_time"].min() >= pd.Timestamp(
        "2021-01-01", tz="UTC"
    )
    assert dataset.decisions["decision_time"].max() < pd.Timestamp(
        "2025-01-01", tz="UTC"
    )
    assert sorted(manifest["fold_id"].unique()) == [0, 1, 2, 3, 4]
    assert audit["feature_count"] == 64
    assert audit["lockbox_2026_q2_used"] is False
