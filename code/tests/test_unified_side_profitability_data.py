from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.unified_2021_ensemble_data import UNIFIED_FEATURES, UnifiedDataset
from experiments.unified_side_profitability_data import (
    attach_profitability_targets,
    load_frozen_development_dataset,
    make_four_role_manifest,
)


def _dataset(rows: int = 1_000) -> UnifiedDataset:
    decision_time = pd.date_range(
        "2021-01-01 00:15", periods=rows, freq="15min", tz="UTC"
    )
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{position}" for position in range(rows)],
            "decision_time": decision_time,
            "label_end": decision_time + pd.Timedelta(minutes=121),
            "adaptive_barrier_bps": 100.0,
            "path_complete": True,
            "net_r_long": np.resize(np.array([0.5, -0.2, -0.1]), rows),
            "net_r_short": np.resize(np.array([-0.4, 0.3, -0.2]), rows),
        }
    )
    tabular = np.arange(rows * len(UNIFIED_FEATURES), dtype=np.float32).reshape(
        rows, len(UNIFIED_FEATURES)
    )
    paths = pd.DataFrame(
        {
            "row_key": np.repeat(decisions["row_key"].to_numpy(), 2),
            "direction": np.tile(["long", "short"], rows),
            "path_complete": True,
        }
    )
    return UnifiedDataset(
        decisions=decisions,
        tabular=tabular,
        sequences=np.empty((0, 0, 0), dtype=np.float32),
        feature_names=UNIFIED_FEATURES,
        economic_paths=paths,
    )


def test_direct_targets_use_completed_net_r_paths_and_allow_wait():
    output = attach_profitability_targets(_dataset(3))

    assert output.decisions["long_profitable"].tolist() == [1, 0, 0]
    assert output.decisions["short_profitable"].tolist() == [0, 1, 0]
    assert not (
        output.decisions["long_profitable"].eq(1)
        & output.decisions["short_profitable"].eq(1)
    ).any()


def test_incomplete_paths_are_ineligible_for_both_targets():
    dataset = _dataset(3)
    decisions = dataset.decisions.copy()
    decisions.loc[0, ["path_complete", "net_r_long", "net_r_short"]] = [
        False,
        0.8,
        0.7,
    ]
    dataset = UnifiedDataset(
        decisions,
        dataset.tabular,
        dataset.sequences,
        dataset.feature_names,
        dataset.economic_paths,
    )

    output = attach_profitability_targets(dataset)

    assert output.decisions.loc[0, "long_profitable"] == 0
    assert output.decisions.loc[0, "short_profitable"] == 0


def test_four_roles_are_disjoint_embargoed_and_label_end_purged():
    manifest = make_four_role_manifest(attach_profitability_targets(_dataset()))

    required = {"fit", "probability_calibration", "policy_selection", "test"}
    assert set(manifest["fold_id"]) == {0, 1, 2, 3, 4}
    assert not manifest.loc[manifest["role"].eq("test"), "row_key"].duplicated().any()
    for _, fold in manifest.groupby("fold_id", sort=True):
        assert required.issubset(set(fold["role"]))
        fit = fold.loc[fold["role"].eq("fit")]
        calibration = fold.loc[fold["role"].eq("probability_calibration")]
        policy = fold.loc[fold["role"].eq("policy_selection")]
        test = fold.loc[fold["role"].eq("test")]
        assert fit["label_end"].max() < calibration["decision_time"].min()
        assert calibration["label_end"].max() < policy["decision_time"].min()
        assert policy["label_end"].max() < test["decision_time"].min()
        assert set(fit["row_key"]).isdisjoint(calibration["row_key"])
        assert set(calibration["row_key"]).isdisjoint(policy["row_key"])
        assert set(policy["row_key"]).isdisjoint(test["row_key"])


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_loader_verifies_hashes_and_reconstructs_feature_order(tmp_path):
    source = _dataset(12)
    decisions = source.decisions.copy()
    for position, name in enumerate(UNIFIED_FEATURES):
        decisions[name] = source.tabular[:, position]
    decisions_path = tmp_path / "decision_dataset.parquet"
    paths_path = tmp_path / "economic_labels.parquet"
    decisions.to_parquet(decisions_path, index=False)
    source.economic_paths.to_parquet(paths_path, index=False)
    manifest = {
        "artifact_hashes": {
            decisions_path.name: _sha256(decisions_path),
            paths_path.name: _sha256(paths_path),
        },
        "maximum_loaded_timestamp": "2024-12-31T23:59:00+00:00",
        "lockbox_2026_q2_used": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    loaded, audit = load_frozen_development_dataset(tmp_path)

    np.testing.assert_array_equal(loaded.tabular, source.tabular)
    assert loaded.feature_names == UNIFIED_FEATURES
    assert audit["verified_artifacts"] == 2

    decisions_path.write_bytes(decisions_path.read_bytes() + b"drift")
    with pytest.raises(AssertionError, match="hash mismatch"):
        load_frozen_development_dataset(tmp_path)
