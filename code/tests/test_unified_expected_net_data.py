from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import UnifiedDataset


def _dataset(rows: int = 1_000, features: int = 4) -> UnifiedDataset:
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
            "net_return_long": np.resize(np.array([0.0020, -0.0015]), rows),
            "net_return_short": np.resize(np.array([-0.0030, 0.0010]), rows),
        }
    )
    return UnifiedDataset(
        decisions=decisions,
        tabular=np.zeros((rows, features), dtype=np.float32),
        sequences=np.empty((0, 0, 0), dtype=np.float32),
        feature_names=tuple(f"feature_{index}" for index in range(features)),
        economic_paths=pd.DataFrame(),
    )


def test_expected_net_targets_are_existing_after_cost_returns_in_bps():
    from experiments.unified_expected_net_data import attach_expected_net_targets

    dataset = _dataset(3)
    decisions = dataset.decisions.copy()
    decisions.loc[2, "path_complete"] = False
    decisions.loc[2, ["net_return_long", "net_return_short"]] = np.nan
    dataset = UnifiedDataset(
        decisions,
        dataset.tabular,
        dataset.sequences,
        dataset.feature_names,
        dataset.economic_paths,
    )

    output = attach_expected_net_targets(dataset)

    np.testing.assert_allclose(
        output.decisions.loc[:1, "target_long_bps"], [20.0, -15.0]
    )
    np.testing.assert_allclose(
        output.decisions.loc[:1, "target_short_bps"], [-30.0, 10.0]
    )
    assert output.decisions.loc[2, "target_long_bps"] != output.decisions.loc[
        2, "target_long_bps"
    ]
    assert output.decisions.loc[2, "target_short_bps"] != output.decisions.loc[
        2, "target_short_bps"
    ]


def test_expected_net_manifest_preserves_four_role_purges_and_renames_preflight():
    from experiments.unified_expected_net_data import (
        attach_expected_net_targets,
        make_expected_net_manifest,
    )

    manifest = make_expected_net_manifest(attach_expected_net_targets(_dataset()))

    required = (
        "fit",
        "probability_calibration",
        "fixed_policy_preflight",
        "test",
    )
    assert set(manifest["fold_id"]) == {0, 1, 2, 3, 4}
    assert "policy_selection" not in set(manifest["role"])
    assert not manifest.loc[manifest["role"].eq("test"), "row_key"].duplicated().any()
    for _, fold in manifest.groupby("fold_id", sort=True):
        assert set(required).issubset(set(fold["role"]))
        ordered = [fold.loc[fold["role"].eq(role)] for role in required]
        for earlier, later in zip(ordered, ordered[1:]):
            assert earlier["label_end"].max() < later["decision_time"].min()
            assert set(earlier["row_key"]).isdisjoint(later["row_key"])


def test_expected_net_loader_reuses_verified_frozen_loader(monkeypatch, tmp_path):
    import experiments.unified_expected_net_data as module

    source = _dataset(12)
    calls = []

    def verified_loader(root):
        calls.append(root)
        return source, {"verified_artifacts": 2, "lockbox_2026_q2_used": False}

    monkeypatch.setattr(module, "load_frozen_development_dataset", verified_loader)

    loaded, audit = module.load_frozen_expected_net_dataset(tmp_path)

    assert calls == [tmp_path]
    assert {"target_long_bps", "target_short_bps"}.issubset(
        loaded.decisions.columns
    )
    assert audit["target_units"] == "after_cost_basis_points"
    assert audit["cost_subtracted_again"] is False
