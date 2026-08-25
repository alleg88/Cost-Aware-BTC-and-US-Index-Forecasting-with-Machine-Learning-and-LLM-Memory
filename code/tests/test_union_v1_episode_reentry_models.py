from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.union_v1_episode_reentry_data import make_union_reentry_manifest
from experiments.union_v1_episode_reentry_models import (
    UnionReentryModelConfig,
    combine_union_reentry_folds,
    fit_union_reentry_fold,
)


def _tiny_dataset_and_manifest() -> tuple[UnifiedDataset, pd.DataFrame]:
    rows = 500
    position = np.arange(rows, dtype=float)
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{index:04d}" for index in range(rows)],
            "decision_time": pd.date_range(
                "2021-01-01 00:15:00", periods=rows, freq="15min", tz="UTC"
            ),
            "union_target_time": pd.date_range(
                "2021-01-01 00:30:00", periods=rows, freq="15min", tz="UTC"
            ),
            "target_dz55": (np.arange(rows) % 3).astype(np.int8),
            "target_dz75": ((np.arange(rows) + 1) % 3).astype(np.int8),
        }
    )
    tabular = np.column_stack(
        [
            np.sin(position / 7.0),
            np.cos(position / 11.0),
            (position % 17.0) / 17.0,
            np.where(np.arange(rows) % 13 == 0, np.nan, position / rows),
        ]
    ).astype(np.float32)
    dataset = UnifiedDataset(
        decisions=decisions,
        tabular=tabular,
        sequences=np.empty((0, 0, 0), dtype=np.float32),
        feature_names=("sin", "cos", "cycle", "trend"),
        economic_paths=pd.DataFrame(),
    )
    return dataset, make_union_reentry_manifest(dataset)


def _tiny_config() -> UnionReentryModelConfig:
    return UnionReentryModelConfig(
        lstm_sequence_length=8,
        lstm_hidden_size=4,
        lstm_epochs=1,
        lstm_batch_size=64,
    )


def test_config_freezes_only_union_member_budgets() -> None:
    config = UnionReentryModelConfig()

    assert config.lstm_sequence_length == 32
    assert config.lstm_hidden_size == 64
    assert config.lstm_epochs == 10
    assert config.lstm_tau == 0.75
    assert config.svm_c == 0.1
    assert config.svm_tau == 0.0
    assert not hasattr(config, "xgb_estimators")


def test_fold_predictions_share_exact_test_keys_and_classes() -> None:
    dataset, manifest = _tiny_dataset_and_manifest()
    result = fit_union_reentry_fold(dataset, manifest, 0, _tiny_config())
    expected = manifest.loc[
        manifest["fold_id"].eq(0) & manifest["role"].eq("test"), "row_key"
    ].tolist()

    assert result.predictions["row_key"].tolist() == expected
    assert result.predictions["row_key"].is_unique
    assert result.predictions[["pred_lstm", "pred_svm_linear"]].notna().all().all()
    assert set(result.predictions["pred_lstm"]).issubset({0, 1, 2})
    assert set(result.predictions["pred_svm_linear"]).issubset({0, 1, 2})
    probabilities = result.predictions[
        ["p_short_lstm", "p_flat_lstm", "p_long_lstm"]
    ].to_numpy(float)
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_lstm_windows_do_not_cross_fold_block_start() -> None:
    dataset, manifest = _tiny_dataset_and_manifest()
    result = fit_union_reentry_fold(dataset, manifest, 0, _tiny_config())

    assert (
        result.training_audit["lstm_context_min_position"]
        >= result.training_audit["block_start"]
    )


def test_same_seed_reproduces_lstm_probabilities_exactly() -> None:
    dataset, manifest = _tiny_dataset_and_manifest()
    left = fit_union_reentry_fold(dataset, manifest, 0, _tiny_config())
    right = fit_union_reentry_fold(dataset, manifest, 0, _tiny_config())

    np.testing.assert_allclose(
        left.predictions[["p_short_lstm", "p_flat_lstm", "p_long_lstm"]],
        right.predictions[["p_short_lstm", "p_flat_lstm", "p_long_lstm"]],
        rtol=0.0,
        atol=0.0,
    )


def test_combiner_rejects_duplicate_oof_keys() -> None:
    dataset, manifest = _tiny_dataset_and_manifest()
    result = fit_union_reentry_fold(dataset, manifest, 0, _tiny_config())

    try:
        combine_union_reentry_folds([result, result])
    except AssertionError as error:
        assert "duplicate" in str(error).lower()
    else:
        raise AssertionError("duplicate OOF keys must fail closed")
