from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.unified_2021_ensemble_models import MODEL_NAMES, UnifiedModelConfig
from experiments.unified_side_profitability_data import (
    PROFITABILITY_HEADS,
    attach_profitability_targets,
    make_four_role_manifest,
)
from experiments.unified_side_profitability_models import fit_profitability_fold


def _dataset(rows: int = 1_000, features: int = 6) -> UnifiedDataset:
    rng = np.random.default_rng(17)
    decision_time = pd.date_range(
        "2021-01-01 00:15", periods=rows, freq="15min", tz="UTC"
    )
    values = rng.normal(size=(rows, features)).astype(np.float32)
    long_positive = values[:, 0] + 0.4 * values[:, 1] > 0.35
    short_positive = (~long_positive) & (values[:, 0] < -0.35)
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{position}" for position in range(rows)],
            "decision_time": decision_time,
            "label_end": decision_time + pd.Timedelta(minutes=121),
            "adaptive_barrier_bps": 100.0,
            "path_complete": True,
            "net_r_long": np.where(long_positive, 0.8, -0.4),
            "net_r_short": np.where(short_positive, 0.7, -0.3),
        }
    )
    return attach_profitability_targets(
        UnifiedDataset(
            decisions=decisions,
            tabular=values,
            sequences=np.empty((0, 0, 0), dtype=np.float32),
            feature_names=tuple(f"feature_{index}" for index in range(features)),
            economic_paths=pd.DataFrame(),
        )
    )


def _tiny_config() -> UnifiedModelConfig:
    return UnifiedModelConfig(
        sequence_length=4,
        lstm_hidden_size=4,
        lstm_epochs=1,
        lstm_batch_size=128,
        xgb_estimators=4,
        xgb_depth=2,
        xgb_min_child_weight=1.0,
        n_jobs=1,
    )


def test_fold_fits_all_six_heads_and_scores_both_later_roles():
    dataset = _dataset()
    manifest = make_four_role_manifest(dataset)

    result = fit_profitability_fold(dataset, manifest, 0, _tiny_config())

    assert set(result.fit_audit["head"]) == set(PROFITABILITY_HEADS)
    assert set(result.fit_audit["model"]) == set(MODEL_NAMES)
    assert len(result.fit_audit) == 6
    expected_probability_columns = {
        "p_long_xgboost",
        "p_long_lstm",
        "p_long_svm_linear",
        "p_short_xgboost",
        "p_short_lstm",
        "p_short_svm_linear",
    }
    assert expected_probability_columns.issubset(result.policy_predictions.columns)
    assert expected_probability_columns.issubset(result.test_predictions.columns)
    assert set(result.policy_predictions["source_role"]) == {"policy_selection"}
    assert set(result.test_predictions["source_role"]) == {"test"}


def test_fit_calibration_policy_and_test_keys_remain_disjoint():
    dataset = _dataset()
    manifest = make_four_role_manifest(dataset)

    result = fit_profitability_fold(dataset, manifest, 0, _tiny_config())

    assert not result.fit_audit["probability_calibration_overlap"].any()
    assert not result.fit_audit["policy_selection_overlap"].any()
    assert not result.fit_audit["test_overlap"].any()
    fold = manifest.loc[manifest["fold_id"].eq(0)]
    expected_policy = set(
        fold.loc[fold["role"].eq("policy_selection"), "row_key"].astype(str)
    )
    expected_test = set(fold.loc[fold["role"].eq("test"), "row_key"].astype(str))
    assert set(result.policy_predictions["row_key"].astype(str)) == expected_policy
    assert set(result.test_predictions["row_key"].astype(str)) == expected_test
    assert expected_policy.isdisjoint(expected_test)


def test_calibration_metrics_cover_each_model_and_side_with_finite_probabilities():
    dataset = _dataset()
    manifest = make_four_role_manifest(dataset)

    result = fit_profitability_fold(dataset, manifest, 0, _tiny_config())

    assert len(result.calibration_metrics) == 6
    assert set(result.calibration_metrics["head"]) == set(PROFITABILITY_HEADS)
    assert set(result.calibration_metrics["model"]) == set(MODEL_NAMES)
    assert {
        "raw_roc_auc",
        "calibrated_roc_auc",
        "raw_pr_auc",
        "calibrated_pr_auc",
    }.issubset(result.calibration_metrics.columns)
    assert np.isfinite(
        result.calibration_metrics[
            ["calibrated_roc_auc", "calibrated_pr_auc"]
        ].to_numpy(float)
    ).all()
    probability = result.test_predictions.filter(regex=r"^p_(long|short)_")
    assert np.isfinite(probability.to_numpy(float)).all()
    assert probability.ge(0.0).all().all()
    assert probability.le(1.0).all().all()
