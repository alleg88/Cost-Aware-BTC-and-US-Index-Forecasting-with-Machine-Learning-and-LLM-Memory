from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.unified_2021_ensemble_models import MODEL_NAMES, UnifiedModelConfig
from experiments.unified_expected_net_data import (
    EXPECTED_NET_TARGETS,
    attach_expected_net_targets,
    make_expected_net_manifest,
)


def _dataset(rows: int = 1_000, features: int = 6) -> UnifiedDataset:
    rng = np.random.default_rng(23)
    values = rng.normal(size=(rows, features)).astype(np.float32)
    decision_time = pd.date_range(
        "2021-01-01 00:15", periods=rows, freq="15min", tz="UTC"
    )
    long_bps = 9.0 * values[:, 0] + 3.0 * values[:, 1] - 2.0
    short_bps = -8.0 * values[:, 0] + 2.0 * values[:, 2] - 2.0
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{position}" for position in range(rows)],
            "decision_time": decision_time,
            "label_end": decision_time + pd.Timedelta(minutes=121),
            "adaptive_barrier_bps": 100.0,
            "path_complete": True,
            "net_return_long": long_bps / 10_000.0,
            "net_return_short": short_bps / 10_000.0,
        }
    )
    return attach_expected_net_targets(
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


def test_shared_scale_uses_both_sides_and_one_bp_floor():
    from experiments.unified_expected_net_models import shared_target_scale

    assert shared_target_scale(
        np.array([-4.0, 2.0]), np.array([8.0, -6.0])
    ) == pytest.approx(5.0)
    assert shared_target_scale(np.zeros(3), np.zeros(3)) == pytest.approx(1.0)


def test_nonnegative_affine_calibrator_clips_negative_slope_and_handles_constant():
    from experiments.unified_expected_net_models import NonNegativeAffineCalibrator

    decreasing = NonNegativeAffineCalibrator.fit(
        np.array([0.0, 1.0, 2.0]), np.array([2.0, 1.0, 0.0])
    )
    constant = NonNegativeAffineCalibrator.fit(
        np.ones(3), np.array([-1.0, 2.0, 5.0])
    )

    assert decreasing.slope == 0.0
    assert decreasing.intercept == pytest.approx(1.0)
    assert constant.slope == 0.0
    assert constant.intercept == pytest.approx(2.0)
    np.testing.assert_allclose(constant.predict(np.array([8.0, -3.0])), 2.0)


def test_lstm_allows_nonfinite_targets_only_on_unselected_context_rows():
    from experiments.unified_expected_net_models import TwoOutputLSTMRegressor

    rng = np.random.default_rng(9)
    features = rng.normal(size=(24, 3)).astype(np.float32)
    targets = rng.normal(size=(24, 2)).astype(np.float32)
    selected = np.ones(24, dtype=bool)
    selected[:3] = False
    targets[:3] = np.nan
    config = UnifiedModelConfig(
        sequence_length=4,
        lstm_hidden_size=4,
        lstm_epochs=1,
        lstm_batch_size=16,
    )

    fitted = TwoOutputLSTMRegressor(config).fit(
        features, targets, sample_mask=selected
    )

    assert np.isfinite(fitted.predict_raw(features[-4:])).all()


@pytest.fixture(scope="module")
def fitted_fold():
    from experiments.unified_expected_net_models import fit_expected_net_fold

    dataset = _dataset()
    manifest = make_expected_net_manifest(dataset)
    return dataset, manifest, fit_expected_net_fold(
        dataset, manifest, fold_id=0, config=_tiny_config()
    )


def test_fold_fits_three_families_and_emits_six_aligned_predictions(fitted_fold):
    _, _, result = fitted_fold

    expected = {
        f"pred_{side}_{model}"
        for side in ("long", "short")
        for model in MODEL_NAMES
    }
    assert expected.issubset(result.preflight_predictions.columns)
    assert expected.issubset(result.test_predictions.columns)
    assert set(result.fit_audit["model"]) == set(MODEL_NAMES)
    assert set(result.fit_audit["target"]) == set(EXPECTED_NET_TARGETS)
    assert len(result.fit_audit) == 6
    assert set(result.preflight_predictions["source_role"]) == {
        "fixed_policy_preflight"
    }
    assert set(result.test_predictions["source_role"]) == {"test"}


def test_fit_calibration_preflight_and_test_keys_are_disjoint(fitted_fold):
    _, manifest, result = fitted_fold

    assert not result.fit_audit["probability_calibration_overlap"].any()
    assert not result.fit_audit["fixed_policy_preflight_overlap"].any()
    assert not result.fit_audit["test_overlap"].any()
    fold = manifest.loc[manifest["fold_id"].eq(0)]
    expected_preflight = set(
        fold.loc[fold["role"].eq("fixed_policy_preflight"), "row_key"].astype(str)
    )
    expected_test = set(
        fold.loc[fold["role"].eq("test"), "row_key"].astype(str)
    )
    assert set(result.preflight_predictions["row_key"].astype(str)) == expected_preflight
    assert set(result.test_predictions["row_key"].astype(str)) == expected_test
    assert expected_preflight.isdisjoint(expected_test)


def test_calibration_metrics_and_predictions_are_finite(fitted_fold):
    _, _, result = fitted_fold

    assert len(result.calibration_metrics) == 6
    assert {
        "raw_mae_bps",
        "calibrated_mae_bps",
        "raw_rmse_bps",
        "calibrated_rmse_bps",
        "slope",
        "intercept_bps",
    }.issubset(result.calibration_metrics.columns)
    numeric = result.calibration_metrics[
        [
            "raw_mae_bps",
            "calibrated_mae_bps",
            "raw_rmse_bps",
            "calibrated_rmse_bps",
            "slope",
            "intercept_bps",
        ]
    ].to_numpy(float)
    assert np.isfinite(numeric).all()
    assert (result.calibration_metrics["slope"] >= 0.0).all()
    predictions = result.test_predictions.filter(regex=r"^pred_(long|short)_")
    assert np.isfinite(predictions.to_numpy(float)).all()
    assert result.target_scale_bps >= 1.0
