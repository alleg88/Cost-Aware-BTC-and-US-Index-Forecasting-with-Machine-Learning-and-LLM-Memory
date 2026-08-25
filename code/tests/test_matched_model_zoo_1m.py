from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from experiments.catboost_execution_resolution import execution_policy_fingerprint
from experiments.catboost_execution_runner import ExecutionResolutionRunner
from experiments.run_catboost_matched_ablation import (
    prediction_cache_fingerprint,
    stage_prediction_cache_fingerprint,
)


def test_matched_model_zoo_protocol_module_exists():
    assert importlib.util.find_spec("experiments.matched_model_zoo_1m") is not None


def test_candidate_manifests_are_deterministic_and_model_specific():
    from experiments.matched_model_zoo_1m import candidate_manifest

    first = candidate_manifest("logreg")
    assert len(first["candidates"]) == 15
    assert first == candidate_manifest("logreg")
    assert first["fingerprint"] != candidate_manifest("svm_linear")["fingerprint"]


def test_expected_full_counts_match_frozen_protocol():
    from experiments.matched_model_zoo_1m import expected_counts

    assert expected_counts() == {
        "classification_2024.parquet": 45,
        "economic_policy_grid_2024.parquet": 2_970,
        "economic_candidate_winners_2024.parquet": 45,
        "selected_candidates_2024.parquet": 3,
        "calibration_policy_grid_2025h1.parquet": 198,
        "selected_policies_2025h1.parquet": 3,
        "forward_monthly.parquet": 27,
        "forward_quarterly.parquet": 9,
        "forward_summary.parquet": 3,
    }


def test_full_artifact_validation_rejects_missing_files(tmp_path: Path):
    from experiments.matched_model_zoo_1m import validate_model_artifacts

    with pytest.raises(FileNotFoundError, match="classification_2024.parquet"):
        validate_model_artifacts(tmp_path, model_name="logreg")


def test_full_artifact_validation_rejects_wrong_row_count(tmp_path: Path):
    from experiments.matched_model_zoo_1m import validate_model_artifacts

    pd.DataFrame({"value": [1]}).to_parquet(
        tmp_path / "classification_2024.parquet"
    )
    with pytest.raises(ValueError, match="classification_2024.parquet.*45"):
        validate_model_artifacts(tmp_path, model_name="logreg")


def _fingerprint_inputs():
    index = pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC")
    X = pd.DataFrame({"feature": range(len(index))}, index=index)
    y = pd.Series([0, 1, 2] * 4, index=index, name="label")
    regimes = pd.Series(
        ["bull", "sideways", "bear"] * 4, index=index, name="regime"
    )
    fold = {
        "fold_id": 0,
        "train_start": index[0],
        "train_end": index[5],
        "test_start": index[6],
        "test_end": index[-1] + pd.Timedelta(minutes=15),
    }
    return X, y, regimes, fold


def test_model_identity_isolates_prediction_fingerprints():
    X, y, regimes, fold = _fingerprint_inputs()
    kwargs = {
        "width_bps": 55,
        "candidate_params": {"C": 0.1},
        "fold_metadata": fold,
        "X": X,
        "y": y,
        "regimes": regimes,
        "data_fingerprint": "m15",
    }

    assert prediction_cache_fingerprint(
        **kwargs, model_name="logreg"
    ) != prediction_cache_fingerprint(**kwargs, model_name="svm_linear")


def test_model_identity_isolates_frozen_and_policy_fingerprints():
    X, y, regimes, _ = _fingerprint_inputs()
    frozen = {
        "stage": "frozen_post_selection",
        "width_bps": 55,
        "candidate_params": {"C": 0.1},
        "X": X,
        "y": y,
        "regimes": regimes,
        "train_end": X.index[6],
        "test_start": X.index[6],
        "test_end": X.index[-1] + pd.Timedelta(minutes=15),
        "data_fingerprint": "m15",
    }
    policy = {
        "stage": "selection",
        "width_bps": 55,
        "candidate_id": 1,
        "prediction_fingerprints": ["prediction"],
        "m15_fingerprint": "m15",
        "resolution": "1m",
        "execution_data_fingerprint": "minute",
        "fee_bps": 5.0,
    }

    assert stage_prediction_cache_fingerprint(
        **frozen, model_name="logreg"
    ) != stage_prediction_cache_fingerprint(**frozen, model_name="svm_linear")
    assert execution_policy_fingerprint(
        **policy, model_name="logreg"
    ) != execution_policy_fingerprint(**policy, model_name="svm_linear")


def test_explicit_none_preserves_legacy_catboost_fingerprints():
    X, y, regimes, fold = _fingerprint_inputs()
    kwargs = {
        "width_bps": 55,
        "candidate_params": {"depth": 6},
        "fold_metadata": fold,
        "X": X,
        "y": y,
        "regimes": regimes,
        "data_fingerprint": "m15",
    }

    assert prediction_cache_fingerprint(**kwargs) == prediction_cache_fingerprint(
        **kwargs, model_name=None
    )


def test_execution_runner_records_optional_model_identity(tmp_path: Path):
    runner = ExecutionResolutionRunner(
        output_root=tmp_path / "output",
        prediction_root=tmp_path / "predictions",
        store=object(),
        prepared=object(),
        candidates=({},),
        model_name="logreg",
    )

    assert runner.model_name == "logreg"


def test_completed_full_study_reconciles_all_frozen_artifacts():
    from experiments.matched_model_zoo_1m import validate_full_study

    report = validate_full_study()
    assert report == {
        "models": 8,
        "selected_candidates": 24,
        "selected_policies": 24,
        "forward_monthly": 216,
        "forward_quarterly": 72,
        "forward_summary": 24,
    }
