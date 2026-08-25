from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.run_lstm_gmadl_shadow as module


def _base_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_key": ["a", "b"],
            "fold_id": [0, 0],
            "decision_time": pd.date_range("2024-01-01", periods=2, freq="15min", tz="UTC"),
            "pred_long_xgboost": [1.0, 2.0],
            "pred_short_xgboost": [-1.0, -2.0],
            "pred_long_lstm": [3.0, 4.0],
            "pred_short_lstm": [-3.0, -4.0],
            "pred_long_svm_linear": [5.0, 6.0],
            "pred_short_svm_linear": [-5.0, -6.0],
            "raw_long_lstm": [0.3, 0.4],
            "raw_short_lstm": [-0.3, -0.4],
        }
    )


def test_shadow_protocol_is_one_seed_development_only_replacement():
    protocol = module.freeze_protocol()

    assert protocol["control_loss"] == "mse_long_plus_mse_short"
    assert protocol["candidate_loss"] == "control_plus_0.25_gmadl"
    assert protocol["gmadl"] == {"alpha": 1.0, "beta": 1.0, "lambda": 0.25}
    assert protocol["seeds"] == [42]
    assert protocol["lstm_role"] == "replacement_not_fourth_vote"
    assert protocol["reuse_04f_xgboost_svm"] is True
    assert protocol["h1_access_allowed"] is False
    assert protocol["forward_access_allowed"] is False
    assert protocol["lockbox_2026_q2_used"] is False


def test_replacement_changes_only_one_lstm_pair_and_preserves_other_models():
    base = _base_predictions()
    replacement = pd.DataFrame(
        {
            "row_key": ["a", "b"],
            "raw_long_lstm": [7.0, 8.0],
            "raw_short_lstm": [-7.0, -8.0],
            "pred_long_lstm": [9.0, 10.0],
            "pred_short_lstm": [-9.0, -10.0],
        }
    )

    output = module.replace_lstm_predictions(base, replacement, arm="candidate")

    for model in ("xgboost", "svm_linear"):
        for side in ("long", "short"):
            column = f"pred_{side}_{model}"
            np.testing.assert_array_equal(output[column], base[column])
    np.testing.assert_array_equal(output["pred_long_lstm"], [9.0, 10.0])
    assert output["lstm_arm"].eq("candidate").all()
    assert not any(column.startswith("pred_long_lstm_") for column in output.columns)


def test_runner_source_contains_no_later_stage_or_non_lstm_refit_path():
    source = inspect.getsource(module).lower()

    for forbidden in (
        "load_bounded_sources",
        "run_walk_forward",
        "xgbregressor",
        "linearsvr",
        "fit_expected_net_fold",
    ):
        assert forbidden not in source


def test_manifest_validation_uses_exact_dataset_keys_not_roundtrip_dtypes():
    decisions = pd.DataFrame({"row_key": ["a", "b", "c"]})
    dataset = SimpleNamespace(decisions=decisions)
    manifest = pd.DataFrame(
        {
            "fold_id": [0, 0, 0],
            "position": [0, 1, 2],
            "row_key": ["a", "b", "c"],
            "role": ["fit", "fixed_policy_preflight", "test"],
            "policy_selection_start_position": pd.Series([1, 1, 1], dtype="int64"),
        }
    )

    assert module._manifest_matches_dataset(dataset, manifest)
    manifest.loc[2, "row_key"] = "drift"
    assert not module._manifest_matches_dataset(dataset, manifest)


def test_completed_shadow_reconciles_if_local_artifacts_exist():
    root = module.CACHE
    if not (root / "summary.json").is_file():
        pytest.skip("completed local 04g run is not present")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    control = pd.read_parquet(root / "control_trade_ledger.parquet")
    candidate = pd.read_parquet(root / "candidate_trade_ledger.parquet")
    training = pd.read_csv(root / "paired_training_audit.csv")

    assert summary["shadow_only"] is True
    assert summary["h1_loaded"] is False
    assert summary["forward_loaded"] is False
    assert summary["lockbox_2026_q2_used"] is False
    assert summary["control"]["trades"] == len(control)
    assert summary["candidate"]["trades"] == len(candidate)
    assert summary["control"]["xgboost_solo_trades"] == 0
    assert summary["candidate"]["xgboost_solo_trades"] == 0
    assert training["initial_state_match"].astype(bool).all()
    assert training["batch_order_match"].astype(bool).all()
    assert training["control_matches_04f"].astype(bool).all()
    for filename, expected in manifest["artifact_hashes"].items():
        assert module._sha256_file(root / filename) == expected
