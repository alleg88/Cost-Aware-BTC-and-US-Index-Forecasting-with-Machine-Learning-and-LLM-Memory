from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from experiments.run_union_v1_episode_reentry import CACHE, LOCKBOX_START, UNION_CACHE
from experiments.union_v1_episode_reentry_policy import build_union_v1_style_signals


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_completed_union_reentry_artifacts_reconcile_independently() -> None:
    if not (CACHE / "summary.json").is_file():
        pytest.skip("completed local Notebook 04h run is not present")
    summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
    development = summary["development"]
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    stage_manifest = json.loads(
        (CACHE / "development_artifacts.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_parquet(CACHE / "development_oof_predictions.parquet")
    signals = pd.read_parquet(CACHE / "development_signals.parquet")
    folds = pd.read_parquet(CACHE / "development_fold_manifest.parquet")
    control = pd.read_parquet(CACHE / "development_control_ledger.parquet")
    reentry = pd.read_parquet(CACHE / "development_reentry_ledger.parquet")
    candidate = pd.read_parquet(CACHE / "development_candidate_ledger.parquet")
    selected = pd.read_parquet(CACHE / "development_selected_reentries.parquet")
    fold_metrics = pd.read_csv(CACHE / "development_fold_metrics.csv")

    for filename, expected in manifest["artifact_hashes"].items():
        assert _sha256(CACHE / filename) == expected
    for filename, expected in stage_manifest["artifact_hashes"].items():
        assert _sha256(CACHE / filename) == expected
    for filename, expected in manifest["union_dependency_hashes"].items():
        assert _sha256(UNION_CACHE / filename) == expected

    assert predictions["row_key"].is_unique
    assert set(predictions["fold_id"]) == set(range(5))
    probability = predictions[
        ["p_short_lstm", "p_flat_lstm", "p_long_lstm"]
    ].to_numpy(float)
    assert np.isfinite(probability).all()
    np.testing.assert_allclose(probability.sum(axis=1), 1.0, atol=1e-6)
    rebuilt = build_union_v1_style_signals(predictions)
    columns = [
        "row_key",
        "lstm_signal",
        "svm_linear_signal",
        "member_conflict",
        "union_signal",
        "episode_id",
    ]
    pd.testing.assert_frame_equal(
        signals[columns].reset_index(drop=True),
        rebuilt[columns].reset_index(drop=True),
        check_dtype=False,
    )

    assert folds["fold_id"].nunique() == 5
    assert folds.loc[folds["role"].eq("test"), "row_key"].is_unique
    assert selected.groupby("episode_id").size().max() <= 1
    assert set(reentry["signal_time"]) == set(selected["signal_time"])
    assert candidate["entry_time"].is_unique
    assert candidate["trade_key"].is_unique
    assert not candidate["position_overlap"].any()
    control_in_candidate = candidate.loc[candidate["route"].eq("union_control")]
    pd.testing.assert_frame_equal(
        control_in_candidate[control.columns].reset_index(drop=True),
        control.reset_index(drop=True),
    )
    assert np.allclose(control["gross_return"] - control["net_return"], 0.001)
    assert np.allclose(reentry["gross_return"] - reentry["net_return"], 0.001)
    assert control["path_complete"].all() and reentry["path_complete"].all()

    assert len(control) == development["control_trades"] == 948
    assert len(reentry) == development["reentry_trades"] == 283
    assert len(candidate) == development["candidate_trades"] == 1231
    assert development["oof_rows"] == len(predictions) == 27_994
    assert development["evaluation_days"] == pytest.approx(len(predictions) / 96.0)
    assert candidate["net_return"].sum() == pytest.approx(
        development["candidate_net_return"]
    )
    assert control["net_return"].sum() == pytest.approx(
        development["control_net_return"]
    )
    assert reentry["net_return"].sum() == pytest.approx(
        development["incremental_net_return"]
    )
    assert fold_metrics["control_net_return"].sum() == pytest.approx(
        control["net_return"].sum()
    )
    assert fold_metrics["incremental_net_return"].sum() == pytest.approx(
        reentry["net_return"].sum()
    )
    assert summary["decision"] == "development_fail_keep_union_v1"
    assert summary["h1_loaded"] is False and summary["forward_loaded"] is False
    assert manifest["lockbox_2026_q2_used"] is False
    assert pd.Timestamp(manifest["maximum_loaded_timestamp"]) < LOCKBOX_START
