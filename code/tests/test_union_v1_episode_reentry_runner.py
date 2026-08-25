from __future__ import annotations

import json

import numpy as np
import pandas as pd

import experiments.run_union_v1_episode_reentry as runner
from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.union_v1_episode_reentry_data import make_union_reentry_manifest
from experiments.union_v1_episode_reentry_models import UnionReentryModelConfig


def _tiny_stage_inputs() -> dict[str, object]:
    rows = 500
    index = pd.date_range("2021-01-01", periods=rows, freq="15min", tz="UTC")
    position = np.arange(rows, dtype=float)
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{value:04d}" for value in range(rows)],
            "decision_time": index,
            "union_target_time": index + pd.Timedelta(minutes=15),
            "target_dz55": (np.arange(rows) % 3).astype(np.int8),
            "target_dz75": ((np.arange(rows) + 1) % 3).astype(np.int8),
        }
    )
    tabular = np.column_stack(
        [
            np.sin(position / 7.0),
            np.cos(position / 11.0),
            (position % 17.0) / 17.0,
            position / rows,
        ]
    ).astype(np.float32)
    dataset = UnifiedDataset(
        decisions=decisions,
        tabular=tabular,
        sequences=np.empty((0, 0, 0), dtype=np.float32),
        feature_names=("sin", "cos", "cycle", "trend"),
        economic_paths=pd.DataFrame(),
    )
    m15 = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=index,
    )
    minute_index = pd.date_range(
        index.min(), periods=rows * 15, freq="1min", tz="UTC"
    )
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=minute_index,
    )
    return {
        "dataset": dataset,
        "manifest": make_union_reentry_manifest(dataset),
        "m15": m15,
        "minute": minute,
        "model_config": UnionReentryModelConfig(
            lstm_sequence_length=8,
            lstm_hidden_size=4,
            lstm_epochs=1,
            lstm_batch_size=64,
        ),
        "source_audit": {
            "fixture": {
                "min_timestamp": index.min().isoformat(),
                "max_timestamp": minute_index.max().isoformat(),
                "rows": len(minute),
            }
        },
    }


def _stage(tmp_path, stage: str, passed: bool) -> runner.ReentryStageRun:
    return runner.ReentryStageRun(
        stage=stage,
        summary={
            "decision": f"{stage}_{'pass' if passed else 'fail'}",
            "maximum_loaded_timestamp": "2024-12-31T23:59:00+00:00",
        },
        gate_passed=passed,
        output_root=tmp_path,
    )


def test_protocol_has_one_candidate_and_no_search_or_xgboost() -> None:
    protocol = runner.freeze_protocol()

    assert protocol["development_start"] == "2021-01-01T00:00:00+00:00"
    assert protocol["n_splits"] == 5
    assert protocol["embargo_bars"] == 8
    assert protocol["candidate_rules"] == [
        "one_earliest_extra_per_same_side_episode"
    ]
    assert protocol["policy_grid"] == []
    assert protocol["models"] == ["lstm_dz55", "svm_linear_dz75"]
    assert protocol["lockbox_2026_q2_used"] is False


def test_checkpoint_identity_changes_with_model_or_scheduler() -> None:
    base = {"data_hash": "fixed", "model_seed": 42, "scheduler": "one_extra"}

    assert runner.fold_checkpoint_identity(base) != runner.fold_checkpoint_identity(
        {**base, "model_seed": 43}
    )
    assert runner.fold_checkpoint_identity(base) != runner.fold_checkpoint_identity(
        {**base, "scheduler": "two_extras"}
    )


def test_development_artifacts_reconcile(tmp_path) -> None:
    run = runner.run_development(output_root=tmp_path, **_tiny_stage_inputs())
    control = pd.read_parquet(tmp_path / "development_control_ledger.parquet")
    extra = pd.read_parquet(tmp_path / "development_reentry_ledger.parquet")
    candidate = pd.read_parquet(tmp_path / "development_candidate_ledger.parquet")

    assert set(control["trade_key"]).issubset(set(candidate["trade_key"]))
    assert len(candidate) == len(control) + len(extra)
    assert np.isclose(
        candidate["net_return"].sum(), run.summary["candidate_net_return"]
    )
    assert np.isclose(
        run.summary["evaluation_days"], run.summary["oof_rows"] / 96.0
    )
    assert not candidate["position_overlap"].any()
    artifact_manifest = json.loads((tmp_path / "development_artifacts.json").read_text())
    assert artifact_manifest["artifact_hashes"]


def test_failed_development_never_calls_h1_loader(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        runner, "run_development", lambda **_: _stage(tmp_path, "development", False)
    )
    monkeypatch.setattr(
        runner,
        "load_h1_sources",
        lambda: (_ for _ in ()).throw(AssertionError("H1 loader was reached")),
    )

    result = runner.run_experiment(output_root=tmp_path)

    assert result["h1_loaded"] is False
    assert result["forward_loaded"] is False


def test_failed_h1_never_calls_forward_loader(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        runner, "run_development", lambda **_: _stage(tmp_path, "development", True)
    )
    monkeypatch.setattr(runner, "run_h1", lambda **_: _stage(tmp_path, "h1", False))
    monkeypatch.setattr(
        runner,
        "load_forward_sources",
        lambda: (_ for _ in ()).throw(AssertionError("forward loader was reached")),
    )

    result = runner.run_experiment(output_root=tmp_path)

    assert result["h1_loaded"] is True
    assert result["forward_loaded"] is False


def test_manifest_binds_union_and_never_reports_q2(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        runner, "run_development", lambda **_: _stage(tmp_path, "development", False)
    )

    summary = runner.run_experiment(output_root=tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())

    assert manifest["union_dependency_hashes"]
    assert manifest["lockbox_2026_q2_used"] is False
    assert pd.Timestamp(summary["maximum_loaded_timestamp"]) < pd.Timestamp(
        "2026-04-01", tz="UTC"
    )
