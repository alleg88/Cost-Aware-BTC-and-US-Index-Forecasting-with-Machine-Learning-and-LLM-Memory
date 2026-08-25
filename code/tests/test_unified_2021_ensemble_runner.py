from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import experiments.run_unified_2021_ensemble as runner
from experiments.run_unified_2021_ensemble import (
    DevelopmentResult,
    SourceBundle,
    SourcePaths,
    StageResult,
    freeze_protocol,
    load_bounded_sources,
    maybe_run_forward,
    maybe_run_h1,
    run_walk_forward,
    verify_frozen_union,
)
from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.unified_2021_ensemble_policy import EnsemblePolicy


def test_bounded_loader_refuses_lockbox_or_later(tmp_path):
    paths = SourcePaths(
        m15=tmp_path / "m15.parquet",
        minute=tmp_path / "minute.parquet",
        positioning=tmp_path / "positioning.parquet",
    )

    with pytest.raises(ValueError, match="Q2-2026"):
        load_bounded_sources(
            pd.Timestamp("2025-07-01", tz="UTC"),
            pd.Timestamp("2026-04-01 00:01", tz="UTC"),
            paths,
        )


def test_bounded_loader_filters_every_source_and_records_identity(tmp_path):
    index_15m = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    index_1m = pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC")
    m15 = pd.DataFrame({"close": np.arange(8.0)}, index=index_15m)
    minute = pd.DataFrame({"close": np.arange(120.0)}, index=index_1m)
    positioning = pd.DataFrame({"funding_rate": np.arange(8.0)}, index=index_15m)
    paths = SourcePaths(
        m15=tmp_path / "m15.parquet",
        minute=tmp_path / "minute.parquet",
        positioning=tmp_path / "positioning.parquet",
    )
    m15.to_parquet(paths.m15)
    minute.to_parquet(paths.minute)
    positioning.to_parquet(paths.positioning)

    bundle = load_bounded_sources(index_15m[1], index_15m[6], paths)

    assert bundle.m15.index.min() >= index_15m[1]
    assert bundle.m15.index.max() < index_15m[6] - pd.Timedelta(minutes=15)
    assert bundle.minute.index.min() >= index_15m[1]
    assert bundle.minute.index.max() < index_15m[6]
    assert set(bundle.source_identities) == {"m15", "minute", "positioning"}
    assert all(len(value["bounded_sha256"]) == 64 for value in bundle.source_identities.values())


def test_union_verifier_checks_manifest_hashes_and_exact_reference(tmp_path):
    artifact = tmp_path / "h1_ledger.parquet"
    artifact.write_bytes(b"immutable-union")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps({"artifact_hashes": {artifact.name: digest}}), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "phase": "h1",
                "trades": 88,
                "net_return": 0.009687734009,
                "sortino": 0.3521029257,
                "max_drawdown": 0.047613386266,
            }
        ]
    ).to_csv(tmp_path / "summary.csv", index=False)

    reference = verify_frozen_union("h1", tmp_path)

    assert reference.summary["trades"] == 88
    assert reference.dependency_hashes[artifact.name] == digest
    artifact.write_bytes(b"changed")
    with pytest.raises(AssertionError, match="hash"):
        verify_frozen_union("h1", tmp_path)


def test_protocol_freeze_is_deterministic_and_declares_54_policies():
    first = freeze_protocol()
    second = freeze_protocol()

    assert first == second
    assert len(first["policy_grid"]) == 54
    assert len(first["protocol_sha256"]) == 64
    assert first["lockbox_start"] == "2026-04-01T00:00:00+00:00"


def test_no_development_policy_makes_h1_loader_unreachable(tmp_path):
    calls: list[str] = []
    development = DevelopmentResult(
        selected_policy=None,
        opportunity_threshold=None,
        summary={"decision": "development_fail_keep_union_v1"},
        work_dir=tmp_path,
        protocol=freeze_protocol(),
    )

    result = maybe_run_h1(development, lambda: calls.append("h1_opened"))

    assert result is None
    assert calls == []


def test_failed_h1_makes_forward_loader_unreachable_and_removes_stale_files(tmp_path):
    stale = tmp_path / "forward_predictions.parquet"
    stale.touch()
    calls: list[str] = []
    h1 = StageResult(
        stage="h1",
        summary={"trades": 0},
        work_dir=tmp_path,
        selected_policy=EnsemblePolicy(2, 0.6, 0.6, 0.75),
        opportunity_threshold=0.7,
    )

    result = maybe_run_forward(h1, lambda: calls.append("forward_opened"), tmp_path)

    assert result is None
    assert calls == []
    assert not stale.exists()


def test_monthly_snapshot_uses_only_prior_label_endpoints(monkeypatch, tmp_path):
    decision_time = pd.date_range(
        "2024-12-01", "2025-06-30 23:45", freq="15min", tz="UTC"
    )
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{position}" for position in range(len(decision_time))],
            "decision_time": decision_time,
            "entry_time": decision_time + pd.Timedelta(minutes=1),
            "label_end": decision_time + pd.Timedelta(minutes=121),
            "path_complete": True,
            "opportunity": 1,
            "side": np.where(np.arange(len(decision_time)) % 2, "long", "short"),
            "side_eligible": True,
            "actual_exit_time_long": decision_time + pd.Timedelta(minutes=10),
            "actual_exit_time_short": decision_time + pd.Timedelta(minutes=10),
        }
    )
    dataset = UnifiedDataset(
        decisions=decisions,
        tabular=np.zeros((len(decisions), 2), dtype=np.float32),
        sequences=np.zeros((len(decisions), 2, 2), dtype=np.float32),
        feature_names=("a", "b"),
        economic_paths=pd.DataFrame(
            columns=["row_key", "direction", "entry_time", "actual_exit_time"]
        ),
    )
    m15_index = pd.date_range("2025-01-01", "2025-06-30 23:45", freq="15min", tz="UTC")
    bundle = SourceBundle(
        m15=pd.DataFrame({"close": 100.0}, index=m15_index),
        minute=pd.DataFrame(),
        positioning=pd.DataFrame(),
        start=pd.Timestamp("2021-01-01", tz="UTC"),
        end=pd.Timestamp("2025-07-01", tz="UTC"),
        source_identities={},
    )
    monkeypatch.setattr(runner, "_dataset_from_bundle", lambda *_: dataset)

    def fake_snapshot(_dataset, cutoff, _config):
        cutoff = pd.Timestamp(cutoff)
        calibration_time = cutoff - pd.Timedelta(days=2)
        calibration = pd.DataFrame(
            {
                "decision_time": [calibration_time],
                "p_opportunity_xgboost": [0.2],
                "p_opportunity_lstm": [0.2],
                "p_opportunity_svm_linear": [0.2],
            }
        )
        return SimpleNamespace(
            fit_max_label_end=cutoff - pd.Timedelta(days=3),
            calibration_start=cutoff - pd.Timedelta(days=2),
            calibration_max_label_end=cutoff - pd.Timedelta(minutes=1),
            calibration_predictions=calibration,
            fit_audit=pd.DataFrame(),
        )

    def fake_score(_snapshot, dataset, positions):
        output = dataset.decisions.iloc[list(positions)].copy()
        output["fold_id"] = -1
        for model in ("xgboost", "lstm", "svm_linear"):
            output[f"p_opportunity_{model}"] = 0.2
            output[f"p_long_{model}"] = 0.5
        return output

    monkeypatch.setattr(runner, "fit_historical_snapshot", fake_snapshot)
    monkeypatch.setattr(runner, "score_historical_snapshot", fake_score)
    monkeypatch.setattr(
        runner,
        "apply_policy",
        lambda predictions, calibration, policy, opportunity_threshold=None: (
            predictions.iloc[0:0].assign(selected_side=pd.Series(dtype="string")),
            predictions.assign(raw_crossing=False, decision_reason="not_rearmed"),
        ),
    )
    monkeypatch.setattr(
        runner,
        "replay_selected_paths",
        lambda *_: pd.DataFrame(
            columns=[
                "direction",
                "entry_time",
                "actual_exit_time",
                "entry_price",
                "exit_price",
                "gross_return",
                "net_return",
            ]
        ),
    )

    result = run_walk_forward(
        "h1",
        bundle,
        EnsemblePolicy(2, 0.6, 0.6, 0.75),
        0.7,
        tmp_path,
    )

    assert len(result.refit_audit) == 6
    assert (
        result.refit_audit["fit_max_label_end"]
        < result.refit_audit["scored_month_start"]
    ).all()
    assert (
        result.refit_audit["calibration_max_label_end"]
        < result.refit_audit["scored_month_start"]
    ).all()
    assert (
        result.refit_audit["calibration_max_decision_time"]
        < result.refit_audit["scored_month_start"]
    ).all()


def test_completed_run_hashes_reconcile_and_lockbox_is_sealed():
    cache = runner.CACHE
    if not (cache / "summary.json").is_file():
        pytest.skip("completed local 04d run is not present")
    summary = json.loads((cache / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))

    for filename, expected_hash in manifest["artifact_hashes"].items():
        assert runner._sha256_file(cache / filename) == expected_hash
    assert summary["lockbox_2026_q2_used"] is False
    assert pd.Timestamp(summary["max_loaded_timestamp"]) < pd.Timestamp(
        "2026-04-01", tz="UTC"
    )
    assert verify_frozen_union("h1").dependency_hashes == manifest[
        "union_dependency_hashes"
    ]
    if summary["forward_loaded"]:
        assert summary["h1_compatibility_passed"] is True
    else:
        assert not list(cache.glob("forward_*"))
