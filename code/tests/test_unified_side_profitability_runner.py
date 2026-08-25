from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import experiments.run_unified_side_profitability_ensemble as module
from experiments.run_unified_side_profitability_ensemble import (
    DevelopmentRun,
    _checkpoint_identity,
    freeze_protocol,
    run_experiment,
)


def test_protocol_freezes_four_roles_targets_grid_and_lockbox():
    protocol = freeze_protocol()

    assert protocol["targets"] == ["long_profitable", "short_profitable"]
    assert protocol["roles"] == [
        "fit",
        "probability_calibration",
        "policy_selection",
        "test",
    ]
    assert len(protocol["policy_grid"]) == 9
    assert protocol["feature_count"] == 64
    assert protocol["lockbox_2026_q2_used"] is False
    assert protocol["development_end_exclusive"] == "2025-01-01T00:00:00+00:00"


def test_checkpoint_identity_binds_protocol_source_manifest_and_outer_keys():
    protocol = freeze_protocol()
    source_audit = {
        "source_manifest_sha256": "source",
        "artifact_hashes": {"decision_dataset.parquet": "decision"},
    }
    first = _checkpoint_identity(protocol, source_audit, "manifest", ["a", "b"])
    assert first == _checkpoint_identity(
        protocol, source_audit, "manifest", ["a", "b"]
    )
    assert first != _checkpoint_identity(
        protocol, source_audit, "manifest", ["b", "a"]
    )
    changed = dict(protocol, protocol_sha256="changed")
    assert first != _checkpoint_identity(changed, source_audit, "manifest", ["a", "b"])


def test_h1_loader_is_unreachable_after_development_failure(monkeypatch, tmp_path):
    failed = DevelopmentRun(
        summary={
            "decision": "development_fail",
            "development_pass": False,
            "trades": 86,
        },
        ledger=pd.DataFrame(),
        fold_selections=[],
        source_audit={"maximum_decision_time": "2024-12-31T23:45:00+00:00"},
        work_dir=tmp_path,
    )
    monkeypatch.setattr(module, "run_development", lambda *args, **kwargs: failed)
    monkeypatch.setattr(module, "verify_frozen_union", lambda stage: object())

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("H1 loader was reached after a development failure")

    monkeypatch.setattr(module, "load_bounded_sources", forbidden_loader)
    summary = run_experiment(root=tmp_path)

    assert summary["decision"] == "development_fail_keep_union_v1"
    assert summary["h1_loaded"] is False
    assert summary["forward_loaded"] is False
    assert summary["lockbox_2026_q2_used"] is False


def test_h1_failure_never_opens_forward_loader(monkeypatch, tmp_path):
    development = DevelopmentRun(
        summary={"decision": "development_pass", "development_pass": True},
        ledger=pd.DataFrame(),
        fold_selections=[],
        source_audit={"maximum_decision_time": "2024-12-31T23:45:00+00:00"},
        work_dir=tmp_path,
    )
    h1 = module.StageRun(
        stage="h1",
        summary={"gate_passed": False},
        ledger=pd.DataFrame(),
        source_audit={"m15": {"max_timestamp": "2025-06-30T23:30:00+00:00"}},
        work_dir=tmp_path,
        gate_passed=False,
    )
    monkeypatch.setattr(module, "run_development", lambda *args, **kwargs: development)
    monkeypatch.setattr(module, "verify_frozen_union", lambda stage: object())
    calls = []

    def bounded_loader(start, end, paths):
        calls.append((pd.Timestamp(start), pd.Timestamp(end)))
        if len(calls) > 1:
            raise AssertionError("forward loader was reached after an H1 failure")
        return object()

    monkeypatch.setattr(module, "load_bounded_sources", bounded_loader)
    monkeypatch.setattr(module, "run_walk_forward", lambda *args, **kwargs: h1)

    summary = run_experiment(root=tmp_path)

    assert len(calls) == 1
    assert summary["h1_loaded"] is True
    assert summary["forward_loaded"] is False
    assert summary["decision"] == "h1_fail_keep_union_v1"


def test_protocol_hash_changes_when_model_configuration_changes():
    base = module.UnifiedModelConfig()
    changed = replace(base, xgb_depth=base.xgb_depth + 1)

    assert freeze_protocol(base)["protocol_sha256"] != freeze_protocol(changed)[
        "protocol_sha256"
    ]


def test_completed_run_pins_reconciled_development_result_and_sealed_stages():
    root = module.CACHE
    if not (root / "summary.json").is_file():
        pytest.skip("completed local 04e run is not present")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    development = summary["development"]
    ledger = pd.read_parquet(root / "development_trade_ledger.parquet")

    assert summary["decision"] == "development_fail_keep_union_v1"
    assert development["decision"] == "development_fail"
    assert development["trades"] == len(ledger) == 336
    assert development["long_trades"] == 123
    assert development["short_trades"] == 213
    assert development["net_return"] == pytest.approx(-0.3362259502052362)
    assert development["gross_return"] == pytest.approx(-0.00022595020523596965)
    assert development["trades_per_observed_day"] == pytest.approx(
        1.1389830508474577
    )
    assert development["total_positive_folds"] == 1
    assert development["long_positive_folds"] == 3
    assert development["short_positive_folds"] == 2
    assert development["fold_policy_selections_passed"] == 0
    assert development["xgb_solo_trades"] == 75
    assert development["xgb_solo_folds"] == 5
    assert development["xgb_solo_positive_folds"] == 3
    assert development["xgb_solo_net_return"] == pytest.approx(
        -0.10782963280650523
    )
    assert development["development_pass"] is False

    trade_net = pd.to_numeric(ledger["net_return"], errors="raise")
    trade_gross = pd.to_numeric(ledger["gross_return"], errors="raise")
    assert float(trade_net.sum()) == pytest.approx(development["net_return"])
    assert float((trade_gross - trade_net).sum()) == pytest.approx(
        len(ledger) * 0.001
    )
    entry = pd.to_datetime(ledger["entry_time"], utc=True).reset_index(drop=True)
    exit_time = pd.to_datetime(ledger["actual_exit_time"], utc=True).reset_index(
        drop=True
    )
    assert not (entry.iloc[1:].to_numpy() <= exit_time.iloc[:-1].to_numpy()).any()

    assert summary["h1_loaded"] is False
    assert summary["forward_loaded"] is False
    assert summary["lockbox_2026_q2_used"] is False
    assert pd.Timestamp(summary["maximum_loaded_timestamp"]) == pd.Timestamp(
        "2024-12-31T23:45:00+00:00"
    )
    assert not list(root.glob("h1_*"))
    assert not list(root.glob("forward_*"))
    assert summary["protocol_sha256"] == freeze_protocol()["protocol_sha256"]
    assert np.isfinite(
        pd.read_csv(root / "calibration_metrics.csv")[
            ["calibrated_roc_auc", "calibrated_pr_auc"]
        ].to_numpy(float)
    ).all()
    for filename, expected in manifest["artifact_hashes"].items():
        assert module._sha256_file(root / filename) == expected
    assert module.verify_frozen_union("h1").dependency_hashes == manifest[
        "union_dependency_hashes"
    ]
