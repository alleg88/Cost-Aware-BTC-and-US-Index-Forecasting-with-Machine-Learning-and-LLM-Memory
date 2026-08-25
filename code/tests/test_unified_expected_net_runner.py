from __future__ import annotations

import json
from dataclasses import replace

import pandas as pd
import pytest

import experiments.run_unified_expected_net_ensemble as module
from experiments.run_unified_expected_net_ensemble import (
    DevelopmentRun,
    _checkpoint_identity,
    freeze_protocol,
    run_experiment,
)


def test_protocol_freezes_expected_net_models_policy_roles_and_lockbox():
    protocol = freeze_protocol()

    assert protocol["targets"] == ["target_long_bps", "target_short_bps"]
    assert protocol["roles"] == [
        "fit",
        "probability_calibration",
        "fixed_policy_preflight",
        "test",
    ]
    assert protocol["consensus"] == "two_of_three_positive_expected_net"
    assert protocol["policy_grid"] == []
    assert protocol["xgboost_solo_allowed"] is False
    assert protocol["target_scale"]["centering"] is False
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
    assert first != _checkpoint_identity(
        changed, source_audit, "manifest", ["a", "b"]
    )


def test_h1_loader_is_unreachable_after_development_failure(monkeypatch, tmp_path):
    failed = DevelopmentRun(
        summary={"decision": "development_fail", "development_pass": False},
        ledger=pd.DataFrame(),
        source_audit={"maximum_decision_time": "2024-12-31T23:45:00+00:00"},
        work_dir=tmp_path,
    )
    monkeypatch.setattr(module, "run_development", lambda *args, **kwargs: failed)
    monkeypatch.setattr(module, "verify_frozen_union", lambda stage: object())

    def forbidden_loader(*args, **kwargs):
        raise AssertionError("H1 loader was reached after development failure")

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
            raise AssertionError("forward loader was reached after H1 failure")
        return object()

    monkeypatch.setattr(module, "load_bounded_sources", bounded_loader)
    monkeypatch.setattr(module, "run_walk_forward", lambda *args, **kwargs: h1)

    summary = run_experiment(root=tmp_path)

    assert len(calls) == 1
    assert summary["h1_loaded"] is True
    assert summary["forward_loaded"] is False
    assert summary["decision"] == "h1_fail_keep_union_v1"


def test_protocol_hash_changes_with_model_configuration():
    base = module.UnifiedModelConfig()
    changed = replace(base, xgb_depth=base.xgb_depth + 1)

    assert freeze_protocol(base)["protocol_sha256"] != freeze_protocol(changed)[
        "protocol_sha256"
    ]


def test_ledger_reconciliation_uses_exact_native_path_not_barrier_bps():
    ledger = pd.DataFrame(
        {
            "row_key": ["row-1"],
            "direction": ["short"],
            "gross_return": [0.0357412432],
            "net_return": [0.0347412432],
            "gross_bps": [363.956005],
            "net_bps": [353.956005],
            "cost_bps": [10.0],
            "path_signature": ["signature"],
        }
    )
    source = ledger.copy()

    assert module._ledger_path_reconciliation(ledger, source)
    source.loc[0, "net_return"] += 0.0001
    assert not module._ledger_path_reconciliation(ledger, source)


def test_completed_run_reconciles_and_preserves_stage_boundaries():
    root = module.CACHE
    if not (root / "summary.json").is_file():
        pytest.skip("completed local 04f run is not present")
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    development = summary["development"]
    ledger = pd.read_parquet(root / "development_trade_ledger.parquet")

    assert development["trades"] == len(ledger)
    assert development["xgboost_solo_trades"] == 0
    assert development["required_side_trades"] == max(
        15, int(__import__("math").ceil(0.20 * len(ledger)))
    )
    assert float(ledger["net_return"].sum()) == pytest.approx(
        development["net_return"]
    )
    assert float((ledger["gross_return"] - ledger["net_return"]).sum()) == pytest.approx(
        len(ledger) * 0.001
    )
    assert not ledger["route"].eq("xgboost_solo").any()
    if len(ledger) > 1:
        entry = pd.to_datetime(ledger["entry_time"], utc=True).reset_index(drop=True)
        exit_time = pd.to_datetime(
            ledger["actual_exit_time"], utc=True
        ).reset_index(drop=True)
        assert not (
            entry.iloc[1:].to_numpy() <= exit_time.iloc[:-1].to_numpy()
        ).any()
    assert summary["lockbox_2026_q2_used"] is False
    assert pd.Timestamp(summary["maximum_loaded_timestamp"]) < module.LOCKBOX_START
    assert summary["protocol_sha256"] == freeze_protocol()["protocol_sha256"]
    for filename, expected in manifest["artifact_hashes"].items():
        assert module._sha256_file(root / filename) == expected
