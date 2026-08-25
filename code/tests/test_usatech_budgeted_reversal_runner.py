from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path

import pandas as pd

import experiments.run_usa500_budgeted_reversal_agent as usa500_runner


CODE_ROOT = Path(__file__).parents[1]
SOURCE = CODE_ROOT / "experiments" / "cache" / "usatech_reflection_weight_agent"
LIVE_ROOT = CODE_ROOT / "experiments" / "cache" / "usatech_budgeted_reversal_agent"


def test_usatech_budgeted_reversal_runner_is_registered():
    assert importlib.util.find_spec(
        "experiments.run_usatech_budgeted_reversal_agent"
    ) is not None


def _runner():
    return importlib.import_module("experiments.run_usatech_budgeted_reversal_agent")


def test_prepare_binds_usatech_test_a_and_restores_usa500_module(tmp_path: Path):
    runner = _runner()
    usa500_hash_before = usa500_runner._implementation_hash()

    manifest = runner.prepare_common_artifacts(
        source_root=SOURCE,
        output_root=tmp_path / "usatech_budgeted_reversal_agent",
    )

    assert manifest["stage_counts"] == {"h1": 366, "forward": 23}
    assert manifest["q2_loaded"] is False
    assert manifest["implementation_hash"] == runner._implementation_hash()
    assert manifest["implementation_hash"] != usa500_hash_before
    assert usa500_runner._implementation_hash() == usa500_hash_before
    copied = pd.read_parquet(
        tmp_path
        / "usatech_budgeted_reversal_agent"
        / "common"
        / "forward_opportunities.parquet"
    )
    assert len(copied) == 23
    assert copied["cost_return"].eq(0.0003).all()
    assert pd.to_datetime(copied["signal_bar_open"], utc=True).max() < pd.Timestamp(
        "2026-04-01T00:00:00Z"
    )


def test_completed_reversal_audit_uses_three_bps_when_live_cache_exists():
    if not (LIVE_ROOT / "manifest.json").is_file():
        return
    runner = _runner()

    summary = runner.finalize_results(output_root=LIVE_ROOT)
    audit = pd.read_parquet(LIVE_ROOT / "leakage_audit.parquet").set_index("check_id")

    assert summary["all_integrity_checks_pass"] is True
    assert bool(audit.loc["immutable_execution_replay", "passed"]) is True
    assert "3 bps" in str(audit.loc["immutable_execution_replay", "detail"])
