"""A new U fit must preserve its own frequency throughout the V runner."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from experiments import run_event_window_direction_head as runner
from test_recomputed_channel_handoffs import MODE, _completed_fixture


def _small_u(root):
    rows = []
    for fold in ("2022H1", *runner.SCORED_FOLDS):
        start = pd.Timestamp(f"{fold[:4]}-{'01' if fold[-1] == '1' else '07'}-02", tz="UTC")
        for index in range(4):
            key = f"{fold}-{index}"
            rows.append({
                "arm": runner.FROZEN_TIMING_ARM, "fold_id": fold,
                "activation_key": key, "window_id": key, "channel_episode_id": key,
                "step": index, "decision_time": start + pd.Timedelta(hours=index * 3),
                "threshold": 0.5, "activation_score": 0.7 + index * 0.02,
                "channel_side": "long" if index % 2 == 0 else "short",
                "reference_price": 100.0, "adaptive_barrier_bps": 100.0,
            })
    ledger = pd.DataFrame(rows)
    paths = []
    for index, row in enumerate(rows):
        for direction in ("long", "short"):
            winning = (direction == "long") == (index % 2 == 0)
            gross = 200.0 if winning else -100.0
            paths.append({
                **row, "direction": direction, "path_complete": True, "censored": False,
                "entry_price": 100.0, "exit_price": 102.0 if winning else 99.0,
                "bars_held": 30, "outcome": "tp" if winning else "sl",
                "gross_bps": gross, "gross_r": gross / 100.0,
                "net_bps": gross - 10.0, "net_r": (gross - 10.0) / 100.0,
                "cost_bps": 10.0, "cost_r": 0.1, "target_multiple_b": 2.0,
                "hold_minutes": 120,
            })
    _completed_fixture(root, runner.FROZEN_U_ARTIFACTS,
        summary={"activation_counts": {runner.FROZEN_TIMING_ARM: len(ledger)},
                 "max_loaded_timestamp": str(ledger["decision_time"].max())},
        frames={"activation_ledger.parquet": ledger,
                "oof_predictions.parquet": ledger.assign(p_t_le_60=ledger["activation_score"]),
                "threshold_audit.csv": ledger[["arm", "fold_id", "threshold"]].drop_duplicates(),
                "economic_paths.parquet": pd.DataFrame(paths)})
    return ledger


def test_recomputed_v_runs_all_stages_on_current_u_counts(tmp_path, monkeypatch):
    root = tmp_path / "input"
    ledger = _small_u(root)
    monkeypatch.setenv(MODE, "recomputed")
    result = runner.run_direction_head(frozen_u_root=root, run_root=tmp_path / "output", smoke=True)
    scored = int(ledger["fold_id"].isin(runner.SCORED_FOLDS).sum())
    assert result.summary["source_activations"] == len(ledger) == 28
    assert result.summary["scored_activations"] == scored == 24
    assert result.summary["research_claim"] == "smoke_only_no_claim"
    assert result.summary["forward_or_lockbox_loaded"] is False
    protocol = json.loads((result.run_dir / "protocol.json").read_text())
    assert protocol["expected_source_activations"] == 28
    assert protocol["expected_scored_activations"] == 24
    for name, group in (("combined_policy_ledger.parquet", "model"), ("policy_paths.parquet", "scenario")):
        frame = pd.read_parquet(result.run_dir / name)
        assert frame.groupby(group)["activation_key"].nunique().eq(scored).all()
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert leakage["passed"].all()
    state = json.loads((result.run_dir / "run_state.json").read_text())
    assert state["status"] == "complete"


def test_viability_uses_declared_count_without_weakening_economic_gates():
    values = dict(path_completeness=1.0, scored_activations=24,
                  expected_scored_activations=24, mean_net_r_ci_low=0.01,
                  versus_channel_ci_low=0.01, leakage_passed=True)
    assert runner.economic_viability(**values)
    for field, value in (("scored_activations", 23), ("path_completeness", 0.98),
                         ("mean_net_r_ci_low", 0.0), ("versus_channel_ci_low", 0.0),
                         ("leakage_passed", False)):
        assert not runner.economic_viability(**{**values, field: value})
    values.pop("expected_scored_activations")
    assert not runner.economic_viability(**values)


def test_default_v_native_path_gate_keeps_historical_count(tmp_path, monkeypatch):
    _small_u(tmp_path)
    monkeypatch.setenv(MODE, "recomputed")
    frozen = runner.load_frozen_u_artifacts(tmp_path)
    with pytest.raises(ValueError, match="native paths changed"):
        runner._smoke_primary_paths(frozen)
