"""Stage guards, identity and resume tests for the Notebook B runner."""

from dataclasses import FrozenInstanceError
import json

import pandas as pd
import pytest

import experiments.run_channel_window_ml as runner
from experiments.run_channel_window_ml import (
    LoadedInputs,
    WindowMLConfig,
    run_experiment,
    select_dev_arm,
)


def _loaded(max_time: str = "2025-06-30 23:59") -> LoadedInputs:
    idx = pd.DatetimeIndex([pd.Timestamp(max_time, tz="UTC")])
    empty = pd.DataFrame(index=idx)
    return LoadedInputs(
        minute=empty,
        five_minute=empty,
        fifteen_minute=empty,
        hourly=empty,
        positioning=empty,
        max_loaded_timestamp=idx.max(),
        input_fingerprint="synthetic-input",
    )


def _fake_pipeline(config, loaded, store, decision_start, stage_end):
    assert (store.run_dir / "protocol_manifest.json").exists()
    return {
        "stage": config.stage,
        "windows": 0,
        "decision_start": decision_start.isoformat(),
        "stage_end": stage_end.isoformat(),
    }


def test_dev_runner_never_accepts_rows_at_or_after_2025_07_01(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "_execute_dev_pipeline", _fake_pipeline)

    result = run_experiment(
        WindowMLConfig(stage="dev"), out_root=tmp_path,
        loaded_inputs=_loaded(),
    )

    assert result.manifest["max_loaded_timestamp"] < "2025-07-01T00:00:00+00:00"
    with pytest.raises(AssertionError, match="stage boundary"):
        run_experiment(
            WindowMLConfig(stage="dev"), out_root=tmp_path,
            loaded_inputs=_loaded("2025-07-01 00:00"), resume=False,
        )


def test_forward_requires_a_frozen_protocol_hash(tmp_path):
    with pytest.raises(PermissionError, match="frozen protocol"):
        run_experiment(WindowMLConfig(stage="forward"), out_root=tmp_path)


def test_tune_requires_a_frozen_dev_protocol_hash(tmp_path):
    with pytest.raises(PermissionError, match="frozen dev protocol"):
        run_experiment(WindowMLConfig(stage="tune"), out_root=tmp_path)


def test_lockbox_is_not_exposed_by_this_runner(tmp_path):
    with pytest.raises(PermissionError, match="sealed"):
        run_experiment(WindowMLConfig(stage="lockbox"), out_root=tmp_path)


def test_config_is_immutable_and_smoke_does_not_change_run_hash():
    config = WindowMLConfig()
    with pytest.raises(FrozenInstanceError):
        config.zone = 0.25

    full = runner.run_identity(config, "input", "source")
    smoke = runner.run_identity(config, "input", "source")
    assert full == smoke


def test_complete_matching_run_resumes_without_reexecution(monkeypatch, tmp_path):
    calls = {"count": 0}

    def counted(config, loaded, store, decision_start, stage_end):
        calls["count"] += 1
        return _fake_pipeline(config, loaded, store, decision_start, stage_end)

    monkeypatch.setattr(runner, "_execute_dev_pipeline", counted)
    config = WindowMLConfig(stage="dev")
    first = run_experiment(config, out_root=tmp_path, loaded_inputs=_loaded())
    second = run_experiment(config, out_root=tmp_path, loaded_inputs=_loaded())

    assert calls["count"] == 1
    assert second.summary["resumed"] is True
    state = json.loads((first.run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"
    assert state["run_hash"] == first.manifest["run_hash"]


def test_dev_arm_selection_uses_primary_capacity_and_support_floor():
    policy = pd.DataFrame(
        {
            "cadence": ["1min", "5min", "5min", "1min"],
            "architecture": ["pooled", "pooled", "separate", "separate"],
            "model_kind": ["catboost_regressor"] * 4,
            "capacity": [3, 3, 3, 5],
            "threshold": [0.10, 0.20, 0.30, 0.40],
            "threshold_quantile": [0.9, 0.9, 0.9, 0.9],
            "mean_r_net": [0.02, 0.03, 0.20, 0.50],
            "total_net_r": [2.0, 3.0, 20.0, 50.0],
            "filled_trades": [80, 70, 20, 100],
            "long_fills": [40, 35, 10, 50],
            "short_fills": [40, 35, 10, 50],
        }
    )

    selected = select_dev_arm(policy)

    assert selected is not None
    assert selected["cadence"] == "5min"
    assert selected["architecture"] == "pooled"
    assert selected["primary_capacity"] == 3


def test_dev_arm_selection_records_no_promotion_without_supported_economics():
    policy = pd.DataFrame(
        {
            "cadence": ["5min"],
            "architecture": ["pooled"],
            "model_kind": ["catboost_regressor"],
            "capacity": [3],
            "threshold": [0.20],
            "threshold_quantile": [0.9],
            "mean_r_net": [-0.01],
            "total_net_r": [-1.0],
            "filled_trades": [100],
            "long_fills": [50],
            "short_fills": [50],
        }
    )

    assert select_dev_arm(policy) is None


def test_dev_arm_selection_can_require_one_calendar_trade_per_day():
    policy = pd.DataFrame(
        {
            "cadence": ["5min", "5min"],
            "architecture": ["pooled", "pooled"],
            "model_kind": ["catboost_regressor", "catboost_regressor"],
            "capacity": [3, 3],
            "threshold": [0.2, 0.1],
            "threshold_quantile": [0.9, 0.7],
            "mean_r_net": [0.10, 0.03],
            "total_net_r": [10.0, 30.0],
            "filled_trades": [100, 1000],
            "long_fills": [50, 500],
            "short_fills": [50, 500],
            "trades_per_day": [0.2, 1.1],
        }
    )

    selected = select_dev_arm(policy, min_trades_per_day=1.0)

    assert selected is not None
    assert selected["dev_diagnostic_quantile"] == 0.7
    assert selected["dev_trades_per_day"] == 1.1


def test_dev_policy_compares_logreg_and_catboost_when_continuation_is_allowed():
    logreg = object()
    catboost = object()

    assert runner._policy_score_results(logreg, catboost) == (logreg, catboost)
    assert runner._policy_score_results(logreg, None) == (logreg,)


def test_tune_dispatches_only_with_the_matching_frozen_protocol(monkeypatch, tmp_path):
    source_hash = "frozen-source"
    protocol_hash = runner.protocol_identity(WindowMLConfig(stage="dev"), source_hash)
    freeze_dir = tmp_path / "frozen_protocols"
    freeze_dir.mkdir()
    (freeze_dir / f"{protocol_hash}.json").write_text("{}", encoding="utf-8")
    calls = {"count": 0}

    def fake_tune(config, loaded, store, decision_start, stage_end, out_root):
        calls["count"] += 1
        assert decision_start == pd.Timestamp("2025-07-01", tz="UTC")
        assert stage_end == pd.Timestamp("2025-10-01", tz="UTC")
        assert out_root == tmp_path
        return {"stage": "tune", "threshold_promoted": True}

    monkeypatch.setattr(runner, "source_version", lambda: source_hash)
    monkeypatch.setattr(runner, "_execute_tune_pipeline", fake_tune)
    result = run_experiment(
        WindowMLConfig(stage="tune", protocol_hash=protocol_hash),
        out_root=tmp_path,
        loaded_inputs=_loaded("2025-09-30 23:59"),
    )

    assert calls["count"] == 1
    assert result.summary["stage"] == "tune"
