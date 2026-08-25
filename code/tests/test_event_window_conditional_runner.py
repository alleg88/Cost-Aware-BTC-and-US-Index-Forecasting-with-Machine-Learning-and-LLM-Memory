from pathlib import Path

import pandas as pd
import pytest

import experiments.run_event_window_conditional_opportunity as runner


def test_protocol_freezes_conditional_hurdle_and_sealed_periods():
    protocol = runner.protocol_dict()
    assert protocol["timing_intervals_minutes"] == [[1, 15], [16, 30], [31, 60], [61, 120]]
    assert protocol["diagnostic_five_minute_timing_only"] is True
    assert protocol["severity_thresholds_b"] == [1.0, 1.5, 2.0]
    assert protocol["frozen_incidence_head"] == "N3_side_neutral_volatility"
    assert protocol["conditional_models"] == ["logreg", "xgboost"]
    assert protocol["primary_activation_target_per_day"] == 2.0
    assert protocol["activation_rate_sensitivities_per_day"] == [1.0, 3.0]
    assert protocol["cooldown_minutes"] == 60
    assert protocol["direction_head_trained"] is False
    assert protocol["economics_evaluated"] is False
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_any_frozen_handoff_load(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("handoff loader should not be called")

    monkeypatch.setattr(runner, "load_frozen_o_artifacts", fail_if_called)
    with pytest.raises(ValueError, match="development only"):
        runner.run_conditional_study(stage="forward")
    assert called is False


def test_frozen_o_loader_accepts_only_the_complete_published_handoff():
    frozen = runner.load_frozen_o_artifacts()
    assert frozen.run_hash == "d07095c0f595826a8c88"
    assert len(frozen.labels) == 171_658
    assert int(frozen.labels["magnitude_target_valid"].sum()) == 171_560
    assert frozen.summary["forward_or_lockbox_loaded"] is False
    assert not frozen.labels.duplicated(["window_id", "step"]).any()


def test_frozen_o_loader_can_resolve_an_explicit_registered_run(monkeypatch):
    read_json = runner._read_json

    def reject_latest(path):
        if path.name == "latest_dev.json":
            raise AssertionError("an explicit frozen run must not read latest_dev.json")
        return read_json(path)

    monkeypatch.setattr(runner, "_read_json", reject_latest)
    frozen = runner.load_frozen_o_artifacts(
        expected_run_hash="73bc742d288a5aabbeb0"
    )
    assert frozen.run_hash == "73bc742d288a5aabbeb0"


def test_completed_smoke_publishes_causal_policy_and_leakage_artifacts(tmp_path):
    result = runner.run_conditional_study(smoke=True, run_root=tmp_path)
    assert result.summary["forward_or_lockbox_loaded"] is False
    assert result.summary["p120_identity_max_abs"] == 0.0
    assert result.summary["timing_monotonic_violations"] == 0
    assert result.summary["severity_monotonic_violations"] == 0
    assert result.summary["policy_is_causal"] is True
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert leakage["passed"].astype(bool).all()
    assert set(runner.READER_ARTIFACTS).issubset(
        {path.name for path in Path(result.run_dir).iterdir()}
    )
