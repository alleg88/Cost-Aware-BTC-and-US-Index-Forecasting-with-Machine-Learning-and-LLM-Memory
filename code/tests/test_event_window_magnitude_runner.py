from pathlib import Path

import pandas as pd
import pytest

import experiments.run_event_window_magnitude_timing as runner


def test_protocol_freezes_full_path_five_horizons_and_sealed_periods():
    protocol = runner.protocol_dict()
    assert protocol["time_to_hit_horizons_minutes"] == [5, 15, 30, 60, 120]
    assert protocol["primary_binary_projection"] == "M>=1.0B"
    assert protocol["label_interval"].endswith("independent of hit time")
    assert protocol["direction_head_trained"] is False
    assert protocol["trading_policy_trained"] is False
    assert protocol["economics_evaluated"] is False
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_any_handoff_load(monkeypatch):
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("handoff loader should not be called")

    monkeypatch.setattr(runner, "load_frozen_n_artifacts", fail_if_called)
    with pytest.raises(ValueError, match="development only"):
        runner.run_magnitude_study(stage="forward")
    assert called is False


def test_frozen_n_loader_accepts_only_the_published_n3_handoff():
    frozen = runner.load_frozen_n_artifacts()
    assert frozen.summary["chosen_arm"] == "N3_side_neutral_volatility"
    assert len(frozen.labels) == 171_658
    assert len(frozen.n3_scores) == 128_271
    assert not frozen.labels.duplicated(["window_id", "step"]).any()
    assert not frozen.n3_scores.duplicated(["window_id", "step"]).any()


def test_frozen_n_loader_can_resolve_an_explicit_registered_run(monkeypatch):
    read_json = runner._read_json

    def reject_latest(path):
        if path.name == "latest_dev.json":
            raise AssertionError("an explicit frozen run must not read latest_dev.json")
        return read_json(path)

    monkeypatch.setattr(runner, "_read_json", reject_latest)
    frozen = runner.load_frozen_n_artifacts(
        expected_run_hash="b3b8086a3f156e5b84aa"
    )
    assert frozen.run_hash == "b3b8086a3f156e5b84aa"


def test_event_metric_does_not_turn_adjacent_positive_rows_into_extra_events():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "window_id": [f"w{i}" for i in range(6)],
            "channel_episode_id": ["e0"] * 6,
            "step": range(6),
            "decision_time": [start + pd.Timedelta(minutes=5 * i) for i in range(6)],
            "y_ge_100": [0, 1, 1, 0, 1, 0],
            "p_ge_100": [0.1, 0.9, 0.8, 0.2, 0.7, 0.3],
            "tth_100_min": [float("nan"), 10, 5, float("nan"), 15, float("nan")],
        }
    )
    original = runner.event_alert_metrics(frame, alerts_per_day=3.0)
    duplicate = frame.iloc[[2]].copy()
    duplicate["window_id"] = "duplicate"
    duplicate["step"] = 20
    repeated = runner.event_alert_metrics(
        pd.concat([frame, duplicate], ignore_index=True), alerts_per_day=3.0
    )
    assert original.iloc[0].truth_events == 2
    assert repeated.iloc[0].truth_events == 2
    stable = [
        "selected_alerts",
        "event_recall",
        "alert_clusters",
        "false_alert_clusters",
        "median_lead_time_minutes",
    ]
    assert original.loc[0, stable].equals(repeated.loc[0, stable])


def test_completed_smoke_publishes_tth_and_leakage_artifacts():
    result = runner.run_magnitude_study(smoke=True)
    assert result.summary["forward_or_lockbox_loaded"] is False
    assert "first_minute_1b_hit_share" in result.summary
    assert result.summary["feature_count"] == 248
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert leakage["passed"].astype(bool).all()
    assert (result.run_dir / "tth_audit.csv").is_file()
    assert set(runner.READER_ARTIFACTS).issubset(
        {path.name for path in Path(result.run_dir).iterdir()}
    )
