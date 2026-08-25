from pathlib import Path

import pandas as pd
import pytest

from experiments import run_event_window_timing_policy_repair as runner


def test_protocol_freezes_r_cell_and_excludes_feature_training():
    protocol = runner.protocol_dict()
    assert protocol["notebook"] == "R_event_window_timing_policy_repair"
    assert protocol["target_activations_per_day"] == 3.0
    assert protocol["target_multiple_b"] == 2.0
    assert protocol["hold_minutes"] == 120
    assert protocol["timing_score"] == "p_t_le_60"
    assert protocol["policies"] == ["crossing", "level_rearm"]
    assert protocol["timing_model_refit"] is False
    assert protocol["new_features_added"] is False
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_any_handoff_loading(monkeypatch):
    monkeypatch.setattr(
        runner,
        "load_frozen_q_handoff",
        lambda *_args, **_kwargs: pytest.fail("loaded frozen Q"),
    )
    monkeypatch.setattr(
        runner,
        "load_frozen_p_artifacts",
        lambda *_args, **_kwargs: pytest.fail("loaded frozen P"),
    )
    with pytest.raises(ValueError, match="development only"):
        runner.run_timing_policy_repair(stage="forward")


def test_exact_frozen_q_handoff_is_validated():
    frozen = runner.load_frozen_q_handoff()
    assert frozen.run_hash == runner.FROZEN_Q_RUN_HASH
    assert frozen.summary["forward_or_lockbox_loaded"] is False


def test_frozen_crossing_and_same_threshold_rearm_supply_are_reproduced():
    frozen = runner.load_frozen_p_artifacts()
    ledger, thresholds, supply = runner.reconstruct_policy_ledger(frozen)

    xgb_crossing = ledger.loc[
        ledger["arm"].eq("xgboost_conditional_crossing")
    ]
    assert len(xgb_crossing) == 3_780

    xgb_supply = supply.loc[
        supply["arm"].eq("xgboost_conditional")
    ].iloc[0]
    assert int(xgb_supply["crossing_activations"]) == 3_780
    assert int(xgb_supply["level_rearm_same_threshold_activations"]) == 7_418
    assert float(xgb_supply["same_threshold_rearm_per_day"]) > 5.0

    assert set(thresholds["threshold_source"]) == {
        "frozen_notebook_p",
        "past_only_policy_calibration",
    }
    assert thresholds["calibration_precedes_outer"].astype(bool).all()

    matched = ledger.loc[ledger["policy"].eq("level_rearm")]
    days = runner.calendar_days(ledger)
    rates = matched.groupby("arm").size().div(days)
    assert rates.between(2.5, 3.5).all()


def test_reader_artifacts_are_tabular_except_metadata_json():
    json_names = {name for name in runner.READER_ARTIFACTS if name.endswith(".json")}
    assert json_names == {"protocol.json", "frozen_protocol.json", "summary.json"}
    assert "activation_ledger.parquet" in runner.READER_ARTIFACTS
    assert "economic_metrics.csv" in runner.READER_ARTIFACTS


def test_paired_policy_bootstrap_resamples_complete_episode_blocks():
    scenarios = pd.DataFrame(
        {
            "arm": ["candidate", "candidate", "control", "control"],
            "channel_episode_id": ["e1", "e2", "e1", "e2"],
            "scenario": ["direction_70"] * 4,
            "censored": [False] * 4,
            "net_r": [1.0, -0.5, 0.2, -1.0],
        }
    )
    result = runner.paired_policy_bootstrap(
        scenarios,
        comparisons=(("candidate", "control", "policy"),),
        calendar_days=2,
        draws=100,
        seed=42,
    )
    row = result.iloc[0]
    assert row["candidate_arm"] == "candidate"
    assert row["control_arm"] == "control"
    assert row["comparison"] == "policy"
    assert int(row["union_episodes"]) == 2
    assert float(row["delta_mean_net_r"]) == pytest.approx(0.65)
    assert float(row["delta_net_r_per_day"]) == pytest.approx(0.65)


def test_completed_smoke_publishes_hashed_economics_and_leakage(tmp_path):
    result = runner.run_timing_policy_repair(smoke=True, run_root=tmp_path)
    assert result.summary["timing_model_refit"] is False
    assert result.summary["new_features_added"] is False
    assert result.summary["forward_or_lockbox_loaded"] is False
    assert result.summary["economic_protocol"] == "RR2_120m"

    published = {path.name for path in result.run_dir.iterdir()}
    assert set(runner.READER_ARTIFACTS).issubset(published)
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert len(leakage) >= 12
    assert leakage["passed"].astype(bool).all()
    metrics = pd.read_csv(result.run_dir / "economic_metrics.csv")
    assert set(metrics["arm"]) == {
        "xgboost_conditional_crossing",
        "xgboost_conditional_level_rearm",
        "logreg_conditional_crossing",
        "logreg_conditional_level_rearm",
        "anchored_empirical_crossing",
        "anchored_empirical_level_rearm",
    }
    assert set(metrics["target_multiple_b"]) == {2.0}
    assert set(metrics["hold_minutes"]) == {120}

    state = runner._read_json(result.run_dir / "run_state.json")
    assert state["status"] == "complete"
    assert state["summary"] == result.summary


def test_runner_default_paths_remain_inside_code_tree():
    assert runner.CODE_ROOT in Path(runner.RUN_ROOT).parents
    assert runner.CODE_ROOT in Path(runner.FROZEN_P_ROOT).parents
    assert runner.CODE_ROOT in Path(runner.FROZEN_Q_ROOT).parents
