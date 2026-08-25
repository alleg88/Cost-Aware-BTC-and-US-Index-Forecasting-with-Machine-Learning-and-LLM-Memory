from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_conditional_oof import (
    ConditionalOOFConfig,
    run_conditional_fold,
)
from experiments.event_window_cost_aware_oof import _partitions
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments import run_event_window_calendar_ablation as runner


def _small_dataset() -> LargeMoveDecisionDataset:
    decisions = pd.DataFrame(
        {
            "window_id": ["w0", "w0"],
            "step": [0, 1],
            "decision_time": pd.to_datetime(
                ["2024-01-02 12:00Z", "2024-01-02 12:05Z"], utc=True
            ),
        }
    )
    return LargeMoveDecisionDataset(
        decisions=decisions,
        tabular=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        tabular_features=("past_a", "past_b"),
        dropped_features=(),
        feature_set="opportunity_side_neutral_volatility",
    )


def _prediction_frame(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["xgboost", "xgboost"],
            "fold_id": ["2024H1", "2024H1"],
            "window_id": ["w0", "w0"],
            "step": [0, 1],
            "decision_time": pd.to_datetime(
                ["2024-01-02 12:00Z", "2024-01-02 12:05Z"], utc=True
            ),
            "p_t_le_15": values,
            "p_t_le_30": values,
            "p_t_le_60": values,
            "p_t_le_120": [0.4, 0.6],
        }
    )


def _conditional_dataset() -> tuple[LargeMoveDecisionDataset, PurgedFold, np.ndarray]:
    rows = []
    features = []
    start = pd.Timestamp("2021-01-01", tz="UTC")
    rng = np.random.default_rng(41)
    classes = [0, 2, 2, 2, 2, 3, 3, 4]
    times = [np.nan, 5.0, 15.0, 30.0, 60.0, 20.0, 80.0, 45.0]
    for episode in range(30):
        for step, (target_class, time_to_hit) in enumerate(
            zip(classes, times, strict=True)
        ):
            decision = start + pd.Timedelta(days=episode, minutes=5 * step)
            rows.append(
                {
                    "window_id": f"w{episode}",
                    "channel_episode_id": f"e{episode}",
                    "side": "long" if episode % 2 == 0 else "short",
                    "step": step,
                    "decision_time": decision,
                    "label_start": decision,
                    "label_end": decision + pd.Timedelta(minutes=120),
                    "magnitude_class": target_class,
                    "magnitude_ratio": [0.5, 1.2, 1.2, 1.2, 1.2, 1.7, 1.7, 2.3][step],
                    "magnitude_target_valid": True,
                    "model_target_valid": True,
                    "tth_100_min": time_to_hit,
                }
            )
            vector = rng.normal(size=6)
            vector[0] += target_class
            vector[1] += 0.0 if np.isnan(time_to_hit) else time_to_hit / 120.0
            features.append(vector)
    decisions = pd.DataFrame(rows)
    valid = np.arange(28 * len(classes), 30 * len(classes))
    fold = PurgedFold(
        fold_id="test",
        train=np.arange(0, 28 * len(classes)),
        valid=valid,
        train_end=decisions.iloc[valid].decision_time.min(),
        valid_start=decisions.iloc[valid].decision_time.min(),
        valid_end=decisions.iloc[valid].decision_time.max() + pd.Timedelta(days=1),
    )
    return (
        LargeMoveDecisionDataset(
            decisions=decisions,
            tabular=np.asarray(features, dtype=np.float32),
            tabular_features=tuple(f"past_f{index}" for index in range(6)),
            dropped_features=(),
            feature_set="base",
        ),
        fold,
        np.linspace(0.25, 0.75, len(decisions)),
    )


def test_protocol_freezes_paired_calendar_ablation_cell():
    protocol = runner.protocol_dict()

    assert protocol["models"] == ["logreg", "xgboost"]
    assert protocol["feature_sets"] == ["base", "base+calendar_v1"]
    assert protocol["calendar_features"] == list(runner.CALENDAR_FEATURE_COLUMNS)
    assert protocol["timing_score"] == "p_t_le_60"
    assert protocol["alert_policy"] == "level_rearm"
    assert protocol["target_activations_per_day"] == 3.0
    assert protocol["target_multiple_b"] == 2.0
    assert protocol["hold_minutes"] == 120
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_any_handoff_loading(monkeypatch):
    monkeypatch.setattr(
        runner,
        "load_frozen_r_handoff",
        lambda *_args, **_kwargs: pytest.fail("loaded frozen R"),
    )

    with pytest.raises(ValueError, match="development only"):
        runner.run_calendar_ablation(stage="forward")


def test_exact_completed_notebook_r_handoff_is_validated():
    frozen = runner.load_frozen_r_handoff()

    assert frozen.run_hash == runner.FROZEN_R_RUN_HASH
    assert frozen.summary["level_rearm_selected_for_direction_head"] is True
    assert frozen.summary["frozen_p_run_hash"] == runner.FROZEN_P_RUN_HASH
    assert frozen.summary["forward_or_lockbox_loaded"] is False
    assert frozen.frozen["frozen_p_run_hash"] == runner.FROZEN_P_RUN_HASH
    assert frozen.state["status"] == "complete"
    assert set(runner.FROZEN_R_READER_ARTIFACTS).issubset(frozen.state["artifacts"])


def test_paired_datasets_share_rows_and_only_calendar_adds_columns():
    source = _small_dataset()

    base, calendar = runner.build_paired_datasets(source)

    assert base is source
    assert base.decisions[["window_id", "step"]].equals(
        calendar.decisions[["window_id", "step"]]
    )
    assert np.array_equal(base.tabular, calendar.tabular[:, : base.tabular.shape[1]])
    assert calendar.tabular.shape[1] == base.tabular.shape[1] + 6
    assert calendar.tabular_features == (
        *base.tabular_features,
        *runner.CALENDAR_FEATURE_COLUMNS,
    )


def test_paired_datasets_preserve_aligned_missing_base_values():
    source = _small_dataset()
    source.tabular[0, 0] = np.nan

    base, calendar = runner.build_paired_datasets(source)

    assert np.array_equal(
        base.tabular,
        calendar.tabular[:, : base.tabular.shape[1]],
        equal_nan=True,
    )


def test_base_reproduction_audit_is_key_aligned_and_detects_probability_drift():
    frozen = _prediction_frame([0.2, 0.3])
    refit = frozen.iloc[::-1].reset_index(drop=True)

    exact = runner.base_reproduction_audit(refit, frozen, source="outer")
    assert len(exact) == 1
    assert bool(exact.loc[0, "row_identity"])
    assert float(exact.loc[0, "max_abs_difference"]) == 0.0

    changed = refit.copy()
    changed.loc[0, "p_t_le_60"] += 0.01
    drift = runner.base_reproduction_audit(changed, frozen, source="outer")
    assert float(drift.loc[0, "max_abs_difference"]) == pytest.approx(0.01)


def test_predictive_tables_measure_calendar_improvement_on_identical_rows():
    rows = []
    for fold_id in ("2023H1", "2023H2"):
        for model in ("logreg", "xgboost"):
            for feature_set, probabilities in (
                ("base", (0.4, 0.6)),
                ("base+calendar_v1", (0.1, 0.9)),
            ):
                arm = f"{model}_{'calendar' if feature_set.endswith('calendar_v1') else 'base'}"
                for step, (target, probability) in enumerate(
                    zip((0, 1), probabilities, strict=True)
                ):
                    rows.append(
                        {
                            "arm": arm,
                            "model": model,
                            "feature_set": feature_set,
                            "fold_id": fold_id,
                            "window_id": f"{fold_id}-w{step}",
                            "step": step,
                            "decision_time": pd.Timestamp(
                                f"2024-01-0{step + 1} 12:00", tz="UTC"
                            ),
                            "y_t_le_60": target,
                            "p_t_le_60": probability,
                            "sample_weight": 1.0,
                        }
                    )
    metrics, deltas = runner.predictive_metric_tables(pd.DataFrame(rows))

    xgb_overall = metrics.loc[
        metrics["arm"].eq("xgboost_calendar")
        & metrics["fold_id"].eq("overall")
    ].iloc[0]
    xgb_delta = deltas.loc[
        deltas["model"].eq("xgboost") & deltas["fold_id"].eq("overall")
    ].iloc[0]
    assert float(xgb_overall["weighted_brier"]) == pytest.approx(0.01)
    assert float(xgb_delta["brier_improvement"]) == pytest.approx(0.15)
    assert float(xgb_delta["log_loss_improvement"]) > 0.0
    assert int(xgb_delta["nonnegative_brier_folds"]) == 2
    assert int(xgb_delta["total_folds"]) == 2


def test_paired_oof_uses_real_frozen_fold_contract_for_all_four_arms():
    base, fold, p_hit = _conditional_dataset()
    _, calendar = runner.build_paired_datasets(base)
    config = ConditionalOOFConfig(
        model=LargeMoveModelConfig(xgb_estimators=3, logreg_max_iter=500, n_jobs=1)
    )
    frozen_base = {
        model: run_conditional_fold(
            model,
            fold,
            base,
            p_hit_by_position=p_hit,
            config=config,
        )
        for model in ("logreg", "xgboost")
    }

    result = runner.run_paired_oof(
        base,
        calendar,
        folds=(fold,),
        p_hit_by_fold={fold.fold_id: p_hit},
        config=config,
    )

    assert set(result.scores["arm"]) == set(runner.ARM_NAMES)
    assert set(result.calibration_scores["arm"]) == set(runner.ARM_NAMES)
    assert result.fold_audit["episode_overlap"].eq(0).all()
    for model, frozen in frozen_base.items():
        refit = result.scores.loc[
            result.scores["arm"].eq(f"{model}_base")
        ].reset_index(drop=True)
        assert np.allclose(
            refit[["p_t_le_15", "p_t_le_30", "p_t_le_60", "p_t_le_120"]],
            frozen.scores[["p_t_le_15", "p_t_le_30", "p_t_le_60", "p_t_le_120"]],
            rtol=0.0,
            atol=0.0,
        )


def test_frozen_p_hit_is_recovered_by_key_for_reserved_and_outer_rows_only():
    dataset, fold, expected = _conditional_dataset()
    config = ConditionalOOFConfig()
    _, _, reserved = _partitions(dataset.decisions, fold.train, config.fold)
    outer = np.asarray(fold.valid, dtype=np.int64)

    def frame(positions: np.ndarray) -> pd.DataFrame:
        rows = []
        for model in ("logreg", "xgboost"):
            for position in positions:
                decision = dataset.decisions.iloc[position]
                rows.append(
                    {
                        "model": model,
                        "fold_id": fold.fold_id,
                        "window_id": decision.window_id,
                        "step": int(decision.step),
                        "decision_time": decision.decision_time,
                        "p_t_le_120": float(expected[position]),
                    }
                )
        return pd.DataFrame(rows)

    oof = frame(outer)
    calibration = frame(reserved)
    recovered = runner.frozen_p_hit_for_fold(
        dataset.decisions,
        fold,
        frozen_oof=oof,
        frozen_calibration=calibration,
        fold_config=config.fold,
    )

    assert np.array_equal(recovered[reserved], expected[reserved])
    assert np.array_equal(recovered[outer], expected[outer])
    assert np.isnan(recovered[np.setdiff1d(np.arange(len(recovered)), np.r_[reserved, outer])]).all()

    changed = oof.copy()
    changed.loc[changed["model"].eq("logreg"), "p_t_le_120"] += 0.01
    with pytest.raises(ValueError, match="models disagree"):
        runner.frozen_p_hit_for_fold(
            dataset.decisions,
            fold,
            frozen_oof=changed,
            frozen_calibration=calibration,
            fold_config=config.fold,
        )


def test_level_rearm_ledger_calibrates_each_arm_on_earlier_rows():
    outer_rows = []
    calibration_rows = []
    for arm, model, feature_set in runner.ARM_SPECS:
        for step, score in enumerate((0.2, 0.8, 0.8, 0.7)):
            calibration_rows.append(
                {
                    "arm": arm,
                    "model": model,
                    "feature_set": feature_set,
                    "fold_id": "2024H1",
                    "window_id": "cal",
                    "channel_episode_id": f"{arm}-cal",
                    "step": step,
                    "decision_time": pd.Timestamp("2023-12-01", tz="UTC")
                    + pd.Timedelta(minutes=60 * step),
                    "p_t_le_60": score,
                }
            )
        for step, score in enumerate((0.8, 0.8, 0.1, 0.9)):
            outer_rows.append(
                {
                    "arm": arm,
                    "model": model,
                    "feature_set": feature_set,
                    "fold_id": "2024H1",
                    "window_id": "outer",
                    "channel_episode_id": f"{arm}-outer",
                    "step": step,
                    "decision_time": pd.Timestamp("2024-01-01", tz="UTC")
                    + pd.Timedelta(minutes=60 * step),
                    "p_t_le_60": score,
                }
            )

    ledger, audit = runner.build_level_rearm_ledger(
        pd.DataFrame(outer_rows),
        pd.DataFrame(calibration_rows),
        config=runner.CalendarAblationConfig(threshold_grid_size=5),
    )

    assert set(ledger["arm"]) == set(runner.ARM_NAMES)
    assert set(ledger["policy"]) == {"level_rearm"}
    assert set(audit["arm"]) == set(runner.ARM_NAMES)
    assert audit["calibration_precedes_outer"].astype(bool).all()
    assert set(audit["threshold_source"]) == {"past_only_policy_calibration"}
    assert not {"outcome", "gross_r", "net_r", "direction"}.intersection(ledger)
    assert not ledger.duplicated(["arm", "channel_episode_id", "decision_time"]).any()


def test_paired_feature_bootstrap_resamples_complete_episode_blocks():
    scenarios = pd.DataFrame(
        {
            "arm": [
                "xgboost_calendar",
                "xgboost_calendar",
                "xgboost_base",
                "xgboost_base",
            ],
            "channel_episode_id": ["e1", "e2", "e1", "e2"],
            "scenario": ["direction_70"] * 4,
            "censored": [False] * 4,
            "net_r": [1.0, -0.5, 0.2, -1.0],
        }
    )

    result = runner.paired_feature_bootstrap(
        scenarios,
        comparisons=(
            ("xgboost_calendar", "xgboost_base", "primary_xgboost_calendar"),
        ),
        calendar_days=2,
        draws=100,
        seed=42,
    )

    row = result.iloc[0]
    assert row["candidate_arm"] == "xgboost_calendar"
    assert row["control_arm"] == "xgboost_base"
    assert int(row["union_episodes"]) == 2
    assert float(row["delta_mean_net_r"]) == pytest.approx(0.65)
    assert float(row["delta_net_r_per_day"]) == pytest.approx(0.65)


def test_reader_artifacts_are_tabular_except_metadata_json():
    json_names = {name for name in runner.READER_ARTIFACTS if name.endswith(".json")}
    assert json_names == {"protocol.json", "frozen_protocol.json", "summary.json"}
    assert "oof_predictions.parquet" in runner.READER_ARTIFACTS
    assert "activation_ledger.parquet" in runner.READER_ARTIFACTS
    assert "economic_metrics.csv" in runner.READER_ARTIFACTS


def test_completed_smoke_publishes_four_arm_oof_economics_and_leakage(tmp_path):
    result = runner.run_calendar_ablation(smoke=True, run_root=tmp_path)

    assert result.summary["timing_model_refit"] is True
    assert result.summary["new_features_added"] is True
    assert result.summary["calendar_feature_count"] == 6
    assert result.summary["base_reproduction_required"] is False
    assert result.summary["economics_evaluated"] is True
    assert result.summary["direction_head_trained"] is False
    assert result.summary["forward_or_lockbox_loaded"] is False

    published = {path.name for path in result.run_dir.iterdir()}
    assert set(runner.READER_ARTIFACTS).issubset(published)
    predictions = pd.read_parquet(result.run_dir / "oof_predictions.parquet")
    assert set(predictions["arm"]) == set(runner.ARM_NAMES)
    metrics = pd.read_csv(result.run_dir / "economic_metrics.csv")
    assert set(metrics["arm"]) == set(runner.ARM_NAMES)
    assert set(metrics["target_multiple_b"]) == {2.0}
    assert set(metrics["hold_minutes"]) == {120}
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert len(leakage) >= 14
    assert leakage["passed"].astype(bool).all()
    state = runner._read_json(result.run_dir / "run_state.json")
    assert state["status"] == "complete"
    assert state["summary"] == result.summary


def test_default_runner_paths_stay_inside_code_tree():
    assert runner.CODE_ROOT in Path(runner.RUN_ROOT).parents
    assert runner.CODE_ROOT in Path(runner.FROZEN_P_ROOT).parents
    assert runner.CODE_ROOT in Path(runner.FROZEN_R_ROOT).parents
