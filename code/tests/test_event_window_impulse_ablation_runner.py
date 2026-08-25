from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


def _subject():
    try:
        return importlib.import_module("experiments.run_event_window_impulse_ablation")
    except ModuleNotFoundError as error:
        pytest.fail(f"Notebook T runner is missing: {error}")


def _small_dataset() -> LargeMoveDecisionDataset:
    decisions = pd.DataFrame(
        {
            "window_id": ["w0", "w0"],
            "channel_episode_id": ["e0", "e0"],
            "step": [0, 1],
            "decision_time": pd.to_datetime(
                ["2024-01-02 12:00Z", "2024-01-02 12:05Z"], utc=True
            ),
        }
    )
    return LargeMoveDecisionDataset(
        decisions=decisions,
        tabular=np.asarray([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32),
        tabular_features=("past_a", "past_b"),
        dropped_features=(),
        feature_set="base",
    )


def _impulse_frame(subject) -> pd.DataFrame:
    return pd.DataFrame(
        np.arange(14, dtype=float).reshape(2, 7),
        index=pd.DatetimeIndex(
            pd.to_datetime(["2024-01-02 12:00Z", "2024-01-02 12:05Z"], utc=True),
            name="decision_time",
        ),
        columns=subject.IMPULSE_FEATURE_COLUMNS,
    )


def test_protocol_freezes_impulse_only_level_rearm_cell():
    runner = _subject()

    protocol = runner.protocol_dict()

    assert protocol["study"] == "notebook_t_impulse_intensity_ablation"
    assert protocol["models"] == ["logreg", "xgboost"]
    assert protocol["feature_sets"] == ["base", "base+impulse_v1"]
    assert protocol["impulse_features"] == list(runner.IMPULSE_FEATURE_COLUMNS)
    assert protocol["impulse_quantile"] == 0.90
    assert protocol["impulse_lookback_bars"] == 2016
    assert protocol["p_hit_frozen"] is True
    assert protocol["timing_heads_refit"] == ["h15", "h30", "h60"]
    assert protocol["timing_score"] == "p_t_le_60"
    assert protocol["alert_policy"] == "level_rearm"
    assert protocol["cooldown_minutes"] == 60
    assert protocol["target_activations_per_day"] == 3.0
    assert protocol["target_multiple_b"] == 2.0
    assert protocol["hold_minutes"] == 120
    assert protocol["calendar_features_included"] is False
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_handoff_loading(monkeypatch):
    runner = _subject()
    monkeypatch.setattr(
        runner,
        "load_frozen_r_handoff",
        lambda *_args, **_kwargs: pytest.fail("loaded frozen R"),
    )

    with pytest.raises(ValueError, match="development only"):
        runner.run_impulse_ablation(stage="forward")


def test_paired_datasets_add_only_seven_impulse_columns_on_identical_rows():
    runner = _subject()
    source = _small_dataset()

    base, impulse = runner.build_paired_datasets(source, _impulse_frame(runner))

    assert base is source
    assert impulse.decisions.equals(base.decisions)
    assert np.array_equal(
        impulse.tabular[:, : base.tabular.shape[1]], base.tabular, equal_nan=True
    )
    assert impulse.tabular.shape[1] == base.tabular.shape[1] + 7
    assert impulse.tabular_features == (
        *base.tabular_features,
        *runner.IMPULSE_FEATURE_COLUMNS,
    )
    assert not {
        "utc_hour_sin",
        "utc_weekday_sin",
        "weekend_flag",
        "us_cash_session_flag",
    }.intersection(impulse.tabular_features)


def test_predictive_tables_measure_impulse_improvement_on_identical_rows():
    runner = _subject()
    rows = []
    for fold_id in ("2023H1", "2023H2"):
        for model in ("logreg", "xgboost"):
            for feature_set, probabilities in (
                ("base", (0.4, 0.6)),
                ("base+impulse_v1", (0.1, 0.9)),
            ):
                suffix = "impulse" if feature_set.endswith("impulse_v1") else "base"
                for step, (target, probability) in enumerate(
                    zip((0, 1), probabilities, strict=True)
                ):
                    rows.append(
                        {
                            "arm": f"{model}_{suffix}",
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

    xgb = metrics.loc[
        metrics["arm"].eq("xgboost_impulse")
        & metrics["fold_id"].eq("overall")
    ].iloc[0]
    delta = deltas.loc[
        deltas["model"].eq("xgboost") & deltas["fold_id"].eq("overall")
    ].iloc[0]
    assert float(xgb["weighted_brier"]) == pytest.approx(0.01)
    assert float(delta["brier_improvement"]) == pytest.approx(0.15)
    assert int(delta["nonnegative_brier_folds"]) == 2
    assert int(delta["total_folds"]) == 2


def test_level_rearm_uses_past_only_thresholds_for_all_four_arms():
    runner = _subject()
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
        config=runner.ImpulseAblationConfig(threshold_grid_size=5),
    )

    assert set(ledger["arm"]) == set(runner.ARM_NAMES)
    assert set(ledger["policy"]) == {"level_rearm"}
    assert audit["calibration_precedes_outer"].astype(bool).all()
    assert set(audit["threshold_source"]) == {"past_only_policy_calibration"}
    assert not {"outcome", "net_r", "direction"}.intersection(ledger)
    assert not ledger.duplicated(["arm", "channel_episode_id", "decision_time"]).any()


def test_completed_smoke_publishes_paired_impulse_evidence(tmp_path):
    runner = _subject()

    result = runner.run_impulse_ablation(smoke=True, run_root=tmp_path)

    assert result.summary["timing_model_refit"] is True
    assert result.summary["p_hit_frozen"] is True
    assert result.summary["impulse_feature_count"] == 7
    assert result.summary["calendar_features_included"] is False
    assert result.summary["base_reproduction_required"] is False
    assert result.summary["economics_evaluated"] is True
    assert result.summary["direction_head_trained"] is False
    assert result.summary["forward_or_lockbox_loaded"] is False
    assert set(runner.READER_ARTIFACTS).issubset(
        {path.name for path in result.run_dir.iterdir()}
    )
    predictions = pd.read_parquet(result.run_dir / "oof_predictions.parquet")
    assert set(predictions["arm"]) == set(runner.ARM_NAMES)
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert len(leakage) >= 18
    assert leakage["passed"].astype(bool).all()
    redundancy = pd.read_csv(result.run_dir / "feature_redundancy_audit.csv")
    assert {"within_base", "impulse_vs_base", "within_impulse"}.issubset(
        set(redundancy["scope"])
    )
    state = runner._read_json(result.run_dir / "run_state.json")
    assert state["status"] == "complete"
    assert state["summary"] == result.summary


def test_default_runner_paths_stay_inside_code_tree():
    runner = _subject()
    assert runner.CODE_ROOT in Path(runner.RUN_ROOT).parents
    assert runner.CODE_ROOT in Path(runner.FROZEN_P_ROOT).parents
    assert runner.CODE_ROOT in Path(runner.FROZEN_R_ROOT).parents
