from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_compact_features import (
    COMPACT_FEATURES,
    NATIVE_COMPACT_FEATURES,
)
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


def _subject():
    try:
        return importlib.import_module(
            "experiments.run_event_window_feature_consolidation"
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"Notebook U runner is missing: {error}")


def _small_dataset() -> LargeMoveDecisionDataset:
    sources = (
        *NATIVE_COMPACT_FEATURES,
        "raw_channel_position",
        "raw_channel_position_mean_3",
        "raw_distance_lower_bps",
        "raw_distance_upper_bps",
        "oi_z_7d",
    )
    first = {name: float(index + 1) for index, name in enumerate(sources)}
    first.update(
        raw_channel_position=0.2,
        raw_channel_position_mean_3=0.3,
        raw_distance_lower_bps=20.0,
        raw_distance_upper_bps=80.0,
    )
    second = {**first, "raw_channel_position": 0.8, "raw_channel_position_mean_3": 0.7}
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
        tabular=np.asarray(
            [[row[name] for name in sources] for row in (first, second)],
            dtype=np.float32,
        ),
        tabular_features=sources,
        dropped_features=(),
        feature_set="frozen_N3_side_neutral_volatility",
    )


def test_protocol_freezes_compact_feature_ablation_only():
    runner = _subject()

    protocol = runner.protocol_dict()

    assert protocol["study"] == "notebook_u_volatility_timing_feature_consolidation"
    assert protocol["models"] == ["logreg", "xgboost"]
    assert protocol["feature_sets"] == ["base", "compact_volatility_timing_v1"]
    assert protocol["base_feature_count"] == 248
    assert protocol["compact_feature_count"] == 28
    assert protocol["compact_features"] == list(COMPACT_FEATURES)
    assert protocol["p_hit_frozen"] is True
    assert protocol["timing_heads_refit"] == ["h15", "h30", "h60"]
    assert protocol["timing_score"] == "p_t_le_60"
    assert protocol["alert_policy"] == "level_rearm"
    assert protocol["cooldown_minutes"] == 60
    assert protocol["target_activations_per_day"] == 3.0
    assert protocol["target_multiple_b"] == 2.0
    assert protocol["hold_minutes"] == 120
    assert protocol["calendar_features_included"] is False
    assert protocol["impulse_features_included"] is False
    assert protocol["direction_head_trained"] is False
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_handoff_loading(monkeypatch):
    runner = _subject()
    monkeypatch.setattr(
        runner,
        "load_frozen_r_handoff",
        lambda *_args, **_kwargs: pytest.fail("loaded frozen R"),
    )

    with pytest.raises(ValueError, match="development only"):
        runner.run_feature_consolidation(stage="forward")


def test_paired_datasets_keep_rows_and_create_only_exact_compact_matrix():
    runner = _subject()
    source = _small_dataset()

    base, compact = runner.build_paired_datasets(source)

    assert base is source
    assert compact.decisions.equals(base.decisions)
    assert compact.tabular_features == COMPACT_FEATURES
    assert compact.tabular.shape == (len(base.decisions), 28)
    assert not {
        "oi_z_7d",
        "utc_hour_sin",
        "impulse_count_60m",
        "funding_z_side",
    }.intersection(compact.tabular_features)


def test_predictive_tables_measure_compact_improvement_on_identical_rows():
    runner = _subject()
    rows = []
    for fold_id in ("2023H1", "2023H2"):
        for model in ("logreg", "xgboost"):
            for feature_set, probabilities in (
                ("base", (0.4, 0.6)),
                ("compact_volatility_timing_v1", (0.1, 0.9)),
            ):
                suffix = "compact" if feature_set.startswith("compact") else "base"
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
        metrics["arm"].eq("xgboost_compact")
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
        config=runner.FeatureConsolidationConfig(threshold_grid_size=5),
    )

    assert set(ledger["arm"]) == set(runner.ARM_NAMES)
    assert set(ledger["policy"]) == {"level_rearm"}
    assert audit["calibration_precedes_outer"].astype(bool).all()
    assert set(audit["threshold_source"]) == {"past_only_policy_calibration"}
    assert not {"outcome", "net_r", "direction"}.intersection(ledger)


def test_promotion_requires_predictive_frequency_economic_and_leakage_gates():
    runner = _subject()
    kwargs = {
        "brier_improvement": 0.01,
        "log_loss_improvement": 0.01,
        "nonnegative_brier_folds": 4,
        "frequency_per_day": 3.0,
        "economic_delta_ci_low": 0.001,
        "path_completeness": 1.0,
        "leakage_passed": True,
        "base_reproduced": True,
    }
    assert runner.promotion_decision(**kwargs) is True
    for key, failing in (
        ("brier_improvement", -0.001),
        ("log_loss_improvement", -0.001),
        ("nonnegative_brier_folds", 3),
        ("frequency_per_day", 2.49),
        ("economic_delta_ci_low", 0.0),
        ("path_completeness", 0.989),
        ("leakage_passed", False),
        ("base_reproduced", False),
    ):
        altered = {**kwargs, key: failing}
        assert runner.promotion_decision(**altered) is False


def test_completed_smoke_publishes_paired_compact_evidence(tmp_path):
    runner = _subject()

    result = runner.run_feature_consolidation(smoke=True, run_root=tmp_path)

    assert result.summary["timing_model_refit"] is True
    assert result.summary["p_hit_frozen"] is True
    assert result.summary["compact_feature_count"] == 28
    assert result.summary["calendar_features_included"] is False
    assert result.summary["impulse_features_included"] is False
    assert result.summary["base_reproduction_required"] is False
    assert result.summary["economics_evaluated"] is True
    assert result.summary["direction_head_trained"] is False
    assert result.summary["forward_or_lockbox_loaded"] is False
    assert result.summary["compact_promoted"] is False
    assert set(runner.READER_ARTIFACTS).issubset(
        {path.name for path in result.run_dir.iterdir()}
    )
    predictions = pd.read_parquet(result.run_dir / "oof_predictions.parquet")
    assert set(predictions["arm"]) == set(runner.ARM_NAMES)
    counts = predictions.groupby("arm").size()
    assert counts.nunique() == 1
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert len(leakage) >= 20
    assert leakage["passed"].astype(bool).all()
    feature_audit = pd.read_csv(result.run_dir / "compact_feature_audit.csv")
    assert len(feature_audit) == 28
    assert feature_audit["included"].astype(bool).all()
    redundancy = pd.read_csv(result.run_dir / "feature_redundancy_audit.csv")
    assert {"within_base", "compact_vs_base", "within_compact"}.issubset(
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
