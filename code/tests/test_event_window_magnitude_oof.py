import numpy as np
import pandas as pd
import pytest

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_magnitude_oof import (
    MagnitudeOOFConfig,
    cumulative_probabilities,
    magnitude_metrics,
    monotonic_violation_count,
    run_magnitude_fold,
)


def _dataset() -> tuple[LargeMoveDecisionDataset, PurgedFold]:
    rows = []
    features = []
    start = pd.Timestamp("2021-01-01", tz="UTC")
    rng = np.random.default_rng(202)
    for episode in range(24):
        for target_class in range(5):
            decision = start + pd.Timedelta(days=episode, minutes=5 * target_class)
            rows.append(
                {
                    "window_id": f"w{episode}",
                    "channel_episode_id": f"e{episode}",
                    "side": "long" if episode % 2 == 0 else "short",
                    "step": target_class,
                    "decision_time": decision,
                    "label_start": decision,
                    "label_end": decision + pd.Timedelta(minutes=120),
                    "magnitude_class": target_class,
                    "magnitude_ratio": [0.5, 0.8, 1.2, 1.7, 2.3][target_class],
                    "magnitude_target_valid": True,
                    "model_target_valid": True,
                    "tth_100_min": np.nan if target_class < 2 else 20.0,
                }
            )
            vector = rng.normal(size=6)
            vector[0] += target_class
            features.append(vector)
    decisions = pd.DataFrame(rows)
    valid = np.arange(22 * 5, 24 * 5)
    fold = PurgedFold(
        fold_id="test",
        train=np.arange(0, 22 * 5),
        valid=valid,
        train_end=decisions.iloc[valid].decision_time.min(),
        valid_start=decisions.iloc[valid].decision_time.min(),
        valid_end=decisions.iloc[valid].decision_time.max() + pd.Timedelta(days=1),
    )
    return (
        LargeMoveDecisionDataset(
            decisions=decisions,
            tabular=np.asarray(features, dtype=np.float32),
            tabular_features=tuple(f"past_f{i}" for i in range(6)),
            dropped_features=(),
            feature_set="test_N3",
        ),
        fold,
    )


def test_cumulative_probabilities_are_nested_by_construction():
    probabilities = np.asarray([[0.1, 0.2, 0.3, 0.25, 0.15]])
    cumulative = cumulative_probabilities(probabilities)
    assert cumulative["075"][0] == pytest.approx(0.9)
    assert cumulative["100"][0] == pytest.approx(0.7)
    assert cumulative["150"][0] == pytest.approx(0.4)
    assert cumulative["200"][0] == pytest.approx(0.15)
    assert monotonic_violation_count(cumulative) == 0


def test_magnitude_fold_is_purged_calibrated_and_nested():
    dataset, fold = _dataset()
    result = run_magnitude_fold(
        fold,
        dataset,
        MagnitudeOOFConfig(
            model=LargeMoveModelConfig(xgb_estimators=10, n_jobs=1)
        ),
    )
    score = result.scores
    assert np.allclose(score[[f"p_bin_{i}" for i in range(5)]].sum(axis=1), 1.0)
    assert (score.p_ge_075 >= score.p_ge_100).all()
    assert (score.p_ge_100 >= score.p_ge_150).all()
    assert (score.p_ge_150 >= score.p_ge_200).all()
    assert result.fold_audit.iloc[0].episode_overlap == 0
    assert result.fold_audit.iloc[0].train_label_end_max <= fold.valid_start
    assert result.fold_audit.iloc[0].monotonic_violation_rate == 0.0
    assert np.isfinite(result.calibration_audit.iloc[0].temperature)


def test_outer_label_mutation_cannot_change_magnitude_predictions():
    dataset, fold = _dataset()
    config = MagnitudeOOFConfig(
        model=LargeMoveModelConfig(xgb_estimators=8, n_jobs=1)
    )
    original = run_magnitude_fold(fold, dataset, config)
    changed = dataset.decisions.copy()
    changed.loc[fold.valid, "magnitude_class"] = (
        changed.loc[fold.valid, "magnitude_class"] + 1
    ) % 5
    replay = run_magnitude_fold(
        fold,
        LargeMoveDecisionDataset(
            decisions=changed,
            tabular=dataset.tabular,
            tabular_features=dataset.tabular_features,
            dropped_features=(),
            feature_set=dataset.feature_set,
        ),
        config,
    )
    columns = [f"p_bin_{i}" for i in range(5)]
    assert np.allclose(original.scores[columns], replay.scores[columns])


def test_metrics_report_rps_and_zero_monotonic_violations():
    labels = np.arange(5)
    probabilities = np.eye(5) * 0.8 + 0.2 / 5
    metrics = magnitude_metrics(labels, probabilities, np.ones(5))
    assert 0.0 <= metrics["ranked_probability_score"] < 0.1
    assert metrics["monotonic_violation_rate"] == 0.0
    assert metrics["ge_100_pr_auc"] > 0.9


def test_label_end_shorter_than_120_minutes_is_rejected():
    dataset, fold = _dataset()
    changed = dataset.decisions.copy()
    changed.loc[0, "label_end"] = changed.loc[0, "decision_time"] + pd.Timedelta(minutes=5)
    with pytest.raises(ValueError, match=r"t\+120"):
        run_magnitude_fold(
            fold,
            LargeMoveDecisionDataset(
                decisions=changed,
                tabular=dataset.tabular,
                tabular_features=dataset.tabular_features,
                dropped_features=(),
                feature_set=dataset.feature_set,
            ),
            MagnitudeOOFConfig(
                model=LargeMoveModelConfig(xgb_estimators=3, n_jobs=1)
            ),
        )
