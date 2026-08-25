import numpy as np
import pandas as pd

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_conditional_oof import (
    ConditionalOOFConfig,
    compose_severity_probabilities,
    compose_timing_probabilities,
    run_conditional_fold,
    timing_interval_index,
)


def _dataset() -> tuple[LargeMoveDecisionDataset, PurgedFold, np.ndarray]:
    rows = []
    features = []
    start = pd.Timestamp("2021-01-01", tz="UTC")
    rng = np.random.default_rng(41)
    classes = [0, 2, 2, 2, 2, 3, 3, 4]
    times = [np.nan, 5.0, 15.0, 30.0, 60.0, 20.0, 80.0, 45.0]
    for episode in range(30):
        for step, (target_class, tth) in enumerate(zip(classes, times, strict=True)):
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
                    "tth_100_min": tth,
                }
            )
            vector = rng.normal(size=6)
            vector[0] += target_class
            vector[1] += 0.0 if np.isnan(tth) else tth / 120.0
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
    p_hit = np.linspace(0.25, 0.75, len(decisions))
    return (
        LargeMoveDecisionDataset(
            decisions=decisions,
            tabular=np.asarray(features, dtype=np.float32),
            tabular_features=tuple(f"past_f{i}" for i in range(6)),
            dropped_features=(),
            feature_set="test_frozen_N3",
        ),
        fold,
        p_hit,
    )


def test_timing_interval_index_merges_first_fifteen_minutes():
    values = np.array([1.0, 5.0, 15.0, 16.0, 30.0, 31.0, 60.0, 61.0, 120.0, np.nan])
    expected = np.array([0, 0, 0, 1, 1, 2, 2, 3, 3, -1])
    assert np.array_equal(timing_interval_index(values), expected)


def test_conditional_hazards_preserve_frozen_n3_at_120_minutes():
    p_hit = np.array([0.20, 0.80])
    result = compose_timing_probabilities(
        p_hit,
        h_15=np.array([0.50, 0.25]),
        h_30=np.array([0.50, 0.50]),
        h_60=np.array([0.50, 0.50]),
    )
    assert np.allclose(result["p_t_le_15"], [0.10, 0.20])
    assert np.allclose(result["p_t_le_30"], [0.15, 0.50])
    assert np.allclose(result["p_t_le_60"], [0.175, 0.65])
    assert np.array_equal(result["p_t_le_120"], p_hit)
    ordered = np.column_stack([result[f"p_t_le_{h}"] for h in (15, 30, 60, 120)])
    assert (np.diff(ordered, axis=1) >= 0.0).all()


def test_nested_tail_heads_cannot_make_two_barrier_more_likely_than_one_barrier():
    result = compose_severity_probabilities(
        p_hit=np.array([0.40, 0.90]),
        p_ge_150_given_100=np.array([0.50, 0.20]),
        p_ge_200_given_150=np.array([0.25, 0.75]),
    )
    assert np.allclose(result["p_ge_100"], [0.40, 0.90])
    assert np.allclose(result["p_ge_150"], [0.20, 0.18])
    assert np.allclose(result["p_ge_200"], [0.05, 0.135])
    assert (result["p_ge_200"] <= result["p_ge_150"]).all()
    assert (result["p_ge_150"] <= result["p_ge_100"]).all()


def test_conditional_fold_is_purged_nested_and_anchored_to_supplied_n3():
    dataset, fold, p_hit = _dataset()
    result = run_conditional_fold(
        "logreg",
        fold,
        dataset,
        p_hit_by_position=p_hit,
        config=ConditionalOOFConfig(
            model=LargeMoveModelConfig(logreg_max_iter=500, n_jobs=1)
        ),
    )
    scores = result.scores
    assert np.array_equal(scores["p_t_le_120"], p_hit[fold.valid])
    assert (scores.p_t_le_15 <= scores.p_t_le_30).all()
    assert (scores.p_t_le_30 <= scores.p_t_le_60).all()
    assert (scores.p_t_le_60 <= scores.p_t_le_120).all()
    assert (scores.p_ge_200 <= scores.p_ge_150).all()
    assert (scores.p_ge_150 <= scores.p_ge_100).all()
    assert result.fold_audit.iloc[0].episode_overlap == 0
    assert result.fold_audit.iloc[0].train_label_end_max <= fold.valid_start
    assert not result.calibration_scores.empty


def test_outer_label_mutation_cannot_change_conditional_predictions():
    dataset, fold, p_hit = _dataset()
    config = ConditionalOOFConfig(
        model=LargeMoveModelConfig(xgb_estimators=5, n_jobs=1)
    )
    original = run_conditional_fold(
        "xgboost", fold, dataset, p_hit_by_position=p_hit, config=config
    )
    changed = dataset.decisions.copy()
    changed.loc[fold.valid, "magnitude_class"] = 4
    changed.loc[fold.valid, "tth_100_min"] = 1.0
    replay = run_conditional_fold(
        "xgboost",
        fold,
        LargeMoveDecisionDataset(
            decisions=changed,
            tabular=dataset.tabular,
            tabular_features=dataset.tabular_features,
            dropped_features=(),
            feature_set=dataset.feature_set,
        ),
        p_hit_by_position=p_hit,
        config=config,
    )
    probability_columns = [
        "p_t_le_15",
        "p_t_le_30",
        "p_t_le_60",
        "p_t_le_120",
        "p_ge_150",
        "p_ge_200",
    ]
    assert np.allclose(original.scores[probability_columns], replay.scores[probability_columns])
