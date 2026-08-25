import numpy as np
import pandas as pd

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_large_move_oof import (
    LargeMoveOOFConfig,
    _fit_binary_platt,
    assert_identical_large_move_keys,
    run_large_move_fold,
)


def _dataset() -> tuple[LargeMoveDecisionDataset, PurgedFold]:
    rows = []
    features = []
    start = pd.Timestamp("2021-01-01", tz="UTC")
    rng = np.random.default_rng(8)
    for episode in range(16):
        for move_code in range(3):
            decision = start + pd.Timedelta(days=episode, minutes=5 * move_code)
            rows.append(
                {
                    "window_id": f"w{episode}_{move_code}",
                    "channel_episode_id": f"e{episode}",
                    "side": "long" if episode % 2 == 0 else "short",
                    "step": move_code,
                    "decision_time": decision,
                    "move_code": move_code,
                    "move_label": ("no_big_move", "up_big", "down_big")[move_code],
                    "adaptive_barrier_bps": 100.0,
                    "terminal_return_bps": 0.0,
                    "label_start": decision,
                    "label_end": decision + pd.Timedelta(minutes=30),
                    "model_target_valid": True,
                }
            )
            vector = rng.normal(size=5)
            vector[0] += move_code
            features.append(vector)
    decisions = pd.DataFrame(rows)
    train = np.arange(0, 42)
    valid = np.arange(42, 48)
    fold = PurgedFold(
        fold_id="test",
        train=train,
        valid=valid,
        train_end=decisions.iloc[valid].decision_time.min(),
        valid_start=decisions.iloc[valid].decision_time.min(),
        valid_end=decisions.iloc[valid].decision_time.max() + pd.Timedelta(days=1),
    )
    dataset = LargeMoveDecisionDataset(
        decisions=decisions,
        tabular=np.asarray(features, dtype=np.float32),
        tabular_features=tuple(f"f{i}" for i in range(5)),
        dropped_features=(),
        feature_set="base",
    )
    return dataset, fold


def test_fold_is_purged_and_outputs_valid_probabilities():
    dataset, fold = _dataset()
    config = LargeMoveOOFConfig(
        model=LargeMoveModelConfig(xgb_estimators=10, n_jobs=1)
    )
    result = run_large_move_fold("logreg", fold, dataset, config)
    probabilities = result.scores[["p_no_big", "p_up_big", "p_down_big"]].to_numpy()
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert not result.scores.duplicated(["window_id", "step"]).any()
    assert result.fold_audit.iloc[0].episode_overlap == 0
    assert result.fold_audit.iloc[0].train_label_end_max <= fold.valid_start
    assert len(result.threshold_frontier) == len(config.policy.threshold_grid)


def test_two_stage_and_multiclass_use_identical_outer_keys():
    dataset, fold = _dataset()
    config = LargeMoveOOFConfig(
        model=LargeMoveModelConfig(xgb_estimators=8, n_jobs=1)
    )
    multiclass = run_large_move_fold("xgboost", fold, dataset, config)
    two_stage = run_large_move_fold("xgboost_two_stage", fold, dataset, config)
    assert_identical_large_move_keys([multiclass, two_stage])
    audit = two_stage.calibration_audit.iloc[0]
    assert np.isnan(audit.temperature)
    assert np.isfinite(audit.opportunity_platt_slope)
    assert np.isfinite(audit.direction_platt_slope)
    assert not audit.opportunity_identity_fallback
    assert not audit.direction_identity_fallback
    assert audit.direction_early_rows > 0


def test_sparse_binary_calibration_uses_registered_identity_fallback():
    slope, intercept, fallback = _fit_binary_platt(
        np.array([-1.0, 0.0, 1.0]),
        np.array([1, 1, 1]),
        np.ones(3),
    )
    assert (slope, intercept, fallback) == (1.0, 0.0, True)


def test_outer_label_mutation_does_not_change_predictions_or_threshold():
    dataset, fold = _dataset()
    config = LargeMoveOOFConfig()
    original = run_large_move_fold("logreg", fold, dataset, config)
    changed = dataset.decisions.copy()
    changed.loc[fold.valid, "move_code"] = np.roll(
        changed.loc[fold.valid, "move_code"].to_numpy(), 1
    )
    mutated = LargeMoveDecisionDataset(
        decisions=changed,
        tabular=dataset.tabular,
        tabular_features=dataset.tabular_features,
        dropped_features=(),
        feature_set="base",
    )
    replay = run_large_move_fold("logreg", fold, mutated, config)
    columns = ["p_no_big", "p_up_big", "p_down_big", "calibration_threshold"]
    assert np.allclose(original.scores[columns], replay.scores[columns])
