import numpy as np
import pandas as pd

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.event_window_large_move_models import LargeMoveModelConfig
from experiments.event_window_opportunity_oof import (
    OpportunityOOFConfig,
    assert_identical_opportunity_keys,
    run_opportunity_fold,
)


def _dataset() -> tuple[LargeMoveDecisionDataset, PurgedFold]:
    rows = []
    features = []
    start = pd.Timestamp("2021-01-01", tz="UTC")
    rng = np.random.default_rng(18)
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
                    "label_start": decision,
                    "label_end": decision + pd.Timedelta(minutes=30),
                    "opportunity_code": int(move_code != 0),
                    "opportunity_target_valid": True,
                    "model_target_valid": True,
                }
            )
            vector = rng.normal(size=5)
            vector[0] += 1.5 * int(move_code != 0)
            features.append(vector)
    decisions = pd.DataFrame(rows)
    valid = np.arange(42, 48)
    fold = PurgedFold(
        fold_id="test",
        train=np.arange(0, 42),
        valid=valid,
        train_end=decisions.iloc[valid].decision_time.min(),
        valid_start=decisions.iloc[valid].decision_time.min(),
        valid_end=decisions.iloc[valid].decision_time.max() + pd.Timedelta(days=1),
    )
    return (
        LargeMoveDecisionDataset(
            decisions=decisions,
            tabular=np.asarray(features, dtype=np.float32),
            tabular_features=tuple(f"f{i}" for i in range(5)),
            dropped_features=(),
            feature_set="opportunity_test",
        ),
        fold,
    )


def test_opportunity_fold_is_purged_calibrated_and_direction_free():
    dataset, fold = _dataset()
    result = run_opportunity_fold(
        "N3_xgboost",
        "xgboost",
        fold,
        dataset,
        OpportunityOOFConfig(
            model=LargeMoveModelConfig(xgb_estimators=10, n_jobs=1)
        ),
    )
    assert result.scores["p_hit"].between(0.0, 1.0).all()
    assert not result.scores.duplicated(["window_id", "step"]).any()
    assert not {"p_up_big", "p_down_big", "direction"}.intersection(result.scores)
    assert result.fold_audit.iloc[0].episode_overlap == 0
    assert result.fold_audit.iloc[0].train_label_end_max <= fold.valid_start
    assert np.isfinite(result.calibration_audit.iloc[0].platt_slope)
    assert result.calibration_audit.iloc[0].threshold_status.startswith("deferred")


def test_outer_label_mutation_cannot_change_opportunity_predictions():
    dataset, fold = _dataset()
    original = run_opportunity_fold("N2_logreg", "logreg", fold, dataset)
    changed = dataset.decisions.copy()
    changed.loc[fold.valid, "opportunity_code"] = 1 - changed.loc[
        fold.valid, "opportunity_code"
    ]
    replay = run_opportunity_fold(
        "N2_logreg",
        "logreg",
        fold,
        LargeMoveDecisionDataset(
            decisions=changed,
            tabular=dataset.tabular,
            tabular_features=dataset.tabular_features,
            dropped_features=(),
            feature_set=dataset.feature_set,
        ),
    )
    assert np.allclose(original.scores["p_hit"], replay.scores["p_hit"])
    assert_identical_opportunity_keys([original, replay])
