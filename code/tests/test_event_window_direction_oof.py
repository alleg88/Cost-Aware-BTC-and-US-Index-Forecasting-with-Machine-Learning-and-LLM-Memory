import numpy as np
import pandas as pd

from experiments.event_window_direction_dataset import DirectionDataset
from experiments.event_window_direction_models import DirectionModelConfig
from experiments.event_window_direction_oof import run_direction_oof


FOLDS = (
    "2022H1",
    "2022H2",
    "2023H1",
    "2023H2",
    "2024H1",
    "2024H2",
    "2025H1",
)
SCORED_FOLDS = FOLDS[1:]


def _dataset(
    scored_counts: tuple[int, ...] = (18, 18, 18, 18, 18, 18),
    *,
    warmup_rows: int = 30,
) -> DirectionDataset:
    counts = (warmup_rows, *scored_counts)
    rows = []
    features = []
    rng = np.random.default_rng(22)
    for fold_position, (fold_id, count) in enumerate(zip(FOLDS, counts)):
        year = int(fold_id[:4])
        month = 1 if fold_id.endswith("H1") else 7
        start = pd.Timestamp(year=year, month=month, day=2, tz="UTC")
        for position in range(count):
            decision_time = start + pd.Timedelta(hours=6 * position)
            delta_r = 0.0 if position % 17 == 0 else (-1.0 if position % 2 else 1.0)
            rows.append(
                {
                    "activation_key": f"{fold_id}-{position}",
                    "fold_id": fold_id,
                    "channel_episode_id": (
                        "shared-episode" if position == 1 else f"{fold_id}-e{position}"
                    ),
                    "channel_side": "long" if position % 3 else "short",
                    "decision_time": decision_time,
                    "delta_r": delta_r,
                }
            )
            vector = rng.normal(size=4)
            vector[0] += np.sign(delta_r)
            vector[1] += fold_position / 10.0
            features.append(vector)
    return DirectionDataset(
        decisions=pd.DataFrame(rows),
        tabular=np.asarray(features, dtype=np.float32),
        tabular_features=("f0", "f1", "f2", "f3"),
    )


def test_direction_oof_starts_after_warmup_and_scores_every_later_activation():
    dataset = _dataset((490, 490, 490, 490, 490, 489), warmup_rows=60)
    result = run_direction_oof(
        dataset, config=DirectionModelConfig(xgb_estimators=2)
    )
    assert set(result.predictions["fold_id"]) == set(SCORED_FOLDS)
    assert len(result.predictions.groupby(["model", "activation_key"])) == 2 * 2939


def test_training_is_purged_and_episode_disjoint():
    audit = run_direction_oof(
        _dataset(), config=DirectionModelConfig(xgb_estimators=2)
    ).fold_audit
    assert (audit["train_label_end_max"] < audit["validation_start"]).all()
    assert audit["episode_overlap"].eq(0).all()


def test_validation_ties_still_receive_both_model_predictions():
    dataset = _dataset()
    result = run_direction_oof(
        dataset, config=DirectionModelConfig(xgb_estimators=2)
    )
    scored = dataset.decisions[dataset.decisions["fold_id"].isin(SCORED_FOLDS)]
    tie_keys = set(scored.loc[scored["delta_r"].eq(0.0), "activation_key"])
    tie_predictions = result.predictions[
        result.predictions["activation_key"].isin(tie_keys)
    ]
    assert set(tie_predictions["activation_key"]) == tie_keys
    assert tie_predictions.groupby("activation_key")["model"].nunique().eq(2).all()
    assert np.isfinite(tie_predictions["direction_score"]).all()
    assert set(tie_predictions["chosen_direction"]) <= {"long", "short"}


def test_training_ties_do_not_change_non_tie_uniqueness_or_predictions():
    baseline = _dataset()
    extra_decisions = pd.concat(
        [
            baseline.decisions,
            pd.DataFrame(
                [
                    {
                        "activation_key": "warmup-extra-tie",
                        "fold_id": "2022H1",
                        "channel_episode_id": "extra-tie-episode",
                        "channel_side": "long",
                        "decision_time": baseline.decisions.iloc[2]["decision_time"],
                        "delta_r": 5e-13,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with_tie = DirectionDataset(
        decisions=extra_decisions,
        tabular=np.vstack([baseline.tabular, np.full((1, 4), 1_000_000.0)]),
        tabular_features=baseline.tabular_features,
    )
    config = DirectionModelConfig(xgb_estimators=2)
    expected = run_direction_oof(baseline, config=config)
    actual = run_direction_oof(with_tie, config=config)
    columns = ["model", "fold_id", "activation_key", "direction_score"]
    pd.testing.assert_frame_equal(
        expected.predictions[columns].reset_index(drop=True),
        actual.predictions[columns].reset_index(drop=True),
    )
    assert np.allclose(actual.fold_audit["uniqueness_mean"], 1.0)


def test_future_fold_mutation_cannot_change_earlier_predictions():
    baseline = _dataset()
    changed_x = baseline.tabular.copy()
    future = baseline.decisions["fold_id"].isin(SCORED_FOLDS[1:]).to_numpy()
    changed_x[future] = 1_000_000.0
    changed_decisions = baseline.decisions.copy()
    changed_decisions.loc[future, "delta_r"] *= -1.0
    changed = DirectionDataset(
        decisions=changed_decisions,
        tabular=changed_x,
        tabular_features=baseline.tabular_features,
    )
    config = DirectionModelConfig(xgb_estimators=2)
    expected = run_direction_oof(baseline, config=config).predictions
    actual = run_direction_oof(changed, config=config).predictions
    columns = ["model", "activation_key", "direction_score"]
    pd.testing.assert_frame_equal(
        expected.loc[expected["fold_id"].eq("2022H2"), columns].reset_index(drop=True),
        actual.loc[actual["fold_id"].eq("2022H2"), columns].reset_index(drop=True),
    )
