import numpy as np
import pandas as pd
import pytest

from experiments.event_window_large_move_dataset import (
    LargeMoveDecisionDataset,
    VOLATILITY_FEATURE_COLUMNS,
)
from experiments.event_window_magnitude_dataset import (
    align_magnitude_dataset,
    assert_magnitude_feature_isolation,
    label_full_path_magnitude,
    magnitude_class,
    time_to_hit_bucket,
)


def _inputs(*, minutes: int = 122):
    decision_time = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    decisions = pd.DataFrame(
        [
            {
                "window_id": "w0",
                "channel_episode_id": "e0",
                "side": "long",
                "step": 0,
                "decision_time": decision_time,
            }
        ]
    )
    index = pd.date_range(decision_time, periods=minutes, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=index,
    )
    causal = decisions[["window_id", "step", "decision_time"]].copy()
    for name in VOLATILITY_FEATURE_COLUMNS:
        causal[name] = 100.0 if name == "adaptive_barrier_bps" else 1.0
    return decisions, minute, causal


def _price(bps: float) -> float:
    return float(100.0 * np.exp(bps / 1e4))


def test_time_to_hit_minute_five_is_included_but_minute_six_is_not():
    decisions, minute, causal = _inputs()
    minute.iloc[4, minute.columns.get_loc("high")] = _price(100.0)
    result = label_full_path_magnitude(decisions, minute, causal_features=causal)
    assert result.iloc[0].tth_100_min == 5.0
    assert bool(result.iloc[0].hit_by_005m)

    minute.loc[:, "high"] = 100.0
    minute.iloc[5, minute.columns.get_loc("high")] = _price(100.0)
    result = label_full_path_magnitude(decisions, minute, causal_features=causal)
    assert result.iloc[0].tth_100_min == 6.0
    assert not bool(result.iloc[0].hit_by_005m)


def test_first_future_bar_is_time_to_hit_minute_one():
    decisions, minute, causal = _inputs()
    minute.iloc[0, minute.columns.get_loc("high")] = _price(100.0)
    result = label_full_path_magnitude(decisions, minute, causal_features=causal)
    assert result.iloc[0].tth_100_min == 1.0


def test_minute_120_is_included_and_minute_121_is_excluded():
    decisions, minute, causal = _inputs()
    minute.iloc[119, minute.columns.get_loc("high")] = _price(100.0)
    included = label_full_path_magnitude(decisions, minute, causal_features=causal)
    assert included.iloc[0].tth_100_min == 120.0

    minute.loc[:, "high"] = 100.0
    minute.iloc[120, minute.columns.get_loc("high")] = _price(200.0)
    excluded = label_full_path_magnitude(decisions, minute, causal_features=causal)
    assert np.isnan(excluded.iloc[0].tth_100_min)
    assert excluded.iloc[0].magnitude_class == 0


def test_full_path_keeps_later_two_barrier_excursion_after_early_one_barrier_hit():
    decisions, minute, causal = _inputs()
    minute.iloc[1, minute.columns.get_loc("high")] = _price(100.0)
    minute.iloc[39, minute.columns.get_loc("low")] = _price(-200.0)
    result = label_full_path_magnitude(decisions, minute, causal_features=causal)
    row = result.iloc[0]
    assert row.tth_100_min == 2.0
    assert row.tth_200_min == 40.0
    assert row.magnitude_ratio == pytest.approx(2.0)
    assert row.magnitude_class == 4


def test_same_minute_double_touch_is_valid_magnitude_without_direction_output():
    decisions, minute, causal = _inputs()
    minute.iloc[2, minute.columns.get_loc("high")] = _price(150.0)
    minute.iloc[2, minute.columns.get_loc("low")] = _price(-150.0)
    result = label_full_path_magnitude(decisions, minute, causal_features=causal)
    assert bool(result.iloc[0].magnitude_target_valid)
    assert result.iloc[0].magnitude_class == 3
    assert result.iloc[0].tth_100_min == 3.0
    assert not {"hit_side", "direction", "move_code"}.intersection(result.columns)


def test_gap_censors_full_path_instead_of_creating_negative_and_end_stays_frozen():
    decisions, minute, causal = _inputs()
    minute = minute.drop(minute.index[20])
    result = label_full_path_magnitude(decisions, minute, causal_features=causal)
    row = result.iloc[0]
    assert not bool(row.magnitude_target_valid)
    assert row.magnitude_class == -1
    assert row.label_end == row.decision_time + pd.Timedelta(minutes=120)


def test_future_mutation_changes_target_not_causal_barrier_or_feature_matrix():
    decisions, minute, causal = _inputs()
    first = label_full_path_magnitude(decisions, minute, causal_features=causal)
    minute.iloc[10, minute.columns.get_loc("high")] = _price(200.0)
    second = label_full_path_magnitude(decisions, minute, causal_features=causal)
    assert first.iloc[0].magnitude_class != second.iloc[0].magnitude_class
    assert first.iloc[0].adaptive_barrier_bps == second.iloc[0].adaptive_barrier_bps
    for name in VOLATILITY_FEATURE_COLUMNS:
        assert first.iloc[0][name] == second.iloc[0][name]


def test_alignment_preserves_n3_matrix_and_rejects_future_features():
    decisions, minute, causal = _inputs()
    labels = label_full_path_magnitude(decisions, minute, causal_features=causal)
    frozen = LargeMoveDecisionDataset(
        decisions=causal,
        tabular=np.asarray([[1.0, 2.0]], dtype=np.float32),
        tabular_features=("past_return_5m", "adaptive_barrier_bps"),
        dropped_features=(),
        feature_set="N3",
    )
    aligned = align_magnitude_dataset(frozen, labels)
    assert np.array_equal(aligned.tabular, frozen.tabular)
    assert aligned.tabular_features == frozen.tabular_features
    with pytest.raises(AssertionError, match="future/target"):
        assert_magnitude_feature_isolation(("future_up_excursion_bps",))


def test_registered_bins_and_tth_buckets_are_exact():
    assert [magnitude_class(value) for value in (0.0, 0.75, 1.0, 1.5, 2.0)] == [
        0,
        1,
        2,
        3,
        4,
    ]
    values = time_to_hit_bucket(pd.Series([5, 6, 16, 31, 61, np.nan]))
    assert list(values.astype(str)) == ["<=5", "6-15", "16-30", "31-60", "61-120", "no hit"]
