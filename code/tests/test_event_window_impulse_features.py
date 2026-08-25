from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


def _subject():
    try:
        return importlib.import_module("experiments.event_window_impulse_features")
    except ModuleNotFoundError as error:
        pytest.fail(f"impulse feature module is missing: {error}")


def _five_minute_frame(returns: list[float]) -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01 00:00Z",
        periods=len(returns) + 1,
        freq="5min",
    )
    close = 100.0 * np.exp(np.r_[0.0, np.cumsum(returns)])
    return pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "minute_count": 5,
            "bar_complete": True,
        },
        index=index,
    )


def _dataset_at(decision_times: list[pd.Timestamp]) -> LargeMoveDecisionDataset:
    rows = len(decision_times)
    return LargeMoveDecisionDataset(
        decisions=pd.DataFrame(
            {
                "window_id": [f"w{index}" for index in range(rows)],
                "channel_episode_id": [f"e{index}" for index in range(rows)],
                "step": np.arange(rows, dtype=int),
                "decision_time": pd.to_datetime(decision_times, utc=True),
            }
        ),
        tabular=np.arange(rows * 2, dtype=np.float32).reshape(rows, 2),
        tabular_features=("past_a", "past_b"),
        dropped_features=(),
        feature_set="base",
    )


def test_current_completed_return_is_excluded_from_q90_and_included_in_counts():
    subject = _subject()
    five = _five_minute_frame([0.01] * 12 + [0.04, 0.01, 0.01, 0.04])
    config = subject.ImpulseFeatureConfig(lookback_bars=4)

    result = subject.build_impulse_feature_frame(five, config=config)

    source_time = five.index[-1]
    decision_time = source_time + pd.Timedelta("5min")
    row = result.loc[decision_time]
    # Previous values have Q90=0.031. The current 0.04 return is therefore an
    # impulse. Including the tied current maximum in the percentile would make
    # the strict comparison false, so this catches look-ahead in the threshold.
    assert row["impulse_count_15m"] == pytest.approx(1.0)
    assert row["minutes_since_last_impulse"] == pytest.approx(0.0)
    assert np.isnan(row["median_interarrival_last_5"])
    assert np.isfinite(row["impulse_excess_energy_60m"])
    assert row["impulse_excess_energy_60m"] > 0.0
    assert result.index.name == "decision_time"
    assert result.index.is_unique


def test_gap_invalidates_history_and_counts_until_contiguous_history_recovers():
    subject = _subject()
    returns = [0.01] * 13 + [0.04]
    complete = _five_minute_frame(returns)
    missing_source_time = complete.index[6]
    gapped = complete.drop(index=missing_source_time)

    result = subject.build_impulse_feature_frame(
        gapped,
        config=subject.ImpulseFeatureConfig(lookback_bars=4),
    )

    first_threshold_after_rebuild = complete.index[12] + pd.Timedelta("5min")
    first_full_15m_count = complete.index[14] + pd.Timedelta("5min")
    assert np.isnan(result.loc[first_threshold_after_rebuild, "impulse_count_15m"])
    assert np.isfinite(result.loc[first_full_15m_count, "impulse_count_15m"])
    assert np.isnan(
        result.loc[first_threshold_after_rebuild, "minutes_since_last_impulse"]
    )


def test_incomplete_bar_is_not_silently_treated_as_a_non_event():
    subject = _subject()
    five = _five_minute_frame([0.01] * 13 + [0.04])
    five.loc[five.index[6], "minute_count"] = 4
    five.loc[five.index[6], "bar_complete"] = False

    result = subject.build_impulse_feature_frame(
        five,
        config=subject.ImpulseFeatureConfig(lookback_bars=4),
    )

    invalid_decision = five.index[7] + pd.Timedelta("5min")
    assert result.loc[invalid_decision, list(subject.IMPULSE_FEATURE_COLUMNS)].isna().all()


def test_future_price_change_cannot_change_earlier_impulse_features():
    subject = _subject()
    five = _five_minute_frame([0.01] * 10 + [0.04, 0.01])
    config = subject.ImpulseFeatureConfig(lookback_bars=4)
    original = subject.build_impulse_feature_frame(five, config=config)
    changed = five.copy()
    changed.loc[changed.index[-1], ["open", "high", "low", "close"]] *= 1.5

    perturbed = subject.build_impulse_feature_frame(changed, config=config)

    cutoff = five.index[-1]
    pd.testing.assert_frame_equal(
        original.loc[original.index <= cutoff],
        perturbed.loc[perturbed.index <= cutoff],
        check_exact=True,
    )


def test_many_window_rows_join_one_impulse_timestamp_without_changing_base():
    subject = _subject()
    decision_time = pd.Timestamp("2024-01-02 12:05Z")
    dataset = _dataset_at([decision_time, decision_time])
    values = pd.DataFrame(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]],
        index=pd.DatetimeIndex([decision_time], name="decision_time"),
        columns=subject.IMPULSE_FEATURE_COLUMNS,
    )

    result = subject.append_impulse_features(dataset, values)

    assert result.tabular.shape == (2, 2 + len(subject.IMPULSE_FEATURE_COLUMNS))
    assert np.array_equal(result.tabular[:, :2], dataset.tabular)
    assert np.array_equal(result.tabular[0, 2:], result.tabular[1, 2:])
    assert result.tabular_features == (
        *dataset.tabular_features,
        *subject.IMPULSE_FEATURE_COLUMNS,
    )
    assert result.feature_set == "base+impulse_v1"
    assert result.decisions.equals(dataset.decisions)


def test_impulse_join_rejects_duplicate_source_times_and_duplicate_decision_keys():
    subject = _subject()
    decision_time = pd.Timestamp("2024-01-02 12:05Z")
    dataset = _dataset_at([decision_time])
    row = [1.0] * len(subject.IMPULSE_FEATURE_COLUMNS)
    duplicate_source = pd.DataFrame(
        [row, row],
        index=pd.DatetimeIndex(
            [decision_time, decision_time], name="decision_time"
        ),
        columns=subject.IMPULSE_FEATURE_COLUMNS,
    )
    duplicate_keys = LargeMoveDecisionDataset(
        decisions=pd.concat([dataset.decisions, dataset.decisions], ignore_index=True),
        tabular=np.vstack([dataset.tabular, dataset.tabular]),
        tabular_features=dataset.tabular_features,
        dropped_features=dataset.dropped_features,
        feature_set=dataset.feature_set,
    )

    with pytest.raises(ValueError, match="unique decision_time"):
        subject.append_impulse_features(dataset, duplicate_source)
    with pytest.raises(ValueError, match="window_id, step"):
        subject.append_impulse_features(duplicate_keys, duplicate_source.iloc[:1])
