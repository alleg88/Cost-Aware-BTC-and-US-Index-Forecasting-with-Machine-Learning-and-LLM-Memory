"""Purging and overlap-weight tests for Notebook B validation."""

import numpy as np
import pandas as pd
import pytest

from evaluation.channel_window_validation import (
    VALID_BLOCKS,
    effective_sample_size,
    expanding_purged_folds,
    interval_uniqueness,
)


def _overlap_fixture() -> pd.DataFrame:
    rows = [
        (1, "2021-06-01 00:00", "2021-06-01 00:00", "2021-06-01 00:20"),
        (2, "2021-12-31 23:50", "2021-12-31 23:50", "2022-01-01 00:10"),
        (3, "2022-01-10 00:00", "2022-01-10 00:00", "2022-01-10 01:00"),
        (3, "2022-01-10 00:05", "2022-01-10 00:05", "2022-01-10 01:05"),
        (4, "2022-07-10 00:00", "2022-07-10 00:00", "2022-07-10 00:30"),
        (5, "2023-02-10 00:00", "2023-02-10 00:00", "2023-02-10 00:30"),
        (6, "2023-08-10 00:00", "2023-08-10 00:00", "2023-08-10 00:30"),
        (7, "2024-02-10 00:00", "2024-02-10 00:00", "2024-02-10 00:30"),
        (8, "2024-08-10 00:00", "2024-08-10 00:00", "2024-08-10 00:30"),
        (9, "2025-02-10 00:00", "2025-02-10 00:00", "2025-02-10 00:30"),
    ]
    frame = pd.DataFrame(
        rows, columns=["channel_episode_id", "decision_time", "label_start", "label_end"]
    )
    for column in ("decision_time", "label_start", "label_end"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    return frame


def _sparse_fixture() -> pd.DataFrame:
    start = pd.to_datetime(
        ["2024-01-01 00:00", "2024-01-01 00:10", "2024-01-01 00:20"], utc=True,
    )
    return pd.DataFrame({"label_start": start, "label_end": start + pd.Timedelta("2min")})


def _dense_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label_start": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02"],
                utc=True,
            ),
            "label_end": pd.to_datetime(
                ["2024-01-01 00:10", "2024-01-01 00:08", "2024-01-01 00:06"],
                utc=True,
            ),
        }
    )


def test_fold_never_splits_episode_or_overlaps_a_label_path():
    events = _overlap_fixture()
    folds = expanding_purged_folds(events)

    assert len(folds) == len(VALID_BLOCKS) == 7
    for fold in folds:
        train, valid = events.iloc[fold.train], events.iloc[fold.valid]
        assert set(train["channel_episode_id"]).isdisjoint(valid["channel_episode_id"])
        if not train.empty and not valid.empty:
            assert (train["label_end"] < valid["decision_time"].min()).all()


def test_boundary_spanning_episode_is_dropped_from_both_sides():
    events = _overlap_fixture()
    first = expanding_purged_folds(events)[0]

    assert set(events.iloc[first.train]["channel_episode_id"]) == {1}
    assert set(events.iloc[first.valid]["channel_episode_id"]) == {3}
    assert 2 not in set(events.iloc[np.r_[first.train, first.valid]]["channel_episode_id"])


def test_raw_uniqueness_falls_when_more_labels_overlap():
    sparse = interval_uniqueness(_sparse_fixture(), np.arange(3), normalize=False)
    dense = interval_uniqueness(_dense_fixture(), np.arange(3), normalize=False)

    assert dense.mean() < sparse.mean()
    assert np.isfinite(dense).all() and (dense > 0).all()


def test_training_weights_are_positive_and_normalised_to_mean_one():
    weights = interval_uniqueness(_dense_fixture(), np.arange(3))

    assert weights.mean() == pytest.approx(1.0)
    assert (weights > 0).all()
    assert 0 < effective_sample_size(weights) <= len(weights)


def test_uniqueness_uses_only_the_requested_training_rows():
    events = pd.concat([_sparse_fixture(), _dense_fixture()], ignore_index=True)

    first_only = interval_uniqueness(events, np.arange(3), normalize=False)

    np.testing.assert_allclose(first_only, np.ones(3))
