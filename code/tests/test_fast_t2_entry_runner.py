"""Protocol and split guards for the resumable Fast-T2 entry runner."""

import json

import numpy as np
import pandas as pd
import pytest

from experiments.fast_t2_action_models import MODEL_NAMES
from experiments.run_fast_t2_entry_policy import (
    build_protocol,
    inner_episode_purged_indices,
    validate_resume,
    write_protocol,
)


def test_entry_protocol_is_fast_only_and_dev_bounded(tmp_path):
    protocol = build_protocol(output_dir=tmp_path)

    assert protocol["arm"] == "fast_t2_2bps"
    assert protocol["period_end_exclusive"] == "2025-07-01T00:00:00+00:00"
    assert protocol["models"] == list(MODEL_NAMES)
    assert protocol["forward_or_lockbox_loaded"] is False
    assert protocol["entry_config"]["decision_minutes"] == 15
    assert len(protocol["protocol_hash"]) == 64


def test_resume_rejects_a_changed_protocol_hash(tmp_path):
    write_protocol(tmp_path, decision_minutes=15)

    with pytest.raises(ValueError, match="protocol hash"):
        validate_resume(tmp_path, decision_minutes=10)


def test_protocol_hash_is_stable_and_written_payload_matches(tmp_path):
    first = write_protocol(tmp_path)
    second = validate_resume(tmp_path)
    stored = json.loads((tmp_path / "protocol.json").read_text(encoding="utf-8"))

    assert first == second == stored
    assert build_protocol(output_dir=tmp_path) == build_protocol(output_dir=None)


def _split_fixture() -> pd.DataFrame:
    utc = "UTC"
    return pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                [
                    "2021-03-01", "2021-06-29", "2021-07-02",
                    "2021-08-01", "2021-12-20", "2021-12-21",
                ],
                utc=True,
            ),
            "label_start": pd.to_datetime(
                [
                    "2021-03-01", "2021-06-29", "2021-07-02",
                    "2021-08-01", "2021-12-20", "2021-12-21",
                ],
                utc=True,
            ),
            "label_end": pd.to_datetime(
                [
                    "2021-03-01 00:30", "2021-07-01 00:01",
                    "2021-07-02 00:30", "2021-08-01 00:30",
                    "2022-01-01 00:01", "2021-12-21 00:30",
                ],
                utc=True,
            ),
            # Episode 20 straddles the inner boundary and must be purged whole.
            "channel_episode_id": [10, 20, 20, 30, 40, 50],
        }
    )


def test_inner_split_is_chronological_label_purged_and_episode_disjoint():
    decisions = _split_fixture()
    outer_train = np.arange(len(decisions), dtype=np.int64)

    fit, valid = inner_episode_purged_indices(
        decisions,
        outer_train,
        inner_cut=pd.Timestamp("2021-07-01", tz="UTC"),
        outer_valid_start=pd.Timestamp("2022-01-01", tz="UTC"),
    )

    assert fit.tolist() == [0]
    assert valid.tolist() == [3, 5]
    assert set(decisions.iloc[fit].channel_episode_id).isdisjoint(
        decisions.iloc[valid].channel_episode_id
    )
    assert decisions.iloc[fit].label_end.max() < pd.Timestamp("2021-07-01", tz="UTC")
    assert decisions.iloc[valid].label_end.max() < pd.Timestamp("2022-01-01", tz="UTC")


def test_inner_split_never_imports_rows_outside_outer_train():
    decisions = _split_fixture()

    fit, valid = inner_episode_purged_indices(
        decisions,
        np.array([0, 3], dtype=np.int64),
        inner_cut=pd.Timestamp("2021-07-01", tz="UTC"),
        outer_valid_start=pd.Timestamp("2022-01-01", tz="UTC"),
    )

    assert fit.tolist() == [0]
    assert valid.tolist() == [3]
