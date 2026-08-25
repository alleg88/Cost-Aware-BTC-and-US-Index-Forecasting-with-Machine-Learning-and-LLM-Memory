from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_dataset import (
    CONTEXT_FEATURES,
    SEQUENCE_FEATURES,
    EventWindowSequences,
)
from experiments.event_window_tail_dataset import build_tail_decision_dataset


UTC = "UTC"
ACTIVE_BARS = 12
PRE_WINDOW_BARS = 24


def _sequences(*, windows: int = 2) -> EventWindowSequences:
    metadata_rows: list[dict[str, object]] = []
    sequence = np.empty(
        (windows, PRE_WINDOW_BARS + ACTIVE_BARS, len(SEQUENCE_FEATURES)),
        dtype=np.float32,
    )
    context = np.empty(
        (windows, ACTIVE_BARS, len(CONTEXT_FEATURES)), dtype=np.float32
    )
    source_times = np.empty((windows, ACTIVE_BARS), dtype="datetime64[ns]")
    decision_times = np.empty((windows, ACTIVE_BARS), dtype="datetime64[ns]")
    for window in range(windows):
        start = pd.Timestamp("2024-01-02", tz=UTC) + pd.Timedelta(days=window)
        metadata_rows.append(
            {
                "window_id": f"w{window:04d}",
                "channel_episode_id": f"ep{window // 2:04d}",
                "side": "long" if window % 2 == 0 else "short",
                "window_start": start,
                "window_end": start + pd.Timedelta(hours=1),
            }
        )
        row = np.arange(PRE_WINDOW_BARS + ACTIVE_BARS, dtype=np.float32)[:, None]
        feature = np.arange(len(SEQUENCE_FEATURES), dtype=np.float32)[None, :]
        sequence[window] = row + feature / 100.0 + window
        step = np.arange(ACTIVE_BARS, dtype=np.float32)[:, None]
        context_feature = np.arange(len(CONTEXT_FEATURES), dtype=np.float32)[None, :]
        context[window] = step + context_feature / 10.0 + window
        decisions = pd.date_range(start, periods=ACTIVE_BARS, freq="5min")
        decision_times[window] = decisions.tz_localize(None).to_numpy(
            dtype="datetime64[ns]"
        )
        source_times[window] = (decisions - pd.Timedelta("5min")).tz_localize(
            None
        ).to_numpy(dtype="datetime64[ns]")
    return EventWindowSequences(
        metadata=pd.DataFrame(metadata_rows),
        sequence=sequence,
        context=context,
        source_bar_times=source_times,
        decision_times=decision_times,
        sequence_valid=np.ones(sequence.shape[:2], dtype=bool),
        decision_valid=np.ones((windows, ACTIVE_BARS), dtype=bool),
    )


def _labels(
    sequences: EventWindowSequences | None = None, *, risk_bps: float = 50.0
) -> pd.DataFrame:
    data = _sequences() if sequences is None else sequences
    rows: list[dict[str, object]] = []
    for window, metadata in data.metadata.reset_index(drop=True).iterrows():
        for step in np.flatnonzero(data.decision_valid[window]):
            decision = pd.Timestamp(data.decision_times[window, step], tz=UTC)
            outcome = ("sl", "tp", "timeout")[int(step) % 3]
            gross_r = {"sl": -1.0, "tp": 2.0, "timeout": 0.25}[outcome]
            rows.append(
                {
                    "window_id": metadata["window_id"],
                    "channel_episode_id": metadata["channel_episode_id"],
                    "side": metadata["side"],
                    "step": int(step),
                    "source_bar_time": decision - pd.Timedelta("5min"),
                    "decision_time": decision,
                    "entry_time": decision,
                    "exit_time": decision + pd.Timedelta("15min"),
                    "risk_bps": risk_bps,
                    "outcome": outcome,
                    "r_net": gross_r - 10.0 / risk_bps,
                    "label_start": decision,
                    "label_end": decision + pd.Timedelta("20min"),
                    "geometry_valid": True,
                    "path_observed": True,
                    "model_target_valid": True,
                }
            )
    return pd.DataFrame(rows)


def _mutate_active_sequence_after_step(
    sequences: EventWindowSequences, *, step: int
) -> EventWindowSequences:
    changed = sequences.sequence.copy()
    changed[:, PRE_WINDOW_BARS + step + 1 :] += np.float32(10_000.0)
    return replace(sequences, sequence=changed)


def _large_sequence_fixture(*, windows: int) -> EventWindowSequences:
    return _sequences(windows=windows)


def _large_labels_fixture(*, windows: int) -> pd.DataFrame:
    sequences = _sequences(windows=windows)
    return _labels(sequences)


def test_tail_dataset_aligns_every_valid_decision_once():
    data = build_tail_decision_dataset(_sequences(), _labels(), rr=2.0, cost_bps=10.0)
    assert not data.decisions.duplicated(["window_id", "step"]).any()
    assert len(data.decisions) == len(data.tabular)
    assert data.tabular.shape[1] == len(data.tabular_features)


def test_tp_and_sl_net_r_are_known_from_geometry():
    data = build_tail_decision_dataset(
        _sequences(), _labels(risk_bps=50.0), rr=2.0, cost_bps=10.0
    )
    row = data.decisions.iloc[0]
    assert row.tp_net_r == pytest.approx(2.0 - 10.0 / 50.0)
    assert row.sl_net_r == pytest.approx(-1.0 - 10.0 / 50.0)


def test_future_active_mutation_cannot_change_prior_tabular_row():
    base = _sequences()
    first = build_tail_decision_dataset(base, _labels(), rr=2.0, cost_bps=10.0)
    changed = _mutate_active_sequence_after_step(base, step=2)
    second = build_tail_decision_dataset(changed, _labels(), rr=2.0, cost_bps=10.0)
    np.testing.assert_allclose(
        first.tabular[first.decisions.step == 2],
        second.tabular[second.decisions.step == 2],
        equal_nan=True,
    )


def test_tail_dataset_rejects_duplicate_label_keys():
    labels = pd.concat([_labels(), _labels().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_tail_decision_dataset(_sequences(), labels, rr=2.0, cost_bps=10.0)


def test_tail_dataset_rejects_missing_label_key():
    with pytest.raises(ValueError, match="decision keys"):
        build_tail_decision_dataset(
            _sequences(), _labels().iloc[1:], rr=2.0, cost_bps=10.0
        )


def test_tail_dataset_preserves_censored_score_row_but_marks_target_invalid():
    labels = _labels().copy()
    censored = tuple(labels.loc[0, ["window_id", "step"]])
    labels.loc[0, "model_target_valid"] = False
    labels.loc[0, "path_observed"] = False
    labels.loc[0, "outcome"] = "censored"
    labels.loc[0, "r_net"] = np.nan
    data = build_tail_decision_dataset(
        _sequences(), labels, rr=2.0, cost_bps=10.0
    )
    row = data.decisions.set_index(["window_id", "step"]).loc[censored]
    assert not bool(row.model_target_valid)
    assert row.outcome_code == -1


def test_tabular_feature_order_is_stable_across_runs():
    first = build_tail_decision_dataset(
        _sequences(), _labels(), rr=2.0, cost_bps=10.0
    )
    second = build_tail_decision_dataset(
        _sequences(), _labels(), rr=2.0, cost_bps=10.0
    )
    assert first.tabular_features == second.tabular_features


def test_tabular_row_contains_positioning_age_and_missingness():
    data = build_tail_decision_dataset(_sequences(), _labels(), rr=2.0, cost_bps=10.0)
    assert {"positioning_age_log", "positioning_stale", "oi_missing"} <= set(
        data.tabular_features
    )


def test_tabular_matrix_is_bounded_float32():
    sequences = _large_sequence_fixture(windows=500)
    data = build_tail_decision_dataset(
        sequences,
        _large_labels_fixture(windows=500),
        rr=2.0,
        cost_bps=10.0,
    )
    assert data.tabular.dtype == np.float32
    assert data.tabular.nbytes == data.tabular.size * 4


def test_timeout_target_is_retained_only_for_observed_timeout_rows():
    data = build_tail_decision_dataset(_sequences(), _labels(), rr=2.0, cost_bps=10.0)
    timeout = data.decisions["outcome"].eq("timeout")
    assert data.decisions.loc[timeout, "timeout_net_r"].notna().all()
    assert data.decisions.loc[~timeout, "timeout_net_r"].isna().all()
