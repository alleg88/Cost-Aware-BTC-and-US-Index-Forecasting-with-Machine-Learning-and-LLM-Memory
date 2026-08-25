from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_direction_dataset import (
    DIRECTION_FEATURES,
    build_direction_dataset,
    interval_uniqueness,
    pair_direction_paths,
)
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.run_event_window_economic_feasibility import replay_brackets


def _paths(*, long_r: float, short_r: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activation_key": ["a1", "a1"],
            "direction": ["long", "short"],
            "net_r": [long_r, short_r],
            "target_multiple_b": [2.0, 2.0],
            "hold_minutes": [120, 120],
            "cost_bps": [10.0, 10.0],
        }
    )


def _attempts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activation_key": ["a1"],
            "window_id": ["w1"],
            "step": [0],
            "channel_episode_id": ["e1"],
            "decision_time": [pd.Timestamp("2024-01-01 00:00:00+00:00")],
            "channel_side": ["long"],
            "adaptive_barrier_bps": [100.0],
            "reference_price": [100.0],
        }
    )


def _minute_path() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=120, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
        index=index,
    )


def test_pair_direction_paths_builds_economic_delta_and_ties():
    paired = pair_direction_paths(_paths(long_r=0.7, short_r=-1.1))
    assert paired.loc[0, "delta_r"] == pytest.approx(1.8)
    assert paired.loc[0, "best_side"] == "long"
    assert paired.loc[0, "economic_value"] == pytest.approx(1.8)


@pytest.mark.parametrize(
    ("column", "invalid", "message"),
    [
        ("target_multiple_b", 3.0, "target multiple"),
        ("hold_minutes", 60, "hold"),
        ("cost_bps", 7.0, "cost"),
    ],
)
def test_pair_direction_paths_rejects_non_v_primary_geometry(
    column: str, invalid: float, message: str
):
    paths = _paths(long_r=0.7, short_r=-1.1)
    paths.loc[0, column] = invalid
    with pytest.raises(ValueError, match=message):
        pair_direction_paths(paths)


@pytest.mark.parametrize(
    ("status_column", "status"),
    [("path_complete", False), ("censored", True)],
)
def test_pair_direction_paths_rejects_non_v_geometry_on_noncomplete_rows(
    status_column: str, status: bool
):
    paths = _paths(long_r=0.7, short_r=-1.1)
    paths[status_column] = status
    paths.loc[0, "cost_bps"] = 7.0
    with pytest.raises(ValueError, match="cost"):
        pair_direction_paths(paths)


def test_direction_contract_is_exact_and_ordered():
    assert len(DIRECTION_FEATURES) == 28
    assert DIRECTION_FEATURES[-3:] == (
        "adaptive_barrier_bps",
        "window_age_fraction",
        "activation_margin",
    )


def test_interval_uniqueness_downweights_overlapping_payoffs():
    times = pd.to_datetime(["2024-01-01 00:00Z", "2024-01-01 01:00Z"])
    weights = interval_uniqueness(times, horizon_minutes=120)
    assert 0.0 < weights[0] < 1.0
    assert 0.0 < weights[1] < 1.0


def test_uniform_round_trip_cost_is_ten_bps():
    paths = replay_brackets(
        _attempts(),
        _minute_path(),
        target_multiples=(2.0,),
        hold_minutes=(120,),
        entry_cost_bps=5.0,
        target_exit_cost_bps=5.0,
        other_exit_cost_bps=5.0,
    )
    assert set(paths["cost_bps"]) == {10.0}


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activation_key": ["a2", "a1"],
            "window_id": ["w2", "w1"],
            "channel_episode_id": ["e2", "e1"],
            "step": [1, 0],
            "decision_time": pd.to_datetime(
                ["2024-01-01 00:05Z", "2024-01-01 00:00Z"]
            ),
            "threshold": [0.40, 0.20],
            "activation_score": [0.55, 0.35],
        }
    )


def _causal_directional() -> LargeMoveDecisionDataset:
    ledger = _ledger()
    return LargeMoveDecisionDataset(
        decisions=ledger.loc[
            :, ["window_id", "channel_episode_id", "step", "decision_time"]
        ].copy(),
        tabular=np.arange(2 * 27, dtype=np.float32).reshape(2, 27),
        tabular_features=DIRECTION_FEATURES[:-1],
        dropped_features=(),
        feature_set="directional",
    )


def _paired_paths() -> pd.DataFrame:
    return pair_direction_paths(
        pd.DataFrame(
            {
                "activation_key": ["a1", "a1", "a2", "a2"],
                "direction": ["long", "short", "long", "short"],
                "net_r": [0.5, -0.5, -0.2, 0.3],
                "target_multiple_b": [2.0] * 4,
                "hold_minutes": [120] * 4,
                "cost_bps": [10.0] * 4,
            }
        )
    )


def test_build_direction_dataset_keeps_exact_feature_alignment_and_margin():
    dataset = build_direction_dataset(_ledger(), _causal_directional(), _paired_paths())
    assert dataset.tabular_features == DIRECTION_FEATURES
    assert dataset.tabular.shape == (2, 28)
    assert dataset.decisions["activation_key"].tolist() == ["a2", "a1"]
    assert dataset.tabular[:, -1] == pytest.approx([0.15, 0.15])
    assert dataset.tabular[0, :-1] == pytest.approx(np.arange(27, dtype=float))


def test_build_direction_dataset_rejects_missing_or_duplicate_paired_coverage():
    paired = _paired_paths()
    with pytest.raises(ValueError, match="do not cover activations"):
        build_direction_dataset(_ledger(), _causal_directional(), paired.iloc[:1])
    duplicated = pd.concat([paired, paired.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="one non-missing row per activation"):
        build_direction_dataset(_ledger(), _causal_directional(), duplicated)
