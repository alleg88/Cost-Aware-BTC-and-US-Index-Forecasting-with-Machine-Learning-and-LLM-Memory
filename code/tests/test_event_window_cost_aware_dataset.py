from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_cost_aware_dataset import (
    MakerExecutionConfig,
    build_cost_aware_dataset,
    label_maker_window_steps,
)
from experiments.event_window_tail_dataset import TailDecisionDataset


def _sequences(valid_steps: int = 2):
    source_times = pd.date_range("2024-01-01 00:35", periods=12, freq="5min", tz="UTC")
    decision_times = source_times + pd.Timedelta("5min")
    source = np.tile(
        source_times.tz_localize(None).to_numpy(dtype="datetime64[ns]"), (1, 1)
    )
    decision = np.tile(
        decision_times.tz_localize(None).to_numpy(dtype="datetime64[ns]"), (1, 1)
    )
    valid = np.zeros((1, 12), dtype=bool)
    valid[0, :valid_steps] = True
    return SimpleNamespace(
        metadata=pd.DataFrame(
            {
                "window_id": ["w1"],
                "channel_episode_id": ["e1"],
                "side": ["long"],
                "window_end": [pd.Timestamp("2024-01-01 01:40", tz="UTC")],
            }
        ),
        source_bar_times=source,
        decision_times=decision,
        decision_valid=valid,
    )


def _five():
    index = pd.date_range("2023-12-31 23:40", "2024-01-01 01:30", freq="5min", tz="UTC")
    return pd.DataFrame({"open": 100.0, "high": 100.4, "low": 99.0, "close": 100.0}, index=index)


def _minute(*, fill: bool = True, tp: bool = True):
    index = pd.date_range("2024-01-01 00:40", periods=180, freq="1min", tz="UTC")
    frame = pd.DataFrame({"open": 100.0, "high": 100.2, "low": 100.0, "close": 100.0}, index=index)
    if fill:
        frame.loc[index[0], "low"] = 99.4
        if tp:
            frame.loc[index[1], "high"] = 103.0
    return frame


def test_maker_label_waits_for_fill_and_charges_maker_tp_fees():
    config = MakerExecutionConfig(limit_offset_bps=5.0, fill_window_minutes=20)
    row = label_maker_window_steps(_sequences(1), _five(), _minute(), config).iloc[0]
    assert row.entry == pytest.approx(99.95)
    assert row.entry_time == row.decision_time
    assert row.outcome == "tp"
    assert row.r_net == pytest.approx(row.r_gross - 4.0 / row.risk_bps)


def test_unfilled_order_is_observed_zero_return_without_a_trade():
    row = label_maker_window_steps(_sequences(1), _five(), _minute(fill=False)).iloc[0]
    assert row.outcome == "unfilled"
    assert row.path_observed and row.model_target_valid
    assert not row.filled
    assert row.r_net == 0.0


def test_mere_limit_touch_is_not_counted_as_a_confirmed_maker_fill():
    path = _minute(fill=False)
    path.loc[path.index[0], "low"] = 99.95
    row = label_maker_window_steps(_sequences(1), _five(), path).iloc[0]
    assert row.outcome == "unfilled"


def test_cost_dataset_extends_wait_label_and_drops_algebraic_duplicates():
    labels = label_maker_window_steps(_sequences(2), _five(), _minute())
    base_decisions = labels[["window_id", "step"]].copy()
    base = TailDecisionDataset(
        decisions=base_decisions,
        tabular=np.arange(10, dtype=np.float32).reshape(2, 5),
        tabular_features=(
            "log_return_side",
            "active_window_mask_mean_3",
            "time_remaining_fraction",
            "window_reason_code",
            "channel_r2",
        ),
        sequences=None,
    )
    dataset = build_cost_aware_dataset(base, labels)
    assert dataset.tabular_features[:2] == ("log_return_side", "channel_r2")
    assert len(dataset.dropped_features) == 3
    first, second = (dataset.decisions.iloc[0], dataset.decisions.iloc[1])
    assert first.advantage_valid
    assert first.enter_advantage_target == pytest.approx(first.r_net - max(0.0, second.r_net))
    assert first.label_end >= second.label_end
    assert dataset.tabular.shape == (2, 6)


def test_wait_target_compares_now_with_best_later_or_skip():
    labels = label_maker_window_steps(_sequences(3), _five(), _minute())
    labels.loc[:, "r_net"] = [-0.1, -1.0, 0.5]
    labels.loc[:, "model_target_valid"] = True
    labels.loc[:, "path_observed"] = True
    base = TailDecisionDataset(
        decisions=labels[["window_id", "step"]].copy(),
        tabular=np.ones((3, 1), dtype=np.float32),
        tabular_features=("channel_r2",),
        sequences=None,
    )
    decisions = build_cost_aware_dataset(base, labels).decisions
    assert decisions.enter_advantage_target.tolist() == pytest.approx([-0.6, -1.5, 0.5])


def test_fee_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="maker_entry_bps"):
        MakerExecutionConfig(maker_entry_bps=-1.0)
