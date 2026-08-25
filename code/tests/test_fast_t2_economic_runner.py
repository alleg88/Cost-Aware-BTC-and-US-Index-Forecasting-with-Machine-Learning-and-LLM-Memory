"""Protocol guards for the dev-only Fast-T2 economic continuation."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest

from evaluation.channel_window_validation import PurgedFold
from experiments.fast_t2_entry_dataset import ENTRY_FEATURE_COLUMNS


def _module():
    try:
        return import_module("experiments.run_fast_t2_economic_entry")
    except ModuleNotFoundError:
        pytest.fail("economic entry runner is not implemented")


def _entry_protocol():
    return {
        "protocol_hash": "entry-protocol",
        "period_start": "2021-01-01T00:00:00+00:00",
        "period_end_exclusive": "2025-07-01T00:00:00+00:00",
        "forward_or_lockbox_loaded": False,
    }


def _manifest():
    return {
        "dataset_hash": "entry-dataset",
        "decision_ledger_hash": "entry-ledger",
        "max_loaded_timestamp": "2025-06-30T23:59:00+00:00",
    }


def test_economic_protocol_is_dev_only_and_frequency_constrained():
    module = _module()
    protocol = module.build_economic_protocol(_entry_protocol(), _manifest())

    assert protocol["target"] == "winsorised_net_r_enter_minus_skip_0R"
    assert protocol["minimum_trades_per_day"] == 1.0
    assert protocol["arms"] == [
        "ridge_pooled", "catboost_pooled", "catboost_split_side"
    ]
    assert protocol["rr_sensitivity"] == [None, 1.0, 1.5, 2.0]
    assert protocol["period_end_exclusive"] == "2025-07-01T00:00:00+00:00"
    assert protocol["forward_or_lockbox_loaded"] is False
    assert len(protocol["protocol_hash"]) == 64


def test_xgboost_arms_exactly_mirror_catboost_variants():
    module = _module()

    assert module._arm_spec("xgboost_pooled") == ("xgboost", "pooled")
    assert module._arm_spec("xgboost_split_side") == ("xgboost", "split_side")


def test_economic_protocol_rejects_an_open_forward_or_lockbox():
    module = _module()
    opened = {**_entry_protocol(), "forward_or_lockbox_loaded": True}

    with pytest.raises(ValueError, match="sealed"):
        module.build_economic_protocol(opened, _manifest())


def _fold_fixture():
    blocks = [
        ("2021-02-01", 12),
        ("2021-08-01", 12),
        ("2022-02-01", 12),
    ]
    rows = []
    position = 0
    for start, count in blocks:
        for offset, timestamp in enumerate(pd.date_range(start, periods=count, freq="1D", tz="UTC")):
            side_sign = -1.0 if offset % 2 else 1.0
            signal = (offset - count / 2) / count
            features = {name: 0.0 for name in ENTRY_FEATURE_COLUMNS}
            features.update(
                {
                    "side_sign": side_sign,
                    "channel_r2": 0.5 + signal,
                    "rr_proxy": 1.0 + (offset % 4),
                    "minutes_since_t2": float(offset % 3),
                }
            )
            rows.append(
                {
                    "window_id": f"w{position}",
                    "decision_id": f"d{position}",
                    "side": "long" if side_sign > 0 else "short",
                    "channel_episode_id": position,
                    "t2_time": timestamp,
                    "decision_time": timestamp,
                    "entry_time": timestamp,
                    "label_start": timestamp,
                    "label_end": timestamp + pd.Timedelta(minutes=30),
                    "active_end_time": timestamp + pd.Timedelta(minutes=30),
                    "entry_price": 100.0,
                    "stop_price": 99.0 if side_sign > 0 else 101.0,
                    "target_price": 102.0 if side_sign > 0 else 98.0,
                    "exit_time": timestamp + pd.Timedelta(minutes=30),
                    "exit_price": 101.0 if signal > 0 else 99.0,
                    "outcome": "tp" if signal > 0 else "sl",
                    "r_net": 2.0 * signal + 0.2 * side_sign,
                    "label_net_positive": int(2.0 * signal + 0.2 * side_sign > 0),
                    "filled": True,
                    "holding_minutes": 30.0,
                    **features,
                }
            )
            position += 1
    decisions = pd.DataFrame(rows)
    fold = PurgedFold(
        fold_id="2022H1",
        train=np.arange(24, dtype=np.int64),
        valid=np.arange(24, 36, dtype=np.int64),
        train_end=pd.Timestamp("2022-01-01", tz="UTC"),
        valid_start=pd.Timestamp("2022-01-01", tz="UTC"),
        valid_end=pd.Timestamp("2022-07-01", tz="UTC"),
    )
    return decisions, fold


def test_outer_economic_fold_scores_only_validation_and_all_rr_sensitivities():
    module = _module()
    decisions, fold = _fold_fixture()

    scores, entries, actions, frontier, audit = module.score_outer_economic_fold(
        decisions, fold, "ridge_pooled"
    )

    assert set(scores["decision_id"]) == set(decisions.iloc[fold.valid]["decision_id"])
    assert set(entries["rr_label"]) == {"none", "rr_1_0", "rr_1_5", "rr_2_0"}
    assert set(actions["rr_label"]) == {"none", "rr_1_0", "rr_1_5", "rr_2_0"}
    assert set(frontier["rr_label"]) == {"none", "rr_1_0", "rr_1_5", "rr_2_0"}
    entry_ids = {
        label: set(group["decision_id"])
        for label, group in entries.groupby("rr_label")
    }
    assert entry_ids["rr_2_0"] <= entry_ids["rr_1_5"] <= entry_ids["rr_1_0"] <= entry_ids["none"]
    chosen = frontier[frontier["chosen"].astype(bool)]
    assert chosen.groupby("rr_label")["quantile"].first().nunique() == 1
    assert audit["episode_overlap"] == 0
    assert audit["validation_rows"] == len(fold.valid)
    assert audit["model"] == "ridge"
    assert audit["variant"] == "pooled"


def test_rr_sensitivities_reuse_one_threshold_and_never_add_windows():
    module = _module()
    rows = []
    for window, side, start in [
        ("a", "long", "2024-01-01 00:00"),
        ("b", "short", "2024-01-01 01:00"),
    ]:
        times = pd.date_range(start, periods=3, freq="1min", tz="UTC")
        for delay, (score, rr) in enumerate(zip([0.9, 0.8, 0.76], [0.5, 1.2, 2.2])):
            rows.append(
                {
                    "window_id": window,
                    "decision_id": f"{window}:{delay}",
                    "decision_time": times[delay],
                    "entry_time": times[delay],
                    "active_end_time": times[delay] + pd.Timedelta(minutes=10),
                    "side": side,
                    "channel_episode_id": 1 if window == "a" else 2,
                    "score": score,
                    "rr_proxy": rr,
                    "r_net": 0.5,
                    "filled": True,
                    "holding_minutes": 10.0,
                    "minutes_since_t2": delay,
                }
            )
    entries, _ = module.replay_rr_sensitivities(
        pd.DataFrame(rows), threshold=0.75
    )
    windows = {
        label: set(group["window_id"])
        for label, group in entries.groupby("rr_label")
    }

    assert windows["rr_2_0"] <= windows["rr_1_5"] <= windows["rr_1_0"] <= windows["none"]
    assert entries.groupby("rr_label")["outer_threshold"].first().nunique() == 1
