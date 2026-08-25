from __future__ import annotations

import importlib

import numpy as np
import pandas as pd


def _runner():
    return importlib.import_module(
        "experiments.run_channel_vs_volatility_ablation"
    )


def test_w_protocol_is_exact_matched_frequency_and_dev_only():
    runner = _runner()
    protocol = runner.protocol_dict()

    assert protocol["study"] == "notebook_w_channel_vs_volatility_ablation"
    assert protocol["stage"] == "dev"
    assert protocol["target_activations_per_day"] == 2939 / 1096
    assert protocol["matched_fold_counts"] == {
        "2022H2": 236,
        "2023H1": 541,
        "2023H2": 305,
        "2024H1": 910,
        "2024H2": 532,
        "2025H1": 415,
    }
    assert protocol["models"] == ["logreg", "xgboost"]
    assert protocol["window_sources"] == ["channel", "channel_blind"]
    assert protocol["round_trip_cost_bps"] == 10.0
    assert protocol["target_multiple_b"] == 2.0
    assert protocol["hold_minutes"] == 120
    assert protocol["same_minute_ambiguity"] == "stop_first"
    assert protocol["forward_or_lockbox_loaded"] is False
    assert not any(
        "channel" in feature.lower()
        for feature in (
            *protocol["opportunity_features"],
            *protocol["direction_features"],
        )
    )


def test_matched_topk_refractory_is_exact_deterministic_and_label_blind():
    runner = _runner()
    frame = pd.DataFrame(
        {
            "decision_time": pd.date_range(
                "2024-01-01", periods=10, freq="20min", tz="UTC"
            ),
            "score": np.linspace(0.1, 1.0, 10),
            "future_outcome": np.arange(10),
        }
    )

    selected = runner.matched_topk_refractory(
        frame,
        count=3,
        cooldown_minutes=60,
        score_column="score",
    )
    repeated = runner.matched_topk_refractory(
        frame.sample(frac=1.0, random_state=7),
        count=3,
        cooldown_minutes=60,
        score_column="score",
    )

    assert len(selected) == 3
    assert selected["decision_time"].tolist() == repeated["decision_time"].tolist()
    gaps = selected["decision_time"].sort_values().diff().dropna()
    assert gaps.ge(pd.Timedelta(minutes=60)).all()
    assert selected["future_outcome"].tolist() == repeated["future_outcome"].tolist()


def test_first_touch_labels_distinguish_up_down_ambiguous_and_no_hit():
    runner = _runner()
    index = pd.date_range("2024-01-01", periods=4, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {
            "open": [100.0] * 4,
            "high": [101.2, 100.2, 101.2, 100.2],
            "low": [99.8, 98.8, 98.8, 99.8],
            "close": [100.5, 99.5, 100.0, 100.0],
        },
        index=index,
    )

    labels = runner.first_touch_labels(
        pd.DataFrame(
            {
                "decision_time": index,
                "reference_price": [100.0] * 4,
                "adaptive_barrier_bps": [100.0] * 4,
            }
        ),
        minute,
        horizon_minutes=1,
    )

    assert labels["move_label"].tolist() == [
        "up_big",
        "down_big",
        "ambiguous",
        "no_big_move",
    ]
    assert labels["opportunity"].tolist() == [1, 1, 1, 0]
    assert labels["direction_valid"].tolist() == [True, True, False, False]
    assert labels["path_complete"].all()


def test_first_touch_continuity_normalises_microsecond_datetime_indices():
    runner = _runner()
    index = pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC").as_unit(
        "us"
    )
    minute = pd.DataFrame(
        {
            "open": [100.0] * 3,
            "high": [100.1] * 3,
            "low": [99.9] * 3,
            "close": [100.0] * 3,
        },
        index=index,
    )
    labels = runner.first_touch_labels(
        pd.DataFrame(
            {
                "decision_time": [index[0]],
                "reference_price": [100.0],
                "adaptive_barrier_bps": [100.0],
            }
        ),
        minute,
        horizon_minutes=3,
    )

    assert labels.loc[0, "path_complete"]
    assert labels.loc[0, "move_label"] == "no_big_move"


def test_decision_universe_excludes_bar_whose_close_equals_dev_end():
    runner = _runner()
    minute_index = pd.date_range(
        "2025-06-30 21:50", "2025-06-30 23:59", freq="1min", tz="UTC"
    )
    minute = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 10.0,
            "quote_volume": 1000.0,
            "trade_count": 10.0,
            "taker_buy_base": 5.0,
        },
        index=minute_index,
    )
    five_index = pd.date_range(
        "2025-06-30 21:50", "2025-06-30 23:55", freq="5min", tz="UTC"
    )
    five = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 50.0,
            "trade_count": 50.0,
            "taker_buy_base": 25.0,
        },
        index=five_index,
    )
    positioning_index = pd.date_range(
        "2025-06-30 21:45", "2025-06-30 23:45", freq="15min", tz="UTC"
    )
    positioning = pd.DataFrame(
        {
            "funding_rate": 0.0,
            "sum_open_interest": np.linspace(1000.0, 1008.0, len(positioning_index)),
            "toptrader_ls": 1.0,
            "taker_ls": 1.0,
            "positioning_stale": False,
            "positioning_age_min": 0.0,
        },
        index=positioning_index,
    )

    decisions = runner.build_channel_free_decisions(
        minute,
        five,
        positioning,
        channel_times=pd.DatetimeIndex([], tz="UTC"),
    )

    assert decisions["decision_time"].max() < pd.Timestamp("2025-07-01", tz="UTC")
    assert pd.Timestamp("2025-07-01", tz="UTC") not in set(decisions["decision_time"])


def test_channel_retention_requires_positive_economics_and_increment():
    runner = _runner()

    assert runner.channel_retention_decision(
        channel_ci_low=0.01,
        channel_minus_blind_ci_low=0.02,
        opportunity_delta_ci_low=0.001,
        leakage_passed=True,
        frequency_matched=True,
    )
    assert not runner.channel_retention_decision(
        channel_ci_low=-0.01,
        channel_minus_blind_ci_low=0.02,
        opportunity_delta_ci_low=0.001,
        leakage_passed=True,
        frequency_matched=True,
    )
    assert not runner.channel_retention_decision(
        channel_ci_low=0.01,
        channel_minus_blind_ci_low=-0.001,
        opportunity_delta_ci_low=0.001,
        leakage_passed=True,
        frequency_matched=True,
    )
