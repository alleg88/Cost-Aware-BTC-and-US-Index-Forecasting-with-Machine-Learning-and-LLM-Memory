"""Feature-contract and alignment tests for Notebook B decision rows."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from data.build_grids import build_grid
from experiments.channel_window_dataset import (
    CHANNEL_WINDOW_FEATURES,
    FORBIDDEN_WINDOW_FEATURES,
    DecisionFrames,
    WindowLabelConfig,
    build_decision_candidates,
    label_decision_candidates,
)


def _minute_frame() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=360, freq="1min", tz="UTC")
    x = np.arange(len(idx), dtype=float)
    close = 100.0 + 0.003 * x + 0.08 * np.sin(x / 11.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 0.04
    low = np.minimum(open_, close) - 0.04
    volume = 10.0 + x % 7
    return pd.DataFrame(
        {
            "open": open_, "high": high, "low": low, "close": close,
            "volume": volume, "taker_buy_base": volume * (0.48 + 0.04 * np.sin(x / 9)),
            "minute_count": 1,
        },
        index=idx,
    )


def _hourly_with_channels(minute: pd.DataFrame) -> pd.DataFrame:
    hourly = build_grid(minute, "1h")
    hourly["channel_slope"] = 8.0
    hourly["channel_r2"] = 0.65
    hourly["channel_lower"] = hourly["close"] - 1.5
    hourly["channel_mid"] = hourly["close"]
    hourly["channel_upper"] = hourly["close"] + 1.5
    hourly["channel_width"] = 3.0
    hourly["channel_confluence"] = 1
    return hourly


def _frames() -> DecisionFrames:
    minute = _minute_frame()
    five = build_grid(minute, "5min")
    fifteen = build_grid(minute, "15min")
    hourly = _hourly_with_channels(minute)
    daily_idx = pd.date_range("2024-05-26", periods=220, freq="1D", tz="UTC")
    daily = pd.DataFrame(
        {"close": 80.0 + np.arange(len(daily_idx), dtype=float) * 0.1},
        index=daily_idx,
    )
    pos_idx = pd.date_range("2025-01-01 00:15", periods=24, freq="15min", tz="UTC")
    positioning = pd.DataFrame(
        {
            "funding_z": np.linspace(-0.5, 0.5, len(pos_idx)),
            "oi_chg_4h": np.linspace(-0.02, 0.02, len(pos_idx)),
            "positioning_stale": False,
            "positioning_age_min": 1.0,
        },
        index=pos_idx,
    )
    return DecisionFrames(minute, five, fifteen, hourly, daily, positioning)


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["window-long-1"],
            "channel_episode_id": [17],
            "side": ["long"],
            "window_start": [pd.Timestamp("2025-01-01 03:00", tz="UTC")],
            "natural_end_time": [pd.Timestamp("2025-01-01 04:00", tz="UTC")],
            "eligible_end_time": [pd.Timestamp("2025-01-01 04:00", tz="UTC")],
            "window_end_reason": ["edge_end"],
            "channel_r2_open": [0.65],
            "channel_regime_open": ["up"],
        }
    )


def test_feature_contract_has_30_causal_columns():
    assert len(CHANNEL_WINDOW_FEATURES) == 30
    assert {
        "positioning_stale",
        "positioning_age_log",
        "minute_count_ratio",
        "edge_recovery_bps_side",
        "bars_since_window_extreme",
        "body_direction_side",
        "reversal_count_3",
        "rejection_mean_3_side",
        "taker_imbalance_mean_3_side",
        "taker_imbalance_delta_side",
        "volume_ratio_12",
    } <= set(CHANNEL_WINDOW_FEATURES)
    assert set(CHANNEL_WINDOW_FEATURES).isdisjoint(FORBIDDEN_WINDOW_FEATURES)


def test_one_minute_and_five_minute_rows_share_windows_and_geometry():
    frames = _frames()
    manifest = _manifest()

    one = build_decision_candidates(manifest, frames, cadence="1min")
    five = build_decision_candidates(manifest, frames, cadence="5min")

    assert len(one) == 60
    assert len(five) == 12
    assert set(five["window_id"]) <= set(one["window_id"])
    aligned = one.loc[one["decision_time"].isin(five["decision_time"])].reset_index(drop=True)
    pd.testing.assert_series_equal(
        aligned["swing_low_15m"], five["swing_low_15m"], check_names=False,
    )
    pd.testing.assert_series_equal(
        aligned["measured_move_15m"], five["measured_move_15m"], check_names=False,
    )


def test_incomplete_current_execution_bar_creates_no_candidate():
    frames = _frames()
    five = frames.five_minute.copy()
    five.loc[pd.Timestamp("2025-01-01 02:55", tz="UTC"), "minute_count"] = 4

    candidates = build_decision_candidates(
        _manifest(), replace(frames, five_minute=five), cadence="5min",
    )

    assert pd.Timestamp("2025-01-01 03:00", tz="UTC") not in set(
        candidates["decision_time"]
    )
    assert len(candidates) == 11


def test_future_changes_cannot_change_prior_candidate_features():
    frames = _frames()
    original = build_decision_candidates(_manifest(), frames, cadence="1min")
    cutoff = pd.Timestamp("2025-01-01 03:30", tz="UTC")
    changed_minute = frames.minute.copy()
    mask = changed_minute.index >= cutoff
    changed_minute.loc[mask, ["open", "high", "low", "close"]] += 50.0
    changed = replace(
        frames,
        minute=changed_minute,
        five_minute=build_grid(changed_minute, "5min"),
        fifteen_minute=build_grid(changed_minute, "15min"),
    )
    perturbed = build_decision_candidates(_manifest(), changed, cadence="1min")

    columns = ["candidate_id", "decision_time", *CHANNEL_WINDOW_FEATURES,
               "swing_low_15m", "swing_high_15m", "measured_move_15m"]
    pd.testing.assert_frame_equal(
        original.loc[original["decision_time"] <= cutoff, columns].reset_index(drop=True),
        perturbed.loc[perturbed["decision_time"] <= cutoff, columns].reset_index(drop=True),
    )


def test_three_faucet_sequence_is_causal_and_resets_for_each_window():
    frames = _frames()
    five = frames.five_minute.copy()
    first_bars = pd.to_datetime(
        ["2025-01-01 02:55", "2025-01-01 03:00", "2025-01-01 03:05"],
        utc=True,
    )
    five.loc[first_bars, ["open", "high", "low", "close", "volume", "taker_buy_base"]] = [
        [100.5, 101.0, 99.5, 100.0, 10.0, 4.0],
        [100.2, 100.4, 99.0, 99.8, 20.0, 8.0],
        [99.7, 100.6, 99.6, 100.4, 30.0, 18.0],
    ]
    reset_bar = pd.Timestamp("2025-01-01 03:15", tz="UTC")
    five.loc[
        reset_bar,
        ["open", "high", "low", "close", "volume", "taker_buy_base"],
    ] = [101.0, 101.3, 100.8, 101.1, 40.0, 24.0]
    manifest = pd.concat(
        [
            _manifest().assign(
                eligible_end_time=pd.Timestamp("2025-01-01 03:15", tz="UTC"),
                natural_end_time=pd.Timestamp("2025-01-01 03:15", tz="UTC"),
            ),
            _manifest().assign(
                window_id="window-long-2",
                channel_episode_id=18,
                window_start=pd.Timestamp("2025-01-01 03:20", tz="UTC"),
                eligible_end_time=pd.Timestamp("2025-01-01 03:25", tz="UTC"),
                natural_end_time=pd.Timestamp("2025-01-01 03:25", tz="UTC"),
            ),
        ],
        ignore_index=True,
    )

    candidates = build_decision_candidates(
        manifest, replace(frames, five_minute=five), cadence="5min"
    ).set_index(["window_id", "decision_time"])

    recovered = candidates.loc[
        ("window-long-1", pd.Timestamp("2025-01-01 03:10", tz="UTC"))
    ]
    assert recovered["edge_recovery_bps_side"] == pytest.approx(
        (100.4 / 99.0 - 1.0) * 1e4
    )
    assert recovered["bars_since_window_extreme"] == 1.0
    assert recovered["body_direction_side"] > 0.0
    assert recovered["taker_imbalance_mean_3_side"] < 0.0
    assert recovered["taker_imbalance_delta_side"] > 0.0

    reset = candidates.loc[
        ("window-long-2", pd.Timestamp("2025-01-01 03:20", tz="UTC"))
    ]
    assert reset["bars_since_window_extreme"] == 0.0
    assert reset["edge_recovery_bps_side"] == pytest.approx(
        (101.1 / 100.8 - 1.0) * 1e4
    )


def test_lifecycle_and_outcome_metadata_are_not_model_inputs():
    candidates = build_decision_candidates(_manifest(), _frames(), cadence="5min")

    assert FORBIDDEN_WINDOW_FEATURES.isdisjoint(CHANNEL_WINDOW_FEATURES)
    assert set(CHANNEL_WINDOW_FEATURES) <= set(candidates.columns)
    assert {"natural_end_time", "eligible_end_time", "window_end_reason"} <= set(
        candidates.columns
    )


def _candidate_fixture() -> pd.DataFrame:
    decisions = pd.to_datetime(
        ["2025-01-01 03:00", "2025-01-01 03:05", "2025-01-01 03:50",
         "2025-01-01 04:10"],
        utc=True,
    )
    source = decisions - pd.Timedelta("5min")
    return pd.DataFrame(
        {
            "candidate_id": ["filled", "unfilled", "cancelled", "censored"],
            "window_id": ["w1", "w1", "w1", "w2"],
            "channel_episode_id": [1, 1, 1, 2],
            "side": ["long"] * 4,
            "cadence": ["5min"] * 4,
            "source_bar_time": source,
            "decision_time": decisions,
            "next_entry_time": decisions,
            "window_start": pd.to_datetime(
                ["2025-01-01 03:00"] * 3 + ["2025-01-01 04:10"], utc=True,
            ),
            "natural_end_time": pd.to_datetime(
                ["2025-01-01 04:00"] * 3 + ["2025-01-01 05:00"], utc=True,
            ),
            "eligible_end_time": pd.to_datetime(
                ["2025-01-01 04:00"] * 3 + ["2025-01-01 05:00"], utc=True,
            ),
            "window_end_reason": ["edge_end"] * 4,
            "decision_close": [100.0] * 4,
            "swing_low_15m": [99.5] * 4,
            "swing_high_15m": [100.5] * 4,
            "measured_move_15m": [1.0] * 4,
            "risk_bps_decision": [49.98] * 4,
            "rr_planned_decision": [2.0] * 4,
        }
    )


def _minute_path() -> pd.DataFrame:
    idx = pd.date_range(
        "2025-01-01 02:50", "2025-01-01 05:00", freq="1min", tz="UTC",
    ).difference(pd.DatetimeIndex([pd.Timestamp("2025-01-01 04:12", tz="UTC")]))
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.02, "low": 100.0, "close": 100.0},
        index=idx,
    )
    minute.loc[pd.Timestamp("2025-01-01 03:00", tz="UTC"), "low"] = 99.90
    minute.loc[pd.Timestamp("2025-01-01 03:01", tz="UTC"), "high"] = 101.10
    return minute


def test_unfilled_and_cancelled_are_zero_but_censored_rows_are_removed():
    labelled = label_decision_candidates(
        _candidate_fixture(), _minute_path(), config=WindowLabelConfig(),
    )

    assert labelled.query("order_status == 'unfilled'")["r_net"].eq(0.0).all()
    assert labelled.query("order_status == 'channel_cancelled'")["r_net"].eq(0.0).all()
    assert "censored" not in set(labelled["order_status"])
    assert set(labelled["candidate_id"]) == {"filled", "unfilled", "cancelled"}


def test_label_interval_covers_order_and_position_lifetime():
    labelled = label_decision_candidates(
        _candidate_fixture(), _minute_path(), config=WindowLabelConfig(),
    )
    row = labelled.set_index("candidate_id").loc["filled"]

    assert row["label_start"] == row["next_entry_time"]
    assert row["label_end"] == row["active_end_time"]
    assert row["label_end"] >= row["label_start"]
    assert row["order_status"] == "filled"
    assert row["r_net"] > 0


def test_unfilled_expiry_and_window_cancellation_have_observed_label_ends():
    labelled = label_decision_candidates(
        _candidate_fixture(), _minute_path(), config=WindowLabelConfig(),
    ).set_index("candidate_id")

    assert labelled.loc["unfilled", "active_end_time"] == pd.Timestamp(
        "2025-01-01 03:25", tz="UTC"
    )
    assert labelled.loc["cancelled", "active_end_time"] == pd.Timestamp(
        "2025-01-01 04:00", tz="UTC"
    )


def test_labels_are_not_changed_by_portfolio_capacity():
    labelled = label_decision_candidates(
        _candidate_fixture(), _minute_path(), config=WindowLabelConfig(),
    )

    assert len(labelled) == 3
    assert labelled["candidate_id"].is_unique
