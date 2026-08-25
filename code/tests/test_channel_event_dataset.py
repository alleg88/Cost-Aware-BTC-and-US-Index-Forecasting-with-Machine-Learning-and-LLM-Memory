"""Leakage and schema contracts for the channel-study E7 event table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _signals() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
    signal = np.zeros(10, dtype=int)
    signal[2], signal[5], signal[8] = 1, -1, 1
    return pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0,
         "signal": signal,
         "channel_slope": [10.0] * 5 + [-10.0] * 5,
         "channel_r2": 0.75,
         "channel_mid": 100.0,
         "channel_upper": 105.0,
         "channel_lower": 95.0,
         "channel_width": 10.0,
         "channel_pos": 0.5,
         "channel_confluence": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
         "rsi_regime_pct": np.linspace(0.1, 0.9, 10),
         "taker_imbalance": np.linspace(-0.5, 0.5, 10),
         "funding_z": np.linspace(-1.0, 1.0, 10),
         "oi_chg_4h": np.linspace(-0.02, 0.02, 10),
         "channel_episode_id": [1] * 5 + [2] * 5},
        index=idx,
    )


def _orders(signals: pd.DataFrame) -> pd.DataFrame:
    idx = signals.index
    return pd.DataFrame(
        {"signal_time": [idx[2], idx[5], idx[8]],
         "decision_time": [idx[3], idx[6], idx[9]],
         "side": ["long", "short", "long"],
         "status": ["filled", "unfilled", "censored"],
         "filled": [True, False, True],
         "entry_time": [idx[3], pd.NaT, idx[9]],
         "exit_time": [idx[4], pd.NaT, pd.NaT],
         "entry": [100.0, 100.0, 100.0],
         "stop": [98.0, 102.0, 98.0],
         "target": [105.0, 95.0, 105.0],
         "outcome": ["tp", "unfilled", "censored"],
         "r_net": [0.5, 0.0, np.nan],
         "channel_episode_id": [1, 2, 2]},
    )


def test_e7_uses_only_decision_time_features_and_keeps_unfilled_orders():
    from experiments.channel_event_dataset import (
        CHANNEL_EVENT_FEATURES,
        FORBIDDEN_EVENT_FEATURES,
        build_channel_event_dataset,
    )

    events = build_channel_event_dataset(
        _signals(), _orders(_signals()), swing_lookback=2,
        min_risk_bps=25.0, max_risk_bps=250.0,
    )

    assert len(events) == 2                         # censored row is not labelled
    assert events["order_status"].tolist() == ["filled", "unfilled"]
    assert events["label_net_positive"].tolist() == [1, 0]
    assert events["r_net"].tolist() == [0.5, 0.0]
    assert events["episode_trade_number"].tolist() == [1, 1]
    assert events["channel_confluence"].tolist() == [1, 0]
    assert list(events.loc[:, CHANNEL_EVENT_FEATURES].columns) == list(CHANNEL_EVENT_FEATURES)
    assert set(CHANNEL_EVENT_FEATURES).isdisjoint(FORBIDDEN_EVENT_FEATURES)
    assert set(events.loc[:, CHANNEL_EVENT_FEATURES]).isdisjoint(FORBIDDEN_EVENT_FEATURES)


def test_e7_stop_and_reward_geometry_are_based_on_signal_close():
    from experiments.channel_event_dataset import build_channel_event_dataset

    signals = _signals()
    events = build_channel_event_dataset(
        signals, _orders(signals), swing_lookback=2,
        min_risk_bps=25.0, max_risk_bps=250.0,
    )

    # Long stop = 98 * (1 - 5 bps); feature geometry must not use next Open/fill.
    expected_risk_bps = (100.0 - 98.0 * 0.9995) / 100.0 * 10_000
    assert events.iloc[0]["risk_bps_decision"] == pytest.approx(expected_risk_bps)
    assert events.iloc[0]["rr_planned_decision"] == pytest.approx(
        5.0 / (100.0 - 98.0 * 0.9995)
    )


def test_e7_measured_reward_uses_the_signal_time_swing_range():
    from experiments.channel_event_dataset import build_channel_event_dataset

    signals = _signals()
    events = build_channel_event_dataset(
        signals,
        _orders(signals),
        swing_lookback=2,
        target_mode="measured",
        min_risk_bps=25.0,
        max_risk_bps=250.0,
    )

    risk = 100.0 - 98.0 * 0.9995
    assert events.iloc[0]["rr_planned_decision"] == pytest.approx(4.0 / risk)


def test_e7_applies_the_same_minimum_reward_to_risk_gate_as_policy():
    from experiments.channel_event_dataset import build_channel_event_dataset

    signals = _signals()
    events = build_channel_event_dataset(
        signals,
        _orders(signals),
        swing_lookback=2,
        target_mode="measured",
        min_risk_bps=25.0,
        max_risk_bps=250.0,
        min_rr=2.0,
    )

    assert events.empty


def test_grid_loader_never_returns_rows_at_or_after_stage_end(tmp_path, monkeypatch):
    from experiments import channel_study

    idx = pd.date_range("2026-03-31 22:00", periods=6, freq="1h", tz="UTC")
    pd.DataFrame({"close": np.arange(6.0)}, index=idx).to_parquet(
        tmp_path / "btcusdt_1h_2021_2026.parquet"
    )
    monkeypatch.setattr(channel_study, "DATA", tmp_path)

    got = channel_study._load_grid(
        "1h", "BTCUSDT", start=pd.Timestamp("2026-03-31 23:00", tz="UTC"),
        end=pd.Timestamp("2026-04-01 01:00", tz="UTC"),
    )
    assert got.index.tolist() == idx[1:3].tolist()


def test_run_tag_changes_when_execution_policy_changes():
    from dataclasses import replace
    from experiments.channel_study import ChannelConfig, run_tag

    base = ChannelConfig()
    assert run_tag(base, "dev") != run_tag(replace(base, max_concurrent=7), "dev")
    assert run_tag(base, "dev") != run_tag(replace(base, maker_fee_bps=3.0), "dev")


def test_cli_exposes_the_frozen_candidate_and_execution_policy():
    from experiments.channel_study import config_from_args, parse_args

    args = parse_args([
        "--stage", "dev", "--grid", "5min",
        "--rsi-oversold", "100", "--rsi-overbought", "0",
        "--no-confirmation", "--max-concurrent", "7",
        "--entry-mode", "maker_limit", "--maker-fee-bps", "2",
        "--taker-fee-bps", "5", "--max-trades-per-day", "12",
    ])
    cfg = config_from_args(args)
    assert cfg.grid == "5min"
    assert not cfg.require_confirmation
    assert cfg.max_concurrent == 7
    assert cfg.entry_mode == "maker_limit"
    assert cfg.maker_fee_bps == 2.0
    assert cfg.taker_fee_bps == 5.0
    assert cfg.max_trades_per_day == 12


def test_cli_defaults_to_the_frozen_measured_confluence_policy():
    from experiments.channel_study import config_from_args, parse_args

    cfg = config_from_args(parse_args([]))

    assert cfg.channel_windows == (60, 90, 120)
    assert cfg.min_channel_agreement == 2
    assert 180 not in cfg.channel_windows
    assert cfg.channel_window == 60
    assert cfg.target_mode == "measured"
    assert cfg.min_r2 == pytest.approx(0.40)
    assert cfg.min_risk_bps == pytest.approx(40.0)
    assert cfg.min_rr == pytest.approx(1.5)
    assert cfg.max_hold_bars == 288
    assert cfg.max_concurrent is None


def test_positioning_features_are_timestamped_when_the_source_bar_closes():
    from experiments.channel_event_dataset import prepare_positioning_features

    idx = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    raw = pd.DataFrame(
        {"funding_rate": [0.0, 1.0, 2.0],
         "sum_open_interest": [100.0, 110.0, 121.0],
         "positioning_stale": [False, False, True]},
        index=idx,
    )
    got = prepare_positioning_features(
        raw, bar_size="15min", funding_window=2, oi_change_periods=2,
    )
    assert got.index[0] == idx[0] + pd.Timedelta("15min")
    assert pd.isna(got.loc[idx[2] + pd.Timedelta("15min"), "oi_chg_4h"])
    assert got.loc[idx[1] + pd.Timedelta("15min"), "funding_z"] == pytest.approx(
        np.sqrt(0.5)
    )


def test_e7_drops_a_row_when_any_allowed_feature_is_unknown():
    from experiments.channel_event_dataset import build_channel_event_dataset

    signals = _signals()
    signals.loc[signals.index[2], "funding_z"] = np.nan
    events = build_channel_event_dataset(
        signals, _orders(signals), swing_lookback=2,
        min_risk_bps=25.0, max_risk_bps=250.0,
    )
    assert events["order_status"].tolist() == ["unfilled"]


def test_regime_percentile_is_causal_and_ranks_the_current_observation():
    from experiments.channel_event_dataset import causal_rolling_percentile

    base = pd.Series([1.0, 3.0, 2.0, 4.0])
    changed_future = pd.Series([1.0, 3.0, 2.0, -100.0])
    got = causal_rolling_percentile(base, window=3, min_periods=2)
    altered = causal_rolling_percentile(changed_future, window=3, min_periods=2)
    assert got.iloc[1] == pytest.approx(1.0)
    assert got.iloc[2] == pytest.approx(2 / 3)
    pd.testing.assert_series_equal(got.iloc[:3], altered.iloc[:3])
