"""Development guard, reproducibility and reader-artifact tests for Notebook J."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import experiments.run_event_window_tcn as runner
from experiments.run_event_window_tcn import (
    REQUIRED_ARTIFACTS,
    EventWindowStudyConfig,
    _same_entries_rr3,
    episode_bootstrap,
    protocol_dict,
    run_event_window_study,
)


def _ohlcv(index: pd.DatetimeIndex, *, minutes: int) -> pd.DataFrame:
    year_start = pd.to_datetime(index.year.astype(str) + "-01-01", utc=True)
    elapsed_hours = (index - year_start).total_seconds().to_numpy() / 3600.0
    position = np.arange(len(index), dtype=float)
    close = (
        100.0
        + 0.015 * elapsed_hours
        + 1.5 * np.sin(elapsed_hours / 5.0)
        + 0.4 * np.sin(elapsed_hours / 1.3)
        + 0.25 * np.sin(elapsed_hours / 0.08)
    )
    open_ = close - 0.02 * np.cos(elapsed_hours / 2.0)
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) + 0.08,
            "low": np.minimum(open_, close) - 0.08,
            "close": close,
            "volume": 10.0 + np.mod(position, 17.0),
            "quote_volume": (10.0 + np.mod(position, 17.0)) * close,
            "count": 20.0 + np.mod(position, 13.0),
            "taker_buy_base": 5.0 + np.mod(position, 7.0) / 10.0,
            "minute_count": minutes,
        },
        index=index,
    )


@pytest.fixture
def tiny_data_root(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    blocks = (
        (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-01-09", tz="UTC")),
        (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-01-09", tz="UTC")),
    )

    def index(freq: str) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(
            np.concatenate(
                [
                    pd.date_range(start, end, freq=freq, inclusive="left").to_numpy()
                    for start, end in blocks
                ]
            )
        ).tz_convert("UTC")

    minute_index = index("1min")
    five_index = index("5min")
    hour_index = index("1h")
    position_index = index("15min")

    minute = _ohlcv(minute_index, minutes=1).drop(columns="minute_count")
    five = _ohlcv(five_index, minutes=5)
    hourly = _ohlcv(hour_index, minutes=60)
    x = np.arange(len(position_index), dtype=float)
    positioning = pd.DataFrame(
        {
            "funding_rate": 0.0001 * np.sin(x / 19.0),
            "sum_open_interest": 1_000_000.0 + 100.0 * x,
            "toptrader_ls": 1.0 + 0.05 * np.sin(x / 23.0),
            "taker_ls": 1.0 + 0.04 * np.cos(x / 17.0),
            "positioning_stale": False,
            "positioning_age_min": 0.0,
        },
        index=position_index,
    )
    symbol = "btcusdt"
    minute.to_parquet(root / f"{symbol}_1m_2021_2026.parquet")
    five.to_parquet(root / f"{symbol}_5min_2021_2026.parquet")
    hourly.to_parquet(root / f"{symbol}_1h_2021_2026.parquet")
    positioning.to_parquet(root / f"{symbol}_positioning_15min_2021_2026.parquet")
    return root


def _paired_episode_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel_episode_id": ["a", "a", "b", "c"],
            "model_net_r": [1.0, -0.25, 0.5, 0.0],
            "baseline_net_r": [0.25, 0.0, -0.5, 0.1],
        }
    )


def test_runner_rejects_non_dev_before_loading(monkeypatch):
    monkeypatch.setattr(runner, "load_inputs", lambda *a, **k: pytest.fail("loaded"))
    with pytest.raises(PermissionError, match="development-only"):
        run_event_window_study(stage="forward")


def test_protocol_freezes_pooled_one_trade_frequency_and_future_boundary():
    config = EventWindowStudyConfig()
    with pytest.raises(FrozenInstanceError):
        config.risk_pct = 2.0

    protocol = protocol_dict(config, stage="dev")
    assert protocol["side_dataset"] == "pooled"
    assert protocol["max_trades_per_window"] == 1
    assert protocol["primary_score_threshold"] == 0.0
    assert protocol["expected_full_dev_windows"] == 14_510
    assert protocol["desired_trades_per_day"] == [3.0, 5.0]
    assert protocol["admissible_trades_per_day"] == [2.0, 5.0]
    assert protocol["selection_rr"] == 2.0
    assert protocol["sensitivity_rr"] == 3.0
    assert protocol["sensitivity_can_select"] is False
    assert protocol["forward_or_lockbox_loaded"] is False


def test_episode_bootstrap_is_seeded_and_reports_paired_baseline_delta():
    first = episode_bootstrap(_paired_episode_ledger(), draws=50, seed=42)
    second = episode_bootstrap(_paired_episode_ledger(), draws=50, seed=42)
    pd.testing.assert_frame_equal(first, second)
    assert {
        "model_total_net_r",
        "baseline_total_net_r",
        "delta_total_net_r",
    } <= set(first)


def test_smoke_run_writes_reader_contract(tmp_path, tiny_data_root):
    result = run_event_window_study(
        stage="dev",
        data_root=tiny_data_root,
        output_root=tmp_path,
        smoke=True,
    )

    for name in REQUIRED_ARTIFACTS:
        assert (result.run_dir / name).exists(), name
    assert result.run_dir.name == "smoke"
    assert not (tmp_path / "latest_dev.json").exists()
    assert result.summary["forward_or_lockbox_loaded"] is False
    assert result.summary["smoke"] is True
    required_summary = {
        "windows_per_calendar_day",
        "trades_per_calendar_day",
        "zero_trade_days",
        "mean_net_r",
        "total_net_r",
        "profit_factor",
        "tp_rate",
        "sl_rate",
        "timeout_rate",
        "censored_rate",
        "paired_delta_total_net_r",
        "paired_delta_ci_low",
        "paired_delta_ci_high",
        "days_profit_ge_2pct",
        "days_profit_ge_3pct",
        "days_profit_ge_5pct",
        "run_hash",
        "rr3_same_entries",
        "rr3_attempted_trades",
        "rr3_mean_net_r",
        "rr3_total_net_r",
    }
    assert required_summary <= set(result.summary)
    assert result.summary["oof_score_rows"] > 0
    assert result.summary["window_calendar_days"] == 28
    protocol = json.loads((result.run_dir / "protocol.json").read_text(encoding="utf-8"))
    assert pd.Timestamp(protocol["max_loaded_timestamp"]) < pd.Timestamp(
        protocol["read_end_exclusive"]
    )
    profile = pd.read_csv(result.run_dir / "feature_profile.csv")
    assert {"finite_fraction", "missing_fraction", "std"} <= set(profile)
    example = pd.read_parquet(result.run_dir / "example_windows.parquet")
    assert "selected" in example
    frontier = pd.read_csv(result.run_dir / "threshold_frontier.csv")
    primary = frontier.loc[frontier["threshold"].eq(0.0)].iloc[0]
    assert bool(primary["primary_registered"])
    assert not bool(primary["exploratory"])
    assert bool(primary["can_establish_success"])


def test_rr3_sensitivity_requires_the_exact_selected_keys():
    selected = pd.DataFrame({"window_id": ["w1"], "step": [2], "r_net": [0.2]})
    matching = pd.DataFrame(
        {
            "window_id": ["w1"],
            "step": [2],
            "r_net": [0.4],
            "path_observed": [True],
            "outcome": ["timeout"],
        }
    )
    merged = _same_entries_rr3(selected, matching)
    assert merged.loc[0, "rr3_r_net"] == pytest.approx(0.4)
    with pytest.raises(ValueError, match="same RR3 key"):
        _same_entries_rr3(selected, matching.assign(step=3))


def test_complete_smoke_resumes_without_loading_again(
    monkeypatch, tmp_path, tiny_data_root
):
    first = run_event_window_study(
        data_root=tiny_data_root, output_root=tmp_path, smoke=True
    )
    monkeypatch.setattr(runner, "load_inputs", lambda *a, **k: pytest.fail("loaded"))
    second = run_event_window_study(
        data_root=tiny_data_root, output_root=tmp_path, smoke=True
    )

    assert second.run_dir == first.run_dir
    assert second.summary["resumed"] is True
    state = json.loads((first.run_dir / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "complete"


def test_atomic_json_retries_a_transient_windows_lock(monkeypatch, tmp_path):
    original_replace = Path.replace
    attempts = []

    def flaky_replace(path, target):
        attempts.append(Path(target))
        if len(attempts) == 1:
            raise PermissionError(5, "transient lock")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    target = tmp_path / "state.json"
    runner._atomic_json(target, {"status": "running"})
    assert len(attempts) == 2
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "running"}
