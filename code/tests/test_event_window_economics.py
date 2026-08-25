from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from evaluation.event_window_economics import (
    EventLabelConfig,
    label_window_steps,
    replay_first_crossing,
    sweep_event_thresholds,
)


START = pd.Timestamp("2024-01-01", tz="UTC")
END = pd.Timestamp("2024-01-03", tz="UTC")


def _sequences(*, valid_steps: int = 1) -> SimpleNamespace:
    source = np.full((1, 12), np.datetime64("NaT"), dtype="datetime64[ns]")
    decision = np.full_like(source, np.datetime64("NaT"))
    source_times = pd.date_range(
        "2024-01-01 00:35", periods=12, freq="5min", tz="UTC"
    )
    decision_times = source_times + pd.Timedelta("5min")
    source[0] = source_times.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    decision[0] = decision_times.tz_localize(None).to_numpy(dtype="datetime64[ns]")
    valid = np.zeros((1, 12), dtype=bool)
    valid[0, :valid_steps] = True
    return SimpleNamespace(
        metadata=pd.DataFrame(
            {
                "window_id": ["w1"],
                "channel_episode_id": ["episode-1"],
                "side": ["long"],
            }
        ),
        source_bar_times=source,
        decision_times=decision,
        decision_valid=valid,
    )


def _five_bars() -> pd.DataFrame:
    index = pd.date_range(
        "2023-12-31 23:40", "2024-01-01 01:30", freq="5min", tz="UTC"
    )
    return pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.4,
            "low": 99.0,
            "close": 100.0,
        },
        index=index,
    )


def _minute_path(*, tie: bool = False, gap_after_tp: bool = False) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:40", periods=180, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": 100.0,
            "high": 100.2,
            "low": 99.5,
            "close": 100.0,
        },
        index=index,
    )
    entry_time = pd.Timestamp("2024-01-01 00:40", tz="UTC")
    frame.loc[entry_time, "high"] = 103.0
    if tie:
        frame.loc[entry_time, "low"] = 98.0
    if gap_after_tp:
        frame = frame.drop(frame.index[1])
    return frame


def test_label_enters_next_open_and_freezes_structural_rr2_geometry():
    labels = label_window_steps(_sequences(), _five_bars(), _minute_path())
    row = labels.query("window_id == 'w1' and step == 0").iloc[0]
    assert row.entry_time == row.decision_time
    assert row.entry_time == row.source_bar_time + pd.Timedelta("5min")
    assert row.entry == _minute_path().loc[row.entry_time, "open"]
    assert row.target - row.entry == pytest.approx(2.0 * (row.entry - row.stop))
    assert row.outcome == "tp"
    assert row.label_end == row.exit_time + pd.Timedelta("1min")


def test_same_minute_tp_sl_tie_is_stop():
    row = label_window_steps(_sequences(), _five_bars(), _minute_path(tie=True)).iloc[0]
    assert row.outcome == "sl"
    assert row.r_gross == pytest.approx(-1.0)


def test_gap_after_early_tp_does_not_censor_already_closed_label():
    row = label_window_steps(
        _sequences(), _five_bars(), _minute_path(gap_after_tp=True)
    ).iloc[0]
    assert row.outcome == "tp"
    assert row.path_observed
    assert row.model_target_valid


def test_gap_before_resolution_separates_geometry_from_path_validity():
    path = _minute_path()
    entry_time = pd.Timestamp("2024-01-01 00:40", tz="UTC")
    path.loc[entry_time, ["high", "low", "close"]] = [100.2, 99.5, 100.0]
    path = path.drop(entry_time + pd.Timedelta(minutes=1))
    row = label_window_steps(_sequences(), _five_bars(), path).iloc[0]
    assert row.geometry_valid
    assert not row.path_observed
    assert not row.model_target_valid
    assert row.outcome == "censored"
    assert pd.isna(row.r_net)
    assert row.label_end == entry_time + pd.Timedelta(minutes=2)


def _policy_labels(*, first_censored: bool = False) -> pd.DataFrame:
    outcomes = ["censored" if first_censored else "timeout", "tp", "sl"]
    values = [np.nan if first_censored else -0.1, 1.8, -1.2]
    return pd.DataFrame(
        {
            "window_id": ["w1"] * 3,
            "channel_episode_id": ["e1"] * 3,
            "side": ["long"] * 3,
            "step": [0, 1, 2],
            "decision_time": pd.date_range(
                "2024-01-01 00:05", periods=3, freq="5min", tz="UTC"
            ),
            "entry_time": pd.date_range(
                "2024-01-01 00:05", periods=3, freq="5min", tz="UTC"
            ),
            "geometry_valid": [True, True, True],
            "path_observed": [not first_censored, True, True],
            "model_target_valid": [not first_censored, True, True],
            "outcome": outcomes,
            "r_net": values,
        }
    )


def _scores(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1"] * len(values),
            "step": range(len(values)),
            "score": values,
        }
    )


def test_policy_takes_first_crossing_and_only_one_trade_per_window():
    replay = replay_first_crossing(
        _scores([-0.1, 0.2, 0.8]),
        _policy_labels(),
        threshold=0.15,
        start=START,
        end=END,
    )
    assert replay.trades.groupby("window_id").size().max() == 1
    assert replay.trades.iloc[0]["step"] == 1


def test_censored_first_crossing_consumes_window_instead_of_becoming_wait():
    replay = replay_first_crossing(
        _scores([0.3, 0.8]),
        _policy_labels(first_censored=True).iloc[:2],
        threshold=0.2,
        start=START,
        end=END,
    )
    assert replay.trades[["step", "outcome"]].iloc[0].tolist() == [0, "censored"]
    assert len(replay.trades) == 1
    assert replay.summary["censored_trades"] == 1


def test_invalid_geometry_crossing_does_not_consume_window():
    labels = _policy_labels()
    labels.loc[0, "geometry_valid"] = False
    replay = replay_first_crossing(
        _scores([0.9, 0.3, 0.8]),
        labels,
        threshold=0.2,
        start=START,
        end=END,
    )
    assert replay.trades.iloc[0]["step"] == 1


def _frontier_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for day in range(2):
        for number in range(6):
            rows.append(
                {
                    "window_id": f"w-{day}-{number}",
                    "channel_episode_id": f"e-{day}",
                    "side": "long" if number % 2 == 0 else "short",
                    "step": 0,
                    "decision_time": START + pd.Timedelta(days=day, minutes=5 * number),
                    "entry_time": START + pd.Timedelta(days=day, minutes=5 * number),
                    "geometry_valid": True,
                    "path_observed": True,
                    "model_target_valid": True,
                    "outcome": "tp" if number < 4 else "sl",
                    "r_net": 1.0 if number < 4 else -1.0,
                    "score": number / 10,
                }
            )
    labels = pd.DataFrame(rows).drop(columns="score")
    scores = pd.DataFrame(rows)[["window_id", "step", "score"]]
    return scores, labels


def test_threshold_frontier_marks_two_to_five_as_admissible():
    scores, labels = _frontier_inputs()
    table = sweep_event_thresholds(scores, labels, start=START, end=END)
    expected = table["trades_per_day"].between(2.0, 5.0)
    assert table["frequency_admissible"].equals(expected)
    economic = (
        (table["threshold"] >= 0.0)
        & (table["mean_net_r"] > 0.0)
        & (table["total_net_r"] > 0.0)
    )
    assert table["economic_admissible"].equals(economic)
    assert table["policy_admissible"].equals(expected & economic)
    assert table["threshold"].ge(0.0).all()


def test_daily_frequency_contains_zero_trade_calendar_days():
    replay = replay_first_crossing(
        _scores([0.5, -0.5, -0.5]),
        _policy_labels(),
        threshold=0.2,
        start=START,
        end=END,
    )
    assert replay.daily_frequency.index.tolist() == [START, START + pd.Timedelta(days=1)]
    assert replay.daily_frequency["attempted_trades"].tolist() == [1, 0]
    assert replay.summary["trades_per_day"] == pytest.approx(0.5)


def test_negative_threshold_is_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        replay_first_crossing(
            _scores([0.1, 0.2, 0.3]),
            _policy_labels(),
            threshold=-0.01,
            start=START,
            end=END,
        )


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError, match="rr_multiple"):
        EventLabelConfig(rr_multiple=0.0)
