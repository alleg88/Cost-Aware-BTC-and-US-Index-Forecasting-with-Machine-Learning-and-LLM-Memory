"""Behavior tests for the Strict/Fast-T2 event-level model study."""

import numpy as np
import pandas as pd
import pytest

from evaluation.channel_window_validation import expanding_purged_folds
from experiments.two_trigger_model_study import (
    FEATURE_COLUMNS,
    MODEL_NAMES,
    EventLabelConfig,
    build_event_dataset,
    build_event_sequences,
    run_oof_models,
    split_statistics,
    summarise_oof_predictions,
)


def _minute_path(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2023-12-31 23:40", periods=periods, freq="1min", tz="UTC")
    close = 104.1 + np.linspace(0.0, 0.2, periods)
    frame = pd.DataFrame(
        {
            "open": close - 0.02,
            "high": close + 0.08,
            "low": close - 0.08,
            "close": close,
            "volume": 10.0,
            "taker_buy_base": 5.5,
            "count": 100,
        },
        index=index,
    )
    frame.loc[pd.Timestamp("2024-01-01 00:09", tz="UTC"), "open"] = 104.2
    frame.loc[pd.Timestamp("2024-01-01 00:12", tz="UTC"), "high"] = 106.2
    return frame


def _channel_context() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "decision_time": pd.to_datetime(
                ["2024-01-01 00:00", "2024-01-01 00:05"], utc=True
            ),
            "open": [103.5, 103.7],
            "high": [104.0, 104.0],
            "low": [102.9, 102.8],
            "close": [103.7, 103.8],
            "channel_lower": [101.9, 102.0],
            "channel_upper": [105.9, 106.0],
            "channel_r2": [0.65, 0.70],
            "channel_regime": ["up", "up"],
            "channel_confluence": [1, 1],
            "channel_confluence_count": [2, 3],
            "channel_episode_id": [7, 7],
            "minute_count": [5, 5],
        }
    )


def _fast_window() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "side": ["long"],
            "channel_episode_id": [7],
            "t1_time": [pd.Timestamp("2024-01-01 00:05", tz="UTC")],
            "t2_time": [pd.Timestamp("2024-01-01 00:09", tz="UTC")],
            "window_start": [pd.Timestamp("2024-01-01 00:09", tz="UTC")],
            "cooldown_until": [pd.Timestamp("2024-01-01 01:09", tz="UTC")],
            "confirmation_lag_minutes": [4],
            "confirmation_buffer_bps": [2.0],
            "confirmation_price": [104.15],
            "t1_high": [104.0],
            "t1_low": [102.8],
        }
    )


def test_model_contest_is_limited_to_the_four_approved_models():
    assert MODEL_NAMES == ("logreg", "catboost", "xgboost", "gru")


def test_event_dataset_labels_next_open_and_frozen_opposite_rail_after_costs():
    events, audit = build_event_dataset(
        _fast_window(),
        _channel_context(),
        _minute_path(),
        arm="fast_t2_2bps",
        config=EventLabelConfig(),
    )

    assert audit["raw_windows"] == 1
    assert audit["labelled_events"] == 1
    assert len(events) == 1
    event = events.iloc[0]
    assert event["entry_time"] == pd.Timestamp("2024-01-01 00:09", tz="UTC")
    assert event["entry_price"] == 104.2
    assert event["target_price"] == 106.0
    assert event["outcome"] == "tp"
    assert event["label_end"] == pd.Timestamp("2024-01-01 00:12", tz="UTC")
    assert event["r_net"] > 0
    assert set(FEATURE_COLUMNS).isdisjoint(
        {"r_net", "outcome", "entry_price", "exit_price", "label_end"}
    )


def test_model_first_dataset_keeps_tight_risk_and_low_rr_for_the_model():
    window = _fast_window()
    window.loc[0, ["t1_high", "t1_low", "confirmation_price"]] = [
        104.0, 103.95, 104.02
    ]
    channel = _channel_context()
    channel.loc[channel.index[-1], "channel_upper"] = 104.10

    events, audit = build_event_dataset(
        window,
        channel,
        _minute_path(),
        arm="fast_t2_2bps",
        config=EventLabelConfig(),
    )

    assert audit["geometry_rejected"] == 0
    assert audit["labelled_events"] == 1
    assert events.iloc[0]["risk_bps_decision"] < 25.0
    assert events.iloc[0]["rr_planned_decision"] < 1.2


def test_event_sequence_ends_before_decision_bar_and_never_reads_current_minute():
    minute = _minute_path()
    events, _ = build_event_dataset(
        _fast_window(), _channel_context(), minute,
        arm="fast_t2_2bps", config=EventLabelConfig(),
    )
    before = build_event_sequences(events, minute, sequence_length=5)
    changed_current = minute.copy()
    changed_current.loc[pd.Timestamp("2024-01-01 00:09", tz="UTC"), "close"] = 999.0
    after_current = build_event_sequences(events, changed_current, sequence_length=5)
    changed_last_closed = minute.copy()
    changed_last_closed.loc[pd.Timestamp("2024-01-01 00:08", tz="UTC"), "close"] = 999.0
    after_last_closed = build_event_sequences(events, changed_last_closed, sequence_length=5)

    np.testing.assert_allclose(before, after_current)
    assert not np.allclose(before, after_last_closed)


def test_split_statistics_reports_train_validation_and_purged_percentages():
    decision = pd.to_datetime(
        [
            "2021-02-01", "2021-04-01", "2021-06-01", "2021-08-01",
            "2022-02-01", "2022-04-01",
        ],
        utc=True,
    )
    events = pd.DataFrame(
        {
            "decision_time": decision,
            "label_start": decision,
            "label_end": decision + pd.Timedelta(hours=1),
            "channel_episode_id": np.arange(1, 7),
        }
    )
    folds = expanding_purged_folds(events)
    stats = split_statistics(events, folds)
    first = stats.loc[stats["fold_id"] == "2022H1"].iloc[0]

    assert first["train_rows"] == 4
    assert first["validation_rows"] == 2
    assert first["purged_rows"] == 0
    assert first["train_pct"] == pytest.approx(66.6666667)
    assert first["validation_pct"] == pytest.approx(33.3333333)
    assert first["train_pct"] + first["validation_pct"] + first["purged_pct"] == pytest.approx(100.0)


def test_oof_runner_uses_only_past_training_rows_and_reports_economics():
    rng = np.random.default_rng(7)
    train_time = pd.date_range("2021-01-01", periods=40, freq="7D", tz="UTC")
    valid_time = pd.date_range("2022-01-03", periods=12, freq="7D", tz="UTC")
    decision = train_time.append(valid_time)
    events = pd.DataFrame(rng.normal(size=(len(decision), len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    events["candidate_id"] = [f"event-{i}" for i in range(len(events))]
    events["side"] = np.where(np.arange(len(events)) % 2, "long", "short")
    events["decision_time"] = decision
    events["label_start"] = decision
    events["label_end"] = decision + pd.Timedelta(hours=1)
    events["channel_episode_id"] = np.arange(len(events))
    events["label_net_positive"] = np.arange(len(events)) % 2
    events["r_net"] = np.where(events["label_net_positive"].eq(1), 0.8, -1.1)
    block = ((pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-07-01", tz="UTC")),)

    predictions, audit = run_oof_models(
        events,
        minute_bars=None,
        model_names=("logreg",),
        valid_blocks=block,
    )
    summary = summarise_oof_predictions(predictions)

    assert len(predictions) == 12
    assert predictions["model"].unique().tolist() == ["logreg"]
    assert audit.iloc[0]["train_rows"] == 40
    assert audit.iloc[0]["validation_rows"] == 12
    assert audit.iloc[0]["episode_overlap"] == 0
    assert summary.iloc[0]["oof_rows"] == 12
    assert {"roc_auc", "brier", "selected_total_net_r", "top30_mean_net_r"} <= set(summary.columns)
