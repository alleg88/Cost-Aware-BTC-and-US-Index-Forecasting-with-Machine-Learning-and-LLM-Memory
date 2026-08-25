from __future__ import annotations

import pandas as pd

from experiments.index_channel_replication import (
    IndexChannelConfig,
    STAGES,
    channel_protocol,
    label_signal_funnel,
    mask_stale_channel_context,
    stage_signal_frame,
    validate_closed_channel_context,
)


def _signals() -> pd.DataFrame:
    index = pd.to_datetime(
        [
            "2025-06-30 21:50",
            "2025-06-30 21:55",
            "2025-06-30 22:00",
            "2025-06-30 23:55",
        ],
        utc=True,
    )
    return pd.DataFrame(
        {
            "signal": [1, -1, 1, -1],
            "availability_time": index + pd.Timedelta(minutes=5),
            "channel_source_time": index.floor("h") - pd.Timedelta(hours=1),
            "channel_availability_time": index.floor("h"),
        },
        index=index,
    )


def test_protocol_freezes_rr2_window60_and_never_selects_on_later_stages():
    protocol = channel_protocol(IndexChannelConfig.for_stream("usa500"))

    assert protocol["primary"] == {"window": 60, "rr": 2.0}
    assert protocol["sensitivities"] == [
        {"window": 60, "rr": 3.0},
        {"window": 60, "rr": 5.0},
    ]
    assert protocol["selection"] == "none_frozen_btc_transfer"
    assert protocol["later_stage_can_tune"] is False
    assert protocol["vix_used"] is False
    assert protocol["sentiment_used"] is False
    assert tuple(protocol["stages"]) == tuple(STAGES)
    assert protocol["channel_window_unit"] == "completed_market_hours"
    assert protocol["stale_context_cutoff_minutes"] == 60


def test_stage_scope_requires_consecutive_next_bar_and_full_hold_before_boundary():
    signals = _signals()
    bars_index = signals.index

    scoped = stage_signal_frame(
        signals,
        bars_index=bars_index,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-07-01", tz="UTC"),
        hold_minutes=120,
    )

    # 21:50 and 21:55 have consecutive entries and complete by the boundary.
    # The 22:00 row has a session gap; the final row crosses the boundary.
    assert scoped.loc[pd.Timestamp("2025-06-30 21:50", tz="UTC"), "signal"] == 1
    assert scoped.loc[pd.Timestamp("2025-06-30 21:55", tz="UTC"), "signal"] == -1
    assert scoped.loc[pd.Timestamp("2025-06-30 22:00", tz="UTC"), "signal"] == 0
    assert scoped.loc[pd.Timestamp("2025-06-30 23:55", tz="UTC"), "signal"] == 0


def test_closed_hourly_context_is_available_before_every_decision():
    signals = _signals()

    audit = validate_closed_channel_context(signals)

    assert audit["passed"]
    assert audit["max_source_availability_lag_minutes"] >= 0
    broken = signals.copy()
    broken.loc[broken.index[0], "channel_availability_time"] = (
        broken.loc[broken.index[0], "availability_time"] + pd.Timedelta(minutes=1)
    )
    assert not validate_closed_channel_context(broken)["passed"]


def test_stream_costs_are_separate_and_forward_ends_before_q2():
    usa500 = IndexChannelConfig.for_stream("usa500")
    usatech = IndexChannelConfig.for_stream("usatech")

    assert usa500.cost_bps == 2.0
    assert usatech.cost_bps == 3.0
    assert STAGES["forward"][1] == pd.Timestamp("2026-04-01", tz="UTC")


def test_funnel_keeps_t1_t2_t3_entry_separate_from_evaluation_period():
    raw = pd.DataFrame(
        {
            "dataset": ["pooled"] * 4,
            "setup_type": ["edge_rejection"] * 4,
            "side": ["long"] * 4,
            "stage": ["T1", "T2", "T3", "ENTRY"],
            "count": [10, 8, 6, 4],
        }
    )

    labelled = label_signal_funnel(raw, stream="usa500", evaluation_stage="forward")

    assert list(labelled["funnel_stage"]) == ["T1", "T2", "T3", "ENTRY"]
    assert labelled["evaluation_stage"].eq("forward").all()
    assert labelled["stream"].eq("usa500").all()
    assert "stage" not in labelled.columns


def test_session_gap_context_is_disabled_until_a_new_hour_closes():
    index = pd.to_datetime(["2025-01-03 21:55", "2025-01-05 23:00"], utc=True)
    frame = pd.DataFrame(
        {
            "availability_time": index + pd.Timedelta(minutes=5),
            "channel_availability_time": pd.to_datetime(
                ["2025-01-03 22:00", "2025-01-03 22:00"], utc=True
            ),
            "channel_mid": [100.0, 100.0],
            "channel_upper": [101.0, 101.0],
            "channel_lower": [99.0, 99.0],
            "channel_slope": [0.001, 0.001],
            "channel_r2": [0.8, 0.8],
            "channel_regime": ["up", "up"],
            "channel_confluence": [1, 1],
            "channel_confluence_count": [3, 3],
        },
        index=index,
    )

    masked = mask_stale_channel_context(frame, cutoff_minutes=60)

    assert masked.loc[index[0], "channel_regime"] == "up"
    assert masked.loc[index[1], "channel_regime"] == "none"
    assert pd.isna(masked.loc[index[1], "channel_slope"])
    assert bool(masked.loc[index[1], "channel_context_stale"])
