"""Protocol, pooled-side, frequency, and 1% risk tests for Notebook J."""
from __future__ import annotations

import pandas as pd
import pytest

from experiments.supervisor_channel_strategy import (
    SupervisorChannelConfig,
    protocol_dict,
    summarise_account_pnl,
    summarise_signal_funnel,
    summarise_trade_breakdown,
    summarise_trade_frequency,
)


START = pd.Timestamp("2025-01-01", tz="UTC")
END = pd.Timestamp("2025-01-04", tz="UTC")


def _trade_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2025-01-01 09:00Z", "2025-01-01 10:00Z", "2025-01-02 10:00Z"]
            ),
            "exit_time": pd.to_datetime(
                ["2025-01-01 09:30Z", "2025-01-01 10:30Z", "2025-01-02 10:30Z"]
            ),
            "side": ["long", "short", "long"],
            "r_net": [2.0, 1.0, -1.0],
        }
    )


def test_one_net_r_equals_one_percent_and_no_profit_stop_is_applied():
    daily, summary = summarise_account_pnl(
        _trade_fixture(), start=START, end=END, risk_pct=1.0
    )

    day = pd.Timestamp("2025-01-01", tz="UTC")
    assert daily.loc[day, "realised_pnl_pct"] == pytest.approx(3.0)
    assert daily.loc[day, "trades_realised"] == 2
    assert summary["days_ge_2pct"] == 1
    assert summary["days_ge_3pct"] == 1
    assert summary["days_ge_5pct"] == 0
    assert summary["daily_stop_applied"] is False


def test_trade_frequency_is_observed_against_three_to_five_not_capped():
    trades = pd.DataFrame(
        {
            "entry_time": pd.to_datetime(
                ["2025-01-01 01:00Z"] * 6 + ["2025-01-02 01:00Z"] * 3
            )
        }
    )

    daily, summary = summarise_trade_frequency(trades, start=START, end=END)

    assert daily["trades"].tolist() == [6, 3, 0]
    assert summary["total_trades"] == 9
    assert summary["trades_per_day"] == pytest.approx(3.0)
    assert summary["inside_target_3_to_5"] is True
    assert summary["trade_cap_applied"] is False


def test_protocol_has_no_three_minute_branch_caps_or_daily_stop():
    protocol = protocol_dict(SupervisorChannelConfig(), stage="dev")

    assert protocol["decision_grid"] == "5min"
    assert protocol["side_dataset"] == "pooled"
    assert protocol["max_trades_per_day"] is None
    assert protocol["max_concurrent"] is None
    assert protocol["daily_stop_pct"] is None
    assert protocol["profit_shutdown_pct"] is None
    assert protocol["target_frequency_trades_per_day"] == [3.0, 5.0]
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_any_data_can_load():
    with pytest.raises(PermissionError, match="development-only"):
        protocol_dict(SupervisorChannelConfig(), stage="forward")


def test_funnel_keeps_long_and_short_in_one_pooled_table():
    signals = pd.DataFrame(
        {
            "edge_stage_1": [1, 1],
            "edge_stage_2": [1, 1],
            "edge_stage_3": [1, 1],
            "edge_stage_1_side": ["long", "short"],
            "edge_stage_2_side": ["long", "short"],
            "edge_stage_3_side": ["long", "short"],
            "midline_stage_1": [0, 0],
            "midline_stage_2": [0, 0],
            "midline_stage_3": [0, 0],
            "midline_stage_1_side": [None, None],
            "midline_stage_2_side": [None, None],
            "midline_stage_3_side": [None, None],
            "signal": [1, -1],
            "setup_type": ["edge_rejection", "edge_rejection"],
        },
        index=pd.date_range("2025-01-01", periods=2, freq="5min", tz="UTC"),
    )

    funnel = summarise_signal_funnel(signals)

    edge_t3 = funnel[(funnel["setup_type"] == "edge_rejection") & (funnel["stage"] == "T3")]
    assert edge_t3.set_index("side")["count"].to_dict() == {"long": 1, "short": 1}
    assert set(funnel["dataset"]) == {"pooled"}


def test_economic_breakdown_keeps_setup_and_side_inside_one_table():
    trades = pd.DataFrame(
        {
            "setup_type": ["edge_rejection", "edge_rejection", "midline_retest"],
            "side": ["long", "short", "long"],
            "outcome": ["tp", "sl", "timeout"],
            "r_gross": [2.0, -1.0, 0.5],
            "r_net": [1.7, -1.3, 0.2],
            "risk_bps": [40.0, 30.0, 50.0],
        }
    )

    breakdown = summarise_trade_breakdown(trades, rr_multiple=2.0)

    assert set(breakdown["dataset"]) == {"pooled"}
    assert breakdown["trades"].sum() == 3
    assert set(zip(breakdown["setup_type"], breakdown["side"])) == {
        ("edge_rejection", "long"),
        ("edge_rejection", "short"),
        ("midline_retest", "long"),
    }
