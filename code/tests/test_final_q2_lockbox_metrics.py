from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.final_q2_lockbox_metrics import (
    PRIMARY_ESTIMAND,
    daily_net_series,
    double_cost_ledger,
    monday_week_blocks,
    paired_weekly_bootstrap,
    summarise_candidate,
)


START = pd.Timestamp("2026-04-01T00:00:00Z")
END = pd.Timestamp("2026-07-01T00:00:00Z")


def _ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "exit_time": pd.to_datetime(
                ["2026-04-01T00:15:00Z", "2026-04-01T23:45:00Z", "2026-05-03T12:00:00Z"],
                utc=True,
            ),
            "side": [1, -1, 1],
            "gross_return": [0.010, -0.004, 0.006],
            "cost_return": [0.001, 0.001, 0.001],
            "net_return": [0.009, -0.005, 0.005],
        }
    )


def test_daily_metrics_zero_fill_all_91_q2_days() -> None:
    ledger = _ledger()
    daily = daily_net_series(ledger, start=START, end=END)

    assert len(daily) == 91
    assert daily.index[0] == START
    assert daily.index[-1] == pd.Timestamp("2026-06-30T00:00:00Z")
    assert daily.sum() == pytest.approx(ledger["net_return"].sum())
    assert daily.loc["2026-04-01"] == pytest.approx(0.004)
    assert daily.loc["2026-04-02"] == 0.0


def test_daily_metrics_reject_rows_outside_the_half_open_interval() -> None:
    ledger = _ledger()
    ledger.loc[0, "exit_time"] = END

    with pytest.raises(ValueError, match="outside"):
        daily_net_series(ledger, start=START, end=END)


def test_double_cost_ledger_reconciles_gross_cost_and_net() -> None:
    stressed = double_cost_ledger(_ledger())

    assert np.allclose(stressed["cost_return"], 0.002)
    assert np.allclose(
        stressed["net_return"],
        stressed["gross_return"] - stressed["cost_return"],
    )
    assert stressed["net_return"].sum() == pytest.approx(0.006)


def test_candidate_summary_uses_daily_sharpe_sortino_and_complete_costs() -> None:
    summary = summarise_candidate(
        _ledger(),
        candidate_id="fixture",
        stream="btcusdt",
        start=START,
        end=END,
    )

    assert summary["trades"] == 3
    assert summary["long_trades"] == 2
    assert summary["short_trades"] == 1
    assert summary["gross_return"] == pytest.approx(0.012)
    assert summary["cost_return"] == pytest.approx(0.003)
    assert summary["net_return"] == pytest.approx(0.009)
    assert summary["trades_per_day"] == pytest.approx(3 / 91)
    assert summary["net_bps_per_trade"] == pytest.approx(30.0)
    assert np.isfinite(summary["daily_sharpe"])
    assert np.isfinite(summary["daily_sortino"])
    assert summary["positive_months"] == 2
    assert summary["stress_2x_net_return"] == pytest.approx(0.006)


def test_boundary_weeks_are_zero_padded_to_seven_days() -> None:
    delta = pd.Series(
        np.arange(91, dtype=float) / 1_000_000.0,
        index=pd.date_range(START, END, inclusive="left", freq="D"),
    )
    blocks = monday_week_blocks(delta, start=START, end=END)

    assert len(blocks) == 14
    assert all(len(block) == 7 for block in blocks)
    assert blocks[0].index[0] == pd.Timestamp("2026-03-30T00:00:00Z")
    assert blocks[-1].index[-1] == pd.Timestamp("2026-07-05T00:00:00Z")
    assert sum(block.sum() for block in blocks) == pytest.approx(delta.sum())


def test_primary_bootstrap_is_union_minus_lstm_only_and_deterministic() -> None:
    index = pd.date_range(START, END, inclusive="left", freq="D")
    union = pd.Series(0.001, index=index)
    lstm = pd.Series(0.0002, index=index)

    first = paired_weekly_bootstrap(union, lstm, reps=5_000, seed=42)
    second = paired_weekly_bootstrap(union, lstm, reps=5_000, seed=42)

    assert first.estimand == PRIMARY_ESTIMAND
    assert first.point_estimate == pytest.approx((union - lstm).sum())
    assert first.lower_95 == second.lower_95
    assert first.upper_95 == second.upper_95
    assert first.primary_confirmatory_support is True
    assert first.block_count == 14


def test_nonprimary_bootstrap_never_emits_a_support_label() -> None:
    index = pd.date_range(START, END, inclusive="left", freq="D")
    policy = pd.Series(0.001, index=index)
    control = pd.Series(0.0, index=index)

    result = paired_weekly_bootstrap(
        policy,
        control,
        reps=100,
        seed=42,
        estimand="usa500_policy_minus_comparator_total_net",
    )

    assert result.primary_confirmatory_support is None


def test_empty_ledger_produces_finite_zero_summary() -> None:
    empty = _ledger().iloc[0:0]
    summary = summarise_candidate(
        empty,
        candidate_id="empty",
        stream="usa500",
        start=START,
        end=END,
    )

    numeric = [
        "gross_return",
        "cost_return",
        "net_return",
        "trades_per_day",
        "net_bps_per_trade",
        "win_rate",
        "daily_sharpe",
        "daily_sortino",
        "max_drawdown",
        "stress_2x_net_return",
    ]
    assert summary["trades"] == 0
    assert all(np.isfinite(summary[column]) for column in numeric)
    assert all(summary[column] == 0.0 for column in numeric)
