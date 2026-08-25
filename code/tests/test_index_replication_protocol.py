from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.index_replication_protocol import (
    VIX_FEATURE_COLS,
    build_vix_block,
    daily_economics,
    decide_vix_admission,
    holm_adjust,
    join_completed_vix,
    select_h1_policy,
)


MODELS = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)


def _vix_bars() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "close": np.linspace(15.0, 18.0, len(index)),
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=index,
    )


def test_join_completed_vix_never_uses_unavailable_bar():
    vix = build_vix_block(_vix_bars())
    decision_time = pd.to_datetime(
        ["2024-01-01 05:15:00", "2024-01-01 05:22:00"], utc=True
    )
    index_rows = pd.DataFrame(
        {"decision_time": decision_time, "close": [100.0, 101.0]},
        index=decision_time - pd.Timedelta(minutes=15),
    )

    joined = join_completed_vix(index_rows, vix)

    assert (joined["vix_available_at"] <= joined["decision_time"]).all()
    assert joined["vix_age_minutes"].tolist() == [0.0, 7.0]
    assert not joined[list(VIX_FEATURE_COLS)].isna().any().any()


def test_build_vix_block_excludes_incomplete_rows_and_has_no_future_fill():
    bars = _vix_bars()
    bars.loc[bars.index[21], "complete_bar"] = False
    block = build_vix_block(bars)

    assert bars.index[21] not in block.index
    assert block.index.min() == bars.index[20]
    assert block.loc[block.index.min(), "vix_available_at"] == (
        block.index.min() + pd.Timedelta(minutes=15)
    )


def _paired_gate_rows(*, family_deltas: list[float], retention: float = 1.0):
    rows = []
    for model, family_delta in zip(MODELS, family_deltas):
        for width in (5, 10, 15):
            for fold in range(5):
                price_trades = 10
                rows.append(
                    {
                        "model_name": model,
                        "width_bps": width,
                        "fold_id": fold,
                        "price_net": 0.01,
                        "vix_net": 0.01 + family_delta,
                        "price_trades": price_trades,
                        "vix_trades": int(price_trades * retention),
                        "common_timestamp_hash": f"fold-{fold}",
                    }
                )
    return pd.DataFrame(rows)


def test_vix_gate_admits_simple_five_of_nine_majority_when_other_conditions_pass():
    decision = decide_vix_admission(
        _paired_gate_rows(family_deltas=[0.002] * 5 + [-0.001] * 4)
    )

    assert decision.selected_base == "price_vix"
    assert decision.admitted is True
    assert decision.conditions == {
        "median_positive": True,
        "families_positive_majority_5_of_9": True,
        "folds_positive_3_of_5": True,
        "trade_retention_80pct": True,
    }
    assert decision.metrics["positive_families"] == 5
    assert decision.metrics["positive_folds"] == 5


def test_vix_gate_rejects_four_of_nine_positive_families():
    decision = decide_vix_admission(
        _paired_gate_rows(family_deltas=[0.002] * 4 + [-0.001] * 5)
    )

    assert decision.selected_base == "price"
    assert decision.admitted is False
    assert decision.conditions["families_positive_majority_5_of_9"] is False
    assert decision.metrics["positive_families"] == 4


def test_vix_gate_ties_or_trade_collapse_choose_price():
    tied = decide_vix_admission(_paired_gate_rows(family_deltas=[0.0] * 9))
    collapsed = decide_vix_admission(
        _paired_gate_rows(family_deltas=[0.001] * 9, retention=0.7)
    )

    assert tied.selected_base == "price"
    assert tied.conditions["median_positive"] is False
    assert collapsed.selected_base == "price"
    assert collapsed.conditions["trade_retention_80pct"] is False


def test_vix_gate_incomplete_pairing_fails_closed_to_price():
    incomplete = _paired_gate_rows(family_deltas=[0.001] * 9).iloc[:-1]
    decision = decide_vix_admission(incomplete)

    assert decision.selected_base == "price"
    assert decision.metrics["complete_pairing"] is False
    assert not any(decision.conditions.values())


def test_daily_economics_uses_common_utc_calendar_and_both_sides():
    bar_index = pd.date_range("2025-01-01", periods=4 * 96, freq="15min", tz="UTC")
    per_bar = pd.Series(0.0, index=bar_index)
    per_bar.iloc[[1, 100, 200]] = [0.01, -0.005, 0.002]
    ledger = pd.DataFrame(
        {
            "entry_time": bar_index[[1, 100, 200]],
            "side": [1, -1, 1],
            "gross_return": [0.012, -0.003, 0.004],
            "net_return": [0.01, -0.005, 0.002],
        }
    )

    result = daily_economics(
        ledger,
        per_bar,
        start=pd.Timestamp("2025-01-01", tz="UTC"),
        end=pd.Timestamp("2025-01-05", tz="UTC"),
    )

    assert result["calendar_days"] == 4
    assert result["trades"] == 3
    assert result["n_long"] == 2
    assert result["n_short"] == 1
    assert result["net_return"] == pytest.approx(0.007)
    assert result["cost_return"] == pytest.approx(0.006)
    assert result["trades_per_day"] == pytest.approx(0.75)
    assert np.isfinite(result["daily_sharpe"])


def test_h1_selection_is_constraint_first_then_sortino_net_and_trades():
    grid = pd.DataFrame(
        [
            {"width_bps": 5, "tau": 0.0, "trades": 100, "n_long": 50,
             "n_short": 50, "positive_months": 3, "daily_sortino": 10.0,
             "net_return": 1.0},
            {"width_bps": 10, "tau": 0.5, "trades": 80, "n_long": 40,
             "n_short": 40, "positive_months": 4, "daily_sortino": 1.0,
             "net_return": 0.2},
            {"width_bps": 15, "tau": 0.7, "trades": 80, "n_long": 40,
             "n_short": 40, "positive_months": 4, "daily_sortino": 1.0,
             "net_return": 0.1},
        ]
    )

    winner = select_h1_policy(grid)

    assert winner["width_bps"] == 10
    assert winner["constraint_violation"] == 0


def test_h1_selection_retains_best_diagnostic_when_none_is_eligible():
    grid = pd.DataFrame(
        [
            {"width_bps": 5, "tau": 0.5, "trades": 10, "n_long": 10,
             "n_short": 0, "positive_months": 1, "daily_sortino": 2.0,
             "net_return": 0.1},
            {"width_bps": 10, "tau": 0.7, "trades": 20, "n_long": 10,
             "n_short": 10, "positive_months": 2, "daily_sortino": 1.0,
             "net_return": 0.05},
        ]
    )

    diagnostic = select_h1_policy(grid)

    assert diagnostic["eligible"] == False
    assert diagnostic["constraint_violation"] > 0


def test_holm_adjust_is_monotone_and_bounded():
    adjusted = holm_adjust(pd.Series([0.01, 0.04, 0.03], index=["a", "b", "c"]))

    assert adjusted.index.tolist() == ["a", "b", "c"]
    assert adjusted.between(0.0, 1.0).all()
    assert adjusted["a"] == pytest.approx(0.03)
    assert adjusted["c"] == pytest.approx(0.06)
    assert adjusted["b"] == pytest.approx(0.06)
