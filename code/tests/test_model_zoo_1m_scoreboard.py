from __future__ import annotations

import pandas as pd


def test_scoreboard_has_nine_models_three_widths_and_required_metrics():
    from experiments.model_zoo_1m_scoreboard import load_comparison

    table = load_comparison()
    required = {
        "model_name",
        "width_bps",
        "sortino",
        "sharpe",
        "net_return",
        "trades",
        "n_long",
        "n_short",
        "long_net",
        "short_net",
        "positive_months",
        "bull_sortino",
        "sideways_sortino",
        "bear_sortino",
    }
    assert len(table) == 27
    assert table["model_name"].nunique() == 9
    assert required <= set(table.columns)
    assert table.groupby("model_name")["width_bps"].apply(set).eq({55, 65, 75}).all()
    assert table["resolution"].eq("1m").all()
    assert pd.to_datetime(table["period_end"], utc=True).le("2026-04-01").all()


def test_best_rows_are_named_and_match_each_metric_maximum():
    from experiments.model_zoo_1m_scoreboard import best_by_metric, load_comparison

    table = load_comparison()
    winners = best_by_metric(table)
    mapping = {
        "Highest Sortino": "sortino",
        "Highest Sharpe": "sharpe",
        "Highest net return": "net_return",
    }
    assert set(winners["criterion"]) == set(mapping)
    for row in winners.to_dict(orient="records"):
        metric = mapping[row["criterion"]]
        assert row[metric] == table[metric].max()


def test_selection_tables_have_one_candidate_and_policy_per_model_width():
    from experiments.model_zoo_1m_scoreboard import load_selection_tables

    candidates, policies = load_selection_tables()
    for table in (candidates, policies):
        assert len(table) == 27
        assert not table.duplicated(["model_name", "width_bps"]).any()
        assert table.groupby("model_name")["width_bps"].apply(set).eq({55, 65, 75}).all()
