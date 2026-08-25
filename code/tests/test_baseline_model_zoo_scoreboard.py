from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "cache"
    / "tuning"
    / "baseline_model_zoo_1m_180d_fixed15_monthly_h1"
)


def test_completed_scoreboards_have_exact_nine_model_contract():
    from experiments.baseline_model_zoo_scoreboard import build_scoreboards

    tables = build_scoreboards(ROOT)
    assert len(tables["classification"]) == 27
    assert len(tables["hold_grid"]) == 27
    assert len(tables["selected_holds"]) == 27
    assert len(tables["calibration_grid"]) == 891
    assert len(tables["selected_policies"]) == 27
    assert len(tables["raw_forward"]) == 27
    assert len(tables["forward"]) == 27
    assert set(tables["forward"]["model_name"]) == set(
        tables["raw_forward"]["model_name"]
    )
    assert set(tables["selected_policies"]["lookback_days"]) == {180}
    assert set(tables["raw_forward"]["lookback_days"]) == {180}
    assert set(tables["forward"]["lookback_days"]) == {180}
    assert set(tables["selected_holds"]["hold_minutes"]) == {15}
    assert set(tables["selected_policies"]["max_hold"]) == {1}
    assert set(tables["raw_forward"]["hold_minutes"]) == {15}
    assert set(tables["forward"]["max_hold"]) == {1}


def test_winners_are_reported_separately_for_raw_and_calibrated_results():
    from experiments.baseline_model_zoo_scoreboard import build_scoreboards

    tables = build_scoreboards(ROOT)
    assert set(tables["raw_winners"]["criterion"]) == {
        "Sortino",
        "Sharpe",
        "Net return",
    }
    assert set(tables["calibrated_winners"]["criterion"]) == {
        "Sortino",
        "Sharpe",
        "Net return",
    }


def test_top_three_tables_contain_three_distinct_model_families():
    from experiments.baseline_model_zoo_scoreboard import build_scoreboards

    tables = build_scoreboards(ROOT)
    for name in ("raw_top_models", "calibrated_top_models"):
        frame = tables[name]
        assert len(frame) == 3
        assert frame["model_name"].nunique() == 3
        assert frame["net_return"].is_monotonic_decreasing


def test_uncalibrated_monthly_evidence_reconciles_to_full_forward_net():
    from experiments.baseline_model_zoo_scoreboard import build_scoreboards

    tables = build_scoreboards(ROOT)
    monthly = tables["raw_forward_monthly"]
    assert len(monthly) == 243
    assert monthly.groupby(["model_name", "width_bps"]).size().eq(9).all()
    reconciled = monthly.groupby(["model_name", "width_bps"])["net_return"].sum()
    expected = tables["raw_forward"].set_index(["model_name", "width_bps"])[
        "net_return"
    ]
    assert (reconciled - expected).abs().max() < 1e-10
