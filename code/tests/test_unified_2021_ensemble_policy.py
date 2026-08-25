from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd

from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.unified_2021_ensemble_policy import (
    POLICY_GRID,
    EnsemblePolicy,
    apply_policy,
    causal_crossings,
    evaluate_oof_policy_grid,
    forward_promotion_gate,
    h1_compatibility_gate,
    ledger_to_common_per_bar,
    replay_selected_paths,
    select_score_only_threshold,
    side_route,
    summarize_candidate,
)


def _probability_row(xgb: float, lstm: float, svm: float) -> pd.Series:
    return pd.Series(
        {
            "p_long_xgboost": xgb,
            "p_long_lstm": lstm,
            "p_long_svm_linear": svm,
        }
    )


def _score_frame(start: str, periods: int, *, fold_id: int = 0) -> pd.DataFrame:
    decision_time = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    opportunity = np.resize(np.array([0.2, 0.8], dtype=float), periods)
    frame = pd.DataFrame(
        {
            "fold_id": fold_id,
            "row_key": [f"row-{fold_id}-{position}" for position in range(periods)],
            "decision_time": decision_time,
            "entry_time": decision_time + pd.Timedelta(minutes=1),
            "actual_exit_time_long": decision_time + pd.Timedelta(minutes=10),
            "actual_exit_time_short": decision_time + pd.Timedelta(minutes=10),
            "p_opportunity_xgboost": opportunity,
            "p_opportunity_lstm": opportunity,
            "p_opportunity_svm_linear": opportunity,
            "p_long_xgboost": 0.51,
            "p_long_lstm": 0.51,
            "p_long_svm_linear": 0.51,
        }
    )
    return frame


def test_policy_grid_is_the_registered_54_combinations():
    assert len(POLICY_GRID) == 54
    assert len(set(POLICY_GRID)) == 54


def test_score_selector_is_cold_rearmed_rate_bounded_and_deterministic():
    index = pd.date_range("2024-01-01", periods=96 * 4, freq="15min", tz="UTC")
    scores = pd.Series(np.tile([0.2, 0.8, 0.9, 0.1], 96), index=index)

    threshold_a, frontier_a = select_score_only_threshold(scores, daily_rate_cap=2)
    threshold_b, frontier_b = select_score_only_threshold(scores, daily_rate_cap=2)

    assert threshold_a == threshold_b
    pd.testing.assert_frame_equal(frontier_a, frontier_b)
    selected = frontier_a.loc[frontier_a["selected"]].iloc[0]
    assert selected["crossings_per_observed_day"] <= 2.0
    crossings = causal_crossings(scores, threshold_a, refractory="60min")
    assert crossings.equals(pd.DatetimeIndex(crossings).sort_values())


def test_side_routes_require_two_votes_or_strict_non_opposed_xgb_solo():
    policy = EnsemblePolicy(2, 0.60, 0.60, 0.75)

    assert side_route(_probability_row(xgb=0.80, lstm=0.70, svm=0.40), policy)[0] == "long"
    assert side_route(_probability_row(xgb=0.80, lstm=0.52, svm=0.49), policy) == (
        "long",
        "xgboost_solo",
    )
    assert side_route(_probability_row(xgb=0.80, lstm=0.20, svm=0.49), policy) == (
        None,
        "opposite_veto",
    )
    assert side_route(_probability_row(xgb=0.65, lstm=0.52, svm=0.49), policy) == (
        None,
        "side_abstain",
    )


def test_raw_crossing_consumes_refractory_even_when_side_abstains():
    predictions = _score_frame("2024-02-01", 8)
    calibration = _score_frame("2024-01-01", 6)

    _, funnel = apply_policy(
        predictions,
        calibration,
        EnsemblePolicy(3, 0.65, 0.65, 0.75),
    )

    first = funnel.loc[funnel["raw_crossing"]].iloc[0]
    within_hour = funnel["decision_time"].between(
        first["decision_time"],
        first["decision_time"] + pd.Timedelta(minutes=59),
    )
    assert first["decision_reason"] == "side_abstain"
    assert not funnel.loc[within_hour, "raw_crossing"].iloc[1:].any()
    assert funnel["refractory_rejection"].any()


def test_frozen_opportunity_threshold_does_not_need_stage_calibration_rows():
    predictions = _score_frame("2025-01-01", 8)

    _, funnel = apply_policy(
        predictions,
        predictions.iloc[0:0],
        EnsemblePolicy(3, 0.65, 0.65, 0.75),
        opportunity_threshold=0.8,
    )

    assert set(funnel["opportunity_threshold"]) == {0.8}
    assert funnel["raw_crossing"].any()


def test_replay_selected_paths_keeps_strictly_non_overlapping_trades():
    activations = pd.DataFrame(
        {
            "row_key": ["a", "b", "c"],
            "selected_side": ["long", "short", "short"],
            "route": ["two_of_three"] * 3,
        }
    )
    paths = pd.DataFrame(
        {
            "row_key": ["a", "b", "c"],
            "direction": ["long", "short", "short"],
            "entry_time": pd.to_datetime(
                ["2024-01-01 00:01Z", "2024-01-01 00:31Z", "2024-01-01 01:01Z"]
            ),
            "actual_exit_time": pd.to_datetime(
                ["2024-01-01 00:45Z", "2024-01-01 01:00Z", "2024-01-01 01:15Z"]
            ),
            "entry_price": [100.0, 100.0, 100.0],
            "exit_price": [101.0, 99.0, 99.0],
            "gross_return": [0.01, 0.01, 0.01],
            "net_return": [0.009, 0.009, 0.009],
            "path_complete": True,
        }
    )

    ledger = replay_selected_paths(activations, paths)

    assert ledger["row_key"].tolist() == ["a", "c"]
    assert (ledger["entry_time"].iloc[1:].to_numpy() > ledger["actual_exit_time"].iloc[:-1].to_numpy()).all()


def test_common_per_bar_reconciles_entry_marks_exit_and_costs():
    index = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0]}, index=index)
    ledger = pd.DataFrame(
        {
            "entry_time": [index[1] + pd.Timedelta(minutes=1)],
            "actual_exit_time": [index[3] + pd.Timedelta(minutes=1)],
            "direction": ["long"],
            "entry_price": [100.0],
            "exit_price": [103.0],
            "gross_return": [0.03],
            "net_return": [0.029],
            "cost_bps": [10.0],
        }
    )

    per_bar = ledger_to_common_per_bar(ledger, bars)
    summary = summarize_candidate(ledger, per_bar, phase="development")

    assert np.isclose(per_bar.sum(), 0.029)
    assert np.isclose(summary["net_return"], ledger["net_return"].sum())
    assert summary["long_trades"] == 1
    assert summary["short_trades"] == 0


def test_common_per_bar_matches_existing_intrabar_equity_transform():
    index = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0, 103.0],
            "high": [100.0, 101.0, 102.0, 103.0, 104.0],
            "low": [100.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0],
        },
        index=index,
    )
    minute_index = pd.date_range(index[1], periods=30, freq="1min", tz="UTC")
    minute_price = np.linspace(100.0, 102.0, len(minute_index))
    minute = pd.DataFrame(
        {column: minute_price for column in ("open", "high", "low", "close")},
        index=minute_index,
    )
    ledger, expected = simulate_bracket_trades_intrabar(
        bars,
        minute,
        pd.Series([2], index=index[:1]),
        tp_bps=10_000.0,
        sl_bps=9_000.0,
        max_hold=2,
        fee_bps=5.0,
        expected_interval=pd.Timedelta(minutes=1),
        include_audit=True,
    )
    ledger["actual_exit_time"] = ledger["intrabar_exit_time"]

    actual = ledger_to_common_per_bar(ledger, bars)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-15)


def test_development_policy_requires_both_sides_positive_in_three_folds(monkeypatch):
    prediction_parts = []
    calibration_parts = []
    path_rows = []
    for fold_id in range(5):
        calibration = _score_frame(f"2023-{fold_id + 1:02d}-01", 12, fold_id=fold_id)
        prediction = _score_frame(f"2024-{fold_id + 1:02d}-01", 12, fold_id=fold_id)
        for frame in (calibration, prediction):
            frame.loc[:, [
                "p_opportunity_xgboost",
                "p_opportunity_lstm",
                "p_opportunity_svm_linear",
            ]] = 0.2
            frame.loc[[1, 6, 11], [
                "p_opportunity_xgboost",
                "p_opportunity_lstm",
                "p_opportunity_svm_linear",
            ]] = 0.8
        # Alternate confident unanimous sides on threshold crossings.
        crossing = prediction["p_opportunity_xgboost"].eq(0.8)
        crossing_number = np.arange(crossing.sum())
        long_mask = crossing.copy()
        long_mask.loc[crossing] = crossing_number % 2 == 0
        prediction.loc[long_mask, ["p_long_xgboost", "p_long_lstm", "p_long_svm_linear"]] = 0.9
        prediction.loc[crossing & ~long_mask, ["p_long_xgboost", "p_long_lstm", "p_long_svm_linear"]] = 0.1
        for row in prediction.loc[crossing].itertuples(index=False):
            direction = "long" if row.p_long_xgboost > 0.5 else "short"
            path_rows.append(
                {
                    "row_key": row.row_key,
                    "direction": direction,
                    "entry_time": row.entry_time,
                    "actual_exit_time": row.entry_time + pd.Timedelta(minutes=5),
                    "entry_price": 100.0,
                    "exit_price": 101.0 if direction == "long" else 99.0,
                    "gross_return": 0.01,
                    "net_return": 0.009,
                    "net_r": 0.9,
                    "path_complete": True,
                }
            )
        prediction_parts.append(prediction)
        calibration_parts.append(calibration)
    oof = SimpleNamespace(
        predictions=pd.concat(prediction_parts, ignore_index=True),
        calibration_predictions=pd.concat(calibration_parts, ignore_index=True),
    )

    from experiments import unified_2021_ensemble_policy as policy_module

    selector_calls = 0
    real_selector = policy_module.select_score_only_threshold

    def counted_selector(*args, **kwargs):
        nonlocal selector_calls
        selector_calls += 1
        return real_selector(*args, **kwargs)

    monkeypatch.setattr(policy_module, "select_score_only_threshold", counted_selector)
    grid, selected = evaluate_oof_policy_grid(oof, pd.DataFrame(path_rows))

    assert len(grid) == 54
    assert selected is not None
    chosen = grid.loc[grid["selected"]].iloc[0]
    assert chosen["long_positive_folds"] >= 3
    assert chosen["short_positive_folds"] >= 3
    assert selector_calls == 15


def _h1_summary() -> dict[str, object]:
    return {
        "trades": 132,
        "long_trades": 15,
        "short_trades": 15,
        "net_return": 0.009687734009,
        "long_net_return": 0.0,
        "short_net_return": 0.0,
        "apr_jun_net_return": 0.0,
        "apr_jun_long_trades": 1,
        "apr_jun_short_trades": 1,
        "sortino": 0.3521029257,
        "max_drawdown": 0.047613386266,
    }


def test_h1_gate_requires_total_both_sides_and_apr_jun_presence():
    passing = _h1_summary()
    assert h1_compatibility_gate(passing)
    for key, failing_value in {
        "trades": 131,
        "long_trades": 14,
        "short_trades": 14,
        "net_return": 0.009,
        "long_net_return": -1e-6,
        "short_net_return": -1e-6,
        "apr_jun_net_return": -1e-6,
        "apr_jun_long_trades": 0,
        "apr_jun_short_trades": 0,
        "sortino": 0.35,
        "max_drawdown": 0.048,
    }.items():
        assert not h1_compatibility_gate({**passing, key: failing_value})


def test_forward_gate_requires_union_economics_and_quarterly_both_sides():
    passing = {
        "trades": 111,
        "long_trades": 15,
        "short_trades": 15,
        "net_return": 0.063149608713,
        "long_net_return": 0.0,
        "short_net_return": 0.0,
        "sortino": 2.1060930178,
        "max_drawdown": 0.029796677136,
        "every_quarter_has_long_and_short": True,
    }
    assert forward_promotion_gate(passing)
    assert not forward_promotion_gate({**passing, "every_quarter_has_long_and_short": False})
    assert not forward_promotion_gate({**passing, "trades": 110})
