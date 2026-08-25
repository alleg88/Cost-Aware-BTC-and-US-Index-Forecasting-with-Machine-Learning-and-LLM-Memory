from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.unified_side_profitability_policy import (
    POLICY_GRID,
    FoldPolicySelection,
    RiskCoveragePolicy,
    apply_frozen_fold_policy,
    evaluate_development_ledger,
    score_routes,
    select_fold_policy,
    xgb_satellite_ablation,
)


MODELS = ("xgboost", "lstm", "svm_linear")


def _prediction_rows() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_key": ["regular", "satellite", "opposed", "ties"],
            "decision_time": pd.date_range(
                "2021-01-01", periods=4, freq="2h", tz="UTC"
            ),
            "fold_id": 0,
            "source_role": "policy_selection",
            "p_long_xgboost": [0.80, 0.82, 0.84, 0.50],
            "p_short_xgboost": [0.20, 0.18, 0.16, 0.50],
            "p_long_lstm": [0.70, 0.40, 0.35, 0.50],
            "p_short_lstm": [0.30, 0.60, 0.65, 0.50],
            "p_long_svm_linear": [0.40, 0.50, 0.30, 0.50],
            "p_short_svm_linear": [0.60, 0.50, 0.70, 0.50],
        }
    )
    return frame


def test_registered_grid_has_nine_risk_coverage_policies():
    assert len(POLICY_GRID) == 9
    assert {policy.regular_side_rate_cap for policy in POLICY_GRID} == {
        0.5,
        1.0,
        1.5,
    }
    assert {policy.xgb_side_rate_cap for policy in POLICY_GRID} == {
        0.125,
        0.25,
        0.5,
    }


def test_route_scores_distinguish_majority_satellite_veto_and_ties():
    scored = score_routes(_prediction_rows()).set_index("row_key")

    assert scored.loc["regular", "regular_side"] == "long"
    assert scored.loc["regular", "regular_votes"] == 2
    assert bool(scored.loc["regular", "xgb_satellite_eligible"])
    assert scored.loc["regular", "xgb_satellite_side"] == "long"
    assert scored.loc["satellite", "regular_side"] == "none"
    assert scored.loc["satellite", "xgb_satellite_side"] == "long"
    assert bool(scored.loc["satellite", "xgb_satellite_eligible"])
    assert scored.loc["opposed", "regular_side"] == "short"
    assert not bool(scored.loc["opposed", "xgb_satellite_eligible"])
    assert scored.loc["ties", "regular_side"] == "none"
    assert scored.loc["ties", "xgb_satellite_side"] == "none"


def _policy_fixture(rows: int = 240) -> tuple[pd.DataFrame, pd.DataFrame]:
    time = pd.date_range("2021-01-01", periods=rows, freq="3h", tz="UTC")
    phase = np.arange(rows) % 4
    long_side = phase < 2
    xgb_edge = np.linspace(0.95, 0.05, rows)
    predictions = pd.DataFrame(
        {
            "row_key": [f"policy-{index}" for index in range(rows)],
            "decision_time": time,
            "fold_id": 0,
            "source_role": "policy_selection",
            "p_long_xgboost": np.where(long_side, 0.5 + xgb_edge / 2, 0.5 - xgb_edge / 2),
            "p_short_xgboost": np.where(long_side, 0.5 - xgb_edge / 2, 0.5 + xgb_edge / 2),
            "p_long_lstm": np.where(long_side, 0.72, np.where(phase == 2, 0.28, 0.50)),
            "p_short_lstm": np.where(long_side, 0.28, np.where(phase == 2, 0.72, 0.50)),
            "p_long_svm_linear": np.where(long_side, 0.68, np.where(phase == 2, 0.32, 0.50)),
            "p_short_svm_linear": np.where(long_side, 0.32, np.where(phase == 2, 0.68, 0.50)),
        }
    )
    directions = np.tile(["long", "short"], rows)
    row_keys = np.repeat(predictions["row_key"].to_numpy(), 2)
    chosen_long = np.repeat(long_side, 2)
    profitable = (directions == "long") == chosen_long
    entry = np.repeat((time + pd.Timedelta(minutes=1)).to_numpy(), 2)
    paths = pd.DataFrame(
        {
            "row_key": row_keys,
            "direction": directions,
            "path_complete": True,
            "entry_time": entry,
            "actual_exit_time": pd.to_datetime(entry, utc=True) + pd.Timedelta(minutes=30),
            "net_return": np.where(profitable, 0.004, -0.003),
            "net_r": np.where(profitable, 0.8, -0.6),
            "net_bps": np.where(profitable, 40.0, -30.0),
        }
    )
    return predictions, paths


def test_fold_policy_is_selected_only_from_policy_role_and_is_replayable():
    predictions, paths = _policy_fixture()

    selection = select_fold_policy(predictions, paths, fold_id=0)
    activations, funnel = apply_frozen_fold_policy(predictions, selection)

    assert selection.source_role == "policy_selection"
    assert len(selection.policy_grid) == 9
    assert selection.policy in POLICY_GRID
    assert set(activations["selected_side"]).issubset({"long", "short"})
    assert set(activations["route"]).issubset(
        {"regular_unanimous", "regular_xgb_decisive", "regular_without_xgb", "xgb_satellite"}
    )
    assert len(funnel) == len(predictions)


def _qualified_ledger(rows: int = 140) -> pd.DataFrame:
    index = np.arange(rows)
    side = np.where(index % 2 == 0, "long", "short")
    route = np.where(index < 15, "xgb_satellite", "regular_xgb_decisive")
    return pd.DataFrame(
        {
            "row_key": [f"trade-{value}" for value in index],
            "fold_id": index % 5,
            "direction": side,
            "selected_side": side,
            "route": route,
            "net_return": 0.001,
            "net_r": 0.1,
            "net_bps": 10.0,
        }
    )


def _selections(passed: int = 5) -> list[FoldPolicySelection]:
    output = []
    for fold_id in range(5):
        output.append(
            FoldPolicySelection(
                fold_id=fold_id,
                policy=RiskCoveragePolicy(0.5, 0.125),
                thresholds={
                    "regular_long": 0.5,
                    "regular_short": 0.5,
                    "xgb_long": 0.5,
                    "xgb_short": 0.5,
                },
                selection_passed=fold_id < passed,
                source_role="policy_selection",
                policy_grid=pd.DataFrame(),
                threshold_frontier=pd.DataFrame(),
            )
        )
    return output


def test_development_gate_requires_frequency_sides_folds_and_xgb_solo():
    result = evaluate_development_ledger(_qualified_ledger(), _selections())

    assert result["development_pass"]
    assert result["trades"] == 140
    assert result["long_trades"] == 70
    assert result["short_trades"] == 70
    assert result["xgb_solo_trades"] == 15

    low_frequency = evaluate_development_ledger(
        _qualified_ledger(86), _selections()
    )
    assert not low_frequency["development_non_regression"]
    assert not low_frequency["development_pass"]


def test_satellite_ablation_reconciles_exact_increment():
    ledger = _qualified_ledger()

    ablation = xgb_satellite_ablation(ledger)

    solo = ledger.loc[ledger["route"].eq("xgb_satellite"), "net_return"]
    assert ablation["satellite_trades"] == len(solo)
    assert ablation["incremental_net_return"] == pytest.approx(float(solo.sum()))
    assert ablation["without_satellite_trades"] + ablation["satellite_trades"] == len(
        ledger
    )
