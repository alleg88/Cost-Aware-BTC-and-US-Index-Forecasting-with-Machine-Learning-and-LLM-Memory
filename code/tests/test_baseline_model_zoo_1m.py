from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def _hold_row(**updates):
    row = {
        "model_name": "logreg",
        "width_bps": 55,
        "candidate_id": 0,
        "hold_bars": 1,
        "hold_minutes": 15,
        "trades": 60,
        "n_long": 30,
        "n_short": 30,
        "positive_segments": 4,
        "pooled_net": 0.02,
        "pooled_sortino": 1.0,
        "pooled_sharpe": 0.8,
        "robust_score": 0.5,
    }
    row.update(updates)
    return row


def test_protocol_is_baseline_only_with_fixed_180_day_history():
    from experiments.baseline_model_zoo_1m import (
        BASELINE_CANDIDATE_ID,
        HOLD_BARS,
        LOOKBACK_DAYS,
        WIDTHS,
    )

    assert BASELINE_CANDIDATE_ID == 0
    assert WIDTHS == (55, 65, 75)
    assert LOOKBACK_DAYS == (180,)
    assert HOLD_BARS == (1,)


def test_h1_policy_grid_has_33_rows_and_fixed_15_minute_hold():
    from experiments.baseline_model_zoo_1m import policy_choices_for_hold

    choices = policy_choices_for_hold(1)
    assert len(choices) == 33
    assert {geometry[2] for _, geometry in choices} == {1}
    with pytest.raises(ValueError, match="hold_bars"):
        policy_choices_for_hold(2)


def test_fixed_hold_row_is_validated_without_15_30_selection():
    from experiments.baseline_model_zoo_1m import select_hold_rows

    selected = select_hold_rows(pd.DataFrame([_hold_row()]))
    assert len(selected) == 1
    assert int(selected.iloc[0]["hold_bars"]) == 1
    assert int(selected.iloc[0]["hold_minutes"]) == 15
    assert float(selected.iloc[0]["constraint_violation"]) == 0.0


def test_fixed_hold_validation_rejects_30_minutes():
    from experiments.baseline_model_zoo_1m import select_hold_rows

    with pytest.raises(ValueError, match="fixed 15-minute hold"):
        select_hold_rows(pd.DataFrame([_hold_row(hold_bars=2, hold_minutes=30)]))


def test_expected_counts_match_one_baseline_and_fixed_hold_protocol():
    from experiments.baseline_model_zoo_1m import expected_model_counts

    assert expected_model_counts() == {
        "classification_2024.parquet": 3,
        "hold_grid_2024.parquet": 3,
        "selected_holds_2024.parquet": 3,
        "calibration_policy_grid_2025h1.parquet": 99,
        "selected_policies_2025h1.parquet": 3,
        "raw_forward_summary.parquet": 3,
        "forward_monthly.parquet": 27,
        "forward_quarterly.parquet": 9,
        "forward_summary.parquet": 3,
    }


@pytest.mark.parametrize("lookback_days", (180,))
def test_fit_plan_is_monthly_h1_then_one_frozen_forward_fit(lookback_days: int):
    from experiments.baseline_model_zoo_1m import fit_plan

    plan = fit_plan(75, lookback_days)
    assert len(plan) == 7
    assert [row["stage"] for row in plan[:6]] == ["calibration"] * 6
    assert [row["test_start"].month for row in plan[:6]] == [1, 2, 3, 4, 5, 6]
    assert all(row["lookback_days"] == lookback_days for row in plan)
    assert plan[-1]["stage"] == "forward"
    assert plan[-1]["train_end"] == pd.Timestamp("2025-07-01", tz="UTC")
    assert plan[-1]["test_end"] == pd.Timestamp("2026-04-01", tz="UTC")


def test_h1_selection_uses_fixed_history_and_selects_policy():
    from experiments.baseline_model_zoo_1m import select_h1_policy_rows

    rows = []
    for policy_id in range(33):
        rows.append(
            _hold_row(
                lookback_days=180,
                policy_id=policy_id,
                pooled_sortino=2.0 if policy_id == 0 else -1.0,
            )
        )
    selected = select_h1_policy_rows(pd.DataFrame(rows))
    assert len(selected) == 1
    assert int(selected.iloc[0]["lookback_days"]) == 180
    assert int(selected.iloc[0]["policy_id"]) == 0


def test_artifact_validation_rejects_missing_file(tmp_path: Path):
    from experiments.baseline_model_zoo_1m import validate_model_artifacts

    with pytest.raises(FileNotFoundError, match="classification_2024.parquet"):
        validate_model_artifacts(tmp_path, model_name="logreg")
