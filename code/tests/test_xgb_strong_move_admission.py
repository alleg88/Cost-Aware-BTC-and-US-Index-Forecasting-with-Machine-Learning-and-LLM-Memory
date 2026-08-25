from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.xgb_strong_move_admission import (
    DIRECTION_THRESHOLDS,
    MOVE_THRESHOLDS,
    apply_binary_logit_calibrator,
    build_addon_signal,
    combine_with_addon,
    decompose_xgb_probabilities,
    fit_binary_logit_calibrator,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "xgb_strong_move_admission"


def test_xgb_probabilities_are_decomposed_into_move_and_conditional_side():
    frame = pd.DataFrame(
        {
            "p_short": [0.2, 0.6],
            "p_flat": [0.5, 0.2],
            "p_long": [0.3, 0.2],
        }
    )

    actual = decompose_xgb_probabilities(frame)

    np.testing.assert_allclose(actual["p_move_raw"], [0.5, 0.8])
    np.testing.assert_allclose(actual["p_long_given_move_raw"], [0.6, 0.25])


def test_binary_logit_calibrator_returns_finite_probabilities_and_intercept():
    raw = np.linspace(0.05, 0.95, 200)
    target = (raw > 0.7).astype(int)

    calibrator = fit_binary_logit_calibrator(raw, target)
    calibrated = apply_binary_logit_calibrator(raw, calibrator)

    assert np.isfinite(calibrator.slope)
    assert np.isfinite(calibrator.intercept)
    assert ((calibrated > 0.0) & (calibrated < 1.0)).all()
    assert np.all(np.diff(calibrated) >= 0.0)


def test_addon_requires_union_flat_three_way_side_support_and_two_thresholds():
    index = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    union = pd.DataFrame(
        {
            "union_signal": [0.0, 1.0, 0.0, 0.0, 0.0],
            "member_conflict": [False, False, True, False, False],
            "lstm_latent_side": [1.0, 1.0, 1.0, -1.0, 1.0],
            "svm_linear_latent_side": [1.0, 1.0, -1.0, -1.0, 1.0],
        },
        index=index,
    )
    xgb = pd.DataFrame(
        {
            "p_move_cal": [0.4, 0.4, 0.4, 0.4, 0.1],
            "p_long_given_move_cal": [0.8, 0.8, 0.8, 0.8, 0.8],
        },
        index=index,
    )

    actual = build_addon_signal(
        union,
        xgb,
        move_threshold=0.2,
        direction_threshold=0.7,
    )

    np.testing.assert_array_equal(actual.to_numpy(), [1.0, 0.0, 0.0, 0.0, 0.0])


def test_combined_signal_never_changes_an_existing_union_trade():
    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    union = pd.Series([1.0, 0.0, -1.0, 0.0], index=index, name="union_signal")
    addon = pd.Series([-1.0, 1.0, 1.0, -1.0], index=index, name="xgb_addon_signal")

    combined = combine_with_addon(union, addon)

    np.testing.assert_array_equal(combined.to_numpy(), [1.0, 1.0, -1.0, -1.0])
    pd.testing.assert_series_equal(
        combined.loc[union.ne(0.0)].rename("union_signal"), union.loc[union.ne(0.0)]
    )


def test_completed_artifacts_obey_h1_before_forward_contract():
    summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(CACHE / "h1_selection_grid.csv")
    calibration = pd.read_csv(CACHE / "calibration_metrics.csv")

    assert len(grid) == len(MOVE_THRESHOLDS) * len(DIRECTION_THRESHOLDS)
    assert set(grid["move_threshold"]) == set(MOVE_THRESHOLDS)
    assert set(grid["direction_threshold"]) == set(DIRECTION_THRESHOLDS)
    assert set(calibration["period"]) == {"2024_oof", "2025_h1"}
    assert set(calibration["target"]) == {"move", "direction_given_move"}
    assert summary["calibration_fit_period"] == "2024_oof"
    assert summary["policy_selection_period"] == "2025_q1"
    assert summary["policy_confirmation_period"] == "2025_q2_calendar"
    assert summary["lockbox_2026_q2_used"] is False
    if summary["h1_pass"]:
        assert summary["forward_loaded"] is True
        forward = pd.read_parquet(CACHE / "forward_combined_per_bar.parquet")
        assert pd.to_datetime(forward["timestamp"], utc=True).max() < pd.Timestamp(
            "2026-04-01", tz="UTC"
        )
    else:
        assert summary["forward_loaded"] is False
        assert not (CACHE / "forward_combined_per_bar.parquet").exists()
