from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.svm_temperature_calibration import (
    TEMPERATURE_GRID,
    apply_temperature,
    select_temperature,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "svm_temperature_calibration"


def test_temperature_preserves_argmax_and_normalisation():
    probabilities = np.array(
        [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.6, 0.2]], dtype=float
    )
    base_class = probabilities.argmax(axis=1)

    for temperature in TEMPERATURE_GRID:
        calibrated = apply_temperature(probabilities, temperature)
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
        np.testing.assert_array_equal(calibrated.argmax(axis=1), base_class)
        assert (calibrated > 0.0).all()


def test_select_temperature_minimises_natural_prevalence_log_loss():
    probabilities = np.array(
        [[0.98, 0.01, 0.01], [0.01, 0.98, 0.01], [0.01, 0.01, 0.98]] * 20,
        dtype=float,
    )
    labels = np.tile([1, 2, 0], 20)

    selected, grid = select_temperature(probabilities, labels)

    assert selected == float(grid.sort_values(["log_loss", "temperature"]).iloc[0]["temperature"])
    assert set(grid["temperature"]) == set(TEMPERATURE_GRID)
    assert grid["changed_classes"].eq(0).all()


def test_svm_temperature_artifacts_are_h1_only_and_class_preserving():
    summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
    h1 = pd.read_csv(CACHE / "h1_confirmation.csv")
    calibrated = pd.read_parquet(CACHE / "calibrated_h1.parquet")

    assert summary["selected_temperature"] in TEMPERATURE_GRID
    assert summary["selection_period"] == "2024_oof"
    assert summary["confirmation_period"] == "2025_h1"
    assert summary["forward_loaded"] is False
    assert summary["lockbox_2026_q2_used"] is False
    assert int(summary["changed_classes_2024"]) == 0
    assert int(summary["changed_classes_h1"]) == 0
    assert set(h1["arm"]) == {"raw", "temperature_scaled"}
    assert pd.to_datetime(calibrated["timestamp"], utc=True).max() < pd.Timestamp(
        "2025-07-01", tz="UTC"
    )
    assert np.allclose(
        calibrated[["p_short_cal", "p_flat_cal", "p_long_cal"]].sum(axis=1),
        1.0,
    )
