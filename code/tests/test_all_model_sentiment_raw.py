from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


def test_protocol_covers_all_models_arms_and_dead_zones_without_policy():
    from experiments.all_model_sentiment_raw import (
        ARMS,
        MODEL_NAMES,
        WIDTHS,
        expected_combination_count,
        protocol_manifest,
    )

    assert ARMS == ("none", "classic", "llm", "llm_full")
    assert WIDTHS == (55, 65, 75)
    assert len(MODEL_NAMES) == 9
    assert expected_combination_count() == 108
    manifest = protocol_manifest("logreg", "classic")
    assert manifest["confidence_threshold"] is None
    assert manifest["tp_bps"] is None
    assert manifest["sl_bps"] is None
    assert manifest["policy_calibration_used"] is False
    assert manifest["one_minute_execution_used"] is False
    assert manifest["hold_bars"] == [1]
    assert manifest["lookback_days"] == [180]


def test_raw_artifact_contract_is_three_rows_per_arm_model():
    from experiments.all_model_sentiment_raw import expected_raw_model_counts

    assert expected_raw_model_counts() == {
        "classification_2024.parquet": 3,
        "raw_forward_summary.parquet": 3,
    }


def test_sequence_runs_each_arm_model_pair_once(monkeypatch, tmp_path: Path):
    from experiments import all_model_sentiment_raw as module

    calls = []

    def fake_run_one(arm, model_name, **kwargs):
        calls.append((arm, model_name))
        return {"status": "complete"}

    monkeypatch.setattr(module, "run_one", fake_run_one)
    result = module.run_sequence(
        arms=("none", "classic"),
        model_names=("logreg", "svm_linear"),
        output_root=tmp_path,
        prepared_by_arm={"none": object(), "classic": object()},
        smoke=False,
    )

    assert calls == [
        ("none", "logreg"),
        ("none", "svm_linear"),
        ("classic", "logreg"),
        ("classic", "svm_linear"),
    ]
    assert result["completed"] == [f"{arm}/{model}" for arm, model in calls]


def test_raw_selection_rejects_policy_columns():
    from experiments.all_model_sentiment_raw import validate_raw_forward

    frame = pd.DataFrame(
        {
            "model_name": ["logreg"] * 3,
            "width_bps": [55, 65, 75],
            "lookback_days": [180] * 3,
            "hold_minutes": [15] * 3,
            "period_start": [pd.Timestamp("2025-07-01", tz="UTC")] * 3,
            "period_end": [pd.Timestamp("2026-04-01", tz="UTC")] * 3,
            "trades": [10, 11, 12],
        }
    )
    validate_raw_forward(frame, model_name="logreg")
    with pytest.raises(ValueError, match="policy columns"):
        validate_raw_forward(frame.assign(tau=0.7), model_name="logreg")


def test_scoreboard_has_one_row_per_arm_model_dead_zone(monkeypatch, tmp_path: Path):
    from experiments import all_model_sentiment_scoreboard as module

    monkeypatch.setattr(module, "MODEL_NAMES", ("logreg",))
    for arm in module.ARMS:
        root = tmp_path / arm / "logreg"
        root.mkdir(parents=True)
        pd.DataFrame(
            {
                "model_name": ["logreg"] * 3,
                "sentiment_arm": [arm] * 3,
                "width_bps": [55, 65, 75],
                "overall_f1": [0.2, 0.3, 0.4],
                "robust_f1": [0.1, 0.2, 0.3],
            }
        ).to_parquet(root / "classification_2024.parquet", index=False)
        pd.DataFrame(
            {
                "model_name": ["logreg"] * 3,
                "sentiment_arm": [arm] * 3,
                "width_bps": [55, 65, 75],
                "lookback_days": [180] * 3,
                "hold_minutes": [15] * 3,
                "period_start": [pd.Timestamp("2025-07-01", tz="UTC")] * 3,
                "period_end": [pd.Timestamp("2026-04-01", tz="UTC")] * 3,
                "trades": [50, 51, 52],
                "n_long": [25, 26, 26],
                "n_short": [25, 25, 26],
                "gross_return": [0.02, 0.03, 0.04],
                "net_return": [0.01, 0.02, 0.03],
                "sortino": [0.1, 0.2, 0.3],
                "sharpe": [0.05, 0.1, 0.15],
                "positive_months": [4, 5, 6],
            }
        ).to_parquet(root / "raw_forward_summary.parquet", index=False)

    monkeypatch.setattr(
        module,
        "validate_raw_model_artifacts",
        lambda *args, **kwargs: {},
    )
    tables = module.build_scoreboards(tmp_path)
    assert len(tables["classification"]) == 12
    assert len(tables["economics"]) == 12
    assert not module.POLICY_COLUMNS.intersection(tables["economics"].columns)
