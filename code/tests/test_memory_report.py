from __future__ import annotations

import pandas as pd
import pytest

from memory.loop import ReflectionWindow
from memory.report import build_error_report


def _window() -> ReflectionWindow:
    return ReflectionWindow(
        name="wf_000_2025-01-01",
        train_start=pd.Timestamp("2024-07-01", tz="UTC"),
        train_end=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_start=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_end=pd.Timestamp("2025-01-02", tz="UTC"),
    )

def test_build_error_report_summarizes_models_conditions_and_worst_rows():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "y_true": [2, 2, 1, 0, 0, 1, 2, 0],
        "lstm_pred": [2, 1, 1, 2, 0, 1, 0, 0],
        "catboost_pred": [2, 2, 1, 0, 1, 1, 2, 0],
        "lstm_conf": [0.2, 0.9, 0.1, 0.8, 0.4, 0.3, 0.95, 0.2],
        "vol_regime": ["low", "high", "low", "high", "high", "low", "high", "low"],
        "event_bucket": ["none", "near", "none", "near", "near", "none", "near", "none"],
        "headline": ["", "CPI surprise", "", "Fed headline", "", "", "ETF headline", ""],
        "forward_return": [0.01, -0.01, 0.0, -0.02, 0.01, 0.0, 0.02, -0.01],
    }, index=idx)

    report = build_error_report(
        _window(),
        df,
        {"lstm": "lstm_pred", "catboost": "catboost_pred"},
        condition_cols=["vol_regime", "event_bucket"],
        news_cols=["headline"],
        focus_model="lstm",
        confidence_col="lstm_conf",
        forward_return_col="forward_return",
        min_group_size=1,
        max_worst=2,
    )

    assert report["window"]["name"] == "wf_000_2025-01-01"
    assert report["n_bars"] == 8
    assert report["label_distribution"] == {"0": 3, "1": 2, "2": 3}
    assert set(report["overall"]) == {"lstm", "catboost"}
    assert report["overall"]["catboost"]["macro_f1"] > report["overall"]["lstm"]["macro_f1"]
    assert "high" in report["by_condition"]["vol_regime"]
    assert report["worst_predictions"][0]["confidence"] == 0.95
    assert report["worst_predictions"][0]["news_context"]["headline"] == "ETF headline"

def test_build_error_report_rejects_missing_prediction_column():
    idx = pd.date_range("2025-01-01", periods=2, freq="15min", tz="UTC")
    df = pd.DataFrame({"y_true": [1, 2]}, index=idx)

    with pytest.raises(ValueError, match="missing prediction column"):
        build_error_report(_window(), df, {"lstm": "lstm_pred"})


def test_error_report_includes_ensemble_diagnostics():
    idx = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "y_true": [2, 2, 0, 0, 1, 1],
        "catboost_pred": [2, 0, 0, 2, 1, 2],
        "gru_pred": [2, 2, 2, 0, 1, 1],
        "ensemble_pred": [2, 0, 2, 2, 1, 2],
        "ensemble_conf": [0.91, 0.88, 0.93, 0.82, 0.76, 0.97],
        "hour": [9, 9, 10, 10, 11, 11],
        "vol_regime": ["low", "low", "high", "high", "low", "high"],
        "forward_return": [0.01, -0.01, -0.02, 0.02, 0.0, -0.01],
    }, index=idx)

    report = build_error_report(
        _window(),
        df,
        {"catboost": "catboost_pred", "gru": "gru_pred", "ensemble3": "ensemble_pred"},
        condition_cols=["hour", "vol_regime"],
        focus_model="ensemble3",
        confidence_col="ensemble_conf",
        forward_return_col="forward_return",
        min_group_size=1,
        max_worst=3,
    )

    assert report["model_disagreement"]["overall_rate"] > 0
    assert report["model_disagreement"]["top_conditions"][0]["condition"] in {"hour", "vol_regime"}
    assert report["single_model_weaknesses"][0]["model"] in {"catboost", "gru", "ensemble3"}
    assert report["single_model_weaknesses"][0]["error_rate"] >= 0.5
    assert report["high_confidence_errors"][0]["confidence"] == 0.97
