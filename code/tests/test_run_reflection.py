from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.run_reflection import (
    apply_lessons_out_of_sample,
    paired_oos_significance,
    paired_oos_significance_frame,
    summarize_oos_memory,
    summarize_oos_windows,
    lessons_to_frame,
    run_reflection_from_cache,
    windows_from_predictions,
)
from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from memory.schema import Lesson


class ToyClassifier:
    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, X):
        out = np.full((len(X), 3), 0.1)
        out[:, 1] = 0.8
        return out


class OneLessonProposer:
    def propose(self, report, active_lessons):
        return [
            Lesson(
                condition="sentiment_flag == 1",
                action="force_flat",
                target="catboost",
                factor=1.0,
                evidence="directional calls were wrong when sentiment flag was active",
                confidence=0.8,
            )
        ]


def test_walkforward_cache_can_carry_validation_feature_columns(tmp_path):
    idx = pd.date_range("2024-12-01", "2025-01-07 23:45", freq="15min", tz="UTC")
    X = pd.DataFrame(
        {
            "signal": np.resize([0.0, 1.0], len(idx)),
            "sentiment_flag": np.resize([0, 1], len(idx)),
        },
        index=idx,
    )
    y = pd.Series(np.resize([1, 2], len(idx)), index=idx)
    windows = weekly_walkforward_windows(idx, "2025-01-01", "2025-01-07", train_lookback="7D")
    cache_path = tmp_path / "wf.parquet"

    preds = run_walkforward_predictions(
        X,
        y,
        windows=windows,
        model_factory=lambda: ToyClassifier(),
        cache_path=cache_path,
        include_features=True,
    )

    disk = pd.read_parquet(cache_path)
    assert {"signal", "sentiment_flag"} <= set(preds.columns)
    assert {"signal", "sentiment_flag"} <= set(disk.columns)


def test_reflection_runner_persists_keep_if_better_lessons(tmp_path):
    idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    predictions = pd.DataFrame(
        {
            "window": ["wf_000_2025-01-01"] * 4,
            "train_start": [pd.Timestamp("2024-07-01", tz="UTC")] * 4,
            "train_end": [pd.Timestamp("2025-01-01", tz="UTC")] * 4,
            "validation_start": [idx.min()] * 4,
            "validation_end": [idx.max()] * 4,
            "y_true": [1, 1, 2, 0],
            "catboost_pred": [0, 1, 2, 0],
            "catboost_conf": [0.9, 0.8, 0.7, 0.6],
            "sentiment_flag": [1, 0, 0, 0],
            "forward_return": [0.0, 0.0, 0.01, -0.01],
        },
        index=idx,
    )
    out_path = tmp_path / "lessons.parquet"

    results = run_reflection_from_cache(
        predictions,
        proposer=OneLessonProposer(),
        feature_columns=("sentiment_flag",),
        condition_cols=("sentiment_flag",),
        out_path=out_path,
    )

    windows = windows_from_predictions(predictions)
    lesson_frame = lessons_to_frame(results)
    disk = pd.read_parquet(out_path)

    assert windows[0].name == "wf_000_2025-01-01"
    assert lesson_frame["status"].tolist() == ["accepted"]
    assert lesson_frame["score_after"].iloc[0] > lesson_frame["score_before"].iloc[0]
    assert disk["status"].tolist() == ["accepted"]


def test_out_of_sample_memory_applies_lessons_only_after_source_window():
    idx = pd.date_range("2025-01-01", periods=4, freq="7D", tz="UTC")
    predictions = pd.DataFrame(
        {
            "window": ["wf_000_2025-01-01", "wf_001_2025-01-08"] * 2,
            "train_start": [pd.Timestamp("2024-07-01", tz="UTC")] * 4,
            "train_end": [pd.Timestamp("2025-01-01", tz="UTC")] * 4,
            "validation_start": [idx.min()] * 4,
            "validation_end": [idx.max()] * 4,
            "y_true": [1, 1, 1, 1],
            "catboost_pred": [0, 0, 0, 0],
            "catboost_conf": [0.9, 0.9, 0.9, 0.9],
            "sentiment_flag": [1, 1, 1, 0],
        },
        index=idx,
    )
    lessons = pd.DataFrame(
        [
            {
                "condition": "sentiment_flag == 1",
                "action": "force_flat",
                "target": "catboost",
                "factor": 1.0,
                "evidence": "test",
                "confidence": 0.8,
                "status": "accepted",
                "source_window": "wf_000_2025-01-01",
                "score_before": 0.1,
                "score_after": 0.2,
                "metadata": "{}",
            }
        ]
    )

    out = apply_lessons_out_of_sample(
        predictions,
        lessons,
        feature_columns=("sentiment_flag",),
    )

    first = out[out["window"] == "wf_000_2025-01-01"]
    second = out[out["window"] == "wf_001_2025-01-08"]
    assert first["memory_pred"].tolist() == first["baseline_pred"].tolist()
    assert second["memory_pred"].tolist() == [1, 0]
    assert first["active_lesson_count"].tolist() == [0, 0]
    assert second["active_lesson_count"].tolist() == [1, 1]



def test_out_of_sample_forgetting_retires_stale_lessons_after_harmful_window():
    idx = pd.date_range("2025-01-01", periods=6, freq="7D", tz="UTC")
    predictions = pd.DataFrame(
        {
            "window": ["wf_000_2025-01-01"] * 2
            + ["wf_001_2025-01-08"] * 2
            + ["wf_002_2025-01-15"] * 2,
            "train_start": [pd.Timestamp("2024-07-01", tz="UTC")] * 6,
            "train_end": [pd.Timestamp("2025-01-01", tz="UTC")] * 6,
            "validation_start": [idx.min()] * 6,
            "validation_end": [idx.max()] * 6,
            "y_true": [1, 1, 0, 0, 0, 0],
            "catboost_pred": [0, 0, 0, 0, 0, 0],
            "catboost_conf": [0.9] * 6,
            "sentiment_flag": [1, 1, 1, 1, 1, 1],
        },
        index=idx,
    )
    lessons = pd.DataFrame(
        [
            {
                "condition": "sentiment_flag == 1",
                "action": "force_flat",
                "target": "catboost",
                "factor": 1.0,
                "evidence": "test",
                "confidence": 0.8,
                "status": "accepted",
                "source_window": "wf_000_2025-01-01",
                "score_before": 0.1,
                "score_after": 0.2,
                "metadata": "{}",
            }
        ]
    )

    out = apply_lessons_out_of_sample(
        predictions,
        lessons,
        feature_columns=("sentiment_flag",),
        max_harmful_windows=1,
    )

    second = out[out["window"] == "wf_001_2025-01-08"]
    third = out[out["window"] == "wf_002_2025-01-15"]
    assert second["memory_pred"].tolist() == [1, 1]
    assert third["memory_pred"].tolist() == third["baseline_pred"].tolist()
    assert second["active_lesson_count"].tolist() == [1, 1]
    assert third["active_lesson_count"].tolist() == [0, 0]
    assert third["retired_lesson_count"].tolist() == [1, 1]


def test_oos_memory_summary_includes_economics_and_paired_significance():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    gated = pd.DataFrame(
        {
            "window": ["wf_000_2025-01-01"] * 2 + ["wf_001_2025-01-08"] * 2,
            "y_true": [2, 1, 0, 1],
            "baseline_pred": [2, 0, 2, 1],
            "memory_pred": [2, 1, 0, 1],
            "changed_by_memory": [False, True, True, False],
            "active_lesson_count": [0, 0, 1, 1],
            "retired_lesson_count": [0, 0, 0, 0],
            "forward_return": [0.01, -0.01, -0.02, 0.0],
        },
        index=idx,
    )

    overall = summarize_oos_memory(gated, fee_bps=5.0)
    weekly = summarize_oos_windows(gated, fee_bps=5.0)
    sig = paired_oos_significance(weekly)

    assert {"net_return_sum", "sharpe", "trade_count"} <= set(overall.columns)
    assert {
        "flattened_baseline_return_mean_bps",
        "kept_directional_baseline_return_mean_bps",
    } <= set(overall.columns)
    assert {"baseline_macro_f1", "memory_macro_f1", "delta_macro_f1"} <= set(weekly.columns)
    assert sig["n_windows"] == 2
    assert "wilcoxon_pvalue" in sig
    assert overall.loc[overall["variant"] == "memory", "net_return_sum"].iloc[0] > overall.loc[
        overall["variant"] == "baseline", "net_return_sum"
    ].iloc[0]



def test_net_return_retention_keeps_economic_lesson_despite_macro_f1_harm():
    idx = pd.date_range("2025-01-01", periods=6, freq="7D", tz="UTC")
    predictions = pd.DataFrame(
        {
            "window": ["wf_000_2025-01-01"] * 2
            + ["wf_001_2025-01-08"] * 2
            + ["wf_002_2025-01-15"] * 2,
            "train_start": [pd.Timestamp("2024-07-01", tz="UTC")] * 6,
            "train_end": [pd.Timestamp("2025-01-01", tz="UTC")] * 6,
            "validation_start": [idx.min()] * 6,
            "validation_end": [idx.max()] * 6,
            "y_true": [1, 1, 0, 2, 0, 2],
            "catboost_pred": [0, 0, 0, 2, 0, 2],
            "catboost_conf": [0.9] * 6,
            "sentiment_flag": [1, 1, 1, 1, 1, 1],
            "forward_return": [0.0, 0.0, 0.01, -0.01, 0.01, -0.01],
        },
        index=idx,
    )
    lessons = pd.DataFrame(
        [
            {
                "condition": "sentiment_flag == 1",
                "action": "force_flat",
                "target": "catboost",
                "factor": 1.0,
                "evidence": "test",
                "confidence": 0.8,
                "status": "accepted",
                "source_window": "wf_000_2025-01-01",
                "score_before": 0.1,
                "score_after": 0.2,
                "metadata": "{}",
            }
        ]
    )

    out = apply_lessons_out_of_sample(
        predictions,
        lessons,
        feature_columns=("sentiment_flag",),
        max_harmful_windows=1,
        retention_metric="net_return",
        fee_bps=0.0,
    )

    third = out[out["window"] == "wf_002_2025-01-15"]
    assert third["memory_pred"].tolist() == [1, 1]
    assert third["active_lesson_count"].tolist() == [1, 1]
    assert third["retired_lesson_count"].tolist() == [0, 0]


def test_significance_frame_reports_macro_f1_and_net_return_rows():
    weekly = pd.DataFrame(
        {
            "delta_macro_f1": [0.1, -0.1, 0.0],
            "delta_net_return_sum": [0.02, 0.01, 0.0],
        }
    )

    sig = paired_oos_significance_frame(weekly)

    assert sig["metric"].tolist() == ["macro_f1", "net_return_sum"]
    assert set(sig.columns) >= {"metric", "n_windows", "wilcoxon_pvalue"}


def test_wf_feature_columns_keep_confidence_drop_class_outputs():
    from experiments.run_reflection import wf_feature_columns

    predictions = pd.DataFrame(
        columns=["window", "train_start", "train_end", "validation_start",
                 "validation_end", "y_true", "forward_return", "gru_pred",
                 "gru_conf", "gru_p0", "gru_p1", "gru_p2", "vol_regime", "hour"]
    )
    cols = wf_feature_columns(predictions, "gru")

    assert "gru_conf" in cols            # confidence is a legitimate condition feature
    assert "gru_pred" not in cols and "gru_p0" not in cols
    assert "vol_regime" in cols and "hour" in cols


def test_oos_confidence_gate_applies_to_baseline_and_memory():
    from experiments.run_reflection import apply_lessons_out_of_sample, summarize_oos_memory

    idx = pd.date_range("2025-01-01", periods=4, freq="7D", tz="UTC")
    predictions = pd.DataFrame(
        {
            "window": ["wf_000", "wf_000", "wf_001", "wf_001"],
            "train_start": [pd.Timestamp("2024-07-01", tz="UTC")] * 4,
            "train_end": [pd.Timestamp("2025-01-01", tz="UTC")] * 4,
            "validation_start": [idx.min()] * 4,
            "validation_end": [idx.max()] * 4,
            "y_true": [2, 2, 2, 2],
            "gru_pred": [2, 2, 2, 2],
            "gru_conf": [0.9, 0.3, 0.9, 0.3],   # half the bars are low confidence
            "forward_return": [0.01, 0.01, 0.01, 0.01],
            "flag": [0, 0, 0, 0],
        },
        index=idx,
    )
    gated = apply_lessons_out_of_sample(
        predictions,
        pd.DataFrame(),                        # no lessons: memory == baseline
        feature_columns=("flag",),
        pred_col="gru_pred",
        conf_col="gru_conf",
    )
    assert "conf" in gated.columns

    free = summarize_oos_memory(gated, fee_bps=0.0, tau=0.0)
    gated_summary = summarize_oos_memory(gated, fee_bps=0.0, tau=0.5)
    base_free = free.loc[free["variant"] == "baseline"].iloc[0]
    base_gated = gated_summary.loc[gated_summary["variant"] == "baseline"].iloc[0]

    assert base_free["exposure"] == pytest.approx(1.0)       # ungated: always long
    assert base_gated["exposure"] == pytest.approx(0.5)      # gate drops low-conf bars
    assert base_gated["net_return_sum"] == pytest.approx(0.02)
