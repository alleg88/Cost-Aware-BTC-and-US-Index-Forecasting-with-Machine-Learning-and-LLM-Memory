from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments.ensemble_memory import (
    BASE_MODELS,
    apply_ensemble_lessons,
    ensemble_acceptance_score,
    ensemble_probabilities,
)
from memory.schema import Lesson


def _frame(n: int = 6) -> pd.DataFrame:
    """Three bases: catboost says down, gru says up (stronger), xgb neutral-flat."""
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    frame = pd.DataFrame(index=idx)
    frame[["catboost_balanced_p0", "catboost_balanced_p1", "catboost_balanced_p2"]] = \
        np.tile([0.5, 0.3, 0.2], (n, 1))
    frame[["gru_p0", "gru_p1", "gru_p2"]] = np.tile([0.1, 0.3, 0.6], (n, 1))
    frame[["xgboost_balanced_p0", "xgboost_balanced_p1", "xgboost_balanced_p2"]] = \
        np.tile([0.2, 0.6, 0.2], (n, 1))
    frame["hour"] = range(n)
    return frame


def _lesson(action: str, target: str = "gru", factor: float = 0.1,
            condition: str = "hour >= 3") -> Lesson:
    return Lesson(condition=condition, action=action, target=target, factor=factor,
                  evidence="test", confidence=0.9, status="accepted")

def test_equal_weights_average_probabilities():
    probs = ensemble_probabilities(_frame())
    np.testing.assert_allclose(probs.iloc[0], [0.8 / 3, 1.2 / 3, 1.0 / 3])

def test_downweight_flips_ensemble_where_condition_holds():
    frame = _frame()
    baseline = apply_ensemble_lessons(frame, [], ("hour",))
    assert baseline.unique().tolist() == [1]          # flat wins the equal-weight vote

    # upweighting gru makes up win, but only on bars where the condition holds
    lesson = _lesson("upweight", target="gru", factor=5.0)
    pred = apply_ensemble_lessons(frame, [lesson], ("hour",))
    assert pred[frame["hour"] < 3].unique().tolist() == [1]
    assert pred[frame["hour"] >= 3].unique().tolist() == [2]

def test_force_flat_overrides_everything():
    frame = _frame()
    lessons = [_lesson("upweight", target="gru", factor=5.0, condition="hour >= 0"),
               _lesson("force_flat", condition="hour >= 0")]
    pred = apply_ensemble_lessons(frame, lessons, ("hour",))
    assert pred.unique().tolist() == [1]

def test_widen_deadzone_tightens_the_gate_conditionally():
    frame = _frame()
    up = _lesson("upweight", target="gru", factor=5.0, condition="hour >= 0")
    # gated at tau=0.4: boosted gru vote clears it everywhere -> all up
    pred = apply_ensemble_lessons(frame, [up], ("hour",), tau=0.4)
    assert pred.unique().tolist() == [2]

    # widen_deadzone x2 -> required confidence 0.8 on hour >= 3 -> those bars go flat
    widen = _lesson("widen_deadzone", factor=2.0, condition="hour >= 3")
    pred = apply_ensemble_lessons(frame, [up, widen], ("hour",), tau=0.4)
    assert pred[frame["hour"] < 3].unique().tolist() == [2]
    assert pred[frame["hour"] >= 3].unique().tolist() == [1]

def test_unknown_target_weight_lesson_is_ignored():
    frame = _frame()
    lesson = _lesson("upweight", target="not_a_model", factor=9.0, condition="hour >= 0")
    pred = apply_ensemble_lessons(frame, [lesson], ("hour",))
    assert pred.unique().tolist() == [1]              # no effect, no crash


def test_oos_forgetting_retires_consecutively_harmful_lesson():
    from experiments.ensemble_memory import ENSEMBLE_TAG, apply_ensemble_lessons_oos

    # 4 weekly windows x 8 bars; truth always flat, market always falls, so any
    # directional (up) prediction loses money every bar
    n_windows, bars = 4, 8
    parts = []
    for w in range(n_windows):
        start = pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=7 * w)
        idx = pd.date_range(start, periods=bars, freq="15min")
        frame = _frame(bars)
        frame.index = idx
        frame["window"] = f"wf_{w:03d}"
        frame["train_start"] = start - pd.Timedelta(days=180)
        frame["train_end"] = start
        frame["validation_start"] = start
        frame["validation_end"] = start + pd.Timedelta(days=7)
        frame["y_true"] = 1
        frame["forward_return"] = -0.01
        parts.append(frame)
    predictions = pd.concat(parts)
    probs = np.tile([0.8 / 3, 1.2 / 3, 1.0 / 3], (len(predictions), 1))
    predictions[f"{ENSEMBLE_TAG}_pred"] = 1
    predictions[f"{ENSEMBLE_TAG}_conf"] = probs.max(axis=1)
    for i in range(3):
        predictions[f"{ENSEMBLE_TAG}_p{i}"] = probs[:, i]

    harmful = Lesson(condition="hour >= 0", action="upweight", target="gru",
                     factor=5.0, evidence="forces up, loses every bar",
                     confidence=0.9, status="accepted", source_window="wf_000")

    kept = apply_ensemble_lessons_oos(
        predictions, [harmful], ("hour",), tau=0.0, fee_bps=0.0)
    assert (kept.loc[kept["window"] != "wf_000", "memory_pred"] == 2).all()
    assert kept["retired_lesson_count"].iloc[-1] == 0

    forgot = apply_ensemble_lessons_oos(
        predictions, [harmful], ("hour",), tau=0.0, fee_bps=0.0,
        max_harmful_windows=2, retention_metric="net_return")
    # harmful in wf_001 and wf_002 -> retired before wf_003
    assert forgot["retired_lesson_count"].iloc[-1] == 1
    assert (forgot.loc[forgot["window"] == "wf_003", "memory_pred"] == 1).all()
    assert (forgot.loc[forgot["window"] == "wf_002", "memory_pred"] == 2).all()


def test_acceptance_score_can_use_net_return_or_hybrid():
    frame = _frame(4)
    frame["y_true"] = [1, 1, 1, 1]
    frame["forward_return"] = [0.01, -0.02, 0.01, -0.02]
    feature_columns = ("hour",)

    baseline_net = ensemble_acceptance_score(
        frame,
        [],
        feature_columns,
        tau=0.0,
        metric="net_return",
        fee_bps=0.0,
    )
    lesson_net = ensemble_acceptance_score(
        frame,
        [_lesson("upweight", target="gru", factor=5.0, condition="hour >= 0")],
        feature_columns,
        tau=0.0,
        metric="net_return",
        fee_bps=0.0,
    )
    assert lesson_net < baseline_net

    hybrid = ensemble_acceptance_score(
        frame,
        [],
        feature_columns,
        tau=0.0,
        metric="hybrid",
        fee_bps=0.0,
    )
    assert hybrid > baseline_net
