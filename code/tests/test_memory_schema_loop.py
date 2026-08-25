from __future__ import annotations

import pandas as pd
import pytest

from memory.gates import LessonGateSet
from memory.loop import ReflectionWindow, WeeklyReflectionLoop
from memory.schema import Lesson


FEATURES = {"minutes_to_high_impact_event", "vol_regime"}
MODELS = {"lstm", "catboost"}


def test_lesson_schema_accepts_design_example():
    lesson = Lesson.from_dict({
        "condition": "minutes_to_high_impact_event < 30 AND vol_regime == 'high'",
        "action": "downweight",
        "target": "lstm",
        "factor": 0.5,
        "evidence": "LSTM underperformed near high-impact events.",
        "confidence": 0.7,
    })

    assert lesson.validate(FEATURES, MODELS) is lesson


def test_lesson_schema_rejects_unknown_condition_feature():
    lesson = Lesson(
        condition="future_return > 0",
        action="downweight",
        target="lstm",
        factor=0.5,
        evidence="leaky feature should fail",
        confidence=0.7,
    )

    with pytest.raises(ValueError, match="unknown condition feature"):
        lesson.validate(FEATURES, MODELS)


def test_gate_set_downweights_only_matching_bars():
    idx = pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC")
    features = pd.DataFrame({
        "minutes_to_high_impact_event": [10, 90, 5],
        "vol_regime": ["high", "high", "low"],
    }, index=idx)
    contrib = pd.DataFrame({"lstm": [1.0, 1.0, 1.0], "catboost": [1.0, 1.0, 1.0]}, index=idx)
    lesson = Lesson(
        condition="minutes_to_high_impact_event < 30 AND vol_regime == 'high'",
        action="downweight",
        target="lstm",
        factor=0.5,
        evidence="test",
        confidence=0.7,
        status="accepted",
    )

    gated = LessonGateSet((lesson,)).apply_model_gates(contrib, features)

    assert gated["lstm"].tolist() == [0.5, 1.0, 1.0]
    assert gated["catboost"].tolist() == [1.0, 1.0, 1.0]


def test_weekly_loop_keeps_only_improving_lesson():
    candidate = Lesson(
        condition="minutes_to_high_impact_event < 30",
        action="downweight",
        target="lstm",
        factor=0.5,
        evidence="test",
        confidence=0.7,
    )

    class Proposer:
        def propose(self, report, active_lessons):
            return [candidate]

    window = ReflectionWindow(
        name="wf_000_2025-01-01",
        train_start=pd.Timestamp("2024-07-01", tz="UTC"),
        train_end=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_start=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_end=pd.Timestamp("2025-01-08", tz="UTC"),
    )

    def score(lessons, _window):
        return 1.1 if lessons else 1.0

    loop = WeeklyReflectionLoop(
        proposer=Proposer(),
        report_fn=lambda _window: {"macro_f1": {"lstm": 0.31}},
        score_fn=score,
        feature_columns=FEATURES,
        model_names=MODELS,
        epsilon=0.01,
    )

    result = loop.run_window(window)

    assert len(result.accepted) == 1
    assert len(loop.store.active_lessons()) == 1
    assert result.accepted[0].score_after == 1.1
