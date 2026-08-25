from __future__ import annotations

import pandas as pd
import pytest

def test_invalid_candidate_is_rejected_not_fatal():
    from memory.loop import ReflectionWindow, WeeklyReflectionLoop
    from memory.schema import Lesson

    bad = Lesson(condition="hour >= 0", action="downweight", target="not_a_model",
                 factor=0.5, evidence="invalid target", confidence=0.9)
    good = Lesson(condition="hour >= 0", action="force_flat", target="catboost",
                  factor=1.0, evidence="ok", confidence=0.9)

    class Proposer:
        def propose(self, report, active):
            return [bad, good]

    loop = WeeklyReflectionLoop(
        proposer=Proposer(),
        report_fn=lambda window: {},
        score_fn=lambda lessons, window: 0.4 + 0.1 * len(lessons),  # more lessons = better
        feature_columns={"hour"},
        model_names={"catboost"},
    )
    window = ReflectionWindow(
        name="w0",
        train_start=pd.Timestamp("2024-07-01", tz="UTC"),
        train_end=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_start=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_end=pd.Timestamp("2025-01-07", tz="UTC"),
    )
    result = loop.run_window(window)

    assert len(result.rejected) == 1
    assert result.rejected[0].metadata.get("invalid_reason", "").startswith("unknown target")
    assert len(result.accepted) == 1


def _window(name: str, start: str) -> "ReflectionWindow":
    from memory.loop import ReflectionWindow

    begin = pd.Timestamp(start, tz="UTC")
    return ReflectionWindow(
        name=name,
        train_start=begin - pd.Timedelta(days=180),
        train_end=begin,
        validation_start=begin,
        validation_end=begin + pd.Timedelta(days=7),
    )


def test_accepted_lessons_accumulate_leave_one_out_track_record():
    from memory.loop import WeeklyReflectionLoop
    from memory.schema import Lesson

    lesson = Lesson(condition="hour >= 20", action="force_flat", target="catboost",
                    factor=1.0, evidence="late-day noise", confidence=0.8)

    class Proposer:
        def __init__(self):
            self.calls = []

        def propose(self, report, active):
            import copy

            self.calls.append([copy.deepcopy(l.metadata) for l in active])
            return [lesson] if not self.calls[1:] else []

    proposer = Proposer()
    loop = WeeklyReflectionLoop(
        proposer=proposer,
        report_fn=lambda window: {},
        score_fn=lambda lessons, window: 0.4 + 0.1 * len(lessons),
        feature_columns={"hour"},
        model_names={"catboost"},
    )
    loop.run_window(_window("w0", "2025-01-01"))
    loop.run_window(_window("w1", "2025-01-08"))

    record = loop.store.accepted[0].metadata["track_record"]
    assert record["windows"] == ["w0", "w1"]
    assert record["marginal_scores"] == [pytest.approx(0.1), pytest.approx(0.1)]
    # the proposer for w1 saw the lesson WITH its w0 history attached
    assert proposer.calls[1][0]["track_record"]["windows"] == ["w0"]


def test_track_records_can_be_disabled():
    from memory.loop import WeeklyReflectionLoop
    from memory.schema import Lesson

    lesson = Lesson(condition="hour >= 20", action="force_flat", target="catboost",
                    factor=1.0, evidence="x", confidence=0.8)

    class Proposer:
        def propose(self, report, active):
            return [lesson]

    loop = WeeklyReflectionLoop(
        proposer=Proposer(),
        report_fn=lambda window: {},
        score_fn=lambda lessons, window: 0.4 + 0.1 * len(lessons),
        feature_columns={"hour"},
        model_names={"catboost"},
        track_records=False,
    )
    loop.run_window(_window("w0", "2025-01-01"))
    assert "track_record" not in loop.store.accepted[0].metadata
