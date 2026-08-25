"""Weekly RQ3 reflection loop.

The loop receives injected report, scoring, and proposal functions.
This keeps model training, LLM calls, and validation scoring testable separately.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol

import pandas as pd

from memory.schema import Lesson


@dataclass(frozen=True)
class ReflectionWindow:
    """One weekly walk-forward step."""

    name: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp


@dataclass
class LessonStore:
    """In-memory lesson store used by the weekly loop."""

    accepted: list[Lesson] = field(default_factory=list)
    rejected: list[Lesson] = field(default_factory=list)

    def active_lessons(self) -> list[Lesson]:
        return list(self.accepted)

    def commit(self, lesson: Lesson) -> None:
        self.accepted.append(lesson)

    def reject(self, lesson: Lesson) -> None:
        self.rejected.append(lesson)


class LessonProposer(Protocol):
    def propose(self, report: dict[str, Any], active_lessons: list[Lesson]) -> Iterable[Lesson]:
        """Return candidate lessons from a compact error report."""


ScoreFn = Callable[[list[Lesson], ReflectionWindow], float]
ReportFn = Callable[[ReflectionWindow], dict[str, Any]]


@dataclass
class WindowResult:
    window: ReflectionWindow
    accepted: list[Lesson]
    rejected: list[Lesson]


def iter_weekly_windows(
    index: pd.DatetimeIndex,
    start: str,
    end: str,
    train_lookback: str = "180D",
) -> list[ReflectionWindow]:
    """Create weekly 2025 walk-forward windows without touching the lockbox."""
    idx = index.sort_values()
    first = pd.Timestamp(start, tz=idx.tz)
    last = pd.Timestamp(end, tz=idx.tz)
    starts = pd.date_range(first, last, freq="7D", tz=idx.tz)
    windows: list[ReflectionWindow] = []
    lookback = pd.Timedelta(train_lookback)
    for i, validation_start in enumerate(starts):
        validation_end = min(validation_start + pd.Timedelta(days=7), last)
        if validation_start >= validation_end:
            continue
        train_end = validation_start
        train_start = train_end - lookback
        windows.append(
            ReflectionWindow(
                name=f"wf_{i:03d}_{validation_start.date()}",
                train_start=train_start,
                train_end=train_end,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
    return windows


class WeeklyReflectionLoop:
    """Run propose -> keep-if-better -> store for each walk-forward window."""

    def __init__(
        self,
        proposer: LessonProposer,
        report_fn: ReportFn,
        score_fn: ScoreFn,
        feature_columns: Iterable[str],
        model_names: Iterable[str],
        epsilon: float = 0.0,
        store: LessonStore | None = None,
        track_records: bool = True,
    ):
        self.proposer = proposer
        self.report_fn = report_fn
        self.score_fn = score_fn
        self.feature_columns = set(feature_columns)
        self.model_names = set(model_names)
        self.epsilon = epsilon
        self.store = store or LessonStore()
        self.track_records = track_records

    def run_window(self, window: ReflectionWindow) -> WindowResult:
        report = self.report_fn(window)
        accepted: list[Lesson] = []
        rejected: list[Lesson] = []
        for candidate in self.proposer.propose(report, self.store.active_lessons()):
            try:
                candidate.validate(self.feature_columns, self.model_names)
            except ValueError as exc:
                # LLM output is untrusted: an invalid proposal is recorded as
                # rejected with the reason, never allowed to abort the run
                invalid = Lesson(
                    condition=candidate.condition,
                    action=candidate.action,
                    target=candidate.target,
                    factor=candidate.factor,
                    evidence=candidate.evidence,
                    confidence=candidate.confidence,
                    status="rejected",
                    source_window=window.name,
                    metadata={**candidate.metadata, "invalid_reason": str(exc)},
                )
                self.store.reject(invalid)
                rejected.append(invalid)
                continue
            active = self.store.active_lessons()
            score_before = self.score_fn(active, window)
            score_after = self.score_fn(active + [candidate], window)
            keep = score_after > score_before + self.epsilon
            decided = candidate.with_decision(
                "accepted" if keep else "rejected",
                score_before=score_before,
                score_after=score_after,
                source_window=window.name,
            )
            if keep:
                self.store.commit(decided)
                accepted.append(decided)
            else:
                self.store.reject(decided)
                rejected.append(decided)
        if self.track_records:
            self._update_track_records(window)
        return WindowResult(window=window, accepted=accepted, rejected=rejected)

    def _update_track_records(self, window: ReflectionWindow) -> None:
        """Record each active lesson's leave-one-out marginal score on this window.

        The history is stored in lesson.metadata (mutable on the frozen dataclass)
        so the proposer can show the LLM how its past lessons performed and the
        persisted decision table keeps the audit trail. Marginals for window N
        only reach prompts from window N+1 on, so nothing leaks.
        """
        active = self.store.active_lessons()
        if not active:
            return
        full_score = self.score_fn(active, window)
        for lesson in active:
            others = [item for item in active if item is not lesson]
            marginal = full_score - self.score_fn(others, window)
            record = lesson.metadata.setdefault(
                "track_record", {"windows": [], "marginal_scores": []}
            )
            record["windows"].append(window.name)
            record["marginal_scores"].append(float(marginal))
