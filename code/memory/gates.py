"""Rule-engine application of accepted reflection lessons."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from memory.schema import Lesson, normalize_condition


def condition_mask(features: pd.DataFrame, condition: str) -> pd.Series:
    """Evaluate one validated lesson condition against bar-level features."""
    mask = features.eval(normalize_condition(condition), engine="python")
    if not isinstance(mask, pd.Series):
        raise ValueError("condition must evaluate to a boolean Series")
    return mask.astype(bool)


@dataclass(frozen=True)
class LessonGateSet:
    """Accepted lessons compiled into per-bar gates."""

    lessons: tuple[Lesson, ...]

    def apply_model_gates(
        self,
        contributions: pd.DataFrame,
        features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return model contributions after conditional up/down weighting.

        `contributions` is expected to contain one column per base model.  The
        meta-learner remains frozen; gates only multiply model-level inputs.
        """
        gated = contributions.copy()
        for lesson in self.lessons:
            if lesson.action not in {"downweight", "upweight"}:
                continue
            if lesson.target not in gated.columns:
                raise ValueError(f"lesson target not in contributions: {lesson.target}")
            mask = condition_mask(features, lesson.condition)
            gated.loc[mask, lesson.target] = gated.loc[mask, lesson.target] * lesson.factor
        return gated

    def force_flat_mask(self, features: pd.DataFrame) -> pd.Series:
        """Bars where at least one active lesson forces a flat signal."""
        force = pd.Series(False, index=features.index)
        for lesson in self.lessons:
            if lesson.action == "force_flat":
                force = force | condition_mask(features, lesson.condition)
        return force
