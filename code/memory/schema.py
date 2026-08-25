"""Lesson schema for the RQ3 reflection-memory layer.

Lessons define validated conditions and actions applied to bar-level features.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any


ALLOWED_ACTIONS = {"downweight", "upweight", "force_flat", "widen_deadzone"}
ALLOWED_IMPACTS = {"candidate", "accepted", "rejected"}

_ALLOWED_AST = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.Compare,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.And,
    ast.Or,
    ast.Not,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
)


def normalize_condition(condition: str) -> str:
    """Convert prompt-style boolean operators into Python/pandas syntax."""
    return (
        condition.replace(" AND ", " and ")
        .replace(" OR ", " or ")
        .replace(" NOT ", " not ")
    )


def condition_feature_names(condition: str) -> set[str]:
    """Return feature names referenced by a safe boolean expression."""
    expr = normalize_condition(condition)
    tree = ast.parse(expr, mode="eval")
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST):
            raise ValueError(f"unsupported condition syntax: {type(node).__name__}")
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


@dataclass(frozen=True)
class Lesson:
    """One proposed, accepted, or rejected memory lesson."""

    condition: str
    action: str
    target: str
    factor: float
    evidence: str
    confidence: float
    status: str = "candidate"
    source_window: str | None = None
    score_before: float | None = None
    score_after: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Lesson":
        required = {"condition", "action", "target", "factor", "evidence", "confidence"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"missing lesson field(s): {sorted(missing)}")
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "action": self.action,
            "target": self.target,
            "factor": self.factor,
            "evidence": self.evidence,
            "confidence": self.confidence,
            "status": self.status,
            "source_window": self.source_window,
            "score_before": self.score_before,
            "score_after": self.score_after,
            "metadata": self.metadata,
        }

    def validate(
        self,
        feature_columns: set[str] | list[str],
        model_names: set[str] | list[str] | None = None,
    ) -> "Lesson":
        """Validate schema and ensure the condition only uses known features."""
        feature_set = set(feature_columns)
        missing_features = condition_feature_names(self.condition) - feature_set
        if missing_features:
            raise ValueError(f"unknown condition feature(s): {sorted(missing_features)}")
        if self.action not in ALLOWED_ACTIONS:
            raise ValueError(f"unsupported action: {self.action}")
        if self.status not in ALLOWED_IMPACTS:
            raise ValueError(f"unsupported status: {self.status}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if float(self.factor) < 0.0:
            raise ValueError("factor must be non-negative")
        if self.action in {"downweight", "upweight"} and model_names is not None:
            if self.target not in set(model_names):
                raise ValueError(f"unknown target model: {self.target}")
        return self

    def with_decision(
        self,
        status: str,
        score_before: float,
        score_after: float,
        source_window: str | None = None,
    ) -> "Lesson":
        return Lesson(
            condition=self.condition,
            action=self.action,
            target=self.target,
            factor=self.factor,
            evidence=self.evidence,
            confidence=self.confidence,
            status=status,
            source_window=source_window or self.source_window,
            score_before=float(score_before),
            score_after=float(score_after),
            metadata=dict(self.metadata),
        )
