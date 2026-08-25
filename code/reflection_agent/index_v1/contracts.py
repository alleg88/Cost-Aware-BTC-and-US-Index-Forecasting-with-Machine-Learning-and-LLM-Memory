"""Strict structured-output contracts for the index reflection agents."""
from __future__ import annotations

import math
from typing import Annotated, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


SCHEMA_VERSION = "1.0"
ReferenceIndex = Annotated[int, Field(ge=0, le=63)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class DirectDecision(StrictModel):
    opportunity_index: Annotated[int, Field(ge=0, le=9)]
    side: Literal["LONG", "SHORT"]
    evidence_indices: Annotated[list[ReferenceIndex], Field(max_length=3)] = Field(
        default_factory=list
    )
    memory_indices: Annotated[list[ReferenceIndex], Field(max_length=12)] = Field(
        default_factory=list
    )

    @field_validator("evidence_indices", "memory_indices")
    @classmethod
    def references_are_unique(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate reference index")
        return values


class DirectBatchDecision(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    decisions: Annotated[list[DirectDecision], Field(min_length=1, max_length=10)]

    @model_validator(mode="after")
    def opportunity_indices_are_unique(self) -> "DirectBatchDecision":
        indices = [item.opportunity_index for item in self.decisions]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate opportunity index")
        return self


class WeeklyWeightDecision(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    weights: Annotated[list[float], Field(min_length=9, max_length=9)]
    evidence_indices: Annotated[list[ReferenceIndex], Field(max_length=3)] = Field(
        default_factory=list
    )
    memory_indices: Annotated[list[ReferenceIndex], Field(max_length=12)] = Field(
        default_factory=list
    )

    @field_validator("evidence_indices", "memory_indices")
    @classmethod
    def references_are_unique(cls, values: list[int]) -> list[int]:
        if len(values) != len(set(values)):
            raise ValueError("duplicate reference index")
        return values


def _references_valid(proposed: Sequence[int], allowed: Sequence[int]) -> bool:
    return set(proposed).issubset(set(allowed))


def validate_direct_batch(
    value: DirectBatchDecision,
    *,
    allowed_opportunity_indices: Sequence[int],
    allowed_evidence_indices: Sequence[int],
    allowed_memory_indices: Sequence[int],
) -> tuple[int, ...]:
    """Return host sides in supplied order after exact ID/reference validation."""
    expected = tuple(int(item) for item in allowed_opportunity_indices)
    by_index = {item.opportunity_index: item for item in value.decisions}
    if set(by_index) != set(expected) or len(by_index) != len(expected):
        raise ValueError("direct output does not cover the exact opportunity indices")
    for item in value.decisions:
        if not _references_valid(item.evidence_indices, allowed_evidence_indices):
            raise ValueError("direct output contains an invalid evidence reference")
        if not _references_valid(item.memory_indices, allowed_memory_indices):
            raise ValueError("direct output contains an invalid memory reference")
    return tuple(1 if by_index[index].side == "LONG" else -1 for index in expected)


def validate_weekly_weights(
    value: WeeklyWeightDecision,
    *,
    allowed_evidence_indices: Sequence[int],
    allowed_memory_indices: Sequence[int],
) -> tuple[float, ...]:
    """Validate the host's exact 0.01-grid influence contract without repair."""
    weights = tuple(float(item) for item in value.weights)
    if len(weights) != 9 or not all(math.isfinite(item) for item in weights):
        raise ValueError("weights must be nine finite values")
    if any(item < 0.0 or item > 0.80 for item in weights):
        raise ValueError("weight lies outside [0.00, 0.80]")
    if any(abs(item * 100.0 - round(item * 100.0)) > 1e-9 for item in weights):
        raise ValueError("weight is not on the 0.01 grid")
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("weights must sum to 1.00")
    if sum(item >= 0.05 for item in weights) < 3:
        raise ValueError("at least three models require weight >= 0.05")
    if not _references_valid(value.evidence_indices, allowed_evidence_indices):
        raise ValueError("weekly output contains an invalid evidence reference")
    if not _references_valid(value.memory_indices, allowed_memory_indices):
        raise ValueError("weekly output contains an invalid memory reference")
    return weights


__all__ = [
    "DirectBatchDecision",
    "DirectDecision",
    "SCHEMA_VERSION",
    "StrictModel",
    "WeeklyWeightDecision",
    "validate_direct_batch",
    "validate_weekly_weights",
]
