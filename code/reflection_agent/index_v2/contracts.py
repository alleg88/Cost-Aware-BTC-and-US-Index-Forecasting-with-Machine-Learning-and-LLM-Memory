"""Strict structured output for pointwise reversal scores."""
from __future__ import annotations

from typing import Annotated, Literal, Sequence

from pydantic import Field, field_validator, model_validator

from reflection_agent.index_v1.contracts import StrictModel


SCHEMA_VERSION = "1.0"
ReferenceIndex = Annotated[int, Field(strict=True, ge=0, le=63)]


class ReversalScore(StrictModel):
    opportunity_index: Annotated[int, Field(strict=True, ge=0, le=9)]
    reversal_score: Annotated[int, Field(strict=True, ge=0, le=1000)]
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


class ReversalScoreBatch(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    decisions: Annotated[list[ReversalScore], Field(min_length=1, max_length=10)]

    @model_validator(mode="after")
    def opportunity_indices_are_unique(self) -> "ReversalScoreBatch":
        indices = [item.opportunity_index for item in self.decisions]
        if len(indices) != len(set(indices)):
            raise ValueError("duplicate opportunity index")
        return self


def _references_valid(proposed: Sequence[int], allowed: Sequence[int]) -> bool:
    return set(proposed).issubset(set(allowed))


def validate_score_batch(
    value: ReversalScoreBatch,
    *,
    allowed_opportunity_indices: Sequence[int],
    allowed_evidence_indices: Sequence[int],
    allowed_memory_indices: Sequence[int],
    uncertainty_priors: Sequence[int],
) -> tuple[int, ...]:
    """Return scores in supplied order after exact ID/reference validation."""
    expected = tuple(int(item) for item in allowed_opportunity_indices)
    by_index = {item.opportunity_index: item for item in value.decisions}
    if set(by_index) != set(expected) or len(by_index) != len(expected):
        raise ValueError("score output does not cover the exact opportunity indices")
    priors = tuple(uncertainty_priors)
    if len(priors) != len(expected) or any(
        not isinstance(item, int) or isinstance(item, bool) or not 0 <= item <= 1000
        for item in priors
    ):
        raise ValueError("uncertainty priors must be aligned integers in [0,1000]")
    for item in value.decisions:
        if not _references_valid(item.evidence_indices, allowed_evidence_indices):
            raise ValueError("score output contains an invalid evidence reference")
        if not _references_valid(item.memory_indices, allowed_memory_indices):
            raise ValueError("score output contains an invalid memory reference")
    scores = tuple(int(by_index[index].reversal_score) for index in expected)
    if any(abs(score - prior) > 250 for score, prior in zip(scores, priors, strict=True)):
        raise ValueError("score output exceeds the 250-point residual boundary")
    return scores


__all__ = [
    "ReversalScore",
    "ReversalScoreBatch",
    "SCHEMA_VERSION",
    "validate_score_batch",
]
