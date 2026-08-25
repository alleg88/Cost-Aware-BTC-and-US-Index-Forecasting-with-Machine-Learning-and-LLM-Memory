"""Strict Cloud transport contract for the v4 policy router."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


SCHEMA_VERSION = "4.0"
IndexValue = Annotated[int, Field(ge=0, le=63)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class RouterChoice(StrictModel):
    """The LLM may select only one host-owned policy index."""

    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    choice_index: Annotated[int, Field(ge=0, le=8)]
    evidence_indices: Annotated[list[IndexValue], Field(max_length=3)] = Field(
        default_factory=list
    )
    memory_indices: Annotated[list[IndexValue], Field(max_length=12)] = Field(
        default_factory=list
    )

    @field_validator("evidence_indices", "memory_indices")
    @classmethod
    def indices_are_unique(cls, value: list[int]) -> list[int]:
        if len(value) != len(set(value)):
            raise ValueError("duplicate index")
        return value


__all__ = ["RouterChoice", "SCHEMA_VERSION", "StrictModel"]
