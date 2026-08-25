"""Deterministic equality-only compiler for bounded re-entry allow rules."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pandas as pd
from pydantic import Field, model_validator

from reflection_agent.v2.contracts import AllowRule, StrictModel


class ActiveAllowRule(StrictModel):
    rule_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_fold_id: Annotated[int, Field(ge=0)]
    activates_at_utc: datetime
    deactivates_at_utc: datetime | None = None
    rule: AllowRule

    @model_validator(mode="after")
    def activation_interval_is_valid(self) -> "ActiveAllowRule":
        if self.activates_at_utc.tzinfo is None:
            raise ValueError("rule activation must be timezone-aware")
        if self.deactivates_at_utc is not None:
            if self.deactivates_at_utc.tzinfo is None:
                raise ValueError("rule deactivation must be timezone-aware")
            if self.deactivates_at_utc <= self.activates_at_utc:
                raise ValueError("rule deactivation must follow activation")
        return self


def compile_allow_mask(frame: pd.DataFrame, rule: AllowRule) -> pd.Series:
    """Compile one conjunction without eval, code generation, OR, or thresholds."""

    if "route" not in frame:
        raise ValueError("opportunity frame is missing route")
    mask = frame["route"].eq("REENTRY")
    for predicate in rule.predicates:
        if predicate.field not in frame:
            raise ValueError(f"missing condition column: {predicate.field}")
        mask &= frame[predicate.field].eq(predicate.value)
    return mask.astype(bool)


def apply_policy(
    opportunities: pd.DataFrame,
    active_rules: list[ActiveAllowRule] | tuple[ActiveAllowRule, ...],
    *,
    static_add_all: bool = False,
) -> pd.DataFrame:
    """Select immutable Union base plus causally active matching re-entries."""

    required = {"opportunity_id", "route", "decision_time"}
    missing = sorted(required.difference(opportunities.columns))
    if missing:
        raise ValueError(f"opportunity frame lacks columns: {missing}")
    if static_add_all and active_rules:
        raise ValueError("static add-all cannot be combined with learned rules")

    output = opportunities.copy().reset_index(drop=True)
    if output["opportunity_id"].duplicated().any():
        raise ValueError("opportunity IDs must be unique")
    output["decision_time"] = pd.to_datetime(output["decision_time"], utc=True)
    if not output["route"].isin({"UNION_BASE", "REENTRY"}).all():
        raise ValueError("policy received an unknown route")

    output["selected"] = output["route"].eq("UNION_BASE")
    selected_by: list[list[str]] = [[] for _ in range(len(output))]
    if static_add_all:
        add_mask = output["route"].eq("REENTRY")
        output.loc[add_mask, "selected"] = True
        for position in output.index[add_mask]:
            selected_by[int(position)].append("STATIC_ADD_ALL")
    else:
        seen_rule_ids: set[str] = set()
        concurrent = pd.Series(0, index=output.index, dtype=int)
        for active in active_rules:
            if active.rule_id in seen_rule_ids:
                raise ValueError(f"duplicate active rule ID: {active.rule_id}")
            seen_rule_ids.add(active.rule_id)
            temporal = output["decision_time"].gt(active.activates_at_utc)
            if active.deactivates_at_utc is not None:
                temporal &= output["decision_time"].lt(active.deactivates_at_utc)
            concurrent += temporal.astype(int)
            if concurrent.gt(3).any():
                raise ValueError("at most three allow rules can be active concurrently")
            mask = compile_allow_mask(output, active.rule) & temporal
            output.loc[mask, "selected"] = True
            for position in output.index[mask]:
                selected_by[int(position)].append(active.rule_id)

    output["selected_rule_ids"] = [",".join(rule_ids) for rule_ids in selected_by]
    if not output.loc[output["route"].eq("UNION_BASE"), "selected"].all():
        raise AssertionError("frozen Union base trade was changed")
    if output.loc[output["route"].eq("UNION_BASE"), "selected_rule_ids"].ne("").any():
        raise AssertionError("an allow rule touched a frozen Union base trade")
    return output


__all__ = ["ActiveAllowRule", "apply_policy", "compile_allow_mask"]
