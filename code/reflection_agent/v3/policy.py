"""Deterministic equality-only compiler and overlap-safe v3 scheduler."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import pandas as pd
from pydantic import Field, model_validator

from reflection_agent.v3.contracts import AllowRule, StrictModel


class ActiveAllowRule(StrictModel):
    rule_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_stage: Literal["development", "h1", "forward"]
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
    """Compile a bounded conjunction without eval, OR, or raw thresholds."""

    if "route" not in frame:
        raise ValueError("opportunity frame is missing route")
    mask = frame["route"].eq("COVERAGE_CANDIDATE")
    for predicate in rule.predicates:
        if predicate.field not in frame:
            raise ValueError(f"missing condition column: {predicate.field}")
        mask &= frame[predicate.field].eq(predicate.value)
    return mask.astype(bool)


def _temporal_mask(frame: pd.DataFrame, active: ActiveAllowRule) -> pd.Series:
    mask = frame["decision_time"].gt(active.activates_at_utc)
    if active.deactivates_at_utc is not None:
        mask &= frame["decision_time"].lt(active.deactivates_at_utc)
    return mask


def _enforce_side_capacity(
    frame: pd.DataFrame, active_rules: list[ActiveAllowRule]
) -> None:
    seen: set[str] = set()
    for active in active_rules:
        if active.rule_id in seen:
            raise ValueError(f"duplicate active rule ID: {active.rule_id}")
        seen.add(active.rule_id)
    for side in ("LONG", "SHORT"):
        rules = [active for active in active_rules if active.rule.side == side]
        if not rules:
            continue
        concurrent = pd.Series(0, index=frame.index, dtype=int)
        for active in rules:
            concurrent += _temporal_mask(frame, active).astype(int)
        if concurrent.gt(3).any():
            raise ValueError(f"at most three active {side} rules are allowed")


def _overlaps(
    entry_time: pd.Timestamp,
    outcome_time: pd.Timestamp,
    interval: tuple[pd.Timestamp, pd.Timestamp],
) -> bool:
    other_entry, other_outcome = interval
    return bool(entry_time <= other_outcome and outcome_time >= other_entry)


def _schedule(output: pd.DataFrame) -> pd.DataFrame:
    union_mask = output["route"].eq("UNION_BASE")
    output["selected"] = union_mask
    output["skip_reason"] = ""
    output.loc[~union_mask & ~output["policy_eligible"], "skip_reason"] = "NO_ALLOW_RULE"

    union_intervals = [
        (pd.Timestamp(row.entry_time), pd.Timestamp(row.outcome_available_time))
        for row in output.loc[union_mask].itertuples(index=False)
    ]
    admitted_intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    candidates = output.loc[
        output["route"].eq("COVERAGE_CANDIDATE") & output["policy_eligible"]
    ].sort_values(
        ["entry_time", "decision_time", "opportunity_id"], kind="stable"
    )
    for row in candidates.itertuples():
        interval = (
            pd.Timestamp(row.entry_time),
            pd.Timestamp(row.outcome_available_time),
        )
        if any(_overlaps(*interval, other) for other in union_intervals):
            output.at[row.Index, "skip_reason"] = "OVERLAP_UNION"
            continue
        if any(_overlaps(*interval, other) for other in admitted_intervals):
            output.at[row.Index, "skip_reason"] = "OVERLAP_CANDIDATE"
            continue
        output.at[row.Index, "selected"] = True
        admitted_intervals.append(interval)
    return output


def apply_policy(
    opportunities: pd.DataFrame,
    active_rules: list[ActiveAllowRule] | tuple[ActiveAllowRule, ...],
    *,
    static_variant: Literal[
        "static_high_extra", "static_all_extra", "union_baseline"
    ]
    | None = None,
) -> pd.DataFrame:
    """Select immutable Union plus causally active, non-overlapping candidates."""

    required = {
        "opportunity_id",
        "route",
        "side",
        "decision_time",
        "entry_time",
        "outcome_available_time",
        "confidence_tier",
    }
    missing = sorted(required.difference(opportunities.columns))
    if missing:
        raise ValueError(f"opportunity frame lacks columns: {missing}")
    rules = list(active_rules)
    if static_variant is not None and rules:
        raise ValueError("static variants cannot be combined with learned rules")

    output = opportunities.copy().reset_index(drop=True)
    for column in ("decision_time", "entry_time", "outcome_available_time"):
        output[column] = pd.to_datetime(output[column], utc=True)
    if output["opportunity_id"].duplicated().any():
        raise ValueError("opportunity IDs must be unique")
    if not output["route"].isin({"UNION_BASE", "COVERAGE_CANDIDATE"}).all():
        raise ValueError("policy received an unknown route")
    if not output["side"].isin({"LONG", "SHORT"}).all():
        raise ValueError("policy received an unknown side")

    output["policy_eligible"] = output["route"].eq("UNION_BASE")
    selected_by: list[list[str]] = [[] for _ in range(len(output))]
    if static_variant is not None:
        if static_variant == "static_high_extra":
            candidate_mask = output["route"].eq("COVERAGE_CANDIDATE") & output[
                "confidence_tier"
            ].eq("HIGH_EXTRA")
        elif static_variant == "static_all_extra":
            candidate_mask = output["route"].eq("COVERAGE_CANDIDATE")
        else:
            candidate_mask = pd.Series(False, index=output.index)
        output.loc[candidate_mask, "policy_eligible"] = True
        for position in output.index[candidate_mask]:
            selected_by[int(position)].append(static_variant.upper())
    else:
        _enforce_side_capacity(output, rules)
        for active in rules:
            mask = compile_allow_mask(output, active.rule) & _temporal_mask(output, active)
            output.loc[mask, "policy_eligible"] = True
            for position in output.index[mask]:
                selected_by[int(position)].append(active.rule_id)

    output["selected_rule_ids"] = [",".join(values) for values in selected_by]
    output = _schedule(output)
    output["action"] = "SKIP"
    output.loc[output["selected"] & output["side"].eq("LONG"), "action"] = "OPEN_LONG"
    output.loc[output["selected"] & output["side"].eq("SHORT"), "action"] = "OPEN_SHORT"
    if not output.loc[output["route"].eq("UNION_BASE"), "selected"].all():
        raise AssertionError("frozen Union base trade was changed")
    if output.loc[output["route"].eq("UNION_BASE"), "selected_rule_ids"].ne("").any():
        raise AssertionError("an allow rule touched a frozen Union base trade")
    return output


__all__ = ["ActiveAllowRule", "apply_policy", "compile_allow_mask"]

