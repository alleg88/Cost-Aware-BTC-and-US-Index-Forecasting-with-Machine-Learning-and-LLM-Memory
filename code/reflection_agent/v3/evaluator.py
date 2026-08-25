"""Deterministic future-shadow evaluator for bounded v3 policy edits."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from evaluation.economics import economics_summary
from reflection_agent.v3.contracts import AllowRule, StrictModel
from reflection_agent.v3.policy import ActiveAllowRule, apply_policy, compile_allow_mask


Q2_START = pd.Timestamp("2026-04-01", tz="UTC")
TOLERANCE = 1e-12


class ShadowCandidate(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    decision: Literal["ADD_ALLOW_RULE", "REMOVE_ALLOW_RULE"]
    source_episode_id: Annotated[str, Field(min_length=1, max_length=128)]
    source_stage: Literal["development", "h1", "forward"]
    source_fold_id: Annotated[int, Field(ge=0)]
    source_episode_cutoff_utc: datetime
    eligible_after_utc: datetime
    proposed_rule: AllowRule | None
    target_rule_id: Annotated[str | None, Field(min_length=1, max_length=128)]

    @model_validator(mode="after")
    def candidate_shape_and_time_are_valid(self) -> "ShadowCandidate":
        if (
            self.source_episode_cutoff_utc.tzinfo is None
            or self.eligible_after_utc.tzinfo is None
        ):
            raise ValueError("candidate timestamps must be timezone-aware")
        if self.eligible_after_utc <= self.source_episode_cutoff_utc:
            raise ValueError("candidate eligibility must be strictly later")
        if self.decision == "ADD_ALLOW_RULE":
            if self.proposed_rule is None or self.target_rule_id is not None:
                raise ValueError("ADD_ALLOW_RULE requires only proposed_rule")
        elif self.proposed_rule is not None or self.target_rule_id is None:
            raise ValueError("REMOVE_ALLOW_RULE requires only target_rule_id")
        return self


class EvaluationMetrics(StrictModel):
    trades: int
    long_trades: int
    short_trades: int
    gross_return: float
    cost_return: float
    net_return: float
    long_net_return: float
    short_net_return: float
    sortino: float
    sharpe: float
    max_drawdown: float


class GateDecision(StrictModel):
    candidate_id: str
    decision: Literal["PROMOTE", "REJECT", "INCONCLUSIVE"]
    source_stage: Literal["development", "h1", "forward"]
    source_fold_id: int
    eligible_after_utc: datetime
    shadow_cutoff_utc: datetime | None
    shadow_opportunities: int
    total_coverage_candidates: int
    matching_candidates: int
    matching_subblocks: int
    triggered_trades: int
    union_metrics: EvaluationMetrics
    control_metrics: EvaluationMetrics
    candidate_metrics: EvaluationMetrics
    delta_net_return: float
    delta_long_net_return: float
    delta_short_net_return: float
    extra_trade_concentration: float
    gate_results: dict[str, bool]
    failure_codes: list[str]
    evaluated_opportunity_ids: list[str]
    evaluated_decision_times: list[datetime]


def _validate_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "opportunity_id",
        "stage",
        "fold_id",
        "route",
        "side",
        "decision_time",
        "entry_time",
        "outcome_available_time",
        "gross_return",
        "net_return",
        "round_trip_cost",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"shadow opportunity frame lacks columns: {missing}")
    output = frame.copy()
    for column in ("decision_time", "entry_time", "outcome_available_time"):
        output[column] = pd.to_datetime(output[column], utc=True)
    if output["opportunity_id"].duplicated().any():
        raise ValueError("shadow opportunity IDs must be unique")
    if not output["decision_time"].lt(output["outcome_available_time"]).all():
        raise ValueError("shadow contains an early outcome")
    if output["outcome_available_time"].ge(Q2_START).any():
        raise ValueError("shadow evaluation reached the sealed Q2 interval")
    observed_cost = output["gross_return"].astype(float) - output[
        "net_return"
    ].astype(float)
    if not np.allclose(
        observed_cost,
        output["round_trip_cost"].astype(float),
        rtol=0.0,
        atol=TOLERANCE,
    ):
        raise ValueError("shadow gross/net/cost columns do not reconcile")
    return output


def _candidate_rule(
    candidate: ShadowCandidate, active_rules: list[ActiveAllowRule]
) -> AllowRule:
    if candidate.decision == "ADD_ALLOW_RULE":
        assert candidate.proposed_rule is not None
        return candidate.proposed_rule
    for active in active_rules:
        if active.rule_id == candidate.target_rule_id:
            return active.rule
    raise ValueError(f"unknown removal target: {candidate.target_rule_id}")


def _select_shadow_pool(
    candidate: ShadowCandidate,
    frame: pd.DataFrame,
    rule: AllowRule,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scope = frame.loc[
        frame["stage"].eq(candidate.source_stage)
        & frame["decision_time"].ge(candidate.eligible_after_utc)
    ].copy()
    if candidate.source_stage == "development":
        scope = scope.loc[scope["fold_id"].eq(candidate.source_fold_id)]
    coverage = scope.loc[scope["route"].eq("COVERAGE_CANDIDATE")].sort_values(
        ["outcome_available_time", "decision_time", "opportunity_id"], kind="stable"
    )
    positions: list[int] = []
    matching = 0
    for index in coverage.index:
        positions.append(int(index))
        if bool(compile_allow_mask(coverage.loc[[index]], rule).iloc[0]):
            matching += 1
        if matching >= 12 or len(positions) >= 40:
            break
    selected_coverage = coverage.loc[positions].copy() if positions else coverage.head(0)
    if selected_coverage.empty:
        return scope.head(0), selected_coverage
    selected_coverage = selected_coverage.reset_index(drop=True)
    count = len(selected_coverage)
    selected_coverage["shadow_subblock"] = np.minimum(
        (np.arange(count) * 2 // max(count, 1)).astype(int), 1
    )
    cutoff = selected_coverage["outcome_available_time"].max()
    selected_ids = set(selected_coverage["opportunity_id"])
    pool = scope.loc[
        scope["opportunity_id"].isin(selected_ids)
        | (
            scope["route"].eq("UNION_BASE")
            & scope["outcome_available_time"].le(cutoff)
        )
    ].copy()
    subblocks = selected_coverage.set_index("opportunity_id")["shadow_subblock"]
    pool["shadow_subblock"] = pool["opportunity_id"].map(subblocks)
    return pool.reset_index(drop=True), selected_coverage


def _returns_series(selected: pd.DataFrame, pool: pd.DataFrame) -> pd.Series:
    if pool.empty:
        return pd.Series(dtype=float)
    start = pool["decision_time"].min().floor("15min")
    end = pool["outcome_available_time"].max().ceil("15min")
    index = pd.date_range(start, end, freq="15min")
    returns = pd.Series(0.0, index=index, name="net_return")
    booking_time = selected["entry_time"].dt.floor("15min")
    booked = selected.assign(_booking_time=booking_time).groupby(
        "_booking_time", sort=False
    )["net_return"].sum()
    returns.loc[booked.index] = booked.to_numpy(float)
    return returns


def _metrics(policy_frame: pd.DataFrame, pool: pd.DataFrame) -> EvaluationMetrics:
    selected = policy_frame.loc[policy_frame["selected"]].copy()
    if pool.empty:
        return EvaluationMetrics(
            trades=0,
            long_trades=0,
            short_trades=0,
            gross_return=0.0,
            cost_return=0.0,
            net_return=0.0,
            long_net_return=0.0,
            short_net_return=0.0,
            sortino=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
        )
    economics = economics_summary(_returns_series(selected, pool))
    side = selected["side"]
    net = selected["net_return"].astype(float)
    gross = selected["gross_return"].astype(float)
    return EvaluationMetrics(
        trades=len(selected),
        long_trades=int(side.eq("LONG").sum()),
        short_trades=int(side.eq("SHORT").sum()),
        gross_return=float(gross.sum()),
        cost_return=float((gross - net).sum()),
        net_return=float(net.sum()),
        long_net_return=float(net.loc[side.eq("LONG")].sum()),
        short_net_return=float(net.loc[side.eq("SHORT")].sum()),
        sortino=float(economics["sortino"]),
        sharpe=float(economics["sharpe"]),
        max_drawdown=float(economics["max_drawdown"]),
    )


def _changed_trade_support(
    pool: pd.DataFrame, control: pd.DataFrame, edited: pd.DataFrame
) -> tuple[int, int, float]:
    control_selected = control.set_index("opportunity_id")["selected"].astype(bool)
    edited_selected = edited.set_index("opportunity_id")["selected"].astype(bool)
    changed_ids = control_selected.index[control_selected.ne(edited_selected)]
    changed = pool.set_index("opportunity_id").loc[changed_ids]
    changed = changed.loc[changed["route"].eq("COVERAGE_CANDIDATE")]
    if changed.empty:
        return 0, 0, 1.0
    counts = changed["shadow_subblock"].dropna().astype(int).value_counts()
    concentration = float(counts.max() / counts.sum()) if len(counts) else 1.0
    return len(changed), len(counts), concentration


def evaluate_shadow(
    candidate: ShadowCandidate,
    opportunities: pd.DataFrame,
    *,
    active_rules: list[ActiveAllowRule] | tuple[ActiveAllowRule, ...],
) -> GateDecision:
    """Evaluate one edit on 12 matching or at most 40 later candidates."""

    frame = _validate_opportunities(opportunities)
    rules = list(active_rules)
    rule = _candidate_rule(candidate, rules)
    pool, coverage = _select_shadow_pool(candidate, frame, rule)
    matching_mask = compile_allow_mask(coverage, rule) if len(coverage) else pd.Series(dtype=bool)
    matching_count = int(matching_mask.sum())
    matching_subblocks = int(
        coverage.loc[matching_mask, "shadow_subblock"].nunique()
    ) if len(coverage) else 0

    union_policy = apply_policy(pool, [])
    control_policy = apply_policy(pool, rules)
    rule_collision = False
    if candidate.decision == "ADD_ALLOW_RULE":
        assert candidate.proposed_rule is not None
        rule_collision = any(active.rule == candidate.proposed_rule for active in rules)
        candidate_rules = list(rules)
        if not rule_collision:
            candidate_rules.append(
                ActiveAllowRule(
                    rule_id=f"shadow_{candidate.candidate_id}",
                    source_candidate_id=candidate.candidate_id,
                    source_stage=candidate.source_stage,
                    source_fold_id=candidate.source_fold_id,
                    activates_at_utc=candidate.source_episode_cutoff_utc,
                    rule=candidate.proposed_rule,
                )
            )
        try:
            candidate_policy = apply_policy(pool, candidate_rules)
        except ValueError as exc:
            if "three active" not in str(exc):
                raise
            rule_collision = True
            candidate_policy = control_policy.copy()
    else:
        candidate_policy = apply_policy(
            pool,
            [active for active in rules if active.rule_id != candidate.target_rule_id],
        )

    union_metrics = _metrics(union_policy, pool)
    control_metrics = _metrics(control_policy, pool)
    candidate_metrics = _metrics(candidate_policy, pool)
    delta_net = candidate_metrics.net_return - control_metrics.net_return
    delta_long = candidate_metrics.long_net_return - control_metrics.long_net_return
    delta_short = candidate_metrics.short_net_return - control_metrics.short_net_return
    triggered, changed_subblocks, concentration = _changed_trade_support(
        pool, control_policy, candidate_policy
    )
    target_side = rule.side
    candidate_side_net = (
        candidate_metrics.long_net_return
        if target_side == "LONG"
        else candidate_metrics.short_net_return
    )
    union_side_net = (
        union_metrics.long_net_return
        if target_side == "LONG"
        else union_metrics.short_net_return
    )
    base_ids = set(pool.loc[pool["route"].eq("UNION_BASE"), "opportunity_id"])
    base_immutable = base_ids.issubset(
        set(candidate_policy.loc[candidate_policy["selected"], "opportunity_id"])
    )
    exact_reconciliation = bool(
        np.isclose(
            candidate_metrics.gross_return - candidate_metrics.cost_return,
            candidate_metrics.net_return,
            rtol=0.0,
            atol=TOLERANCE,
        )
        and np.isclose(
            union_metrics.gross_return - union_metrics.cost_return,
            union_metrics.net_return,
            rtol=0.0,
            atol=TOLERANCE,
        )
    )
    trade_shape = (
        candidate_metrics.trades > control_metrics.trades
        if candidate.decision == "ADD_ALLOW_RULE"
        else candidate_metrics.trades < control_metrics.trades
    )
    gates = {
        "matching_minimum": matching_count >= 12,
        "two_subblock_support": matching_subblocks >= 2,
        "edit_changes_trade_count": triggered >= 1,
        "total_net_noninferior": candidate_metrics.net_return
        >= union_metrics.net_return - 0.005 - TOLERANCE,
        "target_side_net_noninferior": candidate_side_net
        >= union_side_net - 0.0025 - TOLERANCE,
        "sortino_noninferior": candidate_metrics.sortino
        >= union_metrics.sortino - 0.10 - TOLERANCE,
        "drawdown_noninferior": candidate_metrics.max_drawdown
        <= union_metrics.max_drawdown + 0.01 + TOLERANCE,
        "changed_trades_span_two_subblocks": changed_subblocks >= 2,
        "extra_trade_concentration": concentration <= 0.60 + TOLERANCE,
        "base_immutable": base_immutable,
        "trade_shape": trade_shape,
        "rule_not_colliding": not rule_collision,
        "removal_strictly_improves": candidate.decision != "REMOVE_ALLOW_RULE"
        or candidate_metrics.net_return > control_metrics.net_return + TOLERANCE,
        "exact_reconciliation": exact_reconciliation,
    }
    insufficient = not gates["matching_minimum"] or not gates["two_subblock_support"]
    if insufficient:
        decision = "INCONCLUSIVE"
    elif all(gates.values()):
        decision = "PROMOTE"
    else:
        decision = "REJECT"
    failures = [name.upper() for name, passed in gates.items() if not passed]
    return GateDecision(
        candidate_id=candidate.candidate_id,
        decision=decision,
        source_stage=candidate.source_stage,
        source_fold_id=candidate.source_fold_id,
        eligible_after_utc=candidate.eligible_after_utc,
        shadow_cutoff_utc=(
            pool["outcome_available_time"].max().to_pydatetime() if len(pool) else None
        ),
        shadow_opportunities=len(pool),
        total_coverage_candidates=len(coverage),
        matching_candidates=matching_count,
        matching_subblocks=matching_subblocks,
        triggered_trades=triggered,
        union_metrics=union_metrics,
        control_metrics=control_metrics,
        candidate_metrics=candidate_metrics,
        delta_net_return=delta_net,
        delta_long_net_return=delta_long,
        delta_short_net_return=delta_short,
        extra_trade_concentration=concentration,
        gate_results=gates,
        failure_codes=failures,
        evaluated_opportunity_ids=pool["opportunity_id"].astype(str).tolist(),
        evaluated_decision_times=[value.to_pydatetime() for value in pool["decision_time"]],
    )


__all__ = [
    "EvaluationMetrics",
    "GateDecision",
    "ShadowCandidate",
    "evaluate_shadow",
]
