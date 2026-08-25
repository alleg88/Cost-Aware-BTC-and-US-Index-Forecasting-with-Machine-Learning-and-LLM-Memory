"""Deterministic future-shadow evaluation for bounded policy candidates."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from evaluation.economics import economics_summary
from reflection_agent.v2.contracts import AllowRule, StrictModel
from reflection_agent.v2.policy import ActiveAllowRule, apply_policy

Q2_START = pd.Timestamp("2026-04-01", tz="UTC")
TOLERANCE = 1e-12


class ShadowCandidate(StrictModel):
    candidate_id: Annotated[str, Field(min_length=1, max_length=128)]
    decision: Literal["ADD_ALLOW_RULE", "REMOVE_ALLOW_RULE"]
    source_episode_id: Annotated[str, Field(min_length=1, max_length=128)]
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
            raise ValueError("candidate eligibility must be strictly later than source episode")
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
    source_fold_id: int
    eligible_after_utc: datetime
    shadow_cutoff_utc: datetime | None
    shadow_opportunities: int
    shadow_reentry_opportunities: int
    shadow_long_opportunities: int
    shadow_short_opportunities: int
    triggered_trades: int
    union_metrics: EvaluationMetrics
    control_metrics: EvaluationMetrics
    candidate_metrics: EvaluationMetrics
    delta_net_return: float
    delta_long_net_return: float
    delta_short_net_return: float
    positive_return_concentration: float
    gate_results: dict[str, bool]
    failure_codes: list[str]
    evaluated_opportunity_ids: list[str]
    evaluated_decision_times: list[datetime]


def _validate_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "opportunity_id",
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
        raise ValueError("shadow contains an outcome available before its decision")
    if output["outcome_available_time"].ge(Q2_START).any():
        raise ValueError("shadow evaluation reached the sealed Q2 interval")
    observed_cost = output["gross_return"].astype(float) - output["net_return"].astype(float)
    if not np.allclose(
        observed_cost,
        output["round_trip_cost"].astype(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("shadow gross/net/cost columns do not reconcile")
    return output


def _select_shadow_pool(candidate: ShadowCandidate, frame: pd.DataFrame) -> pd.DataFrame:
    eligible = frame.loc[
        frame["fold_id"].eq(candidate.source_fold_id)
        & frame["decision_time"].ge(candidate.eligible_after_utc)
    ].sort_values(
        ["outcome_available_time", "decision_time", "opportunity_id"], kind="stable"
    )
    positions: list[int] = []
    for index in eligible.index:
        positions.append(int(index))
        current = eligible.loc[positions]
        reentries = current.loc[current["route"].eq("REENTRY")]
        if (
            len(reentries) >= 10
            and reentries["side"].eq("LONG").sum() >= 3
            and reentries["side"].eq("SHORT").sum() >= 3
        ):
            break
        if len(current) >= 60:
            break
    return eligible.loc[positions].reset_index(drop=True) if positions else eligible.head(0)


def _returns_series(selected: pd.DataFrame, pool: pd.DataFrame) -> pd.Series:
    if pool.empty:
        return pd.Series(dtype=float)
    start = pool["decision_time"].min().floor("15min")
    end = pool["outcome_available_time"].max().ceil("15min")
    index = pd.date_range(start, end, freq="15min")
    returns = pd.Series(0.0, index=index, name="net_return")
    booked = selected.groupby("entry_time", sort=False)["net_return"].sum()
    missing = booked.index.difference(returns.index)
    if len(missing):
        raise AssertionError("selected return timestamp is outside its shadow horizon")
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
    returns = _returns_series(selected, pool)
    economics = economics_summary(returns)
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


def _positive_change_concentration(
    pool: pd.DataFrame, control: pd.DataFrame, candidate: pd.DataFrame
) -> float:
    indexed = pool.set_index("opportunity_id")
    control_selected = control.set_index("opportunity_id")["selected"].astype(bool)
    candidate_selected = candidate.set_index("opportunity_id")["selected"].astype(bool)
    change = candidate_selected.astype(int) - control_selected.astype(int)
    contribution = indexed["net_return"].astype(float) * change
    positive = contribution.loc[contribution > 0.0]
    if positive.empty:
        return 1.0
    months = indexed.loc[positive.index, "entry_time"].dt.strftime("%Y-%m")
    monthly = positive.groupby(months).sum()
    return float(monthly.max() / monthly.sum())


def evaluate_shadow(
    candidate: ShadowCandidate,
    opportunities: pd.DataFrame,
    *,
    active_rules: list[ActiveAllowRule] | tuple[ActiveAllowRule, ...],
) -> GateDecision:
    """Evaluate one edit on strictly later same-fold outcomes; the LLM has no vote."""

    frame = _validate_opportunities(opportunities)
    pool = _select_shadow_pool(candidate, frame)
    reentries = pool.loc[pool["route"].eq("REENTRY")]
    long_opportunities = int(reentries["side"].eq("LONG").sum())
    short_opportunities = int(reentries["side"].eq("SHORT").sum())
    shadow_minimums = bool(
        len(reentries) >= 10
        and long_opportunities >= 3
        and short_opportunities >= 3
    )

    union_policy = apply_policy(pool, [])
    control_policy = apply_policy(pool, list(active_rules))
    rule_collision = False
    if candidate.decision == "ADD_ALLOW_RULE":
        assert candidate.proposed_rule is not None
        rule_collision = any(active.rule == candidate.proposed_rule for active in active_rules)
        candidate_rules = list(active_rules)
        if not rule_collision and len(candidate_rules) < 3:
            candidate_rules.append(
                ActiveAllowRule(
                    rule_id=f"shadow:{candidate.candidate_id}",
                    source_candidate_id=candidate.candidate_id,
                    source_fold_id=candidate.source_fold_id,
                    activates_at_utc=candidate.eligible_after_utc,
                    deactivates_at_utc=None,
                    rule=candidate.proposed_rule,
                )
            )
        else:
            rule_collision = True
        candidate_policy = apply_policy(pool, candidate_rules)
    else:
        target_ids = {active.rule_id for active in active_rules}
        if candidate.target_rule_id not in target_ids:
            raise ValueError(f"unknown removal target: {candidate.target_rule_id}")
        candidate_policy = apply_policy(
            pool,
            [active for active in active_rules if active.rule_id != candidate.target_rule_id],
        )

    union_metrics = _metrics(union_policy, pool)
    control_metrics = _metrics(control_policy, pool)
    candidate_metrics = _metrics(candidate_policy, pool)
    delta_net = candidate_metrics.net_return - control_metrics.net_return
    delta_long = candidate_metrics.long_net_return - control_metrics.long_net_return
    delta_short = candidate_metrics.short_net_return - control_metrics.short_net_return
    control_selected = control_policy.set_index("opportunity_id")["selected"].astype(bool)
    candidate_selected = candidate_policy.set_index("opportunity_id")["selected"].astype(bool)
    triggered = int(control_selected.ne(candidate_selected).sum())
    concentration = _positive_change_concentration(pool, control_policy, candidate_policy)

    base_ids = set(pool.loc[pool["route"].eq("UNION_BASE"), "opportunity_id"])
    base_immutable = base_ids.issubset(
        set(control_policy.loc[control_policy["selected"], "opportunity_id"])
    ) and base_ids.issubset(
        set(candidate_policy.loc[candidate_policy["selected"], "opportunity_id"])
    )
    trade_shape = (
        candidate_metrics.trades >= control_metrics.trades
        if candidate.decision == "ADD_ALLOW_RULE"
        else candidate_metrics.trades <= control_metrics.trades
    )
    exact_reconciliation = bool(
        np.isclose(
            candidate_metrics.net_return,
            candidate_metrics.gross_return - candidate_metrics.cost_return,
            rtol=0.0,
            atol=1e-10,
        )
        and np.isclose(
            control_metrics.net_return,
            control_metrics.gross_return - control_metrics.cost_return,
            rtol=0.0,
            atol=1e-10,
        )
    )
    drawdown_limit = (
        control_metrics.max_drawdown * 1.10
        if control_metrics.max_drawdown > TOLERANCE
        else TOLERANCE
    )
    gates = {
        "shadow_minimums": shadow_minimums,
        "triggered_minimum": triggered >= 3,
        "incremental_net_positive": delta_net > TOLERANCE,
        "incremental_long_nonnegative": delta_long >= -TOLERANCE,
        "incremental_short_nonnegative": delta_short >= -TOLERANCE,
        "candidate_total_noninferior": candidate_metrics.net_return
        >= control_metrics.net_return - TOLERANCE,
        "candidate_not_below_union": candidate_metrics.net_return
        >= union_metrics.net_return - TOLERANCE,
        "sortino_noninferior": candidate_metrics.sortino
        >= control_metrics.sortino - 0.10 - TOLERANCE,
        "drawdown_noninferior": candidate_metrics.max_drawdown
        <= drawdown_limit + TOLERANCE,
        "positive_return_concentration": concentration <= 0.60 + TOLERANCE,
        "base_immutable": base_immutable,
        "trade_shape": trade_shape,
        "rule_not_colliding": not rule_collision,
        "exact_reconciliation": exact_reconciliation,
    }
    if not shadow_minimums:
        decision = "INCONCLUSIVE"
    elif all(gates.values()):
        decision = "PROMOTE"
    else:
        decision = "REJECT"
    failures = [name.upper() for name, passed in gates.items() if not passed]
    return GateDecision(
        candidate_id=candidate.candidate_id,
        decision=decision,
        source_fold_id=candidate.source_fold_id,
        eligible_after_utc=candidate.eligible_after_utc,
        shadow_cutoff_utc=(
            pool["outcome_available_time"].max().to_pydatetime() if len(pool) else None
        ),
        shadow_opportunities=len(pool),
        shadow_reentry_opportunities=len(reentries),
        shadow_long_opportunities=long_opportunities,
        shadow_short_opportunities=short_opportunities,
        triggered_trades=triggered,
        union_metrics=union_metrics,
        control_metrics=control_metrics,
        candidate_metrics=candidate_metrics,
        delta_net_return=delta_net,
        delta_long_net_return=delta_long,
        delta_short_net_return=delta_short,
        positive_return_concentration=concentration,
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
