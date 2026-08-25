"""Deterministic economic comparison and promotion gates."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from reflection_agent.contracts import EvaluationRecord, MetricBundle

HISTORICAL_PROMISE_GUARDS = (
    "positive_delta_net_return",
    "some_historical_activity",
    "sortino_noninferiority",
    "max_drawdown",
)


@dataclass(frozen=True)
class StrategyOutcome:
    returns: pd.Series
    trades: pd.DataFrame
    turnover: float


def _trade_side_counts(trades: pd.DataFrame) -> tuple[int, int]:
    if "side" not in trades.columns:
        raise ValueError("trade ledger requires a side column")
    side = trades["side"].astype(str).str.lower()
    long_count = int(side.isin({"long", "1", "1.0"}).sum())
    short_count = int(side.isin({"short", "-1", "-1.0"}).sum())
    if long_count + short_count != len(trades):
        raise ValueError("trade side must be long or short")
    return long_count, short_count


def _monthly_gain_concentration(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 1.0
    if "entry_time" not in trades.columns or "net_return" not in trades.columns:
        raise ValueError("trade ledger requires entry_time and net_return")
    timestamps = pd.to_datetime(trades["entry_time"], utc=True)
    monthly = trades.assign(month=timestamps.dt.strftime("%Y-%m")).groupby("month")["net_return"].sum()
    positive = monthly.clip(lower=0.0)
    total = float(positive.sum())
    return float(positive.max() / total) if total > 0 else 1.0


def summarize(outcome: StrategyOutcome) -> MetricBundle:
    returns = outcome.returns.sort_index().astype(float)
    summary = economics_summary(returns)
    long_count, short_count = _trade_side_counts(outcome.trades)
    return MetricBundle(
        trades=len(outcome.trades),
        long_trades=long_count,
        short_trades=short_count,
        net_return=summary["net_return_sum"],
        sortino=summary["sortino"],
        sharpe=summary["sharpe"],
        max_drawdown=summary["max_drawdown"],
        turnover=float(outcome.turnover),
        monthly_gain_concentration=_monthly_gain_concentration(outcome.trades),
    )


def paired_weekly_deltas(baseline_returns: pd.Series, candidate_returns: pd.Series) -> list[float]:
    joined = pd.concat([baseline_returns.rename("baseline"), candidate_returns.rename("candidate")], axis=1, join="inner").dropna()
    if not joined.index.equals(baseline_returns.dropna().index) or not joined.index.equals(candidate_returns.dropna().index):
        raise ValueError("baseline and candidate must use identical timestamps")
    if joined.index.tz is None:
        raise ValueError("return index must be timezone-aware")
    weekly = joined.resample("W-SUN").sum()
    return (weekly["candidate"] - weekly["baseline"]).astype(float).tolist()


def block_bootstrap_ci(deltas: list[float], *, samples: int = 2000, seed: int = 42) -> tuple[float | None, float | None]:
    if len(deltas) < 2:
        return None, None
    values = np.asarray(deltas, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(
    *,
    evaluation_id: str,
    candidate_id: str,
    window_ids: list[str],
    cutoff_utc: datetime,
    baseline_outcome: StrategyOutcome,
    candidate_outcome: StrategyOutcome,
    stage: str,
    min_trades: int = 10,
    min_trades_per_side: int = 3,
    sortino_margin: float = 0.10,
    max_drawdown_ratio: float = 1.10,
) -> EvaluationRecord:
    baseline = summarize(baseline_outcome)
    candidate = summarize(candidate_outcome)
    deltas = paired_weekly_deltas(baseline_outcome.returns, candidate_outcome.returns)
    ci_low, ci_high = block_bootstrap_ci(deltas)
    guards = {
        "positive_delta_net_return": candidate.net_return > baseline.net_return,
        "some_historical_activity": candidate.trades > 0,
        "minimum_trades": candidate.trades >= min_trades,
        "minimum_long_trades": candidate.long_trades >= min_trades_per_side,
        "minimum_short_trades": candidate.short_trades >= min_trades_per_side,
        "sortino_noninferiority": candidate.sortino >= baseline.sortino - sortino_margin,
        "max_drawdown": candidate.max_drawdown <= baseline.max_drawdown * max_drawdown_ratio,
    }
    if stage == "historical":
        passed = all(guards[name] for name in HISTORICAL_PROMISE_GUARDS)
        decision = "historical_keep" if passed else "historical_prune"
    elif stage == "shadow":
        passed = all(guards.values())
        decision = "promote" if passed else "reject"
    else:
        raise ValueError("stage must be historical or shadow")
    return EvaluationRecord(
        evaluation_id=evaluation_id,
        candidate_id=candidate_id,
        window_ids=window_ids,
        cutoff_utc=cutoff_utc,
        baseline=baseline,
        candidate=candidate,
        delta_net_return=candidate.net_return - baseline.net_return,
        paired_weekly_deltas=deltas,
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        guard_results=guards,
        decision=decision,
    )
