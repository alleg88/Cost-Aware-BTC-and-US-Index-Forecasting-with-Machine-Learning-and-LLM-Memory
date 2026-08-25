"""Exact one-minute execution for frozen controls and compiled agent policies."""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Sequence

import pandas as pd

from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.build_reflection_cache import BAR_PATH, DEFAULT_OUTPUT, MINUTE_PATH
from reflection_agent.config import ProtocolConfig
from reflection_agent.contracts import Candidate, EvaluationRecord, MODEL_IDS, PolicyEdit, PolicyRule
from reflection_agent.evaluator import StrategyOutcome, evaluate, summarize
from reflection_agent.manifest import sha256_payload
from reflection_agent.policy import PROBABILITY_COLUMNS, CompiledPolicy, compile_policy

LSTM_RULE = PolicyRule(
    rule_id="frozen-lstm",
    edits=[
        PolicyEdit(edit_id="lstm-expert", action="select_frozen_expert", target="lstm"),
        PolicyEdit(edit_id="lstm-tau", action="set_confidence_threshold", value=0.75),
    ],
)
CONSENSUS_RULE = PolicyRule(
    rule_id="frozen-unanimity-consensus",
    edits=[PolicyEdit(edit_id="consensus-agreement", action="require_minimum_agreement", value=1.0)],
)


def load_agent_inputs(cache_root: Path = DEFAULT_OUTPUT):
    wide = pd.read_parquet(cache_root / "frozen_probability_panel.parquet").set_index("timestamp").sort_index()
    if "y_true" in wide.columns:
        raise RuntimeError("agent probability cache must not contain target labels")
    context = pd.read_parquet(cache_root / "market_context.parquet").set_index("timestamp").sort_index()
    probabilities = {
        model_id: wide[[f"{model_id}_{column}" for column in PROBABILITY_COLUMNS]].set_axis(
            PROBABILITY_COLUMNS, axis=1
        )
        for model_id in MODEL_IDS
    }
    return wide, context, probabilities


def simulate_compiled_policy(
    *,
    rules: Sequence[PolicyRule],
    tp_bps: int,
    sl_bps: int,
    max_hold: int = 1,
    fee_bps: float = 5.0,
    cache_root: Path = DEFAULT_OUTPUT,
    bars_path: Path = BAR_PATH,
    minute_path: Path = MINUTE_PATH,
) -> tuple[StrategyOutcome, CompiledPolicy]:
    wide, context, probabilities = load_agent_inputs(cache_root)
    compiled = compile_policy(probabilities, context, rules)
    start = wide.index.min()
    end_exclusive = pd.Timestamp("2026-04-01", tz="UTC")
    bars = pd.read_parquet(bars_path).sort_index().loc[start:end_exclusive]
    minute = pd.read_parquet(minute_path).sort_index().loc[start:end_exclusive]
    ledger, returns = simulate_bracket_trades_intrabar(
        bars,
        minute,
        compiled.predictions,
        tp_bps=tp_bps,
        sl_bps=sl_bps,
        max_hold=max_hold,
        fee_bps=fee_bps,
        expected_interval=pd.Timedelta(minutes=1),
    )
    returns = returns.reindex(wide.index, fill_value=0.0)
    return StrategyOutcome(returns=returns, trades=ledger, turnover=float(2 * len(ledger))), compiled


def slice_outcome(outcome: StrategyOutcome, *, start: datetime, end: datetime) -> StrategyOutcome:
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end)
    returns = outcome.returns.loc[(outcome.returns.index >= start_at) & (outcome.returns.index < end_at)]
    entry_time = pd.to_datetime(outcome.trades["entry_time"], utc=True)
    exit_time = pd.to_datetime(outcome.trades["exit_time"], utc=True)
    crossing = (entry_time >= start_at) & (entry_time < end_at) & (exit_time >= end_at)
    if crossing.any():
        raise RuntimeError("evaluation interval contains a trade that exits at or after its exclusive end")
    trades = outcome.trades.loc[
        (entry_time >= start_at) & (entry_time < end_at) & (exit_time < end_at)
    ].copy()
    return StrategyOutcome(returns=returns, trades=trades, turnover=float(2 * len(trades)))


def evaluate_candidate_historical(
    candidate: Candidate,
    *,
    config: ProtocolConfig,
    start: datetime,
    end: datetime,
    cache_root: Path = DEFAULT_OUTPUT,
    baseline_full: StrategyOutcome | None = None,
) -> EvaluationRecord:
    from reflection_agent.controls import materialize_candidate_rules

    baseline_full = baseline_full or simulate_compiled_policy(
        rules=[LSTM_RULE], tp_bps=200, sl_bps=100, cache_root=cache_root
    )[0]
    candidate_full = simulate_compiled_policy(
        rules=materialize_candidate_rules(candidate),
        tp_bps=150,
        sl_bps=100,
        cache_root=cache_root,
    )[0]
    baseline = slice_outcome(baseline_full, start=start, end=end)
    candidate_outcome = slice_outcome(candidate_full, start=start, end=end)
    weeks = sorted({
        f"{timestamp.isocalendar().year}-W{timestamp.isocalendar().week:02d}"
        for timestamp in baseline.returns.index
    })
    cutoff = (pd.Timestamp(end) - pd.Timedelta(microseconds=1)).to_pydatetime()
    evaluation_id = "historical-" + sha256_payload({
        "candidate": candidate.model_dump(mode="json"),
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
    })[:20]
    return evaluate(
        evaluation_id=evaluation_id,
        candidate_id=candidate.candidate_id,
        window_ids=weeks,
        cutoff_utc=cutoff,
        baseline_outcome=baseline,
        candidate_outcome=candidate_outcome,
        stage="historical",
        min_trades=config.shadow_min_trades,
        min_trades_per_side=config.shadow_min_trades_per_side,
        sortino_margin=config.sortino_noninferiority_margin,
        max_drawdown_ratio=config.max_drawdown_ratio,
    )


def evaluate_candidate_shadow(
    candidate: Candidate,
    *,
    config: ProtocolConfig,
    start: datetime,
    end: datetime,
    window_ids: list[str],
    cache_root: Path = DEFAULT_OUTPUT,
    baseline_full: StrategyOutcome | None = None,
) -> EvaluationRecord:
    from reflection_agent.controls import materialize_candidate_rules

    if not 2 <= len(window_ids) <= 4:
        raise ValueError("shadow evaluation requires two to four unseen windows")
    baseline_full = baseline_full or simulate_compiled_policy(
        rules=[LSTM_RULE], tp_bps=200, sl_bps=100, cache_root=cache_root
    )[0]
    candidate_full = simulate_compiled_policy(
        rules=materialize_candidate_rules(candidate),
        tp_bps=150,
        sl_bps=100,
        cache_root=cache_root,
    )[0]
    baseline = slice_outcome(baseline_full, start=start, end=end)
    candidate_outcome = slice_outcome(candidate_full, start=start, end=end)
    cutoff = (pd.Timestamp(end) - pd.Timedelta(microseconds=1)).to_pydatetime()
    evaluation_id = "shadow-" + sha256_payload({
        "candidate": candidate.model_dump(mode="json"),
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
        "window_ids": window_ids,
    })[:20]
    record = evaluate(
        evaluation_id=evaluation_id,
        candidate_id=candidate.candidate_id,
        window_ids=window_ids,
        cutoff_utc=cutoff,
        baseline_outcome=baseline,
        candidate_outcome=candidate_outcome,
        stage="shadow",
        min_trades=config.shadow_min_trades,
        min_trades_per_side=config.shadow_min_trades_per_side,
        sortino_margin=config.sortino_noninferiority_margin,
        max_drawdown_ratio=config.max_drawdown_ratio,
    )
    support_guards = (
        "minimum_trades", "minimum_long_trades", "minimum_short_trades"
    )
    lacks_support = not all(record.guard_results[name] for name in support_guards)
    if lacks_support and len(window_ids) < config.shadow_max_weeks:
        return record.model_copy(update={"decision": "shadow_continue"})
    if lacks_support:
        return record.model_copy(update={"decision": "expire"})
    return record


def build_frozen_baselines(cache_root: Path = DEFAULT_OUTPUT) -> pd.DataFrame:
    from reflection_agent.controls import deterministic_router_rules

    controls = {
        "lstm": ([LSTM_RULE], 200, 100),
        "unanimity_consensus": ([CONSENSUS_RULE], 150, 100),
        "deterministic_router": (list(deterministic_router_rules()), 150, 100),
    }
    rows = []
    returns = []
    for control_id, (rules, tp_bps, sl_bps) in controls.items():
        outcome, _ = simulate_compiled_policy(
            rules=rules, tp_bps=tp_bps, sl_bps=sl_bps, cache_root=cache_root
        )
        metrics = summarize(outcome)
        rows.append({
            "control_id": control_id,
            "tp_bps": tp_bps,
            "sl_bps": sl_bps,
            "max_hold": 1,
            "fee_bps_per_side": 5.0,
            **metrics.model_dump(),
        })
        returns.append(outcome.returns.rename(control_id))
        outcome.trades.to_parquet(cache_root / f"{control_id}_trades.parquet", index=False)
    summary = pd.DataFrame(rows)
    summary.to_parquet(cache_root / "frozen_baselines.parquet", index=False)
    pd.concat(returns, axis=1).rename_axis("timestamp").reset_index().to_parquet(
        cache_root / "frozen_baseline_returns.parquet", index=False
    )
    return summary
