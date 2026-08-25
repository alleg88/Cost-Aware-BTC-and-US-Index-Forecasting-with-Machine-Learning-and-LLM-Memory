from datetime import UTC, datetime

import pandas as pd
import pytest

from experiments.build_reflection_cache import CONFIG_PATH, DEFAULT_OUTPUT
from reflection_agent.config import load_config
from reflection_agent.contracts import Candidate, ExpectedEffect, PolicyEdit
from reflection_agent.execution import (
    LSTM_RULE,
    evaluate_candidate_historical,
    evaluate_candidate_shadow,
    simulate_compiled_policy,
    slice_outcome,
)
from reflection_agent.evaluator import StrategyOutcome


def test_historical_candidate_uses_only_prior_interval_and_lstm_benchmark():
    candidate = Candidate(
        candidate_id="c-historical",
        hypothesis="Require a fixed confidence threshold before the source week.",
        edits=[PolicyEdit(edit_id="e1", action="set_confidence_threshold", value=0.75)],
        mechanism="The compiler gates low-confidence consensus signals deterministically.",
        expected_effect=ExpectedEffect(net_return="increase", turnover="decrease"),
        falsifiers=["delta_net_return_lte_0"],
        confidence=0.5,
    )
    config = load_config(CONFIG_PATH)
    baseline = simulate_compiled_policy(
        rules=[LSTM_RULE], tp_bps=200, sl_bps=100, cache_root=DEFAULT_OUTPUT
    )[0]
    end = datetime(2025, 10, 1, tzinfo=UTC)
    record = evaluate_candidate_historical(
        candidate,
        config=config,
        start=config.development_start_utc,
        end=end,
        baseline_full=baseline,
    )
    assert record.cutoff_utc < end
    assert record.cutoff_utc < config.sealed_start_utc
    assert record.baseline.trades > 0
    assert record.decision in {"historical_keep", "historical_prune"}


def test_low_support_shadow_continues_then_expires_at_four_weeks():
    candidate = Candidate(
        candidate_id="c-shadow",
        hypothesis="Use a strict confidence gate during unseen shadow weeks.",
        edits=[PolicyEdit(edit_id="e1", action="set_confidence_threshold", value=0.80)],
        mechanism="The strict gate is expected to create a deliberately sparse shadow.",
        expected_effect=ExpectedEffect(net_return="increase", turnover="decrease"),
        falsifiers=["insufficient unseen trades"],
        confidence=0.5,
    )
    config = load_config(CONFIG_PATH)
    baseline = simulate_compiled_policy(
        rules=[LSTM_RULE], tp_bps=200, sl_bps=100, cache_root=DEFAULT_OUTPUT
    )[0]
    start = datetime(2025, 7, 7, tzinfo=UTC)
    two_week = evaluate_candidate_shadow(
        candidate, config=config, start=start, end=datetime(2025, 7, 21, tzinfo=UTC),
        window_ids=["2025-W28", "2025-W29"], baseline_full=baseline,
    )
    four_week = evaluate_candidate_shadow(
        candidate, config=config, start=start, end=datetime(2025, 8, 4, tzinfo=UTC),
        window_ids=["2025-W28", "2025-W29", "2025-W30", "2025-W31"], baseline_full=baseline,
    )
    assert two_week.decision == "shadow_continue"
    assert four_week.decision == "expire"


def test_slice_outcome_fails_closed_for_trade_crossing_exclusive_end():
    index = pd.date_range("2025-07-06 23:45", periods=2, freq="15min", tz="UTC")
    outcome = StrategyOutcome(
        returns=pd.Series([0.01, 0.02], index=index),
        trades=pd.DataFrame([{
            "entry_time": index[0], "exit_time": index[1], "side": 1,
            "entry_price": 100.0, "exit_price": 101.0, "bars_held": 2,
            "exit_reason": "timeout", "gross_return": 0.01, "net_return": 0.009,
        }]),
        turnover=2.0,
    )
    with pytest.raises(RuntimeError, match="exits at or after"):
        slice_outcome(
            outcome,
            start=datetime(2025, 7, 1, tzinfo=UTC),
            end=datetime(2025, 7, 7, tzinfo=UTC),
        )
