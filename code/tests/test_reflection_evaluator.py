from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from reflection_agent.evaluator import StrategyOutcome, block_bootstrap_ci, evaluate


def _outcome(net_values, sides):
    index = pd.date_range("2025-07-01", periods=len(net_values), freq="D", tz="UTC")
    returns = pd.Series(net_values, index=index)
    trades = pd.DataFrame({
        "entry_time": index[:len(sides)],
        "side": sides,
        "net_return": np.asarray(net_values[:len(sides)]),
    })
    return StrategyOutcome(returns=returns, trades=trades, turnover=float(len(sides)))


def test_shadow_evaluator_promotes_only_when_every_guard_passes():
    baseline = _outcome([-0.001, 0.001] * 10, ["long", "short"] * 5)
    candidate = _outcome([-0.0005, 0.002] * 10, ["long", "short"] * 5)
    record = evaluate(
        evaluation_id="e1",
        candidate_id="c1",
        window_ids=["w1", "w2"],
        cutoff_utc=datetime(2025, 7, 20, tzinfo=UTC),
        baseline_outcome=baseline,
        candidate_outcome=candidate,
        stage="shadow",
    )
    assert record.decision == "promote"
    assert record.delta_net_return > 0
    assert all(record.guard_results.values())


def test_shadow_evaluator_rejects_one_sided_or_lower_net_candidate():
    baseline = _outcome([-0.001, 0.001] * 10, ["long", "short"] * 5)
    candidate = _outcome([-0.001, 0.0005] * 10, ["long"] * 10)
    record = evaluate(
        evaluation_id="e2",
        candidate_id="c2",
        window_ids=["w1", "w2"],
        cutoff_utc=datetime(2025, 7, 20, tzinfo=UTC),
        baseline_outcome=baseline,
        candidate_outcome=candidate,
        stage="shadow",
    )
    assert record.decision == "reject"
    assert not record.guard_results["minimum_short_trades"]


def test_historical_screen_admits_sparse_economic_promise_but_shadow_does_not_promote_it():
    baseline = _outcome([-0.002, 0.0] * 10, ["long", "short"] * 5)
    sparse = _outcome([0.01] + [0.0] * 19, ["long"])
    kwargs = dict(
        evaluation_id="promising", candidate_id="c-promising", window_ids=["history"],
        cutoff_utc=datetime(2025, 7, 20, tzinfo=UTC), baseline_outcome=baseline,
        candidate_outcome=sparse,
    )
    historical = evaluate(stage="historical", **kwargs)
    shadow = evaluate(stage="shadow", **kwargs)
    assert historical.decision == "historical_keep"
    assert historical.guard_results["positive_delta_net_return"]
    assert not historical.guard_results["minimum_trades"]
    assert shadow.decision == "reject"


def test_evaluator_requires_identical_timezone_aware_timestamps():
    baseline = _outcome([0.0] * 10, ["long", "short"] * 5)
    candidate = _outcome([0.0] * 10, ["long", "short"] * 5)
    candidate = StrategyOutcome(
        returns=candidate.returns.iloc[1:], trades=candidate.trades, turnover=candidate.turnover
    )
    with pytest.raises(ValueError, match="identical timestamps"):
        evaluate(
            evaluation_id="e", candidate_id="c", window_ids=["w"], cutoff_utc=datetime.now(UTC),
            baseline_outcome=baseline, candidate_outcome=candidate, stage="historical",
        )


def test_bootstrap_is_deterministic_and_needs_two_weeks():
    assert block_bootstrap_ci([1.0]) == (None, None)
    assert block_bootstrap_ci([1.0, 2.0]) == block_bootstrap_ci([1.0, 2.0])
