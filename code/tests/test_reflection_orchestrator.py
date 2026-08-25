from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from reflection_agent.contracts import (
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    ExpectedEffect,
    MarketContext,
    MetricBundle,
    PolicyEdit,
    ProbabilityVector,
    RefinerBatch,
    ReflectionRecord,
)
from reflection_agent.memory import MemoryManager
from reflection_agent.news import select_balanced_events
from reflection_agent.observation import build_observation
from reflection_agent.orchestrator import WeeklyOrchestrator
from reflection_agent.store import AgentStore
from reflection_agent.transport import StructuredCallResult


def _candidate(candidate_id="c1"):
    return Candidate(
        candidate_id=candidate_id,
        hypothesis="Require stronger agreement in the supplied regime.",
        edits=[PolicyEdit(edit_id=f"{candidate_id}-e", action="require_minimum_agreement", value=0.75)],
        mechanism="Reduce entries unsupported by the frozen experts.",
        expected_effect=ExpectedEffect(net_return="increase", turnover="decrease"),
        falsifiers=["delta_net_return_lte_0"], confidence=0.5,
    )


def _metrics(net=0.01):
    return MetricBundle(
        trades=12, long_trades=6, short_trades=6, net_return=net, sortino=1.0, sharpe=0.5,
        max_drawdown=0.01, turnover=12, monthly_gain_concentration=0.5,
    )


def _evaluation(candidate_id, *, decision="historical_keep", windows=None, cutoff=None):
    base = _metrics(0.01)
    candidate = _metrics(0.02)
    return EvaluationRecord(
        evaluation_id=f"ev-{candidate_id}-{decision}", candidate_id=candidate_id,
        window_ids=windows or ["history"], cutoff_utc=cutoff or datetime(2025, 7, 1, tzinfo=UTC),
        baseline=base, candidate=candidate, delta_net_return=0.01, paired_weekly_deltas=[0.01],
        guard_results={"positive": True}, decision=decision,
    )


class FakeCaller:
    def __init__(self, values):
        self.values = list(values)

    def call(self, **kwargs):
        value = self.values.pop(0)
        return StructuredCallResult(
            status="success", value=value, raw_content="{}", request_hash="h", attempts=1,
            backend="fake", errors=(),
        )


def _observation():
    models = (
        "logreg", "decision_tree", "random_forest", "svm_linear", "xgboost_balanced",
        "catboost_balanced", "mlp", "lstm", "gru",
    )
    cutoff = datetime(2025, 7, 6, 23, 59, tzinfo=UTC)
    empty = pd.DataFrame(columns=[
        "event_id", "available_at_utc", "source_family", "publisher_category", "summary", "impact", "sentiment"
    ])
    return build_observation(
        window_id="2025-W27", cutoff_utc=cutoff, active_policy_id="p0",
        market=MarketContext(vol_regime="normal", trend_regime="flat", realized_volatility=0.1, recent_return=0.0),
        probabilities={model: ProbabilityVector(short=0.2, flat=0.6, long=0.2) for model in models},
        news=select_balanced_events(empty, cutoff_utc=cutoff),
    )


def test_orchestrator_opens_only_refined_future_shadow_and_promotes_later(tmp_path):
    candidate = _candidate()
    reflection = ReflectionRecord(
        reflection_id="r1", candidate_id="c1", evaluation_id="ev-c1-promote",
        evidence=["positive delta"], speculation=[], generalized_lesson="Agreement helped in this regime.",
        memory_recommendation="store_episode",
    )
    caller = FakeCaller([CandidateBatch(diagnosis="test", candidates=[candidate]), RefinerBatch(candidates=[candidate]), reflection])
    store = AgentStore(tmp_path / "state.sqlite")
    orchestrator = WeeklyOrchestrator(
        caller=caller, store=store, memory=MemoryManager(store, protocol_hash="p"), protocol_hash="p"
    )
    observation = _observation()
    shadows = orchestrator.propose_shadows(
        observation=observation, evaluate_historical=lambda item: _evaluation(item.candidate_id)
    )
    assert len(shadows) == 1
    assert shadows[0].eligible_after_utc > observation.cutoff_utc
    close_cutoff = observation.cutoff_utc + timedelta(days=14)
    evaluation = _evaluation(
        "c1", decision="promote", windows=["2025-W28", "2025-W29"], cutoff=close_cutoff
    )
    closed, saved_reflection, policy = orchestrator.close_shadow(
        shadow=shadows[0], evaluation=evaluation, close_cutoff_utc=close_cutoff,
        active_policy_id="p0", memory_tags=["vol:normal"],
    )
    assert closed.status == "promoted"
    assert store.load_record("shadows", shadows[0].shadow_id)["status"] == "promoted"
    assert saved_reflection == reflection
    assert policy.activates_at_utc > close_cutoff


def test_orchestrator_rejects_same_window_or_one_week_shadow(tmp_path):
    candidate = _candidate()
    caller = FakeCaller([CandidateBatch(diagnosis="test", candidates=[candidate]), RefinerBatch(candidates=[candidate])])
    store = AgentStore(tmp_path / "state.sqlite")
    orchestrator = WeeklyOrchestrator(
        caller=caller, store=store, memory=MemoryManager(store, protocol_hash="p"), protocol_hash="p"
    )
    observation = _observation()
    shadow = orchestrator.propose_shadows(
        observation=observation, evaluate_historical=lambda item: _evaluation(item.candidate_id)
    )[0]
    with pytest.raises(ValueError, match="at least two"):
        orchestrator.close_shadow(
            shadow=shadow,
            evaluation=_evaluation("c1", decision="reject", windows=["2025-W28"], cutoff=observation.cutoff_utc + timedelta(days=7)),
            close_cutoff_utc=observation.cutoff_utc + timedelta(days=7),
            active_policy_id="p0", memory_tags=[],
        )


def test_expired_shadow_closes_without_promotion(tmp_path):
    candidate = _candidate()
    reflection = ReflectionRecord(
        reflection_id="r-expire", candidate_id="c1", evaluation_id="ev-c1-expire",
        evidence=["support remained sparse"], speculation=[],
        failure_cause="The candidate did not reach the fixed trade support.",
        memory_recommendation="store_episode",
    )
    caller = FakeCaller([
        CandidateBatch(diagnosis="test", candidates=[candidate]),
        RefinerBatch(candidates=[candidate]),
        reflection,
    ])
    store = AgentStore(tmp_path / "state.sqlite")
    orchestrator = WeeklyOrchestrator(
        caller=caller, store=store, memory=MemoryManager(store, protocol_hash="p"), protocol_hash="p"
    )
    observation = _observation()
    shadow = orchestrator.propose_shadows(
        observation=observation, evaluate_historical=lambda item: _evaluation(item.candidate_id)
    )[0]
    close_cutoff = observation.cutoff_utc + timedelta(days=28)
    evaluation = _evaluation(
        "c1", decision="expire", windows=["2025-W28", "2025-W29", "2025-W30", "2025-W31"],
        cutoff=close_cutoff,
    )
    closed, _, policy = orchestrator.close_shadow(
        shadow=shadow, evaluation=evaluation, close_cutoff_utc=close_cutoff,
        active_policy_id="p0", memory_tags=[],
    )
    assert closed.status == "expired"
    assert policy is None
