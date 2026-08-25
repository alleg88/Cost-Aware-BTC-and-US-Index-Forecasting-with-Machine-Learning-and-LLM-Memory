from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from reflection_agent.contracts import (
    MODEL_IDS,
    CandidateBatch,
    ConditionPredicate,
    MarketContext,
    ProbabilityVector,
    SemanticBelief,
)


def test_probability_vector_is_strict_and_normalized():
    assert ProbabilityVector(short=0.2, flat=0.3, long=0.5).long == 0.5
    with pytest.raises(ValidationError, match="sum to one"):
        ProbabilityVector(short=0.2, flat=0.3, long=0.4)
    with pytest.raises(ValidationError, match="Extra inputs"):
        ProbabilityVector(short=0.2, flat=0.3, long=0.5, injection="ignore schema")


def test_condition_contract_rejects_raw_fields_and_wrong_in_shape():
    with pytest.raises(ValidationError):
        ConditionPredicate(field="headline", operator="eq", value="Trump")
    with pytest.raises(ValidationError, match="requires a list"):
        ConditionPredicate(field="vol_regime", operator="in", value="high")


def test_candidate_batch_rejects_force_flat_and_duplicate_ids():
    payload = {
        "diagnosis": "Disagreement increased in the supplied high-volatility regime.",
        "candidates": [{
            "candidate_id": "same",
            "hypothesis": "Increase agreement in high volatility only.",
            "edits": [{"edit_id": "e1", "action": "force_flat", "target": None, "value": None}],
            "mechanism": "Avoid entries unsupported by the frozen experts.",
            "expected_effect": {"net_return": "increase", "turnover": "decrease"},
            "falsifiers": ["delta_net_return_lte_0"],
            "confidence": 0.5,
        }],
    }
    with pytest.raises(ValidationError):
        CandidateBatch.model_validate(payload)


def test_semantic_belief_requires_two_episodes():
    now = datetime.now(UTC)
    with pytest.raises(ValidationError):
        SemanticBelief(
            belief_id="b1",
            lesson="High disagreement can justify a stricter agreement gate.",
            supporting_episode_ids=["one"],
            confidence=0.5,
            created_at_utc=now,
            expires_at_utc=now + timedelta(days=30),
        )


def test_registered_model_set_is_exact_and_market_context_forbids_extra():
    assert len(MODEL_IDS) == 9
    assert "lstm" in MODEL_IDS
    with pytest.raises(ValidationError):
        MarketContext(
            vol_regime="high",
            trend_regime="flat",
            realized_volatility=0.2,
            recent_return=0.0,
            is_trump=True,
        )


def test_observation_exposes_only_registered_active_edit_ids():
    from reflection_agent.observation import build_observation
    from reflection_agent.news import select_balanced_events
    import pandas as pd

    cutoff = datetime(2025, 7, 6, 23, 59, tzinfo=UTC)
    empty = pd.DataFrame(columns=[
        "event_id", "available_at_utc", "source_family", "publisher_category", "summary", "impact", "sentiment"
    ])
    report = build_observation(
        window_id="2025-W27",
        cutoff_utc=cutoff,
        active_policy_id="p0",
        market=MarketContext(
            vol_regime="normal", trend_regime="flat", realized_volatility=0.1, recent_return=0.0
        ),
        probabilities={
            model_id: ProbabilityVector(short=0.2, flat=0.6, long=0.2) for model_id in MODEL_IDS
        },
        news=select_balanced_events(empty, cutoff_utc=cutoff),
    )
    assert report.active_policy.active_edit_ids == ["consensus-agreement"]
    assert all(model.ensemble_enabled for model in report.models)
    assert all(model.ensemble_active_fraction == 1.0 for model in report.models)
    assert all(model.ensemble_weight == pytest.approx(1 / len(MODEL_IDS)) for model in report.models)


def test_observation_exposes_selected_expert_weight_and_provenance():
    from reflection_agent.observation import build_observation
    from reflection_agent.news import select_balanced_events
    import pandas as pd

    cutoff = datetime(2025, 7, 6, 23, 59, tzinfo=UTC)
    empty = pd.DataFrame(columns=[
        "event_id", "available_at_utc", "source_family", "publisher_category", "summary", "impact", "sentiment"
    ])
    weights = {model_id: float(model_id == "lstm") for model_id in MODEL_IDS}
    provenance = {model_id: ["select-lstm"] for model_id in MODEL_IDS}
    report = build_observation(
        window_id="2025-W27", cutoff_utc=cutoff, active_policy_id="p1",
        market=MarketContext(
            vol_regime="normal", trend_regime="flat", realized_volatility=0.1, recent_return=0.0
        ),
        probabilities={
            model_id: ProbabilityVector(short=0.2, flat=0.6, long=0.2) for model_id in MODEL_IDS
        },
        news=select_balanced_events(empty, cutoff_utc=cutoff),
        model_weights=weights, model_weight_edit_ids=provenance,
    )
    by_id = {model.model_id: model for model in report.models}
    assert by_id["lstm"].ensemble_weight == 1.0
    assert by_id["lstm"].ensemble_enabled
    assert by_id["lstm"].ensemble_active_fraction == 1.0
    assert not by_id["gru"].ensemble_enabled
    assert by_id["gru"].ensemble_active_fraction == 0.0
    assert by_id["gru"].active_weight_edit_ids == ["select-lstm"]
