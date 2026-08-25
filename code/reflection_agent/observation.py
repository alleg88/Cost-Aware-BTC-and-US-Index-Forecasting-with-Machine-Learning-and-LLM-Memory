"""Deterministic construction of the compact weekly LLM observation."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np

from reflection_agent.contracts import (
    MODEL_IDS,
    ActivePolicySummary,
    DataQuality,
    EnsembleContext,
    MarketContext,
    MemorySnippet,
    ModelSignal,
    NewsContext,
    ObservationReport,
    PolicyAction,
    ProbabilityVector,
)

ALLOWED_ACTIONS: tuple[PolicyAction, ...] = (
    "multiply_model_weight",
    "set_model_weight",
    "select_frozen_expert",
    "set_confidence_threshold",
    "require_minimum_agreement",
    "remove_active_edit",
    "reduce_active_edit",
)


def build_observation(
    *,
    window_id: str,
    cutoff_utc: datetime,
    active_policy_id: str,
    active_edit_ids: Sequence[str] = ("consensus-agreement",),
    market: MarketContext,
    probabilities: dict[str, ProbabilityVector],
    news: NewsContext,
    model_weights: dict[str, float] | None = None,
    model_active_fractions: dict[str, float] | None = None,
    model_weight_edit_ids: dict[str, Sequence[str]] | None = None,
    retrieved_memories: Sequence[MemorySnippet] = (),
    data_quality: DataQuality | None = None,
) -> ObservationReport:
    if set(probabilities) != set(MODEL_IDS):
        raise ValueError("observation requires every registered model")
    weights = model_weights or {model_id: 1.0 for model_id in MODEL_IDS}
    if set(weights) != set(MODEL_IDS):
        raise ValueError("observation weights require every registered model")
    weight_values = np.asarray([weights[model_id] for model_id in MODEL_IDS], dtype=float)
    if not np.isfinite(weight_values).all() or (weight_values < 0.0).any() or weight_values.sum() <= 0.0:
        raise ValueError("observation weights must be finite, nonnegative, and nonzero")
    weight_values = weight_values / weight_values.sum()
    active_fractions = model_active_fractions or {
        model_id: float(weight_values[index] > 0.0)
        for index, model_id in enumerate(MODEL_IDS)
    }
    if set(active_fractions) != set(MODEL_IDS):
        raise ValueError("active fractions require every registered model")
    fraction_values = np.asarray([active_fractions[model_id] for model_id in MODEL_IDS], dtype=float)
    if not np.isfinite(fraction_values).all() or (fraction_values < 0.0).any() or (fraction_values > 1.0).any():
        raise ValueError("active fractions must be finite values between zero and one")
    weight_edits = model_weight_edit_ids or {model_id: () for model_id in MODEL_IDS}
    if set(weight_edits) != set(MODEL_IDS):
        raise ValueError("weight edit provenance requires every registered model")
    model_signals = []
    predictions = []
    vectors = []
    for model_index, model_id in enumerate(MODEL_IDS):
        vector = probabilities[model_id]
        values = np.array([vector.short, vector.flat, vector.long], dtype=float)
        prediction = int(values.argmax())
        predictions.append(prediction)
        vectors.append(values)
        model_signals.append(ModelSignal(
            model_id=model_id,
            probabilities=vector,
            predicted_class=prediction,
            confidence=float(values.max()),
            ensemble_weight=float(weight_values[model_index]),
            ensemble_enabled=bool(weight_values[model_index] > 0.0),
            ensemble_active_fraction=float(fraction_values[model_index]),
            active_weight_edit_ids=list(weight_edits[model_id]),
        ))
    ensemble_values = np.average(np.stack(vectors), axis=0, weights=weight_values)
    ensemble_prediction = int(ensemble_values.argmax())
    agreement = float(np.mean(np.asarray(predictions) == ensemble_prediction))
    ensemble = EnsembleContext(
        predicted_class=ensemble_prediction,
        confidence=float(ensemble_values.max()),
        agreement=agreement,
        model_disagreement=1.0 - agreement,
    )
    quality = data_quality or DataQuality(state="ok")
    return ObservationReport(
        window_id=window_id,
        cutoff_utc=cutoff_utc,
        active_policy_id=active_policy_id,
        active_policy=ActivePolicySummary(
            policy_id=active_policy_id,
            base_control="unanimity_consensus",
            active_edit_ids=list(active_edit_ids),
        ),
        market=market,
        models=model_signals,
        ensemble=ensemble,
        news=news,
        retrieved_memories=list(retrieved_memories),
        allowed_actions=list(ALLOWED_ACTIONS),
        data_quality=quality,
    )
