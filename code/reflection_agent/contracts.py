"""Strict JSON contracts shared by every reflection-agent boundary."""
from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "1.0"
MODEL_IDS = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)
CONDITION_FIELDS = (
    "vol_regime",
    "trend_regime",
    "model_disagreement",
    "ensemble_confidence",
    "news_impact",
    "news_dispersion",
    "data_quality_state",
    "hour_block",
    "day_of_week",
)
ConditionOperator = Literal["eq", "in", "gte", "lte"]
PolicyAction = Literal[
    "multiply_model_weight",
    "set_model_weight",
    "select_frozen_expert",
    "set_confidence_threshold",
    "require_minimum_agreement",
    "remove_active_edit",
    "reduce_active_edit",
]


class StrictModel(BaseModel):
    """Reject unspecified fields so prompts cannot expand the action surface."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProbabilityVector(StrictModel):
    short: Annotated[float, Field(ge=0.0, le=1.0)]
    flat: Annotated[float, Field(ge=0.0, le=1.0)]
    long: Annotated[float, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def probabilities_sum_to_one(self) -> "ProbabilityVector":
        if abs(self.short + self.flat + self.long - 1.0) > 1e-6:
            raise ValueError("probabilities must sum to one")
        return self


class ModelSignal(StrictModel):
    model_id: Literal[*MODEL_IDS]
    probabilities: ProbabilityVector
    predicted_class: Literal[0, 1, 2]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    ensemble_weight: Annotated[float, Field(ge=0.0, le=1.0)]
    ensemble_enabled: bool
    ensemble_active_fraction: Annotated[float, Field(ge=0.0, le=1.0)]
    active_weight_edit_ids: list[str] = Field(default_factory=list, max_length=8)


class MarketContext(StrictModel):
    vol_regime: Literal["low", "normal", "high"]
    trend_regime: Literal["down", "flat", "up"]
    realized_volatility: float
    recent_return: float


class EnsembleContext(StrictModel):
    predicted_class: Literal[0, 1, 2]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    agreement: Annotated[float, Field(ge=0.0, le=1.0)]
    model_disagreement: Annotated[float, Field(ge=0.0, le=1.0)]


class NewsItem(StrictModel):
    event_id: str
    available_at_utc: datetime
    source_family: Literal["gdelt_news", "direct_policy_event", "fred_macro", "fear_greed"]
    publisher_category: str
    summary: Annotated[str, Field(max_length=500)]
    impact: Annotated[float, Field(ge=0.0)]
    sentiment: Annotated[float, Field(ge=-1.0, le=1.0)]


class NewsContext(StrictModel):
    source_counts: dict[str, int]
    aggregate_features: dict[str, float]
    top_items: Annotated[list[NewsItem], Field(max_length=12)]


class MemorySnippet(StrictModel):
    memory_id: str
    memory_type: Literal["episodic", "semantic", "procedural"]
    cutoff_utc: datetime
    lesson: Annotated[str, Field(max_length=800)]
    evidence_status: Literal["supported", "contradicted", "rejected", "procedural"]
    tags: list[str] = Field(default_factory=list, max_length=12)


class DataQuality(StrictModel):
    state: Literal["ok", "degraded", "invalid"]
    missing_models: list[Literal[*MODEL_IDS]] = Field(default_factory=list)
    stale_source_families: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, max_length=12)


class ActivePolicySummary(StrictModel):
    policy_id: str
    base_control: Literal["unanimity_consensus"]
    active_edit_ids: list[str] = Field(default_factory=list, max_length=8)


class ObservationReport(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    window_id: str
    cutoff_utc: datetime
    active_policy_id: str
    active_policy: ActivePolicySummary
    market: MarketContext
    models: Annotated[list[ModelSignal], Field(min_length=9, max_length=9)]
    ensemble: EnsembleContext
    news: NewsContext
    retrieved_memories: Annotated[list[MemorySnippet], Field(max_length=8)]
    allowed_actions: list[PolicyAction]
    data_quality: DataQuality

    @field_validator("models")
    @classmethod
    def require_all_models_once(cls, models: list[ModelSignal]) -> list[ModelSignal]:
        ids = [model.model_id for model in models]
        if len(ids) != len(set(ids)) or set(ids) != set(MODEL_IDS):
            raise ValueError("models must contain every registered model exactly once")
        if abs(sum(model.ensemble_weight for model in models) - 1.0) > 1e-6:
            raise ValueError("model ensemble weights must sum to one")
        if any(model.ensemble_enabled != (model.ensemble_weight > 0.0) for model in models):
            raise ValueError("model enabled state must match its ensemble weight")
        return models


class ConditionPredicate(StrictModel):
    field: Literal[*CONDITION_FIELDS]
    operator: ConditionOperator
    value: str | float | int | list[str] | list[float] | list[int]

    @model_validator(mode="after")
    def in_requires_list(self) -> "ConditionPredicate":
        if self.operator == "in" and not isinstance(self.value, list):
            raise ValueError("operator 'in' requires a list value")
        if self.operator != "in" and isinstance(self.value, list):
            raise ValueError(f"operator '{self.operator}' requires a scalar value")
        return self


class ConditionTree(StrictModel):
    all: Annotated[list[ConditionPredicate], Field(min_length=1, max_length=2)]


class PolicyEdit(StrictModel):
    edit_id: str
    action: PolicyAction
    target: str | None = None
    value: float | str | None = None

    @model_validator(mode="after")
    def validate_action_shape(self) -> "PolicyEdit":
        model_actions = {"multiply_model_weight", "set_model_weight"}
        if self.action in model_actions:
            if self.target not in MODEL_IDS or not isinstance(self.value, float | int):
                raise ValueError(f"{self.action} requires a registered model target and numeric value")
        elif self.action == "select_frozen_expert":
            if self.target not in MODEL_IDS or self.value is not None:
                raise ValueError("select_frozen_expert requires only a registered model target")
        elif self.action in {"set_confidence_threshold", "require_minimum_agreement"}:
            if self.target is not None or not isinstance(self.value, float | int):
                raise ValueError(f"{self.action} requires only a numeric value")
        else:
            if not self.target or self.value is not None:
                raise ValueError(f"{self.action} requires only the target active edit id")
        return self


class PolicyRule(StrictModel):
    rule_id: str
    conditions: ConditionTree | None = None
    edits: Annotated[list[PolicyEdit], Field(min_length=1, max_length=2)]


class ExpectedEffect(StrictModel):
    net_return: Literal["increase", "unchanged", "decrease"]
    turnover: Literal["increase", "unchanged", "decrease"]
    drawdown: Literal["increase", "unchanged", "decrease"] = "unchanged"


class Candidate(StrictModel):
    candidate_id: str
    hypothesis: Annotated[str, Field(min_length=10, max_length=600)]
    conditions: ConditionTree | None = None
    edits: Annotated[list[PolicyEdit], Field(min_length=1, max_length=2)]
    mechanism: Annotated[str, Field(min_length=10, max_length=600)]
    expected_effect: ExpectedEffect
    falsifiers: Annotated[list[str], Field(min_length=1, max_length=6)]
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]


class CandidateBatch(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    diagnosis: Annotated[str, Field(max_length=800)]
    candidates: Annotated[list[Candidate], Field(max_length=6)]

    @field_validator("candidates")
    @classmethod
    def candidate_ids_are_unique(cls, candidates: list[Candidate]) -> list[Candidate]:
        ids = [candidate.candidate_id for candidate in candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("candidate ids must be unique")
        return candidates


class RefinerBatch(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    candidates: Annotated[list[Candidate], Field(max_length=3)]


class MetricBundle(StrictModel):
    trades: int
    long_trades: int
    short_trades: int
    net_return: float
    sortino: float
    sharpe: float
    max_drawdown: float
    turnover: float
    monthly_gain_concentration: Annotated[float, Field(ge=0.0, le=1.0)]


class EvaluationRecord(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    evaluation_id: str
    candidate_id: str
    window_ids: list[str]
    cutoff_utc: datetime
    baseline: MetricBundle
    candidate: MetricBundle
    delta_net_return: float
    paired_weekly_deltas: list[float]
    bootstrap_ci_low: float | None = None
    bootstrap_ci_high: float | None = None
    guard_results: dict[str, bool]
    decision: Literal["historical_keep", "historical_prune", "shadow_continue", "promote", "reject", "expire"]


class ReflectionRecord(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    reflection_id: str
    candidate_id: str
    evaluation_id: str
    evidence: Annotated[list[str], Field(max_length=8)]
    speculation: Annotated[list[str], Field(max_length=8)]
    failure_cause: Annotated[str | None, Field(max_length=500)] = None
    generalized_lesson: Annotated[str | None, Field(max_length=800)] = None
    invalidation_conditions: Annotated[list[str], Field(max_length=6)] = Field(default_factory=list)
    memory_recommendation: Literal["store_episode", "propose_semantic", "reject_lesson"]


class SemanticBelief(StrictModel):
    schema_version: Literal[SCHEMA_VERSION] = SCHEMA_VERSION
    belief_id: str
    lesson: Annotated[str, Field(min_length=10, max_length=800)]
    supporting_episode_ids: Annotated[list[str], Field(min_length=2)]
    counterexample_episode_ids: list[str] = Field(default_factory=list)
    valid_tags: list[str] = Field(default_factory=list, max_length=12)
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    created_at_utc: datetime
    expires_at_utc: datetime


class ShadowState(StrictModel):
    shadow_id: str
    candidate: Candidate
    source_window_id: str
    source_cutoff_utc: datetime
    eligible_after_utc: datetime
    observed_window_ids: list[str] = Field(default_factory=list, max_length=4)
    status: Literal["open", "promoted", "rejected", "expired"] = "open"

    @model_validator(mode="after")
    def starts_after_source_window(self) -> "ShadowState":
        if self.eligible_after_utc <= self.source_cutoff_utc:
            raise ValueError("shadow must start after the source window cutoff")
        return self


class PolicyVersion(StrictModel):
    policy_id: str
    parent_policy_id: str
    source_candidate_id: str
    rule: PolicyRule
    activates_at_utc: datetime
    evaluation_id: str
