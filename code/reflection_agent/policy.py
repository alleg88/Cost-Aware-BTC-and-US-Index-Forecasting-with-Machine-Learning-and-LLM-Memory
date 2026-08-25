"""Deterministic compiler for the bounded reflection policy DSL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from reflection_agent.contracts import MODEL_IDS, ConditionTree, PolicyEdit, PolicyRule

PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")
WEIGHT_GRID = (0.50, 0.75, 1.00, 1.25, 1.50)
CONFIDENCE_GRID = (0.65, 0.70, 0.75, 0.80)
AGREEMENT_GRID = (0.55, 0.65, 0.75, 1.00)
NUMERIC_FIELDS = {"model_disagreement", "ensemble_confidence", "news_impact", "news_dispersion"}
CATEGORICAL_FIELDS = {"vol_regime", "trend_regime", "data_quality_state", "hour_block", "day_of_week"}


@dataclass(frozen=True)
class CompiledPolicy:
    probabilities: pd.DataFrame
    weights: pd.DataFrame
    predictions: pd.Series
    confidence: pd.Series
    agreement: pd.Series
    eligible: pd.Series
    active_rules: tuple[PolicyRule, ...]


def _is_grid_value(value: float, grid: Sequence[float]) -> bool:
    return any(np.isclose(float(value), item) for item in grid)


def validate_edit(edit: PolicyEdit) -> None:
    """Apply semantic checks that JSON Schema cannot express."""
    if edit.action in {"multiply_model_weight", "set_model_weight"}:
        if not _is_grid_value(float(edit.value), WEIGHT_GRID):
            raise ValueError(f"weight value is outside fixed grid: {edit.value}")
    elif edit.action == "set_confidence_threshold":
        if not _is_grid_value(float(edit.value), CONFIDENCE_GRID):
            raise ValueError(f"confidence threshold is outside fixed grid: {edit.value}")
    elif edit.action == "require_minimum_agreement":
        if not _is_grid_value(float(edit.value), AGREEMENT_GRID):
            raise ValueError(f"agreement threshold is outside fixed grid: {edit.value}")


def validate_conditions(tree: ConditionTree | None) -> None:
    if tree:
        for predicate in tree.all:
            if predicate.field in NUMERIC_FIELDS and isinstance(predicate.value, str):
                raise ValueError(f"numeric field requires numeric value: {predicate.field}")
            if predicate.field in CATEGORICAL_FIELDS and predicate.operator in {"gte", "lte"}:
                raise ValueError(f"categorical field cannot use {predicate.operator}: {predicate.field}")


def condition_mask(context: pd.DataFrame, tree: ConditionTree | None) -> pd.Series:
    """Compile a validated condition tree without eval, Python, or raw text."""
    mask = pd.Series(True, index=context.index, dtype=bool)
    if tree is None:
        return mask
    for predicate in tree.all:
        if predicate.field not in context.columns:
            raise ValueError(f"missing condition field: {predicate.field}")
        series = context[predicate.field]
        if predicate.operator == "eq":
            current = series.eq(predicate.value)
        elif predicate.operator == "in":
            current = series.isin(predicate.value)
        elif predicate.operator == "gte":
            current = pd.to_numeric(series, errors="raise").ge(float(predicate.value))
        elif predicate.operator == "lte":
            current = pd.to_numeric(series, errors="raise").le(float(predicate.value))
        else:  # pragma: no cover - Literal and Pydantic prevent this
            raise ValueError(f"unsupported operator: {predicate.operator}")
        mask &= current.fillna(False).astype(bool)
    return mask


def _reduce_value(edit: PolicyEdit) -> PolicyEdit:
    if edit.action in {"multiply_model_weight", "set_model_weight"}:
        if np.isclose(float(edit.value), 1.0):
            return edit
        value = min(WEIGHT_GRID, key=lambda item: (abs(item - 1.0), abs(item - float(edit.value))))
        if not np.isclose(value, float(edit.value)):
            return edit.model_copy(update={"value": value})
        candidates = sorted(WEIGHT_GRID, key=lambda item: abs(item - 1.0))
        return edit.model_copy(update={"value": next(item for item in candidates if item != value)})
    if edit.action == "set_confidence_threshold":
        lower = [item for item in CONFIDENCE_GRID if item < float(edit.value)]
        return edit.model_copy(update={"value": max(lower, default=CONFIDENCE_GRID[0])})
    if edit.action == "require_minimum_agreement":
        lower = [item for item in AGREEMENT_GRID if item < float(edit.value)]
        return edit.model_copy(update={"value": max(lower, default=AGREEMENT_GRID[0])})
    return edit


def resolve_edits(active: Sequence[PolicyEdit], requested: Sequence[PolicyEdit]) -> tuple[PolicyEdit, ...]:
    """Apply remove/reduce meta-edits to an immutable active edit set."""
    resolved = {edit.edit_id: edit for edit in active}
    for edit in requested:
        validate_edit(edit)
        if edit.action == "remove_active_edit":
            if edit.target not in resolved:
                raise ValueError(f"unknown active edit id: {edit.target}")
            del resolved[edit.target]
        elif edit.action == "reduce_active_edit":
            if edit.target not in resolved:
                raise ValueError(f"unknown active edit id: {edit.target}")
            resolved[edit.target] = _reduce_value(resolved[edit.target])
        else:
            if edit.edit_id in resolved:
                raise ValueError(f"duplicate edit id: {edit.edit_id}")
            resolved[edit.edit_id] = edit
    return tuple(resolved[key] for key in sorted(resolved))


def _validate_probability_panel(probabilities: Mapping[str, pd.DataFrame]) -> pd.Index:
    if set(probabilities) != set(MODEL_IDS):
        raise ValueError("probability panel must contain every registered model exactly once")
    index = next(iter(probabilities.values())).index
    for model_id, frame in probabilities.items():
        if tuple(frame.columns) != PROBABILITY_COLUMNS:
            raise ValueError(f"{model_id} probability columns must be {PROBABILITY_COLUMNS}")
        if not frame.index.equals(index):
            raise ValueError("all probability frames must share the same index")
        values = frame.to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"invalid probability value for {model_id}")
        if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError(f"probabilities do not sum to one for {model_id}")
    return index


def compile_policy(
    probabilities: Mapping[str, pd.DataFrame],
    context: pd.DataFrame,
    rules: Sequence[PolicyRule],
) -> CompiledPolicy:
    """Apply bounded edits and return ensemble probabilities plus deterministic gates."""
    index = _validate_probability_panel(probabilities)
    if not context.index.equals(index):
        raise ValueError("context and probability panel must share the same index")
    for rule in rules:
        validate_conditions(rule.conditions)
        for edit in rule.edits:
            validate_edit(edit)
            if edit.action in {"remove_active_edit", "reduce_active_edit"}:
                raise ValueError("resolve meta-edits before compilation")

    model_ids = list(MODEL_IDS)
    weights = pd.DataFrame(1.0, index=index, columns=model_ids)
    confidence_threshold = pd.Series(0.0, index=index)
    agreement_threshold = pd.Series(0.0, index=index)
    for rule in rules:
        mask = condition_mask(context, rule.conditions)
        for edit in rule.edits:
            if edit.action == "multiply_model_weight":
                weights.loc[mask, edit.target] *= float(edit.value)
            elif edit.action == "set_model_weight":
                weights.loc[mask, edit.target] = float(edit.value)
            elif edit.action == "select_frozen_expert":
                weights.loc[mask, :] = 0.0
                weights.loc[mask, edit.target] = 1.0
            elif edit.action == "set_confidence_threshold":
                confidence_threshold.loc[mask] = float(edit.value)
            elif edit.action == "require_minimum_agreement":
                agreement_threshold.loc[mask] = float(edit.value)

    totals = weights.sum(axis=1)
    if totals.le(0).any():
        raise ValueError("compiled model weights must have positive row sums")
    normalized = weights.div(totals, axis=0)
    stacked = np.stack([probabilities[model_id].to_numpy(dtype=float) for model_id in model_ids], axis=1)
    ensemble = (stacked * normalized.to_numpy()[:, :, None]).sum(axis=1)
    ensemble_frame = pd.DataFrame(ensemble, index=index, columns=PROBABILITY_COLUMNS)
    raw_pred = pd.Series(ensemble.argmax(axis=1), index=index, dtype=int)
    confidence = pd.Series(ensemble.max(axis=1), index=index, dtype=float)
    model_pred = np.stack([probabilities[model_id].to_numpy().argmax(axis=1) for model_id in model_ids], axis=1)
    agreement = pd.Series((model_pred == raw_pred.to_numpy()[:, None]).mean(axis=1), index=index)
    eligible = confidence.ge(confidence_threshold) & agreement.ge(agreement_threshold)
    predictions = raw_pred.where(eligible, 1).astype(int)
    return CompiledPolicy(
        probabilities=ensemble_frame,
        weights=normalized,
        predictions=predictions,
        confidence=confidence,
        agreement=agreement,
        eligible=eligible,
        active_rules=tuple(rules),
    )
