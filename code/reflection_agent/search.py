"""Deterministic validation and pruning for the bounded Tree-of-Thought search."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

from reflection_agent.contracts import Candidate, EvaluationRecord, PolicyRule, RefinerBatch
from reflection_agent.policy import validate_conditions, validate_edit


def candidate_rule(candidate: Candidate) -> PolicyRule:
    validate_conditions(candidate.conditions)
    for edit in candidate.edits:
        validate_edit(edit)
        if edit.action in {"remove_active_edit", "reduce_active_edit"} and candidate.conditions is not None:
            raise ValueError("meta-edits must be unconditional")
    return PolicyRule(rule_id=candidate.candidate_id, conditions=candidate.conditions, edits=candidate.edits)


def validate_candidates(candidates: Sequence[Candidate], *, maximum: int = 6) -> tuple[Candidate, ...]:
    if len(candidates) > maximum:
        raise ValueError(f"candidate budget exceeded: {len(candidates)} > {maximum}")
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise ValueError("candidate ids must be unique")
    for candidate in candidates:
        candidate_rule(candidate)
    return tuple(candidates)


def evaluation_rank(record: EvaluationRecord) -> tuple[float, float, float, float, float, str]:
    admitted = float(record.decision in {"historical_keep", "shadow_continue", "promote"})
    passed = float(sum(record.guard_results.values()))
    return (
        admitted,
        passed,
        record.delta_net_return,
        record.candidate.sortino,
        -record.candidate.max_drawdown,
        record.candidate_id,
    )


def select_beam(
    candidates: Sequence[Candidate],
    evaluations: Sequence[EvaluationRecord],
    *,
    width: int = 3,
) -> tuple[Candidate, ...]:
    by_candidate: Mapping[str, EvaluationRecord] = {record.candidate_id: record for record in evaluations}
    missing = {candidate.candidate_id for candidate in candidates} - set(by_candidate)
    if missing:
        raise ValueError(f"missing evaluations: {sorted(missing)}")
    ranked = sorted(candidates, key=lambda candidate: evaluation_rank(by_candidate[candidate.candidate_id]), reverse=True)
    return tuple(ranked[:width])


def validate_refinements(original: Sequence[Candidate], refined: RefinerBatch) -> tuple[Candidate, ...]:
    allowed = {candidate.candidate_id for candidate in original}
    ids = [candidate.candidate_id for candidate in refined.candidates]
    unknown = set(ids) - allowed
    if unknown:
        raise ValueError(f"refiner invented candidate ids: {sorted(unknown)}")
    if len(ids) != len(set(ids)):
        raise ValueError("refiner returned duplicate candidate ids")
    if len(ids) > len(original):
        raise ValueError("refiner expanded the branch count")
    return validate_candidates(refined.candidates, maximum=3)


def select_shadows(
    candidates: Sequence[Candidate],
    evaluations: Sequence[EvaluationRecord],
    *,
    maximum: int = 2,
) -> tuple[Candidate, ...]:
    by_candidate = {record.candidate_id: record for record in evaluations}
    missing = {candidate.candidate_id for candidate in candidates} - set(by_candidate)
    if missing:
        raise ValueError(f"missing evaluations: {sorted(missing)}")
    keepable = [
        candidate for candidate in candidates
        if by_candidate[candidate.candidate_id].decision == "historical_keep"
    ]
    return select_beam(keepable, evaluations, width=maximum) if keepable else ()
