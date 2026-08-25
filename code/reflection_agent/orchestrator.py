"""Weekly causal Actor → ToT → shadow → Reflector orchestration."""
from __future__ import annotations

from datetime import timedelta
from typing import Callable, Sequence

from reflection_agent.contracts import (
    Candidate,
    CandidateBatch,
    EvaluationRecord,
    ObservationReport,
    PolicyVersion,
    ReflectionRecord,
    RefinerBatch,
    ShadowState,
)
from reflection_agent.manifest import sha256_payload
from reflection_agent.memory import MemoryManager
from reflection_agent.prompts import actor_messages, refiner_messages, reflector_messages
from reflection_agent.search import candidate_rule, select_beam, select_shadows, validate_candidates, validate_refinements
from reflection_agent.store import AgentStore
from reflection_agent.transport import StructuredCaller

HistoricalEvaluator = Callable[[Candidate], EvaluationRecord]


class WeeklyOrchestrator:
    def __init__(
        self,
        *,
        caller: StructuredCaller,
        store: AgentStore,
        memory: MemoryManager,
        protocol_hash: str,
        candidate_budget: int = 6,
        beam_width: int = 3,
        max_open_shadows: int = 2,
    ):
        self.caller = caller
        self.store = store
        self.memory = memory
        self.protocol_hash = protocol_hash
        self.candidate_budget = candidate_budget
        self.beam_width = beam_width
        self.max_open_shadows = max_open_shadows

    def propose_shadows(
        self,
        *,
        observation: ObservationReport,
        evaluate_historical: HistoricalEvaluator,
    ) -> tuple[ShadowState, ...]:
        actor = self.caller.call(
            role="actor",
            messages=actor_messages(observation),
            response_model=CandidateBatch,
            temperature=0.2,
        )
        if actor.value is None or not actor.value.candidates:
            return ()
        try:
            candidates = validate_candidates(actor.value.candidates, maximum=self.candidate_budget)
        except ValueError:
            return ()
        first_evaluations = [evaluate_historical(candidate) for candidate in candidates]
        self._store_screen(observation.window_id, "actor", candidates, first_evaluations)
        beam = select_beam(candidates, first_evaluations, width=self.beam_width)
        beam_evaluations = [
            next(record for record in first_evaluations if record.candidate_id == candidate.candidate_id)
            for candidate in beam
        ]
        refiner = self.caller.call(
            role="refiner",
            messages=refiner_messages(
                observation=observation,
                candidates=[candidate.model_dump(mode="json") for candidate in beam],
                evaluations=beam_evaluations,
                compiler_warnings=[],
            ),
            response_model=RefinerBatch,
            temperature=0.0,
        )
        if refiner.value is None:
            return ()
        try:
            refined = validate_refinements(beam, refiner.value)
        except ValueError:
            return ()
        refined_evaluations = [evaluate_historical(candidate) for candidate in refined]
        self._store_screen(observation.window_id, "refiner", refined, refined_evaluations)
        selected = select_shadows(refined, refined_evaluations, maximum=self.max_open_shadows)
        shadows = []
        for candidate in selected:
            shadow_id = f"shadow-{sha256_payload({'candidate': candidate.candidate_id, 'window': observation.window_id})[:20]}"
            shadow = ShadowState(
                shadow_id=shadow_id,
                candidate=candidate,
                source_window_id=observation.window_id,
                source_cutoff_utc=observation.cutoff_utc,
                eligible_after_utc=observation.cutoff_utc + timedelta(microseconds=1),
            )
            self.store.save_record(
                "shadows", shadow_id, self.protocol_hash, shadow.model_dump(mode="json"),
                window_id=observation.window_id,
            )
            shadows.append(shadow)
        return tuple(shadows)

    def _store_screen(
        self,
        window_id: str,
        stage: str,
        candidates: Sequence[Candidate],
        evaluations: Sequence[EvaluationRecord],
    ) -> None:
        for candidate in candidates:
            record_id = f"candidate-{sha256_payload({'stage': stage, 'candidate': candidate.model_dump(mode='json')})[:20]}"
            self.store.save_record(
                "candidates", record_id, self.protocol_hash,
                {"stage": stage, "candidate": candidate.model_dump(mode="json")}, window_id=window_id,
            )
        for evaluation in evaluations:
            self.store.save_record(
                "evaluations", f"{stage}:{evaluation.evaluation_id}", self.protocol_hash,
                {"stage": stage, "evaluation": evaluation.model_dump(mode="json")}, window_id=window_id,
            )

    def close_shadow(
        self,
        *,
        shadow: ShadowState,
        evaluation: EvaluationRecord,
        close_cutoff_utc,
        active_policy_id: str,
        memory_tags: Sequence[str],
    ) -> tuple[ShadowState, ReflectionRecord | None, PolicyVersion | None]:
        if shadow.status != "open":
            raise ValueError("only an open shadow can be closed")
        if close_cutoff_utc <= shadow.source_cutoff_utc:
            raise ValueError("shadow outcome must be later than its source window")
        if len(evaluation.window_ids) < 2:
            raise ValueError("shadow requires at least two unseen windows")
        if len(evaluation.window_ids) > 4:
            raise ValueError("shadow cannot exceed four windows")
        if evaluation.candidate_id != shadow.candidate.candidate_id:
            raise ValueError("shadow evaluation candidate changed")
        if evaluation.cutoff_utc > close_cutoff_utc:
            raise ValueError("evaluation cutoff exceeds close cutoff")

        reflector = self.caller.call(
            role="reflector",
            messages=reflector_messages(
                candidate=shadow.candidate.model_dump(mode="json"),
                evaluation=evaluation,
            ),
            response_model=ReflectionRecord,
            temperature=0.0,
        )
        reflection = reflector.value
        if reflection and (
            reflection.candidate_id != shadow.candidate.candidate_id
            or reflection.evaluation_id != evaluation.evaluation_id
        ):
            reflection = None
        if evaluation.decision == "expire":
            status = "expired"
        else:
            status = "promoted" if evaluation.decision == "promote" and reflection is not None else "rejected"
        closed = shadow.model_copy(update={"observed_window_ids": evaluation.window_ids, "status": status})
        self.store.replace_record(
            "shadows", shadow.shadow_id, self.protocol_hash, closed.model_dump(mode="json"),
            window_id=evaluation.window_ids[-1],
        )
        self.store.save_record(
            "evaluations", evaluation.evaluation_id, self.protocol_hash,
            evaluation.model_dump(mode="json"), window_id=evaluation.window_ids[-1],
        )
        if reflection:
            self.store.save_record(
                "reflections", reflection.reflection_id, self.protocol_hash,
                reflection.model_dump(mode="json"), window_id=evaluation.window_ids[-1],
            )
            self.memory.store_episode(
                reflection=reflection,
                evaluation=evaluation,
                cutoff_utc=close_cutoff_utc,
                tags=memory_tags,
            )
        policy = None
        if status == "promoted":
            policy_id = f"policy-{sha256_payload({'shadow': shadow.shadow_id, 'evaluation': evaluation.evaluation_id})[:20]}"
            policy = PolicyVersion(
                policy_id=policy_id,
                parent_policy_id=active_policy_id,
                source_candidate_id=shadow.candidate.candidate_id,
                rule=candidate_rule(shadow.candidate),
                activates_at_utc=close_cutoff_utc + timedelta(microseconds=1),
                evaluation_id=evaluation.evaluation_id,
            )
            self.store.save_record(
                "policies", policy_id, self.protocol_hash, policy.model_dump(mode="json"),
                window_id=evaluation.window_ids[-1],
            )
        return closed, reflection, policy
