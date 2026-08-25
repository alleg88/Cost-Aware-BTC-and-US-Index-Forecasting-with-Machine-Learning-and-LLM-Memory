"""Causal proposal, future-shadow, reflection, memory, and policy transitions."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from typing import Annotated, Any, Literal, Sequence

import pandas as pd
from pydantic import Field

from reflection_agent.v2.contracts import (
    CONDITION_VALUES,
    EvidenceCard,
    MemoryCard,
    ProposalOutput,
    ReflectionOutput,
    StrictModel,
    validate_proposal_references,
)
from reflection_agent.v2.evaluator import GateDecision, ShadowCandidate, evaluate_shadow
from reflection_agent.v2.leakage import (
    FeatureProvenance,
    LeakageAuditor,
    PromptAuditContext,
    STAGE_SCOPES,
    SignalProvenance,
)
from reflection_agent.v2.memory import RealMemory, SemanticSupport
from reflection_agent.v2.policy import ActiveAllowRule
from reflection_agent.v2.prompts import proposal_messages, reflection_messages


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _hash(payload: object) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


class EpisodeProposalResult(StrictModel):
    status: Literal[
        "candidate_opened",
        "no_change",
        "invalid_reference",
        "schema_failure",
        "transport_error",
        "open_candidate_exists",
        "static_control",
    ]
    candidate: ShadowCandidate | None
    proposal: ProposalOutput | None
    evidence_ids: list[str]
    memory_ids: list[str]
    call_status: str | None


class CandidateClosure(StrictModel):
    status: Literal["closed", "still_open"]
    evaluation: GateDecision
    reflection: ReflectionOutput | None
    new_active_rule: ActiveAllowRule | None
    removed_rule_id: str | None
    episodic_memory_id: str | None


def _single_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"episode must contain one {column}")
    return values.iloc[0]


def _dominant_tag(frame: pd.DataFrame, column: str, prefix: str) -> str | None:
    if column not in frame or frame[column].dropna().empty:
        return None
    return f"{prefix}:{frame[column].dropna().astype(str).mode().iloc[0]}"


def _evidence_cards(frame: pd.DataFrame, *, namespace: str) -> list[EvidenceCard]:
    cards: list[EvidenceCard] = []
    for (route, side), group in frame.groupby(["route", "side"], sort=True):
        source_hashes = sorted(set(group["source_artifact_hash"].astype(str)))
        source_hash = source_hashes[0] if len(source_hashes) == 1 else _hash(source_hashes)
        identity = _hash(
            {
                "namespace": namespace,
                "opportunity_ids": sorted(group["opportunity_id"].astype(str)),
            }
        )
        tags = [f"route:{route}", f"side:{side}"]
        for column, prefix in (
            ("member_pattern", "member"),
            ("vol_regime", "vol"),
            ("trend_regime", "trend"),
            ("funding_regime", "funding"),
            ("oi_regime", "oi"),
        ):
            tag = _dominant_tag(group, column, prefix)
            if tag is not None:
                tags.append(tag)
        cards.append(
            EvidenceCard(
                evidence_id=f"evidence-{identity[:24]}",
                stage=str(_single_value(group, "stage")),
                source_role=str(_single_value(group, "source_role")),
                decision_time=pd.to_datetime(group["decision_time"], utc=True).max(),
                outcome_available_time=pd.to_datetime(
                    group["outcome_available_time"], utc=True
                ).max(),
                fold_id=int(_single_value(group, "fold_id")),
                row_key=f"aggregate-{identity[:24]}",
                source_artifact_hash=source_hash,
                route=str(route),
                side=str(side),
                count=len(group),
                net_return=float(group["net_return"].sum()),
                tags=tags,
            )
        )
    if not 1 <= len(cards) <= 4:
        raise ValueError("an episode must produce one to four evidence cards")
    return cards


def _episode_tags(frame: pd.DataFrame) -> tuple[str, ...]:
    tags: set[str] = set()
    for column, prefix in (
        ("side", "side"),
        ("member_pattern", "member"),
        ("vol_regime", "vol"),
        ("trend_regime", "trend"),
        ("funding_regime", "funding"),
        ("oi_regime", "oi"),
    ):
        if column in frame:
            tags.update(f"{prefix}:{value}" for value in frame[column].dropna().astype(str))
    return tuple(sorted(tags))


def _provenance(frame: pd.DataFrame) -> tuple[list[FeatureProvenance], list[SignalProvenance]]:
    features = [
        FeatureProvenance(
            feature_name="registered_row_context",
            available_at_utc=pd.Timestamp(row.feature_available_time).to_pydatetime(),
            source_row_key=str(row.row_key),
        )
        for row in frame.itertuples(index=False)
    ]
    signals = [
        SignalProvenance(
            row_key=str(row.row_key),
            stage=str(row.stage),
            source_role=str(row.source_role),
            decision_time=pd.Timestamp(row.decision_time).to_pydatetime(),
            fold_id=int(row.fold_id),
        )
        for row in frame.itertuples(index=False)
    ]
    return features, signals


class ReflectionOrchestrator:
    def __init__(
        self,
        *,
        caller: Any,
        auditor: LeakageAuditor,
        memory: RealMemory,
        protocol_hash: str,
    ) -> None:
        self.caller = caller
        self.auditor = auditor
        self.memory = memory
        self.protocol_hash = protocol_hash

    def propose_episode(
        self,
        episode: pd.DataFrame,
        *,
        episode_id: str,
        episode_number: int,
        active_rules: Sequence[ActiveAllowRule],
        open_candidate: ShadowCandidate | None = None,
    ) -> EpisodeProposalResult:
        if open_candidate is not None:
            return EpisodeProposalResult(
                status="open_candidate_exists",
                candidate=open_candidate,
                proposal=None,
                evidence_ids=[],
                memory_ids=[],
                call_status=None,
            )
        if not getattr(self.memory, "uses_llm", True):
            return EpisodeProposalResult(
                status="static_control",
                candidate=None,
                proposal=None,
                evidence_ids=[],
                memory_ids=[],
                call_status=None,
            )
        frame = episode.copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError("cannot propose from an empty episode")
        for column in (
            "decision_time",
            "feature_available_time",
            "entry_time",
            "outcome_available_time",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        stage = str(_single_value(frame, "stage"))
        fold_id = int(_single_value(frame, "fold_id"))
        source_role = str(_single_value(frame, "source_role"))
        expected_role = "OOF_TEST" if stage == "development" else "FROZEN_EXACT"
        if source_role != expected_role:
            raise ValueError("episode source role does not match its stage")
        protocol_scope = STAGE_SCOPES[stage]
        if self.memory.protocol_scope != protocol_scope:
            raise ValueError("memory store does not match episode protocol scope")
        cutoff = frame["outcome_available_time"].max().to_pydatetime()
        cards = _evidence_cards(frame, namespace=episode_id)
        tags = _episode_tags(frame)
        memories = self.memory.retrieve(
            cutoff_utc=cutoff,
            tags=tags,
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=4,
        )
        observation = {
            "protocol_summary": {
                "protocol_version": "reflection-agent-v2.2",
                "base_policy": "frozen_union_v1",
                "action": "ALLOW_REENTRY",
            },
            "source_episode_id": episode_id,
            "episode_cutoff_utc": cutoff.isoformat(),
            "episode_metrics": {
                "opportunities": len(frame),
                "long": int(frame["side"].eq("LONG").sum()),
                "short": int(frame["side"].eq("SHORT").sum()),
                "net_return": float(frame["net_return"].sum()),
            },
            "evidence_cards": [card.model_dump(mode="json") for card in cards],
            "active_rules": [rule.model_dump(mode="json") for rule in active_rules],
            "eligible_memories": [memory.model_dump(mode="json") for memory in memories],
            "rejected_edit_buffer": [],
            "condition_values": CONDITION_VALUES,
        }
        messages = proposal_messages(observation)
        prompt_hash = _hash(messages)
        features, signals = _provenance(frame)
        audit_context = PromptAuditContext(
            call_id=f"proposal-{episode_id}",
            call_kind="PROPOSAL",
            stage=stage,
            protocol_scope=protocol_scope,
            cutoff_utc=cutoff,
            fold_id=fold_id,
            source_episode_id=episode_id,
            source_episode_cutoff_utc=cutoff,
            source_episode_outcome_max_utc=cutoff,
            candidate_eligible_after_utc=None,
            opportunity_decision_time=None,
            active_policy_activates_at_utc=None,
            regime_mapping_id="fixed-v2",
            features=features,
            signals=signals,
            evidence_cards=cards,
            memories=memories,
            eligible_memory_ids={memory.memory_id for memory in memories},
            memory_variant=str(self.memory.memory_variant),
            prompt_hash=prompt_hash,
        )
        self.auditor.audit_prompt(audit_context)
        allowed_ids = {
            "source_episode_id": episode_id,
            "evidence_ids": [card.evidence_id for card in cards],
            "memory_ids": [memory.memory_id for memory in memories],
            "rule_ids": [rule.rule_id for rule in active_rules],
        }
        call = self.caller.call(
            role="proposal",
            messages=messages,
            response_model=ProposalOutput,
            allowed_ids=allowed_ids,
        )
        proposal = call.value
        if proposal is None:
            status = (
                "transport_error" if call.status == "transport_error" else "schema_failure"
            )
            return EpisodeProposalResult(
                status=status,
                candidate=None,
                proposal=None,
                evidence_ids=allowed_ids["evidence_ids"],
                memory_ids=allowed_ids["memory_ids"],
                call_status=call.status,
            )
        try:
            if proposal.source_episode_id != episode_id:
                raise ValueError("source episode ID changed")
            validate_proposal_references(
                proposal,
                evidence_ids=set(allowed_ids["evidence_ids"]),
                memory_ids=set(allowed_ids["memory_ids"]),
                active_rule_ids=set(allowed_ids["rule_ids"]),
            )
        except ValueError:
            return EpisodeProposalResult(
                status="invalid_reference",
                candidate=None,
                proposal=proposal,
                evidence_ids=allowed_ids["evidence_ids"],
                memory_ids=allowed_ids["memory_ids"],
                call_status=call.status,
            )
        if proposal.decision == "NO_CHANGE":
            return EpisodeProposalResult(
                status="no_change",
                candidate=None,
                proposal=proposal,
                evidence_ids=allowed_ids["evidence_ids"],
                memory_ids=allowed_ids["memory_ids"],
                call_status=call.status,
            )
        if proposal.decision == "ADD_ALLOW_RULE" and len(active_rules) >= 3:
            return EpisodeProposalResult(
                status="invalid_reference",
                candidate=None,
                proposal=proposal,
                evidence_ids=allowed_ids["evidence_ids"],
                memory_ids=allowed_ids["memory_ids"],
                call_status=call.status,
            )
        identity = _hash(
            {
                "episode_id": episode_id,
                "proposal": proposal.model_dump(mode="json"),
                "protocol_hash": self.protocol_hash,
            }
        )
        candidate = ShadowCandidate(
            candidate_id=f"candidate-{identity[:24]}",
            decision=proposal.decision,
            source_episode_id=episode_id,
            source_fold_id=fold_id,
            source_episode_cutoff_utc=cutoff,
            eligible_after_utc=cutoff + timedelta(microseconds=1),
            proposed_rule=proposal.proposed_rule,
            target_rule_id=proposal.target_rule_id,
        )
        return EpisodeProposalResult(
            status="candidate_opened",
            candidate=candidate,
            proposal=proposal,
            evidence_ids=allowed_ids["evidence_ids"],
            memory_ids=allowed_ids["memory_ids"],
            call_status=call.status,
        )

    def close_candidate(
        self,
        candidate: ShadowCandidate,
        opportunities: pd.DataFrame,
        *,
        active_rules: Sequence[ActiveAllowRule],
        episode_number: int,
        force_close: bool = False,
    ) -> CandidateClosure:
        evaluation = evaluate_shadow(
            candidate, opportunities, active_rules=list(active_rules)
        )
        if (
            evaluation.decision == "INCONCLUSIVE"
            and evaluation.shadow_opportunities < 60
            and not force_close
        ):
            return CandidateClosure(
                status="still_open",
                evaluation=evaluation,
                reflection=None,
                new_active_rule=None,
                removed_rule_id=None,
                episodic_memory_id=None,
            )
        pool = opportunities.loc[
            opportunities["opportunity_id"].isin(evaluation.evaluated_opportunity_ids)
        ].copy()
        if pool.empty or evaluation.shadow_cutoff_utc is None:
            return CandidateClosure(
                status="closed",
                evaluation=evaluation,
                reflection=None,
                new_active_rule=None,
                removed_rule_id=None,
                episodic_memory_id=None,
            )
        for column in (
            "decision_time",
            "feature_available_time",
            "entry_time",
            "outcome_available_time",
        ):
            pool[column] = pd.to_datetime(pool[column], utc=True)
        stage = str(_single_value(pool, "stage"))
        fold_id = int(_single_value(pool, "fold_id"))
        protocol_scope = STAGE_SCOPES[stage]
        cards = _evidence_cards(pool, namespace=f"evaluation:{candidate.candidate_id}")
        evaluation_payload = evaluation.model_dump(mode="json")
        evaluation_payload["evidence_ids"] = [card.evidence_id for card in cards]
        messages = reflection_messages(
            candidate.model_dump(mode="json"), evaluation_payload
        )
        features, signals = _provenance(pool)
        prompt_hash = _hash(messages)
        audit_context = PromptAuditContext(
            call_id=f"reflection-{candidate.candidate_id}",
            call_kind="REFLECTION",
            stage=stage,
            protocol_scope=protocol_scope,
            cutoff_utc=evaluation.shadow_cutoff_utc,
            fold_id=fold_id,
            source_episode_id=candidate.source_episode_id,
            source_episode_cutoff_utc=candidate.source_episode_cutoff_utc,
            source_episode_outcome_max_utc=candidate.source_episode_cutoff_utc,
            candidate_eligible_after_utc=candidate.eligible_after_utc,
            opportunity_decision_time=None,
            active_policy_activates_at_utc=None,
            regime_mapping_id="fixed-v2",
            features=features,
            signals=signals,
            evidence_cards=cards,
            memories=[],
            eligible_memory_ids=set(),
            memory_variant=str(self.memory.memory_variant),
            prompt_hash=prompt_hash,
        )
        self.auditor.audit_prompt(audit_context)
        allowed_ids = {
            "candidate_id": candidate.candidate_id,
            "evaluator_decision": evaluation.decision,
            "evidence_ids": [card.evidence_id for card in cards],
        }
        call = self.caller.call(
            role="reflection",
            messages=messages,
            response_model=ReflectionOutput,
            allowed_ids=allowed_ids,
        )
        reflection = call.value
        if reflection is not None and (
            reflection.candidate_id != candidate.candidate_id
            or reflection.evaluator_decision != evaluation.decision
            or not set(reflection.evidence_ids).issubset(set(allowed_ids["evidence_ids"]))
        ):
            reflection = None

        new_rule: ActiveAllowRule | None = None
        removed_rule_id: str | None = None
        if evaluation.decision == "PROMOTE":
            activation = evaluation.shadow_cutoff_utc + timedelta(microseconds=1)
            if candidate.decision == "ADD_ALLOW_RULE":
                assert candidate.proposed_rule is not None
                rule_identity = _hash(
                    {
                        "candidate_id": candidate.candidate_id,
                        "evaluation": evaluation.model_dump(mode="json"),
                    }
                )
                new_rule = ActiveAllowRule(
                    rule_id=f"rule-{rule_identity[:24]}",
                    source_candidate_id=candidate.candidate_id,
                    source_fold_id=candidate.source_fold_id,
                    activates_at_utc=activation,
                    deactivates_at_utc=None,
                    rule=candidate.proposed_rule,
                )
            else:
                removed_rule_id = candidate.target_rule_id

        evaluation_id = f"evaluation-{_hash(evaluation.model_dump(mode='json'))[:24]}"
        status = {
            "PROMOTE": "SUPPORTED",
            "REJECT": "REJECTED",
            "INCONCLUSIVE": "INCONCLUSIVE",
        }[evaluation.decision]
        lesson = (
            reflection.lesson
            if reflection is not None and reflection.lesson is not None
            else f"Candidate {candidate.candidate_id} ended as {evaluation.decision}."
        )
        memory_id = f"episodic-{_hash({'candidate': candidate.candidate_id, 'evaluation': evaluation_id})[:24]}"
        memory_card = MemoryCard(
            memory_id=memory_id,
            source_stage=stage,
            protocol_scope=protocol_scope,
            memory_type="EPISODIC",
            created_at_utc=evaluation.shadow_cutoff_utc,
            max_support_outcome_time=evaluation.shadow_cutoff_utc,
            lesson=lesson,
            evidence_status=status,
            tags=self._candidate_tags(candidate),
            source_evaluation_ids=[evaluation_id],
            expires_at_utc=None,
            expires_after_episode=None,
        )
        self.memory.store(memory_card)
        if (
            evaluation.decision == "PROMOTE"
            and reflection is not None
            and reflection.memory_recommendation == "PROPOSE_SEMANTIC"
            and reflection.lesson is not None
        ):
            self.memory.store_semantic_support(
                SemanticSupport(
                    evaluation_id=evaluation_id,
                    candidate_id=candidate.candidate_id,
                    normalized_policy_key=self._candidate_policy_key(candidate),
                    source_stage=stage,
                    protocol_scope=protocol_scope,
                    fold_id=fold_id,
                    shadow_start_utc=min(evaluation.evaluated_decision_times),
                    shadow_end_utc=evaluation.shadow_cutoff_utc,
                    max_support_outcome_time=evaluation.shadow_cutoff_utc,
                    lesson=reflection.lesson,
                    tags=self._candidate_tags(candidate),
                    evaluator_decision="PROMOTE",
                )
            )
            self.memory.consolidate(
                now_utc=evaluation.shadow_cutoff_utc,
                current_episode_number=episode_number,
            )
        return CandidateClosure(
            status="closed",
            evaluation=evaluation,
            reflection=reflection,
            new_active_rule=new_rule,
            removed_rule_id=removed_rule_id,
            episodic_memory_id=memory_id,
        )

    @staticmethod
    def _candidate_tags(candidate: ShadowCandidate) -> list[str]:
        if candidate.proposed_rule is None:
            return [f"target_rule:{candidate.target_rule_id}"]
        return [
            f"{predicate.field}:{predicate.value}"
            for predicate in candidate.proposed_rule.predicates
        ]

    @staticmethod
    def _candidate_policy_key(candidate: ShadowCandidate) -> str:
        if candidate.proposed_rule is None:
            return f"REMOVE_ALLOW_RULE|{candidate.target_rule_id}"
        predicates = sorted(
            f"{predicate.field}={predicate.value}"
            for predicate in candidate.proposed_rule.predicates
        )
        return "ALLOW_REENTRY|" + "|".join(predicates)


__all__ = [
    "CandidateClosure",
    "EpisodeProposalResult",
    "ReflectionOrchestrator",
]
