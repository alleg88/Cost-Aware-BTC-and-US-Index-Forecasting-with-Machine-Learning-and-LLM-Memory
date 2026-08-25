"""Causal DeepSeek proposal, shadow, policy, and memory transitions for v3."""
from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from itertools import combinations
from typing import Annotated, Any, Literal, Sequence

import pandas as pd
from pydantic import Field

from reflection_agent.v3.contracts import (
    AllowRule,
    EvidenceCard,
    MemoryCard,
    PolicyChoice,
    ProposalChoiceOutput,
    ProposalOutput,
    ReflectionChoiceOutput,
    ReflectionOutput,
    StrictModel,
    validate_proposal_references,
)
from reflection_agent.v3.evaluator import GateDecision, ShadowCandidate, evaluate_shadow
from reflection_agent.v3.leakage import (
    PROTOCOL_SCOPE,
    FeatureProvenance,
    LeakageAuditor,
    PromptAuditContext,
    SignalProvenance,
)
from reflection_agent.v3.memory import SemanticSupport
from reflection_agent.v3.policy import ActiveAllowRule
from reflection_agent.v3.prompts import proposal_messages, reflection_messages


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


def _prompt_text(messages: Sequence[dict[str, str]]) -> str:
    return "\n\n".join(message["content"] for message in messages)


def _single_value(frame: pd.DataFrame, column: str) -> Any:
    values = frame[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"episode must contain one {column}")
    return values.iloc[0]


def _mode_tag(frame: pd.DataFrame, column: str, prefix: str) -> str | None:
    if column not in frame or frame[column].dropna().empty:
        return None
    return f"{prefix}:{frame[column].dropna().astype(str).mode().iloc[0]}"


def build_evidence_cards(
    frame: pd.DataFrame,
    *,
    namespace: str,
    fold_id_override: int | None = None,
) -> list[EvidenceCard]:
    """Aggregate only resolved economics into at most four opaque cards."""

    if frame.empty:
        raise ValueError("cannot summarize an empty evidence frame")
    required = {
        "opportunity_id",
        "stage",
        "source_role",
        "fold_id",
        "source_artifact_hash",
        "decision_time",
        "outcome_available_time",
        "route",
        "side",
        "gross_return",
        "net_return",
        "round_trip_cost",
        "exit_reason",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"evidence frame lacks columns: {missing}")
    current = frame.copy()
    current["decision_time"] = pd.to_datetime(current["decision_time"], utc=True)
    current["outcome_available_time"] = pd.to_datetime(
        current["outcome_available_time"], utc=True
    )
    if not current["decision_time"].lt(current["outcome_available_time"]).all():
        raise ValueError("unresolved evidence entered aggregation")
    cards: list[EvidenceCard] = []
    for (route, side), group in current.groupby(["route", "side"], sort=True):
        source_hashes = sorted(set(group["source_artifact_hash"].astype(str)))
        source_hash = source_hashes[0] if len(source_hashes) == 1 else _hash(source_hashes)
        identity = _hash(
            {
                "namespace": namespace,
                "opportunity_ids": sorted(group["opportunity_id"].astype(str)),
            }
        )
        outcome_counts = group["exit_reason"].astype(str).value_counts()
        tags = [
            f"route:{route}",
            f"side:{side}",
            f"tp_count:{int(outcome_counts.get('TAKE_PROFIT', 0))}",
            f"sl_count:{int(outcome_counts.get('STOP_LOSS', 0))}",
            f"timeout_count:{int(outcome_counts.get('TIMEOUT', 0))}",
        ]
        for column, prefix in (
            ("confidence_tier", "tier"),
            ("signal_run_bucket", "run"),
            ("vol_regime", "vol"),
            ("trend_regime", "trend"),
            ("funding_regime", "funding"),
            ("oi_regime", "oi"),
        ):
            tag = _mode_tag(group, column, prefix)
            if tag is not None:
                tags.append(tag)
        fold_id = (
            int(fold_id_override)
            if fold_id_override is not None
            else int(_single_value(group, "fold_id"))
        )
        cards.append(
            EvidenceCard(
                evidence_id=f"ev_{identity[:24]}",
                stage=str(_single_value(group, "stage")),
                source_role=str(_single_value(group, "source_role")),
                decision_time=group["decision_time"].max().to_pydatetime(),
                outcome_available_time=group[
                    "outcome_available_time"
                ].max().to_pydatetime(),
                fold_id=fold_id,
                row_key=f"agg_{identity[:24]}",
                source_artifact_hash=source_hash,
                route=str(route),
                side=str(side),
                count=len(group),
                gross_return=float(group["gross_return"].sum()),
                cost_return=float(group["round_trip_cost"].sum()),
                net_return=float(group["net_return"].sum()),
                tags=tags[:16],
            )
        )
    if not 1 <= len(cards) <= 4:
        raise ValueError("resolved evidence must produce one to four cards")
    return cards


def _tag_count(card: EvidenceCard, prefix: str) -> int:
    marker = f"{prefix}:"
    for tag in card.tags:
        if tag.startswith(marker):
            return int(tag[len(marker) :])
    return 0


def _compact_card(card: EvidenceCard) -> dict[str, object]:
    visible_tags = [
        tag
        for tag in card.tags
        if not tag.startswith(("tp_count:", "sl_count:", "timeout_count:"))
    ]
    return {
        "evidence_id": card.evidence_id,
        "route": card.route,
        "side": card.side,
        "count": card.count,
        "tp_count": _tag_count(card, "tp_count"),
        "sl_count": _tag_count(card, "sl_count"),
        "timeout_count": _tag_count(card, "timeout_count"),
        "gross_bps": round(card.gross_return * 10_000.0, 6),
        "cost_bps": round(card.cost_return * 10_000.0, 6),
        "net_bps": round(card.net_return * 10_000.0, 6),
        "tags": visible_tags,
    }


def _compact_memory(card: MemoryCard) -> dict[str, object]:
    return {
        "memory_id": card.memory_id,
        "memory_type": card.memory_type,
        "evidence_status": card.evidence_status,
        "lesson": card.lesson,
        "tags": card.tags,
    }


def _compact_rule(active: ActiveAllowRule) -> dict[str, object]:
    return {
        "rule_id": active.rule_id,
        "action": active.rule.action,
        "predicates": [
            predicate.model_dump(mode="json") for predicate in active.rule.predicates
        ],
    }


def _card_tag(card: EvidenceCard, prefix: str) -> str | None:
    marker = f"{prefix}:"
    return next(
        (tag[len(marker) :] for tag in card.tags if tag.startswith(marker)),
        None,
    )


def build_policy_choices(
    cards: Sequence[EvidenceCard],
    active_rules: Sequence[ActiveAllowRule],
) -> list[PolicyChoice]:
    """Enumerate the complete bounded policy menu owned by the host."""

    choices = [
        PolicyChoice(
            choice_index=0,
            decision="NO_CHANGE",
            proposed_rule=None,
            target_rule_id=None,
        )
    ]
    field_tags = (
        ("confidence_tier", "tier"),
        ("signal_run_bucket", "run"),
        ("vol_regime", "vol"),
        ("trend_regime", "trend"),
        ("funding_regime", "funding"),
        ("oi_regime", "oi"),
    )
    seen_rules: set[str] = set()
    candidate_cards = sorted(
        (card for card in cards if card.route == "COVERAGE_CANDIDATE"),
        key=lambda card: (card.side, card.evidence_id),
    )
    for card in candidate_cards:
        if sum(rule.rule.side == card.side for rule in active_rules) >= 3:
            continue
        extras = [
            {"field": field, "operator": "EQ", "value": value}
            for field, prefix in field_tags
            if (value := _card_tag(card, prefix)) is not None
        ]
        for size in range(3):
            for selected in combinations(extras, size):
                rule = AllowRule(
                    action="ALLOW_CANDIDATE",
                    predicates=[
                        {"field": "side", "operator": "EQ", "value": card.side},
                        *selected,
                    ],
                )
                key = _canonical_json(rule.model_dump(mode="json"))
                if key in seen_rules:
                    continue
                seen_rules.add(key)
                choices.append(
                    PolicyChoice(
                        choice_index=len(choices),
                        decision="ADD_ALLOW_RULE",
                        proposed_rule=rule,
                        target_rule_id=None,
                    )
                )
    for active in sorted(active_rules, key=lambda rule: rule.rule_id):
        choices.append(
            PolicyChoice(
                choice_index=len(choices),
                decision="REMOVE_ALLOW_RULE",
                proposed_rule=None,
                target_rule_id=active.rule_id,
            )
        )
    if len(choices) > 64:
        raise AssertionError("host policy choice menu exceeded its frozen bound")
    return choices


def compact_observation(
    *,
    source_episode_id: str,
    cards: Sequence[EvidenceCard],
    memories: Sequence[MemoryCard],
    active_rules: Sequence[ActiveAllowRule],
    policy_choices: Sequence[PolicyChoice] = (),
) -> dict[str, object]:
    """Build the only anonymous JSON payload permitted in a proposal prompt."""

    return {
        "active_rules": [_compact_rule(rule) for rule in active_rules],
        "choice_menu": [
            choice.model_dump(mode="json") for choice in policy_choices
        ],
        "evidence_cards": [
            {"evidence_index": index, **_compact_card(card)}
            for index, card in enumerate(cards)
        ],
        "memory_cards": [
            {"memory_index": index, **_compact_memory(card)}
            for index, card in enumerate(memories)
        ],
        "schema_version": "3.0",
        "source_episode_id": source_episode_id,
    }


def compile_proposal_choice(
    output: ProposalChoiceOutput,
    *,
    policy_choices: Sequence[PolicyChoice],
    cards: Sequence[EvidenceCard],
    memories: Sequence[MemoryCard],
    source_episode_id: str,
) -> ProposalOutput:
    if output.choice_index >= len(policy_choices):
        raise ValueError("unknown policy choice index")
    if any(index >= len(cards) for index in output.evidence_indices):
        raise ValueError("unknown evidence index")
    if any(index >= len(memories) for index in output.memory_indices):
        raise ValueError("unknown memory index")
    choice = policy_choices[output.choice_index]
    evidence_ids = [cards[index].evidence_id for index in output.evidence_indices]
    memory_ids = [memories[index].memory_id for index in output.memory_indices]
    if choice.decision == "NO_CHANGE":
        return ProposalOutput(
            source_episode_id=source_episode_id,
            decision="NO_CHANGE",
            diagnosis_code="INSUFFICIENT_EVIDENCE",
            evidence_ids=evidence_ids,
            memory_ids_used=memory_ids,
            proposed_rule=None,
            target_rule_id=None,
            hypothesis=None,
            falsifiers=[],
            confidence="LOW",
        )
    return ProposalOutput(
        source_episode_id=source_episode_id,
        decision=choice.decision,
        diagnosis_code=(
            "REGIME_SPECIFIC_EDGE"
            if choice.decision == "ADD_ALLOW_RULE"
            else "STALE_ACTIVE_RULE"
        ),
        evidence_ids=evidence_ids,
        memory_ids_used=memory_ids,
        proposed_rule=choice.proposed_rule,
        target_rule_id=choice.target_rule_id,
        hypothesis="The selected host-owned policy choice requires a strictly later shadow.",
        falsifiers=["Any deterministic future-shadow gate may reject this policy choice."],
        confidence="LOW",
    )


def _host_failure_code(evaluation: GateDecision) -> str:
    if evaluation.decision == "PROMOTE":
        return "NONE"
    joined = "|".join(evaluation.failure_codes)
    if any(token in joined for token in ("MATCHING", "SUBBLOCK", "TRADE_COUNT")):
        return "TOO_FEW_TRIGGERS"
    if "CONCENTRATION" in joined or "SPAN_TWO" in joined:
        return "OVERFIT_CONCENTRATION"
    if "TARGET_SIDE" in joined:
        return "SIDE_IMBALANCE"
    if "TOTAL_NET" in joined:
        return "COST_DRAG"
    if any(token in joined for token in ("COLLID", "BASE_IMMUTABLE", "TRADE_SHAPE", "REMOVAL")):
        return "RULE_COLLISION"
    return "REGIME_MISMATCH"


def compile_reflection_choice(
    output: ReflectionChoiceOutput,
    *,
    candidate: ShadowCandidate,
    evaluation: GateDecision,
    cards: Sequence[EvidenceCard],
) -> ReflectionOutput:
    if any(index >= len(cards) for index in output.evidence_indices):
        raise ValueError("unknown reflection evidence index")
    evidence_ids = [cards[index].evidence_id for index in output.evidence_indices]
    recommendation = (
        "DO_NOT_GENERALIZE",
        "STORE_EPISODE",
        "PROPOSE_SEMANTIC",
    )[output.memory_action_index]
    if evaluation.decision != "PROMOTE" and recommendation == "PROPOSE_SEMANTIC":
        recommendation = "STORE_EPISODE"
    return ReflectionOutput(
        candidate_id=candidate.candidate_id,
        evaluator_decision=evaluation.decision,
        evidence_ids=evidence_ids,
        failure_code=_host_failure_code(evaluation),
        lesson=(
            "The host-owned policy choice passed every deterministic future-shadow gate."
            if evaluation.decision == "PROMOTE"
            else None
        ),
        invalidation_conditions=evaluation.failure_codes[:4],
        memory_recommendation=recommendation,
    )


def _episode_tags(frame: pd.DataFrame) -> tuple[str, ...]:
    tags: set[str] = set()
    for column, prefix in (
        ("side", "side"),
        ("confidence_tier", "tier"),
        ("signal_run_bucket", "run"),
        ("vol_regime", "vol"),
        ("trend_regime", "trend"),
        ("funding_regime", "funding"),
        ("oi_regime", "oi"),
    ):
        if column in frame:
            tags.update(
                f"{prefix}:{value}" for value in frame[column].dropna().astype(str)
            )
    return tuple(sorted(tags))


def _provenance(
    frame: pd.DataFrame,
) -> tuple[list[FeatureProvenance], list[SignalProvenance]]:
    features = [
        FeatureProvenance(
            feature_name="registered_categorical_context",
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


class CandidateClosure(StrictModel):
    status: Literal["closed", "still_open"]
    candidate_id: str
    evaluation: GateDecision
    reflection: ReflectionOutput | None
    new_active_rule: ActiveAllowRule | None
    removed_rule_id: str | None
    episodic_memory_id: str | None
    active_rules_after: list[ActiveAllowRule]


class EpisodeProposalResult(StrictModel):
    status: Literal[
        "candidate_opened",
        "no_change",
        "invalid_reference",
        "schema_failure",
        "transport_error",
        "static_control",
        "insufficient_episode",
    ]
    candidate: ShadowCandidate | None
    proposal: ProposalOutput | None
    closures: list[CandidateClosure]
    active_rules: list[ActiveAllowRule]
    open_candidates: list[ShadowCandidate]
    evidence_ids: list[str]
    memory_ids: list[str]
    call_status: str | None


class ReflectionOrchestrator:
    def __init__(
        self,
        *,
        caller: Any,
        auditor: LeakageAuditor,
        memory: Any,
        protocol_hash: str,
    ) -> None:
        self.caller = caller
        self.auditor = auditor
        self.memory = memory
        self.protocol_hash = protocol_hash
        self.transition_log: list[str] = []

    @staticmethod
    def transport_hashes(
        config: Any,
        *,
        role: str,
        messages: Sequence[dict[str, str]],
        response_model: type[StrictModel],
        allowed_ids: dict[str, Any],
    ) -> tuple[str, str]:
        schema_hash = _hash(response_model.model_json_schema())
        request_payload = {
            "model": config.model,
            "role": role,
            "messages": list(messages),
            "schema_hash": schema_hash,
            "think": config.think,
            "stream": config.stream,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.num_predict,
            },
            "allowed_ids": allowed_ids,
        }
        return schema_hash, _hash(request_payload)

    def _audit_context(
        self,
        *,
        call_id: str,
        call_kind: str,
        stage: str,
        fold_id: int,
        cutoff: pd.Timestamp,
        source_episode_id: str,
        source_episode_cutoff: pd.Timestamp,
        source_episode_outcome_max: pd.Timestamp,
        candidate_eligible_after: pd.Timestamp | None,
        frame: pd.DataFrame,
        cards: Sequence[EvidenceCard],
        memories: Sequence[MemoryCard],
        messages: Sequence[dict[str, str]],
        schema_hash: str,
        request_hash: str,
    ) -> PromptAuditContext:
        features, signals = _provenance(frame)
        prompt_text = _prompt_text(messages)
        return PromptAuditContext(
            call_id=call_id,
            call_kind=call_kind,
            stage=stage,
            protocol_scope=PROTOCOL_SCOPE,
            cutoff_utc=cutoff.to_pydatetime(),
            fold_id=fold_id,
            source_episode_id=source_episode_id,
            source_episode_cutoff_utc=source_episode_cutoff.to_pydatetime(),
            source_episode_outcome_max_utc=source_episode_outcome_max.to_pydatetime(),
            candidate_eligible_after_utc=(
                candidate_eligible_after.to_pydatetime()
                if candidate_eligible_after is not None
                else None
            ),
            regime_mapping_id="fixed-v2",
            features=features,
            signals=signals,
            evidence_cards=list(cards),
            memories=list(memories),
            eligible_memory_ids={memory.memory_id for memory in memories},
            memory_variant=str(self.memory.memory_variant),
            prompt_text=prompt_text,
            prompt_hash=hashlib.sha256(prompt_text.encode("utf-8")).hexdigest(),
            schema_hash=schema_hash,
            request_hash=request_hash,
        )

    def _call(
        self,
        *,
        role: str,
        messages: Sequence[dict[str, str]],
        response_model: type[StrictModel],
        allowed_ids: dict[str, Any],
    ):
        schema_hash, request_hash = self.transport_hashes(
            self.caller.config,
            role=role,
            messages=messages,
            response_model=response_model,
            allowed_ids=allowed_ids,
        )
        call = self.caller.call(
            role=role,
            messages=messages,
            response_model=response_model,
            allowed_ids=allowed_ids,
        )
        if call.schema_hash != schema_hash or call.request_hash != request_hash:
            raise AssertionError("transport hashes drifted after prompt audit")
        self.memory.append_call_audit(
            f"call_{request_hash[:24]}",
            {
                "role": role,
                "status": call.status,
                "schema_hash": call.schema_hash,
                "request_hash": call.request_hash,
                "response_hash": call.response_hash,
                "attempts": call.attempts,
                "errors": list(call.errors),
            },
        )
        return call, schema_hash, request_hash

    def _proposal(
        self,
        episode: pd.DataFrame,
        *,
        episode_id: str,
        episode_number: int,
        active_rules: Sequence[ActiveAllowRule],
    ) -> tuple[str, ShadowCandidate | None, ProposalOutput | None, list[str], list[str], str | None]:
        frame = episode.copy().reset_index(drop=True)
        for column in (
            "decision_time",
            "feature_available_time",
            "entry_time",
            "outcome_available_time",
        ):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        stage = str(_single_value(frame, "stage"))
        fold_id = int(_single_value(frame, "fold_id"))
        expected_role = "OOF_TEST" if stage == "development" else "FROZEN_EXACT"
        if str(_single_value(frame, "source_role")) != expected_role:
            raise ValueError("episode source role does not match its stage")
        cutoff = frame["outcome_available_time"].max()
        cards = build_evidence_cards(frame, namespace=episode_id)
        self.transition_log.append("retrieve_pre_cutoff_memory")
        memories = self.memory.retrieve(
            cutoff_utc=cutoff.to_pydatetime(),
            stage=stage,
            tags=_episode_tags(frame),
            episode_number=episode_number,
            episode_id=episode_id,
            maximum=4,
        )
        policy_choices = build_policy_choices(cards, active_rules)
        observation = compact_observation(
            source_episode_id=episode_id,
            cards=cards,
            memories=memories,
            active_rules=active_rules,
            policy_choices=policy_choices,
        )
        messages = proposal_messages(observation)
        allowed_ids = {
            "choice_indices": [choice.choice_index for choice in policy_choices],
            "evidence_indices": list(range(len(cards))),
            "memory_indices": list(range(len(memories))),
        }
        schema_hash, request_hash = self.transport_hashes(
            self.caller.config,
            role="proposal",
            messages=messages,
            response_model=ProposalChoiceOutput,
            allowed_ids=allowed_ids,
        )
        audit_context = self._audit_context(
            call_id=f"proposal_{_hash(episode_id)[:24]}",
            call_kind="PROPOSAL",
            stage=stage,
            fold_id=fold_id,
            cutoff=cutoff,
            source_episode_id=episode_id,
            source_episode_cutoff=cutoff,
            source_episode_outcome_max=cutoff,
            candidate_eligible_after=None,
            frame=frame,
            cards=cards,
            memories=memories,
            messages=messages,
            schema_hash=schema_hash,
            request_hash=request_hash,
        )
        self.transition_log.append("audit_proposal_prompt")
        self.auditor.audit_prompt(audit_context)
        self.transition_log.append("call_proposal_model")
        call, _, _ = self._call(
            role="proposal",
            messages=messages,
            response_model=ProposalChoiceOutput,
            allowed_ids=allowed_ids,
        )
        proposal_choice = call.value
        if proposal_choice is None:
            status = "transport_error" if call.status == "transport_error" else "schema_failure"
            return (
                status,
                None,
                None,
                [card.evidence_id for card in cards],
                [memory.memory_id for memory in memories],
                call.status,
            )
        try:
            proposal = compile_proposal_choice(
                proposal_choice,
                policy_choices=policy_choices,
                cards=cards,
                memories=memories,
                source_episode_id=episode_id,
            )
            validate_proposal_references(
                proposal,
                evidence_ids={card.evidence_id for card in cards},
                memory_ids={memory.memory_id for memory in memories},
                active_rule_ids={rule.rule_id for rule in active_rules},
                source_sides={card.side for card in cards},
            )
            if proposal.decision == "ADD_ALLOW_RULE":
                assert proposal.proposed_rule is not None
                active_on_side = sum(
                    rule.rule.side == proposal.proposed_rule.side for rule in active_rules
                )
                if active_on_side >= 3:
                    raise ValueError("target side already has three active rules")
        except ValueError:
            return (
                "invalid_reference",
                None,
                None,
                [card.evidence_id for card in cards],
                [memory.memory_id for memory in memories],
                call.status,
            )
        if proposal.decision == "NO_CHANGE":
            return (
                "no_change",
                None,
                proposal,
                [card.evidence_id for card in cards],
                [memory.memory_id for memory in memories],
                call.status,
            )
        identity = _hash(
            {
                "episode_id": episode_id,
                "proposal": proposal.model_dump(mode="json"),
                "protocol_hash": self.protocol_hash,
            }
        )
        candidate = ShadowCandidate(
            candidate_id=f"cand_{identity[:24]}",
            decision=proposal.decision,
            source_episode_id=episode_id,
            source_stage=stage,
            source_fold_id=fold_id,
            source_episode_cutoff_utc=cutoff.to_pydatetime(),
            eligible_after_utc=(cutoff + pd.Timedelta(microseconds=1)).to_pydatetime(),
            proposed_rule=proposal.proposed_rule,
            target_rule_id=proposal.target_rule_id,
        )
        self.transition_log.append("open_strictly_later_shadow")
        return (
            "candidate_opened",
            candidate,
            proposal,
            [card.evidence_id for card in cards],
            [memory.memory_id for memory in memories],
            call.status,
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
        return "ALLOW_CANDIDATE|" + "|".join(predicates)

    @staticmethod
    def _compact_metrics(metrics: Any) -> dict[str, object]:
        return {
            "trades": metrics.trades,
            "net_bps": round(metrics.net_return * 10_000.0, 6),
            "long_net_bps": round(metrics.long_net_return * 10_000.0, 6),
            "short_net_bps": round(metrics.short_net_return * 10_000.0, 6),
            "sortino": round(metrics.sortino, 6),
            "max_drawdown": round(metrics.max_drawdown, 8),
        }

    def _reflect(
        self,
        candidate: ShadowCandidate,
        evaluation: GateDecision,
        pool: pd.DataFrame,
    ) -> ReflectionOutput | None:
        if pool.empty or evaluation.shadow_cutoff_utc is None:
            return None
        cards = build_evidence_cards(
            pool,
            namespace=f"shadow_{candidate.candidate_id}",
            fold_id_override=candidate.source_fold_id,
        )
        candidate_payload = {
            "candidate_id": candidate.candidate_id,
            "decision": candidate.decision,
            "proposed_rule": (
                candidate.proposed_rule.model_dump(mode="json")
                if candidate.proposed_rule is not None
                else None
            ),
            "target_rule_id": candidate.target_rule_id,
        }
        evaluation_payload = {
            "candidate_id": candidate.candidate_id,
            "decision": evaluation.decision,
            "evidence_cards": [
                {"evidence_index": index, **_compact_card(card)}
                for index, card in enumerate(cards)
            ],
            "matching_candidates": evaluation.matching_candidates,
            "total_coverage_candidates": evaluation.total_coverage_candidates,
            "triggered_trades": evaluation.triggered_trades,
            "gate_checks": [
                f"{name}:{'PASS' if passed else 'FAIL'}"
                for name, passed in sorted(evaluation.gate_results.items())
            ],
            "failure_codes": evaluation.failure_codes,
            "union_metrics": self._compact_metrics(evaluation.union_metrics),
            "control_metrics": self._compact_metrics(evaluation.control_metrics),
            "candidate_metrics": self._compact_metrics(evaluation.candidate_metrics),
        }
        messages = reflection_messages(candidate_payload, evaluation_payload)
        allowed_ids = {
            "evidence_indices": list(range(len(cards))),
            "memory_action_indices": [0, 1, 2],
        }
        schema_hash, request_hash = self.transport_hashes(
            self.caller.config,
            role="reflection",
            messages=messages,
            response_model=ReflectionChoiceOutput,
            allowed_ids=allowed_ids,
        )
        cutoff = pd.Timestamp(evaluation.shadow_cutoff_utc)
        audit_context = self._audit_context(
            call_id=f"reflection_{_hash(candidate.candidate_id)[:24]}",
            call_kind="REFLECTION",
            stage=candidate.source_stage,
            fold_id=candidate.source_fold_id,
            cutoff=cutoff,
            source_episode_id=candidate.source_episode_id,
            source_episode_cutoff=pd.Timestamp(candidate.source_episode_cutoff_utc),
            source_episode_outcome_max=pd.Timestamp(
                candidate.source_episode_cutoff_utc
            ),
            candidate_eligible_after=pd.Timestamp(candidate.eligible_after_utc),
            frame=pool,
            cards=cards,
            memories=[],
            messages=messages,
            schema_hash=schema_hash,
            request_hash=request_hash,
        )
        self.transition_log.append("reflect_future_shadow")
        self.auditor.audit_prompt(audit_context)
        call, _, _ = self._call(
            role="reflection",
            messages=messages,
            response_model=ReflectionChoiceOutput,
            allowed_ids=allowed_ids,
        )
        reflection_choice = call.value
        if reflection_choice is None:
            return None
        try:
            reflection = compile_reflection_choice(
                reflection_choice,
                candidate=candidate,
                evaluation=evaluation,
                cards=cards,
            )
        except ValueError:
            return None
        return reflection

    def _store_closure_memory(
        self,
        candidate: ShadowCandidate,
        evaluation: GateDecision,
        reflection: ReflectionOutput | None,
        *,
        episode_number: int,
    ) -> str | None:
        if evaluation.shadow_cutoff_utc is None:
            return None
        cutoff = evaluation.shadow_cutoff_utc
        evaluation_id = f"eval_{_hash(evaluation.model_dump(mode='json'))[:24]}"
        status = {
            "PROMOTE": "SUPPORTED",
            "REJECT": "REJECTED",
            "INCONCLUSIVE": "INCONCLUSIVE",
        }[evaluation.decision]
        lesson = (
            reflection.lesson
            if reflection is not None and reflection.lesson is not None
            else f"Bounded candidate ended with evaluator decision {evaluation.decision}."
        )
        memory_id = f"episodic_{_hash({'candidate': candidate.candidate_id, 'evaluation': evaluation_id})[:24]}"
        card = MemoryCard(
            memory_id=memory_id,
            source_stage=candidate.source_stage,
            protocol_scope=PROTOCOL_SCOPE,
            memory_type="EPISODIC",
            created_at_utc=cutoff,
            max_support_outcome_time=cutoff,
            lesson=lesson,
            evidence_status=status,
            tags=self._candidate_tags(candidate),
            source_evaluation_ids=[evaluation_id],
            expires_at_utc=cutoff + timedelta(days=180),
            expires_after_episode=episode_number + 6,
        )
        self.memory.store(card)
        if (
            evaluation.decision == "PROMOTE"
            and reflection is not None
            and reflection.memory_recommendation == "PROPOSE_SEMANTIC"
            and reflection.lesson is not None
            and evaluation.evaluated_decision_times
        ):
            self.memory.store_semantic_support(
                SemanticSupport(
                    evaluation_id=evaluation_id,
                    candidate_id=candidate.candidate_id,
                    normalized_policy_key=self._candidate_policy_key(candidate),
                    source_stage=candidate.source_stage,
                    protocol_scope=PROTOCOL_SCOPE,
                    fold_id=candidate.source_fold_id,
                    shadow_start_utc=min(evaluation.evaluated_decision_times),
                    shadow_end_utc=cutoff,
                    max_support_outcome_time=cutoff,
                    lesson=reflection.lesson,
                    tags=self._candidate_tags(candidate),
                    evaluator_decision="PROMOTE",
                )
            )
            self.memory.consolidate(
                now_utc=cutoff, current_episode_number=episode_number
            )
        return memory_id

    def close_eligible_candidates(
        self,
        candidates: Sequence[ShadowCandidate],
        opportunities: pd.DataFrame,
        *,
        active_rules: Sequence[ActiveAllowRule],
        episode_number: int,
        force_close: bool = False,
    ) -> list[CandidateClosure]:
        working_rules = list(active_rules)
        closures: list[CandidateClosure] = []
        ordered = sorted(
            candidates, key=lambda item: (item.eligible_after_utc, item.candidate_id)
        )
        current_stage = (
            str(opportunities["stage"].iloc[-1]) if len(opportunities) else None
        )
        for candidate in ordered:
            self.transition_log.append("close_future_shadow")
            evaluation = evaluate_shadow(
                candidate, opportunities, active_rules=working_rules
            )
            stage_ended = current_stage is not None and current_stage != candidate.source_stage
            should_close = bool(
                evaluation.decision != "INCONCLUSIVE"
                or evaluation.total_coverage_candidates >= 40
                or force_close
                or stage_ended
            )
            if not should_close:
                closures.append(
                    CandidateClosure(
                        status="still_open",
                        candidate_id=candidate.candidate_id,
                        evaluation=evaluation,
                        reflection=None,
                        new_active_rule=None,
                        removed_rule_id=None,
                        episodic_memory_id=None,
                        active_rules_after=list(working_rules),
                    )
                )
                continue
            pool = opportunities.loc[
                opportunities["opportunity_id"].isin(
                    evaluation.evaluated_opportunity_ids
                )
            ].copy()
            reflection = self._reflect(candidate, evaluation, pool)
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
                        rule_id=f"rule_{rule_identity[:24]}",
                        source_candidate_id=candidate.candidate_id,
                        source_stage=candidate.source_stage,
                        source_fold_id=candidate.source_fold_id,
                        activates_at_utc=activation,
                        rule=candidate.proposed_rule,
                    )
                    working_rules.append(new_rule)
                else:
                    removed_rule_id = candidate.target_rule_id
                    working_rules = [
                        rule
                        for rule in working_rules
                        if rule.rule_id != removed_rule_id
                    ]
            self.transition_log.append("apply_deterministic_policy")
            memory_id = self._store_closure_memory(
                candidate,
                evaluation,
                reflection,
                episode_number=episode_number,
            )
            self.transition_log.append("store_resolved_memory")
            closures.append(
                CandidateClosure(
                    status="closed",
                    candidate_id=candidate.candidate_id,
                    evaluation=evaluation,
                    reflection=reflection,
                    new_active_rule=new_rule,
                    removed_rule_id=removed_rule_id,
                    episodic_memory_id=memory_id,
                    active_rules_after=list(working_rules),
                )
            )
        return closures

    def run_episode(
        self,
        episode: pd.DataFrame,
        *,
        episode_id: str,
        episode_number: int,
        active_rules: Sequence[ActiveAllowRule],
        open_candidates: Sequence[ShadowCandidate],
        available_opportunities: pd.DataFrame,
    ) -> EpisodeProposalResult:
        if episode.empty:
            raise ValueError("cannot run an empty observation episode")
        frame = episode.copy()
        frame["outcome_available_time"] = pd.to_datetime(
            frame["outcome_available_time"], utc=True
        )
        cutoff = frame["outcome_available_time"].max()
        available = available_opportunities.copy()
        available["outcome_available_time"] = pd.to_datetime(
            available["outcome_available_time"], utc=True
        )
        available = available.loc[available["outcome_available_time"].le(cutoff)]
        closures = self.close_eligible_candidates(
            open_candidates,
            available,
            active_rules=active_rules,
            episode_number=episode_number,
        )
        working_rules = (
            list(closures[-1].active_rules_after) if closures else list(active_rules)
        )
        closed_ids = {
            closure.candidate_id
            for closure in closures
            if closure.status == "closed"
        }
        remaining = [
            candidate
            for candidate in open_candidates
            if candidate.candidate_id not in closed_ids
        ]
        if "episode_can_propose" in frame and not bool(
            frame["episode_can_propose"].all()
        ):
            return EpisodeProposalResult(
                status="insufficient_episode",
                candidate=None,
                proposal=None,
                closures=closures,
                active_rules=working_rules,
                open_candidates=remaining,
                evidence_ids=[],
                memory_ids=[],
                call_status=None,
            )
        if not getattr(self.memory, "uses_llm", True):
            return EpisodeProposalResult(
                status="static_control",
                candidate=None,
                proposal=None,
                closures=closures,
                active_rules=working_rules,
                open_candidates=remaining,
                evidence_ids=[],
                memory_ids=[],
                call_status=None,
            )
        status, candidate, proposal, evidence_ids, memory_ids, call_status = self._proposal(
            frame,
            episode_id=episode_id,
            episode_number=episode_number,
            active_rules=working_rules,
        )
        if candidate is not None:
            remaining.append(candidate)
        return EpisodeProposalResult(
            status=status,
            candidate=candidate,
            proposal=proposal,
            closures=closures,
            active_rules=working_rules,
            open_candidates=remaining,
            evidence_ids=evidence_ids,
            memory_ids=memory_ids,
            call_status=call_status,
        )


__all__ = [
    "CandidateClosure",
    "EpisodeProposalResult",
    "ReflectionOrchestrator",
    "build_evidence_cards",
    "compact_observation",
]
