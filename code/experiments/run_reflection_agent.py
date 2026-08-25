"""Preflight and resumable causal-window runner for the reflection agent."""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.build_reflection_cache import CONFIG_PATH, DEFAULT_OUTPUT
from reflection_agent.config import ProtocolConfig, load_config
from reflection_agent.contracts import (
    CandidateBatch,
    MarketContext,
    MemorySnippet,
    MODEL_IDS,
    CONDITION_FIELDS,
    PolicyRule,
    PolicyVersion,
    ProbabilityVector,
    ShadowState,
)
from reflection_agent.news import select_balanced_events
from reflection_agent.observation import build_observation
from reflection_agent.orchestrator import WeeklyOrchestrator
from reflection_agent.preflight import PreflightReport, run_preflight
from reflection_agent.prompts import actor_messages
from reflection_agent.search import validate_candidates
from reflection_agent.memory import MemoryManager
from reflection_agent.execution import (
    CONSENSUS_RULE,
    LSTM_RULE,
    evaluate_candidate_historical,
    evaluate_candidate_shadow,
    simulate_compiled_policy,
)
from reflection_agent.policy import compile_policy, condition_mask, resolve_edits
from reflection_agent.replay import (
    DisabledMemoryManager,
    REGISTERED_VARIANTS,
    observation_tags,
    shadow_window_ids,
    shuffle_time_eligible_memories,
)
from reflection_agent.store import AgentStore
from reflection_agent.transport import OllamaClientTransport, OllamaHttpTransport, StructuredCaller

DEFAULT_STATE = DEFAULT_OUTPUT / "agent_state.sqlite"
DEFAULT_PREFLIGHT = DEFAULT_OUTPUT / "preflight.json"
DEFAULT_SMOKE = DEFAULT_OUTPUT / "development_smoke.json"
DEFAULT_PIPELINE_SMOKE = DEFAULT_OUTPUT / "pipeline_smoke.json"


def _first_complete_monday(config: ProtocolConfig) -> pd.Timestamp:
    start = pd.Timestamp(config.development_start_utc)
    normalized = start.normalize()
    days = (7 - normalized.weekday()) % 7
    if days == 0 and start == normalized:
        return normalized
    return normalized + pd.Timedelta(days=days)


def build_weekly_observation(
    *,
    config: ProtocolConfig,
    cache_root: Path,
    window_offset: int = 0,
    news_mode: str = "bounded_text",
    retrieved_memories: Iterable[MemorySnippet] = (),
    active_policy_id: str = "policy-unanimity-consensus-v1",
    active_edit_ids: Iterable[str] = ("consensus-agreement",),
    active_rules: Iterable[PolicyRule] = (CONSENSUS_RULE,),
):
    panel = pd.read_parquet(cache_root / "frozen_probability_panel.parquet").set_index("timestamp").sort_index()
    context = pd.read_parquet(cache_root / "market_context.parquet").set_index("timestamp").sort_index()
    events = pd.read_parquet(cache_root / "news_events.parquet")
    start = _first_complete_monday(config) + pd.Timedelta(weeks=window_offset)
    end = start + pd.Timedelta(weeks=1)
    if end > pd.Timestamp(config.development_end_utc):
        raise ValueError("requested window enters the sealed or unavailable interval")
    weekly = panel.loc[(panel.index >= start) & (panel.index < end)]
    weekly_context = context.loc[(context.index >= start) & (context.index < end)]
    if weekly.empty or weekly_context.empty:
        raise ValueError(f"missing frozen data for {start.isoformat()} to {end.isoformat()}")
    cutoff = (end - pd.Timedelta(microseconds=1)).to_pydatetime()
    probabilities = {
        model_id: ProbabilityVector(
            short=float(weekly[f"{model_id}_p_short"].mean()),
            flat=float(weekly[f"{model_id}_p_flat"].mean()),
            long=float(weekly[f"{model_id}_p_long"].mean()),
        )
        for model_id in MODEL_IDS
    }
    last = weekly_context.iloc[-1]
    market = MarketContext(
        vol_regime=str(last["vol_regime"]),
        trend_regime=str(last["trend_regime"]),
        realized_volatility=float(last["realized_volatility"]),
        recent_return=float(last["recent_return"]),
    )
    if news_mode not in {"none", "aggregate", "bounded_text"}:
        raise ValueError("news_mode must be none, aggregate, or bounded_text")
    news_events = events.iloc[0:0] if news_mode == "none" else events
    news = select_balanced_events(
        news_events,
        cutoff_utc=cutoff,
        lookback=timedelta(days=7),
        max_items=config.news.max_items,
        max_per_source_family=config.news.max_items_per_source_family,
    )
    if news_mode == "aggregate":
        news = news.model_copy(update={"top_items": []})
    rules = tuple(active_rules)
    weekly_probabilities = {
        model_id: pd.DataFrame(
            weekly[[f"{model_id}_p_short", f"{model_id}_p_flat", f"{model_id}_p_long"]].to_numpy(),
            index=weekly.index,
            columns=["p_short", "p_flat", "p_long"],
        )
        for model_id in MODEL_IDS
    }
    weekly_policy_context = weekly_context.loc[:, list(CONDITION_FIELDS)]
    compiled_week = compile_policy(weekly_probabilities, weekly_policy_context, rules)
    weight_edit_ids = {model_id: [] for model_id in MODEL_IDS}
    for rule in rules:
        if not bool(condition_mask(weekly_policy_context, rule.conditions).any()):
            continue
        for edit in rule.edits:
            if edit.action == "select_frozen_expert":
                for model_id in MODEL_IDS:
                    weight_edit_ids[model_id].append(edit.edit_id)
            elif edit.action in {"multiply_model_weight", "set_model_weight"}:
                weight_edit_ids[edit.target].append(edit.edit_id)
    window_id = f"{start.isocalendar().year}-W{start.isocalendar().week:02d}"
    return build_observation(
        window_id=window_id,
        cutoff_utc=cutoff,
        active_policy_id=active_policy_id,
        active_edit_ids=list(active_edit_ids),
        market=market,
        probabilities=probabilities,
        news=news,
        model_weights=compiled_week.weights.mean().to_dict(),
        model_active_fractions=compiled_week.weights.gt(0.0).mean().to_dict(),
        model_weight_edit_ids=weight_edit_ids,
        retrieved_memories=list(retrieved_memories),
    )


def run_actor_smoke(
    *,
    config_path: Path = CONFIG_PATH,
    cache_root: Path = DEFAULT_OUTPUT,
    state_path: Path = DEFAULT_STATE,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    output_path: Path = DEFAULT_SMOKE,
    host: str = "http://localhost:11434",
    resume: bool = True,
    window_offset: int = 0,
) -> dict[str, object]:
    config = load_config(config_path)
    report = PreflightReport.model_validate_json(preflight_path.read_text(encoding="utf-8"))
    store = AgentStore(state_path)
    observation = build_weekly_observation(config=config, cache_root=cache_root, window_offset=window_offset)
    existing = store.load_window(observation.window_id, protocol_hash=report.protocol_hash)
    if existing:
        if not resume:
            raise ValueError(f"window already completed: {observation.window_id}; use --resume")
        return {"status": "cached", "window_id": observation.window_id, **existing["payload"]}

    caller = StructuredCaller(
        model=config.model,
        output_mode=report.output_mode,
        primary=OllamaClientTransport(host=host, timeout_seconds=config.timeout_seconds),
        fallback=OllamaHttpTransport(host=host, timeout_seconds=config.timeout_seconds),
        store=store,
        protocol_hash=report.protocol_hash,
        think=config.think,
        retry_delays_seconds=config.retry_delays_seconds,
        repair_attempts=config.repair_attempts,
    )
    call = caller.call(
        role="actor",
        messages=actor_messages(observation),
        response_model=CandidateBatch,
        temperature=config.temperatures["actor"],
    )
    errors = list(call.errors)
    candidates = ()
    status = call.status
    if call.value is not None:
        try:
            candidates = validate_candidates(call.value.candidates, maximum=config.candidate_budget)
        except ValueError as error:
            status = "semantic_noop"
            errors.append(str(error))
    for candidate in candidates:
        store.save_record(
            "candidates",
            f"{observation.window_id}:{candidate.candidate_id}",
            report.protocol_hash,
            candidate.model_dump(mode="json"),
            window_id=observation.window_id,
        )
    payload = {
        "actor_status": status,
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "candidate_count": len(candidates),
        "request_hash": call.request_hash,
        "llm_attempts": call.attempts,
        "backend": call.backend,
        "errors": errors,
        "observation": observation.model_dump(mode="json"),
    }
    store.save_window(
        window_id=observation.window_id,
        protocol_hash=report.protocol_hash,
        cutoff_utc=observation.cutoff_utc,
        status="completed",
        payload=payload,
    )
    result = {"status": "completed", "window_id": observation.window_id, **payload}
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result


def run_pipeline_smoke(
    *,
    config_path: Path = CONFIG_PATH,
    cache_root: Path = DEFAULT_OUTPUT,
    state_path: Path = DEFAULT_STATE,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    output_path: Path = DEFAULT_PIPELINE_SMOKE,
    host: str = "http://localhost:11434",
    resume: bool = True,
    window_offset: int = 12,
) -> dict[str, object]:
    config = load_config(config_path)
    report = PreflightReport.model_validate_json(preflight_path.read_text(encoding="utf-8"))
    store = AgentStore(state_path)
    observation = build_weekly_observation(config=config, cache_root=cache_root, window_offset=window_offset)
    existing = store.load_window(observation.window_id, protocol_hash=report.protocol_hash)
    if existing:
        if not resume:
            raise ValueError(f"window already completed: {observation.window_id}; use --resume")
        return {"status": "cached", "window_id": observation.window_id, **existing["payload"]}

    caller = StructuredCaller(
        model=config.model,
        output_mode=report.output_mode,
        primary=OllamaClientTransport(host=host, timeout_seconds=config.timeout_seconds),
        fallback=OllamaHttpTransport(host=host, timeout_seconds=config.timeout_seconds),
        store=store,
        protocol_hash=report.protocol_hash,
        think=config.think,
        retry_delays_seconds=config.retry_delays_seconds,
        repair_attempts=config.repair_attempts,
    )
    orchestrator = WeeklyOrchestrator(
        caller=caller,
        store=store,
        memory=MemoryManager(store, protocol_hash=report.protocol_hash),
        protocol_hash=report.protocol_hash,
        candidate_budget=config.candidate_budget,
        beam_width=config.beam_width,
        max_open_shadows=config.max_open_shadows,
    )
    source_start = pd.Timestamp(observation.cutoff_utc) + pd.Timedelta(microseconds=1) - pd.Timedelta(weeks=1)
    baseline_full = simulate_compiled_policy(
        rules=[LSTM_RULE], tp_bps=200, sl_bps=100, cache_root=cache_root
    )[0]

    def screen(candidate):
        return evaluate_candidate_historical(
            candidate,
            config=config,
            start=config.development_start_utc,
            end=source_start.to_pydatetime(),
            cache_root=cache_root,
            baseline_full=baseline_full,
        )

    shadows = orchestrator.propose_shadows(observation=observation, evaluate_historical=screen)
    payload = {
        "pipeline_status": "success",
        "shadow_ids": [shadow.shadow_id for shadow in shadows],
        "shadow_candidate_ids": [shadow.candidate.candidate_id for shadow in shadows],
        "shadow_count": len(shadows),
        "historical_end_exclusive": source_start.isoformat(),
        "llm_call_count": store.llm_call_count(),
        "observation": observation.model_dump(mode="json"),
    }
    store.save_window(
        window_id=observation.window_id,
        protocol_hash=report.protocol_hash,
        cutoff_utc=observation.cutoff_utc,
        status="completed",
        payload=payload,
    )
    result = {"status": "completed", "window_id": observation.window_id, **payload}
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return result


def _active_policy_state(
    store: AgentStore, *, protocol_hash: str, cutoff_utc
) -> tuple[str, list[str]]:
    policies = []
    for row in store.list_records("policies", protocol_hash=protocol_hash):
        policy = PolicyVersion.model_validate(row["payload"])
        if policy.activates_at_utc <= cutoff_utc:
            policies.append(policy)
    if not policies:
        return "policy-unanimity-consensus-v1", ["consensus-agreement"]
    latest = max(policies, key=lambda policy: (policy.activates_at_utc, policy.policy_id))
    return latest.policy_id, [edit.edit_id for edit in latest.rule.edits]


def _active_policy_rules(
    store: AgentStore, *, protocol_hash: str, cutoff_utc
) -> tuple[PolicyRule, ...]:
    policies = []
    for row in store.list_records("policies", protocol_hash=protocol_hash):
        policy = PolicyVersion.model_validate(row["payload"])
        if policy.activates_at_utc <= cutoff_utc:
            policies.append(policy)
    if not policies:
        return (CONSENSUS_RULE,)
    latest = max(policies, key=lambda policy: (policy.activates_at_utc, policy.policy_id))
    if any(edit.action in {"remove_active_edit", "reduce_active_edit"} for edit in latest.rule.edits):
        resolved = resolve_edits(CONSENSUS_RULE.edits, latest.rule.edits)
        return (PolicyRule(rule_id=latest.rule.rule_id, edits=list(resolved)),) if resolved else ()
    return CONSENSUS_RULE, latest.rule


def _open_shadows(store: AgentStore, *, protocol_hash: str) -> list[ShadowState]:
    return [
        shadow for shadow in (
            ShadowState.model_validate(row["payload"])
            for row in store.list_records("shadows", protocol_hash=protocol_hash)
        )
        if shadow.status == "open"
    ]


def run_development_replay(
    *,
    variant_id: str,
    config_path: Path = CONFIG_PATH,
    cache_root: Path = DEFAULT_OUTPUT,
    state_path: Path,
    preflight_path: Path = DEFAULT_PREFLIGHT,
    host: str = "http://localhost:11434",
    resume: bool = True,
    start_window_offset: int = 12,
    limit_windows: int = 1,
) -> dict[str, object]:
    if variant_id not in REGISTERED_VARIANTS:
        raise ValueError(f"unknown registered variant: {variant_id}")
    if limit_windows < 1:
        raise ValueError("limit_windows must be positive")
    variant = REGISTERED_VARIANTS[variant_id]
    config = load_config(config_path)
    report = PreflightReport.model_validate_json(preflight_path.read_text(encoding="utf-8"))
    final_end = _first_complete_monday(config) + pd.Timedelta(weeks=start_window_offset + limit_windows)
    if final_end > pd.Timestamp(config.development_end_utc):
        raise ValueError("requested replay enters the sealed or incomplete interval")
    store = AgentStore(state_path)
    store.save_run(
        run_id=f"{report.protocol_hash}:{variant_id}",
        protocol_hash=report.protocol_hash,
        payload={
            "variant_id": variant_id,
            "memory_mode": variant.memory_mode,
            "news_mode": variant.news_mode,
            "start_window_offset": start_window_offset,
        },
    )
    caller = StructuredCaller(
        model=config.model,
        output_mode=report.output_mode,
        primary=OllamaClientTransport(host=host, timeout_seconds=config.timeout_seconds),
        fallback=OllamaHttpTransport(host=host, timeout_seconds=config.timeout_seconds),
        store=store,
        protocol_hash=report.protocol_hash,
        run_scope=variant_id,
        think=config.think,
        retry_delays_seconds=config.retry_delays_seconds,
        repair_attempts=config.repair_attempts,
    )
    real_memory = MemoryManager(store, protocol_hash=report.protocol_hash)
    memory = DisabledMemoryManager() if variant.memory_mode == "none" else real_memory
    baseline_full = simulate_compiled_policy(
        rules=[LSTM_RULE], tp_bps=200, sl_bps=100, cache_root=cache_root
    )[0]
    processed = []
    for window_offset in range(start_window_offset, start_window_offset + limit_windows):
        preliminary = build_weekly_observation(
            config=config,
            cache_root=cache_root,
            window_offset=window_offset,
            news_mode=variant.news_mode,
        )
        existing = store.load_window(preliminary.window_id, protocol_hash=report.protocol_hash)
        if existing:
            if not resume:
                raise ValueError(f"window already completed: {preliminary.window_id}; use --resume")
            processed.append({"window_id": preliminary.window_id, "status": "cached"})
            continue

        active_policy_id, active_edit_ids = _active_policy_state(
            store, protocol_hash=report.protocol_hash, cutoff_utc=preliminary.cutoff_utc
        )
        active_rules = _active_policy_rules(
            store, protocol_hash=report.protocol_hash, cutoff_utc=preliminary.cutoff_utc
        )
        tags = observation_tags(preliminary)
        memories = memory.retrieve(
            cutoff_utc=preliminary.cutoff_utc,
            tags=tags,
            maximum=config.memory.max_retrieved,
        )
        if variant.memory_mode == "shuffled":
            memories = shuffle_time_eligible_memories(memories, window_id=preliminary.window_id)
        observation = build_weekly_observation(
            config=config,
            cache_root=cache_root,
            window_offset=window_offset,
            news_mode=variant.news_mode,
            retrieved_memories=memories,
            active_policy_id=active_policy_id,
            active_edit_ids=active_edit_ids,
            active_rules=active_rules,
        )
        close_end = pd.Timestamp(observation.cutoff_utc) + pd.Timedelta(microseconds=1)
        closed_ids: list[str] = []
        continued_ids: list[str] = []
        promoted_ids: list[str] = []
        closer = WeeklyOrchestrator(
            caller=caller,
            store=store,
            memory=memory,
            protocol_hash=report.protocol_hash,
            candidate_budget=config.candidate_budget,
            beam_width=config.beam_width,
            max_open_shadows=config.max_open_shadows,
        )
        for shadow in _open_shadows(store, protocol_hash=report.protocol_hash):
            all_window_ids = shadow_window_ids(shadow, close_end_exclusive=close_end.to_pydatetime())
            if len(all_window_ids) < config.shadow_min_weeks:
                updated = shadow.model_copy(update={"observed_window_ids": all_window_ids})
                store.replace_record(
                    "shadows", shadow.shadow_id, report.protocol_hash,
                    updated.model_dump(mode="json"), window_id=observation.window_id,
                )
                continued_ids.append(shadow.shadow_id)
                continue
            window_ids = all_window_ids[:config.shadow_max_weeks]
            evaluation_end = pd.Timestamp(shadow.eligible_after_utc) + pd.Timedelta(weeks=len(window_ids))
            evaluation = evaluate_candidate_shadow(
                shadow.candidate,
                config=config,
                start=shadow.eligible_after_utc,
                end=evaluation_end.to_pydatetime(),
                window_ids=window_ids,
                cache_root=cache_root,
                baseline_full=baseline_full,
            )
            if evaluation.decision == "shadow_continue":
                updated = shadow.model_copy(update={"observed_window_ids": window_ids})
                store.replace_record(
                    "shadows", shadow.shadow_id, report.protocol_hash,
                    updated.model_dump(mode="json"), window_id=observation.window_id,
                )
                continued_ids.append(shadow.shadow_id)
                continue
            _, _, policy = closer.close_shadow(
                shadow=shadow,
                evaluation=evaluation,
                close_cutoff_utc=observation.cutoff_utc,
                active_policy_id=active_policy_id,
                memory_tags=tags,
            )
            closed_ids.append(shadow.shadow_id)
            if policy is not None:
                promoted_ids.append(policy.policy_id)

        if variant.memory_mode != "none":
            real_memory.consolidate(now_utc=observation.cutoff_utc)
        open_now = _open_shadows(store, protocol_hash=report.protocol_hash)
        capacity = config.max_open_shadows - len(open_now)
        new_shadows: tuple[ShadowState, ...] = ()
        if capacity > 0:
            source_start = close_end - pd.Timedelta(weeks=1)

            def screen(candidate):
                return evaluate_candidate_historical(
                    candidate,
                    config=config,
                    start=config.development_start_utc,
                    end=source_start.to_pydatetime(),
                    cache_root=cache_root,
                    baseline_full=baseline_full,
                )

            proposer = WeeklyOrchestrator(
                caller=caller,
                store=store,
                memory=memory,
                protocol_hash=report.protocol_hash,
                candidate_budget=config.candidate_budget,
                beam_width=config.beam_width,
                max_open_shadows=capacity,
            )
            new_shadows = proposer.propose_shadows(
                observation=observation, evaluate_historical=screen
            )
        payload = {
            "variant_id": variant_id,
            "memory_mode": variant.memory_mode,
            "news_mode": variant.news_mode,
            "closed_shadow_ids": closed_ids,
            "continued_shadow_ids": continued_ids,
            "new_shadow_ids": [shadow.shadow_id for shadow in new_shadows],
            "promoted_policy_ids": promoted_ids,
            "retrieved_memory_ids": [item.memory_id for item in memories],
            "active_policy_id": active_policy_id,
        }
        store.save_window(
            window_id=observation.window_id,
            protocol_hash=report.protocol_hash,
            cutoff_utc=observation.cutoff_utc,
            status="completed",
            payload=payload,
        )
        processed.append({"window_id": observation.window_id, "status": "completed", **payload})

    summary = {
        "protocol_hash": report.protocol_hash,
        "variant_id": variant_id,
        "windows_requested": limit_windows,
        "windows": processed,
        "llm_calls": store.llm_call_count(),
        "open_shadows": len(_open_shadows(store, protocol_hash=report.protocol_hash)),
        "policies": len(store.list_records("policies", protocol_hash=report.protocol_hash)),
        "memories": len(store.list_records("memories", protocol_hash=report.protocol_hash)),
    }
    output_path = cache_root / f"replay_{variant_id}.json"
    output_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--state", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--host", default="http://localhost:11434")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--mode", choices=["development"], default="development")
    parser.add_argument("--limit-windows", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pipeline-smoke", action="store_true")
    parser.add_argument("--window-offset", type=int, default=12)
    parser.add_argument("--run-replay", action="store_true")
    parser.add_argument("--variant", choices=sorted(REGISTERED_VARIANTS), default="reflection_real_memory")
    args = parser.parse_args(argv)
    if args.preflight:
        report = run_preflight(
            config_path=args.config,
            cache_root=args.cache_root,
            state_path=args.state,
            report_path=args.cache_root / "preflight.json",
            host=args.host,
        )
        print(report.model_dump_json(indent=2))
        return
    if args.run_replay:
        state_path = args.state
        if state_path == DEFAULT_STATE:
            state_path = args.cache_root / f"agent_state_{args.variant}.sqlite"
        result = run_development_replay(
            variant_id=args.variant,
            config_path=args.config,
            cache_root=args.cache_root,
            state_path=state_path,
            preflight_path=args.cache_root / "preflight.json",
            host=args.host,
            resume=args.resume,
            start_window_offset=args.window_offset,
            limit_windows=args.limit_windows,
        )
        print(json.dumps(result, indent=2))
        return
    if args.limit_windows != 1:
        raise ValueError("development smoke currently requires --limit-windows 1")
    if args.pipeline_smoke:
        result = run_pipeline_smoke(
            config_path=args.config,
            cache_root=args.cache_root,
            state_path=args.state,
            preflight_path=args.cache_root / "preflight.json",
            output_path=args.cache_root / "pipeline_smoke.json",
            host=args.host,
            resume=args.resume,
            window_offset=args.window_offset,
        )
    else:
        result = run_actor_smoke(
            config_path=args.config,
            cache_root=args.cache_root,
            state_path=args.state,
            preflight_path=args.cache_root / "preflight.json",
            output_path=args.cache_root / "development_smoke.json",
            host=args.host,
            resume=args.resume,
        )
    print(json.dumps({key: value for key, value in result.items() if key != "observation"}, indent=2))


if __name__ == "__main__":
    main()
