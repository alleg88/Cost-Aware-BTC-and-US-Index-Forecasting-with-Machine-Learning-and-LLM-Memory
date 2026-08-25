"""Resumable continuous runner for the preregistered Reflection Agent v3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from datetime import timedelta
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests

from evaluation.economics import economics_summary
from reflection_agent.v2.transport import DeepSeekSchemaCaller, SchemaCallResult
from reflection_agent.v3.config import REGISTERED_VARIANTS, load_v3_config
from reflection_agent.v3.contracts import (
    ProposalChoiceOutput,
    ProposalOutput,
    ReflectionChoiceOutput,
    ReflectionOutput,
)
from reflection_agent.v3.leakage import LeakageAuditor
from reflection_agent.v3.memory import (
    NoMemory,
    RealMemory,
    ShuffledMemory,
    StaticPolicyControl,
)
from reflection_agent.v3.opportunities import (
    assign_observation_episodes,
    build_development_opportunities,
    build_exact_opportunities,
)
from reflection_agent.v3.orchestrator import ReflectionOrchestrator
from reflection_agent.v3.policy import ActiveAllowRule, apply_policy
from reflection_agent.v3.prompts import SYSTEM_PROMPT_V3, prompt_hashes
from reflection_agent.v3.evaluator import ShadowCandidate


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "reflection_agent_v3"
DEFAULT_CONFIG = CODE_ROOT / "configs" / "reflection_agent_v3.yaml"
Q2_START = pd.Timestamp("2026-04-01", tz="UTC")
STAGE_ORDER = ("development", "h1", "forward")


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    _atomic_text(path, "".join(_canonical_json(row) + "\n" for row in rows))


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values("opportunity_id", kind="stable").copy()
    for column in normalized.columns:
        if isinstance(normalized[column].dtype, pd.DatetimeTZDtype) or pd.api.types.is_datetime64_any_dtype(
            normalized[column]
        ):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).map(
                lambda value: value.isoformat() if pd.notna(value) else None
            )
    records = normalized.where(pd.notna(normalized), None).to_dict(orient="records")
    return _hash({"columns": list(normalized.columns), "records": records})


def _implementation_hash() -> str:
    paths = [
        Path(__file__).resolve(),
        CODE_ROOT / "reflection_agent" / "v2" / "transport.py",
        *sorted((CODE_ROOT / "reflection_agent" / "v3").glob("*.py")),
    ]
    return _hash(
        {
            str(path.relative_to(CODE_ROOT)).replace("\\", "/"): _sha256_file(path)
            for path in paths
        }
    )


def _validate_stage_frame(frame: pd.DataFrame, expected_stage: str) -> pd.DataFrame:
    required = {
        "opportunity_id",
        "stage",
        "source_role",
        "fold_id",
        "row_key",
        "source_artifact_hash",
        "route",
        "side",
        "decision_time",
        "feature_available_time",
        "entry_time",
        "outcome_available_time",
        "gross_return",
        "net_return",
        "round_trip_cost",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"runner opportunity frame lacks columns: {missing}")
    output = frame.copy().reset_index(drop=True)
    for column in (
        "decision_time",
        "feature_available_time",
        "entry_time",
        "exit_time",
        "outcome_available_time",
    ):
        if column in output:
            output[column] = pd.to_datetime(output[column], utc=True)
    if output["opportunity_id"].duplicated().any():
        raise ValueError("runner opportunity IDs must be unique")
    if set(output["stage"]) != {expected_stage}:
        raise ValueError("runner stage frame does not match its registered stage")
    if output["outcome_available_time"].ge(Q2_START).any():
        raise ValueError("Q2 lockbox timestamp entered the runner")
    if not output["feature_available_time"].le(output["decision_time"]).all():
        raise ValueError("future feature entered the runner")
    if not output["decision_time"].lt(output["outcome_available_time"]).all():
        raise ValueError("outcome availability is not later than decision")
    cost = output["gross_return"].astype(float) - output["net_return"].astype(float)
    if not np.allclose(
        cost, output["round_trip_cost"].astype(float), rtol=0.0, atol=1e-12
    ):
        raise ValueError("runner gross/net/cost fields do not reconcile")
    return output


def _load_stage_frames(stages: Sequence[str]) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for stage in stages:
        frames[stage] = (
            build_development_opportunities()
            if stage == "development"
            else build_exact_opportunities(stage)
        )
    return frames


class CachedCaller:
    """Persist terminal schema calls and replay their original result exactly."""

    def __init__(self, inner: Any, database_path: Path, *, protocol_hash: str) -> None:
        self.inner = inner
        self.config = inner.config
        self.database_path = database_path
        self.protocol_hash = protocol_hash
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS runner_call_cache (
                    cache_key TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )"""
            )

    def call(self, *, role, messages, response_model, allowed_ids):
        key = _hash(
            {
                "protocol_hash": self.protocol_hash,
                "role": role,
                "messages": messages,
                "schema": response_model.model_json_schema(),
                "allowed_ids": allowed_ids,
            }
        )
        with sqlite3.connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT protocol_hash, payload_json FROM runner_call_cache "
                "WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is not None:
            if row[0] != self.protocol_hash:
                raise ValueError("cached call protocol hash changed")
            payload = json.loads(row[1])
            value = (
                response_model.model_validate(payload["value"])
                if payload["value"] is not None
                else None
            )
            return SchemaCallResult(
                status=payload["status"],
                value=value,
                raw_content=payload["raw_content"],
                request_hash=payload["request_hash"],
                response_hash=payload["response_hash"],
                schema_hash=payload["schema_hash"],
                attempts=int(payload["attempts"]),
                latency_seconds=float(payload["latency_seconds"]),
                metadata=payload["metadata"],
                errors=tuple(payload["errors"]),
            )
        result = self.inner.call(
            role=role,
            messages=messages,
            response_model=response_model,
            allowed_ids=allowed_ids,
        )
        payload = {
            "status": result.status,
            "value": (
                result.value.model_dump(mode="json")
                if result.value is not None
                else None
            ),
            "raw_content": getattr(result, "raw_content", ""),
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "schema_hash": result.schema_hash,
            "attempts": result.attempts,
            "latency_seconds": getattr(result, "latency_seconds", 0.0),
            "metadata": getattr(result, "metadata", {}),
            "errors": list(result.errors),
        }
        with sqlite3.connect(self.database_path) as connection:
            connection.execute(
                "INSERT INTO runner_call_cache (cache_key, protocol_hash, payload_json) "
                "VALUES (?, ?, ?)",
                (key, self.protocol_hash, _canonical_json(payload)),
            )
        return result


def _memory_control(variant_id: str, real: RealMemory):
    if variant_id == "reflection_real_memory":
        return real
    if variant_id == "reflection_no_memory":
        return NoMemory(real)
    if variant_id == "reflection_shuffled_memory":
        return ShuffledMemory(real)
    if variant_id in {"static_high_extra", "static_all_extra", "union_baseline"}:
        return StaticPolicyControl(real, static_variant=variant_id)
    raise ValueError(f"unknown registered variant: {variant_id}")


def _episode_table(assigned: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "observation_episode_id",
        "fold_id",
        "episode_cutoff_utc",
        "episode_status",
        "episode_can_propose",
        "episode_opportunity_count",
        "episode_candidate_count",
        "episode_long_candidate_count",
        "episode_short_candidate_count",
    ]
    return (
        assigned.loc[:, columns]
        .drop_duplicates("observation_episode_id")
        .sort_values(["fold_id", "episode_cutoff_utc"], kind="stable")
        .reset_index(drop=True)
    )


def _per_bar(selected: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    start = frame["decision_time"].min().floor("15min")
    end = frame["outcome_available_time"].max().ceil("15min")
    result = pd.Series(0.0, index=pd.date_range(start, end, freq="15min"))
    trades = selected.loc[selected["selected"]].copy()
    if len(trades):
        booking = trades["entry_time"].dt.floor("15min")
        booked = trades.assign(_booking=booking).groupby("_booking")["net_return"].sum()
        result.loc[booked.index] = booked.to_numpy(float)
    return result.rename("net_return")


def _policy_metrics(selected: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    trades = selected.loc[selected["selected"]].copy()
    economics = economics_summary(_per_bar(selected, frame))
    net = trades["net_return"].astype(float)
    gross = trades["gross_return"].astype(float)
    effective_days = max(
        (frame["decision_time"].max() - frame["decision_time"].min()).total_seconds()
        / 86_400.0,
        1.0 / 96.0,
    )
    return {
        "selected_trades": len(trades),
        "selected_long_trades": int(trades["side"].eq("LONG").sum()),
        "selected_short_trades": int(trades["side"].eq("SHORT").sum()),
        "additional_trades": int(trades["route"].eq("COVERAGE_CANDIDATE").sum()),
        "trades_per_effective_day": float(len(trades) / effective_days),
        "gross_return": float(gross.sum()),
        "cost_return": float((gross - net).sum()),
        "net_return": float(net.sum()),
        "long_net_return": float(net.loc[trades["side"].eq("LONG")].sum()),
        "short_net_return": float(net.loc[trades["side"].eq("SHORT")].sum()),
        "sortino": float(economics["sortino"]),
        "sharpe": float(economics["sharpe"]),
        "max_drawdown": float(economics["max_drawdown"]),
    }


def _call_audits(database_path: Path) -> list[dict[str, Any]]:
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT payload_json FROM call_audit ORDER BY rowid"
        ).fetchall()
    return [json.loads(row[0]) for row in rows]


def _memory_frame(real: RealMemory) -> pd.DataFrame:
    rows = [card.model_dump(mode="json") for card in real.all_cards()]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=list(MemoryCardColumns))


MemoryCardColumns = (
    "memory_id",
    "source_stage",
    "protocol_scope",
    "memory_type",
    "created_at_utc",
    "max_support_outcome_time",
    "lesson",
    "evidence_status",
    "tags",
    "source_evaluation_ids",
    "expires_at_utc",
    "expires_after_episode",
)


def _state_payload(
    *,
    input_hash: str,
    protocol_hash: str,
    state: dict[str, Any],
) -> dict[str, Any]:
    return {"input_hash": input_hash, "protocol_hash": protocol_hash, **state}


def _default_state() -> dict[str, Any]:
    return {
        "completed_stages": [],
        "stage_episode_counts": {},
        "stage_call_starts": {},
        "global_episode_number": 0,
        "active_rules": [],
        "policy_history": [],
        "open_candidates": [],
        "transitions": [],
        "evaluations": [],
        "stage_summaries": {},
    }


def _load_state(path: Path, *, input_hash: str, protocol_hash: str) -> dict[str, Any]:
    if not path.is_file():
        return _default_state()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("input_hash") != input_hash:
        raise ValueError("resumable state input hash changed")
    if payload.get("protocol_hash") != protocol_hash:
        raise ValueError("resumable state protocol hash changed")
    return {key: payload[key] for key in _default_state()}


def _update_policy_history(
    history: list[ActiveAllowRule], closures: Sequence[Any]
) -> list[ActiveAllowRule]:
    output = list(history)
    for closure in closures:
        if closure.status != "closed":
            continue
        if closure.new_active_rule is not None:
            output.append(closure.new_active_rule)
        if closure.removed_rule_id is not None:
            if closure.evaluation.shadow_cutoff_utc is None:
                raise AssertionError("removal lacks a shadow cutoff")
            deactivation = closure.evaluation.shadow_cutoff_utc + timedelta(
                microseconds=1
            )
            output = [
                rule.model_copy(update={"deactivates_at_utc": deactivation})
                if rule.rule_id == closure.removed_rule_id
                else rule
                for rule in output
            ]
    return output


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(root.rglob("*"), key=lambda item: str(item))
        if path.is_file() and path.name != "manifest.json"
    }


def _resume_if_complete(
    root: Path, *, input_hash: str, protocol_hash: str
) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    summary_path = root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("input_hash") != input_hash:
        raise ValueError("completed variant input hash changed")
    if manifest.get("protocol_hash") != protocol_hash:
        raise ValueError("completed variant protocol hash changed")
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = root / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"completed variant artifact drifted: {name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["resumed"] = True
    return summary


def run_variant(
    variant_id: str,
    *,
    output_root: str | Path = CACHE,
    stage_frames: dict[str, pd.DataFrame] | None = None,
    caller: Any | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    stages: Sequence[str] = STAGE_ORDER,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    config = load_v3_config(config_path)
    if variant_id not in config.registered_variants:
        raise ValueError(f"variant is not registered: {variant_id}")
    normalized_stages = tuple(stages)
    if not normalized_stages or any(stage not in STAGE_ORDER for stage in normalized_stages):
        raise ValueError("runner stages are not registered")
    if tuple(sorted(normalized_stages, key=STAGE_ORDER.index)) != normalized_stages:
        raise ValueError("runner stages must be chronological")
    raw_frames = stage_frames or _load_stage_frames(normalized_stages)
    frames = {
        stage: _validate_stage_frame(raw_frames[stage], stage)
        for stage in normalized_stages
    }
    stage_hashes = {stage: _frame_hash(frame) for stage, frame in frames.items()}
    opportunity_hash = _hash(stage_hashes)
    implementation_hash = _implementation_hash()
    input_hash = _hash(
        {
            "stage_hashes": stage_hashes,
            "stages": normalized_stages,
            "max_episodes": max_episodes,
        }
    )
    protocol_hash = _hash(
        {
            "config": config.model_dump(mode="json"),
            "implementation_hash": implementation_hash,
            "input_hash": input_hash,
            "opportunity_hash": opportunity_hash,
            "prompt_hashes": prompt_hashes(),
            "proposal_choice_schema": ProposalChoiceOutput.model_json_schema(),
            "reflection_choice_schema": ReflectionChoiceOutput.model_json_schema(),
            "compiled_proposal_schema": ProposalOutput.model_json_schema(),
            "compiled_reflection_schema": ReflectionOutput.model_json_schema(),
            "transport_hash": _sha256_file(
                CODE_ROOT / "reflection_agent" / "v2" / "transport.py"
            ),
            "variant_id": variant_id,
        }
    )
    root = Path(output_root).resolve() / variant_id
    resumed = _resume_if_complete(
        root, input_hash=input_hash, protocol_hash=protocol_hash
    )
    if resumed is not None:
        return resumed
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "run_state.json"
    state = _load_state(
        state_path, input_hash=input_hash, protocol_hash=protocol_hash
    )

    real_memory = RealMemory(
        root / "agent_state.sqlite", protocol_hash=protocol_hash
    )
    memory = _memory_control(variant_id, real_memory)
    uses_llm = bool(getattr(memory, "uses_llm", True))
    orchestrator: ReflectionOrchestrator | None = None
    if uses_llm:
        inner = caller or DeepSeekSchemaCaller(
            config, call_log_path=root / "cloud_calls.jsonl"
        )
        cached = CachedCaller(
            inner, root / "agent_state.sqlite", protocol_hash=protocol_hash
        )
        orchestrator = ReflectionOrchestrator(
            caller=cached,
            auditor=LeakageAuditor(root / "prompt_audit.jsonl"),
            memory=memory,
            protocol_hash=protocol_hash,
        )

    active_rules = [ActiveAllowRule.model_validate(row) for row in state["active_rules"]]
    policy_history = [
        ActiveAllowRule.model_validate(row) for row in state["policy_history"]
    ]
    open_candidates = [
        ShadowCandidate.model_validate(row) for row in state["open_candidates"]
    ]

    for stage in normalized_stages:
        if stage in state["completed_stages"]:
            continue
        frame = frames[stage]
        assigned = assign_observation_episodes(
            frame,
            min_candidates=config.episode_min_candidates,
            max_candidates=config.episode_max_candidates,
            min_per_side=config.episode_min_per_side,
        )
        episodes = _episode_table(assigned)
        if max_episodes is not None:
            remaining = max(max_episodes - int(state["global_episode_number"]), 0)
            episodes = episodes.head(remaining).copy()
            included_ids = set(episodes["observation_episode_id"])
            assigned = assigned.loc[
                assigned["observation_episode_id"].isin(included_ids)
            ].copy()
            frame = frame.loc[
                frame["opportunity_id"].isin(set(assigned["opportunity_id"]))
            ].copy()
        if frame.empty:
            break
        stage_start_calls = state["stage_call_starts"].setdefault(
            stage, len(_call_audits(root / "agent_state.sqlite"))
        )
        start_index = int(state["stage_episode_counts"].get(stage, 0))
        stage_transitions: list[dict[str, Any]] = []

        if uses_llm:
            assert orchestrator is not None
            for episode_index in range(start_index, len(episodes)):
                row = episodes.iloc[episode_index]
                episode_id = str(row["observation_episode_id"])
                episode_frame = assigned.loc[
                    assigned["observation_episode_id"].eq(episode_id)
                ].copy()
                result = orchestrator.run_episode(
                    episode_frame,
                    episode_id=episode_id,
                    episode_number=int(state["global_episode_number"]),
                    active_rules=active_rules,
                    open_candidates=open_candidates,
                    available_opportunities=frame,
                )
                policy_history = _update_policy_history(
                    policy_history, result.closures
                )
                for closure in result.closures:
                    if closure.status == "closed":
                        state["evaluations"].append(
                            closure.evaluation.model_dump(mode="json")
                        )
                active_rules = list(result.active_rules)
                open_candidates = list(result.open_candidates)
                transition = {
                    "stage": stage,
                    "episode_id": episode_id,
                    "episode_number": int(state["global_episode_number"]),
                    "status": result.status,
                    "candidate_id": (
                        result.candidate.candidate_id
                        if result.candidate is not None
                        else None
                    ),
                    "closed_candidate_ids": [
                        closure.candidate_id
                        for closure in result.closures
                        if closure.status == "closed"
                    ],
                    "call_status": result.call_status,
                }
                stage_transitions.append(transition)
                state["transitions"].append(transition)
                state["global_episode_number"] += 1
                state["stage_episode_counts"][stage] = episode_index + 1

                next_fold = (
                    int(episodes.iloc[episode_index + 1]["fold_id"])
                    if episode_index + 1 < len(episodes)
                    else None
                )
                current_fold = int(row["fold_id"])
                if stage == "development" and next_fold != current_fold and open_candidates:
                    fold_frame = frame.loc[frame["fold_id"].eq(current_fold)]
                    forced = orchestrator.close_eligible_candidates(
                        open_candidates,
                        fold_frame,
                        active_rules=active_rules,
                        episode_number=int(state["global_episode_number"]),
                        force_close=True,
                    )
                    policy_history = _update_policy_history(policy_history, forced)
                    for closure in forced:
                        if closure.status != "closed":
                            raise AssertionError("forced development shadow remained open")
                        state["evaluations"].append(
                            closure.evaluation.model_dump(mode="json")
                        )
                    active_rules = (
                        list(forced[-1].active_rules_after) if forced else active_rules
                    )
                    open_candidates = []

                state["active_rules"] = [
                    rule.model_dump(mode="json") for rule in active_rules
                ]
                state["policy_history"] = [
                    rule.model_dump(mode="json") for rule in policy_history
                ]
                state["open_candidates"] = [
                    candidate.model_dump(mode="json") for candidate in open_candidates
                ]
                _write_json(
                    state_path,
                    _state_payload(
                        input_hash=input_hash,
                        protocol_hash=protocol_hash,
                        state=state,
                    ),
                )

            if open_candidates:
                forced = orchestrator.close_eligible_candidates(
                    open_candidates,
                    frame,
                    active_rules=active_rules,
                    episode_number=int(state["global_episode_number"]),
                    force_close=True,
                )
                policy_history = _update_policy_history(policy_history, forced)
                for closure in forced:
                    if closure.status != "closed":
                        raise AssertionError("forced stage shadow remained open")
                    state["evaluations"].append(
                        closure.evaluation.model_dump(mode="json")
                    )
                active_rules = list(forced[-1].active_rules_after) if forced else active_rules
                open_candidates = []
        else:
            state["stage_episode_counts"][stage] = len(episodes)
            state["global_episode_number"] += len(episodes)

        if variant_id == "static_high_extra":
            selected = apply_policy(frame, [], static_variant="static_high_extra")
        elif variant_id == "static_all_extra":
            selected = apply_policy(frame, [], static_variant="static_all_extra")
        elif variant_id == "union_baseline":
            selected = apply_policy(frame, [], static_variant="union_baseline")
        else:
            selected = apply_policy(frame, policy_history)
        per_bar = _per_bar(selected, frame)
        metrics = _policy_metrics(selected, frame)
        stage_dir = root / "stages" / stage
        _write_parquet(stage_dir / "selected_ledger.parquet", selected)
        _write_parquet(stage_dir / "episode_ledger.parquet", episodes)
        _write_parquet(
            stage_dir / "per_bar_returns.parquet",
            per_bar.rename_axis("timestamp").reset_index(),
        )
        stage_evaluations = [
            item for item in state["evaluations"] if item["source_stage"] == stage
        ]
        _write_jsonl(stage_dir / "evaluations.jsonl", stage_evaluations)
        _write_jsonl(
            stage_dir / "transitions.jsonl",
            [item for item in state["transitions"] if item["stage"] == stage],
        )
        stage_calls = _call_audits(root / "agent_state.sqlite")[stage_start_calls:]
        failures = sum(
            item.get("status") in {"transport_error", "schema_failure"}
            for item in stage_calls
        )
        failure_fraction = float(failures / len(stage_calls)) if stage_calls else 0.0
        stage_status = (
            "INVALID_TRANSPORT_COVERAGE"
            if failure_fraction > config.max_transport_failure_fraction
            else "complete"
        )
        stage_summary = {
            "stage": stage,
            "status": stage_status,
            "opportunities": len(frame),
            "union_base_trades": int(frame["route"].eq("UNION_BASE").sum()),
            "coverage_candidates": int(
                frame["route"].eq("COVERAGE_CANDIDATE").sum()
            ),
            "episodes": len(episodes),
            "calls": len(stage_calls),
            "call_failures": failures,
            "call_failure_fraction": failure_fraction,
            "active_rules_at_end": len(active_rules),
            "memory_records_at_end": len(real_memory.all_cards()),
            "maximum_outcome_available_time": frame[
                "outcome_available_time"
            ].max().isoformat(),
            "lockbox_2026_q2_used": False,
            **metrics,
        }
        _write_json(stage_dir / "summary.json", stage_summary)
        state["stage_summaries"][stage] = stage_summary
        state["completed_stages"].append(stage)
        state["active_rules"] = [
            rule.model_dump(mode="json") for rule in active_rules
        ]
        state["policy_history"] = [
            rule.model_dump(mode="json") for rule in policy_history
        ]
        state["open_candidates"] = []
        real_memory.checkpoint_stage(
            stage,
            {
                "stage": stage,
                "summary_hash": _hash(stage_summary),
                "active_rule_ids": [rule.rule_id for rule in active_rules],
                "memory_count": len(real_memory.all_cards()),
            },
        )
        _write_json(
            state_path,
            _state_payload(
                input_hash=input_hash,
                protocol_hash=protocol_hash,
                state=state,
            ),
        )

    _write_parquet(root / "memory_ledger.parquet", _memory_frame(real_memory))
    _write_jsonl(
        root / "policy_history.jsonl",
        [rule.model_dump(mode="json") for rule in policy_history],
    )
    _write_jsonl(root / "transitions.jsonl", state["transitions"])
    _write_jsonl(root / "evaluations.jsonl", state["evaluations"])
    if not (root / "prompt_audit.jsonl").exists():
        _atomic_text(root / "prompt_audit.jsonl", "")
    if not (root / "cloud_calls.jsonl").exists():
        _atomic_text(root / "cloud_calls.jsonl", "")
    call_audits = _call_audits(root / "agent_state.sqlite")
    summary: dict[str, Any] = {
        "status": "complete",
        "resumed": False,
        "variant_id": variant_id,
        "completed_stages": state["completed_stages"],
        "stage_summaries": state["stage_summaries"],
        "input_hash": input_hash,
        "implementation_hash": implementation_hash,
        "protocol_hash": protocol_hash,
        "opportunity_hash": opportunity_hash,
        "stage_hashes": stage_hashes,
        "total_calls": len(call_audits),
        "total_call_failures": sum(
            item.get("status") in {"transport_error", "schema_failure"}
            for item in call_audits
        ),
        "memory_records": len(real_memory.all_cards()),
        "policy_records": len(policy_history),
        "candidate_evaluations": len(state["evaluations"]),
        "maximum_outcome_available_time": max(
            frame["outcome_available_time"].max() for frame in frames.values()
        ).isoformat(),
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "summary.json", summary)
    manifest = {
        "variant_id": variant_id,
        "input_hash": input_hash,
        "implementation_hash": implementation_hash,
        "protocol_hash": protocol_hash,
        "opportunity_hash": opportunity_hash,
        "stage_hashes": stage_hashes,
        "model": config.model if uses_llm else None,
        "model_digest": config.required_model_digest if uses_llm else None,
        "prompt_hashes": prompt_hashes(),
        "schema_hashes": {
            "proposal": _hash(ProposalChoiceOutput.model_json_schema()),
            "reflection": _hash(ReflectionChoiceOutput.model_json_schema()),
            "compiled_proposal": _hash(ProposalOutput.model_json_schema()),
            "compiled_reflection": _hash(ReflectionOutput.model_json_schema()),
        },
        "artifact_hashes": _artifact_hashes(root),
        "maximum_outcome_available_time": summary[
            "maximum_outcome_available_time"
        ],
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "manifest.json", manifest)
    return summary


def resume_variant(*args, **kwargs) -> dict[str, Any]:
    return run_variant(*args, **kwargs)


def run_registered_experiment(
    *,
    output_root: str | Path = CACHE,
    stage_frames: dict[str, pd.DataFrame] | None = None,
    variants: Sequence[str] = REGISTERED_VARIANTS,
) -> dict[str, dict[str, Any]]:
    frames = stage_frames or _load_stage_frames(STAGE_ORDER)
    return {
        variant: run_variant(
            variant,
            output_root=output_root,
            stage_frames=frames,
            stages=STAGE_ORDER,
        )
        for variant in variants
    }


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split(".") if part.isdigit())


def _model_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _live_model_record(model: str) -> dict[str, Any]:
    version_response = requests.get("http://localhost:11434/api/version", timeout=10)
    version_response.raise_for_status()
    import ollama

    listing = ollama.list()
    match = next(
        (
            item
            for item in _model_value(listing, "models", [])
            if _model_value(item, "model", _model_value(item, "name")) == model
        ),
        None,
    )
    if match is None:
        raise RuntimeError(f"exact Ollama model is not resolved locally: {model}")
    shown = ollama.show(model)
    return {
        "model": model,
        "digest": str(_model_value(match, "digest", "")),
        "capabilities": list(_model_value(shown, "capabilities", []) or []),
        "ollama_version": str(version_response.json()["version"]),
    }


def run_preflight(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    model_record: dict[str, Any] | None = None,
    stage_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    config = load_v3_config(config_path)
    record = model_record or _live_model_record(config.model)
    if record.get("model") != config.model:
        raise RuntimeError("preflight resolved a different model tag")
    digest = str(record.get("digest", ""))
    if digest != config.required_model_digest:
        raise RuntimeError("preflight model digest does not match the frozen snapshot")
    version = str(record.get("ollama_version", "0.0.0"))
    if _version_tuple(version) < (0, 32, 5):
        raise RuntimeError("Ollama 0.32.5 or newer is required")
    capabilities = set(record.get("capabilities", []))
    if "thinking" not in capabilities:
        raise RuntimeError("exact model does not advertise thinking support")
    raw_frames = stage_frames or _load_stage_frames(STAGE_ORDER)
    frames = {
        stage: _validate_stage_frame(raw_frames[stage], stage)
        for stage in STAGE_ORDER
    }
    stage_hashes = {stage: _frame_hash(frame) for stage, frame in frames.items()}
    opportunity_hash = _hash(stage_hashes)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    active_caller = caller or DeepSeekSchemaCaller(
        config, call_log_path=root / "preflight_calls.jsonl"
    )
    expected = {
        "schema_version": "3.0",
        "choice_index": 0,
        "evidence_indices": [0],
        "memory_indices": [],
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V3},
        {
            "role": "user",
            "content": "TASK: PREFLIGHT_STRICT_SCHEMA\nRETURN_JSON="
            + _canonical_json(expected),
        },
    ]
    allowed_ids = {
        "choice_indices": [0],
        "evidence_indices": [0],
        "memory_indices": [],
    }
    probe = active_caller.call(
        role="proposal",
        messages=messages,
        response_model=ProposalChoiceOutput,
        allowed_ids=allowed_ids,
    )
    if probe.value is None:
        raise RuntimeError(f"strict schema preflight failed: {probe.errors}")
    if (
        probe.value.choice_index != 0
        or probe.value.evidence_indices != [0]
        or probe.value.memory_indices
    ):
        raise RuntimeError("strict schema preflight returned invalid host references")
    result = {
        "passed": True,
        "implementation_hash": _implementation_hash(),
        "model": config.model,
        "model_digest": digest,
        "ollama_version": version,
        "capabilities": sorted(capabilities),
        "probe_status": probe.status,
        "probe_request_hash": probe.request_hash,
        "probe_response_hash": probe.response_hash,
        "prompt_hashes": prompt_hashes(),
        "schema_hashes": {
            "proposal": _hash(ProposalChoiceOutput.model_json_schema()),
            "reflection": _hash(ReflectionChoiceOutput.model_json_schema()),
            "compiled_proposal": _hash(ProposalOutput.model_json_schema()),
            "compiled_reflection": _hash(ReflectionOutput.model_json_schema()),
        },
        "stage_hashes": stage_hashes,
        "opportunity_hash": opportunity_hash,
        "stage_candidate_counts": {
            stage: int(frame["route"].eq("COVERAGE_CANDIDATE").sum())
            for stage, frame in frames.items()
        },
        "stage_union_counts": {
            stage: int(frame["route"].eq("UNION_BASE").sum())
            for stage, frame in frames.items()
        },
        "maximum_outcome_available_time": max(
            frame["outcome_available_time"].max() for frame in frames.values()
        ).isoformat(),
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "preflight.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--variant", choices=REGISTERED_VARIANTS)
    parser.add_argument("--run-all-stages", action="store_true")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args(argv)
    if args.preflight:
        result = run_preflight(output_root=args.cache_dir)
    elif args.variant:
        output_root = args.cache_dir / "smoke" if args.smoke else args.cache_dir
        stages = STAGE_ORDER if args.run_all_stages else ("development",)
        result = run_variant(
            args.variant,
            output_root=output_root,
            stages=stages,
            max_episodes=args.max_episodes,
        )
    else:
        parser.error("choose --preflight or --variant")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE",
    "CachedCaller",
    "main",
    "resume_variant",
    "run_preflight",
    "run_registered_experiment",
    "run_variant",
]
