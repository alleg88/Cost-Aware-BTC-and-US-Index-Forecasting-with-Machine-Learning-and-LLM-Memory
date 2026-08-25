"""Resumable staged runner for the bounded Reflection Agent v2 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Sequence

import pandas as pd
import requests

from evaluation.economics import economics_summary
from reflection_agent.v2.config import load_v2_config
from reflection_agent.v2.contracts import ProposalOutput, ReflectionOutput
from reflection_agent.v2.memory import (
    NoMemory,
    RealMemory,
    ShuffledMemory,
    StaticPolicyControl,
)
from reflection_agent.v2.leakage import LeakageAuditor
from reflection_agent.v2.opportunities import (
    assign_observation_episodes,
    build_development_opportunities,
    build_exact_opportunities,
)
from reflection_agent.v2.orchestrator import ReflectionOrchestrator
from reflection_agent.v2.policy import ActiveAllowRule, apply_policy
from reflection_agent.v2.prompts import SYSTEM_PROMPT_V2, prompt_hashes
from reflection_agent.v2.transport import DeepSeekSchemaCaller, SchemaCallResult


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "reflection_agent_v2"
DEFAULT_CONFIG = CODE_ROOT / "configs" / "reflection_agent_v2.yaml"
Q2_START = pd.Timestamp("2026-04-01", tz="UTC")


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
    content = "".join(_canonical_json(row) + "\n" for row in rows)
    _atomic_text(path, content)


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
        CODE_ROOT / "evaluation" / "economics.py",
        *sorted((CODE_ROOT / "reflection_agent" / "v2").glob("*.py")),
    ]
    return _hash(
        {
            str(path.relative_to(CODE_ROOT)).replace("\\", "/"): _sha256_file(path)
            for path in paths
        }
    )


def _validate_stage_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, str]:
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
        "outcome_available_time",
    ):
        output[column] = pd.to_datetime(output[column], utc=True)
    if output["opportunity_id"].duplicated().any():
        raise ValueError("runner opportunity IDs must be unique")
    stages = output["stage"].drop_duplicates().tolist()
    if len(stages) != 1 or stages[0] not in {"development", "h1", "forward"}:
        raise ValueError("runner requires one registered stage")
    if output["outcome_available_time"].ge(Q2_START).any():
        raise ValueError("Q2 lockbox timestamp entered the runner")
    if not output["feature_available_time"].le(output["decision_time"]).all():
        raise ValueError("future feature entered the runner")
    if not output["decision_time"].lt(output["outcome_available_time"]).all():
        raise ValueError("outcome availability is not later than decision")
    return output, str(stages[0])


class CachedCaller:
    """Persist every terminal structured call so a partial run can replay locally."""

    def __init__(
        self,
        inner: Any,
        database_path: Path,
        *,
        protocol_hash: str,
    ) -> None:
        self.inner = inner
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
                "SELECT protocol_hash, payload_json FROM runner_call_cache WHERE cache_key = ?",
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
                status="cached",
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
            "value": result.value.model_dump(mode="json") if result.value is not None else None,
            "raw_content": result.raw_content,
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "schema_hash": result.schema_hash,
            "attempts": result.attempts,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
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
    if variant_id in {"static_add_all", "union_baseline"}:
        return StaticPolicyControl(real)
    raise ValueError(f"unknown registered variant: {variant_id}")


def _episode_table(assigned: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "observation_episode_id",
        "fold_id",
        "episode_cutoff_utc",
        "episode_status",
        "episode_can_propose",
        "episode_opportunity_count",
        "episode_long_count",
        "episode_short_count",
        "episode_reentry_count",
        "episode_reentry_long_count",
        "episode_reentry_short_count",
    ]
    return (
        assigned.loc[:, columns]
        .drop_duplicates("observation_episode_id")
        .sort_values(["fold_id", "episode_cutoff_utc"], kind="stable")
        .reset_index(drop=True)
    )


def _policy_metrics(selected: pd.DataFrame, stage_frame: pd.DataFrame) -> dict[str, Any]:
    trades = selected.loc[selected["selected"]].copy()
    start = stage_frame["decision_time"].min().floor("15min")
    end = stage_frame["outcome_available_time"].max().ceil("15min")
    per_bar = pd.Series(0.0, index=pd.date_range(start, end, freq="15min"))
    booked = trades.groupby("entry_time")["net_return"].sum()
    per_bar.loc[booked.index] = booked.to_numpy(float)
    economics = economics_summary(per_bar)
    net = trades["net_return"].astype(float)
    gross = trades["gross_return"].astype(float)
    return {
        "selected_trades": len(trades),
        "selected_long_trades": int(trades["side"].eq("LONG").sum()),
        "selected_short_trades": int(trades["side"].eq("SHORT").sum()),
        "gross_return": float(gross.sum()),
        "cost_return": float((gross - net).sum()),
        "net_return": float(net.sum()),
        "long_net_return": float(net.loc[trades["side"].eq("LONG")].sum()),
        "short_net_return": float(net.loc[trades["side"].eq("SHORT")].sum()),
        "sortino": float(economics["sortino"]),
        "sharpe": float(economics["sharpe"]),
        "max_drawdown": float(economics["max_drawdown"]),
    }


def _apply_policy_transition(
    closure: Any,
    *,
    active_rules: Sequence[ActiveAllowRule],
    policy_history: Sequence[ActiveAllowRule],
) -> tuple[list[ActiveAllowRule], list[ActiveAllowRule]]:
    """Apply one closed shadow decision to live and historical policy state."""

    next_active = list(active_rules)
    next_history = list(policy_history)
    if closure.new_active_rule is not None:
        next_active.append(closure.new_active_rule)
        next_history.append(closure.new_active_rule)
    if closure.removed_rule_id is not None:
        if closure.evaluation.shadow_cutoff_utc is None:
            raise ValueError("a removed rule requires an evaluated shadow cutoff")
        if not any(rule.rule_id == closure.removed_rule_id for rule in next_active):
            raise ValueError("a removed rule must be active at transition time")
        removal_time = pd.Timestamp(
            closure.evaluation.shadow_cutoff_utc
        ) + pd.Timedelta(microseconds=1)
        next_active = [
            rule for rule in next_active if rule.rule_id != closure.removed_rule_id
        ]
        next_history = [
            rule.model_copy(
                update={"deactivates_at_utc": removal_time.to_pydatetime()}
            )
            if rule.rule_id == closure.removed_rule_id
            else rule
            for rule in next_history
        ]
    return next_active, next_history


def _record_closed_evaluation(closure: Any, evaluations: list[dict[str, Any]]) -> bool:
    if closure.status != "closed":
        return False
    evaluations.append(closure.evaluation.model_dump(mode="json"))
    return True


def _artifact_hashes(root: Path) -> dict[str, str]:
    return {
        path.name: _sha256_file(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name)
        if path.is_file() and path.name != "manifest.json"
    }


def _resume_if_complete(
    root: Path,
    *,
    input_hash: str,
    protocol_hash: str,
) -> dict[str, Any] | None:
    manifest_path = root / "manifest.json"
    summary_path = root / "evaluation_summary.json"
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
    opportunities: pd.DataFrame | None = None,
    caller: Any | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
    max_episodes: int | None = None,
) -> dict[str, Any]:
    config = load_v2_config(config_path)
    if variant_id not in config.registered_variants:
        raise ValueError(f"variant is not registered: {variant_id}")
    raw = opportunities if opportunities is not None else build_development_opportunities()
    frame, stage = _validate_stage_frame(raw)
    if stage == "forward" and variant_id.startswith("reflection_"):
        raise ValueError("forward agent requires a frozen H1 snapshot")
    input_hash = _frame_hash(frame)
    implementation_hash = _implementation_hash()
    protocol_hash = _hash(
        {
            "config": config.model_dump(mode="json"),
            "implementation_hash": implementation_hash,
            "prompt_hashes": prompt_hashes(),
            "proposal_schema": ProposalOutput.model_json_schema(),
            "reflection_schema": ReflectionOutput.model_json_schema(),
            "input_hash": input_hash,
            "stage": stage,
            "variant_id": variant_id,
        }
    )
    root = Path(output_root).resolve() / stage / variant_id
    resumed = _resume_if_complete(
        root, input_hash=input_hash, protocol_hash=protocol_hash
    )
    if resumed is not None:
        return resumed
    root.mkdir(parents=True, exist_ok=True)

    scope = {
        "development": "development_2021_2024",
        "h1": "exact_h1_2025",
        "forward": "exact_forward_2025_2026",
    }[stage]
    real_memory = RealMemory(
        root / "agent_state.sqlite",
        protocol_hash=protocol_hash,
        protocol_scope=scope,
    )
    memory = _memory_control(variant_id, real_memory)
    if caller is None:
        caller = DeepSeekSchemaCaller(
            config,
            call_log_path=root / "call_log.jsonl",
        )
    cached_caller = CachedCaller(
        caller,
        root / "agent_state.sqlite",
        protocol_hash=protocol_hash,
    )
    auditor_path = root / "prompt_audit.jsonl"
    orchestrator = ReflectionOrchestrator(
        caller=cached_caller,
        auditor=LeakageAuditor(auditor_path),
        memory=memory,
        protocol_hash=protocol_hash,
    )

    assigned = assign_observation_episodes(
        frame,
        min_opportunities=config.episode_min_opportunities,
        max_opportunities=config.episode_max_opportunities,
        min_reentry_opportunities=config.episode_min_reentry_opportunities,
        min_reentry_per_side=config.episode_min_per_side,
    )
    episodes = _episode_table(assigned)
    if max_episodes is not None:
        episodes = episodes.head(max_episodes).copy()
        included_ids = set(episodes["observation_episode_id"])
        assigned = assigned.loc[
            assigned["observation_episode_id"].isin(included_ids)
        ].copy()
        frame = frame.loc[frame["opportunity_id"].isin(set(assigned["opportunity_id"]))].copy()

    transitions: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    policy_history: list[ActiveAllowRule] = []
    active_rules: list[ActiveAllowRule] = []
    open_candidate = None
    proposal_calls = 0
    reflection_calls = 0

    is_agent_variant = variant_id.startswith("reflection_")
    if is_agent_variant:
        for episode_number, row in enumerate(episodes.itertuples(index=False), start=1):
            episode_id = str(row.observation_episode_id)
            episode_frame = assigned.loc[
                assigned["observation_episode_id"].eq(episode_id)
            ].copy()
            cutoff = pd.Timestamp(row.episode_cutoff_utc)
            if open_candidate is not None:
                available = frame.loc[
                    frame["outcome_available_time"].le(cutoff)
                ].copy()
                closure = orchestrator.close_candidate(
                    open_candidate,
                    available,
                    active_rules=active_rules,
                    episode_number=episode_number,
                )
                if closure.status == "closed":
                    _record_closed_evaluation(closure, evaluations)
                    if closure.reflection is not None:
                        reflection_calls += 1
                        transitions.append(
                            {
                                "role": "reflection",
                                "candidate_id": open_candidate.candidate_id,
                                "decision": closure.evaluation.decision,
                            }
                        )
                    active_rules, policy_history = _apply_policy_transition(
                        closure,
                        active_rules=active_rules,
                        policy_history=policy_history,
                    )
                    open_candidate = None
            if bool(row.episode_can_propose) and open_candidate is None:
                result = orchestrator.propose_episode(
                    episode_frame,
                    episode_id=episode_id,
                    episode_number=episode_number,
                    active_rules=active_rules,
                )
                if result.call_status is not None:
                    proposal_calls += 1
                transitions.append(
                    {
                        "role": "proposal",
                        "episode_id": episode_id,
                        "status": result.status,
                        "candidate_id": (
                            result.candidate.candidate_id
                            if result.candidate is not None
                            else None
                        ),
                    }
                )
                open_candidate = result.candidate

            next_fold = (
                int(episodes.iloc[episode_number]["fold_id"])
                if episode_number < len(episodes)
                else None
            )
            if open_candidate is not None and next_fold != int(row.fold_id):
                fold_frame = frame.loc[frame["fold_id"].eq(int(row.fold_id))]
                closure = orchestrator.close_candidate(
                    open_candidate,
                    fold_frame,
                    active_rules=active_rules,
                    episode_number=episode_number,
                    force_close=True,
                )
                if not _record_closed_evaluation(closure, evaluations):
                    raise AssertionError("forced candidate closure remained open")
                if closure.reflection is not None:
                    reflection_calls += 1
                    transitions.append(
                        {
                            "role": "reflection",
                            "candidate_id": open_candidate.candidate_id,
                            "decision": closure.evaluation.decision,
                        }
                    )
                active_rules, policy_history = _apply_policy_transition(
                    closure,
                    active_rules=active_rules,
                    policy_history=policy_history,
                )
                open_candidate = None

    if variant_id == "static_add_all":
        selected = apply_policy(frame, [], static_add_all=True)
    elif variant_id == "union_baseline":
        selected = apply_policy(frame, [])
    else:
        selected = apply_policy(frame, policy_history)

    metrics = _policy_metrics(selected, frame)
    memories = real_memory.all_cards()
    memory_rows = [memory_card.model_dump(mode="json") for memory_card in memories]
    if memory_rows:
        memory_frame = pd.DataFrame(memory_rows)
    else:
        memory_frame = pd.DataFrame(
            columns=[
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
            ]
        )
    _write_parquet(root / "opportunity_ledger.parquet", selected)
    _write_parquet(root / "episode_ledger.parquet", episodes)
    _write_parquet(root / "memory_ledger.parquet", memory_frame)
    _write_jsonl(
        root / "policy_history.jsonl",
        [rule.model_dump(mode="json") for rule in policy_history],
    )
    _write_jsonl(root / "transitions.jsonl", transitions)
    if not (root / "call_log.jsonl").exists():
        _atomic_text(root / "call_log.jsonl", "")
    if not auditor_path.exists():
        _atomic_text(auditor_path, "")
    summary: dict[str, Any] = {
        "status": "complete",
        "resumed": False,
        "stage": stage,
        "variant_id": variant_id,
        "input_hash": input_hash,
        "implementation_hash": implementation_hash,
        "protocol_hash": protocol_hash,
        "opportunities": len(frame),
        "union_base_trades": int(frame["route"].eq("UNION_BASE").sum()),
        "eligible_reentries": int(frame["route"].eq("REENTRY").sum()),
        "episodes": len(episodes),
        "proposal_calls": proposal_calls,
        "reflection_calls": reflection_calls,
        "candidate_evaluations": len(evaluations),
        "promoted_rules": len(policy_history),
        "memory_records": len(memories),
        "maximum_outcome_available_time": frame[
            "outcome_available_time"
        ].max().isoformat(),
        "lockbox_2026_q2_used": False,
        **metrics,
        "evaluations": evaluations,
    }
    _write_json(root / "evaluation_summary.json", summary)
    _write_json(
        root / "run_state.json",
        {
            "status": "complete",
            "input_hash": input_hash,
            "implementation_hash": implementation_hash,
            "protocol_hash": protocol_hash,
            "lockbox_2026_q2_used": False,
        },
    )
    manifest = {
        "stage": stage,
        "variant_id": variant_id,
        "input_hash": input_hash,
        "implementation_hash": implementation_hash,
        "protocol_hash": protocol_hash,
        "model": config.model if is_agent_variant else None,
        "prompt_hashes": prompt_hashes(),
        "schema_hashes": {
            "proposal": _hash(ProposalOutput.model_json_schema()),
            "reflection": _hash(ReflectionOutput.model_json_schema()),
        },
        "artifact_hashes": _artifact_hashes(root),
        "maximum_outcome_available_time": summary[
            "maximum_outcome_available_time"
        ],
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "manifest.json", manifest)
    return summary


def run_frozen_forward(
    variant_id: str,
    *,
    output_root: str | Path = CACHE,
    opportunities: pd.DataFrame | None = None,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Apply a verified H1 policy/memory snapshot to forward data without learning."""

    config = load_v2_config(config_path)
    if variant_id not in config.registered_variants or not variant_id.startswith(
        "reflection_"
    ):
        raise ValueError("frozen forward requires a registered reflection variant")
    raw = opportunities if opportunities is not None else build_exact_opportunities("forward")
    frame, stage = _validate_stage_frame(raw)
    if stage != "forward":
        raise ValueError("frozen forward requires the registered forward opportunity frame")

    output_base = Path(output_root).resolve()
    h1_root = output_base / "h1" / variant_id
    h1_manifest_path = h1_root / "manifest.json"
    if not h1_manifest_path.is_file():
        raise ValueError("frozen forward requires a completed H1 snapshot")
    h1_manifest = json.loads(h1_manifest_path.read_text(encoding="utf-8"))
    if (
        h1_manifest.get("stage") != "h1"
        or h1_manifest.get("variant_id") != variant_id
        or h1_manifest.get("lockbox_2026_q2_used") is not False
    ):
        raise ValueError("H1 snapshot manifest is not registered for this variant")
    freeze_at = pd.Timestamp(config.forward_start_utc)
    h1_maximum = pd.Timestamp(h1_manifest["maximum_outcome_available_time"])
    if h1_maximum >= freeze_at:
        raise ValueError("H1 snapshot reaches or crosses the forward freeze timestamp")

    required_snapshot_files = (
        "agent_state.sqlite",
        "memory_ledger.parquet",
        "policy_history.jsonl",
    )
    source_hashes: dict[str, str] = {}
    for name in required_snapshot_files:
        path = h1_root / name
        expected = h1_manifest.get("artifact_hashes", {}).get(name)
        if not path.is_file() or expected is None or _sha256_file(path) != expected:
            raise ValueError(f"H1 snapshot artifact drifted: {name}")
        source_hashes[name] = expected
    source_manifest_hash = _sha256_file(h1_manifest_path)

    policy_rows = [
        json.loads(line)
        for line in (h1_root / "policy_history.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    policy_history = [ActiveAllowRule.model_validate(row) for row in policy_rows]
    if any(pd.Timestamp(rule.activates_at_utc) >= freeze_at for rule in policy_history):
        raise ValueError("H1 policy activates at or after the forward freeze timestamp")
    memory_frame = pd.read_parquet(h1_root / "memory_ledger.parquet")
    if len(memory_frame) and pd.to_datetime(
        memory_frame["created_at_utc"], utc=True
    ).ge(freeze_at).any():
        raise ValueError("H1 memory was created at or after the forward freeze timestamp")

    input_hash = _frame_hash(frame)
    implementation_hash = _implementation_hash()
    snapshot_payload = {
        "source_stage": "h1",
        "source_variant": variant_id,
        "freeze_at_utc": freeze_at.isoformat(),
        "source_manifest_hash": source_manifest_hash,
        "source_artifact_hashes": source_hashes,
    }
    protocol_hash = _hash(
        {
            "mode": "frozen_forward",
            "config": config.model_dump(mode="json"),
            "implementation_hash": implementation_hash,
            "input_hash": input_hash,
            "snapshot": snapshot_payload,
            "variant_id": variant_id,
        }
    )
    root = output_base / "forward" / variant_id
    resumed = _resume_if_complete(
        root, input_hash=input_hash, protocol_hash=protocol_hash
    )
    if resumed is not None:
        return resumed

    selected = apply_policy(frame, policy_history)
    rules_by_id = {rule.rule_id: rule for rule in policy_history}
    auditor = LeakageAuditor()
    activation_rows: list[dict[str, Any]] = []
    selected_reentries = selected.loc[
        selected["route"].eq("REENTRY") & selected["selected"]
    ]
    for row in selected_reentries.itertuples(index=False):
        for rule_id in str(row.selected_rule_ids).split(","):
            if not rule_id:
                continue
            rule = rules_by_id.get(rule_id)
            if rule is None:
                raise ValueError("forward opportunity cites an unknown frozen rule")
            auditor.audit_activation(
                activates_at_utc=rule.activates_at_utc,
                opportunity_decision_time=pd.Timestamp(row.decision_time).to_pydatetime(),
                stage="forward",
                protocol_scope="exact_forward_2025_2026",
            )
            activation_rows.append(
                {
                    "opportunity_id": row.opportunity_id,
                    "rule_id": rule_id,
                    "activates_at_utc": rule.activates_at_utc,
                    "decision_time": row.decision_time,
                    "passed": True,
                }
            )

    assigned = assign_observation_episodes(
        frame,
        min_opportunities=config.episode_min_opportunities,
        max_opportunities=config.episode_max_opportunities,
        min_reentry_opportunities=config.episode_min_reentry_opportunities,
        min_reentry_per_side=config.episode_min_per_side,
    )
    episodes = _episode_table(assigned)
    metrics = _policy_metrics(selected, frame)

    root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(h1_root / "agent_state.sqlite", root / "agent_state_snapshot.sqlite")
    shutil.copy2(h1_root / "memory_ledger.parquet", root / "memory_ledger.parquet")
    shutil.copy2(h1_root / "policy_history.jsonl", root / "policy_history.jsonl")
    _write_parquet(root / "opportunity_ledger.parquet", selected)
    _write_parquet(root / "episode_ledger.parquet", episodes)
    _write_jsonl(root / "activation_audit.jsonl", activation_rows)
    _write_jsonl(
        root / "transitions.jsonl",
        [
            {
                "role": "frozen_policy",
                "opportunity_id": row["opportunity_id"],
                "rule_ids": row["selected_rule_ids"].split(","),
            }
            for row in selected_reentries.to_dict(orient="records")
        ],
    )
    _atomic_text(root / "call_log.jsonl", "")
    _atomic_text(root / "prompt_audit.jsonl", "")
    _write_json(root / "snapshot_manifest.json", snapshot_payload)

    summary: dict[str, Any] = {
        "status": "complete",
        "resumed": False,
        "stage": "forward",
        "variant_id": variant_id,
        "input_hash": input_hash,
        "implementation_hash": implementation_hash,
        "protocol_hash": protocol_hash,
        "snapshot_source_manifest_hash": source_manifest_hash,
        "snapshot_freeze_at_utc": freeze_at.isoformat(),
        "opportunities": len(frame),
        "union_base_trades": int(frame["route"].eq("UNION_BASE").sum()),
        "eligible_reentries": int(frame["route"].eq("REENTRY").sum()),
        "episodes": len(episodes),
        "proposal_calls": 0,
        "reflection_calls": 0,
        "candidate_evaluations": 0,
        "promoted_rules": len(policy_history),
        "memory_records": len(memory_frame),
        "maximum_outcome_available_time": frame[
            "outcome_available_time"
        ].max().isoformat(),
        "lockbox_2026_q2_used": False,
        **metrics,
        "evaluations": [],
    }
    _write_json(root / "evaluation_summary.json", summary)
    _write_json(
        root / "run_state.json",
        {
            "status": "complete",
            "input_hash": input_hash,
            "implementation_hash": implementation_hash,
            "protocol_hash": protocol_hash,
            "snapshot_source_manifest_hash": source_manifest_hash,
            "lockbox_2026_q2_used": False,
        },
    )
    manifest = {
        "stage": "forward",
        "variant_id": variant_id,
        "input_hash": input_hash,
        "implementation_hash": implementation_hash,
        "protocol_hash": protocol_hash,
        "model": config.model,
        "mode": "frozen_h1_snapshot",
        "snapshot_source_manifest_hash": source_manifest_hash,
        "artifact_hashes": _artifact_hashes(root),
        "maximum_outcome_available_time": summary[
            "maximum_outcome_available_time"
        ],
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "manifest.json", manifest)
    return summary


def _version_tuple(value: str) -> tuple[int, ...]:
    pieces = value.strip().lstrip("v").split(".")
    return tuple(int(piece) for piece in pieces[:3])


def _model_value(record: Any, key: str, default: Any = None) -> Any:
    if isinstance(record, dict):
        return record.get(key, default)
    return getattr(record, key, default)


def _live_model_record(model: str) -> dict[str, Any]:
    version_response = requests.get("http://localhost:11434/api/version", timeout=10)
    version_response.raise_for_status()
    version = str(version_response.json()["version"])
    import ollama

    listing = ollama.list()
    models = _model_value(listing, "models", [])
    match = next(
        (
            item
            for item in models
            if _model_value(item, "model", _model_value(item, "name")) == model
        ),
        None,
    )
    if match is None:
        raise RuntimeError(f"exact Ollama model is not resolved locally: {model}")
    shown = ollama.show(model)
    capabilities = list(_model_value(shown, "capabilities", []) or [])
    return {
        "model": model,
        "digest": str(_model_value(match, "digest", "")),
        "capabilities": capabilities,
        "ollama_version": version,
    }


def run_preflight(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    model_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_v2_config(config_path)
    record = model_record or _live_model_record(config.model)
    if record.get("model") != config.model:
        raise RuntimeError("preflight resolved a different model tag")
    digest = str(record.get("digest", ""))
    if len(digest) != 64:
        raise RuntimeError("preflight model digest is missing")
    version = str(record.get("ollama_version", "0.0.0"))
    if _version_tuple(version) < (0, 32, 5):
        raise RuntimeError("Ollama 0.32.5 or newer is required")
    capabilities = set(record.get("capabilities", []))
    if "thinking" not in capabilities:
        raise RuntimeError("exact model does not advertise thinking support")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if caller is None:
        caller = DeepSeekSchemaCaller(
            config, call_log_path=root / "preflight_calls.jsonl"
        )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_V2},
        {
            "role": "user",
            "content": (
                "TASK: PREFLIGHT_STRICT_SCHEMA\n"
                "Return this exact JSON object with no renamed, omitted, or extra keys:\n"
                + _canonical_json(
                    {
                        "schema_version": "2.0",
                        "source_episode_id": "preflight-episode",
                        "decision": "NO_CHANGE",
                        "diagnosis_code": "INSUFFICIENT_EVIDENCE",
                        "evidence_ids": ["preflight-evidence"],
                        "memory_ids_used": [],
                        "proposed_rule": None,
                        "target_rule_id": None,
                        "hypothesis": None,
                        "falsifiers": [],
                        "confidence": "LOW",
                    }
                )
            ),
        },
    ]
    allowed_ids = {
        "source_episode_id": "preflight-episode",
        "evidence_ids": ["preflight-evidence"],
        "memory_ids": [],
        "rule_ids": [],
    }
    probe = caller.call(
        role="proposal",
        messages=messages,
        response_model=ProposalOutput,
        allowed_ids=allowed_ids,
    )
    if probe.value is None:
        raise RuntimeError(f"strict schema preflight failed: {probe.errors}")
    if (
        probe.value.source_episode_id != "preflight-episode"
        or set(probe.value.evidence_ids) - {"preflight-evidence"}
        or probe.value.decision != "NO_CHANGE"
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
            "proposal": _hash(ProposalOutput.model_json_schema()),
            "reflection": _hash(ReflectionOutput.model_json_schema()),
        },
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "preflight.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--variant", choices=[
        "reflection_real_memory",
        "reflection_no_memory",
        "reflection_shuffled_memory",
        "static_add_all",
        "union_baseline",
    ])
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage", choices=["development", "h1", "forward"], default="development"
    )
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args(argv)
    output_root = args.cache_dir / "smoke" if args.smoke else args.cache_dir
    if args.preflight:
        result = run_preflight(output_root=args.cache_dir)
    elif args.variant:
        opportunities = (
            None
            if args.stage == "development"
            else build_exact_opportunities(args.stage)
        )
        if args.stage == "forward" and args.variant.startswith("reflection_"):
            if args.max_episodes is not None or args.smoke:
                parser.error("frozen forward does not support smoke/episode truncation")
            result = run_frozen_forward(
                args.variant,
                output_root=output_root,
                opportunities=opportunities,
            )
        else:
            result = run_variant(
                args.variant,
                output_root=output_root,
                opportunities=opportunities,
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
    "run_preflight",
    "run_frozen_forward",
    "run_variant",
]
