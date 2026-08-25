"""Independently reconcile the continuous Reflection Agent v3 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from reflection_agent.v3.contracts import MemoryCard
from reflection_agent.v3.evaluator import GateDecision
from reflection_agent.v3.policy import ActiveAllowRule


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = CODE_ROOT / "experiments" / "cache" / "reflection_agent_v3"
Q2_START = pd.Timestamp("2026-04-01", tz="UTC")
STAGES = ("development", "h1", "forward")
VARIANTS = (
    "reflection_real_memory",
    "reflection_no_memory",
    "reflection_shuffled_memory",
    "static_high_extra",
    "static_all_extra",
    "union_baseline",
)
AGENT_VARIANTS = VARIANTS[:3]
CONTROL_VARIANTS = VARIANTS[3:]
EXPECTED_COUNTS = {
    "development": {
        "opportunities": 3456,
        "union_base": 948,
        "coverage_candidates": 2508,
    },
    "h1": {
        "opportunities": 1024,
        "union_base": 88,
        "coverage_candidates": 936,
    },
    "forward": {
        "opportunities": 1002,
        "union_base": 74,
        "coverage_candidates": 928,
    },
}
OPPORTUNITY_COLUMNS = (
    "opportunity_id",
    "stage",
    "source_role",
    "fold_id",
    "row_key",
    "source_artifact_hash",
    "decision_time",
    "feature_available_time",
    "outcome_available_time",
    "entry_time",
    "exit_time",
    "route",
    "side",
    "confidence_tier",
    "signal_run_bucket",
    "gross_return",
    "net_return",
    "round_trip_cost",
    "exit_reason",
    "vol_regime",
    "trend_regime",
    "funding_regime",
    "oi_regime",
    "path_complete",
)
SUMMARY_METRICS = (
    "selected_trades",
    "selected_long_trades",
    "selected_short_trades",
    "additional_trades",
    "trades_per_effective_day",
    "gross_return",
    "cost_return",
    "net_return",
    "long_net_return",
    "short_net_return",
    "sortino",
    "sharpe",
    "max_drawdown",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    normalized = frame.sort_values("opportunity_id", kind="stable").copy()
    for column in normalized.columns:
        if isinstance(
            normalized[column].dtype, pd.DatetimeTZDtype
        ) or pd.api.types.is_datetime64_any_dtype(normalized[column]):
            normalized[column] = pd.to_datetime(normalized[column], utc=True).map(
                lambda value: value.isoformat() if pd.notna(value) else None
            )
    records = normalized.where(pd.notna(normalized), None).to_dict(orient="records")
    return hashlib.sha256(
        _canonical_json(
            {"columns": list(normalized.columns), "records": records}
        ).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL artifact: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _same_number(left: Any, right: Any, *, atol: float = 1e-10) -> bool:
    left_value = float(left)
    right_value = float(right)
    if np.isnan(left_value) and np.isnan(right_value):
        return True
    return bool(np.isclose(left_value, right_value, rtol=0.0, atol=atol))


def paired_policy_delta(
    variant: pd.DataFrame,
    union: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260812,
) -> dict[str, Any]:
    """Return opportunity-paired deltas and deterministic block-bootstrap CIs."""

    required = {
        "opportunity_id",
        "route",
        "side",
        "entry_time",
        "net_return",
        "selected",
    }
    for name, frame in (("variant", variant), ("union", union)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} ledger lacks paired columns: {missing}")
        if frame["opportunity_id"].duplicated().any():
            raise ValueError(f"{name} opportunity IDs are not unique")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")

    left = variant.set_index("opportunity_id").sort_index()
    right = union.set_index("opportunity_id").sort_index()
    if not left.index.equals(right.index):
        raise ValueError("opportunity ledger drift: IDs differ")
    if not left[["route", "side"]].equals(right[["route", "side"]]):
        raise ValueError("opportunity ledger drift: route or side differs")
    left_time = pd.to_datetime(left["entry_time"], utc=True)
    right_time = pd.to_datetime(right["entry_time"], utc=True)
    if not left_time.equals(right_time) or not np.allclose(
        left["net_return"].to_numpy(float),
        right["net_return"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("opportunity ledger drift: time or return differs")

    selection_delta = left["selected"].astype(int) - right["selected"].astype(int)
    if selection_delta.lt(0).any():
        raise ValueError("Union selection was removed by a variant")
    return_delta = left["net_return"].astype(float) * selection_delta
    month = left_time.dt.tz_convert(None).dt.to_period("M").astype(str)
    if "fold_id" in left and left["stage"].eq("development").all():
        block = left["fold_id"].astype(str) + ":" + month
        bootstrap_unit = "development_fold_month"
    else:
        block = month
        bootstrap_unit = "calendar_month"
    block_return = return_delta.groupby(block).sum().sort_index()
    block_trades = selection_delta.groupby(block).sum().reindex(block_return.index)
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0, len(block_return), size=(bootstrap_samples, len(block_return))
    )
    return_distribution = block_return.to_numpy(float)[draws].sum(axis=1)
    trade_distribution = block_trades.to_numpy(int)[draws].sum(axis=1)
    return_low, return_high = np.quantile(return_distribution, [0.025, 0.975])
    trade_low, trade_high = np.quantile(trade_distribution, [0.025, 0.975])
    side = left["side"].astype(str)
    return {
        "selected_trade_delta": int(selection_delta.sum()),
        "selected_long_trade_delta": int(
            selection_delta.loc[side.eq("LONG")].sum()
        ),
        "selected_short_trade_delta": int(
            selection_delta.loc[side.eq("SHORT")].sum()
        ),
        "net_return_delta": float(return_delta.sum()),
        "long_net_return_delta": float(return_delta.loc[side.eq("LONG")].sum()),
        "short_net_return_delta": float(return_delta.loc[side.eq("SHORT")].sum()),
        "blocks": len(block_return),
        "positive_block_share": float((block_return > 0.0).mean()),
        "ci95_low": float(return_low),
        "ci95_high": float(return_high),
        "trade_delta_ci95_low": float(trade_low),
        "trade_delta_ci95_high": float(trade_high),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": bootstrap_unit,
        "bootstrap_seed": seed,
    }


def _recomputed_metrics(
    ledger: pd.DataFrame, per_bar: pd.DataFrame
) -> dict[str, float | int]:
    trades = ledger.loc[ledger["selected"].astype(bool)].copy()
    series = pd.Series(
        per_bar["net_return"].to_numpy(float),
        index=pd.to_datetime(per_bar["timestamp"], utc=True),
        name="net_return",
    )
    if not _same_number(series.sum(), trades["net_return"].sum(), atol=1e-12):
        raise ValueError("per-bar return does not reconcile to selected trades")
    economics = economics_summary(series)
    effective_days = max(
        (
            pd.to_datetime(ledger["decision_time"], utc=True).max()
            - pd.to_datetime(ledger["decision_time"], utc=True).min()
        ).total_seconds()
        / 86_400.0,
        1.0 / 96.0,
    )
    gross = trades["gross_return"].astype(float)
    net = trades["net_return"].astype(float)
    return {
        "selected_trades": len(trades),
        "selected_long_trades": int(trades["side"].eq("LONG").sum()),
        "selected_short_trades": int(trades["side"].eq("SHORT").sum()),
        "additional_trades": int(
            trades["route"].eq("COVERAGE_CANDIDATE").sum()
        ),
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


def _concentration(ledger: pd.DataFrame, stage: str) -> dict[str, Any]:
    additional = ledger.loc[
        ledger["selected"].astype(bool)
        & ledger["route"].eq("COVERAGE_CANDIDATE")
    ].copy()
    if additional.empty:
        return {"blocks": 0, "maximum_share": 0.0, "unit": "fold" if stage == "development" else "month"}
    if stage == "development":
        blocks = additional["fold_id"].astype(str)
        unit = "fold"
    else:
        blocks = (
            pd.to_datetime(additional["entry_time"], utc=True)
            .dt.tz_convert(None)
            .dt.to_period("M")
            .astype(str)
        )
        unit = "month"
    counts = blocks.value_counts()
    return {
        "blocks": len(counts),
        "maximum_share": float(counts.max() / counts.sum()),
        "unit": unit,
    }


def _verify_stage(
    root: Path,
    manifest: dict[str, Any],
    overall: dict[str, Any],
    variant: str,
    stage: str,
) -> dict[str, Any]:
    stage_root = root / "stages" / stage
    summary = _read_json(stage_root / "summary.json")
    if summary != overall["stage_summaries"][stage]:
        raise ValueError(f"overall/stage summary drift: {variant}/{stage}")
    if summary.get("stage") != stage or summary.get("status") not in {
        "complete",
        "INVALID_TRANSPORT_COVERAGE",
    }:
        raise ValueError(f"stage identity/status mismatch: {variant}/{stage}")
    if summary.get("lockbox_2026_q2_used") is not False:
        raise ValueError(f"Q2 flag changed: {variant}/{stage}")
    if pd.Timestamp(summary["maximum_outcome_available_time"]) >= Q2_START:
        raise ValueError(f"Q2 timestamp entered summary: {variant}/{stage}")

    ledger = pd.read_parquet(stage_root / "selected_ledger.parquet")
    missing = sorted(set(OPPORTUNITY_COLUMNS).difference(ledger.columns))
    if missing:
        raise ValueError(f"selected ledger lacks columns: {variant}/{stage}/{missing}")
    for column in (
        "decision_time",
        "feature_available_time",
        "outcome_available_time",
        "entry_time",
        "exit_time",
    ):
        ledger[column] = pd.to_datetime(ledger[column], utc=True)
    if ledger["opportunity_id"].duplicated().any():
        raise ValueError(f"duplicate opportunity ID: {variant}/{stage}")
    if set(ledger["stage"]) != {stage}:
        raise ValueError(f"stage ledger identity mismatch: {variant}/{stage}")
    if ledger["outcome_available_time"].ge(Q2_START).any():
        raise ValueError(f"Q2 row entered ledger: {variant}/{stage}")
    if not ledger["feature_available_time"].le(ledger["decision_time"]).all():
        raise ValueError(f"future feature entered ledger: {variant}/{stage}")
    if not ledger["decision_time"].lt(ledger["outcome_available_time"]).all():
        raise ValueError(f"non-causal outcome entered ledger: {variant}/{stage}")
    if not np.allclose(
        ledger["gross_return"].astype(float)
        - ledger["net_return"].astype(float),
        ledger["round_trip_cost"].astype(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(f"gross/net/cost mismatch: {variant}/{stage}")
    if not np.allclose(
        ledger["round_trip_cost"].astype(float), 0.001, rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"registered fee drift: {variant}/{stage}")
    if _frame_hash(ledger.loc[:, OPPORTUNITY_COLUMNS]) != manifest["stage_hashes"][stage]:
        raise ValueError(f"stage opportunity hash mismatch: {variant}/{stage}")
    union_rows = ledger["route"].eq("UNION_BASE")
    if not ledger.loc[union_rows, "selected"].astype(bool).all():
        raise ValueError(f"Union trade removed: {variant}/{stage}")
    selected_candidates = ledger["selected"].astype(bool) & ~union_rows
    if not ledger.loc[selected_candidates, "policy_eligible"].astype(bool).all():
        raise ValueError(f"candidate traded without policy: {variant}/{stage}")
    if ledger.loc[selected_candidates, "selected_rule_ids"].astype(str).eq("").any():
        raise ValueError(f"candidate traded without rule ID: {variant}/{stage}")

    per_bar = pd.read_parquet(stage_root / "per_bar_returns.parquet")
    per_bar["timestamp"] = pd.to_datetime(per_bar["timestamp"], utc=True)
    if per_bar["timestamp"].ge(Q2_START).any():
        raise ValueError(f"Q2 bar entered returns: {variant}/{stage}")
    metrics = _recomputed_metrics(ledger, per_bar)
    for key in SUMMARY_METRICS:
        if not _same_number(metrics[key], summary[key]):
            raise ValueError(f"metric mismatch {key}: {variant}/{stage}")
    observed_counts = {
        "opportunities": len(ledger),
        "union_base": int(union_rows.sum()),
        "coverage_candidates": int((~union_rows).sum()),
    }
    if observed_counts != EXPECTED_COUNTS[stage]:
        raise ValueError(f"registered opportunity count drift: {variant}/{stage}")
    if summary["union_base_trades"] != observed_counts["union_base"]:
        raise ValueError(f"Union count summary drift: {variant}/{stage}")

    episodes = pd.read_parquet(stage_root / "episode_ledger.parquet")
    if len(episodes) != summary["episodes"]:
        raise ValueError(f"episode count mismatch: {variant}/{stage}")
    complete = episodes["episode_can_propose"].astype(bool)
    if not episodes.loc[complete, "episode_candidate_count"].between(20, 30).all():
        raise ValueError(f"complete episode size drift: {variant}/{stage}")
    if not episodes.loc[complete, "episode_long_candidate_count"].ge(6).all():
        raise ValueError(f"complete episode LONG support drift: {variant}/{stage}")
    if not episodes.loc[complete, "episode_short_candidate_count"].ge(6).all():
        raise ValueError(f"complete episode SHORT support drift: {variant}/{stage}")
    if not episodes.loc[complete, "episode_status"].eq("COMPLETE").all():
        raise ValueError(f"episode status drift: {variant}/{stage}")

    trades = ledger.loc[ledger["selected"].astype(bool)]
    detail = {
        **summary,
        "win_rate": float(trades["net_return"].gt(0.0).mean()) if len(trades) else 0.0,
        "exit_reason_counts": dict(
            sorted(Counter(trades["exit_reason"].astype(str)).items())
        ),
        "selected_candidate_tiers": dict(
            sorted(
                Counter(
                    trades.loc[
                        trades["route"].eq("COVERAGE_CANDIDATE"),
                        "confidence_tier",
                    ].astype(str)
                ).items()
            )
        ),
        "concentration": _concentration(ledger, stage),
    }
    return {
        "summary": summary,
        "detail": detail,
        "ledger": ledger,
        "per_bar": per_bar,
        "episodes": episodes,
    }


def _sqlite_rows(database: Path, table: str) -> list[dict[str, Any]]:
    with sqlite3.connect(database) as connection:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (table,),
        ).fetchone()
        if exists is None:
            return []
        columns = [
            row[1] for row in connection.execute(f"PRAGMA table_info({table})")
        ]
        rows = connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
    return [dict(zip(columns, row)) for row in rows]


def _verify_state_and_calls(
    root: Path,
    manifest: dict[str, Any],
    overall: dict[str, Any],
    stages: dict[str, dict[str, Any]],
    variant: str,
) -> dict[str, Any]:
    state = _read_json(root / "run_state.json")
    if state["completed_stages"] != list(STAGES):
        raise ValueError(f"continuous stages incomplete: {variant}")
    expected_episode_counts = {
        stage: len(stages[stage]["episodes"]) for stage in STAGES
    }
    if state["stage_episode_counts"] != expected_episode_counts:
        raise ValueError(f"stage episode checkpoint drift: {variant}")
    if state["global_episode_number"] != sum(expected_episode_counts.values()):
        raise ValueError(f"global episode continuity drift: {variant}")
    if state["input_hash"] != manifest["input_hash"] or state[
        "protocol_hash"
    ] != manifest["protocol_hash"]:
        raise ValueError(f"state identity drift: {variant}")
    starts = [int(state["stage_call_starts"][stage]) for stage in STAGES]
    if starts != sorted(starts):
        raise ValueError(f"stage call offsets are not monotonic: {variant}")

    database = root / "agent_state.sqlite"
    checkpoints = _sqlite_rows(database, "stage_checkpoints")
    if [row["stage"] for row in checkpoints] != list(STAGES):
        raise ValueError(f"SQLite stage continuity drift: {variant}")
    for row in checkpoints:
        if row["protocol_hash"] != manifest["protocol_hash"]:
            raise ValueError(f"cross-protocol stage checkpoint: {variant}")
    call_rows = _sqlite_rows(database, "call_audit")
    call_payloads = [json.loads(row["payload_json"]) for row in call_rows]
    retrieval_rows = _sqlite_rows(database, "retrieval_audits")
    memory_rows = _sqlite_rows(database, "memory_cards")
    policy_rows = _sqlite_rows(database, "policy_history")
    for table_rows in (call_rows, memory_rows, policy_rows):
        if any(row["protocol_hash"] != manifest["protocol_hash"] for row in table_rows):
            raise ValueError(f"cross-variant protocol row entered {variant}")

    expected_calls = sum(int(stages[stage]["summary"]["calls"]) for stage in STAGES)
    expected_failures = sum(
        int(stages[stage]["summary"]["call_failures"]) for stage in STAGES
    )
    observed_failures = sum(
        payload.get("status") in {"schema_failure", "transport_error"}
        for payload in call_payloads
    )
    if len(call_payloads) != expected_calls or observed_failures != expected_failures:
        raise ValueError(f"call audit count drift: {variant}")

    cloud_calls = _read_jsonl(root / "cloud_calls.jsonl")
    prompt_audits = _read_jsonl(root / "prompt_audit.jsonl")
    if any(not audit.get("passed") for audit in prompt_audits):
        raise ValueError(f"failed prompt audit persisted: {variant}")
    for audit in prompt_audits:
        for key in (
            "cutoff_utc",
            "max_decision_time",
            "max_feature_time",
            "max_outcome_available_time",
            "max_memory_created_at",
            "max_memory_support_outcome_time",
            "active_policy_activates_at_utc",
        ):
            if audit.get(key) is not None and pd.Timestamp(audit[key]) >= Q2_START:
                raise ValueError(f"Q2 prompt provenance entered {variant}")
        if pd.Timestamp(audit["max_feature_time"]) > pd.Timestamp(audit["cutoff_utc"]):
            raise ValueError(f"future feature prompt provenance entered {variant}")
        if pd.Timestamp(audit["max_outcome_available_time"]) > pd.Timestamp(
            audit["cutoff_utc"]
        ):
            raise ValueError(f"future outcome prompt provenance entered {variant}")
        if audit.get("max_memory_created_at") is not None and pd.Timestamp(
            audit["max_memory_created_at"]
        ) > pd.Timestamp(audit["cutoff_utc"]):
            raise ValueError(f"future memory entered prompt: {variant}")

    for row in retrieval_rows:
        if row["protocol_hash"] != manifest["protocol_hash"]:
            raise ValueError(f"cross-protocol retrieval entered {variant}")
        if row["protocol_scope"] != "continuous_2021_2026":
            raise ValueError(f"retrieval scope drift: {variant}")
        if pd.Timestamp(row["cutoff_utc"]) >= Q2_START:
            raise ValueError(f"Q2 retrieval entered {variant}")
        payload = json.loads(row["payload_json"])
        for timestamp_key in ("max_created_at_utc", "max_support_outcome_time"):
            if payload.get(timestamp_key) is not None and pd.Timestamp(
                payload[timestamp_key]
            ) > pd.Timestamp(row["cutoff_utc"]):
                raise ValueError(f"future retrieved memory entered {variant}")

    memory_frame = pd.read_parquet(root / "memory_ledger.parquet")
    if len(memory_frame) != len(memory_rows):
        raise ValueError(f"memory ledger/SQLite count drift: {variant}")
    for raw in memory_frame.to_dict(orient="records"):
        card = MemoryCard.model_validate(raw)
        if pd.Timestamp(card.created_at_utc) >= Q2_START:
            raise ValueError(f"Q2 memory entered {variant}")
        if pd.Timestamp(card.max_support_outcome_time) > pd.Timestamp(
            card.created_at_utc
        ):
            raise ValueError(f"future support entered memory: {variant}")

    policy_history = [
        ActiveAllowRule.model_validate(row)
        for row in _read_jsonl(root / "policy_history.jsonl")
    ]
    evaluations = [
        GateDecision.model_validate(row)
        for row in _read_jsonl(root / "evaluations.jsonl")
    ]
    evaluation_by_candidate = {item.candidate_id: item for item in evaluations}
    for rule in policy_history:
        evaluation = evaluation_by_candidate.get(rule.source_candidate_id)
        if evaluation is None or evaluation.decision != "PROMOTE":
            raise ValueError(f"active rule lacks promoted evaluation: {variant}")
        if evaluation.shadow_cutoff_utc is None or pd.Timestamp(
            rule.activates_at_utc
        ) <= pd.Timestamp(evaluation.shadow_cutoff_utc):
            raise ValueError(f"rule activation is not strictly later: {variant}")
    rule_ids = {rule.rule_id for rule in policy_history}
    for stage in STAGES:
        selected = stages[stage]["ledger"]
        used = {
            rule_id
            for value in selected.loc[
                selected["route"].eq("COVERAGE_CANDIDATE")
                & selected["selected"].astype(bool),
                "selected_rule_ids",
            ].astype(str)
            for rule_id in value.split(",")
            if rule_id
        }
        if variant in AGENT_VARIANTS and not used.issubset(rule_ids):
            raise ValueError(f"selected trade references unknown rule: {variant}/{stage}")
        expected_static = {
            "static_high_extra": {"STATIC_HIGH_EXTRA"},
            "static_all_extra": {"STATIC_ALL_EXTRA"},
            "union_baseline": set(),
        }
        if variant in CONTROL_VARIANTS and used != expected_static[variant]:
            raise ValueError(f"static control rule marker drift: {variant}/{stage}")

    if variant in CONTROL_VARIANTS:
        if call_rows or cloud_calls or prompt_audits or retrieval_rows:
            raise ValueError(f"deterministic control called model or memory: {variant}")
    elif not prompt_audits:
        raise ValueError(f"agent variant lacks prompt audits: {variant}")
    return {
        "state": state,
        "call_payloads": call_payloads,
        "cloud_calls": cloud_calls,
        "prompt_audits": prompt_audits,
        "retrieval_rows": retrieval_rows,
        "memory_records": len(memory_frame),
        "policy_records": len(policy_history),
        "evaluations": evaluations,
    }


def _coverage_gates(
    variant: dict[str, Any], union: dict[str, Any], stage: str
) -> dict[str, bool]:
    candidate = variant["summary"]
    baseline = union["summary"]
    concentration = variant["detail"]["concentration"]
    return {
        "total_trade_growth_25pct": candidate["selected_trades"]
        >= math.ceil(1.25 * baseline["selected_trades"]),
        "additional_long_growth_10pct": candidate["selected_long_trades"]
        - baseline["selected_long_trades"]
        >= math.ceil(0.10 * baseline["selected_long_trades"]),
        "additional_short_growth_10pct": candidate["selected_short_trades"]
        - baseline["selected_short_trades"]
        >= math.ceil(0.10 * baseline["selected_short_trades"]),
        "net_noninferiority": candidate["net_return"]
        >= baseline["net_return"] - 0.005,
        "long_net_noninferiority": candidate["long_net_return"]
        >= baseline["long_net_return"] - 0.0025,
        "short_net_noninferiority": candidate["short_net_return"]
        >= baseline["short_net_return"] - 0.0025,
        "sortino_noninferiority": candidate["sortino"]
        >= baseline["sortino"] - 0.10,
        "drawdown_noninferiority": candidate["max_drawdown"]
        <= baseline["max_drawdown"] + 0.01,
        "distributed_additional_trades": concentration["blocks"] >= 2
        and concentration["maximum_share"] <= 0.60,
        "transport_coverage": candidate["status"] == "complete"
        and candidate["call_failure_fraction"] <= 0.05,
    }


def _cumulative_detail(stage_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ledger = pd.concat([item["ledger"] for item in stage_records], ignore_index=True)
    per_bar = pd.concat([item["per_bar"] for item in stage_records], ignore_index=True)
    per_bar = per_bar.groupby("timestamp", as_index=False)["net_return"].sum()
    metrics = _recomputed_metrics(ledger, per_bar)
    trades = ledger.loc[ledger["selected"].astype(bool)]
    return {
        **metrics,
        "win_rate": float(trades["net_return"].gt(0.0).mean()) if len(trades) else 0.0,
        "exit_reason_counts": dict(
            sorted(Counter(trades["exit_reason"].astype(str)).items())
        ),
    }


def reconcile_final_experiment(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    base = Path(root).resolve()
    records: dict[str, dict[str, Any]] = {}
    prompt_audit_total = 0
    prompt_audit_passed = 0
    protocol_hashes: set[str] = set()
    input_hashes: set[str] = set()
    implementation_hashes: set[str] = set()
    opportunity_hashes: set[str] = set()

    for variant in VARIANTS:
        variant_root = base / variant
        manifest = _read_json(variant_root / "manifest.json")
        overall = _read_json(variant_root / "summary.json")
        if manifest.get("variant_id") != variant or overall.get("variant_id") != variant:
            raise ValueError(f"variant identity mismatch: {variant}")
        if overall.get("status") != "complete" or overall.get(
            "lockbox_2026_q2_used"
        ) is not False:
            raise ValueError(f"variant incomplete or Q2 flag changed: {variant}")
        if pd.Timestamp(overall["maximum_outcome_available_time"]) >= Q2_START:
            raise ValueError(f"Q2 entered overall result: {variant}")
        for name, expected in manifest.get("artifact_hashes", {}).items():
            path = variant_root / name
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"artifact hash mismatch: {variant}/{name}")
        if manifest["input_hash"] != overall["input_hash"]:
            raise ValueError(f"input hash mismatch: {variant}")
        if manifest["opportunity_hash"] != overall["opportunity_hash"]:
            raise ValueError(f"opportunity hash mismatch: {variant}")
        if variant in AGENT_VARIANTS:
            if manifest.get("model") != "deepseek-v4-flash:cloud" or manifest.get(
                "model_digest"
            ) != "5166728b9358990e5f6c34f87cbe48716be2f2cd2d3b98527dff27ea755bf3ba":
                raise ValueError(f"model identity drift: {variant}")
        elif manifest.get("model") is not None or manifest.get("model_digest") is not None:
            raise ValueError(f"control acquired model identity: {variant}")

        stage_records = {
            stage: _verify_stage(
                variant_root, manifest, overall, variant, stage
            )
            for stage in STAGES
        }
        runtime = _verify_state_and_calls(
            variant_root, manifest, overall, stage_records, variant
        )
        prompt_audit_total += len(runtime["prompt_audits"])
        prompt_audit_passed += sum(
            bool(item["passed"]) for item in runtime["prompt_audits"]
        )
        records[variant] = {
            "root": variant_root,
            "manifest": manifest,
            "overall": overall,
            "stages": stage_records,
            "runtime": runtime,
            "cumulative": _cumulative_detail(list(stage_records.values())),
        }
        protocol_hashes.add(manifest["protocol_hash"])
        input_hashes.add(manifest["input_hash"])
        implementation_hashes.add(manifest["implementation_hash"])
        opportunity_hashes.add(manifest["opportunity_hash"])

    if len(protocol_hashes) != len(VARIANTS):
        raise ValueError("variant protocol stores are not isolated")
    if len(input_hashes) != 1 or len(implementation_hashes) != 1 or len(
        opportunity_hashes
    ) != 1:
        raise ValueError("registered variants do not share one implementation/input")

    stage_counts: dict[str, dict[str, int]] = {}
    tier_counts: dict[str, dict[str, int]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    union_invariant = True
    for stage in STAGES:
        union = records["union_baseline"]["stages"][stage]["ledger"]
        stage_counts[stage] = EXPECTED_COUNTS[stage]
        tier_counts[stage] = dict(
            sorted(
                Counter(
                    union.loc[
                        union["route"].eq("COVERAGE_CANDIDATE"),
                        "confidence_tier",
                    ].astype(str)
                ).items()
            )
        )
        comparisons[stage] = {}
        union_rows = union.loc[union["route"].eq("UNION_BASE")].set_index(
            "opportunity_id"
        )
        for variant in VARIANTS:
            ledger = records[variant]["stages"][stage]["ledger"]
            current_union = ledger.loc[ledger["route"].eq("UNION_BASE")].set_index(
                "opportunity_id"
            )
            union_invariant = bool(
                union_invariant
                and current_union.loc[:, OPPORTUNITY_COLUMNS[1:]].equals(
                    union_rows.loc[:, OPPORTUNITY_COLUMNS[1:]]
                )
                and current_union["selected"].astype(bool).all()
            )
            if variant != "union_baseline":
                seed = int.from_bytes(
                    hashlib.sha256(f"v3:{stage}:{variant}".encode()).digest()[:8],
                    "big",
                )
                comparisons[stage][variant] = paired_policy_delta(
                    ledger, union, seed=seed
                )
    if tier_counts["development"] != {
        "HIGH_EXTRA": 524,
        "LOW_EXTRA": 1150,
        "MID_EXTRA": 834,
    }:
        raise ValueError("development candidate tier count drift")
    if not union_invariant:
        raise ValueError("Union subset or economics changed across variants")

    results = {
        stage: {
            variant: records[variant]["stages"][stage]["detail"]
            for variant in VARIANTS
        }
        for stage in STAGES
    }
    results["cumulative"] = {
        variant: records[variant]["cumulative"] for variant in VARIANTS
    }
    coverage_gates = {
        stage: _coverage_gates(
            records["reflection_real_memory"]["stages"][stage],
            records["union_baseline"]["stages"][stage],
            stage,
        )
        for stage in STAGES
    }
    coverage_success = all(
        all(coverage_gates[stage].values()) for stage in ("development", "h1")
    )
    memory_effect_observed = any(
        not records["reflection_real_memory"]["stages"][stage]["ledger"][
            "selected"
        ].astype(bool).equals(
            records[variant]["stages"][stage]["ledger"]["selected"].astype(bool)
        )
        for stage in STAGES
        for variant in (
            "reflection_no_memory",
            "reflection_shuffled_memory",
        )
    )

    def objective(variant: str) -> tuple[int, int, float, float]:
        gates_passed = int(
            all(
                all(
                    _coverage_gates(
                        records[variant]["stages"][stage],
                        records["union_baseline"]["stages"][stage],
                        stage,
                    ).values()
                )
                for stage in ("development", "h1")
            )
        )
        cumulative = records[variant]["cumulative"]
        return (
            gates_passed,
            int(cumulative["selected_trades"]),
            float(cumulative["net_return"]),
            -float(cumulative["max_drawdown"]),
        )

    real_objective = objective("reflection_real_memory")
    memory_benefit = bool(
        coverage_success
        and memory_effect_observed
        and all(
            real_objective > objective(variant)
            for variant in (
                "reflection_no_memory",
                "reflection_shuffled_memory",
            )
        )
    )
    transport_gate = all(
        records[variant]["stages"][stage]["summary"]["status"] == "complete"
        and records[variant]["stages"][stage]["summary"][
            "call_failure_fraction"
        ]
        <= 0.05
        for variant in AGENT_VARIANTS
        for stage in STAGES
    )
    funnel = {
        variant: {
            "transport_calls": len(records[variant]["runtime"]["call_payloads"]),
            "transport_statuses": dict(
                sorted(
                    Counter(
                        item["status"]
                        for item in records[variant]["runtime"]["call_payloads"]
                    ).items()
                )
            ),
            "proposal_choice_indices": dict(
                sorted(
                    Counter(
                        str(item["validated_content"]["choice_index"])
                        for item in records[variant]["runtime"]["cloud_calls"]
                        if item.get("role") == "proposal"
                        and item.get("validated_content") is not None
                    ).items()
                )
            ),
            "transition_statuses": dict(
                sorted(
                    Counter(
                        item["status"]
                        for item in _read_jsonl(
                            records[variant]["root"] / "transitions.jsonl"
                        )
                    ).items()
                )
            ),
            "evaluation_decisions": dict(
                sorted(
                    Counter(
                        item.decision
                        for item in records[variant]["runtime"]["evaluations"]
                    ).items()
                )
            ),
            "memory_records": records[variant]["runtime"]["memory_records"],
            "policy_records": records[variant]["runtime"]["policy_records"],
        }
        for variant in AGENT_VARIANTS
    }
    return {
        "status": "complete",
        "conclusion": (
            "coverage_and_memory_benefit_established"
            if coverage_success and memory_benefit
            else "coverage_success_memory_benefit_not_established"
            if coverage_success
            else "coverage_success_not_established"
        ),
        "coverage_success": coverage_success,
        "memory_benefit_established": memory_benefit,
        "memory_effect_observed": memory_effect_observed,
        "forward_evidence_role": "secondary_reused_forward",
        "lockbox_2026_q2_used": False,
        "artifact_hashes_verified": True,
        "all_prompt_audits_passed": prompt_audit_total > 0
        and prompt_audit_passed == prompt_audit_total,
        "prompt_audit_passed": prompt_audit_passed,
        "prompt_audit_total": prompt_audit_total,
        "controls_called_llm": False,
        "union_invariant_across_variants": union_invariant,
        "continuous_stage_state_verified": True,
        "transport_failure_gate_passed": transport_gate,
        "stage_counts": stage_counts,
        "tier_counts": tier_counts,
        "implementation_hash": next(iter(implementation_hashes)),
        "input_hash": next(iter(input_hashes)),
        "opportunity_hash": next(iter(opportunity_hashes)),
        "coverage_gates": coverage_gates,
        "memory_objectives": {
            variant: list(objective(variant)) for variant in AGENT_VARIANTS
        },
        "funnel": funnel,
        "results": results,
        "comparisons": comparisons,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _write_report_artifacts(base: Path, report: dict[str, Any]) -> dict[str, Any]:
    result_rows = [
        {"stage": stage, "variant": variant, **metrics}
        for stage, variants in report["results"].items()
        for variant, metrics in variants.items()
    ]
    comparison_rows = [
        {"stage": stage, "variant": variant, **metrics}
        for stage, variants in report["comparisons"].items()
        for variant, metrics in variants.items()
    ]
    gate_rows = [
        {"stage": stage, "gate": gate, "passed": passed}
        for stage, gates in report["coverage_gates"].items()
        for gate, passed in gates.items()
    ]
    tables = {
        "results_table.parquet": pd.json_normalize(result_rows, sep="__"),
        "paired_comparisons.parquet": pd.DataFrame(comparison_rows),
        "coverage_gates.parquet": pd.DataFrame(gate_rows),
    }
    for name, frame in tables.items():
        _atomic_parquet(base / name, frame)
    table_hashes = {name: _sha256_file(base / name) for name in tables}
    sealed = {**report, "table_artifacts": table_hashes}
    _atomic_json(base / "final_report.json", sealed)
    manifest = {
        "final_report.json": _sha256_file(base / "final_report.json"),
        **table_hashes,
    }
    _atomic_json(base / "final_report_manifest.json", manifest)
    return sealed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    report = reconcile_final_experiment(args.root)
    sealed = _write_report_artifacts(Path(args.root).resolve(), report)
    print(json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["paired_policy_delta", "reconcile_final_experiment"]
