"""Export concise notebook-ready reflection-agent evidence from frozen artifacts."""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd

from experiments.all_model_sentiment_policy import (
    DEFAULT_ROOT as POLICY_ROOT,
    POLICY_FIELDS,
    validate_policy_model_artifacts,
)
from experiments.build_reflection_cache import DEFAULT_OUTPUT
from reflection_agent.evaluator import StrategyOutcome, summarize
from reflection_agent.execution import build_frozen_baselines
from reflection_agent.replay import REGISTERED_VARIANTS


SECONDARY_SVM_ROOT = POLICY_ROOT / "none" / "svm_linear"


def _secondary_svm_benchmark() -> tuple[dict[str, object], pd.Series]:
    """Load the H1-frozen SVM DZ75 forward result without reselecting it."""
    validate_policy_model_artifacts(SECONDARY_SVM_ROOT, model_name="svm_linear")
    policies = pd.read_parquet(SECONDARY_SVM_ROOT / "selected_policies_2025h1.parquet")
    summaries = pd.read_parquet(SECONDARY_SVM_ROOT / "forward_summary.parquet")
    policy = policies.loc[policies["width_bps"].astype(int).eq(75)]
    summary = summaries.loc[summaries["width_bps"].astype(int).eq(75)]
    if len(policy) != 1 or len(summary) != 1:
        raise ValueError("secondary SVM benchmark requires exactly one frozen DZ75 row")
    policy_row = policy.iloc[0]
    summary_row = summary.iloc[0]
    for field in POLICY_FIELDS:
        if policy_row[field] != summary_row[field]:
            raise ValueError(f"secondary SVM forward {field} differs from its H1-frozen policy")

    trades = pd.read_parquet(SECONDARY_SVM_ROOT / "forward_evidence" / "w75_lb180_ledger.parquet")
    per_bar = pd.read_parquet(
        SECONDARY_SVM_ROOT / "forward_evidence" / "w75_lb180_per_bar.parquet"
    )
    timestamps = pd.to_datetime(per_bar["timestamp"], utc=True)
    if timestamps.max() >= pd.Timestamp("2026-04-01", tz="UTC"):
        raise ValueError("secondary SVM benchmark enters the sealed Q2 interval")
    returns = pd.Series(
        per_bar["net_return"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(timestamps),
        name="svm_linear_dz75",
    )
    metrics = summarize(StrategyOutcome(returns=returns, trades=trades, turnover=2.0 * len(trades)))
    for field in ("trades", "net_return", "sortino", "sharpe", "max_drawdown"):
        if not math.isclose(float(getattr(metrics, field)), float(summary_row[field]), abs_tol=1e-12):
            raise ValueError(f"secondary SVM evidence disagrees on {field}")
    row = {
        "control_id": "svm_linear_dz75",
        "benchmark_role": "secondary_reporting",
        "tp_bps": int(summary_row["tp_bps"]),
        "sl_bps": int(summary_row["sl_bps"]),
        "max_hold": int(summary_row["max_hold"]),
        "fee_bps_per_side": 5.0,
        **metrics.model_dump(),
    }
    return row, returns


def _read_rows(connection: sqlite3.Connection, table: str, protocol_hash: str) -> list[dict]:
    rows = connection.execute(
        f"SELECT record_id, payload_json FROM {table} WHERE protocol_hash = ? ORDER BY record_id",
        (protocol_hash,),
    ).fetchall()
    return [{"record_id": record_id, "payload": json.loads(payload)} for record_id, payload in rows]


def export_results(cache_root: Path = DEFAULT_OUTPUT) -> dict[str, object]:
    manifest = json.loads((cache_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    protocol_hash = manifest["protocol_hash"]
    export_root = cache_root / "exports"
    export_root.mkdir(parents=True, exist_ok=True)
    baselines = build_frozen_baselines(cache_root)
    baselines["benchmark_role"] = baselines["control_id"].map({
        "lstm": "primary",
        "unanimity_consensus": "ensemble_foundation",
        "deterministic_router": "deterministic_comparator",
    })
    svm_row, svm_returns = _secondary_svm_benchmark()
    baselines = pd.concat([baselines, pd.DataFrame([svm_row])], ignore_index=True)
    baselines.to_parquet(export_root / "controls.parquet", index=False)
    frozen_returns = pd.read_parquet(cache_root / "frozen_baseline_returns.parquet")
    lstm_returns = frozen_returns.set_index("timestamp")["lstm"].rename("lstm")
    benchmark_returns = pd.concat([lstm_returns, svm_returns], axis=1, join="inner")
    if benchmark_returns.isna().any().any():
        raise ValueError("benchmark return lines must be fully aligned")
    benchmark_returns.rename_axis("timestamp").reset_index().to_parquet(
        export_root / "benchmark_returns.parquet", index=False
    )

    primary_state = cache_root / "agent_state_reflection_real_memory.sqlite"
    if not primary_state.exists():
        primary_state = cache_root / "agent_state.sqlite"
    connection = sqlite3.connect(primary_state)
    calls = connection.execute(
        "SELECT role, status, payload_json FROM llm_calls WHERE protocol_hash = ? ORDER BY created_at_utc",
        (protocol_hash,),
    ).fetchall()
    call_rows = []
    for role, status, raw_payload in calls:
        payload = json.loads(raw_payload)
        call_rows.append({
            "role": role,
            "status": status,
            "attempts": payload.get("attempts", 0),
            "backend": payload.get("backend"),
            "latency_seconds": payload.get("latency_seconds", 0.0),
            "error_count": len(payload.get("errors", [])),
        })
    reliability = pd.DataFrame(call_rows, columns=[
        "role", "status", "attempts", "backend", "latency_seconds", "error_count"
    ])
    reliability.to_parquet(export_root / "llm_reliability.parquet", index=False)

    candidate_rows = _read_rows(connection, "candidates", protocol_hash)
    candidates = []
    for row in candidate_rows:
        payload = row["payload"]
        if "candidate" in payload:
            candidates.append({"stage": payload["stage"], **payload["candidate"]})
    evaluations_raw = _read_rows(connection, "evaluations", protocol_hash)
    evaluations = []
    for row in evaluations_raw:
        payload = row["payload"]
        if "evaluation" not in payload:
            continue
        record = payload["evaluation"]
        evaluations.append({
            "stage": payload["stage"],
            "candidate_id": record["candidate_id"],
            "decision": record["decision"],
            "delta_net_return": record["delta_net_return"],
            "candidate_trades": record["candidate"]["trades"],
            "candidate_long_trades": record["candidate"]["long_trades"],
            "candidate_short_trades": record["candidate"]["short_trades"],
            "candidate_sortino": record["candidate"]["sortino"],
            "candidate_max_drawdown": record["candidate"]["max_drawdown"],
            **{f"guard_{key}": value for key, value in record["guard_results"].items()},
        })
    evaluation_frame = pd.DataFrame(evaluations)
    evaluation_frame.to_parquet(export_root / "historical_evaluations.parquet", index=False)
    shadows = _read_rows(connection, "shadows", protocol_hash)
    policies = _read_rows(connection, "policies", protocol_hash)
    actor_count = sum(candidate["stage"] == "actor" for candidate in candidates)
    refiner_count = sum(candidate["stage"] == "refiner" for candidate in candidates)
    actor_kept = int((evaluation_frame.loc[evaluation_frame.get("stage") == "actor", "decision"] == "historical_keep").sum()) if not evaluation_frame.empty else 0
    refiner_kept = int((evaluation_frame.loc[evaluation_frame.get("stage") == "refiner", "decision"] == "historical_keep").sum()) if not evaluation_frame.empty else 0
    funnel = pd.DataFrame([
        {"stage": "Actor JSON-valid", "count": actor_count},
        {"stage": "Actor historical keep", "count": actor_kept},
        {"stage": "Refiner JSON-valid", "count": refiner_count},
        {"stage": "Refiner historical keep", "count": refiner_kept},
        {"stage": "Shadow opened", "count": len(shadows)},
    ])
    funnel.to_parquet(export_root / "candidate_funnel.parquet", index=False)

    events = pd.read_parquet(cache_root / "news_events.parquet")
    source_coverage = events.groupby("source_family", as_index=False).agg(
        events=("event_id", "count"),
        first_available=("available_at_utc", "min"),
        last_available=("available_at_utc", "max"),
    )
    source_coverage.to_parquet(export_root / "source_coverage.parquet", index=False)
    ablation_rows = []
    for variant_id, variant in REGISTERED_VARIANTS.items():
        replay_path = cache_root / f"replay_{variant_id}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8")) if replay_path.exists() else None
        manifest_match = bool(replay and replay.get("protocol_hash") == protocol_hash)
        completed_windows = sum(
            row.get("status") in {"completed", "cached"} for row in replay.get("windows", [])
        ) if manifest_match else 0
        status = "not_run"
        if completed_windows:
            status = "complete" if completed_windows >= 26 else "smoke_only"
        ablation_rows.append({
            "variant": variant_id,
            "memory_mode": variant.memory_mode,
            "news_mode": variant.news_mode,
            "status": status,
            "completed_windows": completed_windows,
        })
    ablations = pd.DataFrame(ablation_rows)
    ablations.to_parquet(export_root / "ablation_registry.parquet", index=False)
    connection.close()

    claim_reason = (
        f"The completed causal evidence opened {len(shadows)} shadows, but no promoted policy exists; "
        "there is no agent return series that can beat either frozen benchmark."
        if not policies
        else "Promoted development policies exist, but no final sealed comparison against both benchmarks has run."
    )
    summary = {
        "protocol_hash": protocol_hash,
        "claim_status": "not_established",
        "claim_reason": claim_reason,
        "actor_candidates": actor_count,
        "refiner_candidates": refiner_count,
        "shadows_opened": len(shadows),
        "promoted_policies": len(policies),
        "llm_calls": len(reliability),
        "completed_registered_variants": int((ablations["status"] == "complete").sum()),
        "sealed_start_utc": manifest["config"]["sealed_start_utc"],
    }
    (export_root / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return summary


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(export_results(args.cache_root), indent=2))


if __name__ == "__main__":
    main()
