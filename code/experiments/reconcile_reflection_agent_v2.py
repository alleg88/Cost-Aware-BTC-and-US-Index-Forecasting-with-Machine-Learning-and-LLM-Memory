"""Reconcile the complete Reflection Agent v2 experiment without new model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = CODE_ROOT / "experiments" / "cache" / "reflection_agent_final"
Q2_START = pd.Timestamp("2026-04-01", tz="UTC")
STAGES = ("development", "h1", "forward")
VARIANTS = (
    "reflection_real_memory",
    "reflection_no_memory",
    "reflection_shuffled_memory",
    "static_add_all",
    "union_baseline",
)
AGENT_VARIANTS = VARIANTS[:3]
EXPECTED_COUNTS = {
    "development": {"opportunities": 1231, "union_base": 948, "reentries": 283},
    "h1": {"opportunities": 110, "union_base": 88, "reentries": 22},
    "forward": {"opportunities": 88, "union_base": 74, "reentries": 14},
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"missing JSONL artifact: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def paired_policy_delta(
    variant: pd.DataFrame,
    union: pd.DataFrame,
    *,
    bootstrap_samples: int = 5000,
    seed: int = 20260811,
) -> dict[str, Any]:
    """Return an opportunity-paired policy delta with a deterministic monthly CI."""

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
    return_delta = left["net_return"].astype(float) * selection_delta
    month = left_time.dt.tz_convert(None).dt.to_period("M").astype(str)
    monthly = return_delta.groupby(month).sum().sort_index()
    values = monthly.to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(bootstrap_samples, len(values)))
    distribution = values[draws].sum(axis=1)
    ci_low, ci_high = np.quantile(distribution, [0.025, 0.975])
    side = left["side"].astype(str)
    return {
        "selected_trade_delta": int(selection_delta.sum()),
        "selected_long_trade_delta": int(selection_delta.loc[side.eq("LONG")].sum()),
        "selected_short_trade_delta": int(selection_delta.loc[side.eq("SHORT")].sum()),
        "net_return_delta": float(return_delta.sum()),
        "long_net_return_delta": float(return_delta.loc[side.eq("LONG")].sum()),
        "short_net_return_delta": float(return_delta.loc[side.eq("SHORT")].sum()),
        "months": len(values),
        "positive_month_share": float((monthly > 0.0).mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_unit": "calendar_month",
        "bootstrap_seed": seed,
    }


def _verify_variant(root: Path, stage: str, variant: str) -> dict[str, Any]:
    variant_root = root / stage / variant
    manifest = _read_json(variant_root / "manifest.json")
    summary = _read_json(variant_root / "evaluation_summary.json")
    if manifest.get("stage") != stage or manifest.get("variant_id") != variant:
        raise ValueError(f"manifest identity mismatch: {stage}/{variant}")
    if summary.get("stage") != stage or summary.get("variant_id") != variant:
        raise ValueError(f"summary identity mismatch: {stage}/{variant}")
    if manifest.get("lockbox_2026_q2_used") is not False or summary.get(
        "lockbox_2026_q2_used"
    ) is not False:
        raise ValueError(f"Q2 lockbox flag changed: {stage}/{variant}")
    if pd.Timestamp(summary["maximum_outcome_available_time"]) >= Q2_START:
        raise ValueError(f"Q2 timestamp entered results: {stage}/{variant}")
    for name, expected in manifest.get("artifact_hashes", {}).items():
        path = variant_root / name
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"artifact hash mismatch: {stage}/{variant}/{name}")
    if summary["input_hash"] != manifest["input_hash"]:
        raise ValueError(f"input hash mismatch: {stage}/{variant}")
    if not np.isclose(
        summary["net_return"],
        summary["gross_return"] - summary["cost_return"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(f"economic reconciliation failed: {stage}/{variant}")
    if not np.isclose(
        summary["net_return"],
        summary["long_net_return"] + summary["short_net_return"],
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(f"side reconciliation failed: {stage}/{variant}")
    ledger = pd.read_parquet(variant_root / "opportunity_ledger.parquet")
    if len(ledger) != summary["opportunities"]:
        raise ValueError(f"opportunity count mismatch: {stage}/{variant}")
    if int(ledger["selected"].sum()) != summary["selected_trades"]:
        raise ValueError(f"selected count mismatch: {stage}/{variant}")
    if not ledger.loc[ledger["route"].eq("UNION_BASE"), "selected"].all():
        raise ValueError(f"Union base changed: {stage}/{variant}")
    return {
        "root": variant_root,
        "manifest": manifest,
        "summary": summary,
        "ledger": ledger,
    }


def reconcile_final_experiment(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    base = Path(root).resolve()
    records: dict[str, dict[str, dict[str, Any]]] = {}
    stage_counts: dict[str, dict[str, int]] = {}
    comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    implementation_hashes: dict[str, list[str]] = {}

    for stage in STAGES:
        records[stage] = {
            variant: _verify_variant(base, stage, variant) for variant in VARIANTS
        }
        input_hashes = {
            record["summary"]["input_hash"] for record in records[stage].values()
        }
        if len(input_hashes) != 1:
            raise ValueError(f"variants do not share one {stage} input hash")
        union_ledger = records[stage]["union_baseline"]["ledger"]
        observed = {
            "opportunities": len(union_ledger),
            "union_base": int(union_ledger["route"].eq("UNION_BASE").sum()),
            "reentries": int(union_ledger["route"].eq("REENTRY").sum()),
        }
        if observed != EXPECTED_COUNTS[stage]:
            raise ValueError(f"registered {stage} opportunity count drifted")
        stage_counts[stage] = observed
        implementation_hashes[stage] = sorted(
            {
                record["summary"]["implementation_hash"]
                for record in records[stage].values()
            }
        )
        comparisons[stage] = {
            variant: paired_policy_delta(
                records[stage][variant]["ledger"],
                union_ledger,
                seed=int.from_bytes(
                    hashlib.sha256(f"{stage}:{variant}".encode()).digest()[:8],
                    "big",
                ),
            )
            for variant in VARIANTS
            if variant != "union_baseline"
        }

    prompt_audit_total = 0
    prompt_audit_passed = 0
    development_funnel: dict[str, dict[str, Any]] = {}
    for variant in AGENT_VARIANTS:
        record = records["development"][variant]
        calls = _read_jsonl(record["root"] / "call_log.jsonl")
        audits = _read_jsonl(record["root"] / "prompt_audit.jsonl")
        proposals = [item for item in calls if item["role"] == "proposal"]
        reflections = [item for item in calls if item["role"] == "reflection"]
        prompt_audit_total += len(audits)
        prompt_audit_passed += sum(bool(item["passed"]) for item in audits)
        development_funnel[variant] = {
            "proposal_calls": len(proposals),
            "proposal_decisions": dict(
                sorted(
                    Counter(
                        item["validated_content"]["decision"]
                        for item in proposals
                        if item.get("validated_content") is not None
                    ).items()
                )
            ),
            "reflection_transport_calls": len(reflections),
            "schema_failures": sum(item["status"] == "schema_failure" for item in calls),
            "terminal": dict(
                sorted(
                    Counter(
                        item["decision"]
                        for item in record["summary"]["evaluations"]
                    ).items()
                )
            ),
            "promoted_rules": record["summary"]["promoted_rules"],
            "memory_records": record["summary"]["memory_records"],
        }

    for stage in ("h1", "forward"):
        for variant in AGENT_VARIANTS:
            if _read_jsonl(records[stage][variant]["root"] / "call_log.jsonl"):
                raise ValueError(f"unexpected LLM call in {stage}/{variant}")
    for stage in STAGES:
        for variant in ("static_add_all", "union_baseline"):
            if _read_jsonl(records[stage][variant]["root"] / "call_log.jsonl"):
                raise ValueError(f"control called LLM: {stage}/{variant}")

    for variant in AGENT_VARIANTS:
        snapshot = _read_json(
            records["forward"][variant]["root"] / "snapshot_manifest.json"
        )
        if (
            snapshot["source_stage"] != "h1"
            or snapshot["source_variant"] != variant
            or pd.Timestamp(snapshot["freeze_at_utc"])
            != pd.Timestamp("2025-07-01", tz="UTC")
            or snapshot["source_manifest_hash"]
            != _sha256_file(records["h1"][variant]["root"] / "manifest.json")
        ):
            raise ValueError(f"forward snapshot provenance failed: {variant}")

    union_development = records["development"]["union_baseline"]["ledger"]
    real_selection = records["development"]["reflection_real_memory"]["ledger"][
        "selected"
    ].astype(bool)
    memory_effect_observed = any(
        not real_selection.equals(
            records["development"][variant]["ledger"]["selected"].astype(bool)
        )
        for variant in (
            "reflection_no_memory",
            "reflection_shuffled_memory",
        )
    )
    real_delta = paired_policy_delta(
        records["development"]["reflection_real_memory"]["ledger"],
        union_development,
    )
    real_promotions = records["development"]["reflection_real_memory"]["summary"][
        "promoted_rules"
    ]
    benefit_established = bool(
        real_promotions > 0
        and real_delta["selected_trade_delta"] > 0
        and real_delta["net_return_delta"] > 0.0
        and memory_effect_observed
    )

    result_rows = {
        stage: {
            variant: {
                key: records[stage][variant]["summary"][key]
                for key in (
                    "selected_trades",
                    "selected_long_trades",
                    "selected_short_trades",
                    "net_return",
                    "long_net_return",
                    "short_net_return",
                    "sortino",
                    "sharpe",
                    "max_drawdown",
                    "promoted_rules",
                    "proposal_calls",
                )
            }
            for variant in VARIANTS
        }
        for stage in STAGES
    }
    return {
        "status": "complete",
        "conclusion": "benefit_established" if benefit_established else "benefit_not_established",
        "lockbox_2026_q2_used": False,
        "artifact_hashes_verified": True,
        "all_prompt_audits_passed": prompt_audit_total > 0
        and prompt_audit_passed == prompt_audit_total,
        "prompt_audit_passed": prompt_audit_passed,
        "prompt_audit_total": prompt_audit_total,
        "stage_counts": stage_counts,
        "implementation_hashes": implementation_hashes,
        "memory_effect_observed": memory_effect_observed,
        "development_funnel": development_funnel,
        "results": result_rows,
        "comparisons": comparisons,
    }


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = reconcile_final_experiment(args.root)
    output = args.output or args.root / "reconciliation.json"
    _atomic_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["paired_policy_delta", "reconcile_final_experiment"]
