"""Prepare and run the preregistered causal full-information policy router."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from experiments.run_reflection_agent_v3 import _live_model_record
from reflection_agent.v2.transport import DeepSeekSchemaCaller, SchemaCallResult
from reflection_agent.v3.opportunities import (
    build_development_opportunities,
    build_exact_opportunities,
)
from reflection_agent.v4.config import AGENT_VARIANTS, load_v4_config
from reflection_agent.v4.contracts import RouterChoice
from reflection_agent.v4.policies import (
    POLICY_DESCRIPTIONS,
    POLICY_IDS,
    Q2_START,
    apply_router_policy,
    attach_xgb_support,
    validate_router_opportunities,
)
from reflection_agent.v4.prompts import SYSTEM_PROMPT_V4, prompt_hashes
from reflection_agent.v4.router import FullInformationRouter


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "reflection_policy_router_v4"
DEFAULT_CONFIG = CODE_ROOT / "configs" / "reflection_agent_v4.yaml"
STAGE_ORDER = ("development", "h1", "forward")
EXPECTED_COUNTS = {
    "development": {"rows": 3456, "union": 948, "candidates": 2508, "blocks": 47},
    "h1": {"rows": 1024, "union": 88, "candidates": 936, "blocks": 26},
    "forward": {"rows": 1002, "union": 74, "candidates": 928, "blocks": 40},
}
CONTROL_VARIANTS = (
    "hedge_router",
    "random_router",
    *(f"static_{policy_id.lower()}" for policy_id in POLICY_IDS),
)
REGISTERED_VARIANTS = (*AGENT_VARIANTS, *CONTROL_VARIANTS)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


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


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _frame_hash(frame: pd.DataFrame, sort_columns: Sequence[str]) -> str:
    normalized = frame.sort_values(list(sort_columns), kind="stable").copy()
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
        CODE_ROOT / "configs" / "reflection_agent_v4.yaml",
        CODE_ROOT / "experiments" / "run_reflection_agent_v3.py",
        CODE_ROOT / "reflection_agent" / "v2" / "transport.py",
        *sorted((CODE_ROOT / "reflection_agent" / "v4").glob("*.py")),
    ]
    return _hash(
        {
            str(path.relative_to(CODE_ROOT)).replace("\\", "/"): _sha256_file(path)
            for path in paths
        }
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
    per_bar = _per_bar(selected, frame)
    economics = economics_summary(per_bar)
    net = trades["net_return"].astype(float)
    gross = trades["gross_return"].astype(float)
    effective_days = max(
        (frame["decision_time"].max() - frame["decision_time"].min()).total_seconds()
        / 86_400.0,
        1.0 / 96.0,
    )
    return {
        "selected_trades": int(len(trades)),
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


def _stage_frame(stage: str) -> pd.DataFrame:
    raw = (
        build_development_opportunities()
        if stage == "development"
        else build_exact_opportunities(stage)
    )
    frame, _ = attach_xgb_support(raw, stage)
    frame["stage"] = stage
    return frame


def _block_keys(frame: pd.DataFrame, stage: str) -> tuple[pd.Series, dict[pd.Timestamp, str]]:
    week = frame["decision_time"].dt.floor("D") - pd.to_timedelta(
        frame["decision_time"].dt.dayofweek, unit="D"
    )
    mapping = {
        value: f"{stage}-b{index:03d}"
        for index, value in enumerate(sorted(week.unique()))
    }
    return week, mapping


def build_stage_payoffs(
    stage: str,
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    base = validate_router_opportunities(frame)
    base["stage"] = stage
    week, mapping = _block_keys(base, stage)
    base["block_id"] = week.map(mapping)
    base["block_boundary_eligible"] = (
        base["route"].eq("UNION_BASE")
        | base["outcome_available_time"].lt(week + pd.Timedelta(days=7))
    )
    payoffs: list[dict[str, Any]] = []
    ledgers: dict[str, pd.DataFrame] = {}

    for choice_index, policy_id in enumerate(POLICY_IDS):
        ledger = apply_router_policy(base, policy_id)
        ledger["stage"] = stage
        ledger["block_id"] = base["block_id"].to_numpy()
        ledgers[policy_id] = ledger
        for block_id, block in base.groupby("block_id", sort=False):
            selected_block = ledger.loc[ledger["block_id"].eq(block_id)]
            trades = selected_block.loc[selected_block["selected"]]
            union = trades.loc[trades["route"].eq("UNION_BASE")]
            extra = trades.loc[trades["route"].eq("COVERAGE_CANDIDATE")]
            commit_time = block["decision_time"].min() - pd.Timedelta(microseconds=1)
            available_at = block["outcome_available_time"].max()
            if commit_time >= block["decision_time"].min():
                raise AssertionError("policy commitment is not before its first opportunity")
            payoffs.append(
                {
                    "block_id": str(block_id),
                    "block_start": commit_time,
                    "block_available_at": available_at,
                    "stage": stage,
                    "choice_index": choice_index,
                    "policy_id": policy_id,
                    "union_trades": int(len(union)),
                    "union_long_trades": int(union["side"].eq("LONG").sum()),
                    "union_short_trades": int(union["side"].eq("SHORT").sum()),
                    "union_net": float(union["net_return"].sum()),
                    "union_long_net": float(
                        union.loc[union["side"].eq("LONG"), "net_return"].sum()
                    ),
                    "union_short_net": float(
                        union.loc[union["side"].eq("SHORT"), "net_return"].sum()
                    ),
                    "additional_trades": int(len(extra)),
                    "additional_long_trades": int(extra["side"].eq("LONG").sum()),
                    "additional_short_trades": int(extra["side"].eq("SHORT").sum()),
                    "incremental_gross": float(extra["gross_return"].sum()),
                    "incremental_cost": float(
                        (extra["gross_return"] - extra["net_return"]).sum()
                    ),
                    "incremental_net": float(extra["net_return"].sum()),
                    "incremental_long_net": float(
                        extra.loc[extra["side"].eq("LONG"), "net_return"].sum()
                    ),
                    "incremental_short_net": float(
                        extra.loc[extra["side"].eq("SHORT"), "net_return"].sum()
                    ),
                    "combined_trades": int(len(trades)),
                    "combined_net": float(trades["net_return"].sum()),
                }
            )
    table = pd.DataFrame.from_records(payoffs).sort_values(
        ["block_start", "choice_index"], kind="stable"
    )
    expected = EXPECTED_COUNTS[stage]
    if len(base) != expected["rows"]:
        raise ValueError(f"{stage} router row count drift")
    if int(base["route"].eq("UNION_BASE").sum()) != expected["union"]:
        raise ValueError(f"{stage} router Union count drift")
    if int(base["route"].eq("COVERAGE_CANDIDATE").sum()) != expected["candidates"]:
        raise ValueError(f"{stage} router candidate count drift")
    if table["block_id"].nunique() != expected["blocks"]:
        raise ValueError(f"{stage} router executable-week count drift")
    if table.groupby("block_id")["policy_id"].nunique().ne(len(POLICY_IDS)).any():
        raise AssertionError("a router block lacks a frozen policy")
    return table.reset_index(drop=True), ledgers


def prepare_common_artifacts(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    stage_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    config = load_v4_config(config_path)
    root = Path(output_root).resolve()
    common = root / "common"
    artifacts: dict[str, str] = {}
    source_hashes: dict[str, str] = {}
    stage_counts: dict[str, Any] = {}
    maximum_outcome = pd.Timestamp("1900-01-01", tz="UTC")

    for stage in STAGE_ORDER:
        supplied = stage_frames[stage].copy() if stage_frames is not None else None
        if supplied is None:
            frame = _stage_frame(stage)
        elif {
            "xgb_available",
            "xgb_p_move_raw",
            "xgb_direction_confidence",
            "xgb_side",
        }.issubset(supplied.columns):
            frame = validate_router_opportunities(supplied)
            frame["stage"] = stage
        else:
            frame, _ = attach_xgb_support(supplied, stage)
            frame["stage"] = stage
        if frame["outcome_available_time"].ge(Q2_START).any():
            raise ValueError("Q2 entered v4 common preparation")
        payoffs, ledgers = build_stage_payoffs(stage, frame)
        maximum_outcome = max(maximum_outcome, frame["outcome_available_time"].max())
        paths = {
            f"{stage}_opportunities.parquet": frame,
            f"{stage}_payoffs.parquet": payoffs,
            **{
                f"{stage}/{policy_id}.parquet": ledger
                for policy_id, ledger in ledgers.items()
            },
        }
        for relative, value in paths.items():
            path = common / relative
            _write_parquet(path, value)
            artifacts[relative] = _sha256_file(path)
        source_hashes[stage] = _frame_hash(frame, ("decision_time", "opportunity_id"))
        stage_counts[stage] = {
            "rows": int(len(frame)),
            "union": int(frame["route"].eq("UNION_BASE").sum()),
            "candidates": int(frame["route"].eq("COVERAGE_CANDIDATE").sum()),
            "blocks": int(payoffs["block_id"].nunique()),
            "boundary_vetoes": int(
                ledgers["LSTM_ALL"]["skip_reason"].eq("BLOCK_BOUNDARY").sum()
            ),
        }

    implementation_hash = _implementation_hash()
    protocol_hash = _hash(
        {
            "config": config.model_dump(mode="json"),
            "policy_descriptions": POLICY_DESCRIPTIONS,
            "prompt_hashes": prompt_hashes(),
            "schema": RouterChoice.model_json_schema(),
            "source_hashes": source_hashes,
            "implementation_hash": implementation_hash,
        }
    )
    manifest = {
        "status": "complete",
        "protocol_hash": protocol_hash,
        "implementation_hash": implementation_hash,
        "source_hashes": source_hashes,
        "artifact_hashes": artifacts,
        "stage_counts": stage_counts,
        "policy_ids": list(POLICY_IDS),
        "maximum_outcome_available_time": maximum_outcome.isoformat(),
        "lockbox_2026_q2_used": False,
    }
    _write_json(common / "manifest.json", manifest)
    return manifest


def _verified_common(root: Path) -> dict[str, Any]:
    manifest_path = root / "common" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["implementation_hash"] != _implementation_hash():
        raise ValueError("v4 common implementation hash drift")
    for relative, expected in manifest["artifact_hashes"].items():
        path = root / "common" / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"v4 common artifact hash mismatch: {relative}")
    if manifest.get("lockbox_2026_q2_used") is not False:
        raise ValueError("v4 common manifest opened Q2")
    return manifest


class CachedRouterCaller:
    """Replay terminal schema calls by exact causal request hash."""

    def __init__(self, inner: Any, cache_dir: Path, *, protocol_hash: str) -> None:
        self.inner = inner
        self.config = inner.config
        self.cache_dir = cache_dir
        self.protocol_hash = protocol_hash

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
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["protocol_hash"] != self.protocol_hash:
                raise ValueError("v4 cached call protocol drift")
            value = (
                response_model.model_validate(payload["validated_content"])
                if payload["validated_content"] is not None
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
            "protocol_hash": self.protocol_hash,
            "role": role,
            "messages": messages,
            "allowed_ids": allowed_ids,
            "status": result.status,
            "validated_content": (
                result.value.model_dump(mode="json") if result.value is not None else None
            ),
            "raw_content": result.raw_content,
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "schema_hash": result.schema_hash,
            "attempts": result.attempts,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
            "errors": list(result.errors),
        }
        _write_json(path, payload)
        return result


def run_preflight(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    model_record: dict[str, Any] | None = None,
    stage_frames: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    config = load_v4_config(config_path)
    common = prepare_common_artifacts(
        output_root=root,
        config_path=config_path,
        stage_frames=stage_frames,
    )
    record = model_record or _live_model_record(config.model)
    if record.get("model") != config.model:
        raise RuntimeError("v4 preflight resolved a different model tag")
    if str(record.get("digest", "")) != config.required_model_digest:
        raise RuntimeError("v4 preflight model digest changed")
    if "thinking" not in set(record.get("capabilities", [])):
        raise RuntimeError("v4 model does not advertise thinking support")
    active = caller or DeepSeekSchemaCaller(
        config, call_log_path=root / "preflight_calls.jsonl"
    )
    expected = {
        "schema_version": "4.0",
        "choice_index": 0,
        "evidence_indices": [],
        "memory_indices": [],
    }
    probe = active.call(
        role="router_preflight",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT_V4},
            {
                "role": "user",
                "content": "TASK: PREFLIGHT_STRICT_SCHEMA\nRETURN_JSON="
                + _canonical_json(expected),
            },
        ],
        response_model=RouterChoice,
        allowed_ids={"choice_indices": [0], "evidence_indices": [], "memory_indices": []},
    )
    if probe.value is None or probe.value.model_dump(mode="json") != expected:
        raise RuntimeError(f"v4 strict schema preflight failed: {probe.errors}")
    result = {
        "passed": True,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "model": config.model,
        "model_digest": config.required_model_digest,
        "ollama_version": str(record.get("ollama_version", "")),
        "capabilities": sorted(record.get("capabilities", [])),
        "probe_status": probe.status,
        "probe_request_hash": probe.request_hash,
        "probe_response_hash": probe.response_hash,
        "prompt_hashes": prompt_hashes(),
        "schema_hash": _hash(RouterChoice.model_json_schema()),
        "stage_counts": common["stage_counts"],
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "preflight.json", result)
    return result


def _variant_mode(variant: str) -> tuple[str, int | None]:
    if variant == "reflection_real_memory":
        return "real", None
    if variant == "reflection_no_memory":
        return "no_memory", None
    if variant == "reflection_shuffled_memory":
        return "shuffled", None
    if variant == "hedge_router":
        return "hedge", None
    if variant == "random_router":
        return "random", None
    if variant.startswith("static_"):
        policy = variant.removeprefix("static_").upper()
        if policy in POLICY_IDS:
            return "static", POLICY_IDS.index(policy)
    raise ValueError(f"unknown registered v4 variant: {variant}")


def run_variant(
    variant: str,
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
) -> dict[str, Any]:
    if variant not in REGISTERED_VARIANTS:
        raise ValueError(f"unknown registered v4 variant: {variant}")
    root = Path(output_root).resolve()
    config = load_v4_config(config_path)
    common = _verified_common(root)
    if variant in AGENT_VARIANTS:
        preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
        if not preflight.get("passed") or preflight["protocol_hash"] != common["protocol_hash"]:
            raise ValueError("v4 agent arm requires the matching passed preflight")
    payoffs = pd.concat(
        [pd.read_parquet(root / "common" / f"{stage}_payoffs.parquet") for stage in STAGE_ORDER],
        ignore_index=True,
    ).sort_values(["block_start", "choice_index"], kind="stable")
    mode, static_choice = _variant_mode(variant)
    active_caller = caller
    if variant in AGENT_VARIANTS and active_caller is None:
        inner = DeepSeekSchemaCaller(
            config, call_log_path=root / variant / "transport_calls.jsonl"
        )
        active_caller = CachedRouterCaller(
            inner,
            root / variant / "call_cache",
            protocol_hash=common["protocol_hash"],
        )
    router = FullInformationRouter(
        caller=active_caller,
        memory_mode=mode,
        static_choice=static_choice,
        seed=config.random_seed,
        hedge_eta=config.hedge_eta,
    )
    choices = router.run(payoffs)
    variant_root = root / variant
    _write_parquet(variant_root / "choices.parquet", choices)
    stage_summaries: dict[str, Any] = {}
    output_hashes: dict[str, str] = {"choices.parquet": _sha256_file(variant_root / "choices.parquet")}

    for stage in STAGE_ORDER:
        frame = pd.read_parquet(root / "common" / f"{stage}_opportunities.parquet")
        stage_choices = choices.loc[choices["stage"].eq(stage)]
        policy_ledgers = {
            policy_id: pd.read_parquet(root / "common" / stage / f"{policy_id}.parquet")
            for policy_id in stage_choices["policy_id"].unique()
        }
        selected_blocks: list[pd.DataFrame] = []
        for choice in stage_choices.itertuples(index=False):
            policy_ledger = policy_ledgers[choice.policy_id]
            selected_blocks.append(
                policy_ledger.loc[policy_ledger["block_id"].eq(choice.block_id)].copy()
            )
        selected = pd.concat(selected_blocks, ignore_index=True).sort_values(
            ["decision_time", "route", "opportunity_id"], kind="stable"
        )
        if selected["opportunity_id"].duplicated().any() or len(selected) != len(frame):
            raise AssertionError("v4 routed ledger does not cover each opportunity once")
        metrics = _policy_metrics(selected, frame)
        per_bar = _per_bar(selected, frame).rename_axis("timestamp").reset_index()
        stage_dir = variant_root / "stages" / stage
        _write_parquet(stage_dir / "selected_ledger.parquet", selected)
        _write_parquet(stage_dir / "per_bar_returns.parquet", per_bar)
        _write_json(stage_dir / "summary.json", metrics)
        stage_summaries[stage] = metrics
        for name in ("selected_ledger.parquet", "per_bar_returns.parquet", "summary.json"):
            relative = f"stages/{stage}/{name}"
            output_hashes[relative] = _sha256_file(variant_root / relative)

    call_status = choices["call_status"]
    calls = int(call_status.ne("not_called").sum())
    failures = int((~call_status.isin({"not_called", "success", "repaired"})).sum())
    summary = {
        "status": "complete",
        "variant": variant,
        "memory_mode": mode,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "stage_summaries": stage_summaries,
        "blocks": int(len(choices)),
        "transport_calls": calls,
        "transport_failures": failures,
        "transport_failure_fraction": float(failures / calls) if calls else 0.0,
        "controls_called_llm": bool(calls) if variant not in AGENT_VARIANTS else False,
        "choice_counts": {
            str(key): int(value) for key, value in choices["policy_id"].value_counts().items()
        },
        "lockbox_2026_q2_used": False,
    }
    _write_json(variant_root / "summary.json", summary)
    output_hashes["summary.json"] = _sha256_file(variant_root / "summary.json")
    manifest = {
        "status": "complete",
        "variant": variant,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
        "artifact_hashes": output_hashes,
        "lockbox_2026_q2_used": False,
    }
    _write_json(variant_root / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--variant", choices=REGISTERED_VARIANTS)
    parser.add_argument("--run-controls", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args(argv)
    if args.prepare:
        result: Any = prepare_common_artifacts(output_root=args.cache_dir)
    elif args.preflight:
        result = run_preflight(output_root=args.cache_dir)
    elif args.variant:
        result = run_variant(args.variant, output_root=args.cache_dir)
    elif args.run_controls:
        result = {
            variant: run_variant(variant, output_root=args.cache_dir)
            for variant in CONTROL_VARIANTS
        }
    else:
        parser.error("choose --prepare, --preflight, --variant or --run-controls")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE",
    "CONTROL_VARIANTS",
    "REGISTERED_VARIANTS",
    "build_stage_payoffs",
    "prepare_common_artifacts",
    "run_preflight",
    "run_variant",
]
