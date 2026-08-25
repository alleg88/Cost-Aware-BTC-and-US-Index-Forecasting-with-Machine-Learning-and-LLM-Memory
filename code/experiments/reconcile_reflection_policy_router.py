"""Independently reconcile and seal the registered Reflection v4 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from experiments.run_reflection_policy_router import (
    AGENT_VARIANTS,
    CACHE,
    DEFAULT_CONFIG,
    EXPECTED_COUNTS,
    REGISTERED_VARIANTS,
    STAGE_ORDER,
    _implementation_hash,
)
from reflection_agent.v4.config import load_v4_config
from reflection_agent.v4.contracts import RouterChoice
from reflection_agent.v4.policies import POLICY_IDS, Q2_START
from reflection_agent.v4.prompts import SYSTEM_PROMPT_V4, prompt_hashes
from reflection_agent.v4.router import FullInformationRouter


DEFAULT_ROOT = CACHE
UNION_VARIANT = "static_union_only"
REQUIRED_STAGES = ("development", "h1")
EXPECTED_BOUNDARY_VETOES = {"development": 2, "h1": 2, "forward": 1}
ACCEPTED_CALL_STATUSES = {"success", "repaired"}
FORBIDDEN_PROMPT_KEYS = {
    "available_at",
    "decision_time",
    "entry_time",
    "outcome_available_time",
    "asset",
    "symbol",
    "exchange",
    "price",
    "path",
}
FORBIDDEN_VALUE_PATTERNS = {
    "iso_date": re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    "windows_path": re.compile(r"\b[A-Za-z]:\\"),
    "unix_path": re.compile(r"(?:^|\s)/(?:content|home|mnt|tmp)/", re.IGNORECASE),
    "asset": re.compile(r"\b(?:BTC|BTCUSDT|bitcoin|binance)\b", re.IGNORECASE),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _walk(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk(item, (*path, str(key)))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk(item, (*path, str(index)))
    else:
        yield path, value


def audit_prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Audit only the canonical input JSON; policy words in the system prompt are fixed."""

    reasons: list[str] = []
    expected_keys = {
        "schema_version",
        "coverage_status",
        "policy_menu",
        "policy_statistics",
        "memory_cards",
    }
    if set(payload) != expected_keys:
        reasons.append("top_level_keys")
    for path, value in _walk(payload):
        key = path[-1] if path else ""
        if key.lower() in FORBIDDEN_PROMPT_KEYS:
            reasons.append(f"forbidden_key:{'.'.join(path)}")
        if isinstance(value, str):
            for name, pattern in FORBIDDEN_VALUE_PATTERNS.items():
                if pattern.search(value):
                    reasons.append(f"forbidden_value:{name}:{'.'.join(path)}")
    return {"passed": not reasons, "reasons": sorted(set(reasons))}


def coverage_gates(
    candidate: Mapping[str, Any],
    union: Mapping[str, Any],
    *,
    additional_by_block: Mapping[str, int],
    transport_failure_fraction: float,
    audits_passed: bool,
) -> dict[str, bool]:
    config = load_v4_config(DEFAULT_CONFIG)
    additions = int(sum(int(value) for value in additional_by_block.values()))
    nonzero = [int(value) for value in additional_by_block.values() if int(value) > 0]
    maximum_share = max(nonzero) / additions if additions else 0.0
    return {
        "total_trade_growth_25pct": bool(
            int(candidate["selected_trades"])
            >= math.ceil((1.0 + config.minimum_trade_gain_fraction) * int(union["selected_trades"]))
        ),
        "additional_long_growth_10pct": bool(
            int(candidate["selected_long_trades"]) - int(union["selected_long_trades"])
            >= math.ceil(config.minimum_side_gain_fraction * int(union["selected_long_trades"]))
        ),
        "additional_short_growth_10pct": bool(
            int(candidate["selected_short_trades"])
            - int(union["selected_short_trades"])
            >= math.ceil(config.minimum_side_gain_fraction * int(union["selected_short_trades"]))
        ),
        "net_noninferiority": bool(
            float(candidate["net_return"])
            >= float(union["net_return"]) - config.net_noninferiority_margin
        ),
        "long_net_noninferiority": bool(
            float(candidate["long_net_return"])
            >= float(union["long_net_return"]) - config.side_noninferiority_margin
        ),
        "short_net_noninferiority": bool(
            float(candidate["short_net_return"])
            >= float(union["short_net_return"]) - config.side_noninferiority_margin
        ),
        "sortino_noninferiority": bool(
            float(candidate["sortino"])
            >= float(union["sortino"]) - config.sortino_noninferiority_margin
        ),
        "drawdown_noninferiority": bool(
            float(candidate["max_drawdown"])
            <= float(union["max_drawdown"]) + config.max_drawdown_absolute_margin
        ),
        "distributed_additional_trades": bool(
            len(nonzero) >= 2 and maximum_share <= config.max_extra_trade_concentration
        ),
        "transport_and_audits": bool(
            float(transport_failure_fraction) <= config.max_transport_failure_fraction
            and audits_passed
        ),
    }


def lexicographic_objective(
    gates: Mapping[str, Mapping[str, bool]],
    stage_metrics: Mapping[str, Mapping[str, Any]],
) -> tuple[int, int, int, float, float]:
    required = [gates[stage] for stage in REQUIRED_STAGES]
    all_passed = all(all(values.values()) for values in required)
    passed_count = sum(sum(bool(value) for value in values.values()) for values in required)
    additions = sum(int(stage_metrics[stage]["additional_trades"]) for stage in REQUIRED_STAGES)
    net = sum(float(stage_metrics[stage]["net_return"]) for stage in REQUIRED_STAGES)
    worst_drawdown = max(
        float(stage_metrics[stage]["max_drawdown"]) for stage in REQUIRED_STAGES
    )
    return (
        int(all_passed),
        int(passed_count),
        int(additions),
        float(net),
        -float(worst_drawdown),
    )


def _per_bar(selected: pd.DataFrame, frame: pd.DataFrame) -> pd.Series:
    start = frame["decision_time"].min().floor("15min")
    end = frame["outcome_available_time"].max().ceil("15min")
    result = pd.Series(0.0, index=pd.date_range(start, end, freq="15min"))
    trades = selected.loc[selected["selected"].astype(bool)].copy()
    if len(trades):
        booked = (
            trades.assign(_booking=trades["entry_time"].dt.floor("15min"))
            .groupby("_booking")["net_return"]
            .sum()
        )
        result.loc[booked.index] = booked.to_numpy(float)
    return result.rename("net_return")


def _recomputed_metrics(selected: pd.DataFrame, frame: pd.DataFrame) -> dict[str, Any]:
    trades = selected.loc[selected["selected"].astype(bool)].copy()
    economics = economics_summary(_per_bar(selected, frame))
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


def _assert_metrics_equal(stored: Mapping[str, Any], observed: Mapping[str, Any]) -> None:
    if set(stored) != set(observed):
        raise ValueError("stored metric keys changed")
    for key, value in observed.items():
        expected = stored[key]
        if isinstance(value, int):
            if int(expected) != value:
                raise ValueError(f"stored integer metric changed: {key}")
        elif not math.isfinite(value) or not math.isclose(
            float(expected), value, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"stored floating metric changed: {key}")


def _parse_prompt_payload(messages: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    if len(messages) != 2 or messages[0] != {"role": "system", "content": SYSTEM_PROMPT_V4}:
        raise ValueError("router system prompt changed")
    marker = "INPUT_JSON="
    content = str(messages[1].get("content", ""))
    if messages[1].get("role") != "user" or marker not in content:
        raise ValueError("router user prompt changed")
    return json.loads(content.split(marker, 1)[1])


def _verify_common(base: Path) -> tuple[dict[str, Any], dict[str, dict[str, pd.DataFrame]]]:
    common_root = base / "common"
    manifest = _read_json(common_root / "manifest.json")
    if manifest.get("status") != "complete" or manifest.get("lockbox_2026_q2_used") is not False:
        raise ValueError("common artifacts are incomplete or opened Q2")
    if manifest.get("implementation_hash") != _implementation_hash():
        raise ValueError("registered v4 implementation changed")
    if manifest.get("policy_ids") != list(POLICY_IDS):
        raise ValueError("registered policy menu changed")
    if pd.Timestamp(manifest["maximum_outcome_available_time"]) >= Q2_START:
        raise ValueError("common maximum outcome entered Q2")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        path = common_root / relative
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"common artifact hash mismatch: {relative}")

    stages: dict[str, dict[str, pd.DataFrame]] = {}
    for stage in STAGE_ORDER:
        counts = manifest["stage_counts"][stage]
        if any(int(counts[key]) != int(EXPECTED_COUNTS[stage][key]) for key in EXPECTED_COUNTS[stage]):
            raise ValueError(f"common stage count drift: {stage}")
        if int(counts.get("boundary_vetoes", -1)) != EXPECTED_BOUNDARY_VETOES[stage]:
            raise ValueError(f"weekly boundary-veto count drift: {stage}")
        frame = pd.read_parquet(common_root / f"{stage}_opportunities.parquet")
        payoffs = pd.read_parquet(common_root / f"{stage}_payoffs.parquet")
        for column in ("decision_time", "entry_time", "outcome_available_time"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        for column in ("block_start", "block_available_at"):
            payoffs[column] = pd.to_datetime(payoffs[column], utc=True)
        if frame["outcome_available_time"].ge(Q2_START).any():
            raise ValueError(f"Q2 entered common opportunities: {stage}")
        observed_cost = frame["gross_return"].astype(float) - frame["net_return"].astype(float)
        if not np.allclose(
            observed_cost,
            frame["round_trip_cost"].astype(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(f"common costs do not reconcile: {stage}")
        if stage == "development" and frame.loc[
            frame["decision_time"].lt(pd.Timestamp("2024-01-01", tz="UTC")),
            "xgb_available",
        ].astype(bool).any():
            raise ValueError("XGBoost support appeared before its OOF panel")
        if payoffs.groupby("block_id")["policy_id"].nunique().ne(len(POLICY_IDS)).any():
            raise ValueError(f"incomplete payoff menu: {stage}")
        week = frame["decision_time"].dt.floor("D") - pd.to_timedelta(
            frame["decision_time"].dt.dayofweek, unit="D"
        )
        block_mapping = {
            value: f"{stage}-b{index:03d}"
            for index, value in enumerate(sorted(week.unique()))
        }
        framed = frame.assign(
            block_id=week.map(block_mapping),
            expected_boundary_eligible=(
                frame["route"].eq("UNION_BASE")
                | frame["outcome_available_time"].lt(week + pd.Timedelta(days=7))
            ),
        )
        for block_id, rows in framed.groupby("block_id", sort=False):
            payoff_rows = payoffs.loc[payoffs["block_id"].eq(block_id)]
            if not payoff_rows["block_start"].eq(
                rows["decision_time"].min() - pd.Timedelta(microseconds=1)
            ).all():
                raise ValueError(f"block commitment is not pre-opportunity: {stage}/{block_id}")
            if not payoff_rows["block_available_at"].eq(
                rows["outcome_available_time"].max()
            ).all():
                raise ValueError(f"block feedback cutoff changed: {stage}/{block_id}")
        ledgers = {
            policy_id: pd.read_parquet(common_root / stage / f"{policy_id}.parquet")
            for policy_id in POLICY_IDS
        }
        if int(ledgers["LSTM_ALL"]["skip_reason"].eq("BLOCK_BOUNDARY").sum()) != EXPECTED_BOUNDARY_VETOES[stage]:
            raise ValueError(f"boundary veto ledger drift: {stage}")
        boundary_observed = ledgers["LSTM_ALL"].set_index("opportunity_id").loc[
            framed["opportunity_id"], "block_boundary_eligible"
        ].reset_index(drop=True)
        if not boundary_observed.astype(bool).equals(
            framed["expected_boundary_eligible"].reset_index(drop=True).astype(bool)
        ):
            raise ValueError(f"boundary eligibility formula changed: {stage}")
        for policy_id, ledger in ledgers.items():
            for block_id, rows in ledger.groupby("block_id", sort=False):
                payoff = payoffs.loc[
                    payoffs["block_id"].eq(block_id)
                    & payoffs["policy_id"].eq(policy_id)
                ]
                if len(payoff) != 1:
                    raise ValueError("payoff row identity changed")
                selected = rows.loc[rows["selected"].astype(bool)]
                union = selected.loc[selected["route"].eq("UNION_BASE")]
                extra = selected.loc[selected["route"].eq("COVERAGE_CANDIDATE")]
                checks = {
                    "union_trades": len(union),
                    "union_long_trades": union["side"].eq("LONG").sum(),
                    "union_short_trades": union["side"].eq("SHORT").sum(),
                    "additional_trades": len(extra),
                    "additional_long_trades": extra["side"].eq("LONG").sum(),
                    "additional_short_trades": extra["side"].eq("SHORT").sum(),
                    "combined_trades": len(selected),
                }
                for key, value in checks.items():
                    if int(payoff.iloc[0][key]) != int(value):
                        raise ValueError(f"policy payoff count changed: {stage}/{block_id}/{policy_id}")
                economic_checks = {
                    "union_net": union["net_return"].sum(),
                    "union_long_net": union.loc[union["side"].eq("LONG"), "net_return"].sum(),
                    "union_short_net": union.loc[union["side"].eq("SHORT"), "net_return"].sum(),
                    "incremental_gross": extra["gross_return"].sum(),
                    "incremental_cost": (extra["gross_return"] - extra["net_return"]).sum(),
                    "incremental_net": extra["net_return"].sum(),
                    "incremental_long_net": extra.loc[extra["side"].eq("LONG"), "net_return"].sum(),
                    "incremental_short_net": extra.loc[extra["side"].eq("SHORT"), "net_return"].sum(),
                    "combined_net": selected["net_return"].sum(),
                }
                for key, value in economic_checks.items():
                    if not math.isclose(
                        float(payoff.iloc[0][key]),
                        float(value),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"policy payoff economics changed: {stage}/{block_id}/{policy_id}/{key}"
                        )
        stages[stage] = {"frame": frame, "payoffs": payoffs, **ledgers}
    return manifest, stages


def _verify_prompt_calls(
    base: Path,
    variant: str,
    choices: pd.DataFrame,
    all_payoffs: pd.DataFrame,
) -> list[dict[str, Any]]:
    cache_dir = base / variant / "call_cache"
    payloads = [_read_json(path) for path in sorted(cache_dir.glob("*.json"))]
    by_request: dict[str, dict[str, Any]] = {}
    for payload in payloads:
        request_hash = str(payload["request_hash"])
        if request_hash in by_request:
            raise ValueError(f"duplicate router request cache: {variant}")
        by_request[request_hash] = payload
    if set(choices["request_hash"].astype(str)) != set(by_request):
        raise ValueError(f"router choices and exact call cache disagree: {variant}")
    payoff_availability = (
        all_payoffs.groupby("block_id")["block_available_at"].first().to_dict()
    )
    audits: list[dict[str, Any]] = []
    for choice in choices.itertuples(index=False):
        cached = by_request[str(choice.request_hash)]
        payload = _parse_prompt_payload(cached["messages"])
        audit = audit_prompt_payload(payload)
        reasons = list(audit["reasons"])
        allowed = cached["allowed_ids"]
        if allowed["choice_indices"] != [item["choice_index"] for item in payload["policy_menu"]]:
            reasons.append("choice_allowlist")
        if allowed["evidence_indices"] != list(range(len(payload["policy_statistics"]))):
            reasons.append("evidence_allowlist")
        if allowed["memory_indices"] != list(range(len(payload["memory_cards"]))):
            reasons.append("memory_allowlist")
        if int(choice.memory_cards_visible) != len(payload["memory_cards"]):
            reasons.append("memory_count")
        for card in payload["memory_cards"]:
            block_id = str(card.get("block_id", ""))
            if block_id not in payoff_availability:
                reasons.append("unknown_memory_block")
                continue
            if pd.Timestamp(payoff_availability[block_id]) >= pd.Timestamp(choice.decision_time):
                reasons.append("future_memory")
            policy_payoffs = card.get("policy_payoffs", [])
            if len(policy_payoffs) != len(POLICY_IDS) or {
                str(item.get("policy_id")) for item in policy_payoffs
            } != set(POLICY_IDS):
                reasons.append("incomplete_memory_payoff_vector")
        if variant == "reflection_no_memory" and (
            payload["memory_cards"] or payload["policy_statistics"]
        ):
            reasons.append("no_memory_received_memory")
        validated = cached.get("validated_content")
        if validated is not None:
            value = RouterChoice.model_validate(validated)
            valid_references = (
                value.choice_index in set(allowed["choice_indices"])
                and set(value.evidence_indices).issubset(allowed["evidence_indices"])
                and set(value.memory_indices).issubset(allowed["memory_indices"])
            )
            if valid_references and str(choice.call_status) in ACCEPTED_CALL_STATUSES:
                if int(choice.choice_index) != int(value.choice_index):
                    reasons.append("validated_choice_mismatch")
                if json.loads(choice.evidence_indices) != list(value.evidence_indices):
                    reasons.append("evidence_choice_mismatch")
                if json.loads(choice.memory_indices) != list(value.memory_indices):
                    reasons.append("memory_choice_mismatch")
            elif not valid_references and str(choice.call_status) != "invalid_reference":
                reasons.append("invalid_reference_not_closed")
        elif (
            int(choice.choice_index) != 0
            or str(choice.policy_id) != "UNION_ONLY"
            or json.loads(choice.evidence_indices)
            or json.loads(choice.memory_indices)
            or str(choice.call_status) in ACCEPTED_CALL_STATUSES
        ):
            reasons.append("terminal_transport_failure_not_closed")
        audits.append(
            {
                "variant": variant,
                "stage": str(choice.stage),
                "block_id": str(choice.block_id),
                "request_hash": str(choice.request_hash),
                "passed": not reasons,
                "reasons": json.dumps(sorted(set(reasons))),
            }
        )
    if not all(item["passed"] for item in audits):
        raise ValueError(f"prompt audit failed: {variant}")
    for stage, prior_prefix in (("h1", "development-"), ("forward", "h1-")):
        first = choices.loc[choices["stage"].eq(stage)].sort_values("decision_time").iloc[0]
        payload = _parse_prompt_payload(by_request[str(first["request_hash"])]["messages"])
        if variant != "reflection_no_memory" and not any(
            str(card.get("block_id", "")).startswith(prior_prefix)
            for card in payload["memory_cards"]
        ):
            raise ValueError(f"continuous cross-stage memory missing: {variant}/{stage}")
    return audits


def _assert_no_selected_overlap(ledger: pd.DataFrame) -> None:
    trades = ledger.loc[ledger["selected"].astype(bool)].sort_values(
        ["entry_time", "outcome_available_time", "opportunity_id"], kind="stable"
    )
    previous_end: pd.Timestamp | None = None
    for row in trades.itertuples(index=False):
        entry = pd.Timestamp(row.entry_time)
        outcome = pd.Timestamp(row.outcome_available_time)
        if previous_end is not None and entry <= previous_end:
            raise ValueError("routed selected trades overlap")
        previous_end = outcome


def _block_bootstrap(delta: np.ndarray, *, seed: int, samples: int = 2000) -> dict[str, float]:
    values = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(samples, len(values)))].sum(axis=1)
    return {
        "delta_net": float(values.sum()),
        "delta_net_ci95_low": float(np.quantile(draws, 0.025)),
        "delta_net_ci95_high": float(np.quantile(draws, 0.975)),
    }


def reconcile_final_experiment(root: str | Path = DEFAULT_ROOT) -> dict[str, Any]:
    base = Path(root).resolve()
    config = load_v4_config(DEFAULT_CONFIG)
    common, stages = _verify_common(base)
    preflight = _read_json(base / "preflight.json")
    if (
        preflight.get("passed") is not True
        or preflight.get("protocol_hash") != common["protocol_hash"]
        or preflight.get("implementation_hash") != common["implementation_hash"]
        or preflight.get("model") != config.model
        or preflight.get("model_digest") != config.required_model_digest
        or preflight.get("lockbox_2026_q2_used") is not False
        or preflight.get("prompt_hashes") != prompt_hashes()
    ):
        raise ValueError("registered preflight identity changed")

    all_payoffs = pd.concat(
        [stages[stage]["payoffs"] for stage in STAGE_ORDER], ignore_index=True
    )
    all_payoffs["block_start"] = pd.to_datetime(all_payoffs["block_start"], utc=True)
    all_payoffs["block_available_at"] = pd.to_datetime(
        all_payoffs["block_available_at"], utc=True
    )
    common_manifest_hash = _sha256_file(base / "common" / "manifest.json")
    records: dict[str, dict[str, Any]] = {}
    prompt_audits: list[dict[str, Any]] = []
    union_reference = {
        stage: stages[stage]["UNION_ONLY"].loc[
            stages[stage]["UNION_ONLY"]["route"].eq("UNION_BASE")
        ].sort_values("opportunity_id")[
            ["opportunity_id", "side", "entry_time", "outcome_available_time", "gross_return", "net_return", "round_trip_cost", "selected"]
        ].reset_index(drop=True)
        for stage in STAGE_ORDER
    }

    for variant in REGISTERED_VARIANTS:
        variant_root = base / variant
        manifest = _read_json(variant_root / "manifest.json")
        summary = _read_json(variant_root / "summary.json")
        if (
            manifest.get("status") != "complete"
            or manifest.get("variant") != variant
            or summary.get("status") != "complete"
            or summary.get("variant") != variant
            or manifest.get("protocol_hash") != common["protocol_hash"]
            or summary.get("protocol_hash") != common["protocol_hash"]
            or manifest.get("implementation_hash") != common["implementation_hash"]
            or summary.get("implementation_hash") != common["implementation_hash"]
            or manifest.get("common_manifest_hash") != common_manifest_hash
            or manifest.get("lockbox_2026_q2_used") is not False
            or summary.get("lockbox_2026_q2_used") is not False
        ):
            raise ValueError(f"variant identity or Q2 flag changed: {variant}")
        for relative, expected in manifest.get("artifact_hashes", {}).items():
            path = variant_root / relative
            if not path.is_file() or _sha256_file(path) != expected:
                raise ValueError(f"variant artifact hash mismatch: {variant}/{relative}")

        choices = pd.read_parquet(variant_root / "choices.parquet")
        choices["decision_time"] = pd.to_datetime(choices["decision_time"], utc=True)
        if len(choices) != sum(EXPECTED_COUNTS[stage]["blocks"] for stage in STAGE_ORDER):
            raise ValueError(f"router block count changed: {variant}")
        if choices["decision_time"].ge(Q2_START).any():
            raise ValueError(f"Q2 entered choices: {variant}")
        if any(
            str(row.policy_id) != POLICY_IDS[int(row.choice_index)]
            for row in choices.itertuples(index=False)
        ):
            raise ValueError(f"choice index/policy mismatch: {variant}")
        merged_choice = choices.merge(
            all_payoffs[["block_id", "stage", "choice_index", "policy_id"]],
            on=["block_id", "stage", "choice_index", "policy_id"],
            how="left",
            indicator=True,
        )
        if not merged_choice["_merge"].eq("both").all():
            raise ValueError(f"choice outside frozen payoff menu: {variant}")

        if variant in AGENT_VARIANTS:
            audits = _verify_prompt_calls(base, variant, choices, all_payoffs)
            prompt_audits.extend(audits)
            variant_audits_passed = all(item["passed"] for item in audits)
        else:
            if not choices["call_status"].eq("not_called").all():
                raise ValueError(f"deterministic control called the LLM: {variant}")
            if int(summary["transport_calls"]) != 0 or summary.get("controls_called_llm") is not False:
                raise ValueError(f"control transport summary changed: {variant}")
            variant_audits_passed = True

        stage_metrics: dict[str, dict[str, Any]] = {}
        additions_by_block: dict[str, dict[str, int]] = {}
        failure_fractions: dict[str, float] = {}
        for stage in STAGE_ORDER:
            frame = stages[stage]["frame"]
            ledger = pd.read_parquet(
                variant_root / "stages" / stage / "selected_ledger.parquet"
            )
            for column in ("decision_time", "entry_time", "outcome_available_time"):
                ledger[column] = pd.to_datetime(ledger[column], utc=True)
            if len(ledger) != len(frame) or ledger["opportunity_id"].duplicated().any():
                raise ValueError(f"routed opportunity coverage changed: {variant}/{stage}")
            aligned = ledger.set_index("opportunity_id").loc[
                frame["opportunity_id"]
            ].reset_index()
            for column in frame.columns:
                left = aligned[column]
                right = frame[column]
                if pd.api.types.is_numeric_dtype(right):
                    if not np.allclose(
                        pd.to_numeric(left, errors="coerce"),
                        pd.to_numeric(right, errors="coerce"),
                        equal_nan=True,
                        rtol=0.0,
                        atol=1e-12,
                    ):
                        raise ValueError(f"opportunity economics changed: {variant}/{stage}/{column}")
                elif not left.fillna("<NA>").astype(str).equals(
                    right.fillna("<NA>").astype(str)
                ):
                    raise ValueError(f"opportunity field changed: {variant}/{stage}/{column}")
            selected_union = ledger.loc[ledger["route"].eq("UNION_BASE")]
            if not selected_union["selected"].astype(bool).all():
                raise ValueError(f"Union was modified: {variant}/{stage}")
            union_view = selected_union.sort_values("opportunity_id")[
                ["opportunity_id", "side", "entry_time", "outcome_available_time", "gross_return", "net_return", "round_trip_cost", "selected"]
            ].reset_index(drop=True)
            if not union_view.equals(union_reference[stage]):
                raise ValueError(f"Union economics differ across variants: {variant}/{stage}")
            chosen_map = choices.loc[choices["stage"].eq(stage)].set_index("block_id")["policy_id"]
            if not ledger["block_id"].map(chosen_map).eq(ledger["policy_id"]).all():
                raise ValueError(f"routed ledger does not match weekly choices: {variant}/{stage}")
            if ledger.loc[
                ledger["selected"].astype(bool) & ledger["route"].eq("COVERAGE_CANDIDATE"),
                "block_boundary_eligible",
            ].eq(False).any():
                raise ValueError(f"weekly boundary veto failed: {variant}/{stage}")
            _assert_no_selected_overlap(ledger)
            observed = _recomputed_metrics(ledger, frame)
            stored = _read_json(variant_root / "stages" / stage / "summary.json")
            _assert_metrics_equal(stored, observed)
            _assert_metrics_equal(summary["stage_summaries"][stage], observed)
            stage_metrics[stage] = observed
            additions = (
                ledger.loc[
                    ledger["selected"].astype(bool)
                    & ledger["route"].eq("COVERAGE_CANDIDATE")
                ]
                .groupby("block_id")
                .size()
                .astype(int)
                .to_dict()
            )
            additions_by_block[stage] = {str(key): int(value) for key, value in additions.items()}
            stage_choices = choices.loc[choices["stage"].eq(stage), "call_status"]
            calls = int(stage_choices.ne("not_called").sum())
            failures = int((~stage_choices.isin({"not_called", *ACCEPTED_CALL_STATUSES})).sum())
            failure_fractions[stage] = float(failures / calls) if calls else 0.0

        records[variant] = {
            "choices": choices,
            "summary": summary,
            "stage_metrics": stage_metrics,
            "additions_by_block": additions_by_block,
            "failure_fractions": failure_fractions,
            "audits_passed": variant_audits_passed,
        }

    # Reproduce every zero-call router choice from the frozen payoff stream.
    deterministic = {
        "hedge_router": FullInformationRouter(caller=None, memory_mode="hedge"),
        "random_router": FullInformationRouter(
            caller=None, memory_mode="random", seed=config.random_seed
        ),
        **{
            f"static_{policy_id.lower()}": FullInformationRouter(
                caller=None,
                memory_mode="static",
                static_choice=index,
            )
            for index, policy_id in enumerate(POLICY_IDS)
        },
    }
    for variant, router in deterministic.items():
        reproduced = router.run(all_payoffs)
        if not reproduced[["block_id", "choice_index", "policy_id"]].reset_index(drop=True).equals(
            records[variant]["choices"][["block_id", "choice_index", "policy_id"]].reset_index(drop=True)
        ):
            raise ValueError(f"deterministic control is not reproducible: {variant}")

    gates: dict[str, dict[str, dict[str, bool]]] = {}
    union_metrics = records[UNION_VARIANT]["stage_metrics"]
    for variant, record in records.items():
        gates[variant] = {
            stage: coverage_gates(
                record["stage_metrics"][stage],
                union_metrics[stage],
                additional_by_block=record["additions_by_block"][stage],
                transport_failure_fraction=record["failure_fractions"][stage],
                audits_passed=bool(record["audits_passed"]),
            )
            for stage in STAGE_ORDER
        }

    objectives = {
        variant: lexicographic_objective(gates[variant], records[variant]["stage_metrics"])
        for variant in REGISTERED_VARIANTS
    }
    real = "reflection_real_memory"
    comparators = ("reflection_no_memory", "reflection_shuffled_memory", "hedge_router")
    coverage_success = all(all(gates[real][stage].values()) for stage in REQUIRED_STAGES)
    memory_effect_observed = any(
        not records[real]["choices"]["choice_index"].reset_index(drop=True).equals(
            records[variant]["choices"]["choice_index"].reset_index(drop=True)
        )
        for variant in comparators
    )
    memory_benefit = bool(
        coverage_success
        and memory_effect_observed
        and all(objectives[real] > objectives[variant] for variant in comparators)
    )

    comparisons: dict[str, dict[str, dict[str, float | int]]] = {}
    for stage in STAGE_ORDER:
        comparisons[stage] = {}
        stage_payoffs = stages[stage]["payoffs"]
        for variant, record in records.items():
            selected = record["choices"].loc[record["choices"]["stage"].eq(stage)][
                ["block_id", "choice_index", "policy_id"]
            ]
            joined = selected.merge(
                stage_payoffs[
                    ["block_id", "choice_index", "policy_id", "incremental_net", "additional_trades"]
                ],
                on=["block_id", "choice_index", "policy_id"],
                validate="one_to_one",
            )
            seed = int.from_bytes(
                hashlib.sha256(f"v4:{stage}:{variant}".encode()).digest()[:8], "big"
            )
            comparisons[stage][variant] = {
                **_block_bootstrap(joined["incremental_net"].to_numpy(float), seed=seed),
                "delta_trades": int(joined["additional_trades"].sum()),
                "blocks": int(len(joined)),
            }

    result_rows = {
        stage: {
            variant: records[variant]["stage_metrics"][stage]
            for variant in REGISTERED_VARIANTS
        }
        for stage in STAGE_ORDER
    }
    choice_counts = {
        variant: {
            str(key): int(value)
            for key, value in records[variant]["choices"]["policy_id"].value_counts().items()
        }
        for variant in REGISTERED_VARIANTS
    }
    all_prompt_passed = bool(prompt_audits) and all(item["passed"] for item in prompt_audits)
    best_variant = max(REGISTERED_VARIANTS, key=lambda variant: objectives[variant])
    return {
        "status": "complete",
        "conclusion": (
            "coverage_and_memory_benefit_established"
            if coverage_success and memory_benefit
            else "coverage_success_memory_benefit_not_established"
            if coverage_success
            else "coverage_success_not_established"
        ),
        "coverage_success": bool(coverage_success),
        "memory_benefit_established": bool(memory_benefit),
        "memory_effect_observed": bool(memory_effect_observed),
        "best_variant_by_registered_objective": best_variant,
        "recommended_frozen_policy": (
            real if coverage_success and memory_benefit else UNION_VARIANT
        ),
        "forward_evidence_role": "secondary_reused_forward",
        "lockbox_2026_q2_used": False,
        "artifact_hashes_verified": True,
        "union_invariant_across_variants": True,
        "continuous_stage_state_verified": True,
        "all_prompt_audits_passed": all_prompt_passed,
        "prompt_audit_passed": int(sum(item["passed"] for item in prompt_audits)),
        "prompt_audit_total": int(len(prompt_audits)),
        "controls_called_llm": False,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "reconciliation_hash": _sha256_file(Path(__file__).resolve()),
        "common_manifest_hash": common_manifest_hash,
        "stage_counts": common["stage_counts"],
        "results": result_rows,
        "coverage_gates": gates,
        "memory_objectives": {
            variant: list(value) for variant, value in objectives.items()
        },
        "choice_counts": choice_counts,
        "comparisons": comparisons,
        "prompt_audits": prompt_audits,
    }


def _write_report_artifacts(base: Path, report: dict[str, Any]) -> dict[str, Any]:
    result_rows = [
        {
            "stage": stage,
            "variant": variant,
            "gate_count": sum(report["coverage_gates"][variant][stage].values()),
            "all_gates_passed": all(report["coverage_gates"][variant][stage].values()),
            **metrics,
        }
        for stage, variants in report["results"].items()
        for variant, metrics in variants.items()
    ]
    gate_rows = [
        {"variant": variant, "stage": stage, "gate": gate, "passed": passed}
        for variant, stages in report["coverage_gates"].items()
        for stage, stage_gates in stages.items()
        for gate, passed in stage_gates.items()
    ]
    comparison_rows = [
        {"stage": stage, "variant": variant, **values}
        for stage, variants in report["comparisons"].items()
        for variant, values in variants.items()
    ]
    choice_rows = [
        {"variant": variant, "policy_id": policy_id, "blocks": blocks}
        for variant, values in report["choice_counts"].items()
        for policy_id, blocks in values.items()
    ]
    policy_payoffs = pd.concat(
        [
            pd.read_parquet(base / "common" / f"{stage}_payoffs.parquet")
            for stage in STAGE_ORDER
        ],
        ignore_index=True,
    )
    tables = {
        "results_table.parquet": pd.DataFrame(result_rows),
        "coverage_gates.parquet": pd.DataFrame(gate_rows),
        "paired_comparisons.parquet": pd.DataFrame(comparison_rows),
        "choice_summary.parquet": pd.DataFrame(choice_rows),
        "policy_payoffs.parquet": policy_payoffs,
        "prompt_audits.parquet": pd.DataFrame(report["prompt_audits"]),
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
    base = args.root.resolve()
    report = reconcile_final_experiment(base)
    sealed = _write_report_artifacts(base, report)
    print(json.dumps(sealed, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_prompt_payload",
    "coverage_gates",
    "lexicographic_objective",
    "reconcile_final_experiment",
]
