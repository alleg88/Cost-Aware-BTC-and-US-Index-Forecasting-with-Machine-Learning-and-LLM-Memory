"""Resumable H1-calibrated reversal scorer over the frozen USA500 ensemble."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Sequence

import numpy as np
import pandas as pd

from experiments.index_replication import _atomic_json, _atomic_parquet
from experiments.index_replication_protocol import daily_economics
from experiments.run_reflection_agent_v3 import _live_model_record
from experiments.run_usa500_reflection_weight_agent import (
    _canonical_json,
    _hash,
    _memory_audit_fields,
    _opportunity_prompt_rows,
    _prompt_memory,
    _read_jsonl,
    _sha256_file,
    _verified_common as _verified_source_common,
)
from reflection_agent.index_v1.engine import replay_registered_sides
from reflection_agent.index_v1.transport import CachedIndexSchemaCaller, IndexSchemaCaller
from reflection_agent.index_v2.config import (
    REGISTERED_VARIANTS,
    BudgetedReversalConfig,
    load_budgeted_reversal_config,
)
from reflection_agent.index_v2.contracts import ReversalScoreBatch, validate_score_batch
from reflection_agent.index_v2.policy import (
    ScorePolicy,
    apply_score_policy,
    calibrate_score_policy,
    seeded_hash_scores,
    select_h1_policy,
    uncertainty_scores,
)
from reflection_agent.index_v2.prompts import prompt_hashes, score_messages


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "usa500_budgeted_reversal_agent_v1.yaml"
CACHE = CODE_ROOT / "experiments" / "cache" / "usa500_budgeted_reversal_agent"
SOURCE_CACHE = CODE_ROOT / "experiments" / "cache" / "usa500_reflection_weight_agent"
AGENT_VARIANTS = (
    "budgeted_real_memory",
    "budgeted_no_memory",
    "budgeted_shuffled_memory",
)
CONTROL_VARIANTS = (
    "frozen_parent",
    "uncertainty_control",
    "seeded_hash_control",
)
SOURCE_DATA_FILES = (
    "h1_opportunities.parquet",
    "forward_opportunities.parquet",
    "h1_memory_cards.jsonl",
    "forward_base_memory_cards.jsonl",
    "forward_weekly_state.parquet",
)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    package = CODE_ROOT / "reflection_agent" / "index_v2"
    for path in sorted((Path(__file__), *package.glob("*.py"))):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, target)


def _validate_opportunities(
    frame: pd.DataFrame,
    *,
    split: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    q2_start: pd.Timestamp,
) -> None:
    required = {
        "opportunity_id",
        "signal_bar_open",
        "decision_time",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "original_side",
        "week_start",
    }
    probability_columns = {
        f"m{model_index:02d}_p_{label}"
        for model_index in range(9)
        for label in ("short", "flat", "long")
    }
    missing = required.union(probability_columns).difference(frame.columns)
    if frame.empty or missing:
        raise ValueError(f"{split} opportunities are empty or incomplete: {sorted(missing)}")
    if frame["opportunity_id"].astype(str).eq("").any() or frame["opportunity_id"].duplicated().any():
        raise ValueError(f"{split} opportunity identifiers must be nonempty and unique")
    for column in ("signal_bar_open", "decision_time", "entry_time", "exit_time", "week_start"):
        values = pd.to_datetime(frame[column], utc=True, errors="coerce")
        if values.isna().any():
            raise ValueError(f"{split} contains an invalid {column}")
        if values.max() >= q2_start:
            raise PermissionError(f"{split} source crossed the sealed Q2 boundary")
    signal = pd.to_datetime(frame["signal_bar_open"], utc=True)
    if signal.min() < start or signal.max() >= end:
        raise ValueError(f"{split} opportunities fall outside the frozen split")
    probabilities = frame.loc[:, sorted(probability_columns)].to_numpy(dtype=float)
    prices = frame.loc[:, ["entry_price", "exit_price"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{split} probabilities must be finite and bounded")
    if not np.isfinite(prices).all() or (prices <= 0.0).any():
        raise ValueError(f"{split} prices must be positive and finite")
    if not frame["original_side"].isin((-1, 1)).all():
        raise ValueError(f"{split} parent sides must be binary")


def prepare_common_artifacts(
    *,
    source_root: str | Path = SOURCE_CACHE,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Copy and bind the exact v1 H1/Forward opportunity universe; never open Q2."""
    config = load_budgeted_reversal_config(config_path)
    source = Path(source_root).resolve()
    source_common = source / "common"
    source_manifest = _verified_source_common(source)
    if source_manifest.get("q2_loaded") is not False:
        raise PermissionError("source manifest opened Q2")
    source_hashes = source_manifest.get("artifact_hashes", {})
    for name in ("protocol.json", *SOURCE_DATA_FILES):
        path = source_common / name
        if name not in source_hashes or not path.is_file() or _sha256_file(path) != source_hashes[name]:
            raise ValueError(f"source artifact hash is missing or changed: {name}")

    h1 = pd.read_parquet(source_common / "h1_opportunities.parquet")
    forward = pd.read_parquet(source_common / "forward_opportunities.parquet")
    _validate_opportunities(
        h1,
        split="H1",
        start=_utc(config.h1_start_utc),
        end=_utc(config.h1_end_utc),
        q2_start=_utc(config.q2_start_utc),
    )
    _validate_opportunities(
        forward,
        split="Forward",
        start=_utc(config.forward_start_utc),
        end=_utc(config.forward_end_utc),
        q2_start=_utc(config.q2_start_utc),
    )
    expected_counts = {"h1": int(len(h1)), "forward": int(len(forward))}
    if source_manifest.get("stage_counts") != expected_counts:
        raise ValueError("source stage counts do not match the frozen opportunities")

    cards = [
        *_read_jsonl(source_common / "h1_memory_cards.jsonl"),
        *_read_jsonl(source_common / "forward_base_memory_cards.jsonl"),
    ]
    for card in cards:
        available = _utc(card["available_at"])
        week = _utc(card["week_start"])
        if available < week or available >= _utc(config.q2_start_utc):
            raise PermissionError("memory-card timing crossed its causal/Q2 boundary")
    weekly_state = pd.read_parquet(source_common / "forward_weekly_state.parquet")
    state_week = pd.to_datetime(weekly_state["week_start"], utc=True, errors="coerce")
    state_available = pd.to_datetime(weekly_state["state_available_at"], utc=True, errors="coerce")
    if state_week.isna().any() or state_available.isna().any() or (state_available >= state_week).any():
        raise ValueError("forward weekly state is not strictly pre-commitment")
    if state_week.max() >= _utc(config.q2_start_utc):
        raise PermissionError("forward weekly state opened Q2")

    root = Path(output_root).resolve()
    common = root / "common"
    common.mkdir(parents=True, exist_ok=True)
    _atomic_copy(source_common / "protocol.json", common / "source_protocol.json")
    _atomic_copy(source_common / "manifest.json", common / "source_manifest.json")
    for name in SOURCE_DATA_FILES:
        _atomic_copy(source_common / name, common / name)

    source_identity = {
        "source_protocol_sha256": _sha256_file(common / "source_protocol.json"),
        "source_manifest_sha256": _sha256_file(common / "source_manifest.json"),
        **{
            f"source_{name}_sha256": _sha256_file(common / name)
            for name in SOURCE_DATA_FILES
        },
    }
    protocol_body = {
        "config": config.model_dump(mode="json"),
        "source_identity": dict(sorted(source_identity.items())),
        "implementation_hash": _implementation_hash(),
        "prompt_hashes": prompt_hashes(),
        "schema_hash": _hash(ReversalScoreBatch.model_json_schema()),
        "execution": "immutable_next_consecutive_m15_open_to_same_bar_close",
        "selection": "H1_only_sortino_net_drawdown_lower_rate",
        "q2_loaded": False,
    }
    protocol_hash = _hash(protocol_body)
    _atomic_json({**protocol_body, "protocol_hash": protocol_hash}, common / "protocol.json")
    artifact_names = (
        "protocol.json",
        "source_protocol.json",
        "source_manifest.json",
        *SOURCE_DATA_FILES,
    )
    manifest = {
        "status": "complete",
        "protocol_hash": protocol_hash,
        "implementation_hash": protocol_body["implementation_hash"],
        "prompt_hashes": protocol_body["prompt_hashes"],
        "schema_hash": protocol_body["schema_hash"],
        "source_identity": source_identity,
        "artifact_hashes": {name: _sha256_file(common / name) for name in artifact_names},
        "stage_counts": expected_counts,
        "maximum_signal_time": pd.to_datetime(forward["signal_bar_open"], utc=True).max().isoformat(),
        "q2_loaded": False,
    }
    _atomic_json(manifest, common / "manifest.json")
    return manifest


def _verified_common(root: Path) -> dict[str, Any]:
    manifest_path = root / "common" / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("USA500 budgeted-reversal common artifacts are not prepared")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("implementation_hash") != _implementation_hash():
        raise ValueError("budgeted-reversal common implementation identity changed")
    if manifest.get("q2_loaded") is not False:
        raise PermissionError("budgeted-reversal common artifacts opened Q2")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        path = root / "common" / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"budgeted-reversal common artifact hash changed: {relative}")
    protocol = json.loads((root / "common" / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_hash") != manifest.get("protocol_hash"):
        raise ValueError("budgeted-reversal protocol and manifest differ")
    if protocol.get("q2_loaded") is not False:
        raise PermissionError("budgeted-reversal protocol opened Q2")
    return manifest


def _preflight_payload() -> dict[str, Any]:
    probabilities = [
        {"model_index": index, "short": 0.2, "flat": 0.1, "long": 0.7}
        for index in range(9)
    ]
    return {
        "schema_version": "1.0",
        "opportunities": [
            {
                "opportunity_index": index,
                "original_side": "LONG",
                "model_probabilities": probabilities,
                "agreement": 1.0,
                "probability_dispersion": 0.0,
                "uncertainty_prior": 500,
                "current_market_state": {
                    "vix_regime": 0.0,
                    "trailing_volatility": 0.0,
                    "trailing_trend": 0.0,
                },
            }
            for index in range(10)
        ],
        "model_evidence": [],
        "memory_cards": [],
    }


def run_preflight(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    model_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the exact Ollama model and validate the strict score schema."""
    root = Path(output_root).resolve()
    common = _verified_common(root)
    config = load_budgeted_reversal_config(config_path)
    record = model_record or _live_model_record(config.model)
    if record.get("model") != config.model:
        raise RuntimeError("preflight resolved a different Ollama model tag")
    if str(record.get("digest", "")) != config.required_model_digest:
        raise RuntimeError("preflight Ollama model digest changed")
    if "thinking" not in set(record.get("capabilities", [])):
        raise RuntimeError("preflight model does not advertise thinking support")
    active = caller or IndexSchemaCaller(config, call_log_path=root / "preflight_calls.jsonl")
    allowed = {
        "opportunity_indices": list(range(10)),
        "evidence_indices": [],
        "memory_indices": [],
    }
    call = active.call(
        role="reversal_score_preflight",
        messages=score_messages(_preflight_payload()),
        response_model=ReversalScoreBatch,
        allowed_ids=allowed,
    )
    if call.value is None:
        raise RuntimeError(f"reversal-score strict-schema preflight failed: {call.errors}")
    validate_score_batch(
        call.value,
        allowed_opportunity_indices=allowed["opportunity_indices"],
        allowed_evidence_indices=[],
        allowed_memory_indices=[],
        uncertainty_priors=[
            int(row["uncertainty_prior"])
            for row in _preflight_payload()["opportunities"]
        ],
    )
    result = {
        "passed": True,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "model": config.model,
        "model_digest": config.required_model_digest,
        "ollama_version": str(record.get("ollama_version", "")),
        "capabilities": sorted(record.get("capabilities", [])),
        "score_status": str(call.status),
        "request_hash": str(call.request_hash),
        "response_hash": str(call.response_hash),
        "schema_hash": str(call.schema_hash),
        "q2_loaded": False,
    }
    _atomic_json(result, root / "preflight.json")
    return result


def _memory_mode(variant: str) -> str:
    if variant == "budgeted_real_memory":
        return "real"
    if variant == "budgeted_no_memory":
        return "no_memory"
    if variant == "budgeted_shuffled_memory":
        return "shuffled"
    raise ValueError(f"not a score-producing agent variant: {variant}")


def _checkpoint_identity(root: Path, common: dict[str, Any], variant: str) -> dict[str, Any]:
    return {
        "variant": variant,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
        "q2_loaded": False,
    }


def _validate_or_create_checkpoint(
    root: Path, common: dict[str, Any], variant: str
) -> Path:
    variant_root = root / variant
    variant_root.mkdir(parents=True, exist_ok=True)
    path = variant_root / "checkpoint.json"
    identity = _checkpoint_identity(root, common, variant)
    if path.is_file():
        current = json.loads(path.read_text(encoding="utf-8"))
        for key, expected in identity.items():
            if current.get(key) != expected:
                raise ValueError(f"variant resume identity changed: {key}")
    else:
        if any((variant_root / f"{split}_scores.parquet").exists() for split in ("h1", "forward")):
            raise ValueError("variant scores exist without a bound checkpoint identity")
        _atomic_json({**identity, "status": "running"}, path)
    return path


def _preflight_record(root: Path, common: dict[str, Any], config: BudgetedReversalConfig) -> dict[str, Any]:
    path = root / "preflight.json"
    if not path.is_file():
        raise FileNotFoundError("strict score preflight must pass before agent scoring")
    record = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "passed": True,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "model": config.model,
        "model_digest": config.required_model_digest,
        "q2_loaded": False,
    }
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f"score preflight identity changed: {key}")
    return record


def _existing_scores(path: Path, opportunities: pd.DataFrame) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    scores = pd.read_parquet(path).sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    if scores["opportunity_id"].duplicated().any():
        raise ValueError("score checkpoint contains duplicate opportunity identifiers")
    expected = opportunities["opportunity_id"].astype(str).tolist()[: len(scores)]
    if scores["opportunity_id"].astype(str).tolist() != expected:
        raise ValueError("score checkpoint is not an exact chronological prefix")
    return scores


def _score_prompt_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = _opportunity_prompt_rows(frame)
    priors = uncertainty_scores(frame)
    for row, prior in zip(rows, priors, strict=True):
        row["uncertainty_prior"] = int(prior)
    return rows


def _score_split(
    variant: str,
    *,
    split: str,
    root: Path,
    config: BudgetedReversalConfig,
    caller: Any,
    initial_history: Sequence[dict[str, Any]],
    max_batches: int | None,
) -> tuple[pd.DataFrame, bool, int]:
    opportunities = pd.read_parquet(root / "common" / f"{split}_opportunities.parquet")
    opportunities = opportunities.reset_index(drop=True)
    signal = pd.to_datetime(opportunities["signal_bar_open"], utc=True)
    if not signal.is_monotonic_increasing:
        raise ValueError(f"{split} opportunities are not chronologically ordered")
    scores_path = root / variant / f"{split}_scores.parquet"
    existing = _existing_scores(scores_path, opportunities)
    records = existing.to_dict(orient="records")
    existing_ids = set(existing.get("opportunity_id", pd.Series(dtype=str)).astype(str))
    base_cards = _read_jsonl(
        root
        / "common"
        / ("h1_memory_cards.jsonl" if split == "h1" else "forward_base_memory_cards.jsonl")
    )
    cards_by_start = {_utc(card["week_start"]): card for card in base_cards}
    history = list(initial_history)
    mode = _memory_mode(variant)
    new_batches = 0

    for week_start, group in opportunities.groupby("week_start", sort=True):
        week = _utc(week_start)
        current = group.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
        if week not in cards_by_start:
            raise ValueError(f"{split} memory card is missing for week {week.isoformat()}")
        evidence, cards = _prompt_memory(
            history,
            week_start=week,
            mode=mode,
            seed=config.seed,
            max_cards=config.max_memory_cards,
        )
        memory_audit = _memory_audit_fields(
            history,
            week_start=week,
            mode=mode,
            evidence=evidence,
            cards=cards,
        )
        for offset in range(0, len(current), config.score_batch_size):
            batch = current.iloc[offset : offset + config.score_batch_size].reset_index(drop=True)
            ids = batch["opportunity_id"].astype(str).tolist()
            already = [item in existing_ids for item in ids]
            if any(already) and not all(already):
                raise ValueError("score checkpoint split one immutable request batch")
            if all(already):
                continue
            allowed = {
                "opportunity_indices": list(range(len(batch))),
                "evidence_indices": list(range(len(evidence))),
                "memory_indices": list(range(len(cards))),
            }
            payload = {
                "schema_version": config.schema_version,
                "opportunities": _score_prompt_rows(batch),
                "model_evidence": evidence,
                "memory_cards": cards,
            }
            nonmemory_hash = _hash(
                {
                    "schema_version": payload["schema_version"],
                    "opportunities": payload["opportunities"],
                }
            )
            result = caller.call(
                role="reversal_score",
                messages=score_messages(payload),
                response_model=ReversalScoreBatch,
                allowed_ids=allowed,
            )
            status = str(result.status)
            fallback = True
            batch_scores = uncertainty_scores(batch)
            references: list[tuple[list[int], list[int]]] = [
                ([], []) for _ in range(len(batch))
            ]
            if result.value is not None:
                try:
                    batch_scores = np.asarray(
                        validate_score_batch(
                            result.value,
                            allowed_opportunity_indices=allowed["opportunity_indices"],
                            allowed_evidence_indices=allowed["evidence_indices"],
                            allowed_memory_indices=allowed["memory_indices"],
                            uncertainty_priors=[
                                int(row["uncertainty_prior"])
                                for row in payload["opportunities"]
                            ],
                        ),
                        dtype=int,
                    )
                    by_index = {
                        item.opportunity_index: item for item in result.value.decisions
                    }
                    references = [
                        (
                            list(by_index[index].evidence_indices),
                            list(by_index[index].memory_indices),
                        )
                        for index in range(len(batch))
                    ]
                    fallback = False
                except ValueError:
                    status = "invalid_output"
            call_id = f"{variant}:{split.upper()}:{week.strftime('%Y%m%d')}:{offset:04d}"
            for row_number, row in batch.iterrows():
                evidence_refs, memory_refs = references[row_number]
                records.append(
                    {
                        **{
                            key: row[key]
                            for key in (
                                "opportunity_id",
                                "signal_bar_open",
                                "decision_time",
                                "entry_time",
                                "exit_time",
                            )
                        },
                        "week_start": week,
                        "reversal_score": int(batch_scores[row_number]),
                        "call_id": call_id,
                        "call_status": status,
                        "fallback_used": bool(fallback),
                        "request_hash": str(result.request_hash),
                        "response_hash": str(result.response_hash),
                        "schema_hash": str(result.schema_hash),
                        "evidence_indices": _canonical_json(evidence_refs),
                        "memory_indices": _canonical_json(memory_refs),
                        "nonmemory_hash": nonmemory_hash,
                        **memory_audit,
                        "batch_size": int(len(batch)),
                    }
                )
                existing_ids.add(str(row["opportunity_id"]))
            current_scores = pd.DataFrame(records).sort_values(
                "signal_bar_open", kind="mergesort"
            ).reset_index(drop=True)
            _atomic_parquet(current_scores, scores_path)
            new_batches += 1
            if (
                max_batches is not None
                and new_batches >= max_batches
                and len(current_scores) < len(opportunities)
            ):
                return current_scores, False, new_batches
        completed = {str(item["opportunity_id"]) for item in records}
        if not set(current["opportunity_id"].astype(str)).issubset(completed):
            return pd.DataFrame(records), False, new_batches
        history.append(cards_by_start[week])

    final = pd.DataFrame(records).sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    if len(final) != len(opportunities):
        return final, False, new_batches
    return final, True, new_batches


def run_score_variant(
    variant: str,
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Score H1 then Forward causally; resume only under the exact bound identity."""
    if variant not in AGENT_VARIANTS:
        raise ValueError(f"unknown score-producing variant: {variant}")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive")
    root = Path(output_root).resolve()
    config = load_budgeted_reversal_config(config_path)
    common = _verified_common(root)
    _preflight_record(root, common, config)
    checkpoint_path = _validate_or_create_checkpoint(root, common, variant)
    if caller is None:
        inner = IndexSchemaCaller(
            config,
            call_log_path=root / variant / "calls.jsonl",
        )
        active: Any = CachedIndexSchemaCaller(
            inner,
            root / "call_cache",
            protocol_hash=common["protocol_hash"],
        )
    else:
        active = caller

    h1_scores, h1_complete, used = _score_split(
        variant,
        split="h1",
        root=root,
        config=config,
        caller=active,
        initial_history=[],
        max_batches=max_batches,
    )
    if not h1_complete:
        result = {
            "variant": variant,
            "status": "partial",
            "h1_scores": int(len(h1_scores)),
            "forward_scores": 0,
            "new_batches": int(used),
            "q2_loaded": False,
        }
        _atomic_json({**_checkpoint_identity(root, common, variant), **result}, checkpoint_path)
        return result

    if max_batches is not None and used >= max_batches:
        forward_path = root / variant / "forward_scores.parquet"
        forward_count = len(pd.read_parquet(forward_path)) if forward_path.is_file() else 0
        result = {
            "variant": variant,
            "status": "partial",
            "h1_scores": int(len(h1_scores)),
            "forward_scores": int(forward_count),
            "new_batches": int(used),
            "q2_loaded": False,
        }
        _atomic_json({**_checkpoint_identity(root, common, variant), **result}, checkpoint_path)
        return result

    remaining = None if max_batches is None else max_batches - used
    h1_history = _read_jsonl(root / "common" / "h1_memory_cards.jsonl")
    forward_scores, forward_complete, forward_used = _score_split(
        variant,
        split="forward",
        root=root,
        config=config,
        caller=active,
        initial_history=h1_history,
        max_batches=remaining,
    )
    status = "complete" if forward_complete else "partial"
    result = {
        "variant": variant,
        "status": status,
        "h1_scores": int(len(h1_scores)),
        "forward_scores": int(len(forward_scores)),
        "new_batches": int(used + forward_used),
        "q2_loaded": False,
    }
    _atomic_json({**_checkpoint_identity(root, common, variant), **result}, checkpoint_path)
    return result


def _exact_score_frame(
    root: Path, variant: str, split: str, opportunities: pd.DataFrame
) -> pd.DataFrame:
    path = root / variant / f"{split}_scores.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"complete {split} scores are missing for {variant}")
    scores = pd.read_parquet(path).sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    expected = opportunities["opportunity_id"].astype(str).tolist()
    if scores["opportunity_id"].astype(str).tolist() != expected:
        raise ValueError(f"{variant} {split} scores do not cover the exact opportunity set")
    values = scores["reversal_score"].to_numpy()
    if not all(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError(f"{variant} {split} scores are not integers")
    if ((values < 0) | (values > 1000)).any():
        raise ValueError(f"{variant} {split} scores are outside [0, 1000]")
    return scores


def _policy_sides(
    opportunities: pd.DataFrame,
    scores: Sequence[int],
    policy: ScorePolicy,
) -> tuple[np.ndarray, np.ndarray]:
    actions = apply_score_policy(
        scores,
        opportunities["opportunity_id"].astype(str).tolist(),
        policy,
    )
    original = opportunities["original_side"].to_numpy(dtype=int)
    sides = np.where(actions, -original, original).astype(int)
    return actions, sides


def _economic_metrics(
    opportunities: pd.DataFrame,
    sides: Sequence[int],
    *,
    config: BudgetedReversalConfig,
    split: str,
) -> dict[str, Any]:
    ledger, per_bar = replay_registered_sides(
        opportunities,
        sides,
        cost_bps=config.round_trip_cost_bps,
    )
    if split == "h1":
        start, end = config.h1_start_utc, config.h1_end_utc
    else:
        start, end = config.forward_start_utc, config.forward_end_utc
    return daily_economics(ledger, per_bar, start=start, end=end)


def freeze_h1_policies(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Select one reversal budget on H1 and freeze every arm before Forward actions."""
    root = Path(output_root).resolve()
    config = load_budgeted_reversal_config(config_path)
    common = _verified_common(root)
    h1 = pd.read_parquet(root / "common" / "h1_opportunities.parquet").reset_index(drop=True)
    real = _exact_score_frame(root, "budgeted_real_memory", "h1", h1)
    ids = h1["opportunity_id"].astype(str).tolist()
    grid_rows: list[dict[str, Any]] = []
    candidate_policies: dict[float, ScorePolicy] = {}
    for target_rate in config.target_change_rates:
        policy = calibrate_score_policy(
            real["reversal_score"].tolist(),
            ids,
            float(target_rate),
            config.seed,
        )
        candidate_policies[float(target_rate)] = policy
        actions, sides = _policy_sides(h1, real["reversal_score"].tolist(), policy)
        metrics = _economic_metrics(h1, sides, config=config, split="h1")
        grid_rows.append(
            {
                "target_rate": float(target_rate),
                "target_count": int(policy.target_count),
                "actual_count": int(actions.sum()),
                "actual_rate": float(actions.mean()),
                "threshold": int(policy.threshold),
                "tie_hash_cutoff": float(policy.tie_hash_cutoff),
                **metrics,
            }
        )
    grid = pd.DataFrame(grid_rows)
    selected = select_h1_policy(grid)
    selected_target = float(selected["target_rate"])

    policies: dict[str, ScorePolicy] = {}
    for variant in AGENT_VARIANTS:
        frame = _exact_score_frame(root, variant, "h1", h1)
        policies[variant] = calibrate_score_policy(
            frame["reversal_score"].tolist(),
            ids,
            selected_target,
            config.seed,
        )
    uncertainty = uncertainty_scores(h1)
    policies["uncertainty_control"] = calibrate_score_policy(
        uncertainty.tolist(), ids, selected_target, config.seed
    )
    hashed = seeded_hash_scores(ids, config.seed)
    policies["seeded_hash_control"] = calibrate_score_policy(
        hashed.tolist(), ids, selected_target, config.seed
    )

    grid_path = root / "common" / "h1_policy_grid.parquet"
    _atomic_parquet(
        grid.sort_values(
            ["daily_sortino", "net_return", "max_drawdown", "target_rate"],
            ascending=[False, False, False, True],
            kind="mergesort",
        ).reset_index(drop=True),
        grid_path,
    )
    frozen = {
        "status": "complete",
        "selection_split": "H1",
        "selection_rule": "daily_sortino,net_return,max_drawdown,lower_target_rate",
        "selected_target_rate": selected_target,
        "selected_real_memory_policy": asdict(candidate_policies[selected_target]),
        "policies": {variant: asdict(policy) for variant, policy in policies.items()},
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
        "h1_policy_grid_hash": _sha256_file(grid_path),
        "h1_score_hashes": {
            variant: _sha256_file(root / variant / "h1_scores.parquet")
            for variant in AGENT_VARIANTS
        },
        "forward_actions_observed": False,
        "q2_loaded": False,
    }
    _atomic_json(frozen, root / "common" / "frozen_policies.json")
    return frozen


def _load_frozen_policies(
    root: Path, common: dict[str, Any]
) -> dict[str, Any]:
    path = root / "common" / "frozen_policies.json"
    if not path.is_file():
        raise FileNotFoundError("H1 policies must be frozen before Forward actions")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
        "forward_actions_observed": False,
        "q2_loaded": False,
    }
    for key, value in expected.items():
        if frozen.get(key) != value:
            raise ValueError(f"frozen H1 policy identity changed: {key}")
    grid_path = root / "common" / "h1_policy_grid.parquet"
    if not grid_path.is_file() or _sha256_file(grid_path) != frozen.get("h1_policy_grid_hash"):
        raise ValueError("frozen H1 policy grid hash changed")
    required = set(AGENT_VARIANTS).union({"uncertainty_control", "seeded_hash_control"})
    if set(frozen.get("policies", {})) != required:
        raise ValueError("frozen H1 policies do not contain the exact registered scored arms")
    for variant, expected_hash in frozen.get("h1_score_hashes", {}).items():
        path = root / str(variant) / "h1_scores.parquet"
        if variant not in AGENT_VARIANTS or not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(f"frozen H1 score hash changed: {variant}")
    if set(frozen.get("h1_score_hashes", {})) != set(AGENT_VARIANTS):
        raise ValueError("frozen H1 score hashes are incomplete")
    return frozen


def _policy_from_record(record: dict[str, Any]) -> ScorePolicy:
    return ScorePolicy(
        target_rate=float(record["target_rate"]),
        threshold=int(record["threshold"]),
        tie_hash_cutoff=float(record["tie_hash_cutoff"]),
        seed=int(record["seed"]),
        calibration_size=int(record["calibration_size"]),
        target_count=int(record["target_count"]),
    )


def run_controls(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Apply the H1-frozen policies pointwise to all Forward opportunities."""
    root = Path(output_root).resolve()
    config = load_budgeted_reversal_config(config_path)
    common = _verified_common(root)
    frozen = _load_frozen_policies(root, common)
    frozen_hash = _sha256_file(root / "common" / "frozen_policies.json")
    forward = pd.read_parquet(root / "common" / "forward_opportunities.parquet").reset_index(drop=True)
    ids = forward["opportunity_id"].astype(str).tolist()
    original = forward["original_side"].to_numpy(dtype=int)
    summaries: dict[str, dict[str, Any]] = {}

    for variant in REGISTERED_VARIANTS:
        policy_record: dict[str, Any] | None = None
        if variant == "frozen_parent":
            scores = np.zeros(len(forward), dtype=int)
            actions = np.zeros(len(forward), dtype=bool)
            call_status = np.full(len(forward), "frozen_parent", dtype=object)
            fallback = np.zeros(len(forward), dtype=bool)
        else:
            policy_record = frozen["policies"][variant]
            policy = _policy_from_record(policy_record)
            if variant in AGENT_VARIANTS:
                score_frame = _exact_score_frame(root, variant, "forward", forward)
                scores = score_frame["reversal_score"].to_numpy(dtype=int)
                call_status = score_frame["call_status"].astype(str).to_numpy()
                fallback = score_frame["fallback_used"].to_numpy(dtype=bool)
            elif variant == "uncertainty_control":
                scores = uncertainty_scores(forward)
                call_status = np.full(len(forward), "deterministic_uncertainty", dtype=object)
                fallback = np.zeros(len(forward), dtype=bool)
            elif variant == "seeded_hash_control":
                scores = seeded_hash_scores(ids, config.seed)
                call_status = np.full(len(forward), "deterministic_hash", dtype=object)
                fallback = np.zeros(len(forward), dtype=bool)
            else:
                raise AssertionError(f"unhandled registered variant: {variant}")
            actions = apply_score_policy(scores.tolist(), ids, policy)
        sides = np.where(actions, -original, original).astype(int)
        decisions = forward.loc[
            :,
            [
                "opportunity_id",
                "signal_bar_open",
                "decision_time",
                "entry_time",
                "exit_time",
                "week_start",
            ],
        ].copy()
        decisions["original_side"] = original
        decisions["reversal_score"] = scores
        decisions["reversed"] = actions
        decisions["side"] = sides
        decisions["call_status"] = call_status
        decisions["fallback_used"] = fallback
        decisions["target_rate"] = (
            float(policy_record["target_rate"]) if policy_record is not None else 0.0
        )
        decisions["threshold"] = (
            int(policy_record["threshold"]) if policy_record is not None else 1001
        )
        ledger, per_bar = replay_registered_sides(
            forward,
            sides,
            cost_bps=config.round_trip_cost_bps,
        )
        metrics = daily_economics(
            ledger,
            per_bar,
            start=config.forward_start_utc,
            end=config.forward_end_utc,
        )
        summary = {
            "variant": variant,
            **metrics,
            "direction_changes": int(actions.sum()),
            "direction_change_rate": float(actions.mean()),
            "win_rate": float(ledger["net_return"].gt(0.0).mean()),
            "transport_failures": int(fallback.sum()) if variant in AGENT_VARIANTS else 0,
            "transport_failure_fraction": (
                float(fallback.mean()) if variant in AGENT_VARIANTS else 0.0
            ),
            "protocol_hash": common["protocol_hash"],
            "implementation_hash": common["implementation_hash"],
            "q2_loaded": False,
        }
        variant_root = root / variant
        _atomic_parquet(decisions, variant_root / "decisions.parquet")
        _atomic_parquet(ledger, variant_root / "ledger.parquet")
        _atomic_parquet(per_bar.rename("net_return").reset_index(), variant_root / "per_bar.parquet")
        _atomic_json(summary, variant_root / "summary.json")
        artifacts = {
            name: _sha256_file(variant_root / name)
            for name in ("decisions.parquet", "ledger.parquet", "per_bar.parquet", "summary.json")
        }
        manifest = {
                "status": "complete",
                "variant": variant,
                "protocol_hash": common["protocol_hash"],
                "implementation_hash": common["implementation_hash"],
                "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
                "frozen_policies_hash": frozen_hash,
                "artifact_hashes": artifacts,
                "q2_loaded": False,
            }
        if variant in AGENT_VARIANTS:
            manifest["score_hashes"] = {
                split: _sha256_file(variant_root / f"{split}_scores.parquet")
                for split in ("h1", "forward")
            }
        _atomic_json(manifest, variant_root / "manifest.json")
        summaries[variant] = summary
    return {
        "status": "complete",
        "variants": list(REGISTERED_VARIANTS),
        "selected_target_rate": float(frozen["selected_target_rate"]),
        "forward_opportunities": int(len(forward)),
        "direction_changes": {
            variant: int(summary["direction_changes"])
            for variant, summary in summaries.items()
        },
        "q2_loaded": False,
    }


def _verified_variant(
    root: Path,
    variant: str,
    common: dict[str, Any],
    frozen_hash: str,
) -> dict[str, Any]:
    variant_root = root / variant
    manifest_path = variant_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"registered Forward arm is incomplete: {variant}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "status": "complete",
        "variant": variant,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
        "frozen_policies_hash": frozen_hash,
        "q2_loaded": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"registered arm identity changed: {variant}:{key}")
    for relative, expected_hash in manifest.get("artifact_hashes", {}).items():
        path = variant_root / str(relative)
        if not path.is_file() or _sha256_file(path) != expected_hash:
            raise ValueError(f"registered arm artifact hash changed: {variant}:{relative}")
    required_artifacts = {
        "decisions.parquet",
        "ledger.parquet",
        "per_bar.parquet",
        "summary.json",
    }
    if set(manifest.get("artifact_hashes", {})) != required_artifacts:
        raise ValueError(f"registered arm artifact set changed: {variant}")
    if variant in AGENT_VARIANTS:
        score_hashes = manifest.get("score_hashes", {})
        if set(score_hashes) != {"h1", "forward"}:
            raise ValueError(f"registered agent score hashes are incomplete: {variant}")
        for split, expected_hash in score_hashes.items():
            path = variant_root / f"{split}_scores.parquet"
            if not path.is_file() or _sha256_file(path) != expected_hash:
                raise ValueError(f"registered agent score hash changed: {variant}:{split}")
    return manifest


def _expected_forward_actions(
    root: Path,
    variant: str,
    forward: pd.DataFrame,
    frozen: dict[str, Any],
    config: BudgetedReversalConfig,
) -> np.ndarray:
    ids = forward["opportunity_id"].astype(str).tolist()
    if variant == "frozen_parent":
        return np.zeros(len(forward), dtype=bool)
    policy = _policy_from_record(frozen["policies"][variant])
    if variant in AGENT_VARIANTS:
        scores = _exact_score_frame(root, variant, "forward", forward)[
            "reversal_score"
        ].to_numpy(dtype=int)
    elif variant == "uncertainty_control":
        scores = uncertainty_scores(forward)
    elif variant == "seeded_hash_control":
        scores = seeded_hash_scores(ids, config.seed)
    else:
        raise AssertionError(f"unhandled registered variant: {variant}")
    return apply_score_policy(scores.tolist(), ids, policy)


def _validated_forward_arm(
    root: Path,
    variant: str,
    forward: pd.DataFrame,
    frozen: dict[str, Any],
    config: BudgetedReversalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    variant_root = root / variant
    decisions = pd.read_parquet(variant_root / "decisions.parquet").reset_index(drop=True)
    ledger = pd.read_parquet(variant_root / "ledger.parquet").reset_index(drop=True)
    expected_ids = forward["opportunity_id"].astype(str).tolist()
    if (
        decisions["opportunity_id"].astype(str).tolist() != expected_ids
        or ledger["opportunity_id"].astype(str).tolist() != expected_ids
    ):
        raise ValueError(f"registered arm lost or reordered Forward opportunities: {variant}")
    if not decisions["side"].isin((-1, 1)).all():
        raise ValueError(f"registered arm produced a non-binary side: {variant}")
    expected_actions = _expected_forward_actions(root, variant, forward, frozen, config)
    if not np.array_equal(decisions["reversed"].to_numpy(dtype=bool), expected_actions):
        raise ValueError(f"registered arm actions do not match the H1-frozen policy: {variant}")
    original = forward["original_side"].to_numpy(dtype=int)
    expected_sides = np.where(expected_actions, -original, original).astype(int)
    if not np.array_equal(decisions["side"].to_numpy(dtype=int), expected_sides):
        raise ValueError(f"registered arm sides do not derive from KEEP/REVERSE: {variant}")
    replay, per_bar = replay_registered_sides(
        forward,
        expected_sides,
        cost_bps=config.round_trip_cost_bps,
    )
    numeric = ("entry_price", "exit_price", "gross_return", "cost_return", "net_return")
    if not np.allclose(
        ledger.loc[:, numeric].to_numpy(dtype=float),
        replay.loc[:, numeric].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-12,
    ) or not np.array_equal(
        ledger["side"].to_numpy(dtype=int), replay["side"].to_numpy(dtype=int)
    ):
        raise ValueError(f"registered arm replay does not match immutable prices: {variant}")
    saved_per_bar = pd.read_parquet(variant_root / "per_bar.parquet")
    saved_values = saved_per_bar["net_return"].to_numpy(dtype=float)
    if len(saved_values) != len(per_bar) or not np.allclose(
        saved_values, per_bar.to_numpy(dtype=float), rtol=0.0, atol=1e-12
    ):
        raise ValueError(f"registered arm per-bar replay changed: {variant}")
    return decisions, replay, per_bar


def _results_and_quality(
    root: Path,
    forward: pd.DataFrame,
    arms: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]],
    config: BudgetedReversalConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_ledger = arms["frozen_parent"][1]
    parent_net = parent_ledger["net_return"].to_numpy(dtype=float)
    h1 = pd.read_parquet(root / "common" / "h1_opportunities.parquet").reset_index(drop=True)
    h1_ids = h1["opportunity_id"].astype(str).tolist()
    result_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for variant in REGISTERED_VARIANTS:
        decisions, ledger, per_bar = arms[variant]
        metrics = daily_economics(
            ledger,
            per_bar,
            start=config.forward_start_utc,
            end=config.forward_end_utc,
        )
        changed = decisions["reversed"].to_numpy(dtype=bool)
        fallback = decisions["fallback_used"].to_numpy(dtype=bool)
        failures = int(fallback.sum()) if variant in AGENT_VARIANTS else 0
        fallback_changes = int((changed & fallback).sum()) if variant in AGENT_VARIANTS else 0
        llm_changes = int(changed.sum()) - fallback_changes if variant in AGENT_VARIANTS else 0
        attributed_fraction = float(llm_changes / changed.sum()) if variant in AGENT_VARIANTS and changed.sum() else 0.0
        result_rows.append(
            {
                "variant": variant,
                **metrics,
                "win_rate": float(ledger["net_return"].gt(0.0).mean()),
                "direction_changes": int(changed.sum()),
                "direction_change_rate": float(changed.mean()),
                "transport_failures": failures,
                "transport_failure_fraction": float(failures / len(decisions)),
                "fallback_attributed_changes": fallback_changes,
                "llm_attributed_changes": llm_changes,
                "llm_attributed_change_fraction": attributed_fraction,
            }
        )
        delta = ledger["net_return"].to_numpy(dtype=float) - parent_net
        changed_delta = delta[changed]
        tolerance = 1e-15
        if variant in AGENT_VARIANTS:
            h1_scores = _exact_score_frame(root, variant, "h1", h1)[
                "reversal_score"
            ].to_numpy(dtype=int)
            score_source = "LLM"
        elif variant == "uncertainty_control":
            h1_scores = uncertainty_scores(h1)
            score_source = "uncertainty"
        elif variant == "seeded_hash_control":
            h1_scores = seeded_hash_scores(h1_ids, config.seed)
            score_source = "seeded_hash"
        else:
            h1_scores = np.asarray([], dtype=int)
            score_source = "parent"
        if len(h1_scores):
            distinct = int(pd.Series(h1_scores).nunique())
            concentration = float(pd.Series(h1_scores).value_counts(normalize=True).max())
        else:
            distinct = 0
            concentration = 0.0
        quality_rows.append(
            {
                "variant": variant,
                "score_source": score_source,
                "direction_changes": int(changed.sum()),
                "beneficial_changes": int((changed_delta > tolerance).sum()),
                "harmful_changes": int((changed_delta < -tolerance).sum()),
                "neutral_changes": int((np.abs(changed_delta) <= tolerance).sum()),
                "change_net_contribution": float(changed_delta.sum()),
                "fallback_attributed_changes": fallback_changes,
                "llm_attributed_changes": llm_changes,
                "llm_attributed_change_fraction": attributed_fraction,
                "h1_distinct_scores": distinct,
                "h1_max_score_concentration": concentration,
            }
        )
    results = pd.DataFrame(result_rows).sort_values(
        ["net_return", "daily_sortino", "variant"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    quality = pd.DataFrame(quality_rows).sort_values(
        ["change_net_contribution", "variant"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return results, quality


def _memory_ablation(results: pd.DataFrame) -> pd.DataFrame:
    indexed = results.set_index("variant")
    real = indexed.loc["budgeted_real_memory"]
    rows = []
    for label, variant in (
        ("Real Memory", "budgeted_real_memory"),
        ("No Memory", "budgeted_no_memory"),
        ("Shuffled Memory", "budgeted_shuffled_memory"),
    ):
        current = indexed.loc[variant]
        rows.append(
            {
                "memory": label,
                "variant": variant,
                "trades": int(current["trades"]),
                "n_long": int(current["n_long"]),
                "n_short": int(current["n_short"]),
                "direction_changes": int(current["direction_changes"]),
                "net_return": float(current["net_return"]),
                "daily_sharpe": float(current["daily_sharpe"]),
                "daily_sortino": float(current["daily_sortino"]),
                "max_drawdown": float(current["max_drawdown"]),
                "net_delta_vs_real": float(current["net_return"] - real["net_return"]),
                "sortino_delta_vs_real": float(current["daily_sortino"] - real["daily_sortino"]),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["net_return", "daily_sortino", "variant"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _paired_bootstrap(
    forward: pd.DataFrame,
    arms: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]],
    *,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    parent = arms["frozen_parent"][1]["net_return"].to_numpy(dtype=float)
    week = pd.to_datetime(forward["week_start"], utc=True)
    rows = []
    for variant_index, variant in enumerate(REGISTERED_VARIANTS):
        current = arms[variant][1]["net_return"].to_numpy(dtype=float)
        blocks = pd.DataFrame({"week_start": week, "delta": current - parent}).groupby(
            "week_start", sort=True
        )["delta"].sum()
        values = blocks.to_numpy(dtype=float)
        rng = np.random.default_rng(int(seed) + variant_index)
        draws = rng.choice(values, size=(int(samples), len(values)), replace=True).sum(axis=1)
        rows.append(
            {
                "variant": variant,
                "weeks": int(len(values)),
                "net_delta": float(values.sum()),
                "ci_low": float(np.quantile(draws, 0.025)),
                "ci_high": float(np.quantile(draws, 0.975)),
                "bootstrap_samples": int(samples),
                "seed": int(seed),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["net_delta", "variant"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _leakage_audit(
    root: Path,
    common: dict[str, Any],
    forward: pd.DataFrame,
    arms: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series]],
    frozen_hash: str,
) -> pd.DataFrame:
    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        checks.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    q2 = _utc("2026-04-01T00:00:00Z")
    add(
        "q2_sealed",
        common.get("q2_loaded") is False
        and pd.to_datetime(forward["signal_bar_open"], utc=True).max() < q2,
        "All bound timestamps remain before 2026-04-01 UTC.",
    )
    expected_ids = forward["opportunity_id"].astype(str).tolist()
    add(
        "exact_opportunity_keys",
        all(
            decisions["opportunity_id"].astype(str).tolist() == expected_ids
            and ledger["opportunity_id"].astype(str).tolist() == expected_ids
            for decisions, ledger, _per_bar in arms.values()
        ),
        "Every arm retains the exact registered Forward keys.",
    )
    add(
        "binary_policy_replay",
        all(decisions["side"].isin((-1, 1)).all() for decisions, _ledger, _per_bar in arms.values()),
        "Every saved action is KEEP or REVERSE and every final side is binary.",
    )
    add(
        "immutable_execution_replay",
        True,
        "Every ledger was recomputed and matched immutable prices plus 2 bps cost.",
    )
    bound = all(
        json.loads((root / variant / "manifest.json").read_text(encoding="utf-8")).get(
            "frozen_policies_hash"
        )
        == frozen_hash
        for variant in REGISTERED_VARIANTS
    )
    add(
        "h1_policy_frozen_before_actions",
        bound,
        "Every Forward arm binds the same outcome-blind H1 policy artifact.",
    )
    same_week = True
    nonmemory = True
    for split in ("h1", "forward"):
        score_frames = [
            pd.read_parquet(root / variant / f"{split}_scores.parquet")
            for variant in AGENT_VARIANTS
        ]
        hashes = [frame["nonmemory_hash"].astype(str).tolist() for frame in score_frames]
        nonmemory &= hashes[0] == hashes[1] == hashes[2]
        for frame in score_frames:
            for row in frame.loc[frame["memory_max_available_at"].astype(str).ne("")].itertuples(index=False):
                same_week &= _utc(row.memory_max_available_at) < _utc(row.week_start)
    add(
        "same_week_outcomes_absent",
        same_week,
        "Every visible memory snapshot resolves strictly before its UTC week.",
    )
    add(
        "memory_ablation_nonmemory_parity",
        nonmemory,
        "Real, no-memory and shuffled prompts share identical current inputs.",
    )
    finite = all(
        np.isfinite(ledger.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
        for _decisions, ledger, _per_bar in arms.values()
    )
    add("finite_outputs", finite, "All replayed economic outputs are finite.")
    audit = pd.DataFrame(checks)
    if not audit["passed"].all():
        failed = audit.loc[~audit["passed"], "check_id"].astype(str).tolist()
        raise ValueError(f"leakage/integrity audit failed: {failed}")
    return audit


def _frozen_policy_table(frozen: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for variant, record in frozen["policies"].items():
        rows.append(
            {
                "variant": variant,
                "score_source": (
                    "LLM" if variant in AGENT_VARIANTS else "deterministic_control"
                ),
                "selected_target_rate": float(frozen["selected_target_rate"]),
                "target_rate": float(record["target_rate"]),
                "threshold": int(record["threshold"]),
                "tie_hash_cutoff": float(record["tie_hash_cutoff"]),
                "seed": int(record["seed"]),
                "calibration_size": int(record["calibration_size"]),
                "target_count": int(record["target_count"]),
            }
        )
    return pd.DataFrame(rows).sort_values("variant", kind="mergesort").reset_index(drop=True)


def _primary_gate(
    results: pd.DataFrame,
    quality: pd.DataFrame,
    bootstrap: pd.DataFrame,
    audit: pd.DataFrame,
    config: BudgetedReversalConfig,
    expected_opportunities: int,
) -> dict[str, Any]:
    indexed = results.set_index("variant")
    quality_indexed = quality.set_index("variant")
    bootstrap_indexed = bootstrap.set_index("variant")
    primary = indexed.loc["budgeted_real_memory"]
    parent = indexed.loc["frozen_parent"]
    diagnostic = quality_indexed.loc["budgeted_real_memory"]
    all_rows = bool(results["trades"].eq(expected_opportunities).all())
    influence_checks = {
        "integrity_pass": bool(audit["passed"].all()),
        "all_registered_opportunities_pass": all_rows,
        "minimum_direction_changes_pass": bool(
            primary["direction_change_rate"] >= config.minimum_forward_change_rate
        ),
        "maximum_direction_changes_pass": bool(
            primary["direction_change_rate"] <= config.maximum_forward_change_rate
        ),
        "score_diversity_pass": bool(
            diagnostic["h1_distinct_scores"] >= config.minimum_distinct_h1_scores
        ),
        "score_concentration_pass": bool(
            diagnostic["h1_max_score_concentration"]
            <= config.maximum_h1_score_concentration
        ),
        "transport_failure_pass": bool(
            primary["transport_failure_fraction"]
            <= config.max_transport_failure_fraction
        ),
        "llm_attribution_pass": bool(
            primary["llm_attributed_change_fraction"]
            >= config.minimum_llm_attributed_change_fraction
        ),
    }
    memory_ids = ["budgeted_no_memory", "budgeted_shuffled_memory"]
    control_ids = ["uncertainty_control", "seeded_hash_control"]
    economic_checks = {
        "net_improvement_pass": bool(primary["net_return"] > parent["net_return"]),
        "sortino_improvement_pass": bool(
            primary["daily_sortino"] > parent["daily_sortino"]
        ),
        "drawdown_margin_pass": bool(
            primary["max_drawdown"]
            >= parent["max_drawdown"] - config.max_drawdown_absolute_margin
        ),
        "memory_ablation_pass": bool(
            (primary["net_return"] >= indexed.loc[memory_ids, "net_return"]).all()
            and (
                primary["daily_sortino"]
                >= indexed.loc[memory_ids, "daily_sortino"]
            ).all()
        ),
        "matched_controls_pass": bool(
            (primary["net_return"] > indexed.loc[control_ids, "net_return"]).all()
            and (
                primary["daily_sortino"]
                > indexed.loc[control_ids, "daily_sortino"]
            ).all()
        ),
        "bootstrap_lower_bound_pass": bool(
            bootstrap_indexed.loc["budgeted_real_memory", "ci_low"] > 0.0
        ),
    }
    influence_pass = bool(all(influence_checks.values()))
    economic_pass = bool(all(economic_checks.values()))
    return {
        **influence_checks,
        **economic_checks,
        "influence_gate_pass": influence_pass,
        "economic_gate_pass": economic_pass,
        "candidate_pass": bool(influence_pass and economic_pass),
    }


def finalize_results(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Publish exact six-arm economics, matched comparisons and promotion gates."""
    root = Path(output_root).resolve()
    config = load_budgeted_reversal_config(config_path)
    common = _verified_common(root)
    frozen = _load_frozen_policies(root, common)
    frozen_path = root / "common" / "frozen_policies.json"
    frozen_hash = _sha256_file(frozen_path)
    for variant in REGISTERED_VARIANTS:
        _verified_variant(root, variant, common, frozen_hash)
    forward = pd.read_parquet(root / "common" / "forward_opportunities.parquet").reset_index(drop=True)
    arms = {
        variant: _validated_forward_arm(root, variant, forward, frozen, config)
        for variant in REGISTERED_VARIANTS
    }
    results, quality = _results_and_quality(root, forward, arms, config)
    ablation = _memory_ablation(results)
    bootstrap = _paired_bootstrap(
        forward,
        arms,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    audit = _leakage_audit(root, common, forward, arms, frozen_hash)
    gate = _primary_gate(
        results,
        quality,
        bootstrap,
        audit,
        config,
        len(forward),
    )
    gate_table = pd.DataFrame(
        [
            {
                "variant": "budgeted_real_memory",
                "direction_changes": int(
                    results.set_index("variant").loc[
                        "budgeted_real_memory", "direction_changes"
                    ]
                ),
                "direction_change_rate": float(
                    results.set_index("variant").loc[
                        "budgeted_real_memory", "direction_change_rate"
                    ]
                ),
                **gate,
            }
        ]
    )
    h1_grid = pd.read_parquet(root / "common" / "h1_policy_grid.parquet")
    policy_table = _frozen_policy_table(frozen)
    frames = {
        "results_table.parquet": results,
        "h1_policy_grid.parquet": h1_grid,
        "frozen_policies.parquet": policy_table,
        "change_quality.parquet": quality,
        "memory_ablation.parquet": ablation,
        "paired_bootstrap.parquet": bootstrap,
        "leakage_audit.parquet": audit,
        "candidate_gate.parquet": gate_table,
    }
    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"final comparison artifact is empty: {name}")
        numeric = frame.select_dtypes(include=[np.number])
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"final comparison artifact is non-finite: {name}")
        _atomic_parquet(frame, root / name)
    summary = {
        "status": "complete",
        "primary_variant": "budgeted_real_memory",
        "registered_arms": int(len(REGISTERED_VARIANTS)),
        "forward_opportunities": int(len(forward)),
        "selected_target_rate": float(frozen["selected_target_rate"]),
        "best_variant_by_net": str(results.iloc[0]["variant"]),
        "best_net_return": float(results.iloc[0]["net_return"]),
        "primary_gate": gate,
        "candidate_count": int(gate["candidate_pass"]),
        "all_integrity_checks_pass": bool(audit["passed"].all()),
        "evidence_role": "exploratory_forward_overlay",
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "q2_loaded": False,
    }
    _atomic_json(summary, root / "summary.json")
    artifacts = {
        name: _sha256_file(root / name) for name in (*frames, "summary.json")
    }
    manifest = {
        "status": "complete",
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
        "frozen_policies_hash": frozen_hash,
        "variant_manifest_hashes": {
            variant: _sha256_file(root / variant / "manifest.json")
            for variant in REGISTERED_VARIANTS
        },
        "artifact_hashes": artifacts,
        "q2_loaded": False,
    }
    _atomic_json(manifest, root / "manifest.json")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--prepare", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--score", choices=AGENT_VARIANTS)
    action.add_argument("--score-all", action="store_true")
    action.add_argument("--freeze-h1", action="store_true")
    action.add_argument("--run-controls", action="store_true")
    action.add_argument("--finalize", action="store_true")
    action.add_argument("--all", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--source-cache", type=Path, default=SOURCE_CACHE)
    parser.add_argument("--max-batches", type=int)
    args = parser.parse_args(argv)
    if args.prepare:
        result: Any = prepare_common_artifacts(
            source_root=args.source_cache,
            output_root=args.cache_dir,
        )
    elif args.preflight:
        result = run_preflight(output_root=args.cache_dir)
    elif args.score:
        result = run_score_variant(
            args.score,
            output_root=args.cache_dir,
            max_batches=args.max_batches,
        )
    elif args.score_all:
        result = {
            variant: run_score_variant(
                variant,
                output_root=args.cache_dir,
                max_batches=args.max_batches,
            )
            for variant in AGENT_VARIANTS
        }
    elif args.freeze_h1:
        result = freeze_h1_policies(output_root=args.cache_dir)
    elif args.run_controls:
        result = run_controls(output_root=args.cache_dir)
    elif args.finalize:
        result = finalize_results(output_root=args.cache_dir)
    else:
        result = {
            "prepare": prepare_common_artifacts(
                source_root=args.source_cache,
                output_root=args.cache_dir,
            )
        }
        result["preflight"] = run_preflight(output_root=args.cache_dir)
        result["scores"] = {
            variant: run_score_variant(variant, output_root=args.cache_dir)
            for variant in AGENT_VARIANTS
        }
        result["freeze_h1"] = freeze_h1_policies(output_root=args.cache_dir)
        result["controls"] = run_controls(output_root=args.cache_dir)
        result["finalize"] = finalize_results(output_root=args.cache_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGENT_VARIANTS",
    "CACHE",
    "CONTROL_VARIANTS",
    "finalize_results",
    "freeze_h1_policies",
    "main",
    "prepare_common_artifacts",
    "run_controls",
    "run_preflight",
    "run_score_variant",
]
