"""Resumable USA500 Direct and weekly reflection-weight agent experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.index_all_model_ensemble import (
    AlignedPanel,
    IndexAllModelEnsembleConfig,
    IndexAllModelEnsembleRunner,
    validate_source_contract,
)
from experiments.index_replication import _atomic_json, _atomic_parquet, _frame_hash
from experiments.index_replication_protocol import daily_economics
from experiments.run_reflection_agent_v3 import _live_model_record
from reflection_agent.index_v1 import engine as agent_engine
from reflection_agent.index_v1.config import (
    REGISTERED_VARIANTS,
    IndexAgentConfig,
    load_index_agent_config,
)
from reflection_agent.index_v1.contracts import (
    DirectBatchDecision,
    WeeklyWeightDecision,
    validate_direct_batch,
    validate_weekly_weights,
)
from reflection_agent.index_v1.engine import (
    add_control_payoffs,
    build_causal_week_states,
    build_registered_opportunities,
    build_weekly_cards,
    causal_hedge_schedule,
    eligible_memory,
    equal_weights,
    memory_statistics,
    opportunity_keys,
    replay_registered_sides,
    shuffled_memory,
    tune_static_h1_weights,
    weighted_sides,
)
from reflection_agent.index_v1.prompts import (
    direct_messages,
    prompt_hashes,
    weekly_weight_messages,
)
from reflection_agent.index_v1.transport import CachedIndexSchemaCaller, IndexSchemaCaller


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "usa500_reflection_weight_agent_v1.yaml"
CACHE = CODE_ROOT / "experiments" / "cache" / "usa500_reflection_weight_agent"
PARENT_CACHE = CODE_ROOT / "experiments" / "cache" / "index_all_model_ensemble" / "usa500"
CONTROL_VARIANTS = ("original_frozen", "hedge_weekly", "static_h1")
AGENT_VARIANTS = tuple(
    variant for variant in REGISTERED_VARIANTS if variant not in CONTROL_VARIANTS
)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    package_files = Path(agent_engine.__file__).parent.glob("*.py")
    for path in sorted((Path(__file__), *package_files)):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _panel_hash(panel: AlignedPanel) -> str:
    frame = pd.DataFrame({"timestamp": panel.timestamp, "y_true": panel.y_true})
    for model, values in panel.probabilities.items():
        for class_index, label in enumerate(("short", "flat", "long")):
            frame[f"{model}_{label}"] = np.asarray(values, dtype=float)[:, class_index]
    return _frame_hash(frame)


def _parent_inputs(
    config: IndexAgentConfig,
) -> tuple[
    pd.DataFrame,
    AlignedPanel,
    AlignedPanel,
    pd.DataFrame,
    dict[str, str],
    pd.DataFrame,
]:
    parent_protocol_path = PARENT_CACHE / "protocol.json"
    candidate_path = PARENT_CACHE / "forward" / f"{config.source_candidate_id}.json"
    if not parent_protocol_path.is_file() or not candidate_path.is_file():
        raise FileNotFoundError("frozen USA500 ensemble artifacts are incomplete")
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    policy = candidate.get("policy", {})
    expected_policy = {
        "candidate_id": config.source_candidate_id,
        "arm": config.source_arm,
        "variant": config.source_variant,
        "width_bps": config.source_width_bps,
        "tau": config.source_tau,
    }
    for key, expected in expected_policy.items():
        if policy.get(key) != expected:
            raise ValueError(f"frozen USA500 ensemble policy changed: {key}")
    if candidate.get("protocol_hash") != parent_protocol.get("protocol_hash"):
        raise ValueError("candidate and parent ensemble protocols differ")
    ledger_record = candidate.get("artifacts", {}).get("ledger", {})
    ledger_path = (PARENT_CACHE / str(ledger_record.get("path", ""))).resolve()
    if PARENT_CACHE.resolve() not in ledger_path.parents or not ledger_path.is_file():
        raise ValueError("candidate ledger path left the frozen USA500 cache")
    if _sha256_file(ledger_path) != ledger_record.get("sha256"):
        raise ValueError("candidate ledger hash changed")

    ensemble_config = IndexAllModelEnsembleConfig.for_stream("usa500")
    source_contract = validate_source_contract(ensemble_config)
    parent_runner = IndexAllModelEnsembleRunner(ensemble_config)
    source_runner = parent_runner._source_runner()
    h1_panel = parent_runner._h1_panel(
        source_runner, config.source_arm, config.source_width_bps
    )
    forward_panel = parent_runner._forward_panel(
        source_runner, config.source_arm, config.source_width_bps
    )
    features, _ = source_runner.dataset(config.source_arm, config.source_width_bps)
    vix_level = features["vix_log_close"].astype(float)
    rolling_mean = vix_level.rolling(520, min_periods=20).mean()
    rolling_scale = vix_level.rolling(520, min_periods=20).std(ddof=0)
    state_frame = pd.DataFrame(
        {
            "available_at": pd.to_datetime(
                source_runner.bars["available_at"].reindex(features.index), utc=True
            ),
            "state_vix_regime": vix_level.sub(rolling_mean).div(
                rolling_scale.replace(0.0, np.nan)
            ).fillna(0.0),
            "state_trailing_vol": features["vol_20"].astype(float),
            "state_trailing_trend": features["r20"].astype(float),
        },
        index=features.index,
    )
    if state_frame.isna().any().any() or state_frame.index.max() >= config.q2_start_utc:
        raise ValueError("causal state frame is incomplete or crossed Q2")
    source_identity = {
        "parent_protocol_sha256": _sha256_file(parent_protocol_path),
        "candidate_sha256": _sha256_file(candidate_path),
        "candidate_ledger_sha256": _sha256_file(ledger_path),
        "source_protocol_hash": str(source_contract["source_protocol_hash"]),
        "bars_frame_sha256": _frame_hash(source_runner.bars),
        "h1_panel_sha256": _panel_hash(h1_panel),
        "forward_panel_sha256": _panel_hash(forward_panel),
        "causal_state_frame_sha256": _frame_hash(state_frame),
    }
    return (
        source_runner.bars,
        h1_panel,
        forward_panel,
        pd.read_parquet(ledger_path),
        source_identity,
        state_frame,
    )


def _atomic_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )
    os.replace(temporary, path)


def _reconcile_parent_ledger(
    generated: pd.DataFrame, expected: pd.DataFrame
) -> dict[str, bool]:
    columns = (
        "signal_bar_open",
        "decision_time",
        "entry_bar_open",
        "exit_bar_open",
        "entry_time",
        "exit_time",
        "side",
        "entry_price",
        "exit_price",
        "confidence",
        "gross_return",
        "cost_return",
        "net_return",
    )
    missing = set(columns).difference(generated.columns).union(
        set(columns).difference(expected.columns)
    )
    if missing or len(generated) != len(expected):
        raise ValueError("generated opportunities do not reconcile with parent ledger rows")
    left = generated.loc[:, columns].reset_index(drop=True)
    right = expected.loc[:, columns].reset_index(drop=True)
    time_columns = columns[:6]
    for column in time_columns:
        if not pd.to_datetime(left[column], utc=True).equals(
            pd.to_datetime(right[column], utc=True)
        ):
            raise ValueError("generated opportunities do not reconcile with parent ledger keys")
    if not np.array_equal(left["side"].to_numpy(int), right["side"].to_numpy(int)):
        raise ValueError("generated opportunities do not reconcile with parent ledger sides")
    numeric = columns[7:]
    if not np.allclose(
        left.loc[:, numeric].to_numpy(float),
        right.loc[:, numeric].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("generated opportunities do not reconcile with parent ledger economics")
    return {"exact_rows": True, "exact_keys": True, "exact_sides": True, "exact_economics": True}


def prepare_common_artifacts(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    bars: pd.DataFrame | None = None,
    h1_panel: AlignedPanel | None = None,
    forward_panel: AlignedPanel | None = None,
    expected_forward_ledger: pd.DataFrame | None = None,
    source_identity: Mapping[str, str] | None = None,
    state_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Freeze and hash the parent ensemble opportunities before any LLM call."""
    config = load_index_agent_config(config_path)
    supplied = (
        bars,
        h1_panel,
        forward_panel,
        expected_forward_ledger,
        source_identity,
        state_frame,
    )
    if all(item is None for item in supplied):
        (
            bars,
            h1_panel,
            forward_panel,
            expected_forward_ledger,
            identity,
            state_frame,
        ) = _parent_inputs(config)
        source_identity = identity
    elif any(item is None for item in supplied):
        raise ValueError("synthetic preparation requires every frozen input")
    assert bars is not None
    assert h1_panel is not None
    assert forward_panel is not None
    assert expected_forward_ledger is not None
    assert source_identity is not None
    assert state_frame is not None
    source_identity = {
        **dict(source_identity),
        "causal_state_frame_sha256": _frame_hash(state_frame),
    }
    if not source_identity or any(len(str(value)) != 64 for value in source_identity.values()):
        raise ValueError("source identity must contain 64-hex hashes")

    h1_opportunities, _ = build_registered_opportunities(
        h1_panel,
        bars,
        start=config.h1_start_utc,
        end=config.h1_end_utc,
        tau=config.source_tau,
        cost_bps=config.round_trip_cost_bps,
        state_frame=state_frame,
    )
    forward_opportunities, generated_ledger = build_registered_opportunities(
        forward_panel,
        bars,
        start=config.forward_start_utc,
        end=config.forward_end_utc,
        tau=config.source_tau,
        cost_bps=config.round_trip_cost_bps,
        state_frame=state_frame,
    )
    reconciliation = _reconcile_parent_ledger(generated_ledger, expected_forward_ledger)
    h1_cards = build_weekly_cards(
        h1_opportunities, cost_bps=config.round_trip_cost_bps
    )
    forward_cards = build_weekly_cards(
        forward_opportunities, cost_bps=config.round_trip_cost_bps
    )
    static_weights = tune_static_h1_weights(
        h1_opportunities, cost_bps=config.round_trip_cost_bps
    )
    all_cards = sorted(
        [*h1_cards, *forward_cards], key=lambda item: pd.Timestamp(item["week_start"])
    )
    hedge_schedule = causal_hedge_schedule(all_cards, eta=config.hedge_eta)
    static_schedule = {card["week_id"]: static_weights for card in all_cards}
    control_weights = {
        "static_h1": static_schedule,
        "hedge_weekly": hedge_schedule,
    }
    h1_cards = add_control_payoffs(
        h1_cards,
        h1_opportunities,
        control_weights=control_weights,
        cost_bps=config.round_trip_cost_bps,
    )
    forward_cards = add_control_payoffs(
        forward_cards,
        forward_opportunities,
        control_weights=control_weights,
        cost_bps=config.round_trip_cost_bps,
    )
    weekly_state = build_causal_week_states(
        state_frame,
        pd.to_datetime(forward_opportunities["week_start"], utc=True).unique(),
    )
    root = Path(output_root).resolve()
    common = root / "common"
    common.mkdir(parents=True, exist_ok=True)
    protocol_body = {
        "config": config.model_dump(mode="json"),
        "source_identity": dict(sorted(source_identity.items())),
        "implementation_hash": _implementation_hash(),
        "prompt_hashes": prompt_hashes(),
        "schema_hashes": {
            "direct_batch": _hash(DirectBatchDecision.model_json_schema()),
            "weekly_weights": _hash(WeeklyWeightDecision.model_json_schema()),
        },
        "execution": "next_consecutive_m15_open_to_same_bar_close",
        "q2_loaded": False,
    }
    protocol_hash = _hash(protocol_body)
    protocol = {**protocol_body, "protocol_hash": protocol_hash}
    _atomic_json(protocol, common / "protocol.json")
    _atomic_parquet(h1_opportunities, common / "h1_opportunities.parquet")
    _atomic_parquet(forward_opportunities, common / "forward_opportunities.parquet")
    _atomic_jsonl(h1_cards, common / "h1_memory_cards.jsonl")
    _atomic_jsonl(forward_cards, common / "forward_base_memory_cards.jsonl")
    _atomic_parquet(weekly_state, common / "forward_weekly_state.parquet")
    _atomic_json(
        {
            "static_h1_weights": static_weights,
            "hedge_schedule": hedge_schedule,
            "h1_only_static_tuning": True,
        },
        common / "control_weights.json",
    )
    artifacts = {
        name: _sha256_file(common / name)
        for name in (
            "protocol.json",
            "h1_opportunities.parquet",
            "forward_opportunities.parquet",
            "h1_memory_cards.jsonl",
            "forward_base_memory_cards.jsonl",
            "forward_weekly_state.parquet",
            "control_weights.json",
        )
    }
    manifest = {
        "status": "complete",
        "protocol_hash": protocol_hash,
        "implementation_hash": protocol_body["implementation_hash"],
        "prompt_hashes": protocol_body["prompt_hashes"],
        "schema_hashes": protocol_body["schema_hashes"],
        "source_identity": dict(sorted(source_identity.items())),
        "artifact_hashes": artifacts,
        "stage_counts": {
            "h1": int(len(h1_opportunities)),
            "forward": int(len(forward_opportunities)),
        },
        "reconciliation": reconciliation,
        "maximum_signal_time": pd.to_datetime(
            forward_opportunities["signal_bar_open"], utc=True
        ).max().isoformat(),
        "q2_loaded": False,
    }
    _atomic_json(manifest, common / "manifest.json")
    return manifest


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _verified_common(root: Path) -> dict[str, Any]:
    path = root / "common" / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError("USA500 agent common artifacts are not prepared")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("implementation_hash") != _implementation_hash():
        raise ValueError("USA500 agent common implementation identity changed")
    if manifest.get("q2_loaded") is not False:
        raise PermissionError("USA500 agent common artifacts opened Q2")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        artifact = root / "common" / str(relative)
        if not artifact.is_file() or _sha256_file(artifact) != expected:
            raise ValueError(f"USA500 agent common artifact hash changed: {relative}")
    protocol = json.loads((root / "common" / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("protocol_hash") != manifest.get("protocol_hash"):
        raise ValueError("USA500 agent protocol and common manifest differ")
    if protocol.get("q2_loaded") is not False:
        raise PermissionError("USA500 agent protocol opened Q2")
    return manifest


def _preflight_direct_payload() -> dict[str, Any]:
    probabilities = [
        {"model_index": index, "short": 0.2, "flat": 0.1, "long": 0.7}
        for index in range(9)
    ]
    return {
        "schema_version": "1.0",
        "market_state": {
            "vix_regime": 0.0,
            "trailing_volatility": 0.0,
            "trailing_trend": 0.0,
        },
        "opportunities": [
            {
                "opportunity_index": index,
                "original_side": "LONG",
                "model_probabilities": probabilities,
                "agreement": 1.0,
                "probability_dispersion": 0.0,
            }
            for index in range(10)
        ],
        "model_evidence": [],
        "memory_cards": [],
    }


def _empty_model_evidence() -> list[dict[str, Any]]:
    empty = {
        "weeks": 0,
        "sample_count": 0,
        "directional_accuracy": 0.0,
        "net_return": 0.0,
        "brier_score": 0.0,
        "mean_confidence": 0.0,
    }
    return [
        {
            "evidence_index": index,
            "model_index": index,
            "rolling_1": dict(empty),
            "rolling_4": dict(empty),
            "rolling_12": dict(empty),
        }
        for index in range(9)
    ]


def _preflight_weekly_payload() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "market_state": {
            "vix_regime": 0.0,
            "trailing_volatility": 0.0,
            "trailing_trend": 0.0,
        },
        "model_evidence": _empty_model_evidence(),
        "memory_cards": [],
        "previous_weights": list(equal_weights()),
    }


def run_preflight(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    model_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify exact model identity and both strict schemas before agent arms."""
    root = Path(output_root).resolve()
    common = _verified_common(root)
    config = load_index_agent_config(config_path)
    record = model_record or _live_model_record(config.model)
    if record.get("model") != config.model:
        raise RuntimeError("preflight resolved a different Ollama model tag")
    if str(record.get("digest", "")) != config.required_model_digest:
        raise RuntimeError("preflight Ollama model digest changed")
    if "thinking" not in set(record.get("capabilities", [])):
        raise RuntimeError("preflight model does not advertise thinking support")
    active = caller or IndexSchemaCaller(
        config, call_log_path=root / "preflight_calls.jsonl"
    )
    direct_allowed = {
        "opportunity_indices": list(range(10)),
        "evidence_indices": [],
        "memory_indices": [],
    }
    direct = active.call(
        role="direct_preflight",
        messages=direct_messages(_preflight_direct_payload()),
        response_model=DirectBatchDecision,
        allowed_ids=direct_allowed,
    )
    if direct.value is None:
        raise RuntimeError(f"Direct strict-schema preflight failed: {direct.errors}")
    validate_direct_batch(
        direct.value,
        allowed_opportunity_indices=direct_allowed["opportunity_indices"],
        allowed_evidence_indices=[],
        allowed_memory_indices=[],
    )
    weekly_allowed = {
        "evidence_indices": list(range(9)),
        "memory_indices": [],
    }
    weekly = active.call(
        role="weekly_preflight",
        messages=weekly_weight_messages(_preflight_weekly_payload()),
        response_model=WeeklyWeightDecision,
        allowed_ids=weekly_allowed,
    )
    if weekly.value is None:
        raise RuntimeError(f"Weekly strict-schema preflight failed: {weekly.errors}")
    validate_weekly_weights(
        weekly.value,
        allowed_evidence_indices=weekly_allowed["evidence_indices"],
        allowed_memory_indices=[],
    )
    result = {
        "passed": True,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "model": config.model,
        "model_digest": config.required_model_digest,
        "ollama_version": str(record.get("ollama_version", "")),
        "capabilities": sorted(record.get("capabilities", [])),
        "direct_status": direct.status,
        "weekly_status": weekly.status,
        "direct_request_hash": direct.request_hash,
        "weekly_request_hash": weekly.request_hash,
        "q2_loaded": False,
    }
    _atomic_json(result, root / "preflight.json")
    return result


def _memory_mode(variant: str) -> str:
    if variant.endswith("real_memory"):
        return "real"
    if variant.endswith("no_memory"):
        return "no_memory"
    if variant.endswith("shuffled_memory"):
        return "shuffled"
    return "control"


def _prompt_memory(
    history: Sequence[dict[str, Any]],
    *,
    week_start: pd.Timestamp,
    mode: str,
    seed: int,
    max_cards: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = eligible_memory(history, week_start)
    if mode == "no_memory":
        visible: list[dict[str, Any]] = []
    elif mode == "shuffled":
        visible = shuffled_memory(eligible, seed=seed)
    else:
        visible = eligible
    evidence = memory_statistics(visible)
    evidence_rows = [
        {"evidence_index": index, **row} for index, row in enumerate(evidence)
    ]
    cards = []
    for index, card in enumerate(visible[-max_cards:]):
        cards.append(
            {
                "memory_index": index,
                **{
                    key: value
                    for key, value in card.items()
                    if key not in {"week_id", "week_start", "available_at"}
                },
            }
        )
    return evidence_rows, cards


def _memory_audit_fields(
    history: Sequence[dict[str, Any]],
    *,
    week_start: pd.Timestamp,
    mode: str,
    evidence: Sequence[dict[str, Any]],
    cards: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    eligible = eligible_memory(history, week_start)
    visible = [] if mode == "no_memory" else eligible[-len(cards) :] if cards else []
    maximum = max((_utc(card["available_at"]) for card in visible), default=None)
    if maximum is not None and maximum >= week_start:
        raise AssertionError("same-week outcome entered an agent prompt")
    return {
        "memory_cards_visible": int(len(cards)),
        "memory_max_available_at": maximum.isoformat() if maximum is not None else "",
        "memory_snapshot_hash": _hash(
            {"model_evidence": list(evidence), "memory_cards": list(cards)}
        ),
    }


def _week_state_lookup(root: Path) -> dict[pd.Timestamp, dict[str, float]]:
    frame = pd.read_parquet(root / "common" / "forward_weekly_state.parquet")
    frame["week_start"] = pd.to_datetime(frame["week_start"], utc=True)
    frame["state_available_at"] = pd.to_datetime(frame["state_available_at"], utc=True)
    output = {}
    for row in frame.itertuples(index=False):
        week = pd.Timestamp(row.week_start)
        if pd.Timestamp(row.state_available_at) >= week:
            raise ValueError("weekly state is not strictly pre-commitment")
        output[week] = {
            "vix_regime": float(row.state_vix_regime),
            "trailing_volatility": float(row.state_trailing_vol),
            "trailing_trend": float(row.state_trailing_trend),
        }
    return output


def _opportunity_prompt_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for local_index, row in enumerate(frame.itertuples(index=False)):
        probabilities = []
        votes = []
        margins = []
        for model_index in range(9):
            short = float(getattr(row, f"m{model_index:02d}_p_short"))
            flat = float(getattr(row, f"m{model_index:02d}_p_flat"))
            long = float(getattr(row, f"m{model_index:02d}_p_long"))
            probabilities.append(
                {
                    "model_index": model_index,
                    "short": round(short, 8),
                    "flat": round(flat, 8),
                    "long": round(long, 8),
                }
            )
            votes.append(1 if long > short else -1 if long < short else int(row.original_side))
            margins.append(long - short)
        long_share = float(np.mean(np.asarray(votes) == 1))
        rows.append(
            {
                "opportunity_index": local_index,
                "original_side": "LONG" if int(row.original_side) == 1 else "SHORT",
                "model_probabilities": probabilities,
                "agreement": round(max(long_share, 1.0 - long_share), 6),
                "probability_dispersion": round(float(np.std(margins)), 8),
                "current_market_state": {
                    "vix_regime": round(float(row.state_vix_regime), 8),
                    "trailing_volatility": round(float(row.state_trailing_vol), 8),
                    "trailing_trend": round(float(row.state_trailing_trend), 8),
                },
            }
        )
    return rows


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
        for key, value in identity.items():
            if current.get(key) != value:
                raise ValueError(f"variant resume identity changed: {key}")
    else:
        if (variant_root / "decisions.parquet").exists():
            raise ValueError("variant decisions exist without a bound checkpoint")
        _atomic_json({**identity, "status": "running"}, path)
    return path


def _existing_frame(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.is_file() else pd.DataFrame()


def _validate_decision_prefix(decisions: pd.DataFrame, opportunities: pd.DataFrame) -> None:
    if decisions.empty:
        return
    if decisions["opportunity_id"].duplicated().any():
        raise ValueError("checkpoint contains duplicate opportunity decisions")
    expected = opportunities["opportunity_id"].astype(str).tolist()[: len(decisions)]
    if decisions["opportunity_id"].astype(str).tolist() != expected:
        raise ValueError("checkpoint decisions are not an exact chronological prefix")


def _append_agent_result(
    card: dict[str, Any], current: pd.DataFrame, sides: Sequence[int], *, cost_bps: float,
    weights: Sequence[float] | None = None,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(card))
    ledger, _ = replay_registered_sides(current, sides, cost_bps=cost_bps)
    updated["prior_agent"] = {
        "net_return": float(ledger["net_return"].sum()),
        "long_trades": int(ledger["side"].eq(1).sum()),
        "short_trades": int(ledger["side"].eq(-1).sum()),
        "direction_changes": int(
            np.sum(np.asarray(sides, dtype=int) != current["original_side"].to_numpy(int))
        ),
    }
    if weights is not None:
        updated["prior_weights"] = [float(value) for value in weights]
    return updated


def _direct_variant(
    variant: str,
    *,
    root: Path,
    config: IndexAgentConfig,
    caller: Any,
    opportunities: pd.DataFrame,
    max_batches: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    variant_root = root / variant
    decisions_path = variant_root / "decisions.parquet"
    existing = _existing_frame(decisions_path)
    if not existing.empty:
        existing = existing.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    _validate_decision_prefix(existing, opportunities)
    records = existing.to_dict(orient="records")
    existing_ids = set(existing.get("opportunity_id", pd.Series(dtype=str)).astype(str))
    h1_cards = _read_jsonl(root / "common" / "h1_memory_cards.jsonl")
    forward_cards = _read_jsonl(root / "common" / "forward_base_memory_cards.jsonl")
    cards_by_start = {_utc(card["week_start"]): card for card in forward_cards}
    state_by_start = _week_state_lookup(root)
    history = list(h1_cards)
    mode = _memory_mode(variant)
    new_batches = 0

    for week_start, group in opportunities.groupby("week_start", sort=True):
        week = _utc(pd.Timestamp(week_start))
        current = group.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
        evidence, cards = _prompt_memory(
            history,
            week_start=week,
            mode=mode,
            seed=config.seed,
            max_cards=config.max_memory_cards,
        )
        for offset in range(0, len(current), config.direct_batch_size):
            batch = current.iloc[offset : offset + config.direct_batch_size].reset_index(drop=True)
            ids = batch["opportunity_id"].astype(str).tolist()
            already = [item in existing_ids for item in ids]
            if any(already) and not all(already):
                raise ValueError("Direct checkpoint split one immutable request batch")
            if all(already):
                continue
            allowed = {
                "opportunity_indices": list(range(len(batch))),
                "evidence_indices": list(range(len(evidence))),
                "memory_indices": list(range(len(cards))),
            }
            payload = {
                "schema_version": "1.0",
                "market_state": state_by_start[week],
                "opportunities": _opportunity_prompt_rows(batch),
                "model_evidence": evidence,
                "memory_cards": cards,
            }
            memory_audit = _memory_audit_fields(
                history,
                week_start=week,
                mode=mode,
                evidence=evidence,
                cards=cards,
            )
            nonmemory_hash = _hash(
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {"model_evidence", "memory_cards"}
                }
            )
            result = caller.call(
                role="direct",
                messages=direct_messages(payload),
                response_model=DirectBatchDecision,
                allowed_ids=allowed,
            )
            status = str(result.status)
            sides = batch["original_side"].to_numpy(int)
            references = [([], []) for _ in range(len(batch))]
            if result.value is not None:
                try:
                    sides = np.asarray(
                        validate_direct_batch(
                            result.value,
                            allowed_opportunity_indices=allowed["opportunity_indices"],
                            allowed_evidence_indices=allowed["evidence_indices"],
                            allowed_memory_indices=allowed["memory_indices"],
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
                except ValueError:
                    status = "invalid_output"
            call_id = f"{variant}:D:{len(records):06d}"
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
                        "side": int(sides[row_number]),
                        "original_side": int(row["original_side"]),
                        "call_id": call_id,
                        "call_status": status,
                        "request_hash": str(result.request_hash),
                        "response_hash": str(result.response_hash),
                        "evidence_indices": _canonical_json(evidence_refs),
                        "memory_indices": _canonical_json(memory_refs),
                        "nonmemory_hash": nonmemory_hash,
                        **memory_audit,
                    }
                )
                existing_ids.add(str(row["opportunity_id"]))
            decisions = pd.DataFrame(records).sort_values(
                "signal_bar_open", kind="mergesort"
            ).reset_index(drop=True)
            _atomic_parquet(decisions, decisions_path)
            new_batches += 1
            if max_batches is not None and new_batches >= max_batches and len(decisions) < len(opportunities):
                return decisions, pd.DataFrame(), False
        week_rows = pd.DataFrame(records)
        week_rows["week_start"] = pd.to_datetime(week_rows["week_start"], utc=True)
        chosen = week_rows.loc[week_rows["week_start"].eq(week)].sort_values(
            "signal_bar_open", kind="mergesort"
        )
        if len(chosen) != len(current):
            return pd.DataFrame(records), pd.DataFrame(), False
        history.append(
            _append_agent_result(
                cards_by_start[week],
                current,
                chosen["side"].to_numpy(int),
                cost_bps=config.round_trip_cost_bps,
            )
        )
    return pd.DataFrame(records), pd.DataFrame(), True


def _weekly_variant(
    variant: str,
    *,
    root: Path,
    config: IndexAgentConfig,
    caller: Any,
    opportunities: pd.DataFrame,
    max_batches: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    variant_root = root / variant
    decisions_path = variant_root / "decisions.parquet"
    weights_path = variant_root / "weekly_weights.parquet"
    decisions = _existing_frame(decisions_path)
    weights_frame = _existing_frame(weights_path)
    if not decisions.empty:
        decisions = decisions.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    _validate_decision_prefix(decisions, opportunities)
    decision_records = decisions.to_dict(orient="records")
    weight_records = weights_frame.to_dict(orient="records")
    weights_by_week = {
        str(row["week_id"]): row for row in weight_records
    }
    h1_cards = _read_jsonl(root / "common" / "h1_memory_cards.jsonl")
    forward_cards = _read_jsonl(root / "common" / "forward_base_memory_cards.jsonl")
    cards_by_start = {_utc(card["week_start"]): card for card in forward_cards}
    state_by_start = _week_state_lookup(root)
    history = list(h1_cards)
    previous = equal_weights()
    mode = _memory_mode(variant)
    new_batches = 0

    for week_start, group in opportunities.groupby("week_start", sort=True):
        week = _utc(pd.Timestamp(week_start))
        current = group.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
        card = cards_by_start[week]
        if card["week_id"] in weights_by_week:
            record = weights_by_week[card["week_id"]]
            selected_weights = tuple(float(record[f"w{index:02d}"]) for index in range(9))
        else:
            evidence, cards = _prompt_memory(
                history,
                week_start=week,
                mode=mode,
                seed=config.seed,
                max_cards=config.max_memory_cards,
            )
            if not evidence:
                evidence = _empty_model_evidence()
            allowed = {
                "evidence_indices": list(range(len(evidence))),
                "memory_indices": list(range(len(cards))),
            }
            payload = {
                "schema_version": "1.0",
                "market_state": state_by_start[week],
                "model_evidence": evidence,
                "memory_cards": cards,
                "previous_weights": [float(value) for value in previous],
            }
            memory_audit = _memory_audit_fields(
                history,
                week_start=week,
                mode=mode,
                evidence=evidence,
                cards=cards,
            )
            nonmemory_hash = _hash(
                {
                    "schema_version": payload["schema_version"],
                    "market_state": payload["market_state"],
                }
            )
            result = caller.call(
                role="weekly_weights",
                messages=weekly_weight_messages(payload),
                response_model=WeeklyWeightDecision,
                allowed_ids=allowed,
            )
            status = str(result.status)
            selected_weights = equal_weights()
            evidence_refs: list[int] = []
            memory_refs: list[int] = []
            if result.value is not None:
                try:
                    selected_weights = validate_weekly_weights(
                        result.value,
                        allowed_evidence_indices=allowed["evidence_indices"],
                        allowed_memory_indices=allowed["memory_indices"],
                    )
                    evidence_refs = list(result.value.evidence_indices)
                    memory_refs = list(result.value.memory_indices)
                except ValueError:
                    status = "invalid_output"
            record = {
                "week_id": card["week_id"],
                "week_start": week,
                "call_id": f"{variant}:W:{len(weight_records):04d}",
                "call_status": status,
                "request_hash": str(result.request_hash),
                "response_hash": str(result.response_hash),
                "evidence_indices": _canonical_json(evidence_refs),
                "memory_indices": _canonical_json(memory_refs),
                "nonmemory_hash": nonmemory_hash,
                **memory_audit,
                **{
                    f"w{index:02d}": float(value)
                    for index, value in enumerate(selected_weights)
                },
            }
            weight_records.append(record)
            weights_by_week[card["week_id"]] = record
            weights_frame = pd.DataFrame(weight_records).sort_values(
                "week_start", kind="mergesort"
            ).reset_index(drop=True)
            _atomic_parquet(weights_frame, weights_path)
            new_batches += 1
        existing_week = {
            str(item["opportunity_id"])
            for item in decision_records
            if _utc(item["week_start"]) == week
        }
        if existing_week and existing_week != set(current["opportunity_id"].astype(str)):
            raise ValueError("Weekly checkpoint contains an incomplete committed week")
        if not existing_week:
            selected_sides = weighted_sides(current, selected_weights)
            for offset, (_, row) in enumerate(current.iterrows()):
                decision_records.append(
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
                        "side": int(selected_sides[offset]),
                        "original_side": int(row["original_side"]),
                        "call_id": str(record["call_id"]),
                        "call_status": str(record["call_status"]),
                        "request_hash": str(record["request_hash"]),
                        "response_hash": str(record["response_hash"]),
                        "evidence_indices": str(record["evidence_indices"]),
                        "memory_indices": str(record["memory_indices"]),
                        "nonmemory_hash": str(record["nonmemory_hash"]),
                        "memory_cards_visible": int(record["memory_cards_visible"]),
                        "memory_max_available_at": str(record["memory_max_available_at"]),
                        "memory_snapshot_hash": str(record["memory_snapshot_hash"]),
                    }
                )
            decisions = pd.DataFrame(decision_records).sort_values(
                "signal_bar_open", kind="mergesort"
            ).reset_index(drop=True)
            _atomic_parquet(decisions, decisions_path)
        else:
            decisions = pd.DataFrame(decision_records)
            selected_sides = decisions.loc[
                pd.to_datetime(decisions["week_start"], utc=True).eq(week), "side"
            ].to_numpy(int)
        history.append(
            _append_agent_result(
                card,
                current,
                selected_sides,
                cost_bps=config.round_trip_cost_bps,
                weights=selected_weights,
            )
        )
        previous = selected_weights
        if max_batches is not None and new_batches >= max_batches and len(decision_records) < len(opportunities):
            return pd.DataFrame(decision_records), pd.DataFrame(weight_records), False
    return pd.DataFrame(decision_records), pd.DataFrame(weight_records), True


def _control_variant(
    variant: str,
    *,
    root: Path,
    config: IndexAgentConfig,
    opportunities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    controls = json.loads((root / "common" / "control_weights.json").read_text(encoding="utf-8"))
    cards = _read_jsonl(root / "common" / "forward_base_memory_cards.jsonl")
    cards_by_start = {_utc(card["week_start"]): card for card in cards}
    decision_records = []
    weight_records = []
    for week_start, group in opportunities.groupby("week_start", sort=True):
        week = _utc(pd.Timestamp(week_start))
        current = group.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
        card = cards_by_start[week]
        if variant == "original_frozen":
            sides = current["original_side"].to_numpy(int)
            weights: Sequence[float] | None = None
        elif variant == "static_h1":
            weights = controls["static_h1_weights"]
            sides = weighted_sides(current, weights)
        elif variant == "hedge_weekly":
            weights = controls["hedge_schedule"][card["week_id"]]
            sides = weighted_sides(current, weights)
        else:
            raise ValueError(f"unknown control variant: {variant}")
        call_id = f"{variant}:control:{card['week_id']}"
        for offset, (_, row) in enumerate(current.iterrows()):
            decision_records.append(
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
                    "side": int(sides[offset]),
                    "original_side": int(row["original_side"]),
                    "call_id": call_id,
                    "call_status": "not_called",
                    "request_hash": "",
                    "response_hash": "",
                    "evidence_indices": "[]",
                    "memory_indices": "[]",
                    "nonmemory_hash": "",
                    "memory_cards_visible": 0,
                    "memory_max_available_at": "",
                    "memory_snapshot_hash": "",
                }
            )
        if weights is not None:
            weight_records.append(
                {
                    "week_id": card["week_id"],
                    "week_start": week,
                    "call_id": call_id,
                    "call_status": "not_called",
                    "request_hash": "",
                    "response_hash": "",
                    "evidence_indices": "[]",
                    "memory_indices": "[]",
                    **{
                        f"w{index:02d}": float(value)
                        for index, value in enumerate(weights)
                    },
                }
            )
    return pd.DataFrame(decision_records), pd.DataFrame(weight_records), True


def _weight_turnover(weights: pd.DataFrame) -> float:
    if len(weights) <= 1:
        return 0.0
    columns = [f"w{index:02d}" for index in range(9)]
    values = weights.sort_values("week_start", kind="mergesort")[columns].to_numpy(float)
    return float(np.abs(np.diff(values, axis=0)).sum(axis=1).mean())


def _finish_variant(
    variant: str,
    *,
    root: Path,
    common: dict[str, Any],
    config: IndexAgentConfig,
    opportunities: pd.DataFrame,
    decisions: pd.DataFrame,
    weights: pd.DataFrame,
) -> dict[str, Any]:
    decisions = decisions.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    if len(decisions) != len(opportunities) or not decisions["opportunity_id"].is_unique:
        raise AssertionError("variant did not decide every registered opportunity exactly once")
    if not opportunity_keys(decisions).equals(opportunity_keys(opportunities)):
        raise AssertionError("variant changed immutable opportunity keys")
    ledger, per_bar = replay_registered_sides(
        opportunities,
        decisions["side"].to_numpy(int),
        cost_bps=config.round_trip_cost_bps,
    )
    metrics = daily_economics(
        ledger,
        per_bar,
        start=config.forward_start_utc,
        end=config.forward_end_utc,
    )
    call_rows = decisions.loc[decisions["call_status"].ne("not_called")].drop_duplicates(
        "call_id"
    )
    calls = int(len(call_rows))
    failures = int((~call_rows["call_status"].isin({"success", "repaired"})).sum())
    summary = {
        "status": "complete",
        "variant": variant,
        "memory_mode": _memory_mode(variant),
        **metrics,
        "win_rate": float(ledger["net_return"].gt(0.0).mean()),
        "direction_changes": int(
            np.sum(decisions["side"].to_numpy(int) != decisions["original_side"].to_numpy(int))
        ),
        "weight_turnover": _weight_turnover(weights),
        "transport_calls": calls,
        "transport_failures": failures,
        "transport_failure_fraction": float(failures / calls) if calls else 0.0,
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "q2_loaded": False,
    }
    variant_root = root / variant
    _atomic_parquet(decisions, variant_root / "decisions.parquet")
    if not weights.empty:
        _atomic_parquet(weights, variant_root / "weekly_weights.parquet")
    _atomic_parquet(ledger, variant_root / "ledger.parquet")
    _atomic_parquet(per_bar.rename_axis("timestamp").reset_index(), variant_root / "per_bar.parquet")
    _atomic_json(summary, variant_root / "summary.json")
    artifact_names = ["decisions.parquet", "ledger.parquet", "per_bar.parquet", "summary.json"]
    if not weights.empty:
        artifact_names.append("weekly_weights.parquet")
    manifest = {
        **_checkpoint_identity(root, common, variant),
        "status": "complete",
        "artifact_hashes": {
            name: _sha256_file(variant_root / name) for name in artifact_names
        },
    }
    _atomic_json(manifest, variant_root / "manifest.json")
    _atomic_json(
        {**_checkpoint_identity(root, common, variant), "status": "complete"},
        variant_root / "checkpoint.json",
    )
    return summary


def run_variant(
    variant: str,
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    """Run or resume one registered arm on the exact frozen opportunity set."""
    if variant not in REGISTERED_VARIANTS:
        raise ValueError(f"unknown USA500 agent variant: {variant}")
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    root = Path(output_root).resolve()
    config = load_index_agent_config(config_path)
    common = _verified_common(root)
    _validate_or_create_checkpoint(root, common, variant)
    opportunities = pd.read_parquet(root / "common" / "forward_opportunities.parquet")
    opportunities["week_start"] = pd.to_datetime(opportunities["week_start"], utc=True)
    if variant.startswith(("direct_", "weekly_")):
        preflight_path = root / "preflight.json"
        if not preflight_path.is_file():
            raise ValueError("agent arm requires a passed exact-model preflight")
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        if preflight.get("passed") is not True or preflight.get("protocol_hash") != common["protocol_hash"]:
            raise ValueError("agent arm preflight identity changed")
        if caller is None:
            inner = IndexSchemaCaller(
                config, call_log_path=root / variant / "transport_calls.jsonl"
            )
            caller = CachedIndexSchemaCaller(
                inner,
                root / variant / "call_cache",
                protocol_hash=common["protocol_hash"],
            )
    if variant.startswith("direct_"):
        decisions, weights, complete = _direct_variant(
            variant,
            root=root,
            config=config,
            caller=caller,
            opportunities=opportunities,
            max_batches=max_batches,
        )
    elif variant.startswith("weekly_"):
        decisions, weights, complete = _weekly_variant(
            variant,
            root=root,
            config=config,
            caller=caller,
            opportunities=opportunities,
            max_batches=max_batches,
        )
    else:
        decisions, weights, complete = _control_variant(
            variant, root=root, config=config, opportunities=opportunities
        )
    if not complete:
        partial = {
            "status": "partial",
            "variant": variant,
            "completed_decisions": int(len(decisions)),
            "total_decisions": int(len(opportunities)),
            "q2_loaded": False,
        }
        _atomic_json(
            {**_checkpoint_identity(root, common, variant), "status": "partial"},
            root / variant / "checkpoint.json",
        )
        return partial
    return _finish_variant(
        variant,
        root=root,
        common=common,
        config=config,
        opportunities=opportunities,
        decisions=decisions,
        weights=weights,
    )


def _verified_variant(
    root: Path, variant: str, common: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    variant_root = root / variant
    manifest_path = variant_root / "manifest.json"
    summary_path = variant_root / "summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"registered arm is incomplete: {variant}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = _checkpoint_identity(root, common, variant)
    for key, expected in identity.items():
        if manifest.get(key) != expected:
            raise ValueError(f"registered arm identity changed: {variant}:{key}")
    if manifest.get("status") != "complete" or manifest.get("q2_loaded") is not False:
        raise ValueError(f"registered arm is not a sealed completion: {variant}")
    for relative, expected in manifest.get("artifact_hashes", {}).items():
        path = variant_root / str(relative)
        if not path.is_file() or _sha256_file(path) != expected:
            raise ValueError(f"registered arm artifact hash changed: {variant}:{relative}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("q2_loaded") is not False:
        raise ValueError(f"registered arm summary is not sealed: {variant}")
    return manifest, summary


def _results_table(summaries: Mapping[str, dict[str, Any]]) -> pd.DataFrame:
    columns = (
        "variant",
        "memory_mode",
        "calendar_days",
        "trades",
        "trades_per_day",
        "n_long",
        "n_short",
        "gross_return",
        "cost_return",
        "net_return",
        "net_bps_per_trade",
        "daily_sharpe",
        "daily_sortino",
        "max_drawdown",
        "win_rate",
        "direction_changes",
        "weight_turnover",
        "transport_calls",
        "transport_failures",
        "transport_failure_fraction",
    )
    frame = pd.DataFrame([{key: value for key, value in row.items() if key in columns} for row in summaries.values()])
    if set(frame["variant"]) != set(REGISTERED_VARIANTS):
        raise ValueError("results table does not contain the exact nine registered arms")
    numeric = [column for column in columns if column not in {"variant", "memory_mode"}]
    if not np.isfinite(frame[numeric].to_numpy(float)).all():
        raise ValueError("results table contains non-finite economics")
    return frame.sort_values(
        ["net_return", "daily_sortino", "variant"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _side_results(
    root: Path, config: IndexAgentConfig
) -> pd.DataFrame:
    rows = []
    for variant in REGISTERED_VARIANTS:
        ledger = pd.read_parquet(root / variant / "ledger.parquet")
        for side, label in ((1, "LONG"), (-1, "SHORT")):
            current = ledger.loc[ledger["side"].eq(side)].copy()
            per_bar = current.groupby("entry_bar_open")["net_return"].sum()
            per_bar.index = pd.to_datetime(per_bar.index, utc=True)
            metrics = daily_economics(
                current,
                per_bar,
                start=config.forward_start_utc,
                end=config.forward_end_utc,
            )
            rows.append(
                {
                    "variant": variant,
                    "side_label": label,
                    **metrics,
                    "win_rate": float(current["net_return"].gt(0.0).mean()) if len(current) else 0.0,
                }
            )
    frame = pd.DataFrame(rows)
    numeric = frame.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("side results contain non-finite values")
    return frame.sort_values(
        ["net_return", "daily_sortino", "variant", "side_label"],
        ascending=[False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _weekly_results(root: Path, config: IndexAgentConfig) -> pd.DataFrame:
    opportunities = pd.read_parquet(root / "common" / "forward_opportunities.parquet")
    week_map = opportunities.set_index("opportunity_id")["week_start"]
    rows = []
    for variant in REGISTERED_VARIANTS:
        ledger = pd.read_parquet(root / variant / "ledger.parquet")
        ledger["week_start"] = pd.to_datetime(
            ledger["opportunity_id"].map(week_map), utc=True
        )
        for week_start, current in ledger.groupby("week_start", sort=True):
            start = max(_utc(week_start), _utc(config.forward_start_utc))
            end = min(_utc(week_start) + pd.Timedelta(days=7), _utc(config.forward_end_utc))
            per_bar = current.groupby("entry_bar_open")["net_return"].sum()
            per_bar.index = pd.to_datetime(per_bar.index, utc=True)
            metrics = daily_economics(current, per_bar, start=start, end=end)
            rows.append(
                {
                    "variant": variant,
                    "week_start": _utc(week_start),
                    **metrics,
                    "win_rate": float(current["net_return"].gt(0.0).mean()),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty or not np.isfinite(frame.select_dtypes(include=[np.number]).to_numpy(float)).all():
        raise ValueError("weekly results are empty or non-finite")
    return frame.sort_values(["week_start", "variant"], kind="mergesort").reset_index(drop=True)


def _memory_ablation(results: pd.DataFrame) -> pd.DataFrame:
    indexed = results.set_index("variant")
    rows = []
    for family in ("direct", "weekly"):
        real_id = f"{family}_real_memory"
        real = indexed.loc[real_id]
        for mode, variant in (
            ("Real Memory", real_id),
            ("No Memory", f"{family}_no_memory"),
            ("Shuffled Memory", f"{family}_shuffled_memory"),
        ):
            current = indexed.loc[variant]
            rows.append(
                {
                    "family": family,
                    "memory": mode,
                    "variant": variant,
                    "trades": int(current["trades"]),
                    "n_long": int(current["n_long"]),
                    "n_short": int(current["n_short"]),
                    "net_return": float(current["net_return"]),
                    "daily_sharpe": float(current["daily_sharpe"]),
                    "daily_sortino": float(current["daily_sortino"]),
                    "max_drawdown": float(current["max_drawdown"]),
                    "net_delta_vs_real": float(current["net_return"] - real["net_return"]),
                    "sortino_delta_vs_real": float(
                        current["daily_sortino"] - real["daily_sortino"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["net_return", "daily_sortino", "variant"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def _paired_bootstrap(
    root: Path, opportunities: pd.DataFrame, *, samples: int, seed: int
) -> pd.DataFrame:
    original = pd.read_parquet(root / "original_frozen" / "ledger.parquet").set_index(
        "opportunity_id"
    )["net_return"]
    week = opportunities.set_index("opportunity_id")["week_start"]
    rows = []
    for variant_index, variant in enumerate(REGISTERED_VARIANTS):
        current = pd.read_parquet(root / variant / "ledger.parquet").set_index(
            "opportunity_id"
        )["net_return"]
        if not current.index.equals(original.index):
            current = current.reindex(original.index)
        if current.isna().any():
            raise ValueError("paired bootstrap opportunity alignment failed")
        block = current.sub(original).groupby(week.reindex(original.index)).sum()
        values = block.to_numpy(dtype=float)
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
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["net_delta", "variant"], ascending=[False, True], kind="mergesort"
    ).reset_index(drop=True)


def _call_audit_frame(root: Path, variant: str) -> pd.DataFrame:
    if variant.startswith("weekly_"):
        frame = pd.read_parquet(root / variant / "weekly_weights.parquet")
    else:
        frame = pd.read_parquet(root / variant / "decisions.parquet").drop_duplicates(
            "call_id"
        )
    return frame.sort_values("week_start", kind="mergesort").reset_index(drop=True)


def _leakage_audit(
    root: Path,
    common: dict[str, Any],
    opportunities: pd.DataFrame,
) -> pd.DataFrame:
    details: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, detail: str) -> None:
        details.append({"check_id": check_id, "passed": bool(passed), "detail": detail})

    q2_pass = common.get("q2_loaded") is False and pd.to_datetime(
        opportunities["signal_bar_open"], utc=True
    ).max() < pd.Timestamp("2026-04-01T00:00:00Z")
    exact_keys = True
    binary = True
    execution = True
    reference_keys = opportunity_keys(opportunities)
    for variant in REGISTERED_VARIANTS:
        decisions = pd.read_parquet(root / variant / "decisions.parquet")
        ledger = pd.read_parquet(root / variant / "ledger.parquet")
        exact_keys &= opportunity_keys(decisions).equals(reference_keys)
        exact_keys &= opportunity_keys(ledger).equals(reference_keys)
        binary &= bool(decisions["side"].isin((-1, 1)).all())
        replay, _ = replay_registered_sides(
            opportunities, decisions["side"].to_numpy(int), cost_bps=2.0
        )
        execution &= bool(
            np.allclose(
                replay["net_return"].to_numpy(float),
                ledger["net_return"].to_numpy(float),
                rtol=0.0,
                atol=1e-12,
            )
        )
        summary = json.loads((root / variant / "summary.json").read_text(encoding="utf-8"))
        q2_pass &= summary.get("q2_loaded") is False
    add("q2_sealed", q2_pass, "All source and arm maxima remain before 2026-04-01 UTC.")
    add("exact_opportunity_keys", exact_keys, "Every arm preserves all immutable opportunity keys.")
    add("binary_decisions", binary, "Every registered opportunity is LONG or SHORT.")
    add("execution_reconciled", execution, "Every ledger recomputes from immutable entry/exit prices and 2 bps cost.")

    same_week = True
    for variant in AGENT_VARIANTS:
        calls = _call_audit_frame(root, variant)
        for row in calls.itertuples(index=False):
            maximum = str(row.memory_max_available_at)
            if maximum:
                same_week &= _utc(maximum) < _utc(row.week_start)
    add("same_week_outcomes_absent", same_week, "Visible memory always resolves strictly before weekly commitment.")

    weekly_precedes = True
    for variant in ("weekly_real_memory", "weekly_no_memory", "weekly_shuffled_memory"):
        weights = pd.read_parquet(root / variant / "weekly_weights.parquet")
        decisions = pd.read_parquet(root / variant / "decisions.parquet")
        for row in weights.itertuples(index=False):
            first_signal = pd.to_datetime(
                decisions.loc[
                    pd.to_datetime(decisions["week_start"], utc=True).eq(_utc(row.week_start)),
                    "signal_bar_open",
                ],
                utc=True,
            ).min()
            weekly_precedes &= _utc(row.week_start) <= first_signal
    add("weekly_weights_precede_opportunities", weekly_precedes, "Each weekly vector is committed before its first opportunity.")

    nonmemory = True
    shuffled_timing = True
    for family in ("direct", "weekly"):
        variants = [
            f"{family}_real_memory",
            f"{family}_no_memory",
            f"{family}_shuffled_memory",
        ]
        call_frames = [_call_audit_frame(root, variant) for variant in variants]
        hashes = [frame["nonmemory_hash"].astype(str).tolist() for frame in call_frames]
        nonmemory &= hashes[0] == hashes[1] == hashes[2]
        real = call_frames[0]
        shuffled = call_frames[2]
        shuffled_timing &= real["memory_cards_visible"].astype(int).tolist() == shuffled[
            "memory_cards_visible"
        ].astype(int).tolist()
        shuffled_timing &= real["memory_max_available_at"].astype(str).tolist() == shuffled[
            "memory_max_available_at"
        ].astype(str).tolist()
    add("memory_ablation_nonmemory_parity", nonmemory, "Real, no-memory and shuffled arms share identical current inputs.")
    add("shuffled_memory_timing_parity", shuffled_timing, "Shuffling changes attribution but preserves card timing and count.")
    return pd.DataFrame(details)


def _candidate_gate(
    results: pd.DataFrame, audit: pd.DataFrame, config: IndexAgentConfig
) -> pd.DataFrame:
    indexed = results.set_index("variant")
    original = indexed.loc["original_frozen"]
    hedge = indexed.loc["hedge_weekly"]
    integrity = bool(audit["passed"].all())
    rows = []
    for family in ("direct", "weekly"):
        variant = f"{family}_real_memory"
        current = indexed.loc[variant]
        ablations = indexed.loc[
            [f"{family}_no_memory", f"{family}_shuffled_memory"]
        ]
        checks = {
            "integrity_pass": integrity,
            "minimum_side_trades_pass": bool(
                int(current["n_long"]) >= config.minimum_side_trades
                and int(current["n_short"]) >= config.minimum_side_trades
            ),
            "net_improvement_pass": bool(current["net_return"] > original["net_return"]),
            "sortino_improvement_pass": bool(
                current["daily_sortino"] > original["daily_sortino"]
            ),
            "drawdown_margin_pass": bool(
                current["max_drawdown"]
                >= original["max_drawdown"] - config.max_drawdown_absolute_margin
            ),
            "transport_pass": bool(
                current["transport_failure_fraction"]
                <= config.max_transport_failure_fraction
            ),
            "memory_controls_pass": bool(
                (current["net_return"] >= ablations["net_return"]).all()
                and (current["daily_sortino"] >= ablations["daily_sortino"]).all()
            ),
            "hedge_control_pass": bool(
                current["net_return"] >= hedge["net_return"]
                and current["daily_sortino"] >= hedge["daily_sortino"]
            ),
        }
        rows.append(
            {
                "variant": variant,
                **checks,
                "candidate_pass": bool(all(checks.values())),
            }
        )
    return pd.DataFrame(rows)


def finalize_experiment(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    """Validate all arms and publish paired finite comparison artifacts."""
    root = Path(output_root).resolve()
    config = load_index_agent_config(config_path)
    common = _verified_common(root)
    summaries = {}
    for variant in REGISTERED_VARIANTS:
        _manifest, summaries[variant] = _verified_variant(root, variant, common)
    opportunities = pd.read_parquet(root / "common" / "forward_opportunities.parquet")
    opportunities["week_start"] = pd.to_datetime(opportunities["week_start"], utc=True)
    results = _results_table(summaries)
    sides = _side_results(root, config)
    weekly = _weekly_results(root, config)
    ablation = _memory_ablation(results)
    bootstrap = _paired_bootstrap(
        root,
        opportunities,
        samples=config.bootstrap_samples,
        seed=config.seed,
    )
    audit = _leakage_audit(root, common, opportunities)
    gates = _candidate_gate(results, audit, config)
    frames = {
        "results_table.parquet": results,
        "side_results.parquet": sides,
        "weekly_results.parquet": weekly,
        "memory_ablation.parquet": ablation,
        "paired_bootstrap.parquet": bootstrap,
        "leakage_audit.parquet": audit,
        "candidate_gate.parquet": gates,
    }
    for name, frame in frames.items():
        if frame.empty:
            raise ValueError(f"final comparison artifact is empty: {name}")
        _atomic_parquet(frame, root / name)
    passing = gates.loc[gates["candidate_pass"], "variant"].astype(str).tolist()
    summary = {
        "status": "complete",
        "registered_arms": len(REGISTERED_VARIANTS),
        "best_variant_by_net": str(results.iloc[0]["variant"]),
        "best_net_return": float(results.iloc[0]["net_return"]),
        "candidate_variants": passing,
        "candidate_count": len(passing),
        "all_integrity_checks_pass": bool(audit["passed"].all()),
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "evidence_role": "exploratory_forward_overlay",
        "q2_loaded": False,
    }
    _atomic_json(summary, root / "summary.json")
    artifacts = {name: _sha256_file(root / name) for name in [*frames, "summary.json"]}
    manifest = {
        "status": "complete",
        "protocol_hash": common["protocol_hash"],
        "implementation_hash": common["implementation_hash"],
        "common_manifest_hash": _sha256_file(root / "common" / "manifest.json"),
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
    action.add_argument("--variant", choices=REGISTERED_VARIANTS)
    action.add_argument("--run-controls", action="store_true")
    action.add_argument("--run-agents", action="store_true")
    action.add_argument("--finalize", action="store_true")
    action.add_argument("--all", action="store_true")
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
    elif args.run_agents:
        result = {
            variant: run_variant(variant, output_root=args.cache_dir)
            for variant in AGENT_VARIANTS
        }
    elif args.finalize:
        result = finalize_experiment(output_root=args.cache_dir)
    else:
        result = {"prepare": prepare_common_artifacts(output_root=args.cache_dir)}
        result["preflight"] = run_preflight(output_root=args.cache_dir)
        result["controls"] = {
            variant: run_variant(variant, output_root=args.cache_dir)
            for variant in CONTROL_VARIANTS
        }
        result["agents"] = {
            variant: run_variant(variant, output_root=args.cache_dir)
            for variant in AGENT_VARIANTS
        }
        result["finalize"] = finalize_experiment(output_root=args.cache_dir)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AGENT_VARIANTS",
    "CACHE",
    "CONTROL_VARIANTS",
    "finalize_experiment",
    "main",
    "prepare_common_artifacts",
    "run_preflight",
    "run_variant",
]
