"""Resumable USATECH Direct and weekly reflection-weight replication."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from experiments.index_all_model_ensemble import (
    AlignedPanel,
    IndexAllModelEnsembleConfig,
    IndexAllModelEnsembleRunner,
    validate_source_contract,
)
from experiments.index_replication import _frame_hash
import experiments.run_usa500_reflection_weight_agent as base
from reflection_agent.usatech_v1.config import (
    REGISTERED_VARIANTS,
    USATechAgentConfig,
    load_usatech_agent_config,
)
from reflection_agent.usatech_v1.engine import build_soft_vote_opportunities


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "usatech_reflection_weight_agent_v1.yaml"
CACHE = CODE_ROOT / "experiments" / "cache" / "usatech_reflection_weight_agent"
PARENT_CACHE = CODE_ROOT / "experiments" / "cache" / "index_all_model_ensemble" / "usatech"
CONTROL_VARIANTS = base.CONTROL_VARIANTS
AGENT_VARIANTS = base.AGENT_VARIANTS
_ORIGINAL_LEAKAGE_AUDIT = base._leakage_audit


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    dependencies = [
        Path(__file__),
        Path(base.__file__),
        *sorted((CODE_ROOT / "reflection_agent" / "index_v1").glob("*.py")),
        *sorted((CODE_ROOT / "reflection_agent" / "usatech_v1").glob("*.py")),
    ]
    for path in dependencies:
        digest.update(str(path.relative_to(CODE_ROOT)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _usatech_leakage_audit(
    root: Path, common: dict[str, Any], opportunities: pd.DataFrame
) -> pd.DataFrame:
    audit = _ORIGINAL_LEAKAGE_AUDIT(root, common, opportunities)
    execution = True
    for variant in REGISTERED_VARIANTS:
        decisions = pd.read_parquet(root / variant / "decisions.parquet")
        ledger = pd.read_parquet(root / variant / "ledger.parquet")
        replay, _ = base.replay_registered_sides(
            opportunities,
            decisions["side"].to_numpy(dtype=int),
            cost_bps=3.0,
        )
        execution &= bool(
            np.allclose(
                replay["net_return"].to_numpy(dtype=float),
                ledger["net_return"].to_numpy(dtype=float),
                rtol=0.0,
                atol=1e-12,
            )
        )
    mask = audit["check_id"].eq("execution_reconciled")
    if int(mask.sum()) != 1:
        raise ValueError("execution reconciliation audit row changed")
    audit.loc[mask, "passed"] = execution
    audit.loc[mask, "detail"] = (
        "Every ledger recomputes from immutable entry/exit prices and 3 bps cost."
    )
    return audit


@contextmanager
def _bound_base_module():
    replacements = {
        "load_index_agent_config": load_usatech_agent_config,
        "build_registered_opportunities": build_soft_vote_opportunities,
        "_implementation_hash": _implementation_hash,
        "PARENT_CACHE": PARENT_CACHE,
        "_leakage_audit": _usatech_leakage_audit,
    }
    originals = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def _parent_inputs(
    config: USATechAgentConfig,
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
        raise FileNotFoundError("frozen USATECH ensemble artifacts are incomplete")
    parent_protocol = json.loads(parent_protocol_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    if parent_protocol.get("stream") != "usatech":
        raise ValueError("parent ensemble stream changed")
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
            raise ValueError(f"frozen USATECH ensemble policy changed: {key}")
    if candidate.get("protocol_hash") != parent_protocol.get("protocol_hash"):
        raise ValueError("candidate and parent ensemble protocols differ")
    ledger_record = candidate.get("artifacts", {}).get("ledger", {})
    ledger_path = (PARENT_CACHE / str(ledger_record.get("path", ""))).resolve()
    if PARENT_CACHE.resolve() not in ledger_path.parents or not ledger_path.is_file():
        raise ValueError("candidate ledger path left the frozen USATECH cache")
    if base._sha256_file(ledger_path) != ledger_record.get("sha256"):
        raise ValueError("candidate ledger hash changed")

    ensemble_config = IndexAllModelEnsembleConfig.for_stream("usatech")
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
            "state_vix_regime": vix_level.sub(rolling_mean)
            .div(rolling_scale.replace(0.0, np.nan))
            .fillna(0.0),
            "state_trailing_vol": features["vol_20"].astype(float),
            "state_trailing_trend": features["r20"].astype(float),
        },
        index=features.index,
    )
    if state_frame.isna().any().any() or state_frame.index.max() >= config.q2_start_utc:
        raise ValueError("causal state frame is incomplete or crossed Q2")
    source_identity = {
        "parent_protocol_sha256": base._sha256_file(parent_protocol_path),
        "candidate_sha256": base._sha256_file(candidate_path),
        "candidate_ledger_sha256": base._sha256_file(ledger_path),
        "source_protocol_hash": str(source_contract["source_protocol_hash"]),
        "bars_frame_sha256": _frame_hash(source_runner.bars),
        "h1_panel_sha256": base._panel_hash(h1_panel),
        "forward_panel_sha256": base._panel_hash(forward_panel),
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
            source_identity,
            state_frame,
        ) = _parent_inputs(load_usatech_agent_config(config_path))
    with _bound_base_module():
        return base.prepare_common_artifacts(
            output_root=output_root,
            config_path=config_path,
            bars=bars,
            h1_panel=h1_panel,
            forward_panel=forward_panel,
            expected_forward_ledger=expected_forward_ledger,
            source_identity=source_identity,
            state_frame=state_frame,
        )


def run_preflight(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    model_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.run_preflight(
            output_root=output_root,
            config_path=config_path,
            caller=caller,
            model_record=model_record,
        )


def run_variant(
    variant: str,
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.run_variant(
            variant,
            output_root=output_root,
            config_path=config_path,
            caller=caller,
            max_batches=max_batches,
        )


def finalize_experiment(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.finalize_experiment(
            output_root=output_root,
            config_path=config_path,
        )


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
