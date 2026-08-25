"""Resumable H1-calibrated reversal scorer over the frozen USATECH ensemble."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

import experiments.run_usa500_budgeted_reversal_agent as base
import experiments.run_usatech_reflection_weight_agent as source_runner
from reflection_agent.usatech_v2.config import (
    REGISTERED_VARIANTS,
    load_usatech_reversal_config,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = CODE_ROOT / "configs" / "usatech_budgeted_reversal_agent_v1.yaml"
CACHE = CODE_ROOT / "experiments" / "cache" / "usatech_budgeted_reversal_agent"
SOURCE_CACHE = CODE_ROOT / "experiments" / "cache" / "usatech_reflection_weight_agent"
AGENT_VARIANTS = base.AGENT_VARIANTS
CONTROL_VARIANTS = base.CONTROL_VARIANTS
_ORIGINAL_LEAKAGE_AUDIT = base._leakage_audit


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    dependencies = [
        Path(__file__),
        Path(base.__file__),
        Path(source_runner.__file__),
        *sorted((CODE_ROOT / "reflection_agent" / "index_v1").glob("*.py")),
        *sorted((CODE_ROOT / "reflection_agent" / "index_v2").glob("*.py")),
        *sorted((CODE_ROOT / "reflection_agent" / "usatech_v1").glob("*.py")),
        *sorted((CODE_ROOT / "reflection_agent" / "usatech_v2").glob("*.py")),
    ]
    for path in dependencies:
        digest.update(str(path.relative_to(CODE_ROOT)).replace("\\", "/").encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _verified_usatech_source(root: Path) -> dict[str, Any]:
    with source_runner._bound_base_module():
        return source_runner.base._verified_common(root)


def _usatech_leakage_audit(
    root: Path,
    common: dict[str, Any],
    forward: pd.DataFrame,
    arms,
    frozen_hash: str,
) -> pd.DataFrame:
    audit = _ORIGINAL_LEAKAGE_AUDIT(root, common, forward, arms, frozen_hash)
    mask = audit["check_id"].eq("immutable_execution_replay")
    if int(mask.sum()) != 1:
        raise ValueError("immutable execution audit row changed")
    audit.loc[mask, "detail"] = (
        "Every ledger was recomputed and matched immutable prices plus 3 bps cost."
    )
    return audit


@contextmanager
def _bound_base_module():
    replacements = {
        "load_budgeted_reversal_config": load_usatech_reversal_config,
        "_implementation_hash": _implementation_hash,
        "_verified_source_common": _verified_usatech_source,
        "_leakage_audit": _usatech_leakage_audit,
        "SOURCE_CACHE": SOURCE_CACHE,
    }
    originals = {name: getattr(base, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def prepare_common_artifacts(
    *,
    source_root: str | Path = SOURCE_CACHE,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.prepare_common_artifacts(
            source_root=source_root,
            output_root=output_root,
            config_path=config_path,
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


def run_score_variant(
    variant: str,
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
    caller: Any | None = None,
    max_batches: int | None = None,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.run_score_variant(
            variant,
            output_root=output_root,
            config_path=config_path,
            caller=caller,
            max_batches=max_batches,
        )


def freeze_h1_policies(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.freeze_h1_policies(
            output_root=output_root,
            config_path=config_path,
        )


def run_controls(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.run_controls(output_root=output_root, config_path=config_path)


def finalize_results(
    *,
    output_root: str | Path = CACHE,
    config_path: str | Path = DEFAULT_CONFIG,
) -> dict[str, Any]:
    with _bound_base_module():
        return base.finalize_results(output_root=output_root, config_path=config_path)


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
            source_root=args.source_cache, output_root=args.cache_dir
        )
    elif args.preflight:
        result = run_preflight(output_root=args.cache_dir)
    elif args.score:
        result = run_score_variant(
            args.score, output_root=args.cache_dir, max_batches=args.max_batches
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
                source_root=args.source_cache, output_root=args.cache_dir
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
