"""Run the eight non-CatBoost models under the frozen one-minute protocol."""
from __future__ import annotations

import traceback
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.catboost_execution_resolution import (
    PartitionedIntrabarStore,
    write_run_state,
)
from experiments.catboost_execution_runner import ExecutionResolutionRunner
from experiments.matched_model_zoo_1m import (
    NEW_MODELS,
    WIDTHS,
    candidate_manifest,
    validate_model_artifacts,
)
from experiments.run_catboost_matched_ablation import MINUTE_PATH, load_prepared_data
from models.zoo import MODELS


def _atomic_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _matching_complete(
    model_root: Path,
    *,
    model_name: str,
    manifest: dict[str, Any],
) -> dict[str, Any] | None:
    cached_manifest = _load_json(model_root / "candidate_manifest.json")
    state = _load_json(model_root / "run_state.json")
    result = _load_json(model_root / "result.json")
    if (
        cached_manifest != manifest
        or not state
        or state.get("status") != "complete"
        or not result
        or result.get("status") != "complete"
    ):
        return None
    try:
        validate_model_artifacts(model_root, model_name=model_name)
    except (FileNotFoundError, ValueError):
        return None
    return {**result, "resumed": True}


def run_model(
    model_name: str,
    *,
    output_root: Path,
    prepared: Any,
    store: Any,
    smoke: bool,
) -> dict[str, Any]:
    """Run one model in its isolated resumable directory."""
    manifest = candidate_manifest(model_name)
    model_root = Path(output_root) / model_name
    if not smoke:
        completed = _matching_complete(
            model_root, model_name=model_name, manifest=manifest
        )
        if completed is not None:
            return completed
    _atomic_json(manifest, model_root / "candidate_manifest.json")
    candidates = manifest["candidates"][:1] if smoke else manifest["candidates"]
    runner = ExecutionResolutionRunner(
        output_root=model_root,
        prediction_root=model_root / "prediction_cache",
        store=store,
        prepared=prepared,
        widths=(WIDTHS[0],) if smoke else WIDTHS,
        candidates=candidates,
        candidate_ids=(0,) if smoke else tuple(range(len(candidates))),
        model_factory=MODELS[model_name],
        model_name=model_name,
        smoke=smoke,
        fold_limit=1 if smoke else 5,
        stage1_only=smoke,
    )
    return runner.run()


def run_sequence(
    model_names: Sequence[str],
    *,
    output_root: Path,
    prepared: Any,
    store: Any,
    smoke: bool,
) -> dict[str, Any]:
    """Run models sequentially and persist stop-on-failure state."""
    output_root = Path(output_root)
    state_path = output_root / "run_state.json"
    completed: list[str] = []
    write_run_state(state_path, status="running", detail={"completed_models": completed})
    try:
        for model_name in model_names:
            write_run_state(
                state_path,
                status="running",
                detail={"active_model": model_name, "completed_models": completed},
            )
            run_model(
                model_name,
                output_root=output_root,
                prepared=prepared,
                store=store,
                smoke=smoke,
            )
            completed.append(model_name)
        result = {"status": "complete", "completed_models": completed}
        write_run_state(state_path, status="complete", detail=result)
        return result
    except Exception:
        write_run_state(
            state_path,
            status="failed",
            detail={
                "active_model": model_names[len(completed)],
                "completed_models": completed,
                "traceback": traceback.format_exc(),
            },
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    code_root = Path(__file__).resolve().parents[1]
    default_root = (
        code_root / "experiments" / "cache" / "tuning" / "matched_model_zoo_1m"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", choices=("all", *NEW_MODELS), default="all")
    parser.add_argument("--output-root", type=Path, default=default_root)
    parser.add_argument("--minute-path", type=Path, default=MINUTE_PATH)
    args = parser.parse_args(argv)

    output_root = args.output_root
    if args.smoke and output_root == default_root:
        output_root = default_root.with_name(default_root.name + "_smoke")
    model_names = NEW_MODELS if args.model == "all" else (args.model,)
    widths = (WIDTHS[0],) if args.smoke else WIDTHS
    prepared = load_prepared_data(widths=widths, minute_path=args.minute_path)
    store = PartitionedIntrabarStore.one_minute(args.minute_path)
    result = run_sequence(
        model_names,
        output_root=output_root,
        prepared=prepared,
        store=store,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
