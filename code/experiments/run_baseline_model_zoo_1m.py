"""Run nine fixed baseline models through the Notebook 02e protocol."""
from __future__ import annotations

import argparse
import json
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from experiments.baseline_model_zoo_1m import (
    BASELINE_CANDIDATE_ID,
    LOOKBACK_DAYS,
    MODEL_NAMES,
    WIDTHS,
    BaselineModelRunner,
    validate_model_artifacts,
)
from experiments.catboost_execution_resolution import write_run_state
from experiments.run_catboost_matched_ablation import MINUTE_PATH, load_prepared_data
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "baseline_model_zoo_1m_180d_fixed15_monthly_h1"
)


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


def _write_state(
    path: Path, *, status: str, detail: dict[str, Any], attempts: int = 5
) -> None:
    """Retry a transient Windows lock around atomic state-file replacement."""
    for attempt in range(attempts):
        try:
            write_run_state(path, status=status, detail=detail)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def protocol_manifest(model_name: str) -> dict[str, Any]:
    return {
        "protocol_version": "baseline-model-zoo-1m-monthly-h1-180d-fixed15-v4",
        "model_name": model_name,
        "candidate_id": BASELINE_CANDIDATE_ID,
        "candidate_params": {},
        "widths": list(WIDTHS),
        "lookback_days": list(LOOKBACK_DAYS),
        "hold_bars": [1],
        "sentiment": "none",
    }


def _matching_complete(model_root: Path, model_name: str) -> dict[str, Any] | None:
    if _load_json(model_root / "protocol.json") != protocol_manifest(model_name):
        return None
    result = _load_json(model_root / "result.json")
    if not result or result.get("status") != "complete":
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
    smoke: bool,
) -> dict[str, Any]:
    model_root = Path(output_root) / model_name
    if not smoke:
        complete = _matching_complete(model_root, model_name)
        if complete is not None:
            return complete
    _atomic_json(protocol_manifest(model_name), model_root / "protocol.json")
    runner = BaselineModelRunner(
        output_root=model_root,
        prepared=prepared,
        model_name=model_name,
        model_factory=MODELS[model_name],
        widths=(WIDTHS[0],) if smoke else WIDTHS,
        candidate_id=BASELINE_CANDIDATE_ID,
        candidate_params={},
        smoke=smoke,
    )
    return runner.run()


def run_sequence(
    model_names: Sequence[str],
    *,
    output_root: Path,
    prepared: Any,
    smoke: bool,
) -> dict[str, Any]:
    output_root = Path(output_root)
    state_path = output_root / "run_state.json"
    completed: list[str] = []
    _write_state(state_path, status="running", detail={"completed_models": []})
    try:
        for model_name in model_names:
            _write_state(
                state_path,
                status="running",
                detail={"active_model": model_name, "completed_models": completed},
            )
            run_model(
                model_name,
                output_root=output_root,
                prepared=prepared,
                smoke=smoke,
            )
            completed.append(model_name)
        result = {"status": "complete", "completed_models": completed}
        _write_state(state_path, status="complete", detail=result)
        return result
    except Exception:
        _write_state(
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
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", choices=("all", *MODEL_NAMES), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--minute-path", type=Path, default=MINUTE_PATH)
    args = parser.parse_args(argv)

    output_root = args.output_root
    if args.smoke and output_root == DEFAULT_ROOT:
        output_root = DEFAULT_ROOT.with_name(DEFAULT_ROOT.name + "_smoke")
    widths = (WIDTHS[0],) if args.smoke else WIDTHS
    prepared = load_prepared_data(
        widths=widths,
        minute_path=args.minute_path,
        sentiment="none",
    )
    model_names = MODEL_NAMES if args.model == "all" else (args.model,)
    result = run_sequence(
        model_names,
        output_root=output_root,
        prepared=prepared,
        smoke=args.smoke,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
