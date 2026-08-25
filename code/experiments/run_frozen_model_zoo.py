"""Sequential resumable launcher for the frozen nine-model BTC study."""
from __future__ import annotations

import argparse
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

from experiments.frozen_model_study import STUDY_ROOT, run_model_study
from experiments.model_zoo_protocol import BASE_MODELS, protocol_fingerprint

DEFAULT_RUN_STATE_PATH = STUDY_ROOT / "run_state.json"
RUN_STATE_PATH = DEFAULT_RUN_STATE_PATH


def parse_models(raw: str) -> list[str]:
    """Parse an explicit subset without changing the frozen registry order."""
    if raw == "all":
        return list(BASE_MODELS)
    models = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = [model for model in models if model not in BASE_MODELS]
    if unknown:
        raise ValueError(f"unknown model(s): {unknown}")
    if not models:
        raise ValueError("at least one model is required")
    if len(set(models)) != len(models):
        raise ValueError("duplicate models are not allowed")
    return models


def completed_state(model: str, fingerprint: str) -> dict:
    """Return the smallest valid completed state for tests and recovery."""
    return {
        "protocol_fingerprint": fingerprint,
        "models": {model: {"status": "complete"}},
    }


def should_resume(state: dict, model: str, fingerprint: str) -> bool:
    return bool(
        state.get("protocol_fingerprint") == fingerprint
        and state.get("models", {}).get(model, {}).get("status") == "complete"
    )


def _load_state(path: Path, fingerprint: str) -> dict:
    if not path.exists():
        return {"protocol_fingerprint": fingerprint, "models": {}}
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("protocol_fingerprint") != fingerprint:
        return {"protocol_fingerprint": fingerprint, "models": {}}
    state.setdefault("models", {})
    return state


def _write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    temporary.replace(path)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="all")
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--fold-limit", type=int)
    parser.add_argument("--candidate-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    models = parse_models(args.models)
    fingerprint = protocol_fingerprint()
    smoke = args.fold_limit is not None or args.candidate_limit is not None
    state_path = RUN_STATE_PATH
    if RUN_STATE_PATH == DEFAULT_RUN_STATE_PATH and smoke:
        state_path = STUDY_ROOT / "smoke" / "run_state.json"
    state = _load_state(state_path, fingerprint)

    for model in models:
        if args.resume and should_resume(state, model, fingerprint):
            print(f"skip complete -> {model}")
            continue

        state["models"][model] = {
            "status": "running",
            "started_at": _timestamp(),
            "trials": args.trials,
            "fold_limit": args.fold_limit,
            "candidate_limit": args.candidate_limit,
        }
        _write_state(state_path, state)
        try:
            result = run_model_study(
                model,
                n_trials=args.trials,
                fold_limit=args.fold_limit,
                candidate_limit=args.candidate_limit,
            )
        except Exception:
            state["models"][model].update(
                {
                    "status": "failed",
                    "finished_at": _timestamp(),
                    "traceback": traceback.format_exc(),
                }
            )
            _write_state(state_path, state)
            raise

        state["models"][model].update(
            {
                "status": "complete",
                "finished_at": _timestamp(),
                "result": result,
            }
        )
        _write_state(state_path, state)
        print(f"complete -> {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
