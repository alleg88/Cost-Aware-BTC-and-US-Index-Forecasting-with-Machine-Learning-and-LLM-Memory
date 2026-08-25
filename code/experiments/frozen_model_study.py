"""Paths and cache contracts for the frozen nine-model primary study."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.model_zoo_protocol import BASE_MODELS, protocol_fingerprint

CODE_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "antibull_model_zoo"


@dataclass(frozen=True)
class StudyPaths:
    root: Path
    predictions: Path
    candidates: Path
    classification: Path
    economics: Path
    audit: Path
    ungated: Path
    objective: Path
    manifest: Path
    result: Path

    @classmethod
    def for_model(cls, model: str, *, smoke: bool = False) -> "StudyPaths":
        if model not in BASE_MODELS:
            raise ValueError(f"unsupported base model: {model}")
        root = STUDY_ROOT / ("smoke" if smoke else "") / model
        return cls(
            root=root,
            predictions=root / "predictions",
            candidates=root / "candidates.json",
            classification=root / "classification_grid.parquet",
            economics=root / "economic_grid.parquet",
            audit=root / "outer_audit.parquet",
            ungated=root / "ungated_summary.parquet",
            objective=root / "objective_summary.parquet",
            manifest=root / "manifest.json",
            result=root / "result.json",
        )


def _params_fingerprint(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]


def prediction_cache_path(
    paths: StudyPaths,
    model: str,
    *,
    width: int,
    candidate_id: int,
    params: dict,
    fold_id: int,
    month: str,
) -> Path:
    """Return a cache name isolated by protocol, model, parameters, and fold."""
    if model not in BASE_MODELS:
        raise ValueError(f"unsupported base model: {model}")
    return paths.predictions / (
        f"{protocol_fingerprint()}_{model}_w{width}_candidate_{candidate_id:02d}_"
        f"{_params_fingerprint(params)}_fold_{fold_id:02d}_{month}.parquet"
    )


def validate_prediction_frame(
    frame: pd.DataFrame,
    *,
    model: str,
    development_end: pd.Timestamp,
) -> pd.DataFrame:
    """Reject malformed, non-probabilistic, or forward-period prediction caches."""
    required = {
        "y_true",
        f"{model}_pred",
        f"{model}_conf",
        f"{model}_p0",
        f"{model}_p1",
        f"{model}_p2",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"prediction cache missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("prediction cache index must be timezone-aware")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("prediction cache index must be unique and sorted")
    if len(frame) and frame.index.max() >= development_end:
        raise ValueError("prediction cache crosses development boundary")

    probability_columns = [f"{model}_p{i}" for i in range(3)]
    probabilities = frame[probability_columns].to_numpy(dtype=float)
    confidence = frame[f"{model}_conf"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.isfinite(confidence).all():
        raise ValueError("prediction probabilities and confidence must be finite")
    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("prediction probabilities must be between zero and one")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("prediction probabilities must sum to one")
    if not np.allclose(confidence, probabilities.max(axis=1), atol=1e-6):
        raise ValueError("prediction confidence must equal maximum probability")
    for column in ("y_true", f"{model}_pred"):
        if not set(frame[column].astype(int).unique()).issubset({0, 1, 2}):
            raise ValueError(f"{column} contains an unknown class")
    return frame

def study_command(
    model: str,
    *,
    n_trials: int = 15,
    fold_limit: int | None = None,
    candidate_limit: int | None = None,
) -> list[str]:
    """Build the isolated CLI command for one frozen base-model study."""
    if model not in BASE_MODELS:
        raise ValueError(f"unsupported base model: {model}")
    command = [
        sys.executable,
        "-m",
        "experiments.run_tune_antibull_widths",
        "--model-zoo",
        "--model",
        model,
        "--trials",
        str(n_trials),
    ]
    for flag, value in (
        ("--fold-limit", fold_limit),
        ("--candidate-limit", candidate_limit),
    ):
        if value is not None:
            if value < 1:
                raise ValueError(f"{flag} must be positive")
            command.extend([flag, str(value)])
    return command


def run_model_study(
    model: str,
    *,
    n_trials: int = 15,
    fold_limit: int | None = None,
    candidate_limit: int | None = None,
) -> dict:
    """Run one isolated study and return its written result payload."""
    command = study_command(
        model,
        n_trials=n_trials,
        fold_limit=fold_limit,
        candidate_limit=candidate_limit,
    )
    subprocess.run(command, cwd=CODE_ROOT, check=True)
    smoke = fold_limit is not None or candidate_limit is not None
    result_path = StudyPaths.for_model(model, smoke=smoke).result
    if not result_path.exists():
        raise RuntimeError(f"study completed without result: {result_path}")
    return json.loads(result_path.read_text(encoding="utf-8"))
