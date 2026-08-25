"""Frozen contracts for the eight-model one-minute economic study."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.model_zoo_protocol import candidate_pool

NEW_MODELS = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "mlp",
    "lstm",
    "gru",
)
WIDTHS = (55, 65, 75)
CANDIDATE_COUNT = 15
SEED = 42
PROTOCOL_VERSION = "matched-model-zoo-1m-v1"
DEFAULT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "cache"
    / "tuning"
    / "matched_model_zoo_1m"
)

_EXPECTED_COUNTS = {
    "classification_2024.parquet": 45,
    "economic_policy_grid_2024.parquet": 2_970,
    "economic_candidate_winners_2024.parquet": 45,
    "selected_candidates_2024.parquet": 3,
    "calibration_policy_grid_2025h1.parquet": 198,
    "selected_policies_2025h1.parquet": 3,
    "forward_monthly.parquet": 27,
    "forward_quarterly.parquet": 9,
    "forward_summary.parquet": 3,
}


def _content_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def candidate_manifest(model_name: str) -> dict[str, Any]:
    """Return the deterministic outcome-independent candidate pool identity."""
    if model_name not in NEW_MODELS:
        raise ValueError(f"unsupported matched model: {model_name}")
    candidates = candidate_pool(model_name, n_trials=CANDIDATE_COUNT, seed=SEED)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "model_name": model_name,
        "candidate_count": CANDIDATE_COUNT,
        "seed": SEED,
        "candidates": candidates,
    }
    return {**payload, "fingerprint": _content_hash(payload)}


def expected_counts() -> dict[str, int]:
    """Return exact full-run parquet row counts for one model."""
    return dict(_EXPECTED_COUNTS)


def validate_model_artifacts(root: Path, *, model_name: str) -> dict[str, int]:
    """Validate the exact full-run parquet inventory for one model."""
    if model_name not in NEW_MODELS:
        raise ValueError(f"unsupported matched model: {model_name}")
    root = Path(root)
    actual: dict[str, int] = {}
    for name, expected in _EXPECTED_COUNTS.items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"missing full artifact: {name}")
        rows = len(pd.read_parquet(path))
        if rows != expected:
            raise ValueError(f"{name} has {rows} rows; expected {expected}")
        actual[name] = rows
    return actual


def validate_full_study(root: Path = DEFAULT_ROOT) -> dict[str, int]:
    """Reconcile full artifacts, frozen policies, and the sealed boundary."""
    root = Path(root)
    totals = {
        "models": 0,
        "selected_candidates": 0,
        "selected_policies": 0,
        "forward_monthly": 0,
        "forward_quarterly": 0,
        "forward_summary": 0,
    }
    policy_columns = (
        "candidate_id",
        "policy_id",
        "tau",
        "tp_bps",
        "sl_bps",
        "max_hold",
        "fit_id",
    )
    for model_name in NEW_MODELS:
        model_root = root / model_name
        validate_model_artifacts(model_root, model_name=model_name)
        selected_candidates = pd.read_parquet(
            model_root / "selected_candidates_2024.parquet"
        )
        selected_policies = pd.read_parquet(
            model_root / "selected_policies_2025h1.parquet"
        )
        monthly = pd.read_parquet(model_root / "forward_monthly.parquet")
        quarterly = pd.read_parquet(model_root / "forward_quarterly.parquet")
        summary = pd.read_parquet(model_root / "forward_summary.parquet")
        if set(selected_candidates["width_bps"].astype(int)) != set(WIDTHS):
            raise ValueError(f"{model_name}: selected candidates omit a width")
        if set(selected_policies["width_bps"].astype(int)) != set(WIDTHS):
            raise ValueError(f"{model_name}: selected policies omit a width")
        for width in WIDTHS:
            candidate = selected_candidates.loc[
                selected_candidates["width_bps"] == width
            ].iloc[0]
            policy = selected_policies.loc[selected_policies["width_bps"] == width].iloc[0]
            if int(candidate["candidate_id"]) != int(policy["candidate_id"]):
                raise ValueError(f"{model_name} DZ{width}: candidate changed in H1")
            for label, frame, expected_rows in (
                ("monthly", monthly, 9),
                ("quarterly", quarterly, 3),
                ("summary", summary, 1),
            ):
                rows = frame.loc[frame["width_bps"] == width]
                if len(rows) != expected_rows:
                    raise ValueError(
                        f"{model_name} DZ{width}: expected {expected_rows} {label} rows"
                    )
                for column in policy_columns:
                    if not rows[column].eq(policy[column]).all():
                        raise ValueError(
                            f"{model_name} DZ{width}: frozen {column} changed in {label}"
                        )
        manifest = json.loads((model_root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("sealed_lockbox") is not True:
            raise ValueError(f"{model_name}: lockbox is not sealed")
        if pd.Timestamp(manifest["forward_end_exclusive"]) != pd.Timestamp(
            "2026-04-01", tz="UTC"
        ):
            raise ValueError(f"{model_name}: wrong forward boundary")
        if pd.Timestamp(manifest["forward_execution_max_timestamp"]) >= pd.Timestamp(
            "2026-04-01", tz="UTC"
        ):
            raise ValueError(f"{model_name}: forward execution reached the lockbox")
        totals["models"] += 1
        totals["selected_candidates"] += len(selected_candidates)
        totals["selected_policies"] += len(selected_policies)
        totals["forward_monthly"] += len(monthly)
        totals["forward_quarterly"] += len(quarterly)
        totals["forward_summary"] += len(summary)
    return totals
