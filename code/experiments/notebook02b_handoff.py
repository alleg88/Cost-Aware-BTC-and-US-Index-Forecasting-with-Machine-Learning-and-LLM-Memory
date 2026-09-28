"""Frozen model handoff from Notebook 02b to the raw sentiment ablation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.notebook02_handoff import (
    BASE_FEATURE_COLUMNS,
    PIPELINE_HANDOFF,
    load_pipeline_handoff,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "notebook02b_handoff"
HANDOFF_PATH = HANDOFF_ROOT / "handoff.json"
BASELINE_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.10,
    "l2_leaf_reg": 3.0,
    "auto_class_weights": "Balanced",
    "random_seed": 42,
    "loss_function": "MultiClass",
}


def _fingerprint(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def write_notebook02b_handoff(path: str | Path = HANDOFF_PATH) -> Path:
    upstream = load_pipeline_handoff(PIPELINE_HANDOFF)
    payload: dict[str, Any] = {
        "upstream_notebook": "03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb",
        "downstream_notebook": "13_RQ3_B_BTC_nine_model_sentiment_ablation.ipynb",
        "decision": "fixed_baseline_retained",
        "decision_reason": "F1 and economic tuning did not produce a robust profitable improvement after costs",
        "objective": "baseline",
        "candidate_id": 0,
        "model": "CatBoost",
        "hyperparameters": BASELINE_PARAMS,
        "widths": [55, 65, 75],
        "training_history_days": 180,
        "base_features": list(BASE_FEATURE_COLUMNS),
        "sentiment": "disabled_at_handoff",
        "downstream_evaluation": {
            "mode": "raw_fixed_hold",
            "hold_bars": 1,
            "hold_minutes": 15,
            "fee_bps_per_side": 5.0,
            "confidence_threshold": None,
            "tp_bps": None,
            "sl_bps": None,
        },
        "sealed_lockbox_start": "2026-04-01T00:00:00+00:00",
        "upstream_handoff_fingerprint": upstream["handoff_fingerprint"],
    }
    payload["handoff_fingerprint"] = _fingerprint(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_notebook02b_handoff(path: str | Path = HANDOFF_PATH) -> Mapping[str, Any]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    fingerprint = payload.pop("handoff_fingerprint", None)
    if fingerprint != _fingerprint(payload):
        raise ValueError("Notebook 02b handoff fingerprint mismatch")
    payload["handoff_fingerprint"] = fingerprint
    if payload.get("decision") != "fixed_baseline_retained":
        raise ValueError("Notebook 02b must hand off the fixed baseline")
    if payload.get("candidate_id") != 0 or payload.get("objective") != "baseline":
        raise ValueError("Notebook 02b baseline identity changed")
    if payload.get("hyperparameters") != BASELINE_PARAMS:
        raise ValueError("Notebook 02b baseline hyperparameters changed")
    if payload.get("widths") != [55, 65, 75]:
        raise ValueError("Notebook 02b handoff must retain DZ55, DZ65 and DZ75")
    if payload.get("training_history_days") != 180:
        raise ValueError("Notebook 02b handoff must use 180 days")
    if payload.get("base_features") != list(BASE_FEATURE_COLUMNS):
        raise ValueError("Notebook 02b base feature schema changed")
    execution = payload.get("downstream_evaluation", {})
    if execution != {
        "mode": "raw_fixed_hold",
        "hold_bars": 1,
        "hold_minutes": 15,
        "fee_bps_per_side": 5.0,
        "confidence_threshold": None,
        "tp_bps": None,
        "sl_bps": None,
    }:
        raise ValueError("Notebook 02d raw evaluation contract changed")
    return payload


if __name__ == "__main__":
    print(write_notebook02b_handoff())
