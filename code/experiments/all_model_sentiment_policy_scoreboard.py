"""Validated all-model sentiment policy tables for Notebook 02e."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.all_model_sentiment_policy import (
    ARMS,
    DEFAULT_ROOT,
    FORWARD_END,
    FORWARD_START,
    POLICY_FIELDS,
    WIDTHS,
    validate_policy_model_artifacts,
)
from experiments.all_model_sentiment_scoreboard import ARM_LABELS
from experiments.raw_hold_control import MODEL_NAMES
from experiments.run_catboost_matched_ablation import _atomic_json, _atomic_parquet


def _load(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        for model_name in MODEL_NAMES:
            model_root = Path(root) / arm / model_name
            validate_policy_model_artifacts(model_root, model_name=model_name)
            frame = pd.read_parquet(model_root / filename).copy()
            if set(frame["model_name"].astype(str)) != {model_name}:
                raise ValueError(f"{arm}/{model_name}: embedded model changed")
            frame.insert(0, "sentiment_arm", arm)
            frame.insert(1, "Arm", ARM_LABELS[arm])
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _validate_keys(frame: pd.DataFrame, *, table: str) -> None:
    keys = ["sentiment_arm", "model_name", "width_bps"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{table}: duplicate Arm/Model/DZ rows")
    expected = len(ARMS) * len(MODEL_NAMES) * len(WIDTHS)
    if len(frame) != expected:
        raise ValueError(f"{table}: expected {expected} rows")
    if set(frame["width_bps"].astype(int)) != set(WIDTHS):
        raise ValueError(f"{table}: dead-zone coverage changed")


def build_scoreboards(root: Path = DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    root = Path(root)
    policies = _load(root, "selected_policies_2025h1.parquet")
    economics = _load(root, "forward_summary.parquet")
    _validate_keys(policies, table="policies")
    _validate_keys(economics, table="economics")
    if not policies["monthly_fit_count"].astype(int).eq(6).all():
        raise ValueError("policy table contains a non-six-month calibration")
    keys = ["sentiment_arm", "model_name", "width_bps"]
    for field in POLICY_FIELDS:
        left = policies.set_index(keys)[field].sort_index()
        right = economics.set_index(keys)[field].sort_index()
        if not left.equals(right):
            raise ValueError(f"forward {field} differs from frozen H1 policy")
    if not pd.to_datetime(economics["period_start"], utc=True).eq(FORWARD_START).all():
        raise ValueError("forward start changed")
    if not pd.to_datetime(economics["period_end"], utc=True).eq(FORWARD_END).all():
        raise ValueError("forward end changed")
    return {
        "policies": policies.sort_values(
            ["width_bps", "model_name", "sentiment_arm"]
        ).reset_index(drop=True),
        "economics": economics.sort_values(
            ["width_bps", "model_name", "sentiment_arm"]
        ).reset_index(drop=True),
    }


def write_scoreboards(root: Path = DEFAULT_ROOT) -> dict[str, pd.DataFrame]:
    root = Path(root)
    tables = build_scoreboards(root)
    output = root / "combined"
    for name, frame in tables.items():
        _atomic_parquet(frame, output / f"{name}.parquet")
        frame.to_csv(output / f"{name}.csv", index=False)
    _atomic_json(
        {
            "protocol": "notebook02e-all-model-sentiment-policy-only-v2",
            "models": list(MODEL_NAMES),
            "sentiment_arms": list(ARMS),
            "widths": list(WIDTHS),
            "lookback_days": 180,
            "hold_minutes": 15,
            "policy_count_per_model_arm_width": 33,
            "calibration_months": 6,
            "reuse_notebook02c_raw_forward": True,
            "rerun_2024_folds": False,
            "one_minute_execution_used": True,
            "sealed_lockbox": True,
            "artifact_rows": {name: len(frame) for name, frame in tables.items()},
        },
        output / "manifest.json",
    )
    return tables


if __name__ == "__main__":
    written = write_scoreboards()
    print({name: len(frame) for name, frame in written.items()})
