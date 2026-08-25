"""Validated two-table scoreboard for the Notebook 02d raw sentiment study."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.all_model_sentiment_raw import (
    ARMS,
    DEFAULT_ROOT,
    POLICY_COLUMNS,
    WIDTHS,
    validate_raw_model_artifacts,
)
from experiments.raw_hold_control import MODEL_NAMES
from experiments.run_catboost_matched_ablation import _atomic_json, _atomic_parquet


ARM_LABELS = {
    "none": "No sentiment",
    "classic": "DeBERTa",
    "llm": "LLM-matched",
    "llm_full": "LLM-full",
}


def _load(root: Path, filename: str) -> pd.DataFrame:
    frames = []
    for arm in ARMS:
        for model_name in MODEL_NAMES:
            model_root = Path(root) / arm / model_name
            validate_raw_model_artifacts(model_root, model_name=model_name)
            frame = pd.read_parquet(model_root / filename).copy()
            if set(frame["model_name"].astype(str)) != {model_name}:
                raise ValueError(f"{arm}/{model_name}: embedded model name changed")
            if set(frame["sentiment_arm"].astype(str)) != {arm}:
                raise ValueError(f"{arm}/{model_name}: embedded sentiment arm changed")
            frame["Arm"] = ARM_LABELS[arm]
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
    classification = _load(root, "classification_2024.parquet")
    economics = _load(root, "raw_forward_summary.parquet")
    _validate_keys(classification, table="classification")
    _validate_keys(economics, table="economics")
    forbidden = POLICY_COLUMNS.intersection(economics.columns)
    if forbidden:
        raise ValueError(f"economics contains policy columns: {sorted(forbidden)}")
    if not economics["lookback_days"].astype(int).eq(180).all():
        raise ValueError("economics contains a non-180-day fit")
    if not economics["hold_minutes"].astype(int).eq(15).all():
        raise ValueError("economics contains a non-15-minute hold")
    return {
        "classification": classification.sort_values(
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
            "protocol": "notebook02c-all-model-sentiment-raw-v1",
            "models": list(MODEL_NAMES),
            "sentiment_arms": list(ARMS),
            "widths": list(WIDTHS),
            "lookback_days": 180,
            "hold_minutes": 15,
            "confidence_threshold": None,
            "tp_bps": None,
            "sl_bps": None,
            "policy_calibration_used": False,
            "one_minute_execution_used": False,
            "sealed_lockbox": True,
            "artifact_rows": {name: len(frame) for name, frame in tables.items()},
        },
        output / "manifest.json",
    )
    return tables


if __name__ == "__main__":
    written = write_scoreboards()
    print({name: len(frame) for name, frame in written.items()})
