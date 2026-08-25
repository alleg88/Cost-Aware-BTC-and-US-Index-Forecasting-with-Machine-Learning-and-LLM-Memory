"""Class-preserving SVM DZ75 temperature selection and H1 confirmation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from experiments.qualified_union import CODE_ROOT, FORWARD_START, load_member_panel


TEMPERATURE_GRID = (0.50, 0.75, 1.00, 1.25, 1.50, 2.00)
RAW_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "all_model_sentiment_raw_180d_fixed15"
    / "none"
    / "svm_linear"
)
OUTPUT_ROOT = CODE_ROOT / "experiments" / "cache" / "svm_temperature_calibration"
PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (n_rows, 3)")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("probabilities must be finite and non-negative")
    if not np.isfinite(temperature) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    row_sum = values.sum(axis=1, keepdims=True)
    values = np.divide(
        values,
        row_sum,
        out=np.full_like(values, 1.0 / 3.0),
        where=row_sum > 0.0,
    )
    logits = np.log(np.clip(values, 1e-12, 1.0)) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _multiclass_brier(probabilities: np.ndarray, labels: np.ndarray) -> float:
    target = np.eye(3, dtype=float)[np.asarray(labels, dtype=int)]
    return float(np.mean(np.sum((np.asarray(probabilities) - target) ** 2, axis=1)))


def _ece(probabilities: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    values = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    confidence = values.max(axis=1)
    correct = values.argmax(axis=1).eq(y) if isinstance(values, pd.DataFrame) else values.argmax(axis=1) == y
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(values)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence >= left) & (
            (confidence <= right) if right == 1.0 else (confidence < right)
        )
        if mask.any():
            error += float(mask.sum() / total) * abs(
                float(np.mean(correct[mask])) - float(np.mean(confidence[mask]))
            )
    return float(error)


def select_temperature(
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, pd.DataFrame]:
    values = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if len(values) != len(y) or len(y) == 0:
        raise ValueError("probabilities and labels must be non-empty and aligned")
    base_class = values.argmax(axis=1)
    rows = []
    for temperature in TEMPERATURE_GRID:
        scaled = apply_temperature(values, temperature)
        rows.append(
            {
                "temperature": float(temperature),
                "log_loss": float(log_loss(y, scaled, labels=[0, 1, 2])),
                "brier": _multiclass_brier(scaled, y),
                "ece": _ece(scaled, y),
                "mean_confidence": float(scaled.max(axis=1).mean()),
                "changed_classes": int((scaled.argmax(axis=1) != base_class).sum()),
            }
        )
    grid = pd.DataFrame(rows)
    selected = float(grid.sort_values(["log_loss", "temperature"]).iloc[0]["temperature"])
    return selected, grid


def load_2024_oof() -> tuple[pd.DataFrame, list[Path]]:
    paths = sorted((RAW_ROOT / "predictions").glob("w75_candidate_00_fold_*.parquet"))
    if len(paths) != 5:
        raise ValueError(f"expected five SVM DZ75 2024 OOF files, found {len(paths)}")
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.drop_duplicates("timestamp").sort_values("timestamp")
    if frame.empty or frame["timestamp"].max() >= pd.Timestamp("2025-01-01", tz="UTC"):
        raise ValueError("SVM temperature selection must be confined to 2024 OOF")
    return frame, paths


def _metric_row(arm: str, probabilities: np.ndarray, labels: np.ndarray) -> dict[str, object]:
    return {
        "arm": arm,
        "rows": int(len(labels)),
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1, 2])),
        "brier": _multiclass_brier(probabilities, labels),
        "ece": _ece(probabilities, labels),
        "mean_confidence": float(np.asarray(probabilities).max(axis=1).mean()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run() -> dict[str, object]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    oof, source_paths = load_2024_oof()
    oof_probability = oof.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    oof_labels = oof["y_true"].to_numpy(int)
    selected, grid = select_temperature(oof_probability, oof_labels)
    grid.to_csv(OUTPUT_ROOT / "temperature_grid.csv", index=False)

    h1 = load_member_panel("svm_linear", 75, "h1").reset_index()
    h1_probability = h1.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    h1_labels = h1["y_true"].to_numpy(int)
    h1_scaled = apply_temperature(h1_probability, selected)
    h1_metrics = pd.DataFrame(
        [
            _metric_row("raw", h1_probability, h1_labels),
            _metric_row("temperature_scaled", h1_scaled, h1_labels),
        ]
    )
    h1_metrics.to_csv(OUTPUT_ROOT / "h1_confirmation.csv", index=False)
    calibrated = h1[
        ["timestamp", "y_true", "pred", "confidence", *PROBABILITY_COLUMNS, "refit_id"]
    ].copy()
    calibrated[["p_short_cal", "p_flat_cal", "p_long_cal"]] = h1_scaled
    calibrated["confidence_cal"] = h1_scaled.max(axis=1)
    calibrated["temperature"] = selected
    calibrated.to_parquet(OUTPUT_ROOT / "calibrated_h1.parquet", index=False)

    base_class_2024 = oof_probability.argmax(axis=1)
    scaled_2024 = apply_temperature(oof_probability, selected)
    changed_2024 = int((base_class_2024 != scaled_2024.argmax(axis=1)).sum())
    changed_h1 = int((h1_probability.argmax(axis=1) != h1_scaled.argmax(axis=1)).sum())
    raw_h1 = h1_metrics.set_index("arm").loc["raw"]
    scaled_h1 = h1_metrics.set_index("arm").loc["temperature_scaled"]
    summary = {
        "study": "svm_dz75_class_preserving_temperature",
        "temperature_grid": list(TEMPERATURE_GRID),
        "selected_temperature": selected,
        "selection_period": "2024_oof",
        "confirmation_period": "2025_h1",
        "selection_rows": int(len(oof)),
        "confirmation_rows": int(len(h1)),
        "changed_classes_2024": changed_2024,
        "changed_classes_h1": changed_h1,
        "h1_log_loss_improvement": float(raw_h1["log_loss"] - scaled_h1["log_loss"]),
        "h1_brier_improvement": float(raw_h1["brier"] - scaled_h1["brier"]),
        "h1_confirmation_pass": bool(
            scaled_h1["log_loss"] <= raw_h1["log_loss"] + 1e-12
            and scaled_h1["brier"] <= raw_h1["brier"] + 1e-12
            and changed_h1 == 0
        ),
        "union_v1_economics_changed": False,
        "reason_union_unchanged": "svm_tau_zero_and_temperature_preserves_classes",
        "forward_loaded": False,
        "lockbox_2026_q2_used": False,
        "max_loaded_timestamp": str(h1["timestamp"].max()),
    }
    _write_json(OUTPUT_ROOT / "summary.json", summary)

    artifact_paths = sorted(path for path in OUTPUT_ROOT.iterdir() if path.name != "manifest.json")
    manifest = {
        "source_hashes": {
            str(path.relative_to(CODE_ROOT)).replace("\\", "/"): _sha256(path)
            for path in source_paths
        },
        "artifact_hashes": {path.name: _sha256(path) for path in artifact_paths},
        "forward_loaded": False,
        "lockbox_2026_q2_used": False,
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return summary


def main() -> int:
    summary = run()
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OUTPUT_ROOT",
    "TEMPERATURE_GRID",
    "apply_temperature",
    "load_2024_oof",
    "run",
    "select_temperature",
]
