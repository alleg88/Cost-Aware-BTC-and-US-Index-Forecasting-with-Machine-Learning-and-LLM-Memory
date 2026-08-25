"""Frozen data adapter and four-role chronology for Notebook 04e.

Notebook 04e reuses Notebook 04d's verified development decisions, causal
features and paired economic paths.  Only the targets and the inner role split
change: raw fitting, probability calibration and policy selection are kept
strictly chronological before each untouched outer test block.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import (
    UNIFIED_FEATURES,
    UnifiedDataConfig,
    UnifiedDataset,
    make_blocking_fold_manifest,
)


PROFITABILITY_HEADS = ("long_profitable", "short_profitable")
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")
FROZEN_DATA_ARTIFACTS = (
    "decision_dataset.parquet",
    "economic_labels.parquet",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_profitability_targets(dataset: UnifiedDataset) -> UnifiedDataset:
    """Attach direct after-cost LONG and SHORT profitability labels."""
    decisions = dataset.decisions.copy()
    required = {"path_complete", "net_r_long", "net_r_short"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"profitability targets need columns: {missing}")
    complete = decisions["path_complete"].fillna(False).astype(bool)
    decisions["long_profitable"] = np.where(
        complete, pd.to_numeric(decisions["net_r_long"], errors="coerce").gt(0), False
    ).astype(np.int8)
    decisions["short_profitable"] = np.where(
        complete, pd.to_numeric(decisions["net_r_short"], errors="coerce").gt(0), False
    ).astype(np.int8)
    both = decisions["long_profitable"].eq(1) & decisions["short_profitable"].eq(1)
    if both.any():
        raise AssertionError("paired RR2 paths cannot make both side targets positive")
    return replace(dataset, decisions=decisions)


def make_four_role_manifest(
    dataset: UnifiedDataset,
    config: UnifiedDataConfig = UnifiedDataConfig(),
    *,
    embargo_bars: int = 8,
) -> pd.DataFrame:
    """Build five outer folds with fit/calibration/policy/test isolation."""
    if embargo_bars != 8:
        raise ValueError("Notebook 04e keeps the registered eight-bar embargo")
    base_config = replace(
        config,
        calibration_fraction=0.30,
        embargo_bars=embargo_bars,
    )
    labels = dataset.decisions[["row_key", "path_complete", "label_end"]]
    manifest = make_blocking_fold_manifest(
        dataset.decisions,
        labels,
        base_config,
    ).copy()
    manifest["label_end"] = pd.to_datetime(manifest["label_end"], utc=True)
    manifest["decision_time"] = pd.to_datetime(manifest["decision_time"], utc=True)
    manifest["probability_calibration_start_position"] = pd.NA
    manifest["policy_selection_start_position"] = pd.NA

    for fold_id, fold in manifest.groupby("fold_id", sort=True):
        fold_index = fold.index
        calibration_start = int(fold["calibration_start_position"].iloc[0])
        outer_cut = int(fold["outer_cut_position"].iloc[0])
        calibration_span = outer_cut - calibration_start
        if calibration_span < 2 * embargo_bars + 4:
            raise ValueError(f"fold {fold_id} has too few rows for four roles")
        policy_start = calibration_start + int(np.ceil(calibration_span / 2.0))
        policy_embargo_start = policy_start - embargo_bars
        policy_start_time = pd.Timestamp(
            dataset.decisions.iloc[policy_start]["decision_time"]
        )

        original_calibration = fold["role"].eq("calibration")
        position = fold["position"].astype(int)
        probability_valid = (
            original_calibration
            & position.lt(policy_embargo_start)
            & fold["label_end"].lt(policy_start_time)
        )
        policy_valid = original_calibration & position.ge(policy_start)
        policy_embargo = original_calibration & ~(probability_valid | policy_valid)

        manifest.loc[fold_index[probability_valid], "role"] = "probability_calibration"
        manifest.loc[fold_index[policy_valid], "role"] = "policy_selection"
        manifest.loc[fold_index[policy_embargo], "role"] = "policy_embargo"
        manifest.loc[fold_index, "probability_calibration_start_position"] = (
            calibration_start
        )
        manifest.loc[fold_index, "policy_selection_start_position"] = policy_start

    if manifest["role"].eq("calibration").any():
        raise AssertionError("legacy calibration role escaped the four-role split")
    test_keys = manifest.loc[manifest["role"].eq("test"), "row_key"]
    if test_keys.duplicated().any():
        raise AssertionError("outer test keys overlap across folds")

    required_roles = ("fit", "probability_calibration", "policy_selection", "test")
    for fold_id, fold in manifest.groupby("fold_id", sort=True):
        roles = set(fold["role"])
        if not set(required_roles).issubset(roles):
            raise ValueError(f"fold {fold_id} lacks a required four-role partition")
        ordered = [fold.loc[fold["role"].eq(role)] for role in required_roles]
        for earlier, later in zip(ordered, ordered[1:]):
            if not earlier["label_end"].max() < later["decision_time"].min():
                raise AssertionError(f"fold {fold_id} label endpoints cross a role boundary")
            if not set(earlier["row_key"]).isdisjoint(later["row_key"]):
                raise AssertionError(f"fold {fold_id} role keys overlap")
    return manifest


def load_frozen_development_dataset(
    root: str | Path,
) -> tuple[UnifiedDataset, dict[str, object]]:
    """Hash-verify and reconstruct Notebook 04d's frozen development dataset."""
    frozen_root = Path(root).resolve()
    manifest_path = frozen_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if bool(manifest.get("lockbox_2026_q2_used", False)):
        raise AssertionError("frozen source reports Q2-2026 lockbox use")
    expected_hashes = manifest.get("artifact_hashes", {})
    verified: dict[str, str] = {}
    for filename in FROZEN_DATA_ARTIFACTS:
        path = (frozen_root / filename).resolve()
        if path.parent != frozen_root or not path.is_file():
            raise AssertionError(f"frozen development artifact is missing: {filename}")
        expected = expected_hashes.get(filename)
        actual = _sha256_file(path)
        if expected != actual:
            raise AssertionError(f"frozen development hash mismatch: {filename}")
        verified[filename] = actual

    decisions = pd.read_parquet(frozen_root / "decision_dataset.parquet")
    economic_paths = pd.read_parquet(frozen_root / "economic_labels.parquet")
    missing_features = sorted(set(UNIFIED_FEATURES).difference(decisions.columns))
    if missing_features:
        raise ValueError(f"frozen decisions lack features: {missing_features}")
    decisions["decision_time"] = pd.to_datetime(decisions["decision_time"], utc=True)
    decisions["label_end"] = pd.to_datetime(decisions["label_end"], utc=True)
    if decisions["decision_time"].max() >= DEVELOPMENT_END:
        raise AssertionError("frozen development decisions reached 2025")
    if decisions["row_key"].duplicated().any():
        raise AssertionError("frozen development decision keys are not unique")
    tabular = decisions.loc[:, list(UNIFIED_FEATURES)].to_numpy(dtype=np.float32)
    dataset = UnifiedDataset(
        decisions=decisions.reset_index(drop=True),
        tabular=tabular,
        sequences=np.empty((0, 0, 0), dtype=np.float32),
        feature_names=UNIFIED_FEATURES,
        economic_paths=economic_paths.reset_index(drop=True),
    )
    dataset = attach_profitability_targets(dataset)
    audit: dict[str, object] = {
        "source_root": str(frozen_root),
        "source_manifest_sha256": _sha256_file(manifest_path),
        "verified_artifacts": len(verified),
        "artifact_hashes": verified,
        "decision_rows": len(dataset.decisions),
        "complete_rows": int(dataset.decisions["path_complete"].fillna(False).sum()),
        "feature_count": len(UNIFIED_FEATURES),
        "maximum_decision_time": dataset.decisions["decision_time"].max().isoformat(),
    }
    return dataset, audit


__all__ = [
    "DEVELOPMENT_END",
    "FROZEN_DATA_ARTIFACTS",
    "PROFITABILITY_HEADS",
    "attach_profitability_targets",
    "load_frozen_development_dataset",
    "make_four_role_manifest",
]
