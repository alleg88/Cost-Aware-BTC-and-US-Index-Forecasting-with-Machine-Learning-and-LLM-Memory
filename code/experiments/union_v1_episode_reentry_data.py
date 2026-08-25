"""Frozen 2021 data contract for the Union-v1-style re-entry experiment."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.splits import BlockingTimeSeriesSplit
from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.unified_side_profitability_data import (
    load_frozen_development_dataset,
)


UNION_TARGETS = {
    "lstm": "target_dz55",
    "svm_linear": "target_dz75",
}
DEVELOPMENT_START = pd.Timestamp("2021-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")
BAR_INTERVAL = pd.Timedelta(minutes=15)


def _dead_zone_target(
    forward_return: pd.Series,
    valid: pd.Series,
    width_bps: float,
) -> np.ndarray:
    target = np.full(len(forward_return), -1, dtype=np.int8)
    usable = valid.to_numpy(bool) & np.isfinite(forward_return.to_numpy(float))
    values = forward_return.to_numpy(float)
    threshold = float(width_bps) / 10_000.0
    target[usable] = 1
    target[usable & (values >= threshold)] = 2
    target[usable & (values <= -threshold)] = 0
    return target


def attach_union_dead_zone_targets(dataset: UnifiedDataset) -> UnifiedDataset:
    """Attach exact next-M15 close-to-close DZ55 and DZ75 targets."""
    decisions = dataset.decisions.copy()
    required = {"row_key", "decision_time", "close"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Union dead-zone targets need columns: {missing}")
    decisions["decision_time"] = pd.to_datetime(
        decisions["decision_time"], utc=True, errors="raise"
    )
    if not decisions["decision_time"].is_monotonic_increasing:
        raise ValueError("Union decisions must be chronological")
    if decisions["row_key"].duplicated().any():
        raise ValueError("Union decision keys must be unique")

    next_time = decisions["decision_time"].shift(-1)
    valid = next_time.sub(decisions["decision_time"]).eq(BAR_INTERVAL)
    close = pd.to_numeric(decisions["close"], errors="coerce")
    forward_return = close.shift(-1).div(close).sub(1.0)
    valid &= np.isfinite(close) & close.gt(0.0)
    valid &= np.isfinite(close.shift(-1)) & close.shift(-1).gt(0.0)

    decisions["union_target_time"] = next_time.where(valid)
    decisions["target_dz55"] = _dead_zone_target(forward_return, valid, 55.0)
    decisions["target_dz75"] = _dead_zone_target(forward_return, valid, 75.0)
    return replace(dataset, decisions=decisions)


def make_union_reentry_manifest(
    dataset: UnifiedDataset,
    *,
    n_splits: int = 5,
    train_fraction: float = 0.8,
    embargo_bars: int = 8,
) -> pd.DataFrame:
    """Build five independent train/embargo/test blocks for both Union targets."""
    if n_splits != 5 or train_fraction != 0.8 or embargo_bars != 8:
        raise ValueError("Notebook 04h keeps the registered 5x80/20 split and embargo 8")
    decisions = dataset.decisions.reset_index(drop=True).copy()
    required = {
        "row_key",
        "decision_time",
        "union_target_time",
        *UNION_TARGETS.values(),
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"Union re-entry manifest needs columns: {missing}")
    decisions["decision_time"] = pd.to_datetime(decisions["decision_time"], utc=True)
    decisions["union_target_time"] = pd.to_datetime(
        decisions["union_target_time"], utc=True
    )
    if not decisions["decision_time"].is_monotonic_increasing:
        raise ValueError("Union manifest decisions must be chronological")

    splitter = BlockingTimeSeriesSplit(
        n_splits=n_splits,
        train_frac=train_fraction,
        embargo=embargo_bars,
    )
    block_size = len(decisions) // n_splits
    rows: list[pd.DataFrame] = []
    for fold_id, (fit_positions, test_positions) in enumerate(splitter.split(decisions)):
        block_start = fold_id * block_size
        block_stop = block_start + block_size
        positions = np.arange(block_start, block_stop, dtype=np.int64)
        block = decisions.iloc[positions].copy()
        block.insert(0, "position", positions)
        block.insert(0, "fold_id", fold_id)
        role = np.full(len(block), "outer_embargo", dtype=object)
        role[np.isin(positions, fit_positions)] = "fit"
        role[np.isin(positions, test_positions)] = "test"

        valid = block[list(UNION_TARGETS.values())].ge(0).all(axis=1)
        valid &= block["union_target_time"].notna()
        if block_stop < len(decisions):
            block_end = decisions.loc[block_stop, "decision_time"]
        else:
            block_end = decisions.loc[block_stop - 1, "decision_time"] + BAR_INTERVAL
        valid &= block["union_target_time"].lt(block_end)
        role[(role != "outer_embargo") & ~valid.to_numpy()] = "target_censored"
        block["role"] = role
        rows.append(
            block[
                [
                    "fold_id",
                    "row_key",
                    "position",
                    "decision_time",
                    "union_target_time",
                    "target_dz55",
                    "target_dz75",
                    "role",
                ]
            ]
        )

    manifest = pd.concat(rows, ignore_index=True)
    test_keys = manifest.loc[manifest["role"].eq("test"), "row_key"]
    if not test_keys.is_unique:
        raise AssertionError("Union OOF test keys must be unique")
    for fold_id, fold in manifest.groupby("fold_id", sort=True):
        fit = fold.loc[fold["role"].eq("fit")]
        test = fold.loc[fold["role"].eq("test")]
        if fit.empty or test.empty:
            raise AssertionError(f"Union fold {fold_id} needs fit and test rows")
        if int(fit["position"].max()) + embargo_bars >= int(test["position"].min()):
            raise AssertionError(f"Union fold {fold_id} violates the embargo")
        if not fit["union_target_time"].max() < test["decision_time"].min():
            raise AssertionError(f"Union fold {fold_id} target crosses the test boundary")
    return manifest


def load_frozen_union_reentry_dataset(
    root: str | Path,
) -> tuple[UnifiedDataset, pd.DataFrame, dict[str, object]]:
    """Hash-verify Notebook 04d development data and attach the 04h contract."""
    dataset, source_audit = load_frozen_development_dataset(root)
    decisions = dataset.decisions
    if decisions["decision_time"].min() < DEVELOPMENT_START:
        raise AssertionError("frozen Union development starts before 2021")
    if decisions["decision_time"].max() >= DEVELOPMENT_END:
        raise AssertionError("frozen Union development reaches 2025")
    dataset = attach_union_dead_zone_targets(dataset)
    manifest = make_union_reentry_manifest(dataset)
    audit = dict(source_audit)
    audit.update(
        {
            "target_names": list(UNION_TARGETS.values()),
            "target_horizon_bars": 1,
            "n_splits": 5,
            "embargo_bars": 8,
            "manifest_rows": len(manifest),
            "oof_test_rows": int(manifest["role"].eq("test").sum()),
            "lockbox_2026_q2_used": False,
        }
    )
    return dataset, manifest, audit


__all__ = [
    "BAR_INTERVAL",
    "DEVELOPMENT_END",
    "DEVELOPMENT_START",
    "UNION_TARGETS",
    "attach_union_dead_zone_targets",
    "load_frozen_union_reentry_dataset",
    "make_union_reentry_manifest",
]

