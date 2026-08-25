"""Frozen expected-net data contract for Notebook 04f.

The adapter reuses Notebook 04d/04e's hash-verified development rows and
four-role chronology.  Its only semantic changes are two already-costed
continuous targets in basis points and a diagnostic name for the unused
policy-selection role.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import (
    UnifiedDataConfig,
    UnifiedDataset,
)
from experiments.unified_side_profitability_data import (
    load_frozen_development_dataset,
    make_four_role_manifest,
)


EXPECTED_NET_TARGETS = ("target_long_bps", "target_short_bps")
EXPECTED_NET_ROLES = (
    "fit",
    "probability_calibration",
    "fixed_policy_preflight",
    "test",
)


def attach_expected_net_targets(dataset: UnifiedDataset) -> UnifiedDataset:
    """Attach conditional LONG/SHORT after-cost returns in basis points."""
    decisions = dataset.decisions.copy()
    required = {"path_complete", "net_return_long", "net_return_short"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"expected-net targets need columns: {missing}")

    complete = decisions["path_complete"].fillna(False).astype(bool)
    for side in ("long", "short"):
        source = pd.to_numeric(
            decisions[f"net_return_{side}"], errors="coerce"
        ).astype(float)
        if not np.isfinite(source.loc[complete].to_numpy(float)).all():
            raise ValueError(f"complete paths need finite net_return_{side}")
        decisions[f"target_{side}_bps"] = (10_000.0 * source).where(complete)
    return replace(dataset, decisions=decisions)


def make_expected_net_manifest(
    dataset: UnifiedDataset,
    config: UnifiedDataConfig = UnifiedDataConfig(),
    *,
    embargo_bars: int = 8,
) -> pd.DataFrame:
    """Reuse the purged four-role manifest and mark preflight as diagnostic."""
    manifest = make_four_role_manifest(
        dataset,
        config,
        embargo_bars=embargo_bars,
    ).copy()
    manifest.loc[
        manifest["role"].eq("policy_selection"), "role"
    ] = "fixed_policy_preflight"
    required = set(EXPECTED_NET_ROLES)
    for fold_id, fold in manifest.groupby("fold_id", sort=True):
        if not required.issubset(set(fold["role"])):
            raise AssertionError(f"fold {fold_id} lacks an expected-net role")
        ordered = [fold.loc[fold["role"].eq(role)] for role in EXPECTED_NET_ROLES]
        for earlier, later in zip(ordered, ordered[1:]):
            if not earlier["label_end"].max() < later["decision_time"].min():
                raise AssertionError(
                    f"fold {fold_id} label endpoints cross an expected-net role"
                )
    return manifest


def load_frozen_expected_net_dataset(
    root: str | Path,
) -> tuple[UnifiedDataset, dict[str, object]]:
    """Hash-verify the frozen source and attach the registered regression targets."""
    dataset, source_audit = load_frozen_development_dataset(root)
    dataset = attach_expected_net_targets(dataset)
    audit = dict(source_audit)
    complete = dataset.decisions["path_complete"].fillna(False).astype(bool)
    audit.update(
        {
            "target_names": list(EXPECTED_NET_TARGETS),
            "target_units": "after_cost_basis_points",
            "target_rows": int(complete.sum()),
            "cost_subtracted_again": False,
            "lockbox_2026_q2_used": False,
        }
    )
    return dataset, audit


__all__ = [
    "EXPECTED_NET_ROLES",
    "EXPECTED_NET_TARGETS",
    "attach_expected_net_targets",
    "load_frozen_expected_net_dataset",
    "make_expected_net_manifest",
]
