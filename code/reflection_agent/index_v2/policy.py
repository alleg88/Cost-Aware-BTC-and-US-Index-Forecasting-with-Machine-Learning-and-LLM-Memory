"""Pure H1 calibration and deterministic reversal-score controls."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScorePolicy:
    target_rate: float
    threshold: int
    tie_hash_cutoff: float
    seed: int
    calibration_size: int
    target_count: int

    def __post_init__(self) -> None:
        if not 0.0 < float(self.target_rate) < 1.0:
            raise ValueError("target_rate must lie inside (0, 1)")
        if not isinstance(self.threshold, int) or isinstance(self.threshold, bool):
            raise ValueError("threshold must be an integer")
        if not 0 <= self.threshold <= 1000:
            raise ValueError("threshold must lie inside [0, 1000]")
        if not math.isfinite(self.tie_hash_cutoff) or not 0.0 <= self.tie_hash_cutoff <= 1.0:
            raise ValueError("tie_hash_cutoff must lie inside [0, 1]")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be an integer")
        if self.calibration_size < 1:
            raise ValueError("calibration_size must be positive")
        if not 1 <= self.target_count <= self.calibration_size:
            raise ValueError("target_count is outside the calibration sample")


def _validated_scores_ids(
    scores: Sequence[int], opportunity_ids: Sequence[str]
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = list(scores)
    identifiers = tuple(str(item) for item in opportunity_ids)
    if not values or len(values) != len(identifiers):
        raise ValueError("scores and opportunity IDs must be nonempty and aligned")
    if any(not isinstance(item, (int, np.integer)) or isinstance(item, (bool, np.bool_)) for item in values):
        raise ValueError("every reversal score must be an integer")
    array = np.asarray(values, dtype=int)
    if ((array < 0) | (array > 1000)).any():
        raise ValueError("every reversal score must lie inside [0, 1000]")
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("opportunity IDs must be nonempty and unique")
    return array, identifiers


def _hash_fraction(opportunity_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{opportunity_id}".encode("utf-8")).digest()
    return int.from_bytes(digest, "big") / float((1 << 256) - 1)


def calibrate_score_policy(
    scores: Sequence[int],
    opportunity_ids: Sequence[str],
    target_rate: float,
    seed: int,
) -> ScorePolicy:
    """Freeze a score threshold and deterministic tie cutoff on H1 only."""
    score, identifiers = _validated_scores_ids(scores, opportunity_ids)
    if not math.isfinite(float(target_rate)) or not 0.0 < float(target_rate) < 1.0:
        raise ValueError("target_rate must lie inside (0, 1)")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    target_count = max(1, min(len(score), int(math.floor(target_rate * len(score) + 0.5))))
    threshold = int(np.sort(score)[::-1][target_count - 1])
    greater_count = int((score > threshold).sum())
    tie_needed = target_count - greater_count
    tie_hashes = sorted(
        _hash_fraction(identifier, seed)
        for identifier, value in zip(identifiers, score, strict=True)
        if int(value) == threshold
    )
    if not 1 <= tie_needed <= len(tie_hashes):
        raise AssertionError("threshold tie count cannot attain the H1 target")
    return ScorePolicy(
        target_rate=float(target_rate),
        threshold=threshold,
        tie_hash_cutoff=float(tie_hashes[tie_needed - 1]),
        seed=seed,
        calibration_size=len(score),
        target_count=target_count,
    )


def apply_score_policy(
    scores: Sequence[int], opportunity_ids: Sequence[str], policy: ScorePolicy
) -> np.ndarray:
    """Apply a frozen H1 score policy without observing outcomes."""
    if not isinstance(policy, ScorePolicy):
        raise ValueError("policy must be a ScorePolicy")
    score, identifiers = _validated_scores_ids(scores, opportunity_ids)
    tie_hash = np.asarray(
        [_hash_fraction(identifier, policy.seed) for identifier in identifiers],
        dtype=float,
    )
    return (score > policy.threshold) | (
        (score == policy.threshold) & (tie_hash <= policy.tie_hash_cutoff)
    )


def uncertainty_scores(opportunities: pd.DataFrame) -> np.ndarray:
    """Score current ensemble uncertainty without labels, returns or execution data."""
    columns: list[str] = []
    for model_index in range(9):
        columns.extend(
            [f"m{model_index:02d}_p_short", f"m{model_index:02d}_p_long"]
        )
    missing = set(columns).difference(opportunities.columns)
    if missing:
        raise ValueError(f"opportunities miss probability columns: {sorted(missing)}")
    values = opportunities.loc[:, columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("probabilities must be finite")
    if ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probabilities must lie inside [0, 1]")
    margins = []
    for model_index in range(9):
        margins.append(
            opportunities[f"m{model_index:02d}_p_long"].to_numpy(float)
            - opportunities[f"m{model_index:02d}_p_short"].to_numpy(float)
        )
    mean_margin = np.abs(np.mean(np.column_stack(margins), axis=1))
    return np.rint(1000.0 * (1.0 - np.clip(mean_margin, 0.0, 1.0))).astype(int)


def seeded_hash_scores(opportunity_ids: Sequence[str], seed: int) -> np.ndarray:
    """Return a deterministic no-market-information score control."""
    identifiers = tuple(str(item) for item in opportunity_ids)
    if not identifiers or any(not item for item in identifiers):
        raise ValueError("opportunity IDs must be nonempty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("opportunity IDs must be unique")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    return np.asarray(
        [int(round(1000.0 * _hash_fraction(identifier, seed))) for identifier in identifiers],
        dtype=int,
    )


def select_h1_policy(candidate_metrics: pd.DataFrame) -> pd.Series:
    """Choose one H1 candidate by Sortino, Net, drawdown and lower target rate."""
    required = {"target_rate", "daily_sortino", "net_return", "max_drawdown"}
    missing = required.difference(candidate_metrics.columns)
    if candidate_metrics.empty or missing:
        raise ValueError(f"H1 candidate metrics are empty or incomplete: {sorted(missing)}")
    work = candidate_metrics.copy()
    values = work.loc[:, sorted(required)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("H1 candidate metrics must be finite")
    if ((work["target_rate"] <= 0.0) | (work["target_rate"] >= 1.0)).any():
        raise ValueError("H1 target rates must lie inside (0, 1)")
    ranked = work.sort_values(
        ["daily_sortino", "net_return", "max_drawdown", "target_rate"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    return ranked.iloc[0].copy()


__all__ = [
    "ScorePolicy",
    "apply_score_policy",
    "calibrate_score_policy",
    "seeded_hash_scores",
    "select_h1_policy",
    "uncertainty_scores",
]
