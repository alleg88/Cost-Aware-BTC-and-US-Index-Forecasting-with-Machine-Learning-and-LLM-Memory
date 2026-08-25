"""Fixed protocol contracts for the matched CatBoost objective ablation."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path

import pandas as pd

from evaluation.splits import BlockingTimeSeriesSplit

SELECTION_START = pd.Timestamp("2024-01-01", tz="UTC")
SELECTION_END = pd.Timestamp("2025-01-01", tz="UTC")
CALIBRATION_END = pd.Timestamp("2025-07-01", tz="UTC")
FORWARD_END = pd.Timestamp("2026-04-01", tz="UTC")
LOCKBOX_START = FORWARD_END

WIDTHS = (55, 65, 75)
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
GEOMETRIES = ((150, 75, 1), (150, 100, 1), (200, 100, 1))
CANDIDATE_COUNT = 15
REGIMES = ("bull", "sideways", "bear")
REGIME_LOOKBACK_BARS = 672
REGIME_RETURN_THRESHOLD = 0.02
MIN_REGIME_ROWS = 50
MIN_TRADES = 50
MIN_SIDE_TRADES = 15
CALIBRATION_SEGMENTS = 6
WORST_RANK_VALUE = -1_000_000.0
CANDIDATE_POOL_FINGERPRINT = (
    "17a1b1999630d824f8bd257aacf8541f67058c31d05d572d36f60c85516288d7"
)

EXPECTED_ARTIFACT_ROWS = {
    "classification": 45,
    "selection_policy_grid": 1485,
    "candidate_policy_winners": 45,
    "selected_candidates": 9,
    "calibration_policy_grid": 297,
    "selected_policies": 9,
    "forward_monthly": 81,
    "forward_quarterly": 27,
    "forward_summary": 9,
}

DEFAULT_CANDIDATE_PATH = (
    Path(__file__).resolve().parent / "catboost_candidates_15.json"
)


def span_mask(
    index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
):
    """Return a left-inclusive, right-exclusive UTC span mask."""
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None:
        raise ValueError("span index must be timezone-aware")
    timestamps = timestamps.tz_convert("UTC")
    return (timestamps >= start) & (timestamps < end)


def fixed_splitter() -> BlockingTimeSeriesSplit:
    return BlockingTimeSeriesSplit(n_splits=5, train_frac=0.8, embargo=4)


def policy_choices() -> tuple[tuple[float, tuple[int, int, int]], ...]:
    return tuple((tau, geometry) for geometry in GEOMETRIES for tau in TAUS)


def past_regime_labels(
    close: pd.Series,
    *,
    lookback: int = REGIME_LOOKBACK_BARS,
    threshold: float = REGIME_RETURN_THRESHOLD,
) -> pd.Series:
    """Classify state from only the current and trailing M15 closes."""
    values = close.astype(float)
    trailing_return = values / values.shift(lookback) - 1.0
    regimes = pd.Series("sideways", index=close.index, dtype="object", name="regime")
    regimes.loc[trailing_return > threshold] = "bull"
    regimes.loc[trailing_return < -threshold] = "bear"
    regimes.loc[trailing_return.isna()] = "unknown"
    return regimes


def validate_regime_counts(counts: Mapping[str, int]) -> None:
    too_small = {
        regime: int(counts.get(regime, 0))
        for regime in REGIMES
        if int(counts.get(regime, 0)) < MIN_REGIME_ROWS
    }
    if too_small:
        raise ValueError(
            f"every validation regime needs at least {MIN_REGIME_ROWS} rows: "
            f"{too_small}"
        )


def validate_fold_regime_counts(
    fold_counts: Iterable[Mapping[str, int]],
) -> None:
    folds = tuple(fold_counts)
    if len(folds) != 5:
        raise ValueError("regime adequacy requires exactly five validation folds")
    for fold_id, counts in enumerate(folds):
        try:
            validate_regime_counts(counts)
        except ValueError as error:
            raise ValueError(f"fold {fold_id}: {error}") from error


def _finite_values(*values: object) -> tuple[float, ...]:
    numbers = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in numbers):
        raise ValueError("ranking metrics must be finite")
    return numbers


def robust_f1_score(fold_regime_scores: Iterable[Sequence[float]]) -> float:
    """Mean of the weakest regime macro-F1 in each of exactly five folds."""
    folds = tuple(fold_regime_scores)
    if len(folds) != 5:
        raise ValueError("robust F1 requires exactly five validation folds")
    weakest = []
    for scores in folds:
        if len(scores) != 3:
            raise ValueError("each fold must contain bull, sideways, and bear F1")
        weakest.append(min(_finite_values(*scores)))
    return float(sum(weakest) / len(weakest))


def robust_score(
    *,
    pooled_sortino: float,
    pooled_sharpe: float,
    bull_sortino: float,
    sideways_sortino: float,
    bear_sortino: float,
) -> float:
    values = tuple(
        float(value)
        for value in (
            pooled_sortino,
            pooled_sharpe,
            bull_sortino,
            sideways_sortino,
            bear_sortino,
        )
    )
    if not all(math.isfinite(value) for value in values):
        return WORST_RANK_VALUE
    return min(values)


def _validated_id(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return int(value)


def _validated_count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{field} must be a finite nonnegative integer")
    return int(value)


def f1_ranking_key(row: Mapping[str, object]) -> tuple[float, float, int]:
    robust_f1, overall_f1 = _finite_values(row["robust_f1"], row["overall_f1"])
    return (
        -robust_f1,
        -overall_f1,
        _validated_id(row["candidate_id"], "candidate_id"),
    )


def constraint_values(
    row: Mapping[str, object], *, n_segments: int
) -> tuple[float, float, float, float]:
    if not isinstance(n_segments, Integral) or n_segments <= 0:
        raise ValueError("n_segments must be a positive integer")
    required_positive = math.ceil(2 * int(n_segments) / 3)
    return (
        float(MIN_TRADES - _validated_count(row["trades"], "trades")),
        float(MIN_SIDE_TRADES - _validated_count(row["n_long"], "n_long")),
        float(MIN_SIDE_TRADES - _validated_count(row["n_short"], "n_short")),
        float(
            required_positive
            - _validated_count(row["positive_segments"], "positive_segments")
        ),
    )


def total_constraint_violation(
    row: Mapping[str, object], *, n_segments: int
) -> float:
    return float(
        sum(
            max(0.0, value)
            for value in constraint_values(row, n_segments=n_segments)
        )
    )


def economic_ranking_key(
    row: Mapping[str, object], *, n_segments: int
) -> tuple[float, float, float, float, int, int, int]:
    robust, sortino, pooled_net = _finite_values(
        row["robust_score"], row["pooled_sortino"], row["pooled_net"]
    )
    return (
        total_constraint_violation(row, n_segments=n_segments),
        -robust,
        -sortino,
        -pooled_net,
        -_validated_count(row["trades"], "trades"),
        _validated_id(row["policy_id"], "policy_id"),
        _validated_id(row["candidate_id"], "candidate_id"),
    )


def _candidate_pool_fingerprint(payload: Mapping[str, object]) -> str:
    normalized = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def load_candidates(path: Path = DEFAULT_CANDIDATE_PATH) -> tuple[dict, ...]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("candidate payload must be a mapping")
    if payload.get("generated_before_economics") is not True:
        raise ValueError("candidate pool must be outcome-independent")
    if tuple(payload.get("widths", ())) != WIDTHS:
        raise ValueError(f"candidate widths must be {WIDTHS}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != CANDIDATE_COUNT:
        raise ValueError(f"candidate pool must contain exactly {CANDIDATE_COUNT} rows")
    if not all(isinstance(candidate, dict) and candidate for candidate in candidates):
        raise ValueError("each candidate must be a non-empty parameter mapping")
    normalized_candidates = [
        json.dumps(candidate, sort_keys=True, separators=(",", ":"))
        for candidate in candidates
    ]
    if len(set(normalized_candidates)) != CANDIDATE_COUNT:
        raise ValueError("candidate parameter mappings must be unique")
    actual_fingerprint = _candidate_pool_fingerprint(payload)
    if actual_fingerprint != CANDIDATE_POOL_FINGERPRINT:
        raise ValueError(
            "candidate pool fingerprint mismatch: "
            f"expected {CANDIDATE_POOL_FINGERPRINT}, got {actual_fingerprint}"
        )
    return tuple(dict(candidate) for candidate in candidates)


def assert_artifact_row_counts(actual: Mapping[str, int]) -> None:
    mismatches = {
        name: (expected, actual.get(name))
        for name, expected in EXPECTED_ARTIFACT_ROWS.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise AssertionError(f"artifact row-count mismatch: {mismatches}")
