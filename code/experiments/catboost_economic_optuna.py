"""True economic Optuna protocol for CatBoost on BTC M15.

The primary study tunes on 2024 only and freezes its hyperparameters and
execution policy for 2025-H1.  Three rolling studies retune on nine earlier
monthly folds and audit April, May, and June 2025 respectively.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import optuna
import pandas as pd

from memory.loop import ReflectionWindow

WIDTHS = (55, 65, 75)
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
GEOMETRIES = ((150, 75, 1), (150, 100, 1), (200, 100, 1))
LOOKBACK_DAYS = 90
TRIALS = 15
SEED = 42
MIN_TRADES = 50
MIN_SIDE_TRADES = 15
DEVELOPMENT_END = pd.Timestamp("2025-07-01", tz="UTC")

BASELINE_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "random_strength": 1.0,
    "bagging_temperature": 1.0,
    "rsm": 1.0,
}


@dataclass(frozen=True)
class StudyScope:
    name: str
    inner_fold_ids: tuple[int, ...]
    outer_fold_id: int | None


def monthly_folds() -> list[ReflectionWindow]:
    starts = pd.date_range("2024-04-01", "2025-06-01", freq="MS", tz="UTC")
    return [
        ReflectionWindow(
            name=f"catboost_econ_{i:02d}_{start:%Y-%m}",
            train_start=start - pd.Timedelta(days=LOOKBACK_DAYS),
            train_end=start,
            validation_start=start,
            validation_end=start
            + pd.offsets.MonthBegin(1)
            - pd.Timedelta(minutes=15),
        )
        for i, start in enumerate(starts)
    ]


def primary_tuning_fold_ids() -> tuple[int, ...]:
    return tuple(range(9))


def primary_evaluation_fold_ids() -> tuple[int, ...]:
    return tuple(range(9, 15))


def rolling_schedule() -> list[tuple[tuple[int, ...], int]]:
    return [
        (tuple(range(3, 12)), 12),
        (tuple(range(4, 13)), 13),
        (tuple(range(5, 14)), 14),
    ]


def study_scopes() -> list[StudyScope]:
    folds = monthly_folds()
    scopes = [StudyScope("primary_2024", primary_tuning_fold_ids(), None)]
    scopes.extend(
        StudyScope(
            f"rolling_pre_{folds[outer].validation_start:%Y_%m}", inner, outer
        )
        for inner, outer in rolling_schedule()
    )
    return scopes


def policy_choices() -> tuple[tuple[float, tuple[int, int, int]], ...]:
    return tuple((tau, geometry) for geometry in GEOMETRIES for tau in TAUS)


def constraints_from_trial(trial) -> tuple[float, ...]:
    return tuple(float(value) for value in trial.user_attrs["constraints"])


def completed_trial_count(study: optuna.Study) -> int:
    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )


def robust_score(
    *,
    pooled_sortino: float,
    pooled_sharpe: float,
    bull_sortino: float,
    sideways_sortino: float,
    bear_sortino: float,
) -> float:
    values = np.asarray(
        [
            pooled_sortino,
            pooled_sharpe,
            bull_sortino,
            sideways_sortino,
            bear_sortino,
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        return -1_000_000.0
    return float(values.min())


def constraint_values(
    row: Mapping[str, object] | pd.Series, *, n_folds: int
) -> tuple[float, float, float, float]:
    required_positive = math.ceil(2 * n_folds / 3)
    return (
        float(MIN_TRADES - int(row["trades"])),
        float(MIN_SIDE_TRADES - int(row["n_long"])),
        float(MIN_SIDE_TRADES - int(row["n_short"])),
        float(required_positive - int(row["positive_folds"])),
    )


def rank_policy(grid: pd.DataFrame, *, n_folds: int) -> pd.Series:
    """Return the best row without deleting failed-diagnostic policies."""
    if grid.empty:
        raise ValueError("policy grid is empty")
    ranked = grid.copy()
    ranked["constraint_violation"] = [
        sum(max(0.0, value) for value in constraint_values(row, n_folds=n_folds))
        for _, row in ranked.iterrows()
    ]
    return ranked.sort_values(
        [
            "constraint_violation",
            "robust_score",
            "pooled_sortino",
            "pooled_net",
            "trades",
        ],
        ascending=[True, False, False, False, False],
    ).iloc[0]


def sample_catboost_params(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 300, 800, step=100),
        "depth": trial.suggest_int("depth", 5, 7),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.03, 0.10, log=True
        ),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 3.0, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.5),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.0, 1.5
        ),
        "rsm": trial.suggest_float("rsm", 0.65, 1.0),
    }


def parameter_fingerprint(params: Mapping[str, object]) -> str:
    raw = json.dumps(dict(params), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def prediction_cache_path(
    root: Path,
    *,
    width: int,
    params: Mapping[str, object],
    fold_id: int,
    month: str,
) -> Path:
    return Path(root) / (
        f"w{width}_{parameter_fingerprint(params)}_fold_{fold_id:02d}_{month}.parquet"
    )


def protocol_fingerprint() -> str:
    payload = {
        "widths": WIDTHS,
        "taus": TAUS,
        "geometries": GEOMETRIES,
        "lookback_days": LOOKBACK_DAYS,
        "trials": TRIALS,
        "seed": SEED,
        "minimum_trades": MIN_TRADES,
        "minimum_side_trades": MIN_SIDE_TRADES,
        "objective": "min pooled Sortino, pooled Sharpe, bull/sideways/bear Sortino",
        "primary_tuning": primary_tuning_fold_ids(),
        "primary_evaluation": primary_evaluation_fold_ids(),
        "rolling_schedule": rolling_schedule(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
