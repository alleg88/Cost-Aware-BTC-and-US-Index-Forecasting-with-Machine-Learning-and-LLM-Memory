"""Pure contracts for the like-for-like CatBoost F1 control."""
from __future__ import annotations

import pandas as pd

from experiments.catboost_economic_optuna import StudyScope, study_scopes


def control_scopes() -> list[StudyScope]:
    """Use exactly the economic study's primary and rolling windows."""
    return study_scopes()


def select_f1_candidate(grid: pd.DataFrame) -> pd.Series:
    """Select robust regime F1, then overall F1, with a stable ID tie-break."""
    if grid.empty:
        raise ValueError("F1 candidate grid is empty")
    return grid.sort_values(
        ["robust_f1", "overall_f1", "candidate"],
        ascending=[False, False, True],
    ).iloc[0]


def combine_objective_tables(
    economic: pd.DataFrame, f1_control: pd.DataFrame
) -> pd.DataFrame:
    """Stack both objectives without dropping diagnostic failures."""
    key = ["width", "mode"]
    for label, frame in (("Economic-tuned", economic), ("F1-tuned", f1_control)):
        missing = set(key) - set(frame.columns)
        if missing:
            raise ValueError(f"{label} table missing columns: {sorted(missing)}")
        if frame.duplicated(key).any():
            raise ValueError(f"{label} table has duplicate width/mode rows")
    left = economic.copy()
    left.insert(0, "objective", "Economic-tuned")
    right = f1_control.copy()
    right.insert(0, "objective", "F1-tuned")
    return pd.concat([left, right], ignore_index=True)