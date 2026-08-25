"""Nested long/short expected-net regressors for frozen TP200/SL100/hold1."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.run_joint_path_selection import PRIMARY_TAU
from experiments.run_side_path_selection import GEOMETRY
from experiments.run_tune_antibull_widths import OUT_DIR, select_economic_candidate

OUTPUT_DIR = OUT_DIR / "side_net_regression_selection"


def realized_net_target(
    candidates: pd.DataFrame, *, fee_bps: float
) -> pd.Series:
    round_trip_fee = 2.0 * float(fee_bps) / 10_000.0
    return (
        candidates["gross_return"].astype(float) - round_trip_fee
    ).rename("realized_net")


def net_training_mask(
    candidates: pd.DataFrame, cutoff: pd.Timestamp, *, side: int
) -> pd.Series:
    close_time = pd.to_datetime(candidates["outcome_close_time"], utc=True)
    return (close_time < pd.Timestamp(cutoff)) & candidates["side"].eq(side)


def apply_net_filter(
    prediction: pd.Series,
    confidence: pd.Series,
    expected_net: pd.Series,
    *,
    primary_tau: float = PRIMARY_TAU,
    long_tau: float,
    short_tau: float,
) -> pd.Series:
    filtered = prediction.astype(int).copy()
    score = expected_net.reindex(filtered.index)
    threshold = pd.Series(np.nan, index=filtered.index, dtype=float)
    threshold.loc[filtered.eq(2)] = float(long_tau)
    threshold.loc[filtered.eq(0)] = float(short_tau)
    keep = (
        filtered.isin((0, 2))
        & confidence.reindex(filtered.index).ge(float(primary_tau))
        & score.notna()
        & score.ge(threshold)
    )
    filtered.loc[~keep] = 1
    return filtered


def select_net_policy(
    grid: pd.DataFrame, *, n_folds: int
) -> pd.Series | None:
    return select_economic_candidate(grid, n_folds=n_folds)


def main() -> int:
    from experiments.side_net_regression_study import run_study

    return run_study()


if __name__ == "__main__":
    raise SystemExit(main())

