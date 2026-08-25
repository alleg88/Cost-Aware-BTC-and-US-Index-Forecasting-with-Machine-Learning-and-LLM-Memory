"""Paired positioning ablation — walk-forward protocol, inside the dev window only.

Notebook 01b's second protocol. The CV protocol (blocking splits, pooled
out-of-fold) has more statistical power but averages over locally stationary
blocks, so it is blind to drift — and drift is exactly what decides whether
funding/OI features survive deployment. This runner supplies the drift-exposed
twin: weekly retraining on a rolling 180-day history.

Three arms, identical rows / windows / hyperparameters, differing ONLY in the
feature block:

  base      price + order flow                       (no sentiment, no tuning)
  pos       base + the 7 positioning features
  placebo   base + the same columns circularly shifted 41 days — same marginals
            and autocorrelation, alignment to price destroyed

The walk-forward span stops at the end of the development window
(2024-07-01 .. 2025-06-30, after a 180-day warm-up), so this experiment never
touches the forward evaluation window or the lockbox.

Run:  python -m experiments.run_positioning_ablation
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import build_dataset
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WF_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"

WIDTHS = (55, 60, 65, 75)
MODEL = "catboost_balanced"
DEV_START, DEV_END = "2024-01-01", "2025-06-30"
WALK_START, WALK_END = "2024-07-01", "2025-06-30"      # after the 180d warm-up
PLACEBO_SHIFT_BARS = 41 * 96                            # 41 days of M15 bars


def cache_path(arm: str, width: int) -> Path:
    return WF_DIR / f"btc_of-{arm}_dz{width}_devwf.parquet"


def arm_frames(df: pd.DataFrame, pos: pd.DataFrame, width: int):
    """base / pos / placebo design matrices on one shared row index."""
    dfp = df.join(pos.reindex(df.index))
    shifted = pd.DataFrame(np.roll(pos.reindex(df.index).to_numpy(), PLACEBO_SHIFT_BARS,
                                   axis=0),
                           index=df.index, columns=pos.columns)
    dfq = df.join(shifted)

    frames = {
        "base": build_dataset(dfp, threshold_bps=width, horizon=1),
        "pos": build_dataset(dfp, threshold_bps=width, horizon=1, positioning=True),
        "placebo": build_dataset(dfq, threshold_bps=width, horizon=1, positioning=True),
    }
    common = frames["base"][0].index
    for X, _ in frames.values():
        common = common.intersection(X.index)
    return {k: (X.loc[common], y.loc[common]) for k, (X, y) in frames.items()}, common


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    working = pd.read_parquet(CODE_ROOT / cfg["paths"]["working_parquet"])
    df = working.loc[DEV_START:DEV_END]
    pos = pd.read_parquet(CODE_ROOT / "data" / "btcusdt_positioning_m15_2024_2026.parquet")
    fwd = df["close"].pct_change().shift(-1).rename("forward_return")

    WF_DIR.mkdir(parents=True, exist_ok=True)
    for width in WIDTHS:
        frames, common = arm_frames(df, pos, width)
        windows = weekly_walkforward_windows(
            common, walk_start=WALK_START, walk_end=WALK_END, train_lookback="180D")
        print(f"dz{width}: {len(common):,} shared rows, {len(windows)} weekly windows")

        for arm, (X, y) in frames.items():
            out = cache_path(arm, width)
            if out.exists():
                print(f"  [cached] {out.name}")
                continue
            preds = run_walkforward_predictions(
                X, y, windows=windows, model_factory=MODELS[MODEL], model_name=MODEL,
                params=None, min_train_rows=500, min_validation_rows=50,
                include_features=False, progress_label=f"dz{width}:{arm}",
                train_tail_trim=1)
            if preds.empty:
                raise RuntimeError(f"no predictions for dz{width}:{arm}")
            preds = preds.join(fwd, how="left")
            preds.to_parquet(out)
            print(f"  wrote {len(preds):,} rows -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
