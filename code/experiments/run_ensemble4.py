"""Economics of the 4-model soft-vote ensemble (the sweep-selected seeds).

Seeds chosen by the order-flow threshold sweep on mean/best evaluation Sortino,
spanning both model families for decorrelation: two deep (gru, mlp) + two tree
(catboost, random_forest). XGBoost and LSTM were dropped (see run_xgboost_ab and
the sweep).

Uncalibrated path (default): soft-vote the base models' cached walk-forward
probabilities at each dead-zone threshold, freeze tau on Q1 by Sortino, evaluate
Q2-Q4. This reuses the sweep caches — no retraining.

The `--calibrated` path is handled by run_ensemble4_calibrated (a walk-forward
re-run with per-window isotonic-calibrated bases); this module's job is the
uncalibrated baseline and the head-to-head table.

Run:  python -m experiments.run_ensemble4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.run_threshold_sweep import (
    THRESHOLDS, cache_path, calibrated_economics, sweep_tag,
)
from experiments.spans import CACHE_SUFFIX

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
ECON = CODE_ROOT / "experiments" / "cache" / "economics"
WALKFORWARD_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"

ENSEMBLE4_MODELS = ("gru", "catboost_balanced", "random_forest", "mlp")
ENSEMBLE4_TAG = "ensemble4"
BOOKKEEPING = ("window", "train_start", "train_end", "validation_start",
               "validation_end", "y_true", "forward_return", "vol_regime")


def soft_vote(frames: dict[str, pd.DataFrame], models) -> pd.DataFrame:
    """Equal-weight average of the base models' class probabilities."""
    first = frames[models[0]]
    total = np.zeros((len(first), 3))
    for m in models:
        total += frames[m][[f"{m}_p{i}" for i in range(3)]].to_numpy(dtype=float)
    total /= len(models)
    out = first[[c for c in BOOKKEEPING if c in first.columns]].copy()
    out[f"{ENSEMBLE4_TAG}_pred"] = total.argmax(axis=1)
    out[f"{ENSEMBLE4_TAG}_conf"] = total.max(axis=1)
    for i in range(3):
        out[f"{ENSEMBLE4_TAG}_p{i}"] = total[:, i]
    return out


def build_from_caches(instrument: str, sentiment_tag: str, tag: str,
                      models=ENSEMBLE4_MODELS) -> pd.DataFrame:
    frames = {}
    for m in models:
        p = cache_path(instrument, sentiment_tag, m, tag)
        d = pd.read_parquet(p)
        d.index = pd.to_datetime(d.index, utc=True)
        frames[m] = d.sort_index()
    idx0 = frames[models[0]].index
    for m in models:
        if not frames[m].index.equals(idx0):
            raise ValueError(f"{m} cache index differs at {tag}")
    merged = soft_vote(frames, models)
    out = cache_path(instrument, sentiment_tag, ENSEMBLE4_TAG, tag)
    merged.to_parquet(out)
    return merged


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])

    rows = []
    for thr in THRESHOLDS:
        tag = sweep_tag("deadzone", thr, 16)
        merged = build_from_caches("btc", "bothof", tag)
        merged.index = pd.to_datetime(merged.index, utc=True)
        s = calibrated_economics(merged.sort_index(), ENSEMBLE4_TAG, fee)
        s["threshold_bps"] = thr
        rows.append(s)
        print(f"  {tag}: tau={s['tau']:.2f} Sortino {s['sortino']:+.2f} "
              f"Sharpe {s['sharpe']:+.2f} F1 {s['macro_f1']:.3f} "
              f"net {s['net_return_sum']:+.4f} trades {s['trade_count']}")

    ens = pd.DataFrame(rows)
    ECON.mkdir(parents=True, exist_ok=True)
    ens.to_parquet(ECON / f"btc_bothof_ensemble4_{CACHE_SUFFIX}.parquet", index=False)

    best = ens.sort_values("sortino", ascending=False).iloc[0]
    sweep = pd.read_parquet(ECON / f"btc_bothof_sweep_deadzone_sortino_{CACHE_SUFFIX}.parquet")
    best_single = sweep.sort_values("sortino", ascending=False).iloc[0]
    credible = (sweep[sweep["trade_count"] >= 20]
                .sort_values("sortino", ascending=False).iloc[0])

    print(f"\nensemble4 seeds: {ENSEMBLE4_MODELS}")
    print(f"ensemble4 best : {best['threshold_bps']:.0f} bps  Sortino {best['sortino']:+.2f} "
          f"Sharpe {best['sharpe']:+.2f} F1 {best['macro_f1']:.3f} "
          f"net {best['net_return_sum']:+.4f} trades {int(best['trade_count'])}")
    print(f"best single    : {best_single['model']} @ {best_single['threshold_bps']:.0f} bps "
          f"Sortino {best_single['sortino']:+.2f} (trades {int(best_single['trade_count'])})")
    print(f"best credible  : {credible['model']} @ {credible['threshold_bps']:.0f} bps "
          f"Sortino {credible['sortino']:+.2f} (trades {int(credible['trade_count'])}, >=20)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
