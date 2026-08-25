"""Calibrated ensemble4: per-window isotonic-calibrated bases, then soft-vote.

Re-runs the four seed models (gru, catboost, random_forest, mlp) through the
walk-forward with a ChronoIsotonicCalibrated wrapper (leak-free: base on an
earlier fit slice, isotonic on a later calibration slice, both inside each
training window), soft-votes the calibrated probabilities, and computes the
frozen-tau economics. Reports calibrated vs uncalibrated ensemble4 vs the best
single model, to decide whether ensemble4 is worth carrying into RQ3.

Cache tags: `{model}_cal` for calibrated bases, `ensemble4cal` for the vote.
Resumable. Runs a subset of thresholds by default to bound compute.

Run:  python -m experiments.run_ensemble4_calibrated --thresholds 25,35,40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from experiments.run_ensemble4 import ENSEMBLE4_MODELS, soft_vote
from experiments.run_threshold_sweep import cache_path, calibrated_economics, sweep_tag
from experiments.run_walkforward import build_walkforward_xy, resolve_params
from experiments.spans import CACHE_SUFFIX
from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import make_label
from models.calibrate import make_calibrated
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
ECON = CODE_ROOT / "experiments" / "cache" / "economics"
CAL_TAG = "ensemble4cal"


def calibrated_base_cache(instrument, thr, model, cfg):
    """Walk-forward one seed with a per-window calibrated base; cache as {model}_cal."""
    tag = sweep_tag("deadzone", thr, 16)
    name = f"{model}_cal"
    out = cache_path(instrument, "bothof", name, tag)
    if out.exists():
        print(f"  [cached] {out.name}")
        return out
    X, y, aux = build_walkforward_xy(
        instrument, cfg, horizon=1, sentiment="both", orderflow=True,
        label_fn=lambda feat, t=thr: make_label(feat, threshold_bps=t, horizon=1))
    ws, we = cfg["dates"]["walkforward"]
    windows = weekly_walkforward_windows(X.index, walk_start=ws, walk_end=we,
                                         train_lookback="180D")
    params = resolve_params(model, instrument, "both", 1, cfg)
    preds = run_walkforward_predictions(
        X, y, windows=windows,
        model_factory=make_calibrated(MODELS[model]),
        model_name=name, params=params, min_train_rows=500, min_validation_rows=50,
        include_features=True, progress_label=f"{tag}:{name}", train_tail_trim=1)
    preds = preds.join(aux[["forward_return", "vol_regime"]], how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    print(f"  wrote {len(preds):,} rows -> {out.name}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thresholds", default="25,35,40")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    thresholds = [float(t) for t in args.thresholds.split(",")]

    rows = []
    for thr in thresholds:
        tag = sweep_tag("deadzone", thr, 16)
        print(f"[{tag}] calibrated bases")
        frames = {}
        for model in ENSEMBLE4_MODELS:
            p = calibrated_base_cache("btc", thr, model, cfg)
            d = pd.read_parquet(p); d.index = pd.to_datetime(d.index, utc=True)
            frames[f"{model}_cal"] = d.sort_index()
        # soft_vote keys are model names; remap calibrated frames to plain names
        vote_frames = {m: frames[f"{m}_cal"].rename(
            columns={f"{m}_cal_p{i}": f"{m}_p{i}" for i in range(3)})
            for m in ENSEMBLE4_MODELS}
        merged = soft_vote(vote_frames, ENSEMBLE4_MODELS)
        merged.index = pd.to_datetime(merged.index, utc=True)
        merged = merged.rename(columns={"ensemble4_pred": f"{CAL_TAG}_pred",
                                        "ensemble4_conf": f"{CAL_TAG}_conf",
                                        **{f"ensemble4_p{i}": f"{CAL_TAG}_p{i}" for i in range(3)}})
        s = calibrated_economics(merged.sort_index(), CAL_TAG, fee)
        s["threshold_bps"] = thr
        rows.append(s)
        print(f"  {tag} calibrated ensemble4: tau={s['tau']:.2f} "
              f"Sortino {s['sortino']:+.2f} Sharpe {s['sharpe']:+.2f} "
              f"F1 {s['macro_f1']:.3f} net {s['net_return_sum']:+.4f} "
              f"trades {s['trade_count']}")

    cal = pd.DataFrame(rows)
    ECON.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(ECON / f"btc_bothof_ensemble4cal_{CACHE_SUFFIX}.parquet", index=False)

    uncal = pd.read_parquet(ECON / f"btc_bothof_ensemble4_{CACHE_SUFFIX}.parquet")
    print("\n=== calibrated vs uncalibrated ensemble4 (same thresholds) ===")
    hdr = f"{'bps':>5}{'cal_sortino':>13}{'uncal_sortino':>15}{'cal_net':>10}{'cal_trades':>11}"
    print(hdr)
    for thr in thresholds:
        c = cal[cal.threshold_bps == thr].iloc[0]
        u = uncal[uncal.threshold_bps == thr]
        us = float(u["sortino"].iloc[0]) if len(u) else float("nan")
        print(f"{thr:>5.0f}{c['sortino']:>13.2f}{us:>15.2f}"
              f"{c['net_return_sum']:>+10.4f}{int(c['trade_count']):>11d}")
    best = cal.sort_values("sortino", ascending=False).iloc[0]
    print(f"\ncalibrated ensemble4 best: {best['threshold_bps']:.0f} bps "
          f"Sortino {best['sortino']:+.2f} net {best['net_return_sum']:+.4f} "
          f"trades {int(best['trade_count'])}")
    print("best credible single: catboost_balanced @ 35 bps Sortino +3.39 (48 trades)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
