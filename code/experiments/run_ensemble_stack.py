"""Stacking meta-learner over the four calibrated base models (notebook 03).

The deployed ensemble combines the sweep-selected seeds (gru, catboost,
random_forest, mlp). The default combiner is a flat soft-vote ("trust all four
equally"). This module asks whether a LEARNED combiner does better: a level-2
multinomial logistic regression that reads the four bases' calibrated class
probabilities (12 features) and learns how much to trust each base per class.

Why logistic and not another boosted tree: the level-0 models already did the
nonlinear work and their outputs are isotonic-calibrated, so the residual job is
close to linear; a heavy meta-learner over 12 features and thin weekly windows
overfits (the CatBoost-meta stacks in notebook 02 lose to their best base). L2
logistic is the minimal principled upgrade, and its softmax output keeps the
directional gate and confidence sizing well-behaved.

Leak-free walk-forward stacking: the level-1 training rows are the bases' own
out-of-fold validation predictions. For validation window w the meta trains only
on rows from windows strictly before w, so no future information reaches it (the
same past-only discipline as BlockingTimeSeriesSplit).

Economics use the DIRECTIONAL-probability gate, not argmax: isotonic pushes the
directional probabilities below the flat class, so argmax collapses to flat. The
gate trades long when P(up) >= tau and P(up) > P(down) (short mirrored), tau
calibrated on Q1-2025 by Sortino, frozen, evaluated Q2-Q4. Both flat unit sizing
and confidence-weighted sizing are reported.

Also writes an ensemble prediction cache in the {model}_pred/_conf/_p* format so
the bracket and meta-labeling layers can use the stack as their primary signal.

Run:  python -m experiments.run_ensemble_stack --thresholds 35,40,45
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.economics import (
    BARS_PER_YEAR_M15, confidence_weights, economics_summary,
)
from evaluation.metrics import classification_scores
from experiments.run_ensemble4 import ENSEMBLE4_MODELS
from experiments.run_threshold_sweep import cache_path, sweep_tag
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WALKFORWARD_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
ECON = CODE_ROOT / "experiments" / "cache" / "economics"

STACK_TAG = "ensemble4stack"
# Directional-probability gate grid: calibrated directional probs live well
# below the max-prob range, so the useful thresholds are low.
DIR_TAUS = (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
MIN_STACK_TRAIN = 300


def load_calibrated_bases(instrument: str, tag: str,
                          models=ENSEMBLE4_MODELS) -> dict[str, pd.DataFrame]:
    """The four {model}_cal walk-forward caches at one threshold, index-aligned."""
    frames = {}
    for m in models:
        p = cache_path(instrument, "bothof", f"{m}_cal", tag)
        d = pd.read_parquet(p)
        d.index = pd.to_datetime(d.index, utc=True)
        frames[m] = d.sort_index()
    idx0 = frames[models[0]].index
    for m in models:
        if not frames[m].index.equals(idx0):
            raise ValueError(f"{m}_cal index differs at {tag}")
    return frames


def stack_probabilities(frames: dict[str, pd.DataFrame],
                        models=ENSEMBLE4_MODELS) -> pd.DataFrame:
    """Walk-forward multinomial-logistic stack over the bases' calibrated probs.

    Returns a frame with p0/p1/p2 (meta probabilities), forward_return and
    y_true. Early windows without enough history keep the soft-vote as a
    fallback so the series is complete.
    """
    first = frames[models[0]]
    feat_cols, blocks = [], []
    for m in models:
        cols = [f"{m}_cal_p{i}" for i in range(3)]
        blocks.append(frames[m][cols].to_numpy(dtype=float))
        feat_cols += cols
    X = np.hstack(blocks)                                   # (n, 12)
    y = first["y_true"].to_numpy(dtype=int)
    windows = first["window"].to_numpy()
    soft = np.mean([b for b in blocks], axis=0)             # (n, 3) fallback

    order = np.array(sorted(pd.unique(windows)))
    out = np.zeros((len(X), 3), dtype=float)
    for w in order:
        cur = windows == w
        past = windows < w
        y_past = y[past]
        if past.sum() < MIN_STACK_TRAIN or len(np.unique(y_past)) < 2:
            out[cur] = soft[cur]                            # not enough history yet
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[past], y_past)
        proba = clf.predict_proba(X[cur])
        aligned = np.zeros((cur.sum(), 3))
        for j, cls in enumerate(clf.named_steps["logisticregression"].classes_):
            aligned[:, int(cls)] = proba[:, j]
        out[cur] = aligned

    res = pd.DataFrame(out, index=first.index, columns=["p0", "p1", "p2"])
    res["forward_return"] = first["forward_return"].to_numpy(dtype=float)
    res["y_true"] = y
    return res


def directional_positions(p_down: pd.Series, p_up: pd.Series, tau: float) -> pd.Series:
    """+1 when P(up)>=tau and P(up)>P(down); -1 mirrored; else flat."""
    long = (p_up >= tau) & (p_up > p_down)
    short = (p_down >= tau) & (p_down > p_up)
    return pd.Series(np.where(long, 1.0, np.where(short, -1.0, 0.0)),
                     index=p_up.index)


def directional_returns(df: pd.DataFrame, tau: float, fee_bps: float, *,
                        sized: bool = False) -> tuple[pd.Series, int]:
    """Per-bar net returns of the directional gate; optional confidence sizing."""
    pos = directional_positions(df["p0"], df["p2"], tau)
    if sized:
        conf = df[["p0", "p2"]].max(axis=1)                # directional conviction
        pos = pos * confidence_weights(conf, tau, cap=1.0, floor=0.0)
    turn = (pos - pos.shift(1, fill_value=0.0)).abs()
    r = pos * df["forward_return"].fillna(0.0) - turn * (float(fee_bps) / 1e4)
    return r, int((turn > 0).sum())


def gated_economics(df: pd.DataFrame, fee_bps: float, *, sized: bool) -> dict:
    """Calibrate tau on Q1 by Sortino, freeze, evaluate Q2-Q4 (directional gate)."""
    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    lockbox = pd.Timestamp(LOCKBOX_START, tz="UTC")
    cal = df[df.index < split]
    ev = df[(df.index >= split) & (df.index < lockbox)]
    best_tau, best_s = None, -np.inf
    for tau in DIR_TAUS:
        r, n = directional_returns(cal, tau, fee_bps, sized=sized)
        s = economics_summary(r)["sortino"]
        if n >= 4 and s > best_s:
            best_s, best_tau = s, tau
    if best_tau is None:
        best_tau = DIR_TAUS[len(DIR_TAUS) // 2]
    r, n = directional_returns(ev, best_tau, fee_bps, sized=sized)
    summ = economics_summary(r)
    pos = directional_positions(ev["p0"], ev["p2"], best_tau)
    yp = pos.map({-1.0: 0, 0.0: 1, 1.0: 2}).astype(int)
    cls = classification_scores(ev["y_true"].astype(int), yp)
    summ.update({"tau": best_tau, "trade_count": n,
                 "calibration_sortino": best_s,
                 "macro_f1": float(cls["macro_f1"]),
                 "sizing": "confidence" if sized else "flat"})
    return summ


def soft_vote_frame(frames: dict[str, pd.DataFrame],
                    models=ENSEMBLE4_MODELS) -> pd.DataFrame:
    """Flat soft-vote of the calibrated bases in the same p0/p1/p2 layout."""
    first = frames[models[0]]
    total = np.zeros((len(first), 3))
    for m in models:
        total += frames[m][[f"{m}_cal_p{i}" for i in range(3)]].to_numpy(dtype=float)
    total /= len(models)
    res = pd.DataFrame(total, index=first.index, columns=["p0", "p1", "p2"])
    res["forward_return"] = first["forward_return"].to_numpy(dtype=float)
    res["y_true"] = first["y_true"].to_numpy(dtype=int)
    return res


def write_primary_cache(stack: pd.DataFrame, instrument: str, tag: str,
                        base_frame: pd.DataFrame) -> Path:
    """Emit the stack as a {model}_* prediction cache for brackets/meta-labeling.

    `_pred` is the directional lean (2 if P(up)>P(down) else 0, never flat) and
    `_conf` the directional conviction, so the exit-discipline layers get a dense
    candidate set and let their own gates/theta do the filtering. Signal-time
    features are carried over from a base cache for the meta-labeling model.
    """
    m = STACK_TAG
    out = pd.DataFrame(index=stack.index)
    out[f"{m}_pred"] = np.where(stack["p2"] >= stack["p0"], 2, 0)
    out[f"{m}_conf"] = stack[["p0", "p2"]].max(axis=1).to_numpy()
    for i in range(3):
        out[f"{m}_p{i}"] = stack[f"p{i}"].to_numpy()
    out["y_true"] = stack["y_true"].to_numpy()
    out["forward_return"] = stack["forward_return"].to_numpy()
    carry = ["r1", "vol_10", "vol_20", "vol_60", "hl_range", "rsi_14", "vol_z",
             "hour", "dayofweek", "vol_regime"]
    for c in carry:
        if c in base_frame.columns:
            out[c] = base_frame[c].to_numpy()
    path = cache_path(instrument, "bothof", STACK_TAG, tag)
    out.to_parquet(path)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc")
    ap.add_argument("--thresholds", default="35,40,45")
    args = ap.parse_args()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"][args.instrument]["taker_fee_bps"])
    thresholds = [float(t) for t in args.thresholds.split(",")]

    rows = []
    for thr in thresholds:
        tag = sweep_tag("deadzone", thr, 16)
        print(f"[{tag}] stacking meta-learner over {ENSEMBLE4_MODELS}")
        frames = load_calibrated_bases(args.instrument, tag)
        stack = stack_probabilities(frames)
        soft = soft_vote_frame(frames)

        write_primary_cache(stack, args.instrument, tag, frames[ENSEMBLE4_MODELS[0]])

        for name, frame, sized in (
            ("soft_vote", soft, False),
            ("stack", stack, False),
            ("stack_sized", stack, True),
        ):
            s = gated_economics(frame, fee, sized=sized)
            s.update({"combiner": name, "threshold_bps": thr})
            rows.append(s)
            print(f"  {name:12s} tau*={s['tau']:.2f} "
                  f"Sortino {s['sortino']:+.2f} Sharpe {s['sharpe']:+.2f} "
                  f"F1 {s['macro_f1']:.3f} net {s['net_return_sum']:+.4f} "
                  f"trades {s['trade_count']}")

    table = pd.DataFrame(rows)
    ECON.mkdir(parents=True, exist_ok=True)
    table.to_parquet(ECON / f"btc_bothof_ensemble_stack_{CACHE_SUFFIX}.parquet", index=False)

    print("\n=== best per combiner (by eval Sortino) ===")
    for name in ("soft_vote", "stack", "stack_sized"):
        sub = table[table["combiner"] == name].sort_values("sortino", ascending=False)
        b = sub.iloc[0]
        print(f"  {name:12s} {b['threshold_bps']:.0f} bps  Sortino {b['sortino']:+.2f} "
              f"net {b['net_return_sum']:+.4f} trades {int(b['trade_count'])}")
    print("reference: catboost_balanced @ 35 bps Sortino +3.39 (48 trades, +3.7% net)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
