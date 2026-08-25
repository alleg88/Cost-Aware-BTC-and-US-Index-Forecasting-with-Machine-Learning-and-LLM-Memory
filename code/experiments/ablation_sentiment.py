"""Ablation: does the news-sentiment layer improve short-horizon direction prediction?

Trains the same class-balanced CatBoost under the same BlockingTimeSeriesSplit twice — on
price features only, then on price + leak-free news-sentiment features — and reports the
macro-F1 delta. This is the core RQ3 evidence.

Run:  python ablation_sentiment.py --instrument btc
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from features.build import FEATURE_COLS, add_features, make_label
from features.sentiment import assemble_sentiment_features, build_llm_features
from evaluation.splits import BlockingTimeSeriesSplit
from models.zoo import make_catboost, run_cv

CODE_ROOT = Path(__file__).resolve().parents[1]   # .../code
CONFIG = CODE_ROOT / "configs" / "default.yaml"


def build_xy(instrument: str, cfg: dict, horizon: int = 1, sentiment: str = "classic"):
    """Return (X_price, X_price_plus_sentiment, y) sharing one valid-row mask.

    sentiment: "classic" = DeBERTa tone + GDELT tone + F&G + macro pulse;
               "llm"     = structured LLM features (relevance-weighted / high-impact);
               "both"    = classic + llm.
    """
    inst = cfg["instruments"][instrument]
    df = pd.read_parquet(CODE_ROOT / inst["working_parquet"])
    feat = add_features(df)
    y = make_label(feat, threshold_bps=inst["threshold_bps"], horizon=horizon)

    parts = []
    if sentiment in ("classic", "both"):
        parts.append(assemble_sentiment_features(instrument, df.index))
    if sentiment in ("llm", "both"):
        parts.append(build_llm_features(instrument, df.index))
    sent = pd.concat(parts, axis=1)                            # all leak-free, bar-aligned
    X_price = feat[FEATURE_COLS]
    X_full = X_price.join(sent)

    valid = X_price.notna().all(axis=1) & (y != -1)   # sentiment cols are NaN-free (0-filled)
    return X_price[valid], X_full[valid], y[valid]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    from experiments.horizons import DEFAULT_LABEL, parse_horizon
    ap.add_argument("--horizon", type=parse_horizon, default=DEFAULT_LABEL,
                    help="m15 (default) / h1 / h4, or a bar count")
    ap.add_argument("--sentiment", default="classic", choices=["classic", "llm", "both"],
                    help="which sentiment block joins the price features")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text())
    sp = cfg["split"]
    params = cfg["model"]["catboost"]

    X_price, X_full, y = build_xy(args.instrument, cfg, args.horizon, args.sentiment)
    # embargo must cover the label horizon (an h-bar label reaches h bars past its row)
    splitter = BlockingTimeSeriesSplit(sp["n_splits"], sp["train_frac"],
                                       max(sp["embargo_bars"], args.horizon))

    n_sent = X_full.shape[1] - X_price.shape[1]
    print(f"[{args.instrument}] horizon={args.horizon} bar(s) | sentiment={args.sentiment} | "
          f"{len(y):,} rows | price feats {len(FEATURE_COLS)} | +sentiment feats {n_sent} | "
          f"label dist {dict((y.value_counts(normalize=True) * 100).round(1))}")

    res_price = run_cv(make_catboost, X_price, y, splitter, params)
    res_full = run_cv(make_catboost, X_full, y, splitter, params)

    d = res_full["mean_macro_f1"] - res_price["mean_macro_f1"]
    label = f"price + {args.sentiment}"
    print("\n                     mean macro-F1   (per fold)")
    print(f"  price only       : {res_price['mean_macro_f1']:.4f} "
          f"± {res_price['std_macro_f1']:.4f}   {[round(f,3) for f in res_price['fold_macro_f1']]}")
    print(f"  {label:17s}: {res_full['mean_macro_f1']:.4f} "
          f"± {res_full['std_macro_f1']:.4f}   {[round(f,3) for f in res_full['fold_macro_f1']]}")
    print(f"  delta            : {d:+.4f}  ({'sentiment helps' if d > 0 else 'no gain'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
