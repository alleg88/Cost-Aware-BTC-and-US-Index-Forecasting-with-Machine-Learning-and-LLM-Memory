"""RQ1 model-zoo comparison: every registry model under the same blocking CV.

Runs each model family (classical / boosting / deep / stacks) on identical features,
labels, and BlockingTimeSeriesSplit folds, restricted to the configured train period
(2024) so the 2025 walk-forward year stays untouched by model selection.

Each model's fold metrics are cached as JSON, so interrupted runs resume where they
stopped and the notebook reads results without retraining.

Run:  python -m experiments.run_model_zoo --instrument btc --sentiment both
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.splits import BlockingTimeSeriesSplit
from experiments.ablation_sentiment import build_xy
from experiments.horizons import DEFAULT_LABEL, horizon_label, parse_horizon
from models.zoo import MODELS, run_cv

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
CACHE_DIR = CODE_ROOT / "experiments" / "cache" / "model_zoo"


def load_xy(instrument: str, sentiment: str, horizon: int, cfg: dict):
    """Features/labels for the configured TRAIN period only."""
    X_price, X_full, y = build_xy(instrument, cfg, horizon, "both" if sentiment == "none" else sentiment)
    X = X_price if sentiment == "none" else X_full
    start, end = cfg["dates"]["train"]
    mask = (X.index >= pd.Timestamp(start, tz="UTC")) & (
        X.index <= pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    )
    return X[mask], y[mask]


def cache_path(instrument: str, sentiment: str, horizon: int, model: str,
               tuned: bool = False) -> Path:
    suffix = "_tuned" if tuned else ""
    label = horizon_label(horizon)
    return CACHE_DIR / f"{instrument}_{sentiment}_{label}_{model}{suffix}.json"


def run_one(name: str, X, y, splitter, *, force: bool = False,
            instrument: str, sentiment: str, horizon: int,
            params: dict | None = None, tuned: bool = False) -> dict:
    path = cache_path(instrument, sentiment, horizon, name, tuned)
    if path.exists() and not force:
        return json.loads(path.read_text(encoding="utf-8"))

    started = time.time()
    result = run_cv(MODELS[name], X, y, splitter, params)
    payload = {
        "model": name,
        "instrument": instrument,
        "sentiment": sentiment,
        "horizon": horizon,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "elapsed_s": round(time.time() - started, 1),
        "mean_macro_f1": result["mean_macro_f1"],
        "std_macro_f1": result["std_macro_f1"],
        "mean_balanced_accuracy": result["mean_balanced_accuracy"],
        "fold_macro_f1": result["fold_macro_f1"],
        "fold_balanced_accuracy": result["fold_balanced_accuracy"],
        "confusion_matrix": np.asarray(result["confusion_matrix"]).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def results_table(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([
        {
            "model": r["model"],
            "mean_macro_f1": r["mean_macro_f1"],
            "std_macro_f1": r["std_macro_f1"],
            "mean_balanced_accuracy": r["mean_balanced_accuracy"],
            "elapsed_s": r.get("elapsed_s"),
        }
        for r in rows
    ])
    return frame.sort_values("mean_macro_f1", ascending=False).reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    ap.add_argument("--horizon", type=parse_horizon, default=DEFAULT_LABEL,
                    help="m15 (default) / h1 / h4, or a bar count")
    ap.add_argument("--models", default="all",
                    help="comma-separated registry names, or 'all'")
    ap.add_argument("--force", action="store_true", help="retrain even if cached")
    ap.add_argument("--tuned", action="store_true",
                    help="use Optuna-tuned params from experiments/cache/tuning/ where present")
    args = ap.parse_args()

    names = list(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    unknown = [m for m in names if m not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}; available: {list(MODELS)}")

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sp = cfg["split"]
    # the embargo must cover the label horizon, or an h-bar label at the end of a
    # train segment would overlap the first test bars
    embargo = max(sp["embargo_bars"], args.horizon)
    splitter = BlockingTimeSeriesSplit(sp["n_splits"], sp["train_frac"], embargo)
    X, y = load_xy(args.instrument, args.sentiment, args.horizon, cfg)
    print(f"[{args.instrument}] sentiment={args.sentiment} horizon={horizon_label(args.horizon)} | "
          f"{len(y):,} rows x {X.shape[1]} features | models: {len(names)}")

    rows = []
    for name in names:
        params = None
        if args.tuned:
            from experiments.run_tuning import load_tuned_params

            params = load_tuned_params(args.instrument, args.sentiment, args.horizon, name)
        tag = " (tuned)" if params else ""
        print(f"  {name}{tag} ...", flush=True)
        payload = run_one(name, X, y, splitter, force=args.force,
                          instrument=args.instrument, sentiment=args.sentiment,
                          horizon=args.horizon, params=params,
                          tuned=args.tuned and params is not None)
        rows.append(payload)
        print(f"    macro-F1 {payload['mean_macro_f1']:.4f} ± {payload['std_macro_f1']:.4f} "
              f"({payload['elapsed_s']}s)")

    # rebuild the table from every cached result, not just this invocation's subset;
    # a tuned cache supersedes the default-params cache for the same model
    cached = []
    for name in MODELS:
        for tuned in (True, False):
            p = cache_path(args.instrument, args.sentiment, args.horizon, name, tuned)
            if p.exists():
                entry = json.loads(p.read_text(encoding="utf-8"))
                entry["model"] = f"{name} (tuned)" if tuned else name
                cached.append(entry)
                break
    table = results_table(cached)
    out = CACHE_DIR / f"{args.instrument}_{args.sentiment}_{horizon_label(args.horizon)}_table.parquet"
    table.to_parquet(out, index=False)
    print(f"\n{table.to_string(index=False)}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
