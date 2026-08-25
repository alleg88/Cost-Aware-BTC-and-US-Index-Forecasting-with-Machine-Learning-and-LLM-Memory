"""Label-threshold sweep with economics-first evaluation (notebook 03).

Retrains the walk-forward contenders at several label thresholds and reports
net-of-cost economics for each, because the dead-zone width is a POLICY choice,
not a fixed fact: wider thresholds mean fewer, larger signals (more flat bars)
and the right width is the one that maximises risk-adjusted return, not
macro-F1. Macro-F1 is NOT comparable across thresholds (the class balance
changes with the threshold); economics is the comparable axis.

Two label kinds:
  deadzone  close-to-close dead-zone label at each threshold (horizon = 1 bar)
  barrier   triple-barrier first-touch label (up = down = threshold,
            time-out after --max-hold bars)

For every (kind, threshold, model) the walk-forward retrains weekly over 2025
with the already-tuned hyperparameters (tuning is NOT repeated per threshold —
a stated limitation), the confidence gate tau is calibrated on Q1-2025 by the
--select-metric (default Sortino), frozen, and evaluated on Q2-Q4. The
ensemble4 soft-vote is built separately (see experiments.run_ensemble4).

Caches are resumable: existing walk-forward parquets are reused, so the sweep
can be re-run after an interruption without repeating finished work.

Run:  python -m experiments.run_threshold_sweep --label-kind deadzone
      python -m experiments.run_threshold_sweep --label-kind barrier --max-hold 16
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from evaluation.economics import economics_summary, strategy_returns, sweep_tau
from evaluation.metrics import classification_scores
from experiments.run_walkforward import build_walkforward_xy, resolve_params
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START
from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import (
    FEATURE_COLS, ORDERFLOW_FEATURE_COLS, make_barrier_label, make_label,
)
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WALKFORWARD_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"

# The ensemble-candidate pool: every model with an Optuna space + cached tuned
# params. The sweep ranks these so the best few can seed the ensemble.
CANDIDATE_MODELS = ("catboost_balanced", "xgboost_balanced", "random_forest",
                    "gru", "lstm", "mlp")
# The tradeable end of notebook 01's dead-zone stack (from 25 bps up): tighter
# zones sit under the ~10 bps round-trip cost and are not worth modelling. 25
# bps = the m15 default label (reuses those caches).
THRESHOLDS = (25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 60.0, 75.0, 80.0, 85.0, 90.0)
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
# CALIBRATION_END now comes from experiments.spans (config-driven, unchanged Q1-2025)


def sweep_tag(kind: str, threshold: float, max_hold: int) -> str:
    """Cache-name tag for one sweep cell (fills the horizon slot of the name).

    The 25 bps dead-zone is exactly the default `m15` label, so it reuses the
    canonical m15 caches instead of minting a redundant dz25 set.
    """
    if kind == "deadzone" and threshold == 25.0:
        return "m15"
    thr = f"{threshold:g}".replace(".", "p")
    if kind == "deadzone":
        return f"dz{thr}"
    return f"tb{thr}x{thr}h{max_hold}"


def cache_path(instrument: str, sentiment: str, model: str, tag: str) -> Path:
    return WALKFORWARD_DIR / f"{instrument}_{sentiment}_{model}_{tag}_{CACHE_SUFFIX}.parquet"


def label_fn_for(kind: str, threshold: float, max_hold: int):
    if kind == "deadzone":
        return lambda feat: make_label(feat, threshold_bps=threshold, horizon=1)
    return lambda feat: make_barrier_label(
        feat, up_bps=threshold, down_bps=threshold, max_hold=max_hold)


def run_walkforward_cell(
    instrument: str, sentiment: str, sentiment_tag: str, model: str, tag: str, *,
    label_fn, orderflow: bool, train_tail_trim: int, cfg: dict,
    limit_windows: int | None = None,
) -> Path:
    """One (threshold, model) walk-forward; skipped when its cache exists.

    `sentiment` drives the sentiment-feature join and tuned-param lookup;
    `sentiment_tag` (e.g. "bothof") names the cache. Tuned params are always
    loaded under the base `sentiment` — order-flow reuses them (no per-feature
    re-tuning), a stated limitation.
    """
    out = cache_path(instrument, sentiment_tag, model, tag)
    if out.exists():
        print(f"  [cached] {out.name}")
        return out

    X, y, aux = build_walkforward_xy(
        instrument, cfg, horizon=1, sentiment=sentiment, label_fn=label_fn,
        orderflow=orderflow)
    walk_start, walk_end = cfg["dates"]["walkforward"]
    windows = weekly_walkforward_windows(
        X.index, walk_start=walk_start, walk_end=walk_end, train_lookback="180D")
    if limit_windows is not None:
        windows = windows[:limit_windows]

    params = resolve_params(model, instrument, sentiment, 1, cfg)
    preds = run_walkforward_predictions(
        X, y,
        windows=windows,
        model_factory=MODELS[model],
        model_name=model,
        params=params,
        min_train_rows=500,
        min_validation_rows=50,
        include_features=True,
        progress_label=f"{tag}:{model}",
        train_tail_trim=train_tail_trim,
    )
    if preds.empty:
        raise RuntimeError(f"no predictions for {tag}:{model}")
    preds = preds.join(aux[["forward_return", "vol_regime"]], how="left")
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out)
    print(f"  wrote {len(preds):,} rows -> {out.name}")
    return out


def calibrated_economics(
    df: pd.DataFrame, model: str, fee_bps: float, *,
    select_metric: str = "sortino", slippage_bps: float = 0.0,
) -> dict:
    """Calibrate tau on Q1 by `select_metric`, freeze, evaluate Q2-Q4."""
    cost_bps = float(fee_bps) + float(slippage_bps)
    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    lockbox = pd.Timestamp(LOCKBOX_START, tz="UTC")
    # evaluation is [calibration_end, lockbox_start): sealed lockbox never scored here
    calib = df[df.index < split]
    evaluation = df[(df.index >= split) & (df.index < lockbox)]
    if calib.empty or evaluation.empty:
        raise ValueError("calibration or evaluation period is empty")

    sweep = sweep_tau(calib[f"{model}_pred"], calib[f"{model}_conf"],
                      calib["forward_return"], cost_bps, TAUS)
    best = sweep.loc[sweep[select_metric].idxmax()]
    best_tau = float(best["tau"])

    eval_returns = strategy_returns(
        evaluation[f"{model}_pred"], evaluation["forward_return"], cost_bps,
        evaluation[f"{model}_conf"], best_tau)
    summary = economics_summary(
        eval_returns, evaluation[f"{model}_pred"], evaluation[f"{model}_conf"], best_tau)
    summary.update({
        "model": model,
        "select_metric": select_metric,
        "slippage_bps": float(slippage_bps),
        f"calibration_{select_metric}": float(best[select_metric]),
    })
    # Full score + shape bundle. Economics (Sortino/Sharpe) is the headline and
    # the calibration criterion; classification scores are the secondary check
    # (ungated, evaluation period) and are NOT comparable across thresholds
    # because the class balance shifts with the label — but valid within one.
    if "y_true" in evaluation.columns:
        yt = evaluation["y_true"].astype(int)
        yp = evaluation[f"{model}_pred"].astype(int)
        cls = classification_scores(yt, yp)
        summary["macro_f1"] = float(cls["macro_f1"])
        summary["balanced_accuracy"] = float(cls["balanced_accuracy"])
        for name, f1 in cls["per_class_f1"].items():
            summary[f"f1_{name}"] = float(f1)
        vc = yt.value_counts()
        summary["n_eval_rows"] = int(len(yt))               # shape
        summary["n_down"] = int(vc.get(0, 0))
        summary["n_flat"] = int(vc.get(1, 0))
        summary["n_up"] = int(vc.get(2, 0))
        summary["flat_share"] = float(vc.get(1, 0) / len(yt)) if len(yt) else 0.0
    # feature-set shape: how many predictor columns the model actually saw
    feat_cols = [c for c in df.columns if c in set(FEATURE_COLS) | set(ORDERFLOW_FEATURE_COLS)]
    summary["n_features"] = int(len(feat_cols))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    ap.add_argument("--label-kind", default="deadzone", choices=["deadzone", "barrier"])
    ap.add_argument("--thresholds", default=",".join(f"{t:g}" for t in THRESHOLDS))
    ap.add_argument("--models", default=",".join(CANDIDATE_MODELS),
                    help="ensemble-candidate pool to rank (default: the 6 tuned models)")
    ap.add_argument("--max-hold", type=int, default=16,
                    help="barrier label time-out, in bars (barrier kind only)")
    ap.add_argument("--select-metric", default="sortino", choices=["sortino", "sharpe"])
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--price-only", action="store_true",
                    help="drop the order-flow block (price+sentiment ablation)")
    ap.add_argument("--limit-windows", type=int, default=None,
                    help="smoke-test with the first N weekly windows only")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee_bps = float(cfg["instruments"][args.instrument]["taker_fee_bps"])
    thresholds = [float(t) for t in args.thresholds.split(",")]
    models = [m.strip() for m in args.models.split(",")]
    trim = 1 if args.label_kind == "deadzone" else args.max_hold
    orderflow = not args.price_only               # order-flow is the default feature set
    sentiment_tag = args.sentiment + ("" if args.price_only else "of")

    rows = []
    for thr in thresholds:
        tag = sweep_tag(args.label_kind, thr, args.max_hold)
        print(f"[{tag}] walk-forward ({len(models)} model(s), "
              f"features={'price+sentiment' if args.price_only else 'price+sentiment+orderflow'})")
        label_fn = label_fn_for(args.label_kind, thr, args.max_hold)
        for model in models:
            run_walkforward_cell(
                args.instrument, args.sentiment, sentiment_tag, model, tag,
                label_fn=label_fn, orderflow=orderflow, train_tail_trim=trim, cfg=cfg,
                limit_windows=args.limit_windows)

        eval_models = list(models)
        for model in eval_models:
            df = pd.read_parquet(cache_path(args.instrument, sentiment_tag, model, tag))
            df.index = pd.to_datetime(df.index, utc=True)
            try:
                summary = calibrated_economics(
                    df.sort_index(), model, fee_bps,
                    select_metric=args.select_metric, slippage_bps=args.slippage_bps)
            except ValueError as exc:   # e.g. smoke runs that stop before Q2
                print(f"  [skip economics] {model}: {exc}")
                continue
            summary.update({"label_kind": args.label_kind, "threshold_bps": thr,
                            "max_hold": args.max_hold if args.label_kind == "barrier" else 1,
                            "tag": tag})
            rows.append(summary)
            print(f"  {model}: tau={summary['tau']:.2f} "
                  f"eval Sortino {summary['sortino']:+.2f} Sharpe {summary['sharpe']:+.2f} "
                  f"net {summary['net_return_sum']:+.4f} trades {summary['trade_count']:,}")

    table = pd.DataFrame(rows)
    if table.empty:
        print("\nno economics rows produced (smoke run?) — nothing written")
        return 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (f"{args.instrument}_{sentiment_tag}_sweep_{args.label_kind}"
                     f"_{args.select_metric}_{CACHE_SUFFIX}.parquet")
    table.to_parquet(out, index=False)
    print(f"\nwrote {len(table)} sweep rows -> {out}")
    cols = ["tag", "model", "threshold_bps", "tau",
            "sortino", "sharpe", "net_return_sum", "max_drawdown",
            "macro_f1", "balanced_accuracy", "f1_down", "f1_up",
            "trade_count", "exposure", "n_eval_rows", "flat_share", "n_features"]
    cols = [c for c in cols if c in table.columns]
    # rank models by MEAN eval Sortino AND mean macro-F1 across the full
    # threshold stack — the aggregate that names the ensemble seeds
    ranking = table.groupby("model").agg(
        mean_sortino=("sortino", "mean"), mean_sharpe=("sharpe", "mean"),
        mean_macro_f1=("macro_f1", "mean"), best_sortino=("sortino", "max"),
    ).sort_values("mean_sortino", ascending=False)
    print(f"\nmodel ranking across thresholds:\n{ranking.round(4).to_string()}")
    print(f"\nall sweep rows by Sortino:")
    print(table[cols].sort_values("sortino", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
