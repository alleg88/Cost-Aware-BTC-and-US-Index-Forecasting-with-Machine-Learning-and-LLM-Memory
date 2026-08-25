"""Meta-labeling on the stacked-ensemble primary signal (notebook 03).

The stacking meta-learner (run_ensemble_stack) is a level-2 COMBINER; this is a
different second model on top of it -- a Lopez de Prado meta-label that decides
whether to ACT on the ensemble's directional call. It reuses the single-model
meta-labeling machinery (experiments.meta_labeling) with the ensemble stack cache
as the primary, so the direction comes from the ensemble and a walk-forward
logistic win-probability model filters out the bracket trades likely to lose.

The stack cache stores `_pred` as the directional lean (never flat) and `_conf`
as the directional conviction, giving a dense candidate set; the candidate gate
`--tau`, the bracket (tp, sl, max_hold) and the filter threshold theta then do
the selecting -- theta calibrated on Q1-2025 by Sortino, frozen, evaluated Q2-Q4
against the unfiltered bracket strategy (Diebold-Mariano on per-bar returns).

Run:  python -m experiments.run_ensemble_meta --tag dz40 --tau 0.40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import diebold_mariano, economics_summary
from evaluation.trades import trade_stats
from experiments.meta_labeling import (
    CALIBRATION_END, MIN_TRAIN_CANDIDATES, THETA_GRID, build_candidates,
    filtered_strategy, walkforward_pwin,
)
from experiments.run_ensemble_stack import STACK_TAG
from experiments.run_threshold_sweep import cache_path

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"


def load_stack_inputs(instrument: str, tag: str, cfg: dict):
    """Bars over the prediction span + the ensemble stack prediction cache."""
    bars = pd.read_parquet(CODE_ROOT / cfg["instruments"][instrument]["working_parquet"])
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()
    preds = pd.read_parquet(cache_path(instrument, "bothof", STACK_TAG, tag))
    preds.index = pd.to_datetime(preds.index, utc=True)
    preds = preds.sort_index()
    bars = bars.loc[preds.index.min():preds.index.max()]
    return bars, preds


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc")
    ap.add_argument("--tag", default="dz40", help="stack cache tag (dz35/dz40/dz45)")
    ap.add_argument("--tau", type=float, default=0.40,
                    help="directional-conviction gate for candidate signals")
    ap.add_argument("--tp-bps", type=float, default=50.0)
    ap.add_argument("--sl-bps", type=float, default=50.0)
    ap.add_argument("--max-hold", type=int, default=16)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--select-metric", default="sortino", choices=["sortino", "sharpe"])
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee_bps = float(cfg["instruments"][args.instrument]["taker_fee_bps"])
    bars, preds = load_stack_inputs(args.instrument, args.tag, cfg)

    candidates = build_candidates(
        bars, preds, STACK_TAG, tau=args.tau, tp_bps=args.tp_bps,
        sl_bps=args.sl_bps, max_hold=args.max_hold,
        fee_bps=fee_bps, slippage_bps=args.slippage_bps)
    if candidates.empty:
        raise SystemExit("no candidate trades at this gate -- loosen --tau")
    candidates["p_win"] = walkforward_pwin(candidates)

    from experiments.spans import LOCKBOX_START
    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    lockbox = pd.Timestamp(LOCKBOX_START, tz="UTC")
    n_q1 = int((candidates.index < split).sum())
    print(f"{len(candidates)} candidates ({n_q1} in Q1), "
          f"base win rate {candidates['win'].mean():.1%}")
    if n_q1 < MIN_TRAIN_CANDIDATES:
        print(f"WARNING: only {n_q1} Q1 candidates -- theta calibration is fragile")

    bars_cal = bars[bars.index < split]
    bars_eval = bars[(bars.index >= split) & (bars.index < lockbox)]

    # ---- calibrate theta on Q1 (theta=0 = unfiltered reference) ------------
    cal_rows = []
    for theta in (0.0, *THETA_GRID):
        _, per_bar = filtered_strategy(
            bars_cal, preds[preds.index < split], STACK_TAG,
            candidates["p_win"], tau=args.tau, theta=theta,
            tp_bps=args.tp_bps, sl_bps=args.sl_bps, max_hold=args.max_hold,
            fee_bps=fee_bps, slippage_bps=args.slippage_bps)
        row = economics_summary(per_bar); row["theta"] = theta
        cal_rows.append(row)
    cal_table = pd.DataFrame(cal_rows)
    best_theta = float(cal_table.loc[cal_table[args.select_metric].idxmax(), "theta"])
    print(f"theta* = {best_theta:.2f} "
          f"(Q1 {args.select_metric} {cal_table[args.select_metric].max():+.2f})")

    # ---- frozen evaluation Q2-Q4 -------------------------------------------
    results = {}
    for name, theta in (("unfiltered", 0.0), ("meta_filtered", best_theta)):
        ledger, per_bar = filtered_strategy(
            bars_eval, preds[preds.index >= split], STACK_TAG,
            candidates["p_win"], tau=args.tau, theta=theta,
            tp_bps=args.tp_bps, sl_bps=args.sl_bps, max_hold=args.max_hold,
            fee_bps=fee_bps, slippage_bps=args.slippage_bps)
        summary = economics_summary(per_bar)
        summary.update(trade_stats(ledger))
        summary.update({"variant": name, "theta": theta, "tau": args.tau,
                        "primary": STACK_TAG, "tag": args.tag})
        results[name] = (summary, ledger, per_bar)

    dm = diebold_mariano(results["unfiltered"][2], results["meta_filtered"][2])

    eval_cands = candidates[candidates.index >= split]
    quality = {"variant": "filter_quality", "primary": STACK_TAG, "tag": args.tag,
               "theta": best_theta}
    scored = eval_cands[eval_cands["p_win"] != 0.5]
    if len(scored) >= 20 and scored["win"].nunique() == 2:
        from sklearn.metrics import roc_auc_score
        quality["auc"] = float(roc_auc_score(scored["win"], scored["p_win"]))
    taken = eval_cands[eval_cands["p_win"] >= best_theta]
    skipped = eval_cands[eval_cands["p_win"] < best_theta]
    quality.update({
        "taken_win_rate": float(taken["win"].mean()) if len(taken) else np.nan,
        "skipped_win_rate": float(skipped["win"].mean()) if len(skipped) else np.nan,
        "taken_expectancy_bps": float(taken["net_return"].mean() * 1e4) if len(taken) else np.nan,
        "n_taken": int(len(taken)), "n_skipped": int(len(skipped)),
    })

    table = pd.DataFrame([results["unfiltered"][0], results["meta_filtered"][0], quality])
    table.loc[table["variant"] == "meta_filtered", "dm_stat"] = dm["dm_stat"]
    table.loc[table["variant"] == "meta_filtered", "dm_p_value"] = dm["p_value"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.instrument}_bothof_{STACK_TAG}_{args.tag}_meta"
    table.to_parquet(OUT_DIR / f"{stem}_summary.parquet", index=False)
    cal_table.to_parquet(OUT_DIR / f"{stem}_theta_grid.parquet", index=False)
    candidates.reset_index().to_parquet(OUT_DIR / f"{stem}_candidates.parquet", index=False)

    cols = ["variant", "theta", "sortino", "sharpe", "net_return_sum", "n_trades",
            "win_rate", "expectancy_bps", "auc", "taken_win_rate", "skipped_win_rate"]
    have = [c for c in cols if c in table.columns]
    print(f"\n{table[have].to_string(index=False)}")
    print(f"\nmeta filter vs unfiltered: DM {dm['dm_stat']:+.2f} (p={dm['p_value']:.3f}, "
          f"positive favours the filter)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
