"""Bracket-exit economics: take-profit / stop-loss / time-out on top of a gate.

Converts a walk-forward prediction cache into DISCRETE trades via
evaluation.trades.simulate_bracket_trades: a gated signal opens a position at
the next bar's open and exits at the first of take-profit, stop-loss, or
time-out — the profit is banked at the target regardless of what price does
afterwards. This replaces the per-bar position-flipping policy with the exit
discipline a real trader would use, and asks whether it adds economic value.

Calibration protocol (mirrors the tau protocol of run_economics):
  1. Q1-2025: joint grid over (tau, tp, sl, max_hold), best by --select-metric
     (default Sortino). Optional stages, each with earlier choices frozen:
       --asymmetric   long and short sides get independent (tp, sl);
       --split-tau    long and short sides get independent gates.
  2. The full grid is written to disk (multiple-testing transparency).
  3. Q2-Q4: evaluated once with everything frozen, plus slippage sensitivity
     and a Diebold-Mariano test against the per-bar gated baseline.

Run:  python -m experiments.run_brackets --model gru --tag m15
      python -m experiments.run_brackets --model ensemble4 --tag m15 --asymmetric --split-tau
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd
import yaml

from evaluation.economics import diebold_mariano, economics_summary, strategy_returns, sweep_tau
from evaluation.trades import simulate_bracket_trades, trade_stats
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WALKFORWARD_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"

TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
TP_GRID = (25.0, 50.0, 75.0, 100.0)
SL_GRID = (25.0, 50.0, 75.0)
HOLD_GRID = (16, 32)
# CALIBRATION_END / LOCKBOX_START from experiments.spans (config-driven)
SLIPPAGE_SENSITIVITY = (0.0, 2.0, 5.0)


def load_inputs(instrument: str, sentiment: str, model: str, tag: str,
                cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (bars over the prediction span, prediction cache)."""
    bars = pd.read_parquet(CODE_ROOT / cfg["instruments"][instrument]["working_parquet"])
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()

    path = WALKFORWARD_DIR / f"{instrument}_{sentiment}_{model}_{tag}_{CACHE_SUFFIX}.parquet"
    preds = pd.read_parquet(path)
    preds.index = pd.to_datetime(preds.index, utc=True)
    preds = preds.sort_index()
    bars = bars.loc[preds.index.min():preds.index.max()]
    return bars, preds


def gate_per_side(pred: pd.Series, conf: pd.Series,
                  tau_long: float, tau_short: float) -> pd.Series:
    """Force below-gate signals to flat, with separate long/short gates."""
    out = pred.astype(int).copy()
    out[(out == 2) & (conf.astype(float) < tau_long)] = 1
    out[(out == 0) & (conf.astype(float) < tau_short)] = 1
    return out


def bracket_economics(bars: pd.DataFrame, pred: pd.Series, conf: pd.Series | None,
                      *, tau: float, tp: float, sl: float, hold: int,
                      fee_bps: float, slippage_bps: float,
                      tp_short: float | None = None, sl_short: float | None = None,
                      ) -> tuple[dict, pd.DataFrame, pd.Series]:
    ledger, per_bar = simulate_bracket_trades(
        bars, pred, conf, tau=tau, tp_bps=tp, sl_bps=sl, max_hold=hold,
        fee_bps=fee_bps, slippage_bps=slippage_bps,
        tp_bps_short=tp_short, sl_bps_short=sl_short)
    summary = economics_summary(per_bar)
    summary.update(trade_stats(ledger))
    summary["exposure"] = float((per_bar != 0.0).mean())
    return summary, ledger, per_bar


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both",
                    choices=["none", "classic", "llm", "both", "bothof"],
                    help="feature set / cache tag; 'bothof' = sentiment + order-flow")
    ap.add_argument("--model", default="gru")
    ap.add_argument("--tag", default="m15",
                    help="cache tag: m15, dz30, tb30x30h16, ... (the horizon slot)")
    ap.add_argument("--select-metric", default="sortino", choices=["sortino", "sharpe"])
    ap.add_argument("--slippage-bps", type=float, default=0.0,
                    help="slippage assumed during calibration and headline eval")
    ap.add_argument("--tp-grid", default=",".join(f"{v:g}" for v in TP_GRID),
                    help="take-profit grid in bps (add the label threshold for "
                         "label-matched brackets on tb* tags)")
    ap.add_argument("--sl-grid", default=",".join(f"{v:g}" for v in SL_GRID))
    ap.add_argument("--hold-grid", default=",".join(str(v) for v in HOLD_GRID))
    ap.add_argument("--asymmetric", action="store_true",
                    help="stage 2: independent (tp, sl) per side")
    ap.add_argument("--split-tau", action="store_true",
                    help="stage 3: independent confidence gates per side")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee_bps = float(cfg["instruments"][args.instrument]["taker_fee_bps"])
    tp_grid = tuple(float(v) for v in args.tp_grid.split(","))
    sl_grid = tuple(float(v) for v in args.sl_grid.split(","))
    hold_grid = tuple(int(v) for v in args.hold_grid.split(","))
    bars, preds = load_inputs(args.instrument, args.sentiment, args.model, args.tag, cfg)
    pred = preds[f"{args.model}_pred"]
    conf = preds[f"{args.model}_conf"]

    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    lockbox = pd.Timestamp(LOCKBOX_START, tz="UTC")

    def _ev(idx):                     # evaluation window [split, lockbox): sealed lockbox excluded
        return (idx >= split) & (idx < lockbox)

    bars_cal, bars_eval = bars[bars.index < split], bars[_ev(bars.index)]
    pred_cal, conf_cal = pred[pred.index < split], conf[conf.index < split]
    pred_eval, conf_eval = pred[_ev(pred.index)], conf[_ev(conf.index)]

    # ---- stage 1: symmetric joint grid on Q1 ------------------------------
    grid_rows = []
    for tau, tp, sl, hold in product(TAUS, tp_grid, sl_grid, hold_grid):
        summary, _, _ = bracket_economics(
            bars_cal, pred_cal, conf_cal, tau=tau, tp=tp, sl=sl, hold=hold,
            fee_bps=fee_bps, slippage_bps=args.slippage_bps)
        summary.update({"tau": tau, "tp_bps": tp, "sl_bps": sl, "max_hold": hold,
                        "stage": "symmetric"})
        grid_rows.append(summary)
    grid = pd.DataFrame(grid_rows)
    traded = grid[grid["n_trades"] > 0]
    best = (traded if len(traded) else grid).sort_values(
        args.select_metric, ascending=False).iloc[0]
    tau = float(best["tau"]); hold = int(best["max_hold"])
    tp_long = tp_short = float(best["tp_bps"])
    sl_long = sl_short = float(best["sl_bps"])
    print(f"[stage 1] tau={tau:.2f} tp={tp_long:g} sl={sl_long:g} hold={hold} "
          f"(Q1 {args.select_metric} {best[args.select_metric]:+.2f}, "
          f"{int(best['n_trades'])} trades)")

    # ---- stage 2 (optional): asymmetric brackets, tau/hold frozen ---------
    if args.asymmetric:
        rows = []
        for tpl, sll, tps, sls in product(tp_grid, sl_grid, tp_grid, sl_grid):
            summary, _, _ = bracket_economics(
                bars_cal, pred_cal, conf_cal, tau=tau, tp=tpl, sl=sll, hold=hold,
                fee_bps=fee_bps, slippage_bps=args.slippage_bps,
                tp_short=tps, sl_short=sls)
            summary.update({"tau": tau, "tp_bps": tpl, "sl_bps": sll,
                            "tp_bps_short": tps, "sl_bps_short": sls,
                            "max_hold": hold, "stage": "asymmetric"})
            rows.append(summary)
        stage2 = pd.DataFrame(rows)
        grid = pd.concat([grid, stage2], ignore_index=True)
        traded = stage2[stage2["n_trades"] > 0]
        if len(traded):
            best2 = traded.sort_values(args.select_metric, ascending=False).iloc[0]
            if best2[args.select_metric] > best[args.select_metric]:
                tp_long, sl_long = float(best2["tp_bps"]), float(best2["sl_bps"])
                tp_short, sl_short = float(best2["tp_bps_short"]), float(best2["sl_bps_short"])
                print(f"[stage 2] long tp/sl={tp_long:g}/{sl_long:g} "
                      f"short tp/sl={tp_short:g}/{sl_short:g} "
                      f"(Q1 {args.select_metric} {best2[args.select_metric]:+.2f})")
            else:
                print("[stage 2] asymmetric brackets do not beat symmetric on Q1 — kept symmetric")

    # ---- stage 3 (optional): split gates, brackets frozen -----------------
    tau_long = tau_short = tau
    if args.split_tau:
        rows = []
        base_score = None
        for tl, ts in product(TAUS, TAUS):
            gated = gate_per_side(pred_cal, conf_cal, tl, ts)
            summary, _, _ = bracket_economics(
                bars_cal, gated, None, tau=0.0, tp=tp_long, sl=sl_long, hold=hold,
                fee_bps=fee_bps, slippage_bps=args.slippage_bps,
                tp_short=tp_short, sl_short=sl_short)
            summary.update({"tau_long": tl, "tau_short": ts, "stage": "split_tau",
                            "tp_bps": tp_long, "sl_bps": sl_long,
                            "tp_bps_short": tp_short, "sl_bps_short": sl_short,
                            "max_hold": hold})
            rows.append(summary)
            if tl == tau and ts == tau:
                base_score = summary[args.select_metric]
        stage3 = pd.DataFrame(rows)
        grid = pd.concat([grid, stage3], ignore_index=True)
        traded = stage3[stage3["n_trades"] > 0]
        if len(traded):
            best3 = traded.sort_values(args.select_metric, ascending=False).iloc[0]
            if base_score is None or best3[args.select_metric] > base_score:
                tau_long, tau_short = float(best3["tau_long"]), float(best3["tau_short"])
                print(f"[stage 3] tau_long={tau_long:.2f} tau_short={tau_short:.2f} "
                      f"(Q1 {args.select_metric} {best3[args.select_metric]:+.2f})")

    # ---- frozen evaluation on Q2-Q4 ---------------------------------------
    summaries = []
    gated_eval = gate_per_side(pred_eval, conf_eval, tau_long, tau_short)
    for slip in sorted({args.slippage_bps, *SLIPPAGE_SENSITIVITY}):
        summary, ledger, per_bar = bracket_economics(
            bars_eval, gated_eval, None, tau=0.0, tp=tp_long, sl=sl_long, hold=hold,
            fee_bps=fee_bps, slippage_bps=slip,
            tp_short=tp_short, sl_short=sl_short)
        summary.update({"scope": "all", "slippage_bps": slip,
                        "tau_long": tau_long, "tau_short": tau_short,
                        "tp_bps": tp_long, "sl_bps": sl_long,
                        "tp_bps_short": tp_short, "sl_bps_short": sl_short,
                        "max_hold": hold, "model": args.model, "tag": args.tag})
        if slip == args.slippage_bps:
            headline_ledger, headline_returns = ledger, per_bar
            # long-only / short-only decomposition at the headline slippage
            for scope, side in (("long", 1), ("short", -1)):
                side_pred = gated_eval.where(
                    gated_eval == (2 if side == 1 else 0), 1)
                s, led, _ = bracket_economics(
                    bars_eval, side_pred, None, tau=0.0, tp=tp_long, sl=sl_long,
                    hold=hold, fee_bps=fee_bps, slippage_bps=slip,
                    tp_short=tp_short, sl_short=sl_short)
                s.update({"scope": scope, "slippage_bps": slip, "model": args.model,
                          "tag": args.tag})
                summaries.append(s)
        summaries.append(summary)

    # per-bar gated baseline on the same eval bars, same selection metric
    cal_preds = preds[preds.index < split]
    eval_preds = preds[_ev(preds.index)]
    base_sweep = sweep_tau(cal_preds[f"{args.model}_pred"], cal_preds[f"{args.model}_conf"],
                           cal_preds["forward_return"], fee_bps + args.slippage_bps, TAUS)
    base_tau = float(base_sweep.loc[base_sweep[args.select_metric].idxmax(), "tau"])
    baseline_returns = strategy_returns(
        eval_preds[f"{args.model}_pred"], eval_preds["forward_return"],
        fee_bps + args.slippage_bps, eval_preds[f"{args.model}_conf"], base_tau)
    base_summary = economics_summary(
        baseline_returns, eval_preds[f"{args.model}_pred"],
        eval_preds[f"{args.model}_conf"], base_tau)
    base_summary.update({"scope": "per_bar_baseline", "slippage_bps": args.slippage_bps,
                         "model": args.model, "tag": args.tag})
    dm = diebold_mariano(baseline_returns,
                         headline_returns.reindex(baseline_returns.index, fill_value=0.0))
    base_summary.update({f"dm_{k}": v for k, v in dm.items()})
    summaries.append(base_summary)

    table = pd.DataFrame(summaries)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.instrument}_{args.sentiment}_{args.model}_{args.tag}_brk"
    grid.to_parquet(OUT_DIR / f"{stem}_grid.parquet", index=False)
    table.to_parquet(OUT_DIR / f"{stem}_summary.parquet", index=False)
    headline_ledger.to_parquet(OUT_DIR / f"{stem}_trades.parquet", index=False)
    pd.DataFrame({"bracket": headline_returns,
                  "per_bar_baseline": baseline_returns}).to_parquet(
        OUT_DIR / f"{stem}_returns.parquet")

    cols = ["scope", "slippage_bps", "sortino", "sharpe", "net_return_sum",
            "max_drawdown", "n_trades", "win_rate", "profit_factor",
            "expectancy_bps", "tp_rate", "sl_rate", "timeout_rate"]
    have = [c for c in cols if c in table.columns]
    print(f"\n{table[have].to_string(index=False)}")
    print(f"\nbracket vs per-bar baseline: DM {dm['dm_stat']:+.2f} (p={dm['p_value']:.3f}, "
          f"positive favours brackets)")
    print(f"wrote artifacts -> {OUT_DIR / (stem + '_*.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
