"""Intrabar (1-minute) exit engine on the champion, vs the M15 exit engine.

Step 1 of the post-economics improvement plan ([[improving-results-directions]]):
does resolving bracket exits on 1-minute bars — and the path-dependent stops it
unlocks (breakeven, trailing) plus volatility-scaled targets — beat the M15 exit
engine on the credible champion (order-flow CatBoost @ dz35)?

Protocol (unchanged freeze discipline):
  * The champion's gate + bracket geometry (tau, tp, sl, max_hold) are the Q1-2025
    frozen values already selected in run_brackets — read from its cached summary
    and held fixed here, so the ONLY thing that varies across engines is the exit.
  * Each new-policy knob (breakeven trigger, trailing distance, vol multipliers)
    is calibrated on Q1-2025 with the 1-minute engine, frozen, then evaluated once
    on Q2-Q4. DM tests vs the M15-exit champion on per-bar net returns.
  * Q1-2026 lockbox stays sealed (only 2025 minute data is loaded).

Run:  python -m experiments.run_brackets_intrabar
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import diebold_mariano, economics_summary
from evaluation.trades import simulate_bracket_trades, trade_stats
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WF_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
ECON_DIR = CODE_ROOT / "experiments" / "cache" / "economics"

SENT = "bothof"
SPLIT = pd.Timestamp(CALIBRATION_END, tz="UTC")
LOCKBOX = pd.Timestamp(LOCKBOX_START, tz="UTC")
MINUTE_PARQUET = "btcusdt_1m_2025_2026.parquet"
SLIPPAGE_SENSITIVITY = (0.0, 2.0, 5.0)

# Q1 calibration grids for the new policy knobs (tau/tp/sl/hold stay frozen).
BE_GRID = (25.0, 50.0, 75.0, 100.0)                       # breakeven trigger, bps
TRAIL_GRID = (25.0, 50.0, 75.0, 100.0, 150.0)             # trailing distance, bps
VOL_TP_MULT = (1.5, 2.0, 3.0, 4.0)                        # take-profit, xsigma
VOL_SL_MULT = (1.0, 1.5, 2.0)                             # stop-loss, xsigma


def _add_vol(bars: pd.DataFrame) -> pd.DataFrame:
    """Rolling 20-bar log-return volatility (fraction) for vol-scaled brackets."""
    out = bars.copy()
    out["vol_20"] = np.log(out["close"]).diff().rolling(20).std()
    return out


def _summary(ledger, per_bar, **extra) -> dict:
    s = economics_summary(per_bar)
    s.update(trade_stats(ledger))
    s["exposure"] = float((per_bar != 0.0).mean())
    s.update(extra)
    return s


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="catboost_balanced")
    ap.add_argument("--tag", default="dz35")
    args = ap.parse_args()
    MODEL, TAG = args.model, args.tag
    STEM = f"btc_{SENT}_{MODEL}_{TAG}"

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])

    # ---- inputs -----------------------------------------------------------
    bars = pd.read_parquet(CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"])
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = _add_vol(bars.sort_index())

    minute = pd.read_parquet(CODE_ROOT / "data" / MINUTE_PARQUET)
    minute.index = pd.to_datetime(minute.index, utc=True)
    minute = minute.sort_index()

    preds = pd.read_parquet(WF_DIR / f"{STEM}_{CACHE_SUFFIX}.parquet")
    preds.index = pd.to_datetime(preds.index, utc=True)
    preds = preds.sort_index()
    pred, conf = preds[f"{MODEL}_pred"], preds[f"{MODEL}_conf"]
    bars = bars.loc[pred.index.min():pred.index.max()]

    # ---- champion's frozen Q1 bracket geometry (from run_brackets cache) ---
    champ = pd.read_parquet(ECON_DIR / f"{STEM}_brk_summary.parquet")
    row = champ[champ["scope"] == "all"].iloc[0]
    tau = float(row["tau_long"]); tp = float(row["tp_bps"])
    sl = float(row["sl_bps"]); hold = int(row["max_hold"])
    print(f"champion frozen geometry: tau={tau:.2f} tp={tp:g} sl={sl:g} hold={hold}")

    def gate(p, c):
        g = p.astype(int).copy()
        g[(g != 1) & (c.astype(float) < tau)] = 1
        return g

    def _ev(idx):                     # evaluation window [SPLIT, LOCKBOX): sealed lockbox excluded
        return (idx >= SPLIT) & (idx < LOCKBOX)

    b_cal, b_eval = bars[bars.index < SPLIT], bars[_ev(bars.index)]
    gated_cal = gate(pred[pred.index < SPLIT], conf[conf.index < SPLIT])
    gated_eval = gate(pred[_ev(pred.index)], conf[_ev(conf.index)])

    def run_1m(bars_, gated, *, tp_=tp, sl_=sl, slip=0.0, vol=None,
               trail=None, be=None):
        return simulate_bracket_trades_intrabar(
            bars_, minute, gated, None, tau=0.0, tp_bps=tp_, sl_bps=sl_,
            max_hold=hold, fee_bps=fee, slippage_bps=slip,
            vol_scale_col=vol, trail_bps=trail, be_trigger_bps=be)

    # ---- Q1 calibration of the new knobs (1m engine, Sortino) --------------
    def calibrate(grid_iter, run_fn, label):
        best, best_kw = None, None
        for kw in grid_iter:
            led, pb = run_fn(kw)
            if len(led) == 0:
                continue
            sc = economics_summary(pb)["sortino"]
            if best is None or sc > best:
                best, best_kw = sc, kw
        print(f"[Q1] {label}: {best_kw}  (Q1 Sortino {best:+.2f})" if best_kw
              else f"[Q1] {label}: no trades on Q1")
        return best_kw

    be_star = calibrate(
        ({"be": b} for b in BE_GRID),
        lambda kw: run_1m(b_cal, gated_cal, be=kw["be"]), "breakeven")
    trail_star = calibrate(
        ({"trail": t} for t in TRAIL_GRID),
        lambda kw: run_1m(b_cal, gated_cal, trail=kw["trail"]), "trailing")
    vol_star = calibrate(
        ({"tp": a, "sl": b} for a, b in product(VOL_TP_MULT, VOL_SL_MULT)),
        lambda kw: run_1m(b_cal, gated_cal, tp_=kw["tp"], sl_=kw["sl"], vol="vol_20"),
        "vol-scaled")

    # ---- frozen Q2-Q4 evaluation across engines ---------------------------
    engines = {}
    led, pb = simulate_bracket_trades(b_eval, gated_eval, None, tau=0.0, tp_bps=tp,
                                      sl_bps=sl, max_hold=hold, fee_bps=fee)
    engines["M15 exits (champion)"] = (led, pb)
    engines["1m exits"] = run_1m(b_eval, gated_eval)
    if be_star:
        engines[f"1m + breakeven@{be_star['be']:g}"] = run_1m(
            b_eval, gated_eval, be=be_star["be"])
    if trail_star:
        engines[f"1m + trailing@{trail_star['trail']:g}"] = run_1m(
            b_eval, gated_eval, trail=trail_star["trail"])
    if vol_star:
        engines[f"1m + vol tp{vol_star['tp']:g}/sl{vol_star['sl']:g}x"] = run_1m(
            b_eval, gated_eval, tp_=vol_star["tp"], sl_=vol_star["sl"], vol="vol_20")

    base_returns = engines["M15 exits (champion)"][1]
    rows, ret_frame = [], {}
    for name, (led, pb) in engines.items():
        # positive dm_vs_m15 = this engine beats the M15 champion
        dm = diebold_mariano(base_returns.reindex(pb.index, fill_value=0.0), pb)
        rows.append(_summary(led, pb, engine=name,
                             dm_vs_m15=dm["dm_stat"], dm_p=dm["p_value"]))
        ret_frame[name] = pb
    table = pd.DataFrame(rows)

    # slippage sensitivity on the two headline engines
    slip_rows = []
    for name, kw in [("M15 exits (champion)", {}), ("1m exits", {})]:
        for slip in SLIPPAGE_SENSITIVITY:
            if name.startswith("M15"):
                led, pb = simulate_bracket_trades(
                    b_eval, gated_eval, None, tau=0.0, tp_bps=tp, sl_bps=sl,
                    max_hold=hold, fee_bps=fee, slippage_bps=slip)
            else:
                led, pb = run_1m(b_eval, gated_eval, slip=slip)
            slip_rows.append(_summary(led, pb, engine=name, slippage_bps=slip))
    slip_table = pd.DataFrame(slip_rows)

    ECON_DIR.mkdir(parents=True, exist_ok=True)
    out = f"{STEM}_intrabar"
    table.to_parquet(ECON_DIR / f"{out}_summary.parquet", index=False)
    slip_table.to_parquet(ECON_DIR / f"{out}_slippage.parquet", index=False)
    pd.DataFrame(ret_frame).to_parquet(ECON_DIR / f"{out}_returns.parquet")

    cols = ["engine", "sortino", "sharpe", "net_return_sum", "n_trades", "win_rate",
            "profit_factor", "expectancy_bps", "tp_rate", "sl_rate", "timeout_rate",
            "dm_vs_m15", "dm_p"]
    pd.set_option("display.width", 200)
    print("\nFrozen Q2-Q4 evaluation (0 bps slippage):")
    print(table[[c for c in cols if c in table.columns]].round(4).to_string(index=False))
    print(f"\nwrote -> {ECON_DIR / (out + '_*.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
