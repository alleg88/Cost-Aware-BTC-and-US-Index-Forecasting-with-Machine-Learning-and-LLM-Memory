"""Frozen forward comparison grid for the walk-forward arms — all widths, all
gates, all engines.

Grid: width {40,55,75} x params {legacy, f1-tuned, econ-tuned; SqrtBalanced at dz40} x gate
{global tau, E1 funding-regime tau} x engine {per-bar, m15 brackets,
1m brackets + trailing}. For EVERY cell the knobs are chosen on the calibration
window only (walk-forward predictions before CALIBRATION_END = 2025-H1; tau(s)
by Sortino with the >=50-event floor; bracket geometry tp x sl x trail with the
1-minute engine, same floor), frozen, then evaluated once on
[CALIBRATION_END, LOCKBOX_START) = 2025-Q3..2026-Q1. DM tests vs the legacy
global-tau cell of the same width+engine. Lockbox (Q2-2026) sealed.

The full grid is TRANSPARENCY — 18 gate-configs are inspected, so any single
standout must be discounted for multiplicity. The pre-registered headline is
the per-arm honest pick (best calibration Sortino across width x gate),
flagged in the `picked` column.

Requires run_walkforward_arms caches.  Run:  python -m experiments.run_arms_economics
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import diebold_mariano, economics_summary, strategy_returns
from evaluation.trades import simulate_bracket_trades
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.run_walkforward_arms import WIDTHS, arms_for_width, cache_path
from experiments.spans import CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
ECON = CODE_ROOT / "experiments" / "cache" / "economics"
MODEL = "catboost_balanced"

TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
FLOOR = 50
TP_GRID = (50.0, 100.0, 150.0)
SL_GRID = (25.0, 50.0, 75.0)
TRAIL_GRID = (None, 50.0, 100.0, 150.0)
MAX_HOLD = 16


def dm_lag_for_engine(engine: str) -> int:
    """Return a Newey-West lag matching the return series' holding horizon."""
    if engine == "per-bar":
        return 1
    if engine in {"m15 brackets", "1m brackets+trail"}:
        return MAX_HOLD
    raise ValueError(f"unknown engine: {engine}")


SPLIT = pd.Timestamp(CALIBRATION_END, tz="UTC")
LOCKBOX = pd.Timestamp(LOCKBOX_START, tz="UTC")


def _utc(df):
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def funding_z(index: pd.Index) -> pd.Series:
    pos = _utc(pd.read_parquet(
        CODE_ROOT / "data" / "btcusdt_positioning_m15_2024_2026.parquet"))
    fr = pos["funding_rate"].astype(float)
    z = (fr - fr.rolling(672).mean()) / fr.rolling(672).std().replace(0.0, np.nan)
    return z.reindex(index)


def calibrate_global(pred, conf, fwd, fee):
    best = None
    for tau in TAUS:
        s = economics_summary(strategy_returns(pred, fwd, fee, conf, tau),
                              pred, conf, tau)
        if s["trade_count"] >= FLOOR and (best is None or s["sortino"] > best[1]):
            best = (tau, s["sortino"])
    return best                                  # (tau, cal_sortino) | None


def calibrate_regime(pred, conf, fwd, fee, reg):
    taus = {}
    for side in (True, False):
        m = reg == side
        best = None
        for tau in TAUS:
            s = economics_summary(
                strategy_returns(pred[m], fwd[m], fee, conf[m], tau),
                pred[m], conf[m], tau)
            if s["trade_count"] >= FLOOR and (best is None or s["sortino"] > best[1]):
                best = (tau, s["sortino"])
        if best is None:
            return None
        taus[side] = best[0]
    sig = apply_regime(pred, conf, reg, taus)
    s = economics_summary(strategy_returns(sig, fwd, fee), sig)
    if s["trade_count"] < FLOOR:
        return None
    return (taus, s["sortino"])                  # ({side: tau}, cal_sortino) | None


def apply_regime(pred, conf, reg, taus):
    sig = pred.copy()
    for side, tau in taus.items():
        m = reg == side
        sig[m & (conf < tau) & (pred != 1)] = 1
    return sig


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    bars = _utc(pd.read_parquet(CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"]))
    minute = _utc(pd.read_parquet(CODE_ROOT / "data" / "btcusdt_1m_2025_2026.parquet"))

    rows, ret_store = [], {}
    for width in WIDTHS:
        for arm in arms_for_width(width):
            path = cache_path(arm, width)
            if not path.exists():
                print(f"[missing cache] {path.name} — run run_walkforward_arms first")
                continue
            d = _utc(pd.read_parquet(path))
            pred, conf = d[f"{MODEL}_pred"].astype(int), d[f"{MODEL}_conf"]
            fwd = d["forward_return"]
            fz = funding_z(d.index)
            cal = d.index < SPLIT
            ev = (d.index >= SPLIT) & (d.index < LOCKBOX)
            bars_arm = bars.loc[d.index.min():d.index.max()]
            b_cal = bars_arm[bars_arm.index < SPLIT]
            b_ev = bars_arm[(bars_arm.index >= SPLIT) & (bars_arm.index < LOCKBOX)]

            gates = {}
            g = calibrate_global(pred[cal], conf[cal], fwd[cal], fee)
            if g:
                gates["global tau"] = {"sig": pred.where(conf >= g[0], 1),
                                       "knobs": f"tau={g[0]:.2f}", "cal_sortino": g[1]}
            r = calibrate_regime(pred[cal], conf[cal], fwd[cal], fee, (fz.abs() > 1)[cal])
            if r:
                gates["E1 funding-regime"] = {
                    "sig": apply_regime(pred, conf, fz.abs() > 1, r[0]),
                    "knobs": "tau " + "/".join(f"{v:.2f}" for v in r[0].values()),
                    "cal_sortino": r[1]}

            for gname, ginfo in gates.items():
                sig = ginfo["sig"]
                # bracket geometry + trailing on calibration, 1m engine, floored
                best_geo = None
                for tp, sl, tr in product(TP_GRID, SL_GRID, TRAIL_GRID):
                    led, pb = simulate_bracket_trades_intrabar(
                        b_cal, minute, sig[cal], None, tau=0.0, tp_bps=tp, sl_bps=sl,
                        max_hold=MAX_HOLD, fee_bps=fee, trail_bps=tr)
                    if len(led) < FLOOR:
                        continue
                    sc = economics_summary(pb)["sortino"]
                    if best_geo is None or sc > best_geo[3]:
                        best_geo = (tp, sl, tr, sc)
                geo = best_geo[:3] if best_geo else (100.0, 50.0, None)

                engines = {"per-bar": (None, strategy_returns(sig[ev], fwd[ev], fee))}
                led, pb = simulate_bracket_trades(
                    b_ev, sig[ev], None, tau=0.0, tp_bps=geo[0], sl_bps=geo[1],
                    max_hold=MAX_HOLD, fee_bps=fee)
                engines["m15 brackets"] = (led, pb)
                led, pb = simulate_bracket_trades_intrabar(
                    b_ev, minute, sig[ev], None, tau=0.0, tp_bps=geo[0], sl_bps=geo[1],
                    max_hold=MAX_HOLD, fee_bps=fee, trail_bps=geo[2])
                engines["1m brackets+trail"] = (led, pb)

                for eng, (led, pb) in engines.items():
                    s = economics_summary(pb, sig[ev] if eng == "per-bar" else None)
                    rows.append({
                        "width": width, "arm": arm, "gate": gname,
                        "knobs": ginfo["knobs"],
                        "tp": geo[0], "sl": geo[1],
                        "trail": geo[2] if geo[2] is not None else np.nan,
                        "engine": eng, "cal_sortino": ginfo["cal_sortino"],
                        "eval_sortino": s["sortino"], "eval_sharpe": s["sharpe"],
                        "eval_net": s["net_return_sum"],
                        "eval_trades": int(s["trade_count"]) if eng == "per-bar" else len(led),
                    })
                    ret_store[(width, arm, gname, eng)] = pb
            print(f"done dz{width} {arm}: gates={list(gates)}")

    table = pd.DataFrame(rows)

    # DM vs the legacy global-tau cell of the same width+engine
    dm_col, dmp_col = [], []
    for _, r in table.iterrows():
        base = ret_store.get((r["width"], "legacy", "global tau", r["engine"]))
        this = ret_store.get((r["width"], r["arm"], r["gate"], r["engine"]))
        if base is None or this is None or (r["arm"] == "legacy" and r["gate"] == "global tau"):
            dm_col.append(np.nan); dmp_col.append(np.nan)
            continue
        idx = base.index.union(this.index)
        dm = diebold_mariano(
            base.reindex(idx, fill_value=0.0),
            this.reindex(idx, fill_value=0.0),
            lag=dm_lag_for_engine(r["engine"]),
        )
        dm_col.append(dm["dm_stat"]); dmp_col.append(dm["p_value"])
    table["dm_vs_legacy"] = dm_col
    table["dm_p"] = dmp_col

    # per-arm honest pick: best calibration Sortino across width x gate
    picks = (table[table["engine"] == "per-bar"]
             .sort_values("cal_sortino", ascending=False)
             .groupby("arm").first().reset_index())
    picked = set(zip(picks["arm"], picks["width"], picks["gate"]))
    table["picked"] = [
        "*" if (r["arm"], r["width"], r["gate"]) in picked else ""
        for _, r in table.iterrows()]

    ECON.mkdir(parents=True, exist_ok=True)
    table.to_parquet(ECON / "btc_arms_frozen_eval.parquet", index=False)

    pd.set_option("display.width", 240)
    cols = ["picked", "arm", "gate", "knobs", "tp", "sl", "trail", "cal_sortino",
            "eval_sortino", "eval_sharpe", "eval_net", "eval_trades",
            "dm_vs_legacy", "dm_p"]
    for eng in ("per-bar", "m15 brackets", "1m brackets+trail"):
        print(f"\n===== engine: {eng} | frozen eval [2025-07-01 .. 2026-04-01) =====")
        sub = table[table["engine"] == eng]
        for w in WIDTHS:
            blk = sub[sub["width"] == w]
            if blk.empty:
                continue
            print(f"\n-- dz{w} --")
            print(blk[cols].round(4).to_string(index=False))
    print(f"\nwrote -> {ECON / 'btc_arms_frozen_eval.parquet'}")
    print("`picked` = per-arm honest selection (best calibration Sortino); "
          "everything else is transparency and pays a multiplicity discount.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
