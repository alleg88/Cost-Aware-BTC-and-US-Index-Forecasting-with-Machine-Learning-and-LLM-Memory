"""Step 2: does a smarter SELECTION gate beat the champion's confidence gate?

The economics notebooks show selectivity is the only lever that monetises the M15
signal. The default selectivity is a per-model confidence threshold (tau) on the
champion (order-flow CatBoost @ dz35, +3.7% net, Sortino 3.39). This script asks
whether a **multi-model** gate on the four seeds beats it, using only **symmetric**
mechanisms (long and short treated identically — no long-only shortcut, which would
merely harvest the 2025 up-market):

  * K-of-4 agreement       (evaluation.selection.agreement_signal)
  * conformal singleton set (evaluation.selection.conformal_signal)

Substrate note (a finding in itself): the multi-model gates run on the **calibrated**
member outputs. On RAW member predictions the agreement gate over-trades
catastrophically (members rarely abstain), so per-window isotonic calibration — the
same step the deployed ensemble uses — is what makes agreement selective. The champion
benchmark is the RAW CatBoost gate (calibration is not applied to the single model, so
this row reproduces the deployed +3.7% exactly). Everything is at dz35; each knob
(tau / K / alpha) is Q1-calibrated, frozen, evaluated once on Q2-Q4; DM vs the champion;
long/short P&L decomposed so a trending-year long bias cannot hide.

Run:  python -m experiments.run_selection_gates
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import (
    diebold_mariano, economics_summary, strategy_returns, sweep_tau)
from evaluation.selection import agreement_signal, conformal_qhat, conformal_signal
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WF = CODE_ROOT / "experiments" / "cache" / "walkforward"
OUT = CODE_ROOT / "experiments" / "cache" / "economics"

MEMBERS = ("gru", "catboost_balanced", "random_forest", "mlp")
TAG = "dz35"                       # champion's best label
SPLIT = pd.Timestamp(CALIBRATION_END, tz="UTC")
LOCKBOX = pd.Timestamp(LOCKBOX_START, tz="UTC")
TAUS = (0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
K_GRID = (2, 3, 4)
ALPHA_GRID = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40)


def _load() -> pd.DataFrame:
    """Aligned panel: calibrated member preds/probs, the calibrated ensemble, and
    the RAW CatBoost champion columns (cb_raw_*)."""
    frames = {}
    for m in MEMBERS:
        d = pd.read_parquet(WF / f"btc_bothof_{m}_cal_{TAG}_{CACHE_SUFFIX}.parquet")
        d.index = pd.to_datetime(d.index, utc=True)
        frames[m] = d.sort_index()
    ens = pd.read_parquet(WF / f"btc_bothof_ensemble4_{TAG}_{CACHE_SUFFIX}.parquet")
    ens.index = pd.to_datetime(ens.index, utc=True)
    ens = ens.sort_index()
    raw_cb = pd.read_parquet(WF / f"btc_bothof_catboost_balanced_{TAG}_{CACHE_SUFFIX}.parquet")
    raw_cb.index = pd.to_datetime(raw_cb.index, utc=True)
    raw_cb = raw_cb.sort_index()

    idx = ens.index
    panel = pd.DataFrame(index=idx)
    panel["y_true"] = ens["y_true"].astype(int)
    panel["forward_return"] = ens["forward_return"].astype(float)
    for c in ("pred", "conf", "p0", "p1", "p2"):
        panel[f"ens_{c}"] = ens[f"ensemble4_{c}"]
    panel["cb_raw_pred"] = raw_cb.reindex(idx)["catboost_balanced_pred"]
    panel["cb_raw_conf"] = raw_cb.reindex(idx)["catboost_balanced_conf"]
    for m in MEMBERS:
        f = frames[m].reindex(idx)
        panel[f"{m}_pred"] = f[f"{m}_cal_pred"]
        panel[f"{m}_conf"] = f[f"{m}_cal_conf"]
    panel = panel.dropna(subset=[f"{m}_pred" for m in MEMBERS] + ["ens_pred", "cb_raw_pred"])
    panel["votes_up"] = sum((panel[f"{m}_pred"] == 2).astype(int) for m in MEMBERS)
    panel["votes_down"] = sum((panel[f"{m}_pred"] == 0).astype(int) for m in MEMBERS)
    return panel


def _econ(pred: pd.Series, fwd: pd.Series, fee: float) -> tuple[dict, pd.Series]:
    """Economics of an already-gated 0/1/2 signal + long/short decomposition."""
    ret = strategy_returns(pred, fwd, fee)
    s = economics_summary(ret, pred, None, 0.0)
    s["long_net"] = float(strategy_returns(pred.where(pred == 2, 1), fwd, fee).sum())
    s["short_net"] = float(strategy_returns(pred.where(pred == 0, 1), fwd, fee).sum())
    return s, ret


def _best_tau(cal, pred_col, conf_col, fee):
    sw = sweep_tau(cal[pred_col], cal[conf_col], cal["forward_return"], fee, TAUS)
    return float(sw.loc[sw["sortino"].idxmax(), "tau"])


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    panel = _load()
    cal, ev = panel[panel.index < SPLIT], panel[(panel.index >= SPLIT) & (panel.index < LOCKBOX)]
    fwd = ev["forward_return"]
    print(f"panel {len(panel)} bars | cal {len(cal)} | eval {len(ev)} | fee {fee:g} bps")

    results, returns = [], {}

    def record(name, knob, pred_eval):
        s, ret = _econ(pred_eval, fwd, fee)
        s.update({"policy": name, "knob": knob})
        results.append(s)
        returns[name] = ret
        return ret

    # ---- benchmark: RAW CatBoost champion gate -----------------------------
    t = _best_tau(cal, "cb_raw_pred", "cb_raw_conf", fee)
    base_ret = record("CatBoost gate (raw, champion)", f"tau={t:.2f}",
                      ev["cb_raw_pred"].where(ev["cb_raw_conf"] >= t, 1))

    # ---- calibrated single-model gate (does calibration help the gate?) ----
    t = _best_tau(cal, "catboost_balanced_pred", "catboost_balanced_conf", fee)
    record("CatBoost gate (calibrated)", f"tau={t:.2f}",
           ev["catboost_balanced_pred"].where(ev["catboost_balanced_conf"] >= t, 1))

    # ---- calibrated ensemble soft-vote gate --------------------------------
    t = _best_tau(cal, "ens_pred", "ens_conf", fee)
    record("ensemble4 soft-vote gate", f"tau={t:.2f}",
           ev["ens_pred"].where(ev["ens_conf"] >= t, 1))

    # ---- G1: K-of-4 agreement (calibrated votes) ---------------------------
    best = max(K_GRID, key=lambda k: economics_summary(
        strategy_returns(agreement_signal(cal["votes_up"], cal["votes_down"], k),
                         cal["forward_return"], fee))["sortino"])
    record("agreement K-of-4", f"K={best}",
           agreement_signal(ev["votes_up"], ev["votes_down"], best))

    # ---- G2: agreement + calibrated-ensemble confidence --------------------
    best = None
    for k in K_GRID:
        for tau in TAUS:
            sig = agreement_signal(cal["votes_up"], cal["votes_down"], k).where(
                cal["ens_conf"] >= tau, 1)
            sc = economics_summary(strategy_returns(sig, cal["forward_return"], fee))["sortino"]
            if best is None or sc > best[0]:
                best = (sc, k, tau)
    k2, tau2 = best[1], best[2]
    record("agreement + ens-conf", f"K={k2}, tau={tau2:.2f}",
           agreement_signal(ev["votes_up"], ev["votes_down"], k2).where(ev["ens_conf"] >= tau2, 1))

    # ---- G3: conformal singleton on the calibrated ensemble probs ----------
    cal_p = cal[["ens_p0", "ens_p1", "ens_p2"]].rename(
        columns={"ens_p0": "p0", "ens_p1": "p1", "ens_p2": "p2"})
    ev_p = ev[["ens_p0", "ens_p1", "ens_p2"]].rename(
        columns={"ens_p0": "p0", "ens_p1": "p1", "ens_p2": "p2"})
    p_true = cal_p.to_numpy()[np.arange(len(cal)), cal["y_true"].to_numpy()]
    best = None
    for alpha in ALPHA_GRID:
        qhat = conformal_qhat(p_true, alpha)
        sc = economics_summary(strategy_returns(
            conformal_signal(cal_p, qhat), cal["forward_return"], fee))["sortino"]
        if best is None or sc > best[0]:
            best = (sc, alpha, qhat)
    record("conformal singleton", f"alpha={best[1]:.2f}", conformal_signal(ev_p, best[2]))

    # ---- assemble + DM vs the champion -------------------------------------
    table = pd.DataFrame(results)
    dm = [{"policy": n, "dm_vs_champ": diebold_mariano(
              base_ret, r.reindex(base_ret.index, fill_value=0.0))["dm_stat"],
           "dm_p": diebold_mariano(base_ret, r.reindex(base_ret.index, fill_value=0.0))["p_value"]}
          for n, r in returns.items()]
    table = table.merge(pd.DataFrame(dm), on="policy")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = f"btc_bothof_selection_gates_{TAG}"
    table.to_parquet(OUT / f"{stem}_summary.parquet", index=False)
    pd.DataFrame(returns).to_parquet(OUT / f"{stem}_returns.parquet")

    cols = ["policy", "knob", "sortino", "sharpe", "net_return_sum", "trade_count",
            "long_net", "short_net", "dm_vs_champ", "dm_p"]
    pd.set_option("display.width", 200)
    print("\nFrozen Q2-Q4 (dz35, 5 bps/side; multi-model gates on calibrated members):")
    print(table[cols].round(4).to_string(index=False))
    print(f"\nwrote -> {OUT / (stem + '_*.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
