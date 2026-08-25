"""Meta-labeling: a second model filters bracket trades by predicted win odds.

The primary model answers "which direction?"; a SECONDARY model answers "given
this signal, will the bracket trade actually pay?" (Lopez de Prado, Advances in
Financial Machine Learning, 2018, ch. 3). The secondary model is trained on the
outcomes of past candidate trades and used as a trade filter: take the trade
only when the predicted win probability clears a threshold theta.

Pipeline:
  1. Candidates: every gated signal gets a bracket outcome simulated on its own
     path (take-profit / stop-loss / time-out, same conventions as
     evaluation.trades). Candidate outcomes ignore the one-trade-at-a-time
     constraint so the training set stays dense.
  2. Win-probability model: logistic regression on signal-time features
     (confidence, class probabilities, volatility, range, RSI, time of day,
     side), refit each week on candidates whose outcome window CLOSED before
     the week starts — leak-free by construction. Weeks with too little
     history predict a neutral 0.5.
  3. Filter: signals with p_win < theta are forced flat, then the sequential
     bracket simulator runs as usual. Theta is calibrated on Q1-2025 by the
     --select-metric (default Sortino), frozen, evaluated on Q2-Q4 against the
     unfiltered bracket strategy (Diebold-Mariano on per-bar returns).

The gate/bracket parameters are CLI arguments: set them from the frozen
choices of experiments.run_brackets. If Q1 yields too few candidates at that
gate, loosen tau here and say so — a filter cannot be learned from a handful
of trades.

Run:  python -m experiments.meta_labeling --model gru --tag m15 --tau 0.55
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import diebold_mariano, economics_summary
from evaluation.trades import simulate_bracket_trades, trade_stats
from experiments.run_brackets import gate_per_side, load_inputs
from experiments.spans import CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"

# CALIBRATION_END / LOCKBOX_START from experiments.spans (config-driven)
THETA_GRID = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
META_FEATURES = ("conf", "p0", "p2", "side", "r1", "vol_10", "vol_20", "vol_60",
                 "hl_range", "rsi_14", "vol_z", "hour", "dayofweek")
MIN_TRAIN_CANDIDATES = 40


def build_candidates(
    bars: pd.DataFrame, preds: pd.DataFrame, model: str, *,
    tau: float, tp_bps: float, sl_bps: float, max_hold: int,
    fee_bps: float, slippage_bps: float = 0.0,
) -> pd.DataFrame:
    """One row per gated signal: outcome of ITS OWN bracket path + features."""
    gated = gate_per_side(preds[f"{model}_pred"], preds[f"{model}_conf"], tau, tau)
    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    positions = bars.index.get_indexer(gated.index)
    cost = 2.0 * (float(fee_bps) + float(slippage_bps)) / 10_000.0

    rows = []
    for ts, cls, bar_pos in zip(gated.index, gated.to_numpy(), positions):
        if cls == 1 or bar_pos < 0 or bar_pos >= len(bars) - 1:
            continue
        side = 1 if cls == 2 else -1
        entry_idx = bar_pos + 1
        entry_px = open_[entry_idx]
        tp_px = entry_px * (1.0 + side * tp_bps / 10_000.0)
        sl_px = entry_px * (1.0 - side * sl_bps / 10_000.0)
        last_idx = min(entry_idx + max_hold - 1, len(bars) - 1)
        exit_idx, exit_px = last_idx, close[last_idx]
        for j in range(entry_idx, last_idx + 1):
            hit_sl = low[j] <= sl_px if side == 1 else high[j] >= sl_px
            hit_tp = high[j] >= tp_px if side == 1 else low[j] <= tp_px
            if hit_sl:
                exit_idx, exit_px = j, sl_px
                break
            if hit_tp:
                exit_idx, exit_px = j, tp_px
                break
        net = side * (exit_px / entry_px - 1.0) - cost

        row = {"signal_time": ts, "exit_time": bars.index[exit_idx],
               "side": float(side), "net_return": float(net),
               "win": int(net > 0.0),
               "conf": float(preds.at[ts, f"{model}_conf"]),
               "p0": float(preds.at[ts, f"{model}_p0"]),
               "p2": float(preds.at[ts, f"{model}_p2"])}
        for col in META_FEATURES:
            if col in preds.columns and col not in row:
                row[col] = float(preds.at[ts, col])
        rows.append(row)
    return pd.DataFrame(rows).set_index("signal_time") if rows else pd.DataFrame()


def walkforward_pwin(candidates: pd.DataFrame) -> pd.Series:
    """Weekly-refit logistic win probability; neutral 0.5 with thin history."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = [c for c in META_FEATURES if c in candidates.columns]
    X = candidates[feats].astype(float).fillna(0.0)
    p_win = pd.Series(0.5, index=candidates.index, name="p_win")

    weeks = candidates.index.tz_convert("UTC").tz_localize(None).to_period("W")
    for week in weeks.unique():
        week_mask = weeks == week
        week_start = candidates.index[week_mask].min()
        train_mask = candidates["exit_time"] < week_start   # outcome fully in the past
        y_train = candidates.loc[train_mask, "win"]
        if train_mask.sum() < MIN_TRAIN_CANDIDATES or y_train.nunique() < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0))
        clf.fit(X.loc[train_mask], y_train)
        p_win.loc[week_mask] = clf.predict_proba(X.loc[week_mask])[:, 1]
    return p_win


def filtered_strategy(
    bars: pd.DataFrame, preds: pd.DataFrame, model: str, p_win: pd.Series, *,
    tau: float, theta: float, tp_bps: float, sl_bps: float, max_hold: int,
    fee_bps: float, slippage_bps: float = 0.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sequential bracket run with sub-theta signals forced flat."""
    gated = gate_per_side(preds[f"{model}_pred"], preds[f"{model}_conf"], tau, tau)
    keep = p_win.reindex(gated.index)
    filtered = gated.where(~((gated != 1) & (keep < theta)), 1)
    return simulate_bracket_trades(
        bars, filtered, None, tau=0.0, tp_bps=tp_bps, sl_bps=sl_bps,
        max_hold=max_hold, fee_bps=fee_bps, slippage_bps=slippage_bps)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    ap.add_argument("--model", default="gru")
    ap.add_argument("--tag", default="m15")
    ap.add_argument("--tau", type=float, default=0.55,
                    help="confidence gate for candidate signals (from run_brackets)")
    ap.add_argument("--tp-bps", type=float, default=50.0)
    ap.add_argument("--sl-bps", type=float, default=50.0)
    ap.add_argument("--max-hold", type=int, default=16)
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    ap.add_argument("--select-metric", default="sortino", choices=["sortino", "sharpe"])
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee_bps = float(cfg["instruments"][args.instrument]["taker_fee_bps"])
    bars, preds = load_inputs(args.instrument, args.sentiment, args.model, args.tag, cfg)

    candidates = build_candidates(
        bars, preds, args.model, tau=args.tau, tp_bps=args.tp_bps,
        sl_bps=args.sl_bps, max_hold=args.max_hold,
        fee_bps=fee_bps, slippage_bps=args.slippage_bps)
    if candidates.empty:
        raise SystemExit("no candidate trades at this gate — loosen --tau")
    candidates["p_win"] = walkforward_pwin(candidates)

    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    n_q1 = int((candidates.index < split).sum())
    print(f"{len(candidates)} candidates ({n_q1} in Q1), "
          f"base win rate {candidates['win'].mean():.1%}")
    if n_q1 < MIN_TRAIN_CANDIDATES:
        print(f"WARNING: only {n_q1} Q1 candidates — theta calibration is fragile; "
              f"consider a looser --tau")

    lockbox = pd.Timestamp(LOCKBOX_START, tz="UTC")
    bars_cal = bars[bars.index < split]
    bars_eval = bars[(bars.index >= split) & (bars.index < lockbox)]

    # ---- calibrate theta on Q1 (theta=0 row = unfiltered reference) --------
    cal_rows = []
    for theta in (0.0, *THETA_GRID):
        _, per_bar = filtered_strategy(
            bars_cal, preds[preds.index < split], args.model,
            candidates["p_win"], tau=args.tau, theta=theta,
            tp_bps=args.tp_bps, sl_bps=args.sl_bps, max_hold=args.max_hold,
            fee_bps=fee_bps, slippage_bps=args.slippage_bps)
        row = economics_summary(per_bar)
        row["theta"] = theta
        cal_rows.append(row)
    cal_table = pd.DataFrame(cal_rows)
    best_theta = float(cal_table.loc[cal_table[args.select_metric].idxmax(), "theta"])
    print(f"theta* = {best_theta:.2f} "
          f"(Q1 {args.select_metric} {cal_table[args.select_metric].max():+.2f})")

    # ---- frozen evaluation on Q2-Q4 ----------------------------------------
    results = {}
    for name, theta in (("unfiltered", 0.0), ("meta_filtered", best_theta)):
        ledger, per_bar = filtered_strategy(
            bars_eval, preds[preds.index >= split], args.model,
            candidates["p_win"], tau=args.tau, theta=theta,
            tp_bps=args.tp_bps, sl_bps=args.sl_bps, max_hold=args.max_hold,
            fee_bps=fee_bps, slippage_bps=args.slippage_bps)
        summary = economics_summary(per_bar)
        summary.update(trade_stats(ledger))
        summary.update({"variant": name, "theta": theta, "tau": args.tau,
                        "model": args.model, "tag": args.tag})
        results[name] = (summary, ledger, per_bar)

    dm = diebold_mariano(results["unfiltered"][2], results["meta_filtered"][2])

    # ---- filter quality on eval candidates ---------------------------------
    eval_cands = candidates[candidates.index >= split]
    quality = {"variant": "filter_quality", "model": args.model, "tag": args.tag,
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
        "skipped_expectancy_bps": float(skipped["net_return"].mean() * 1e4) if len(skipped) else np.nan,
        "n_taken": int(len(taken)), "n_skipped": int(len(skipped)),
    })

    table = pd.DataFrame([results["unfiltered"][0], results["meta_filtered"][0], quality])
    table.loc[table["variant"] == "meta_filtered", "dm_stat"] = dm["dm_stat"]
    table.loc[table["variant"] == "meta_filtered", "dm_p_value"] = dm["p_value"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.instrument}_{args.sentiment}_{args.model}_{args.tag}_meta"
    table.to_parquet(OUT_DIR / f"{stem}_summary.parquet", index=False)
    cal_table.to_parquet(OUT_DIR / f"{stem}_theta_grid.parquet", index=False)
    candidates.reset_index().to_parquet(OUT_DIR / f"{stem}_candidates.parquet", index=False)
    pd.DataFrame({"unfiltered": results["unfiltered"][2],
                  "meta_filtered": results["meta_filtered"][2]}).to_parquet(
        OUT_DIR / f"{stem}_returns.parquet")

    cols = ["variant", "theta", "sortino", "sharpe", "net_return_sum", "n_trades",
            "win_rate", "expectancy_bps", "auc", "taken_win_rate", "skipped_win_rate"]
    have = [c for c in cols if c in table.columns]
    print(f"\n{table[have].to_string(index=False)}")
    print(f"\nmeta filter vs unfiltered: DM {dm['dm_stat']:+.2f} (p={dm['p_value']:.3f}, "
          f"positive favours the filter)")
    print(f"wrote artifacts -> {OUT_DIR / (stem + '_*.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
