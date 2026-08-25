"""Equal-capital portfolio of independently gated configurations.

A single gated model trades a handful of times per year — the signal is real
but thin. Capacity comes from breadth: running several LOW-CORRELATION frozen
configurations in parallel (different models, label thresholds, and feature
sets) and splitting capital equally across them.

Honesty protocol:
  * Every candidate's gate tau is calibrated on Q1-2025 by --select-metric.
  * Portfolio MEMBERSHIP is decided by Q1 information only: a candidate joins
    iff its calibration score clears --min-cal-score. Evaluation-period
    results play no part in selection.
  * The frozen portfolio (equal weight over members) is evaluated once on
    Q2-Q4, with a Diebold-Mariano test against the best single member —
    "best" again chosen by the Q1 calibration score.

Default candidate pool: for each model, its best label threshold BY THE Q1
SCORE from each sweep summary, plus the m15 caches for both feature sets
(price+sentiment and price+sentiment+order-flow).

Run:  python -m experiments.run_portfolio
      python -m experiments.run_portfolio --configs gru:both:m15,gru:bothof:m15
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from evaluation.economics import (
    diebold_mariano, economics_summary, strategy_returns, sweep_tau,
)
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WALKFORWARD_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"

TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
# CALIBRATION_END / LOCKBOX_START from experiments.spans (config-driven)
M15_MODELS = ("catboost_balanced", "gru", "random_forest", "mlp", "ensemble4")


def cache_path(instrument: str, sentiment: str, model: str, tag: str) -> Path:
    return WALKFORWARD_DIR / f"{instrument}_{sentiment}_{model}_{tag}_{CACHE_SUFFIX}.parquet"


def default_candidates(instrument: str) -> list[tuple[str, str, str]]:
    """(model, sentiment, tag) candidates chosen on Q1 information only."""
    cands: list[tuple[str, str, str]] = []
    for kind in ("deadzone", "barrier"):
        path = OUT_DIR / f"{instrument}_both_sweep_{kind}_sortino_{CACHE_SUFFIX}.parquet"
        if not path.exists():
            continue
        summary = pd.read_parquet(path)
        cal_col = next((c for c in summary.columns if c.startswith("calibration_")), None)
        if cal_col is None:
            continue
        for model, grp in summary.groupby("model"):
            best = grp.loc[grp[cal_col].idxmax()]
            cands.append((str(model), "both", str(best["tag"])))
    for sentiment in ("both", "bothof"):
        for model in M15_MODELS:
            if cache_path(instrument, sentiment, model, "m15").exists():
                cands.append((model, sentiment, "m15"))
    # dz25 duplicates the m15 recipe (same label, retrained); keep only m15
    seen, out = set(), []
    for model, sentiment, tag in cands:
        if tag == "dz25" and (model, "both", "m15") in cands:
            continue
        key = (model, sentiment, tag)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def config_economics(
    df: pd.DataFrame, model: str, fee_bps: float, *,
    select_metric: str = "sortino",
) -> tuple[dict, pd.Series]:
    """Q1-calibrated, frozen, Q2-Q4-evaluated economics + the eval return series."""
    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    lockbox = pd.Timestamp(LOCKBOX_START, tz="UTC")
    calib = df[df.index < split]
    evaluation = df[(df.index >= split) & (df.index < lockbox)]
    if calib.empty or evaluation.empty:
        raise ValueError("calibration or evaluation period is empty")
    sweep = sweep_tau(calib[f"{model}_pred"], calib[f"{model}_conf"],
                      calib["forward_return"], fee_bps, TAUS)
    best = sweep.loc[sweep[select_metric].idxmax()]
    tau = float(best["tau"])
    returns = strategy_returns(evaluation[f"{model}_pred"], evaluation["forward_return"],
                               fee_bps, evaluation[f"{model}_conf"], tau)
    summary = economics_summary(returns, evaluation[f"{model}_pred"],
                                evaluation[f"{model}_conf"], tau)
    summary[f"calibration_{select_metric}"] = float(best[select_metric])
    # Q1 trade count of the chosen gate: a calibration score earned on a
    # handful of trades is noise, and downstream rules may demand evidence.
    summary["calibration_trades"] = int(best.get("trade_count", 0))
    return summary, returns


def select_members(rows: list[dict], *, cal_col: str, min_cal_score: float,
                   min_cal_trades: int = 0) -> list[dict]:
    """Membership rule — uses ONLY Q1 information (score and trade evidence)."""
    return [r for r in rows
            if r[cal_col] >= min_cal_score
            and r.get("calibration_trades", 0) >= min_cal_trades]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--configs", default=None,
                    help="comma list model:sentiment:tag (default: Q1-picked pool)")
    ap.add_argument("--select-metric", default="sortino", choices=["sortino", "sharpe"])
    ap.add_argument("--min-cal-score", type=float, default=0.0,
                    help="join the portfolio iff Q1 calibration score >= this")
    ap.add_argument("--min-cal-trades", type=int, default=0,
                    help="also require >= this many Q1 trades behind the score "
                         "(exploratory minimum-evidence variant; artifacts get "
                         "a _mint{N} suffix so the pre-registered rule stays)")
    ap.add_argument("--slippage-bps", type=float, default=0.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"][args.instrument]["taker_fee_bps"]) + args.slippage_bps
    cal_col = f"calibration_{args.select_metric}"

    if args.configs:
        candidates = [tuple(c.split(":")) for c in args.configs.split(",")]
    else:
        candidates = default_candidates(args.instrument)

    rows, streams = [], {}
    for model, sentiment, tag in candidates:
        path = cache_path(args.instrument, sentiment, model, tag)
        if not path.exists():
            print(f"  [skip] no cache: {path.name}")
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        summary, returns = config_economics(
            df.sort_index(), model, fee, select_metric=args.select_metric)
        name = f"{model}|{sentiment}|{tag}"
        summary.update({"config": name, "model": model, "sentiment": sentiment, "tag": tag})
        rows.append(summary)
        streams[name] = returns
        print(f"  {name}: cal {summary[cal_col]:+.2f} -> "
              f"eval sortino {summary['sortino']:+.2f} net {summary['net_return_sum']:+.4f} "
              f"trades {summary['trade_count']}")

    if not rows:
        raise SystemExit("no candidate caches found")

    members = select_members(rows, cal_col=cal_col, min_cal_score=args.min_cal_score,
                             min_cal_trades=args.min_cal_trades)
    if not members:
        raise SystemExit(f"no candidate clears {cal_col} >= {args.min_cal_score}")
    member_names = [m["config"] for m in members]
    print(f"\nportfolio members (Q1 rule: {cal_col} >= {args.min_cal_score}, "
          f"calibration_trades >= {args.min_cal_trades}): {member_names}")

    S = pd.DataFrame({n: streams[n] for n in member_names}).fillna(0.0)
    portfolio = S.mean(axis=1)                     # equal capital split across members
    port_summary = economics_summary(portfolio)
    port_summary.update({
        "config": "PORTFOLIO", "n_members": len(members),
        "trade_count": int(sum(m["trade_count"] for m in members)),
        "exposure": float((S != 0.0).any(axis=1).mean()),
    })

    best_single = max(members, key=lambda r: r[cal_col])   # Q1 choice again
    dm = diebold_mariano(streams[best_single["config"]], portfolio)
    port_summary.update({"vs_best_single": best_single["config"],
                         **{f"dm_{k}": v for k, v in dm.items()}})

    table = pd.DataFrame(rows + [port_summary])
    table["member"] = table["config"].isin(member_names + ["PORTFOLIO"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.instrument}_portfolio"
    if args.min_cal_trades > 0:
        stem += f"_mint{args.min_cal_trades}"
    table.to_parquet(OUT_DIR / f"{stem}_summary.parquet", index=False)
    S.assign(PORTFOLIO=portfolio).to_parquet(OUT_DIR / f"{stem}_returns.parquet")
    S.corr().to_parquet(OUT_DIR / f"{stem}_corr.parquet")

    cols = ["config", cal_col, "sortino", "sharpe", "net_return_sum", "max_drawdown",
            "trade_count", "exposure"]
    have = [c for c in cols if c in table.columns]
    print(f"\n{table[have].to_string(index=False)}")
    print(f"\nportfolio vs best single ({best_single['config']}): "
          f"DM {dm['dm_stat']:+.2f} (p={dm['p_value']:.3f}, positive favours the portfolio)")
    print(f"member return correlations:\n{S.corr().round(2).to_string()}")
    print(f"\nwrote artifacts -> {OUT_DIR / (stem + '_*.parquet')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
