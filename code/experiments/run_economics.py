"""Walk-forward economics for the RQ2 contenders (notebooks 03 and 06).

For each contender's configured walk-forward cache:
  1. Sweep the confidence threshold tau on the CALIBRATION quarter (Q1-2025) and
     keep the tau with the best calibration Sharpe.
  2. Evaluate CALIBRATION_END..LOCKBOX_START with that frozen tau — strictly out-of-sample for the policy.
  3. Report net-of-cost return, Sharpe, Sortino, max drawdown, trades, exposure.
Adds a Diebold-Mariano test of every contender against the best single model on
the evaluation-period net returns.

Run:  python -m experiments.run_economics --instrument btc --sentiment both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from evaluation.economics import diebold_mariano, economics_summary, strategy_returns, sweep_tau
from experiments.horizons import DEFAULT_LABEL, horizon_label, parse_horizon
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
WALKFORWARD_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"

CONTENDERS = ("catboost_balanced", "xgboost_balanced", "random_forest", "gru", "stack_all")
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
# CALIBRATION_END from experiments.spans (config-driven Q1-2025 freeze, unchanged)


def cache_file(instrument: str, sentiment: str, model: str, label: str) -> Path:
    return WALKFORWARD_DIR / f"{instrument}_{sentiment}_{model}_{label}_{CACHE_SUFFIX}.parquet"


def load_cache(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index, utc=True)
    need = {f"{model}_pred", f"{model}_conf", "forward_return"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing column(s): {sorted(missing)}")
    return df.sort_index()


def evaluate_model(df: pd.DataFrame, model: str, fee_bps: float) -> tuple[dict, pd.DataFrame, pd.Series]:
    """Calibrate tau on Q1, evaluate CALIBRATION_END..LOCKBOX_START; return summary, sweep, eval returns."""
    split = pd.Timestamp(CALIBRATION_END, tz="UTC")
    lockbox = pd.Timestamp(LOCKBOX_START, tz="UTC")
    calib = df[df.index < split]
    evaluation = df[(df.index >= split) & (df.index < lockbox)]
    if calib.empty or evaluation.empty:
        raise ValueError("calibration or evaluation period is empty")

    sweep = sweep_tau(calib[f"{model}_pred"], calib[f"{model}_conf"],
                      calib["forward_return"], fee_bps, TAUS)
    best_tau = float(sweep.loc[sweep["sharpe"].idxmax(), "tau"])

    eval_returns = strategy_returns(evaluation[f"{model}_pred"], evaluation["forward_return"],
                                    fee_bps, evaluation[f"{model}_conf"], best_tau)
    summary = economics_summary(eval_returns, evaluation[f"{model}_pred"],
                                evaluation[f"{model}_conf"], best_tau)
    # the ungated policy on the same evaluation window, for the "did tau help" column
    raw_returns = strategy_returns(evaluation[f"{model}_pred"], evaluation["forward_return"], fee_bps)
    summary.update({
        "model": model,
        "calibration_sharpe": float(sweep.loc[sweep["sharpe"].idxmax(), "sharpe"]),
        "raw_sharpe_no_tau": economics_summary(raw_returns)["sharpe"],
        "raw_net_return_no_tau": float(raw_returns.sum()),
    })
    return summary, sweep.assign(model=model), eval_returns


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    ap.add_argument("--horizon", type=parse_horizon, default=DEFAULT_LABEL)
    ap.add_argument("--models", default=",".join(CONTENDERS))
    args = ap.parse_args()

    label = horizon_label(args.horizon)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee_bps = float(cfg["instruments"][args.instrument]["taker_fee_bps"])
    models = [m.strip() for m in args.models.split(",")]

    summaries, sweeps, eval_returns = [], [], {}
    for model in models:
        path = cache_file(args.instrument, args.sentiment, model, label)
        if not path.exists():
            print(f"  [skip] {model}: no cache at {path.name}")
            continue
        df = load_cache(path, model)
        summary, sweep, returns = evaluate_model(df, model, fee_bps)
        summaries.append(summary)
        sweeps.append(sweep)
        eval_returns[model] = returns
        print(f"  {model}: tau={summary['tau']:.2f} eval Sharpe {summary['sharpe']:+.2f} "
              f"net {summary['net_return_sum']:+.4f} trades {summary['trade_count']:,} "
              f"exposure {summary['exposure']:.1%}")

    if not summaries:
        raise SystemExit("no walk-forward caches found — run experiments.run_walkforward first")

    table = pd.DataFrame(summaries).sort_values("sharpe", ascending=False).reset_index(drop=True)
    singles = table.loc[table["model"] != "stack_all", "model"]
    best_single = singles.iloc[0] if len(singles) else table.iloc[0]["model"]
    dm_rows = []
    for model, returns in eval_returns.items():
        if model == best_single:
            continue
        dm = diebold_mariano(eval_returns[best_single], returns)
        dm_rows.append({"model": model, "vs": best_single, **dm})
    dm_table = pd.DataFrame(dm_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.instrument}_{args.sentiment}_{label}_{CACHE_SUFFIX}"
    table.to_parquet(OUT_DIR / f"{stem}_summary.parquet", index=False)
    pd.concat(sweeps).to_parquet(OUT_DIR / f"{stem}_tau_sweep.parquet", index=False)
    if not dm_table.empty:
        dm_table.to_parquet(OUT_DIR / f"{stem}_dm.parquet", index=False)
    returns_frame = pd.DataFrame(eval_returns)
    returns_frame.to_parquet(OUT_DIR / f"{stem}_eval_returns.parquet")

    cols = ["model", "tau", "sharpe", "sortino", "net_return_sum", "max_drawdown",
            "trade_count", "exposure", "raw_sharpe_no_tau"]
    print(f"\n{table[cols].to_string(index=False)}")
    if not dm_table.empty:
        print(f"\nDiebold-Mariano vs best single ({best_single}):")
        print(dm_table.to_string(index=False))
    print(f"\nwrote artifacts -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
