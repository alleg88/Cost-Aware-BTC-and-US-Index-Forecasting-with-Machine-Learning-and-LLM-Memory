"""Apply accepted reflection lessons to later walk-forward windows."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from experiments.run_reflection import (
    DEFAULT_OUT_DIR,
    apply_lessons_out_of_sample,
    default_cache_path,
    default_output_path,
    load_predictions,
    paired_oos_significance_frame,
    summarize_oos_memory,
    summarize_oos_windows,
    wf_cols,
    wf_feature_columns,
)


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def default_oos_path(
    instrument: str,
    sentiment: str,
    horizon: int,
    model: str,
    max_harmful_windows: int | None = None,
    retention_metric: str = "macro_f1",
    wf_model: str = "catboost",
    tau: float = 0.0,
) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    suffix = f"_forget{max_harmful_windows}" if max_harmful_windows else ""
    if wf_model != "catboost":
        suffix = f"_{wf_model}{suffix}"
    if tau > 0.0:
        suffix = f"{suffix}_tau{int(round(tau * 100)):02d}"
    if max_harmful_windows:
        metric_suffix = "netret" if retention_metric == "net_return" else "macrof1"
        suffix = f"{suffix}_{metric_suffix}"
    from experiments.horizons import horizon_label

    return DEFAULT_OUT_DIR / f"{instrument}_{sentiment}_{horizon_label(horizon)}_{safe_model}_oos_memory{suffix}_2025.parquet"


def resolve_tau(raw: str, instrument: str, sentiment: str, horizon: int,
                wf_model: str) -> float:
    """A float, or 'auto' = the tau the economics run calibrated on Q1 for this model."""
    if str(raw).lower() != "auto":
        return float(raw)
    from experiments.horizons import horizon_label

    path = (Path(__file__).resolve().parent / "cache" / "economics"
            / f"{instrument}_{sentiment}_{horizon_label(horizon)}_2025_summary.parquet")
    summary = pd.read_parquet(path)
    row = summary.loc[summary["model"] == wf_model]
    if row.empty:
        raise SystemExit(f"no economics summary row for {wf_model} in {path.name}; "
                         f"run experiments.run_economics first or pass --tau explicitly")
    return float(row.iloc[0]["tau"])


def default_fee_bps(instrument: str) -> float:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    inst = cfg.get("instruments", {}).get(instrument, {})
    return float(inst.get("taker_fee_bps", cfg.get("taker_fee_bps", 0.0)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    from experiments.horizons import DEFAULT_LABEL, parse_horizon
    ap.add_argument("--horizon", type=parse_horizon, default=DEFAULT_LABEL,
                    help="m15 (default) / h1 / h4, or a bar count")
    ap.add_argument("--model", default="glm-5.2:cloud")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--lessons", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fee-bps", type=float, default=None)
    ap.add_argument("--max-harmful-windows", type=int, default=None)
    ap.add_argument("--wf-model", default="catboost",
                    help="walk-forward cache/model tag the lessons were learned on")
    ap.add_argument("--tau", default="0",
                    help="confidence gate: a float, or 'auto' to read the tau the "
                         "economics run calibrated for this wf-model")
    ap.add_argument("--retention-metric", choices=["macro_f1", "net_return"], default="macro_f1")
    args = ap.parse_args()

    cache_path = Path(args.cache) if args.cache else default_cache_path(
        args.instrument, args.sentiment, args.horizon, args.wf_model)
    lessons_path = Path(args.lessons) if args.lessons else default_output_path(
        args.instrument,
        args.sentiment,
        args.horizon,
        args.model,
        args.wf_model,
    )
    fee_bps = default_fee_bps(args.instrument) if args.fee_bps is None else float(args.fee_bps)
    tau = resolve_tau(args.tau, args.instrument, args.sentiment, args.horizon, args.wf_model)
    out_path = Path(args.out) if args.out else default_oos_path(
        args.instrument,
        args.sentiment,
        args.horizon,
        args.model,
        args.max_harmful_windows,
        args.retention_metric,
        args.wf_model,
        tau,
    )

    predictions = load_predictions(cache_path)
    lessons = pd.read_parquet(lessons_path)
    feature_columns = wf_feature_columns(predictions, args.wf_model)
    cols = wf_cols(args.wf_model)
    gated = apply_lessons_out_of_sample(
        predictions,
        lessons,
        feature_columns=feature_columns,
        pred_col=cols["pred"],
        conf_col=cols["conf"],
        max_harmful_windows=args.max_harmful_windows,
        retention_metric=args.retention_metric,
        fee_bps=fee_bps,
    )
    if gated.empty:
        raise RuntimeError("no out-of-sample rows produced")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gated.to_parquet(out_path)
    metrics = summarize_oos_memory(gated, fee_bps=fee_bps, tau=tau)
    metrics_path = out_path.with_suffix(".metrics.parquet")
    metrics.to_parquet(metrics_path, index=False)
    windows = summarize_oos_windows(gated, fee_bps=fee_bps, tau=tau)
    windows_path = out_path.with_suffix(".windows.parquet")
    windows.to_parquet(windows_path, index=False)
    significance = paired_oos_significance_frame(windows)
    significance_path = out_path.with_suffix(".significance.parquet")
    significance.to_parquet(significance_path, index=False)

    baseline = metrics.loc[metrics["variant"] == "baseline"].iloc[0]
    memory = metrics.loc[metrics["variant"] == "memory"].iloc[0]
    print(f"wrote {len(gated):,} OOS memory rows -> {out_path}")
    print(f"wrote metrics -> {metrics_path}")
    print(f"wrote weekly metrics -> {windows_path}")
    print(f"wrote significance -> {significance_path}")
    print(
        "macro_f1 "
        f"baseline={baseline['macro_f1']:.6f} "
        f"memory={memory['macro_f1']:.6f} "
        f"delta={memory['macro_f1'] - baseline['macro_f1']:.6f} "
        f"p={significance.loc[significance['metric'] == 'macro_f1', 'wilcoxon_pvalue'].iloc[0]:.6f}"
    )
    print(
        "net_return_sum "
        f"baseline={baseline.get('net_return_sum', 0.0):.6f} "
        f"memory={memory.get('net_return_sum', 0.0):.6f} "
        f"delta={memory.get('net_return_sum', 0.0) - baseline.get('net_return_sum', 0.0):.6f}"
    )
    net_p = significance.loc[significance['metric'] == 'net_return_sum', 'wilcoxon_pvalue'].iloc[0]
    print(f"net_return_wilcoxon_p={net_p:.6f}")
    print(
        "baseline_return_split_bps "
        f"flattened={memory.get('flattened_baseline_return_mean_bps', float('nan')):.3f} "
        f"kept={memory.get('kept_directional_baseline_return_mean_bps', float('nan')):.3f}"
    )
    from evaluation.economics import diebold_mariano, strategy_returns

    conf = gated["conf"] if (tau > 0.0 and "conf" in gated.columns) else None
    r_base = strategy_returns(gated["baseline_pred"], gated["forward_return"], fee_bps, conf, tau)
    r_mem = strategy_returns(gated["memory_pred"], gated["forward_return"], fee_bps, conf, tau)
    dm = diebold_mariano(r_base, r_mem)
    print(f"tau={tau:g} (baseline AND memory both gated)")
    print(f"DM gate-alone vs gate+memory: stat={dm['dm_stat']:+.3f} p={dm['p_value']:.4f}")
    print(f"changed_rows={int(memory['changed_rows'])} fee_bps={fee_bps:g} retention_metric={args.retention_metric}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
