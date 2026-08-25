"""Balanced CatBoost dz40 sensitivity to rolling training-history length.

Only the 60D and 90D caches are new; the frozen 180D economic-arm cache is
reused exactly. Gates are calibrated before 2025-07-01 and evaluated on
2025-Q3 through 2026-Q1. The Q2-2026 lockbox remains sealed.

Run:  python -m experiments.run_balanced_lookbacks
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from evaluation.economics import economics_summary, strategy_returns
from experiments.run_arms_economics import (
    LOCKBOX,
    MODEL,
    SPLIT,
    apply_regime,
    calibrate_global,
    calibrate_regime,
    funding_z,
)
from experiments.run_walkforward import build_walkforward_xy
from experiments.run_walkforward_arms import (
    CONFIG,
    WF_DIR,
    arm_params,
    cache_path as arm_cache_path,
)
from experiments.spans import CACHE_SUFFIX
from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import make_label
from models.zoo import MODELS

WIDTH = 40
LOOKBACK_DAYS = (60, 90, 180)
ECON_DIR = Path(__file__).resolve().parent / "cache" / "economics"
OUTPUT = ECON_DIR / "btc_balanced_dz40_lookback_eval.parquet"


def lookback_cache_path(days: int) -> Path:
    if days not in LOOKBACK_DAYS:
        raise ValueError(f"unsupported lookback: {days}D")
    if days == 180:
        return arm_cache_path("econ", WIDTH)
    return WF_DIR / f"btc_bothofpos_cb-econ_dz40_lb{days}d_{CACHE_SUFFIX}.parquet"


def evaluation_quarter_masks(index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    starts = {
        "2025Q3": pd.Timestamp("2025-07-01", tz="UTC"),
        "2025Q4": pd.Timestamp("2025-10-01", tz="UTC"),
        "2026Q1": pd.Timestamp("2026-01-01", tz="UTC"),
    }
    ends = {
        "2025Q3": pd.Timestamp("2025-10-01", tz="UTC"),
        "2025Q4": pd.Timestamp("2026-01-01", tz="UTC"),
        "2026Q1": LOCKBOX,
    }
    return {name: (idx >= start) & (idx < ends[name]) for name, start in starts.items()}


def _utc(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def build_missing_caches(cfg: dict) -> None:
    X, y, aux = build_walkforward_xy(
        "btc",
        cfg,
        horizon=1,
        sentiment="both",
        label_fn=lambda feat: make_label(feat, threshold_bps=WIDTH, horizon=1),
        orderflow=True,
        positioning=True,
    )
    walk_start, walk_end = cfg["dates"]["walkforward"]
    for days in LOOKBACK_DAYS:
        out = lookback_cache_path(days)
        if out.exists():
            print(f"[cached] {out.name}")
            continue
        windows = weekly_walkforward_windows(
            X.index,
            walk_start=walk_start,
            walk_end=walk_end,
            train_lookback=f"{days}D",
        )
        preds = run_walkforward_predictions(
            X,
            y,
            windows=windows,
            model_factory=MODELS[MODEL],
            model_name=MODEL,
            params=arm_params("econ", WIDTH),
            min_train_rows=500,
            min_validation_rows=50,
            include_features=False,
            progress_label=f"dz{WIDTH}:balanced:{days}D",
            train_tail_trim=1,
        )
        if preds.empty:
            raise RuntimeError(f"no predictions for {days}D lookback")
        preds = preds.join(aux[["forward_return", "vol_regime"]], how="left")
        out.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out)
        print(f"wrote {len(preds):,} rows -> {out.name}")


def _confidence_rho(pred: pd.Series, conf: pd.Series, fwd: pd.Series) -> float:
    directional = pred != 1
    signed = fwd.where(pred == 2, -fwd)
    result = spearmanr(conf[directional], signed[directional], nan_policy="omit")
    return float(result.statistic) if np.isfinite(result.statistic) else np.nan


def evaluate(cfg: dict) -> pd.DataFrame:
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    rows: list[dict] = []
    for days in LOOKBACK_DAYS:
        d = _utc(pd.read_parquet(lookback_cache_path(days)))
        pred = d[f"{MODEL}_pred"].astype(int)
        conf = d[f"{MODEL}_conf"].astype(float)
        fwd = d["forward_return"].astype(float)
        fz = funding_z(d.index)
        cal = d.index < SPLIT
        ev = (d.index >= SPLIT) & (d.index < LOCKBOX)

        gates: dict[str, dict] = {}
        global_gate = calibrate_global(pred[cal], conf[cal], fwd[cal], fee)
        if global_gate:
            tau, cal_sortino = global_gate
            gates["global tau"] = {
                "sig": pred.where(conf >= tau, 1),
                "knobs": f"tau={tau:.2f}",
                "cal_sortino": cal_sortino,
            }
        regime_gate = calibrate_regime(
            pred[cal], conf[cal], fwd[cal], fee, (fz.abs() > 1)[cal]
        )
        if regime_gate:
            taus, cal_sortino = regime_gate
            gates["E1 funding-regime"] = {
                "sig": apply_regime(pred, conf, fz.abs() > 1, taus),
                "knobs": "tau " + "/".join(f"{value:.2f}" for value in taus.values()),
                "cal_sortino": cal_sortino,
            }

        rho = _confidence_rho(pred[ev], conf[ev], fwd[ev])
        quarters = evaluation_quarter_masks(d.index)
        for gate_name, gate in gates.items():
            signal = gate["sig"]
            gross = strategy_returns(signal[ev], fwd[ev], 0.0)
            net = strategy_returns(signal[ev], fwd[ev], fee)
            summary = economics_summary(net, signal[ev])
            long_net = strategy_returns(signal[ev].where(signal[ev] == 2, 1), fwd[ev], fee)
            short_net = strategy_returns(signal[ev].where(signal[ev] == 0, 1), fwd[ev], fee)
            row = {
                "lookback_days": days,
                "gate": gate_name,
                "knobs": gate["knobs"],
                "cal_sortino": gate["cal_sortino"],
                "eval_gross": float(gross.sum()),
                "eval_net": float(net.sum()),
                "eval_sortino": summary["sortino"],
                "eval_events": int(summary["trade_count"]),
                "long_net": float(long_net.sum()),
                "short_net": float(short_net.sum()),
                "confidence_rho": rho,
            }
            positive_quarters = 0
            for quarter, mask in quarters.items():
                qmask = mask & ev
                qgross = strategy_returns(signal[qmask], fwd[qmask], 0.0)
                qnet = strategy_returns(signal[qmask], fwd[qmask], fee)
                row[f"{quarter}_gross"] = float(qgross.sum())
                row[f"{quarter}_net"] = float(qnet.sum())
                positive_quarters += int(qgross.sum() > 0)
            row["positive_gross_quarters"] = positive_quarters
            row["screen_pass"] = bool(
                row["eval_gross"] > 0
                and row["eval_net"] > 0
                and positive_quarters >= 2
                and row["eval_events"] >= 50
                and rho > 0
            )
            rows.append(row)
        print(f"done Balanced dz40 {days}D: gates={list(gates)}")

    table = pd.DataFrame(rows)
    table["picked"] = ""
    if not table.empty:
        pick_idx = table.groupby("lookback_days")["cal_sortino"].idxmax()
        table.loc[pick_idx, "picked"] = "*"
    ECON_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(OUTPUT, index=False)
    return table


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    build_missing_caches(cfg)
    table = evaluate(cfg)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", None)
    print(table.to_string(index=False))
    print(f"wrote {len(table)} rows -> {OUTPUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
