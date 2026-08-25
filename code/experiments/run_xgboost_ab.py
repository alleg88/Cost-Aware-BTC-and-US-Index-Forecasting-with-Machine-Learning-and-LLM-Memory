"""A/B the resolution-retuned XGBoost against its baseline and random_forest.

The multi-objective retune (experiments.run_tuning --objective multi) writes
`..._xgboost_balanced__multi.json`. This script walk-forwards an XGBoost carrying
those params across the dead-zone threshold stack (order-flow features, same
protocol as the sweep), computes the frozen-tau economics, and reports whether
the resolution-fixed model now beats random_forest's best Sortino — the current
holder of the 4th ensemble seat.

Cache/model tag: `xgb_mo` (multi-objective), so it never collides with the
baseline `xgboost_balanced` caches. Resumable.

Run:  python -m experiments.run_xgboost_ab
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from experiments.run_threshold_sweep import (
    THRESHOLDS, cache_path, calibrated_economics, sweep_tag,
)
from experiments.run_walkforward import build_walkforward_xy
from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import make_label
from models.zoo import make_xgboost

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
TUNING = CODE_ROOT / "experiments" / "cache" / "tuning"
ECON = CODE_ROOT / "experiments" / "cache" / "economics"
TAG_MODEL = "xgb_mo"


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    params = json.loads(
        (TUNING / "btc_both_m15_xgboost_balanced__multi.json").read_text())["best_params"]
    print(f"resolution-tuned XGBoost params: {params}")

    rows = []
    for thr in THRESHOLDS:
        tag = sweep_tag("deadzone", thr, 16)
        out = cache_path("btc", "bothof", TAG_MODEL, tag)
        if not out.exists():
            X, y, aux = build_walkforward_xy(
                "btc", cfg, horizon=1, sentiment="both", orderflow=True,
                label_fn=lambda feat, t=thr: make_label(feat, threshold_bps=t, horizon=1))
            ws, we = cfg["dates"]["walkforward"]
            windows = weekly_walkforward_windows(X.index, walk_start=ws, walk_end=we,
                                                 train_lookback="180D")
            preds = run_walkforward_predictions(
                X, y, windows=windows, model_factory=make_xgboost,
                model_name=TAG_MODEL, params=params, min_train_rows=500,
                min_validation_rows=50, include_features=True,
                progress_label=f"{tag}:{TAG_MODEL}", train_tail_trim=1)
            preds = preds.join(aux[["forward_return", "vol_regime"]], how="left")
            out.parent.mkdir(parents=True, exist_ok=True)
            preds.to_parquet(out)
            print(f"  wrote {len(preds):,} rows -> {out.name}")
        else:
            print(f"  [cached] {out.name}")

        df = pd.read_parquet(out)
        df.index = pd.to_datetime(df.index, utc=True)
        s = calibrated_economics(df.sort_index(), TAG_MODEL, fee)
        s["threshold_bps"] = thr
        rows.append(s)
        print(f"  {tag}: tau={s['tau']:.2f} Sortino {s['sortino']:+.2f} "
              f"Sharpe {s['sharpe']:+.2f} F1 {s['macro_f1']:.3f} "
              f"net {s['net_return_sum']:+.4f} trades {s['trade_count']}")

    ab = pd.DataFrame(rows)
    ECON.mkdir(parents=True, exist_ok=True)
    ab.to_parquet(ECON / "btc_bothof_xgb_mo_ab_2025.parquet", index=False)

    # compare best configs against the baseline sweep
    sweep = pd.read_parquet(ECON / "btc_bothof_sweep_deadzone_sortino_2025.parquet")
    def best(model_df):
        r = model_df.sort_values("sortino", ascending=False).iloc[0]
        return r["sortino"], r["sharpe"], r["macro_f1"], r["net_return_sum"], int(r["trade_count"]), r["threshold_bps"]
    mo = best(ab)
    xgb0 = best(sweep[sweep["model"] == "xgboost_balanced"])
    rf = best(sweep[sweep["model"] == "random_forest"])

    print("\n=== best config by Sortino (the 4th-seat contest) ===")
    hdr = f"{'variant':<22}{'bps':>5}{'sortino':>10}{'sharpe':>9}{'f1':>8}{'net':>10}{'trades':>8}"
    print(hdr)
    for name, b in [("xgboost (retuned MO)", mo), ("xgboost (baseline)", xgb0),
                    ("random_forest (seat)", rf)]:
        print(f"{name:<22}{b[5]:>5.0f}{b[0]:>10.2f}{b[1]:>9.2f}{b[2]:>8.3f}{b[3]:>+10.4f}{b[4]:>8d}")
    verdict = ("TAKES the seat from random_forest" if mo[0] > rf[0]
               else "does NOT beat random_forest — seat unchanged")
    print(f"\nverdict: resolution-tuned XGBoost {verdict} "
          f"(Sortino {mo[0]:+.2f} vs rf {rf[0]:+.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
