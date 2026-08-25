"""Run the current anti-bull dz75 model and compare it with frozen dz40.

The dz75 model uses the 2024-only regime-robust parameters and, during every
weekly 90-day refit, gives bull, sideways, and bear samples equal aggregate
weight. Each model then selects its own confidence threshold on 2025-H1 net
Sortino with a 50-trade floor. Execution is fixed at TP150/SL75/one M15 bar and
resolved from corrected 1-minute data. Q2-2026 remains sealed.

Run:  python -m experiments.run_walkforward_dz75_regime
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from evaluation.economics import economics_summary
from evaluation.trades import trade_stats
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.run_tune_dz75_regime import (
    REGIMES,
    past_regime_labels,
    regime_balanced_weights,
)
from experiments.run_walkforward import build_walkforward_xy
from experiments.spans import CACHE_SUFFIX, CALIBRATION_END, LOCKBOX_START
from experiments.walkforward import run_walkforward_predictions, weekly_walkforward_windows
from features.build import make_label
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
TUNING_PATH = (
    CODE_ROOT / "experiments" / "cache" / "tuning"
    / "btc_bothofpos_dz75_regime_catboost_balanced.json"
)
WF_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
ECON_DIR = CODE_ROOT / "experiments" / "cache" / "economics"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2024_2026.parquet"
DZ40_PATH = WF_DIR / f"btc_bothofpos_cb-econ_dz40_lb90d_{CACHE_SUFFIX}.parquet"
SUMMARY_PATH = ECON_DIR / "btc_balanced_dz40_vs_dz75_regime_intrabar.parquet"
GRID_PATH = ECON_DIR / "btc_balanced_dz40_vs_dz75_regime_tau_grid.parquet"

MODEL = "catboost_balanced"
WIDTH = 75
LOOKBACK_DAYS = 90
TP_BPS = 150.0
SL_BPS = 75.0
MAX_HOLD = 1
FLOOR = 50
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
SPLIT = pd.Timestamp(CALIBRATION_END, tz="UTC")
LOCKBOX = pd.Timestamp(LOCKBOX_START, tz="UTC")


def prediction_cache_path() -> Path:
    return WF_DIR / f"btc_bothofpos_cb-regime_dz75_lb90d_{CACHE_SUFFIX}.parquet"


def load_tuned_params() -> dict:
    return json.loads(TUNING_PATH.read_text(encoding="utf-8"))["best_params"]


def select_tau(grid: pd.DataFrame, *, floor: int = FLOOR) -> pd.Series:
    eligible = grid[grid["cal_trades"] >= floor]
    if eligible.empty:
        raise ValueError(f"no threshold meets the {floor}-trade floor")
    return eligible.sort_values(
        ["cal_sortino", "cal_trades", "tau"],
        ascending=[False, False, True],
    ).iloc[0]


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    out = prediction_cache_path()
    if out.exists():
        print(f"[cached] {out.name}")
    else:
        X, y, aux = build_walkforward_xy(
            "btc",
            cfg,
            horizon=1,
            sentiment="both",
            label_fn=lambda feat: make_label(feat, threshold_bps=WIDTH, horizon=1),
            orderflow=True,
            positioning=True,
        )
        bars_for_regime = pd.read_parquet(
            CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"]
        )
        bars_for_regime.index = pd.to_datetime(bars_for_regime.index, utc=True)
        regimes = past_regime_labels(
            bars_for_regime.sort_index()["close"]
        ).reindex(X.index)
        walk_start, walk_end = cfg["dates"]["walkforward"]
        windows = weekly_walkforward_windows(
            X.index,
            walk_start=walk_start,
            walk_end=walk_end,
            train_lookback=f"{LOOKBACK_DAYS}D",
        )

        def training_weights(X_train: pd.DataFrame, _y_train: pd.Series):
            window_regimes = regimes.reindex(X_train.index)
            if not window_regimes.isin(REGIMES).all():
                raise ValueError("rolling training window contains unknown regimes")
            return regime_balanced_weights(window_regimes)

        preds = run_walkforward_predictions(
            X,
            y,
            windows=windows,
            model_factory=MODELS[MODEL],
            model_name=MODEL,
            params=load_tuned_params(),
            min_train_rows=500,
            min_validation_rows=50,
            progress_label="dz75:regime-balanced:90D",
            train_tail_trim=1,
            sample_weight_fn=training_weights,
        )
        if preds.empty:
            raise RuntimeError("no dz75 regime-balanced walk-forward predictions")
        preds = preds.join(aux[["forward_return", "vol_regime"]], how="left")
        out.parent.mkdir(parents=True, exist_ok=True)
        preds.to_parquet(out)
        print(f"wrote {len(preds):,} rows -> {out.name}")

    minute = pd.read_parquet(MINUTE_PATH)
    minute.index = pd.to_datetime(minute.index, utc=True)
    minute = minute.sort_index().loc[lambda d: d.index < LOCKBOX]
    bars = pd.read_parquet(CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"])
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index().loc[lambda d: d.index < LOCKBOX]
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    hold_path = pd.Timedelta(minutes=15 * (MAX_HOLD + 1))
    cache_paths = {"dz40": DZ40_PATH, "dz75_regime": out}
    grid_rows: list[dict] = []
    result_rows: list[dict] = []
    quarters = {
        "2025Q3": (pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC")),
        "2025Q4": (pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        "2026Q1": (pd.Timestamp("2026-01-01", tz="UTC"), LOCKBOX),
    }

    for variant, cache_path in cache_paths.items():
        predictions = pd.read_parquet(cache_path)
        predictions.index = pd.to_datetime(predictions.index, utc=True)
        predictions = predictions.sort_index().loc[lambda d: d.index < LOCKBOX]
        pred = predictions[f"{MODEL}_pred"].astype(int)
        conf = predictions[f"{MODEL}_conf"].astype(float)
        cal_mask = (pred.index + hold_path <= SPLIT)
        eval_mask = (pred.index >= SPLIT) & (pred.index + hold_path <= LOCKBOX)
        cal_bars = bars[(bars.index >= pred.index.min()) & (bars.index < SPLIT)]
        eval_bars = bars[(bars.index >= SPLIT) & (bars.index < LOCKBOX)]

        variant_grid = []
        for tau in TAUS:
            ledger, per_bar = simulate_bracket_trades_intrabar(
                cal_bars,
                minute,
                pred.loc[cal_mask],
                conf.loc[cal_mask],
                tau=tau,
                tp_bps=TP_BPS,
                sl_bps=SL_BPS,
                max_hold=MAX_HOLD,
                fee_bps=fee,
            )
            row = {
                "variant": variant,
                "tau": tau,
                "cal_sortino": economics_summary(per_bar)["sortino"],
                "cal_gross": float(ledger["gross_return"].sum()),
                "cal_net": float(ledger["net_return"].sum()),
                "cal_trades": len(ledger),
            }
            variant_grid.append(row)
            grid_rows.append(row)
        chosen = select_tau(pd.DataFrame(variant_grid))
        tau = float(chosen["tau"])
        ledger, per_bar = simulate_bracket_trades_intrabar(
            eval_bars,
            minute,
            pred.loc[eval_mask],
            conf.loc[eval_mask],
            tau=tau,
            tp_bps=TP_BPS,
            sl_bps=SL_BPS,
            max_hold=MAX_HOLD,
            fee_bps=fee,
        )
        summary = economics_summary(per_bar)
        stats = trade_stats(ledger)
        row = {
            "variant": variant,
            "tau": tau,
            "cal_sortino": float(chosen["cal_sortino"]),
            "cal_trades": int(chosen["cal_trades"]),
            "eval_gross": float(ledger["gross_return"].sum()),
            "eval_net": float(ledger["net_return"].sum()),
            "eval_sortino": summary["sortino"],
            "eval_sharpe": summary["sharpe"],
            "eval_trades": len(ledger),
            "gross_per_trade_bps": (
                float(ledger["gross_return"].mean() * 10_000.0) if len(ledger) else 0.0
            ),
            "long_net": float(ledger.loc[ledger["side"] == 1, "net_return"].sum()),
            "short_net": float(ledger.loc[ledger["side"] == -1, "net_return"].sum()),
            **stats,
        }
        entry_time = pd.to_datetime(ledger["entry_time"], utc=True)
        positive = 0
        for quarter, (start, end) in quarters.items():
            q = ledger[(entry_time >= start) & (entry_time < end)]
            qgross = float(q["gross_return"].sum())
            row[f"{quarter}_gross"] = qgross
            row[f"{quarter}_net"] = float(q["net_return"].sum())
            row[f"{quarter}_trades"] = len(q)
            positive += int(qgross > 0.0)
        row["positive_gross_quarters"] = positive
        row["screen_pass"] = bool(
            row["eval_gross"] > 0.0
            and row["eval_net"] > 0.0
            and row["eval_trades"] >= FLOOR
            and positive >= 2
            and row["long_net"] > 0.0
            and row["short_net"] > 0.0
        )
        result_rows.append(row)
        print(
            f"{variant}: tau={tau:.2f} cal Sortino={row['cal_sortino']:+.3f} "
            f"({row['cal_trades']} trades)"
        )

    table = pd.DataFrame(result_rows)
    ECON_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(SUMMARY_PATH, index=False)
    pd.DataFrame(grid_rows).to_parquet(GRID_PATH, index=False)
    columns = [
        "variant", "tau", "cal_sortino", "cal_trades", "eval_gross", "eval_net",
        "eval_sortino", "eval_sharpe", "eval_trades", "gross_per_trade_bps",
        "long_net", "short_net", "2025Q3_gross", "2025Q3_net", "2025Q4_gross",
        "2025Q4_net", "2026Q1_gross", "2026Q1_net", "positive_gross_quarters",
        "screen_pass",
    ]
    pd.set_option("display.width", 260)
    print(table[columns].to_string(index=False))
    print(f"wrote -> {SUMMARY_PATH.name}, {GRID_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
