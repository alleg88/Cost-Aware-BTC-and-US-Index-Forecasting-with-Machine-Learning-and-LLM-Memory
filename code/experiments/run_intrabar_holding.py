"""Corrected-1m longer-hold and asymmetric execution on frozen BTC dz40 signals.

Stage 1 selects one symmetric TP/SL/hold on calibration. Stage 2 freezes the
hold and selects long and short TP/SL independently, each with the 50-trade
floor. Both policies are then evaluated once on 2025-Q3 through 2026-Q1.
Q2-2026 remains sealed.

Run:  python -m experiments.run_intrabar_holding
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import diebold_mariano, economics_summary
from evaluation.trades import trade_stats
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.spans import CALIBRATION_END, LOCKBOX_START

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2024_2026.parquet"
PREDICTION_PATH = (
    CODE_ROOT / "experiments" / "cache" / "walkforward"
    / "btc_bothofpos_cb-econ_dz40_lb90d_to2026.parquet"
)
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"
SUMMARY_PATH = OUT_DIR / "btc_balanced_90d_intrabar_holding_summary.parquet"
GRID_PATH = OUT_DIR / "btc_balanced_90d_intrabar_holding_grid.parquet"
RETURN_PATH = OUT_DIR / "btc_balanced_90d_intrabar_holding_returns.parquet"

MODEL = "catboost_balanced"
PRIMARY_TAU = 0.60
FLOOR = 50
HOLD_GRID = (4, 8, 16)
TP_GRID = (100.0, 150.0, 200.0, 300.0)
SL_GRID = (50.0, 75.0, 100.0)
SPLIT = pd.Timestamp(CALIBRATION_END, tz="UTC")
LOCKBOX = pd.Timestamp(LOCKBOX_START, tz="UTC")


def path_safe_mask(index: pd.DatetimeIndex, *, hold: int, boundary: pd.Timestamp):
    """Require the next-open entry and full holding path to close by boundary."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return idx + pd.Timedelta(minutes=15 * (hold + 1)) <= boundary


def select_best_geometry(grid: pd.DataFrame, *, floor: int = FLOOR) -> pd.Series:
    """Select calibration Sortino only among geometries meeting the trade floor."""
    eligible = grid[grid["cal_trades"] >= floor]
    if eligible.empty:
        raise ValueError(f"no geometry meets the {floor}-trade floor")
    return eligible.sort_values("cal_sortino", ascending=False).iloc[0]


def main() -> int:
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])

    minute = pd.read_parquet(MINUTE_PATH)
    minute.index = pd.to_datetime(minute.index, utc=True)
    minute = minute.sort_index()
    minute = minute[minute.index < LOCKBOX]

    bars = pd.read_parquet(CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"])
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index()
    bars = bars[bars.index < LOCKBOX]

    predictions = pd.read_parquet(PREDICTION_PATH)
    predictions.index = pd.to_datetime(predictions.index, utc=True)
    predictions = predictions.sort_index()
    predictions = predictions[predictions.index < LOCKBOX]
    pred = predictions[f"{MODEL}_pred"].astype(int)
    conf = predictions[f"{MODEL}_conf"].astype(float)
    signals = pred.where(conf >= PRIMARY_TAU, 1)
    bars = bars[bars.index >= predictions.index.min()]
    bars_cal = bars[bars.index < SPLIT]
    bars_eval = bars[(bars.index >= SPLIT) & (bars.index < LOCKBOX)]

    def scoped_signals(hold: int, boundary: pd.Timestamp, *, start=None):
        mask = path_safe_mask(signals.index, hold=hold, boundary=boundary)
        if start is not None:
            mask &= signals.index >= start
        return signals.loc[mask]

    def run_policy(
        scope_bars: pd.DataFrame,
        scope_signals: pd.Series,
        *,
        tp_long: float,
        sl_long: float,
        hold: int,
        tp_short: float | None = None,
        sl_short: float | None = None,
    ):
        return simulate_bracket_trades_intrabar(
            scope_bars,
            minute,
            scope_signals,
            None,
            tau=0.0,
            tp_bps=tp_long,
            sl_bps=sl_long,
            max_hold=hold,
            fee_bps=fee,
            tp_bps_short=tp_short,
            sl_bps_short=sl_short,
        )

    grid_rows: list[dict] = []
    for hold, tp, sl in product(HOLD_GRID, TP_GRID, SL_GRID):
        cal_signals = scoped_signals(hold, SPLIT)
        ledger, per_bar = run_policy(
            bars_cal, cal_signals, tp_long=tp, sl_long=sl, hold=hold
        )
        grid_rows.append(
            {
                "stage": "symmetric",
                "side": "both",
                "tp_bps": tp,
                "sl_bps": sl,
                "max_hold": hold,
                "cal_sortino": economics_summary(per_bar)["sortino"],
                "cal_gross": float(ledger["gross_return"].sum()),
                "cal_net": float(ledger["net_return"].sum()),
                "cal_trades": len(ledger),
            }
        )
    grid = pd.DataFrame(grid_rows)
    symmetric = select_best_geometry(grid)
    hold = int(symmetric["max_hold"])
    symmetric_tp = float(symmetric["tp_bps"])
    symmetric_sl = float(symmetric["sl_bps"])
    print(
        f"symmetric calibration: hold={hold} tp/sl={symmetric_tp:g}/{symmetric_sl:g} "
        f"Sortino={symmetric['cal_sortino']:+.3f}, trades={int(symmetric['cal_trades'])}"
    )

    side_choices: dict[str, tuple[float, float]] = {}
    cal_signals = scoped_signals(hold, SPLIT)
    for side_name, class_label in (("long", 2), ("short", 0)):
        side_signal = cal_signals.where(cal_signals == class_label, 1)
        rows = []
        for tp, sl in product(TP_GRID, SL_GRID):
            ledger, per_bar = run_policy(
                bars_cal, side_signal, tp_long=tp, sl_long=sl, hold=hold
            )
            row = {
                "stage": "side",
                "side": side_name,
                "tp_bps": tp,
                "sl_bps": sl,
                "max_hold": hold,
                "cal_sortino": economics_summary(per_bar)["sortino"],
                "cal_gross": float(ledger["gross_return"].sum()),
                "cal_net": float(ledger["net_return"].sum()),
                "cal_trades": len(ledger),
            }
            rows.append(row)
            grid_rows.append(row)
        side_grid = pd.DataFrame(rows)
        try:
            selected = select_best_geometry(side_grid)
            side_choices[side_name] = (
                float(selected["tp_bps"]), float(selected["sl_bps"])
            )
            print(
                f"{side_name} calibration: tp/sl={selected['tp_bps']:g}/"
                f"{selected['sl_bps']:g} Sortino={selected['cal_sortino']:+.3f}, "
                f"trades={int(selected['cal_trades'])}"
            )
        except ValueError:
            side_choices[side_name] = (symmetric_tp, symmetric_sl)
            print(f"{side_name}: no geometry met floor; using symmetric levels")

    long_tp, long_sl = side_choices["long"]
    short_tp, short_sl = side_choices["short"]
    eval_signals = scoped_signals(hold, LOCKBOX, start=SPLIT)
    policies = {
        "symmetric": (symmetric_tp, symmetric_sl, symmetric_tp, symmetric_sl),
        "asymmetric": (long_tp, long_sl, short_tp, short_sl),
    }
    policy_results = {}
    quarters = {
        "2025Q3": (pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC")),
        "2025Q4": (pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        "2026Q1": (pd.Timestamp("2026-01-01", tz="UTC"), LOCKBOX),
    }
    summaries = []
    for name, (tp_l, sl_l, tp_s, sl_s) in policies.items():
        ledger, per_bar = run_policy(
            bars_eval,
            eval_signals,
            tp_long=tp_l,
            sl_long=sl_l,
            hold=hold,
            tp_short=tp_s,
            sl_short=sl_s,
        )
        summary = economics_summary(per_bar)
        summary.update(trade_stats(ledger))
        summary.update(
            {
                "variant": name,
                "primary_tau": PRIMARY_TAU,
                "max_hold": hold,
                "hold_hours": hold / 4.0,
                "tp_long": tp_l,
                "sl_long": sl_l,
                "tp_short": tp_s,
                "sl_short": sl_s,
                "eval_gross": float(ledger["gross_return"].sum()),
                "eval_net": float(ledger["net_return"].sum()),
                "eval_sortino": summary["sortino"],
                "eval_events": len(ledger),
                "turnover_sides": 2 * len(ledger),
                "long_net": float(ledger.loc[ledger["side"] == 1, "net_return"].sum()),
                "short_net": float(ledger.loc[ledger["side"] == -1, "net_return"].sum()),
            }
        )
        positive = 0
        entry_time = pd.to_datetime(ledger["entry_time"], utc=True)
        for quarter, (start, end) in quarters.items():
            q = ledger[(entry_time >= start) & (entry_time < end)]
            gross = float(q["gross_return"].sum())
            summary[f"{quarter}_gross"] = gross
            summary[f"{quarter}_net"] = float(q["net_return"].sum())
            positive += int(gross > 0.0)
        summary["positive_gross_quarters"] = positive
        summary["screen_pass"] = bool(
            summary["eval_gross"] > 0.0
            and summary["eval_net"] > 0.0
            and positive >= 2
            and summary["eval_events"] >= FLOOR
        )
        summaries.append(summary)
        policy_results[name] = (ledger, per_bar)

    dm = diebold_mariano(
        policy_results["symmetric"][1],
        policy_results["asymmetric"][1],
        lag=hold,
    )
    summaries[0].update({"dm_vs_symmetric": np.nan, "dm_p": np.nan})
    summaries[1].update({"dm_vs_symmetric": dm["dm_stat"], "dm_p": dm["p_value"]})
    table = pd.DataFrame(summaries)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(SUMMARY_PATH, index=False)
    pd.DataFrame(grid_rows).to_parquet(GRID_PATH, index=False)
    pd.DataFrame(
        {name: result[1] for name, result in policy_results.items()}
    ).to_parquet(RETURN_PATH)

    columns = [
        "variant", "hold_hours", "tp_long", "sl_long", "tp_short", "sl_short",
        "eval_gross", "eval_net", "eval_sortino", "eval_events", "long_net",
        "short_net", "tp_rate", "sl_rate", "timeout_rate",
        "positive_gross_quarters", "dm_vs_symmetric", "dm_p", "screen_pass",
    ]
    pd.set_option("display.width", 240)
    print(table[columns].to_string(index=False))
    print(f"wrote -> {SUMMARY_PATH.name}, {GRID_PATH.name}, {RETURN_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
