"""Cooldown-only economic test on frozen dz75 candidate-6 predictions."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.run_tune_antibull_widths import (
    CONFIG,
    DEVELOPMENT_END,
    MAX_HOLD,
    MINUTE_PATH,
    MODEL,
    OUT_DIR,
    SL_BPS,
    TP_BPS,
    monthly_development_folds,
    prediction_cache_path,
    select_economic_candidate,
)
from experiments.run_tune_dz75_regime import REGIMES, past_regime_labels

WIDTH = 75
CANDIDATE = 6
TAU = 0.75
CONTROL_COOLDOWN = 0
COOLDOWNS = (4, 8, 16)

OUTPUT_DIR = OUT_DIR / "turnover_control"
GRID_PATH = OUTPUT_DIR / "economic_grid.parquet"
AUDIT_PATH = OUTPUT_DIR / "outer_audit.parquet"
RESULT_PATH = OUTPUT_DIR / "result.json"


def outer_audit_schedule() -> list[tuple[tuple[int, ...], int]]:
    return [
        (tuple(range(12)), 12),
        (tuple(range(13)), 13),
        (tuple(range(14)), 14),
    ]


def select_cooldown(grid: pd.DataFrame, *, n_folds: int) -> pd.Series | None:
    experimental = grid[grid["cooldown_bars"].isin(COOLDOWNS)]
    return select_economic_candidate(experimental, n_folds=n_folds)


def _json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main() -> int:
    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    folds = monthly_development_folds()

    candidate_file = OUT_DIR / "candidates.json"
    candidates = json.loads(candidate_file.read_text(encoding="utf-8"))[
        "candidates"
    ]
    params = candidates[CANDIDATE]

    predictions = {}
    for fold_id, fold in enumerate(folds):
        path = prediction_cache_path(
            WIDTH,
            CANDIDATE,
            params,
            fold_id,
            f"{fold.validation_start:%Y-%m}",
        )
        if not path.exists():
            raise FileNotFoundError(f"missing frozen prediction cache: {path}")
        frame = pd.read_parquet(path)
        frame.index = pd.to_datetime(frame.index, utc=True)
        predictions[fold_id] = frame.sort_index()

    code_root = Path(__file__).resolve().parents[1]
    bars = pd.read_parquet(
        code_root / cfg["instruments"]["btc"]["working_parquet"]
    )
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index().loc[lambda frame: frame.index < DEVELOPMENT_END]
    minute = pd.read_parquet(MINUTE_PATH)
    minute.index = pd.to_datetime(minute.index, utc=True)
    minute = minute.sort_index().loc[lambda frame: frame.index < DEVELOPMENT_END]
    bar_regimes = past_regime_labels(bars["close"])

    simulations: dict[tuple[int, int], tuple[pd.DataFrame, pd.Series]] = {}

    def simulate(fold_id: int, cooldown: int):
        key = (fold_id, cooldown)
        if key in simulations:
            return simulations[key]
        fold = folds[fold_id]
        frame = predictions[fold_id]
        prediction = frame[f"{MODEL}_pred"].astype(int)
        confidence = frame[f"{MODEL}_conf"].astype(float)
        boundary = fold.validation_end + pd.Timedelta(minutes=15)
        path_safe = (
            prediction.index + pd.Timedelta(minutes=15 * (MAX_HOLD + 1))
            <= boundary
        )
        scope_bars = bars[
            (bars.index >= fold.validation_start) & (bars.index < boundary)
        ]
        result = simulate_bracket_trades_intrabar(
            scope_bars,
            minute,
            prediction.loc[path_safe],
            confidence.loc[path_safe],
            tau=TAU,
            tp_bps=TP_BPS,
            sl_bps=SL_BPS,
            max_hold=MAX_HOLD,
            fee_bps=fee,
            cooldown_bars=cooldown,
        )
        simulations[key] = result
        return result

    def economic_scope(scope: str, fold_ids: tuple[int, ...]) -> pd.DataFrame:
        rows = []
        for cooldown in (CONTROL_COOLDOWN, *COOLDOWNS):
            ledgers = []
            returns = []
            fold_nets = []
            for fold_id in fold_ids:
                ledger, per_bar = simulate(fold_id, cooldown)
                ledgers.append(ledger)
                returns.append(per_bar)
                fold_nets.append(float(ledger["net_return"].sum()))
            ledger = pd.concat(ledgers, ignore_index=True)
            per_bar = pd.concat(returns).sort_index()
            summary = economics_summary(per_bar)
            regimes = bar_regimes.reindex(per_bar.index)
            regime_sortino = {
                regime: economics_summary(per_bar[regimes == regime])["sortino"]
                for regime in REGIMES
            }
            robust = min(
                summary["sortino"],
                summary["sharpe"],
                *regime_sortino.values(),
            )
            rows.append(
                {
                    "scope": scope,
                    "width": WIDTH,
                    "candidate": CANDIDATE,
                    "tau": TAU,
                    "cooldown_bars": cooldown,
                    "trades": len(ledger),
                    "pooled_gross": float(ledger["gross_return"].sum()),
                    "pooled_net": float(ledger["net_return"].sum()),
                    "pooled_sortino": summary["sortino"],
                    "pooled_sharpe": summary["sharpe"],
                    "positive_folds": sum(value > 0.0 for value in fold_nets),
                    "n_long": int((ledger["side"] == 1).sum()),
                    "n_short": int((ledger["side"] == -1).sum()),
                    "long_net": float(
                        ledger.loc[ledger["side"] == 1, "net_return"].sum()
                    ),
                    "short_net": float(
                        ledger.loc[ledger["side"] == -1, "net_return"].sum()
                    ),
                    "bull_sortino": regime_sortino["bull"],
                    "sideways_sortino": regime_sortino["sideways"],
                    "bear_sortino": regime_sortino["bear"],
                    "robust_score": robust,
                }
            )
        return pd.DataFrame(rows)

    grid_parts = []
    audit_rows = []
    for inner_ids, outer_id in outer_audit_schedule():
        month = f"{folds[outer_id].validation_start:%Y-%m}"
        inner = economic_scope(f"outer_{month}_inner", inner_ids)
        grid_parts.append(inner)
        selected = select_cooldown(inner, n_folds=len(inner_ids))
        if selected is None:
            audit_rows.append(
                {
                    "outer_month": month,
                    "decision": "no_trade",
                    "cooldown_bars": None,
                    "outer_trades": 0,
                    "outer_net": 0.0,
                    "outer_sortino": 0.0,
                    "outer_sharpe": 0.0,
                }
            )
            continue
        cooldown = int(selected["cooldown_bars"])
        ledger, per_bar = simulate(outer_id, cooldown)
        summary = economics_summary(per_bar)
        audit_rows.append(
            {
                "outer_month": month,
                "decision": "trade",
                "cooldown_bars": cooldown,
                "outer_trades": len(ledger),
                "outer_net": float(ledger["net_return"].sum()),
                "outer_sortino": summary["sortino"],
                "outer_sharpe": summary["sharpe"],
            }
        )

    final = economic_scope("final_development", tuple(range(len(folds))))
    grid_parts.append(final)
    grid = pd.concat(grid_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    selected = select_cooldown(final, n_folds=len(folds))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(GRID_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)

    payload = {
        "model_source": "frozen anti-bull prediction cache; no retraining",
        "width": WIDTH,
        "candidate": CANDIDATE,
        "tau": TAU,
        "candidate_params": params,
        "control_cooldown_bars": CONTROL_COOLDOWN,
        "tested_cooldown_bars": list(COOLDOWNS),
        "execution": {
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "max_hold_m15_bars": MAX_HOLD,
            "fee_bps_per_side": fee,
        },
        "decision": "no_trade" if selected is None else "trade",
        "selected_cooldown_bars": (
            None if selected is None else int(selected["cooldown_bars"])
        ),
        "selected_metrics": (
            None
            if selected is None
            else {key: _json_value(value) for key, value in selected.items()}
        ),
        "outer_audit": audit.to_dict(orient="records"),
        "elapsed_s": round(time.time() - started, 1),
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(final.to_string(index=False))
    print(audit.to_string(index=False))
    if selected is None:
        print("FINAL TURNOVER DECISION: NO TRADE")
    else:
        print(
            "FINAL TURNOVER DECISION: "
            f"{int(selected['cooldown_bars'])} BARS"
        )
    print(f"wrote -> {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
