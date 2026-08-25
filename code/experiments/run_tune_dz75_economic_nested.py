"""Nested, regime-balanced dz75 economic study using 2024 only.

Candidate parameters are fixed before any economics are observed. Nine monthly
out-of-sample folds use the preceding 90 days for training. October, November,
and December act as outer audits: each may select a candidate/tau only from
earlier folds. The final 2024 decision permits an explicit no-trade result.

Run:  python -m experiments.run_tune_dz75_economic_nested --trials 15
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import optuna
import pandas as pd
import yaml
from tqdm import tqdm

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.run_tune_dz75_regime import (
    REGIMES,
    past_regime_labels,
    regime_balanced_weights,
)
from experiments.run_walkforward import build_walkforward_xy
from experiments.walkforward import run_walkforward_predictions
from features.build import make_label
from memory.loop import ReflectionWindow
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
F1_TUNING_PATH = (
    CODE_ROOT / "experiments" / "cache" / "tuning"
    / "btc_bothofpos_dz75_regime_catboost_balanced.json"
)
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "tuning" / "dz75_economic_nested"
PRED_DIR = OUT_DIR / "predictions"
CANDIDATES_PATH = OUT_DIR / "candidates.json"
GRID_PATH = OUT_DIR / "economic_grid.parquet"
AUDIT_PATH = OUT_DIR / "outer_audit.parquet"
RESULT_PATH = OUT_DIR / "result.json"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2024_2026.parquet"

MODEL = "catboost_balanced"
WIDTH = 75
LOOKBACK_DAYS = 90
TP_BPS = 150.0
SL_BPS = 75.0
MAX_HOLD = 1
FLOOR = 50
MIN_SIDE_TRADES = 15
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")


def monthly_study_folds() -> list[ReflectionWindow]:
    starts = pd.date_range("2024-04-01", "2024-12-01", freq="MS", tz="UTC")
    folds = []
    for i, start in enumerate(starts):
        end_exclusive = start + pd.offsets.MonthBegin(1)
        folds.append(
            ReflectionWindow(
                name=f"econ_{i:02d}_{start:%Y-%m}",
                train_start=start - pd.Timedelta(days=LOOKBACK_DAYS),
                train_end=start,
                validation_start=start,
                validation_end=end_exclusive - pd.Timedelta(minutes=15),
            )
        )
    return folds


def outer_audit_schedule() -> list[tuple[tuple[int, ...], int]]:
    return [
        (tuple(range(6)), 6),
        (tuple(range(7)), 7),
        (tuple(range(8)), 8),
    ]


def select_candidate(grid: pd.DataFrame, *, n_folds: int) -> pd.Series | None:
    required_positive = math.ceil(2 * n_folds / 3)
    eligible = grid[
        (grid["trades"] >= FLOOR)
        & (grid["pooled_net"] > 0.0)
        & (grid["pooled_sortino"] > 0.0)
        & (grid["positive_folds"] >= required_positive)
        & (grid["n_long"] >= MIN_SIDE_TRADES)
        & (grid["n_short"] >= MIN_SIDE_TRADES)
        & (grid["long_net"] > 0.0)
        & (grid["short_net"] > 0.0)
        & (grid["bull_sortino"] > 0.0)
        & (grid["sideways_sortino"] > 0.0)
        & (grid["bear_sortino"] > 0.0)
        & (grid["robust_score"] > 0.0)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values(
        ["robust_score", "pooled_sortino", "pooled_net", "trades"],
        ascending=[False, False, False, False],
    ).iloc[0]


def _sample_params(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 400, 800, step=100),
        "depth": trial.suggest_int("depth", 5, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.10, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 10.0, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.5),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.5),
        "rsm": trial.suggest_float("rsm", 0.65, 1.0),
    }


def _candidate_params(n_trials: int) -> list[dict]:
    if n_trials < 2:
        raise ValueError("trials must be at least 2 so the F1 control has a comparator")
    f1 = json.loads(F1_TUNING_PATH.read_text(encoding="utf-8"))["best_params"]
    candidates = [dict(f1)]
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
    )
    for _ in range(n_trials - 1):
        trial = study.ask()
        candidates.append(_sample_params(trial))
        study.tell(trial, 0.0)
    return candidates


def _fingerprint(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=15)
    args = ap.parse_args()

    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    candidates = _candidate_params(args.trials)
    folds = monthly_study_folds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATES_PATH.write_text(
        json.dumps(
            {
                "generated_before_economics": True,
                "sampler": "seeded RandomSampler; outcome-independent candidate pool",
                "candidate_0": "frozen regime-robust F1 parameters",
                "candidates": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    X, y, _ = build_walkforward_xy(
        "btc",
        cfg,
        horizon=1,
        sentiment="both",
        label_fn=lambda feat: make_label(feat, threshold_bps=WIDTH, horizon=1),
        orderflow=True,
        positioning=True,
    )
    X = X[X.index < DEVELOPMENT_END]
    y = y.reindex(X.index)
    bars = pd.read_parquet(CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"])
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index().loc[lambda d: d.index < DEVELOPMENT_END]
    minute = pd.read_parquet(MINUTE_PATH)
    minute.index = pd.to_datetime(minute.index, utc=True)
    minute = minute.sort_index().loc[lambda d: d.index < DEVELOPMENT_END]
    regimes = past_regime_labels(bars["close"]).reindex(X.index)

    predictions: dict[tuple[int, int], pd.DataFrame] = {}
    jobs = [(ci, fi) for ci in range(len(candidates)) for fi in range(len(folds))]
    for candidate_id, fold_id in tqdm(jobs, desc="dz75 economic 2024", unit="fit"):
        params = candidates[candidate_id]
        fold = folds[fold_id]
        cache = PRED_DIR / (
            f"candidate_{candidate_id:02d}_{_fingerprint(params)}_"
            f"fold_{fold_id:02d}_{fold.validation_start:%Y-%m}.parquet"
        )
        if cache.exists():
            pred = pd.read_parquet(cache)
            pred.index = pd.to_datetime(pred.index, utc=True)
            predictions[(candidate_id, fold_id)] = pred.sort_index()
            continue

        def training_weights(X_train: pd.DataFrame, _y_train: pd.Series):
            window_regimes = regimes.reindex(X_train.index)
            known = window_regimes.isin(REGIMES)
            if not known.all():
                raise ValueError("unknown regime remained inside a model training fold")
            return regime_balanced_weights(window_regimes)

        known_X = X.loc[regimes.isin(REGIMES)]
        known_y = y.reindex(known_X.index)
        pred = run_walkforward_predictions(
            known_X,
            known_y,
            windows=[fold],
            model_factory=MODELS[MODEL],
            model_name=MODEL,
            params=params,
            cache_path=cache,
            min_train_rows=500,
            min_validation_rows=50,
            train_tail_trim=1,
            sample_weight_fn=training_weights,
        )
        if pred.empty:
            raise RuntimeError(f"empty predictions: candidate={candidate_id} {fold.name}")
        predictions[(candidate_id, fold_id)] = pred

    simulations: dict[tuple[int, int, float], tuple[pd.DataFrame, pd.Series]] = {}

    def simulate(candidate_id: int, fold_id: int, tau: float):
        key = (candidate_id, fold_id, tau)
        if key in simulations:
            return simulations[key]
        fold = folds[fold_id]
        pred_frame = predictions[(candidate_id, fold_id)]
        pred = pred_frame[f"{MODEL}_pred"].astype(int)
        conf = pred_frame[f"{MODEL}_conf"].astype(float)
        boundary = fold.validation_end + pd.Timedelta(minutes=15)
        safe = pred.index + pd.Timedelta(minutes=15 * (MAX_HOLD + 1)) <= boundary
        scope_bars = bars[
            (bars.index >= fold.validation_start) & (bars.index < boundary)
        ]
        ledger, per_bar = simulate_bracket_trades_intrabar(
            scope_bars,
            minute,
            pred.loc[safe],
            conf.loc[safe],
            tau=tau,
            tp_bps=TP_BPS,
            sl_bps=SL_BPS,
            max_hold=MAX_HOLD,
            fee_bps=fee,
        )
        simulations[key] = (ledger, per_bar)
        return ledger, per_bar

    def scope_grid(scope: str, fold_ids: tuple[int, ...]) -> pd.DataFrame:
        rows = []
        for candidate_id in range(len(candidates)):
            for tau in TAUS:
                ledgers, returns, fold_nets = [], [], []
                for fold_id in fold_ids:
                    ledger, per_bar = simulate(candidate_id, fold_id, tau)
                    tagged = ledger.copy()
                    tagged["fold_id"] = fold_id
                    ledgers.append(tagged)
                    returns.append(per_bar)
                    fold_nets.append(float(ledger["net_return"].sum()))
                ledger = pd.concat(ledgers, ignore_index=True)
                per_bar = pd.concat(returns).sort_index()
                summary = economics_summary(per_bar)
                regime_sortino = {}
                bar_regimes = past_regime_labels(bars["close"]).reindex(per_bar.index)
                for regime in REGIMES:
                    regime_returns = per_bar[bar_regimes == regime]
                    regime_sortino[regime] = economics_summary(regime_returns)["sortino"]
                long_net = float(ledger.loc[ledger["side"] == 1, "net_return"].sum())
                short_net = float(ledger.loc[ledger["side"] == -1, "net_return"].sum())
                robust = min(summary["sortino"], *regime_sortino.values())
                rows.append(
                    {
                        "scope": scope,
                        "candidate": candidate_id,
                        "tau": tau,
                        "trades": len(ledger),
                        "pooled_gross": float(ledger["gross_return"].sum()),
                        "pooled_net": float(ledger["net_return"].sum()),
                        "pooled_sortino": summary["sortino"],
                        "pooled_sharpe": summary["sharpe"],
                        "positive_folds": sum(value > 0.0 for value in fold_nets),
                        "n_long": int((ledger["side"] == 1).sum()),
                        "n_short": int((ledger["side"] == -1).sum()),
                        "long_net": long_net,
                        "short_net": short_net,
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
        scope = f"outer_{folds[outer_id].validation_start:%Y-%m}_inner"
        inner_grid = scope_grid(scope, inner_ids)
        grid_parts.append(inner_grid)
        selected = select_candidate(inner_grid, n_folds=len(inner_ids))
        if selected is None:
            audit_rows.append(
                {
                    "outer_month": f"{folds[outer_id].validation_start:%Y-%m}",
                    "decision": "no_trade",
                    "candidate": -1,
                    "tau": None,
                    "outer_trades": 0,
                    "outer_gross": 0.0,
                    "outer_net": 0.0,
                    "outer_sortino": 0.0,
                }
            )
            continue
        candidate_id = int(selected["candidate"])
        tau = float(selected["tau"])
        ledger, per_bar = simulate(candidate_id, outer_id, tau)
        audit_rows.append(
            {
                "outer_month": f"{folds[outer_id].validation_start:%Y-%m}",
                "decision": "trade",
                "candidate": candidate_id,
                "tau": tau,
                "outer_trades": len(ledger),
                "outer_gross": float(ledger["gross_return"].sum()),
                "outer_net": float(ledger["net_return"].sum()),
                "outer_sortino": economics_summary(per_bar)["sortino"],
            }
        )

    all_ids = tuple(range(len(folds)))
    final_grid = scope_grid("final_2024", all_ids)
    grid_parts.append(final_grid)
    selected = select_candidate(final_grid, n_folds=len(all_ids))
    grid = pd.concat(grid_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    grid.to_parquet(GRID_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)

    result = {
        "instrument": "btc",
        "label": "dz75",
        "development_window": "2024 only",
        "control": "candidate 0 is frozen regime-robust F1 parameters",
        "candidate_count": len(candidates),
        "fold_count": len(folds),
        "retrain_cadence_for_tuning": "monthly",
        "training_history": "90D rolling",
        "execution": {
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "max_hold_m15_bars": MAX_HOLD,
            "fee_bps_per_side": fee,
        },
        "guardrails": {
            "minimum_trades": FLOOR,
            "minimum_trades_per_side": MIN_SIDE_TRADES,
            "positive_fold_fraction": "at least 2/3",
            "positive_total_net": True,
            "positive_long_and_short_net": True,
            "positive_bull_sideways_bear_sortino": True,
            "no_trade_allowed": True,
        },
        "outer_audit": audit.to_dict(orient="records"),
        "decision": "no_trade" if selected is None else "trade",
        "selected_candidate": None if selected is None else int(selected["candidate"]),
        "selected_tau": None if selected is None else float(selected["tau"]),
        "selected_params": None
        if selected is None
        else candidates[int(selected["candidate"])],
        "selected_2024_metrics": None
        if selected is None
        else {
            key: (int(value) if key in {"candidate", "trades", "positive_folds", "n_long", "n_short"} else float(value))
            for key, value in selected.items()
            if key != "scope"
        },
        "elapsed_s": round(time.time() - started, 1),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    if selected is None:
        print("FINAL 2024 DECISION: NO TRADE (no candidate passed all guardrails)")
    else:
        print(
            "FINAL 2024 DECISION: "
            f"candidate={int(selected['candidate'])} tau={float(selected['tau']):.2f} "
            f"net={float(selected['pooled_net']):+.2%} "
            f"robust Sortino={float(selected['robust_score']):+.3f}"
        )
    print(f"wrote -> {RESULT_PATH}, {GRID_PATH}, {AUDIT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
