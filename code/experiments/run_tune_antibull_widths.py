"""Selected-width anti-bull CatBoost study for notebook 02.

The width grid (dz55/dz65/dz75) comes from notebook 01's net-return ranking.
All model and policy decisions stay inside monthly chronological development
folds ending 2025-06-30. Expensive fitting and economic aggregation are added
below these small, testable selection contracts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from tqdm import tqdm

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.frozen_model_study import (
    StudyPaths,
    prediction_cache_path as frozen_prediction_cache_path,
    validate_prediction_frame,
)
from experiments.model_zoo_protocol import (
    BASE_MODELS,
    candidate_pool as frozen_candidate_pool,
    protocol_fingerprint,
    protocol_payload,
)
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

WIDTHS = (55, 65, 75)
LOOKBACK_DAYS = 90
DEVELOPMENT_END = pd.Timestamp("2025-07-01", tz="UTC")
FLOOR = 50
MIN_SIDE_TRADES = 15
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "tuning" / "antibull_widths"
PRED_DIR = OUT_DIR / "predictions"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2024_2026.parquet"
CANDIDATES_PATH = OUT_DIR / "candidates.json"
CLASSIFICATION_PATH = OUT_DIR / "classification_grid.parquet"
ECONOMIC_PATH = OUT_DIR / "economic_grid.parquet"
AUDIT_PATH = OUT_DIR / "outer_audit.parquet"
UNGATED_PATH = OUT_DIR / "ungated_summary.parquet"
OBJECTIVE_PATH = OUT_DIR / "objective_summary.parquet"
RESULT_PATH = OUT_DIR / "result.json"

MODEL = "catboost_balanced"
TP_BPS = 150.0
SL_BPS = 75.0
MAX_HOLD = 1


def _fingerprint(params: dict) -> str:
    raw = json.dumps(params, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:10]


def prediction_cache_path(
    width: int,
    candidate_id: int,
    params: dict,
    fold_id: int,
    month: str,
) -> Path:
    return PRED_DIR / (
        f"w{width}_candidate_{candidate_id:02d}_{_fingerprint(params)}_"
        f"fold_{fold_id:02d}_{month}.parquet"
    )


def outer_audit_schedule() -> list[tuple[tuple[int, ...], int]]:
    return [
        (tuple(range(12)), 12),
        (tuple(range(13)), 13),
        (tuple(range(14)), 14),
    ]


def monthly_development_folds() -> list[ReflectionWindow]:
    starts = pd.date_range("2024-04-01", "2025-06-01", freq="MS", tz="UTC")
    folds = []
    for i, start in enumerate(starts):
        end_exclusive = start + pd.offsets.MonthBegin(1)
        folds.append(
            ReflectionWindow(
                name=f"antibull_{i:02d}_{start:%Y-%m}",
                train_start=start - pd.Timedelta(days=LOOKBACK_DAYS),
                train_end=start,
                validation_start=start,
                validation_end=end_exclusive - pd.Timedelta(minutes=15),
            )
        )
    return folds


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


def candidate_pool(n_trials: int = 15) -> list[dict]:
    if n_trials < 2:
        raise ValueError("at least two candidates are required")
    candidates = [
        {"iterations": 300, "depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3.0}
    ]
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.RandomSampler(seed=42)
    )
    for _ in range(n_trials - 1):
        trial = study.ask()
        candidates.append(_sample_params(trial))
        study.tell(trial, 0.0)
    return candidates


@dataclass(frozen=True)
class ConfiguredStudy:
    model: str
    paths: StudyPaths
    candidates: list[dict]
    folds: list[ReflectionWindow]
    smoke: bool


def _legacy_paths() -> StudyPaths:
    return StudyPaths(
        root=OUT_DIR,
        predictions=PRED_DIR,
        candidates=CANDIDATES_PATH,
        classification=CLASSIFICATION_PATH,
        economics=ECONOMIC_PATH,
        audit=AUDIT_PATH,
        ungated=UNGATED_PATH,
        objective=OBJECTIVE_PATH,
        manifest=OUT_DIR / "manifest.json",
        result=RESULT_PATH,
    )


def configure_study(
    *,
    model: str,
    model_zoo: bool,
    n_trials: int,
    fold_limit: int | None = None,
    candidate_limit: int | None = None,
) -> ConfiguredStudy:
    """Resolve isolated paths and bounded candidates/folds before fitting."""
    if model not in BASE_MODELS:
        raise ValueError(f"unsupported base model: {model}")
    if not model_zoo and model != MODEL:
        raise ValueError("legacy output is reserved for catboost_balanced")
    if fold_limit is not None and fold_limit < 1:
        raise ValueError("fold_limit must be positive")
    if candidate_limit is not None and candidate_limit < 1:
        raise ValueError("candidate_limit must be positive")

    smoke = model_zoo and (fold_limit is not None or candidate_limit is not None)
    paths = StudyPaths.for_model(model, smoke=smoke) if model_zoo else _legacy_paths()
    candidates = (
        frozen_candidate_pool(model, n_trials)
        if model_zoo
        else candidate_pool(n_trials)
    )
    folds = monthly_development_folds()
    if candidate_limit is not None:
        candidates = candidates[:candidate_limit]
    if fold_limit is not None:
        folds = folds[:fold_limit]
    return ConfiguredStudy(model, paths, candidates, folds, smoke)

def select_f1_candidate(grid: pd.DataFrame) -> pd.Series:
    if grid.empty:
        raise ValueError("F1 grid is empty")
    return grid.sort_values(
        ["robust_f1", "overall_f1", "candidate"],
        ascending=[False, False, True],
    ).iloc[0]


def select_economic_candidate(
    grid: pd.DataFrame, *, n_folds: int
) -> pd.Series | None:
    required_positive = math.ceil(2 * n_folds / 3)
    eligible = grid[
        (grid["trades"] >= FLOOR)
        & (grid["pooled_net"] > 0.0)
        & (grid["pooled_sortino"] > 0.0)
        & (grid["pooled_sharpe"] > 0.0)
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




def _macro_f1(y_true: pd.Series, prediction: pd.Series) -> float:
    return float(
        f1_score(
            y_true.astype(int),
            prediction.astype(int),
            labels=[0, 1, 2],
            average="macro",
            zero_division=0,
        )
    )


def _json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=15)
    ap.add_argument("--model", choices=BASE_MODELS, default=MODEL)
    ap.add_argument("--model-zoo", action="store_true")
    ap.add_argument("--fold-limit", type=int)
    ap.add_argument("--candidate-limit", type=int)
    args = ap.parse_args(argv)

    started = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    study = configure_study(
        model=args.model,
        model_zoo=args.model_zoo,
        n_trials=args.trials,
        fold_limit=args.fold_limit,
        candidate_limit=args.candidate_limit,
    )
    candidates = study.candidates
    folds = study.folds

    study.paths.root.mkdir(parents=True, exist_ok=True)
    study.paths.predictions.mkdir(parents=True, exist_ok=True)
    study.paths.manifest.write_text(
        json.dumps(
            {
                **protocol_payload(),
                "protocol_fingerprint": protocol_fingerprint(),
                "model": study.model,
                "smoke": study.smoke,
                "actual_candidate_count": len(candidates),
                "actual_fold_count": len(folds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    study.paths.candidates.write_text(
        json.dumps(
            {
                "generated_before_economics": True,
                "source_widths": "notebook 01 net (+positioning) ranking",
                "widths": list(WIDTHS),
                "sampler": "seeded RandomSampler; outcome-independent candidate pool",
                "candidate_0": "current project factory defaults",
                "model": study.model,
                "protocol_fingerprint": protocol_fingerprint(),
                "candidates": candidates,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    bars = pd.read_parquet(
        CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"]
    )
    bars.index = pd.to_datetime(bars.index, utc=True)
    bars = bars.sort_index().loc[lambda frame: frame.index < DEVELOPMENT_END]
    minute = pd.read_parquet(MINUTE_PATH)
    minute.index = pd.to_datetime(minute.index, utc=True)
    minute = minute.sort_index().loc[lambda frame: frame.index < DEVELOPMENT_END]
    bar_regimes = past_regime_labels(bars["close"])

    features: dict[int, tuple[pd.DataFrame, pd.Series]] = {}
    for width in WIDTHS:
        X, y, _ = build_walkforward_xy(
            "btc",
            cfg,
            horizon=1,
            sentiment="both",
            label_fn=lambda feat, w=width: make_label(
                feat, threshold_bps=w, horizon=1
            ),
            orderflow=True,
            positioning=True,
        )
        X = X[X.index < DEVELOPMENT_END]
        y = y.reindex(X.index)
        known = bar_regimes.reindex(X.index).isin(REGIMES)
        features[width] = (X.loc[known], y.loc[known])

    predictions: dict[tuple[int, int, int], pd.DataFrame] = {}
    jobs = [
        (width, candidate_id, fold_id)
        for width in WIDTHS
        for candidate_id in range(len(candidates))
        for fold_id in range(len(folds))
    ]
    for width, candidate_id, fold_id in tqdm(
        jobs, desc="selected-width anti-bull", unit="fit"
    ):
        params = candidates[candidate_id]
        fold = folds[fold_id]
        cache = (
            frozen_prediction_cache_path(
                study.paths,
                study.model,
                width=width,
                candidate_id=candidate_id,
                params=params,
                fold_id=fold_id,
                month=f"{fold.validation_start:%Y-%m}",
            )
            if args.model_zoo
            else prediction_cache_path(
                width,
                candidate_id,
                params,
                fold_id,
                f"{fold.validation_start:%Y-%m}",
            )
        )
        if cache.exists():
            pred = pd.read_parquet(cache)
            pred.index = pd.to_datetime(pred.index, utc=True)
            pred = validate_prediction_frame(
                pred.sort_index(), model=study.model, development_end=DEVELOPMENT_END
            )
            predictions[(width, candidate_id, fold_id)] = pred
            continue

        X, y = features[width]

        def training_weights(
            X_train: pd.DataFrame, _y_train: pd.Series
        ) -> pd.Series:
            window_regimes = bar_regimes.reindex(X_train.index)
            if not window_regimes.isin(REGIMES).all():
                raise ValueError("unknown regime remained inside a training fold")
            return regime_balanced_weights(window_regimes)

        pred = run_walkforward_predictions(
            X,
            y,
            windows=[fold],
            model_factory=MODELS[study.model],
            model_name=study.model,
            params=params,
            cache_path=cache,
            min_train_rows=500,
            min_validation_rows=50,
            train_tail_trim=1,
            sample_weight_fn=training_weights,
        )
        if pred.empty:
            raise RuntimeError(
                f"empty predictions: width={width} candidate={candidate_id} "
                f"fold={fold.name}"
            )
        pred = validate_prediction_frame(
            pred.sort_index(), model=study.model, development_end=DEVELOPMENT_END
        )
        predictions[(width, candidate_id, fold_id)] = pred

    if study.smoke:
        result = {
            "status": "smoke_complete",
            "model": study.model,
            "protocol_fingerprint": protocol_fingerprint(),
            "candidate_count": len(candidates),
            "fold_count": len(folds),
            "prediction_cells": len(predictions),
        }
        study.paths.result.write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        print(f"wrote -> {study.paths.result}")
        return 0

    def classification_scope(
        scope: str, fold_ids: tuple[int, ...]
    ) -> pd.DataFrame:
        rows = []
        for width in WIDTHS:
            for candidate_id in range(len(candidates)):
                frames = [
                    predictions[(width, candidate_id, fold_id)]
                    for fold_id in fold_ids
                ]
                frame = pd.concat(frames).sort_index()
                actual = frame["y_true"].astype(int)
                predicted = frame[f"{study.model}_pred"].astype(int)
                regimes = bar_regimes.reindex(frame.index)
                regime_scores = {}
                for regime in REGIMES:
                    mask = regimes == regime
                    regime_scores[regime] = _macro_f1(
                        actual.loc[mask], predicted.loc[mask]
                    )
                rows.append(
                    {
                        "scope": scope,
                        "width": width,
                        "candidate": candidate_id,
                        "overall_f1": _macro_f1(actual, predicted),
                        "bull_f1": regime_scores["bull"],
                        "sideways_f1": regime_scores["sideways"],
                        "bear_f1": regime_scores["bear"],
                        "robust_f1": min(regime_scores.values()),
                    }
                )
        return pd.DataFrame(rows)

    simulations: dict[
        tuple[int, int, int, float], tuple[pd.DataFrame, pd.Series]
    ] = {}

    def simulate(width: int, candidate_id: int, fold_id: int, tau: float):
        key = (width, candidate_id, fold_id, tau)
        if key in simulations:
            return simulations[key]
        fold = folds[fold_id]
        frame = predictions[(width, candidate_id, fold_id)]
        pred = frame[f"{study.model}_pred"].astype(int)
        conf = frame[f"{study.model}_conf"].astype(float)
        boundary = fold.validation_end + pd.Timedelta(minutes=15)
        path_safe = (
            pred.index + pd.Timedelta(minutes=15 * (MAX_HOLD + 1))
            <= boundary
        )
        scope_bars = bars[
            (bars.index >= fold.validation_start) & (bars.index < boundary)
        ]
        ledger, per_bar = simulate_bracket_trades_intrabar(
            scope_bars,
            minute,
            pred.loc[path_safe],
            conf.loc[path_safe],
            tau=tau,
            tp_bps=TP_BPS,
            sl_bps=SL_BPS,
            max_hold=MAX_HOLD,
            fee_bps=fee,
        )
        simulations[key] = (ledger, per_bar)
        return ledger, per_bar

    def economic_scope(scope: str, fold_ids: tuple[int, ...]) -> pd.DataFrame:
        rows = []
        for width in WIDTHS:
            for candidate_id in range(len(candidates)):
                for tau in TAUS:
                    ledgers = []
                    returns = []
                    fold_nets = []
                    for fold_id in fold_ids:
                        ledger, per_bar = simulate(
                            width, candidate_id, fold_id, tau
                        )
                        tagged = ledger.copy()
                        tagged["fold_id"] = fold_id
                        ledgers.append(tagged)
                        returns.append(per_bar)
                        fold_nets.append(float(ledger["net_return"].sum()))
                    ledger = pd.concat(ledgers, ignore_index=True)
                    per_bar = pd.concat(returns).sort_index()
                    summary = economics_summary(per_bar)
                    regimes = bar_regimes.reindex(per_bar.index)
                    regime_sortino = {}
                    for regime in REGIMES:
                        regime_return = per_bar[regimes == regime]
                        regime_sortino[regime] = economics_summary(
                            regime_return
                        )["sortino"]
                    robust = min(
                        summary["sortino"],
                        summary["sharpe"],
                        *regime_sortino.values(),
                    )
                    rows.append(
                        {
                            "scope": scope,
                            "width": width,
                            "candidate": candidate_id,
                            "tau": tau,
                            "trades": len(ledger),
                            "pooled_gross": float(
                                ledger["gross_return"].sum()
                            ),
                            "pooled_net": float(ledger["net_return"].sum()),
                            "pooled_sortino": summary["sortino"],
                            "pooled_sharpe": summary["sharpe"],
                            "positive_folds": sum(
                                value > 0.0 for value in fold_nets
                            ),
                            "n_long": int((ledger["side"] == 1).sum()),
                            "n_short": int((ledger["side"] == -1).sum()),
                            "long_net": float(
                                ledger.loc[
                                    ledger["side"] == 1, "net_return"
                                ].sum()
                            ),
                            "short_net": float(
                                ledger.loc[
                                    ledger["side"] == -1, "net_return"
                                ].sum()
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
        inner_grid = economic_scope(f"outer_{month}_inner", inner_ids)
        grid_parts.append(inner_grid)
        selected = select_economic_candidate(
            inner_grid, n_folds=len(inner_ids)
        )
        if selected is None:
            audit_rows.append(
                {
                    "outer_month": month,
                    "decision": "no_trade",
                    "width": None,
                    "candidate": None,
                    "tau": None,
                    "outer_trades": 0,
                    "outer_gross": 0.0,
                    "outer_net": 0.0,
                    "outer_sortino": 0.0,
                    "outer_sharpe": 0.0,
                }
            )
            continue
        width = int(selected["width"])
        candidate_id = int(selected["candidate"])
        tau = float(selected["tau"])
        ledger, per_bar = simulate(width, candidate_id, outer_id, tau)
        summary = economics_summary(per_bar)
        audit_rows.append(
            {
                "outer_month": month,
                "decision": "trade",
                "width": width,
                "candidate": candidate_id,
                "tau": tau,
                "outer_trades": len(ledger),
                "outer_gross": float(ledger["gross_return"].sum()),
                "outer_net": float(ledger["net_return"].sum()),
                "outer_sortino": summary["sortino"],
                "outer_sharpe": summary["sharpe"],
            }
        )

    all_ids = tuple(range(len(folds)))
    classification = classification_scope("final_development", all_ids)
    final_economic = economic_scope("final_development", all_ids)
    grid_parts.append(final_economic)
    economic_grid = pd.concat(grid_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)

    f1_choices = {}
    ungated_rows = []
    objective_rows = []
    for width in WIDTHS:
        class_width = classification[classification["width"] == width]
        f1_choice = select_f1_candidate(class_width)
        f1_candidate = int(f1_choice["candidate"])
        f1_choices[width] = f1_candidate

        for arm, candidate_id in (
            ("project baseline", 0),
            ("F1-selected", f1_candidate),
        ):
            row = final_economic[
                (final_economic["width"] == width)
                & (final_economic["candidate"] == candidate_id)
                & (final_economic["tau"] == 0.0)
            ].iloc[0].to_dict()
            row.update({"arm": arm, "decision": "diagnostic"})
            ungated_rows.append(row)

        f1_econ_grid = final_economic[
            (final_economic["width"] == width)
            & (final_economic["candidate"] == f1_candidate)
        ]
        f1_policy = select_economic_candidate(
            f1_econ_grid, n_folds=len(all_ids)
        )
        if f1_policy is None:
            objective_rows.append(
                {
                    "width": width,
                    "objective": "F1",
                    "decision": "no_trade",
                    "candidate": f1_candidate,
                    "tau": None,
                }
            )
        else:
            row = f1_policy.to_dict()
            row.update({"objective": "F1", "decision": "trade"})
            objective_rows.append(row)

        width_grid = final_economic[final_economic["width"] == width]
        econ_policy = select_economic_candidate(
            width_grid, n_folds=len(all_ids)
        )
        if econ_policy is None:
            objective_rows.append(
                {
                    "width": width,
                    "objective": "economic",
                    "decision": "no_trade",
                    "candidate": None,
                    "tau": None,
                }
            )
        else:
            row = econ_policy.to_dict()
            row.update({"objective": "economic", "decision": "trade"})
            objective_rows.append(row)

    selected = select_economic_candidate(
        final_economic, n_folds=len(all_ids)
    )
    classification.to_parquet(study.paths.classification, index=False)
    economic_grid.to_parquet(study.paths.economics, index=False)
    audit.to_parquet(study.paths.audit, index=False)
    pd.DataFrame(ungated_rows).to_parquet(study.paths.ungated, index=False)
    pd.DataFrame(objective_rows).to_parquet(study.paths.objective, index=False)

    result = {
        "instrument": "btc",
        "model": study.model,
        "protocol_fingerprint": protocol_fingerprint(),
        "development_end_exclusive": str(DEVELOPMENT_END),
        "width_source": "notebook 01 net (+positioning) ranking",
        "widths": list(WIDTHS),
        "candidate_count": len(candidates),
        "fold_count": len(folds),
        "training_history": "90D rolling",
        "retrain_cadence_for_tuning": "monthly",
        "anti_bull_training_weights": True,
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
            "positive_total_long_short_net": True,
            "positive_pooled_sortino_and_sharpe": True,
            "positive_bull_sideways_bear_sortino": True,
            "no_trade_allowed": True,
        },
        "f1_selected_candidates": {
            str(width): candidate_id
            for width, candidate_id in f1_choices.items()
        },
        "outer_audit": audit.to_dict(orient="records"),
        "decision": "no_trade" if selected is None else "trade",
        "selected_width": None if selected is None else int(selected["width"]),
        "selected_candidate": (
            None if selected is None else int(selected["candidate"])
        ),
        "selected_tau": None if selected is None else float(selected["tau"]),
        "selected_params": (
            None
            if selected is None
            else candidates[int(selected["candidate"])]
        ),
        "selected_development_metrics": (
            None
            if selected is None
            else {
                key: _json_value(value)
                for key, value in selected.items()
                if key != "scope"
            }
        ),
        "elapsed_s": round(time.time() - started, 1),
    }
    study.paths.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(audit.to_string(index=False))
    if selected is None:
        print("FINAL DEVELOPMENT DECISION: NO TRADE")
    else:
        print(
            "FINAL DEVELOPMENT DECISION: "
            f"dz{int(selected['width'])} candidate="
            f"{int(selected['candidate'])} tau={float(selected['tau']):.2f} "
            f"net={float(selected['pooled_net']):+.2%} "
            f"robust={float(selected['robust_score']):+.3f}"
        )
    print(f"wrote -> {study.paths.result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
