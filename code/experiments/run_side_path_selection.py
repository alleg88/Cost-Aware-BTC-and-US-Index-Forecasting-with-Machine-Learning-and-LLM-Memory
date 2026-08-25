"""Nested long/short TP-first models for the frozen 66-trade geometry."""
from __future__ import annotations

import argparse
import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.intrabar_candidates import TP_FIRST, build_first_touch_candidates
from experiments.run_joint_path_selection import (
    CANDIDATE,
    PATH_PARAMS,
    PRIMARY_FEATURES,
    PRIMARY_TAU,
    WIDTH,
)
from experiments.run_tune_antibull_widths import (
    CONFIG,
    DEVELOPMENT_END,
    MINUTE_PATH,
    MODEL,
    OUT_DIR,
    monthly_development_folds,
    prediction_cache_path,
    select_economic_candidate,
)
from experiments.run_tune_dz75_regime import (
    REGIMES,
    past_regime_labels,
    regime_balanced_weights,
)
from features.intrabar import (
    INTRABAR_FEATURES,
    REGIME_STAGE_FEATURES,
    STAGE_FEATURES,
    build_intrabar_features,
    build_market_stage_features,
)
from models.zoo import make_catboost

GEOMETRY = (200, 100, 1)
MIN_PATH_TRAIN = 50
THRESHOLD_QUANTILES = (0.0, 0.25, 0.5, 0.75, 0.9)
PATH_FEATURES = (*PRIMARY_FEATURES, *INTRABAR_FEATURES, *STAGE_FEATURES)

OUTPUT_DIR = OUT_DIR / "side_path_selection"
GRID_PATH = OUTPUT_DIR / "economic_grid.parquet"
AUDIT_PATH = OUTPUT_DIR / "outer_audit.parquet"
DIAGNOSTIC_PATH = OUTPUT_DIR / "path_model_diagnostics.parquet"
PROBABILITY_PATH = OUTPUT_DIR / "side_probabilities.parquet"
RESULT_PATH = OUTPUT_DIR / "result.json"


def variant_settings(variant: str) -> tuple[Path, tuple[str, ...]]:
    if variant == "baseline":
        return OUTPUT_DIR, PATH_FEATURES
    if variant == "stage7d":
        return (
            OUT_DIR / "side_path_stage7d_selection",
            (*PATH_FEATURES, *REGIME_STAGE_FEATURES),
        )
    raise ValueError(f"unknown side-path variant: {variant}")


def side_training_mask(
    candidates: pd.DataFrame, cutoff: pd.Timestamp, *, side: int
) -> pd.Series:
    close_time = pd.to_datetime(candidates["outcome_close_time"], utc=True)
    return (close_time < pd.Timestamp(cutoff)) & candidates["side"].eq(side)


def apply_side_filter(
    prediction: pd.Series,
    confidence: pd.Series,
    p_tp: pd.Series,
    *,
    primary_tau: float,
    long_tau: float,
    short_tau: float,
) -> pd.Series:
    filtered = prediction.astype(int).copy()
    score = p_tp.reindex(filtered.index)
    threshold = pd.Series(np.nan, index=filtered.index, dtype=float)
    threshold.loc[filtered.eq(2)] = float(long_tau)
    threshold.loc[filtered.eq(0)] = float(short_tau)
    keep = (
        filtered.isin((0, 2))
        & confidence.reindex(filtered.index).ge(float(primary_tau))
        & score.notna()
        & score.ge(threshold)
    )
    filtered.loc[~keep] = 1
    return filtered


def select_side_policy(
    grid: pd.DataFrame, *, n_folds: int
) -> pd.Series | None:
    return select_economic_candidate(grid, n_folds=n_folds)


def outer_audit_schedule() -> list[tuple[tuple[int, ...], int]]:
    return [
        (tuple(range(12)), 12),
        (tuple(range(13)), 13),
        (tuple(range(14)), 14),
    ]


def _thresholds(
    probabilities: pd.Series,
    candidates: pd.DataFrame,
    fold_ids: tuple[int, ...],
    *,
    side: int,
) -> tuple[float, ...]:
    mask = candidates["fold_id"].isin(fold_ids) & candidates["side"].eq(side)
    values = probabilities.loc[mask].dropna().astype(float)
    if values.empty:
        return (0.0,)
    cutoffs = np.quantile(values, np.asarray(THRESHOLD_QUANTILES))
    return tuple(sorted({float(value) for value in cutoffs}))


def _json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def main(*, variant: str = "baseline") -> int:
    started = time.time()
    output_dir, path_features = variant_settings(variant)
    grid_path = output_dir / "economic_grid.parquet"
    audit_path = output_dir / "outer_audit.parquet"
    diagnostic_path = output_dir / "path_model_diagnostics.parquet"
    probability_path = output_dir / "side_probabilities.parquet"
    result_path = output_dir / "result.json"
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    folds = monthly_development_folds()
    params = json.loads(
        (OUT_DIR / "candidates.json").read_text(encoding="utf-8")
    )["candidates"][CANDIDATE]

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
        frame = frame.sort_index()
        frame["fold_id"] = fold_id
        predictions[fold_id] = frame
    prediction_frame = pd.concat(predictions.values()).sort_index()

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

    intrabar = build_intrabar_features(minute)
    stage = build_market_stage_features(bars, include_regime_stage=variant == "stage7d")
    features = pd.DataFrame(
        {
            "primary_conf": prediction_frame[f"{MODEL}_conf"].astype(float),
            "primary_p0": prediction_frame[f"{MODEL}_p0"].astype(float),
            "primary_p1": prediction_frame[f"{MODEL}_p1"].astype(float),
            "primary_p2": prediction_frame[f"{MODEL}_p2"].astype(float),
        },
        index=prediction_frame.index,
    ).join(intrabar).join(stage)
    features = features.dropna(subset=list(path_features))
    all_signals = prediction_frame[f"{MODEL}_pred"].astype(int).reindex(
        features.index
    )
    tp_bps, sl_bps, max_hold = GEOMETRY
    candidates = build_first_touch_candidates(
        minute,
        all_signals,
        features,
        tp_bps=tp_bps,
        sl_bps=sl_bps,
        max_hold=max_hold,
    )
    candidates["fold_id"] = prediction_frame["fold_id"].reindex(
        candidates.index
    ).astype(int)
    candidates["regime"] = bar_regimes.reindex(candidates.index)
    candidates = candidates[candidates["regime"].isin(REGIMES)].copy()

    output_dir.mkdir(parents=True, exist_ok=True)
    if probability_path.exists():
        probabilities = pd.read_parquet(probability_path)
        probabilities.index = pd.to_datetime(probabilities.index, utc=True)
        probabilities = probabilities.sort_index()
    else:
        probabilities = pd.DataFrame(
            np.nan,
            index=candidates.index,
            columns=["path_p_sl", "path_p_timeout", "path_p_tp"],
        )
        probabilities["path_trained"] = False
        probabilities["model_side"] = 0
        for fold_id, fold in enumerate(folds):
            for side in (1, -1):
                predict_mask = candidates["fold_id"].eq(fold_id) & candidates[
                    "side"
                ].eq(side)
                train_mask = side_training_mask(
                    candidates, fold.validation_start, side=side
                )
                y_train = candidates.loc[train_mask, "outcome"].astype(int)
                if (
                    int(train_mask.sum()) < MIN_PATH_TRAIN
                    or int(predict_mask.sum()) == 0
                    or y_train.nunique() < 2
                ):
                    continue
                model = make_catboost(PATH_PARAMS)
                weights = regime_balanced_weights(
                    candidates.loc[train_mask, "regime"]
                )
                model.fit(
                    candidates.loc[train_mask, list(path_features)].astype(float),
                    y_train,
                    sample_weight=weights,
                )
                raw = np.asarray(
                    model.predict_proba(
                        candidates.loc[predict_mask, list(path_features)].astype(float)
                    ),
                    dtype=float,
                )
                mapped = np.zeros((int(predict_mask.sum()), 3), dtype=float)
                for source, label in enumerate(model.classes_):
                    mapped[:, int(label)] = raw[:, source]
                probabilities.loc[
                    predict_mask, ["path_p_sl", "path_p_timeout", "path_p_tp"]
                ] = mapped
                probabilities.loc[predict_mask, "path_trained"] = True
                probabilities.loc[predict_mask, "model_side"] = side
        probabilities.to_parquet(probability_path)

    diagnostics = []
    for side, label in ((1, "long"), (-1, "short")):
        trained = (
            probabilities["path_trained"].fillna(False).astype(bool)
            & candidates["side"].eq(side)
        )
        actual_tp = candidates.loc[trained, "outcome"].eq(TP_FIRST).astype(int)
        p_tp = probabilities.loc[trained, "path_p_tp"].astype(float)
        diagnostics.append(
            {
                "side": label,
                "candidates": int(candidates["side"].eq(side).sum()),
                "trained_predictions": int(trained.sum()),
                "tp_rate": float(
                    candidates.loc[candidates["side"].eq(side), "outcome"]
                    .eq(TP_FIRST)
                    .mean()
                ),
                "tp_auc": (
                    float(roc_auc_score(actual_tp, p_tp))
                    if actual_tp.nunique() == 2
                    else np.nan
                ),
                "tp_brier": (
                    float(brier_score_loss(actual_tp, p_tp))
                    if len(actual_tp)
                    else np.nan
                ),
            }
        )
    diagnostics = pd.DataFrame(diagnostics)

    simulations = {}

    def simulate(fold_id: int, long_tau: float, short_tau: float):
        key = (fold_id, long_tau, short_tau)
        if key in simulations:
            return simulations[key]
        fold = folds[fold_id]
        frame = predictions[fold_id]
        filtered = apply_side_filter(
            frame[f"{MODEL}_pred"].astype(int),
            frame[f"{MODEL}_conf"].astype(float),
            probabilities["path_p_tp"],
            primary_tau=PRIMARY_TAU,
            long_tau=long_tau,
            short_tau=short_tau,
        )
        boundary = fold.validation_end + pd.Timedelta(minutes=15)
        path_safe = (
            filtered.index + pd.Timedelta(minutes=15 * (max_hold + 1))
            <= boundary
        )
        scope_bars = bars[
            (bars.index >= fold.validation_start) & (bars.index < boundary)
        ]
        result = simulate_bracket_trades_intrabar(
            scope_bars,
            minute,
            filtered.loc[path_safe],
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            max_hold=max_hold,
            fee_bps=fee,
        )
        simulations[key] = result
        return result

    def economic_scope(scope: str, fold_ids: tuple[int, ...]) -> pd.DataFrame:
        long_thresholds = _thresholds(
            probabilities["path_p_tp"], candidates, fold_ids, side=1
        )
        short_thresholds = _thresholds(
            probabilities["path_p_tp"], candidates, fold_ids, side=-1
        )
        rows = []
        for long_tau, short_tau in product(long_thresholds, short_thresholds):
            ledgers = []
            returns = []
            fold_nets = []
            for fold_id in fold_ids:
                ledger, per_bar = simulate(fold_id, long_tau, short_tau)
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
                    "tau": PRIMARY_TAU,
                    "tp_bps": tp_bps,
                    "sl_bps": sl_bps,
                    "max_hold": max_hold,
                    "long_tau": long_tau,
                    "short_tau": short_tau,
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
        selected = select_side_policy(inner, n_folds=len(inner_ids))
        if selected is None:
            audit_rows.append(
                {
                    "outer_month": month,
                    "decision": "no_trade",
                    "long_tau": None,
                    "short_tau": None,
                    "outer_trades": 0,
                    "outer_net": 0.0,
                    "outer_sortino": 0.0,
                    "outer_sharpe": 0.0,
                }
            )
            continue
        long_tau = float(selected["long_tau"])
        short_tau = float(selected["short_tau"])
        ledger, per_bar = simulate(outer_id, long_tau, short_tau)
        summary = economics_summary(per_bar)
        audit_rows.append(
            {
                "outer_month": month,
                "decision": "trade",
                "long_tau": long_tau,
                "short_tau": short_tau,
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
    selected = select_side_policy(final, n_folds=len(folds))

    grid.to_parquet(grid_path, index=False)
    audit.to_parquet(audit_path, index=False)
    diagnostics.to_parquet(diagnostic_path, index=False)
    joint_grid = pd.read_parquet(
        OUT_DIR / "joint_path_selection" / "economic_grid.parquet"
    )
    control = joint_grid[
        joint_grid["scope"].eq("final_development")
        & joint_grid["tp_bps"].eq(tp_bps)
        & joint_grid["sl_bps"].eq(sl_bps)
        & joint_grid["max_hold"].eq(max_hold)
        & joint_grid["trades"].eq(66)
    ].sort_values("pooled_net", ascending=False).iloc[0]
    payload = {
        "method": "causal side-specific regime-balanced TP-first CatBoost gates",
        "variant": variant,
        "path_features": list(path_features),
        "geometry": list(GEOMETRY),
        "primary": {
            "width": WIDTH,
            "candidate": CANDIDATE,
            "tau": PRIMARY_TAU,
        },
        "control_66": {key: _json_value(value) for key, value in control.items()},
        "decision": "no_trade" if selected is None else "trade",
        "selected_policy": (
            None
            if selected is None
            else {key: _json_value(value) for key, value in selected.items()}
        ),
        "outer_audit": audit.to_dict(orient="records"),
        "elapsed_s": round(time.time() - started, 1),
    }
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(diagnostics.to_string(index=False))
    print(audit.to_string(index=False))
    if selected is None:
        print("FINAL SIDE-MODEL DECISION: NO TRADE")
    else:
        print(
            "FINAL SIDE-MODEL DECISION: "
            f"long>={float(selected['long_tau']):.4f}, "
            f"short>={float(selected['short_tau']):.4f}"
        )
    print(f"wrote -> {result_path}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("baseline", "stage7d"), default="baseline"
    )
    args = parser.parse_args()
    raise SystemExit(main(variant=args.variant))

