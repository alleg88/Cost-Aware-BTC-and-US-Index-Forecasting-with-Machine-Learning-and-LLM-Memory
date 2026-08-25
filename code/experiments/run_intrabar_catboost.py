"""Balanced CatBoost TP-first meta-filter over frozen 90D BTC predictions."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

from evaluation.economics import diebold_mariano, economics_summary
from evaluation.trades import trade_stats
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.intrabar_candidates import TP_FIRST, build_first_touch_candidates
from experiments.spans import CALIBRATION_END, LOCKBOX_START
from features.build import POSITIONING_FEATURE_COLS, add_features
from features.intrabar import (
    INTRABAR_FEATURES,
    STAGE_FEATURES,
    build_intrabar_features,
    build_market_stage_features,
)
from models.zoo import MODELS

SPLIT = pd.Timestamp(CALIBRATION_END, tz="UTC")
LOCKBOX = pd.Timestamp(LOCKBOX_START, tz="UTC")
CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2024_2026.parquet"
PREDICTION_PATH = (
    CODE_ROOT / "experiments" / "cache" / "walkforward"
    / "btc_bothofpos_cb-econ_dz40_lb90d_to2026.parquet"
)
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "economics"
SUMMARY_PATH = OUT_DIR / "btc_balanced_90d_intrabar_meta_summary.parquet"
CANDIDATE_PATH = OUT_DIR / "btc_balanced_90d_intrabar_meta_candidates.parquet"
RETURN_PATH = OUT_DIR / "btc_balanced_90d_intrabar_meta_returns.parquet"

MODEL = "catboost_balanced"
PRIMARY_TAU = 0.60
MAX_HOLD = 1
FLOOR = 50
TP_GRID = (50.0, 100.0, 150.0)
SL_GRID = (25.0, 50.0, 75.0)
PRIMARY_FEATURES = ("primary_conf", "primary_p0", "primary_p1", "primary_p2")
META_FEATURES = (
    *PRIMARY_FEATURES, *INTRABAR_FEATURES, *STAGE_FEATURES,
    *POSITIONING_FEATURE_COLS, "side",
)


def weekly_meta_probabilities(
    candidates: pd.DataFrame,
    feature_cols: tuple[str, ...],
    *,
    min_train: int = 50,
    model_factory: Callable | None = None,
) -> pd.DataFrame:
    """Predict each week using only candidate outcomes closed before it starts."""
    required = {"outcome", "outcome_close_time", "prediction_week_start", *feature_cols}
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"missing candidate columns: {sorted(missing)}")
    factory = model_factory or (lambda: MODELS["catboost_balanced"]())
    output = pd.DataFrame(
        {
            "meta_p_sl": 1.0 / 3.0,
            "meta_p_timeout": 1.0 / 3.0,
            "meta_p_tp": 1.0 / 3.0,
            "meta_trained": False,
        },
        index=candidates.index,
    )
    outcome_close = pd.to_datetime(candidates["outcome_close_time"], utc=True)
    week_start = pd.to_datetime(candidates["prediction_week_start"], utc=True)
    X = candidates.loc[:, feature_cols].astype(float)

    for week in sorted(week_start.unique()):
        predict_mask = week_start == week
        train_mask = outcome_close < week
        y_train = candidates.loc[train_mask, "outcome"].astype(int)
        if train_mask.sum() < min_train or y_train.nunique() < 2:
            continue
        model = factory()
        model.fit(X.loc[train_mask], y_train)
        raw = np.asarray(model.predict_proba(X.loc[predict_mask]), dtype=float)
        probabilities = np.zeros((predict_mask.sum(), 3), dtype=float)
        for source, label in enumerate(model.classes_):
            label = int(label)
            if label in (0, 1, 2):
                probabilities[:, label] = raw[:, source]
        output.loc[predict_mask, ["meta_p_sl", "meta_p_timeout", "meta_p_tp"]] = probabilities
        output.loc[predict_mask, "meta_trained"] = True
    return output


def experiment_masks(index: pd.DatetimeIndex):
    """Return calibration and forward-evaluation masks with the lockbox excluded."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    return idx < SPLIT, (idx >= SPLIT) & (idx < LOCKBOX)


def outcome_safe_signal_mask(
    index: pd.DatetimeIndex, *, max_hold: int
):
    """Exclude signals whose next-open holding path would enter the lockbox."""
    idx = pd.DatetimeIndex(pd.to_datetime(index, utc=True))
    path_end = idx + pd.Timedelta(minutes=15 * (max_hold + 1))
    return path_end <= LOCKBOX


def apply_meta_filter(
    signals: pd.Series, p_tp: pd.Series, *, threshold: float
) -> pd.Series:
    """Keep a directional signal only when its TP-first probability clears the gate."""
    filtered = signals.astype(int).copy()
    score = p_tp.reindex(filtered.index)
    reject = (filtered != 1) & (score.isna() | (score < float(threshold)))
    filtered.loc[reject] = 1
    return filtered


def calibration_probability_thresholds(
    probabilities: pd.Series,
    trained: pd.Series,
    *,
    quantiles: Sequence[float] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
) -> tuple[float, ...]:
    """Build candidate gates from trained calibration probabilities only."""
    index = pd.DatetimeIndex(pd.to_datetime(probabilities.index, utc=True))
    mask = (index < SPLIT) & trained.reindex(probabilities.index).fillna(False).to_numpy(bool)
    values = probabilities.loc[mask].dropna().astype(float).to_numpy()
    if not len(values):
        return (0.0,)
    cutoffs = np.quantile(values, np.asarray(tuple(quantiles), dtype=float))
    return tuple(sorted({0.0, *(float(value) for value in cutoffs)}))


def calibrate_meta_threshold(
    thresholds: Sequence[float],
    evaluate: Callable[[float], dict[str, float]],
    *,
    floor: int = 50,
) -> tuple[float, float] | None:
    """Choose the highest calibration Sortino among policies meeting the floor."""
    best: tuple[float, float] | None = None
    for threshold in thresholds:
        summary = evaluate(float(threshold))
        if int(summary["trade_count"]) < floor:
            continue
        score = float(summary["sortino"])
        if best is None or score > best[1]:
            best = (float(threshold), score)
    return best



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

    positioning = pd.read_parquet(
        CODE_ROOT / "data" / "btcusdt_positioning_m15_2024_2026.parquet"
    )
    positioning.index = pd.to_datetime(positioning.index, utc=True)
    positioning = positioning.sort_index()
    engineered = add_features(bars.join(positioning.reindex(bars.index)))

    print("building causal 1m and market-stage features...")
    intrabar = build_intrabar_features(minute)
    stage = build_market_stage_features(bars)

    predictions = pd.read_parquet(PREDICTION_PATH)
    predictions.index = pd.to_datetime(predictions.index, utc=True)
    predictions = predictions.sort_index()
    predictions = predictions[predictions.index < LOCKBOX]
    pred = predictions[f"{MODEL}_pred"].astype(int)
    conf = predictions[f"{MODEL}_conf"].astype(float)
    signals = pred.where(conf >= PRIMARY_TAU, 1)

    features = pd.DataFrame(
        {
            "primary_conf": conf,
            "primary_p0": predictions[f"{MODEL}_p0"].astype(float),
            "primary_p1": predictions[f"{MODEL}_p1"].astype(float),
            "primary_p2": predictions[f"{MODEL}_p2"].astype(float),
            "prediction_week_start": pd.to_datetime(
                predictions["validation_start"], utc=True
            ),
        },
        index=predictions.index,
    )
    features = features.join(intrabar).join(stage)
    features = features.join(engineered[POSITIONING_FEATURE_COLS])
    feature_required = [
        *PRIMARY_FEATURES,
        *INTRABAR_FEATURES,
        *STAGE_FEATURES,
        *POSITIONING_FEATURE_COLS,
    ]
    features = features.dropna(subset=feature_required)
    signals = signals.reindex(features.index)
    safe = outcome_safe_signal_mask(signals.index, max_hold=MAX_HOLD)
    signals = signals.loc[safe]
    features = features.reindex(signals.index)

    bars_from_predictions = bars[bars.index >= predictions.index.min()]
    calibration_path_safe = (
        signals.index + pd.Timedelta(minutes=15 * (MAX_HOLD + 1)) <= SPLIT
    )
    calibration_signals = signals.loc[calibration_path_safe]
    calibration_bars = bars_from_predictions[bars_from_predictions.index < SPLIT]

    best_geometry = None
    for tp_bps, sl_bps in product(TP_GRID, SL_GRID):
        ledger, per_bar = simulate_bracket_trades_intrabar(
            calibration_bars,
            minute,
            calibration_signals,
            None,
            tau=0.0,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            max_hold=MAX_HOLD,
            fee_bps=fee,
        )
        if len(ledger) < FLOOR:
            continue
        score = economics_summary(per_bar)["sortino"]
        if best_geometry is None or score > best_geometry[2]:
            best_geometry = (tp_bps, sl_bps, score)
    if best_geometry is None:
        raise RuntimeError("no calibration bracket geometry meets the 50-trade floor")
    tp_bps, sl_bps, geometry_sortino = best_geometry
    print(
        f"calibration geometry: tp={tp_bps:g} sl={sl_bps:g} "
        f"hold={MAX_HOLD} (Sortino {geometry_sortino:+.3f})"
    )

    candidates = build_first_touch_candidates(
        minute,
        signals,
        features,
        tp_bps=tp_bps,
        sl_bps=sl_bps,
        max_hold=MAX_HOLD,
    )
    print(f"fitting weekly Balanced CatBoost meta-models on {len(candidates):,} candidates...")
    meta = weekly_meta_probabilities(candidates, META_FEATURES, min_train=FLOOR)
    p_tp = meta["meta_p_tp"]

    def run_policy(scope_bars: pd.DataFrame, scope_signals: pd.Series):
        return simulate_bracket_trades_intrabar(
            scope_bars,
            minute,
            scope_signals,
            None,
            tau=0.0,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            max_hold=MAX_HOLD,
            fee_bps=fee,
        )

    def evaluate_calibration(threshold: float) -> dict[str, float]:
        gated = (
            calibration_signals
            if threshold == 0.0
            else apply_meta_filter(calibration_signals, p_tp, threshold=threshold)
        )
        ledger, per_bar = run_policy(calibration_bars, gated)
        return {
            "sortino": economics_summary(per_bar)["sortino"],
            "trade_count": len(ledger),
        }

    meta_thresholds = calibration_probability_thresholds(
        p_tp, meta["meta_trained"]
    )
    threshold_pick = calibrate_meta_threshold(
        meta_thresholds, evaluate_calibration, floor=FLOOR
    )
    if threshold_pick is None:
        raise RuntimeError("no calibration meta threshold meets the 50-trade floor")
    meta_threshold, threshold_sortino = threshold_pick
    print(
        f"calibration meta threshold: {meta_threshold:.2f} "
        f"(Sortino {threshold_sortino:+.3f})"
    )

    evaluation_bars = bars_from_predictions[
        (bars_from_predictions.index >= SPLIT) & (bars_from_predictions.index < LOCKBOX)
    ]
    evaluation_signals = signals[
        (signals.index >= SPLIT) & (signals.index < LOCKBOX)
    ]
    control_ledger, control_returns = run_policy(evaluation_bars, evaluation_signals)
    filtered_signals = (
        evaluation_signals
        if meta_threshold == 0.0
        else apply_meta_filter(evaluation_signals, p_tp, threshold=meta_threshold)
    )
    meta_ledger, meta_returns = run_policy(evaluation_bars, filtered_signals)

    scored = candidates.join(meta)
    scored_eval = scored[
        (scored.index >= SPLIT)
        & (pd.to_datetime(scored["outcome_close_time"], utc=True) <= LOCKBOX)
        & scored["meta_trained"].astype(bool)
    ]
    binary_tp = (scored_eval["outcome"].astype(int) == TP_FIRST).astype(int)
    if len(scored_eval) and binary_tp.nunique() == 2:
        auc = float(roc_auc_score(binary_tp, scored_eval["meta_p_tp"]))
        brier = float(brier_score_loss(binary_tp, scored_eval["meta_p_tp"]))
    else:
        auc = np.nan
        brier = np.nan

    quarters = {
        "2025Q3": (pd.Timestamp("2025-07-01", tz="UTC"), pd.Timestamp("2025-10-01", tz="UTC")),
        "2025Q4": (pd.Timestamp("2025-10-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")),
        "2026Q1": (pd.Timestamp("2026-01-01", tz="UTC"), LOCKBOX),
    }

    def policy_summary(name: str, ledger: pd.DataFrame, per_bar: pd.Series) -> dict:
        summary = economics_summary(per_bar)
        summary.update(trade_stats(ledger))
        summary.update(
            {
                "variant": name,
                "primary_tau": PRIMARY_TAU,
                "tp_bps": tp_bps,
                "sl_bps": sl_bps,
                "max_hold": MAX_HOLD,
                "meta_threshold": 0.0 if name == "unfiltered" else meta_threshold,
                "cal_geometry_sortino": geometry_sortino,
                "cal_threshold_sortino": threshold_sortino,
                "eval_gross": float(ledger["gross_return"].sum()),
                "eval_net": float(ledger["net_return"].sum()),
                "eval_sortino": summary["sortino"],
                "eval_events": int(len(ledger)),
                "turnover_sides": int(2 * len(ledger)),
                "long_net": float(ledger.loc[ledger["side"] == 1, "net_return"].sum()),
                "short_net": float(ledger.loc[ledger["side"] == -1, "net_return"].sum()),
                "meta_auc": auc,
                "meta_brier": brier,
            }
        )
        positive = 0
        entry = pd.to_datetime(ledger["entry_time"], utc=True)
        for quarter, (start, end) in quarters.items():
            q = ledger[(entry >= start) & (entry < end)]
            gross = float(q["gross_return"].sum())
            summary[f"{quarter}_gross"] = gross
            summary[f"{quarter}_net"] = float(q["net_return"].sum())
            positive += int(gross > 0.0)
        summary["positive_gross_quarters"] = positive
        return summary

    control_summary = policy_summary("unfiltered", control_ledger, control_returns)
    filtered_summary = policy_summary("meta_filtered", meta_ledger, meta_returns)
    dm = diebold_mariano(control_returns, meta_returns, lag=MAX_HOLD)
    filtered_summary["dm_vs_unfiltered"] = dm["dm_stat"]
    filtered_summary["dm_p"] = dm["p_value"]
    control_summary["dm_vs_unfiltered"] = np.nan
    control_summary["dm_p"] = np.nan
    control_summary["screen_pass"] = False
    filtered_summary["screen_pass"] = bool(
        filtered_summary["eval_gross"] > 0.0
        and filtered_summary["eval_net"] > 0.0
        and filtered_summary["positive_gross_quarters"] >= 2
        and filtered_summary["eval_events"] >= FLOOR
        and np.isfinite(auc)
        and auc > 0.5
    )

    table = pd.DataFrame([control_summary, filtered_summary])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_parquet(SUMMARY_PATH, index=False)
    scored.reset_index().to_parquet(CANDIDATE_PATH, index=False)
    pd.DataFrame(
        {"unfiltered": control_returns, "meta_filtered": meta_returns}
    ).to_parquet(RETURN_PATH)

    columns = [
        "variant", "primary_tau", "tp_bps", "sl_bps", "meta_threshold",
        "eval_gross", "eval_net", "eval_sortino", "eval_events",
        "turnover_sides", "long_net", "short_net", "tp_rate", "sl_rate",
        "timeout_rate", "positive_gross_quarters", "meta_auc", "meta_brier",
        "dm_vs_unfiltered", "dm_p", "screen_pass",
    ]
    pd.set_option("display.width", 240)
    print(table[columns].to_string(index=False))
    print(f"wrote -> {SUMMARY_PATH.name}, {CANDIDATE_PATH.name}, {RETURN_PATH.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

