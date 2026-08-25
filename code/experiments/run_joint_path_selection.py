"""Joint nested selection of top-Sortino geometry and TP-first path gate."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import brier_score_loss, roc_auc_score

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.intrabar_candidates import TP_FIRST, build_first_touch_candidates
from experiments.run_geometry_selection import (
    GRID_PATH as GEOMETRY_GRID_PATH,
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
from experiments.run_tune_dz75_regime import REGIMES, past_regime_labels
from features.intrabar import (
    INTRABAR_FEATURES,
    STAGE_FEATURES,
    build_intrabar_features,
    build_market_stage_features,
)
from models.zoo import make_catboost

WIDTH = 75
CANDIDATE = 6
PRIMARY_TAU = 0.75
TOP_K = 3
MIN_PATH_TRAIN = 50
PATH_PARAMS = {
    "iterations": 300,
    "depth": 6,
    "learning_rate": 0.1,
    "l2_leaf_reg": 3.0,
    "random_seed": 42,
}
PRIMARY_FEATURES = (
    "primary_conf",
    "primary_p0",
    "primary_p1",
    "primary_p2",
)
PATH_FEATURES = (*PRIMARY_FEATURES, *INTRABAR_FEATURES, *STAGE_FEATURES, "side")

OUTPUT_DIR = OUT_DIR / "joint_path_selection"
GRID_PATH = OUTPUT_DIR / "economic_grid.parquet"
AUDIT_PATH = OUTPUT_DIR / "outer_audit.parquet"
DIAGNOSTIC_PATH = OUTPUT_DIR / "path_model_diagnostics.parquet"
RESULT_PATH = OUTPUT_DIR / "result.json"
PROBABILITY_DIR = OUTPUT_DIR / "probabilities"


def shortlist_sortino(grid: pd.DataFrame, *, top_k: int = TOP_K) -> pd.DataFrame:
    return grid.sort_values(
        ["pooled_sortino", "pooled_net"], ascending=[False, False]
    ).head(top_k)


def causal_training_mask(
    candidates: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.Series:
    close_time = pd.to_datetime(candidates["outcome_close_time"], utc=True)
    return close_time < pd.Timestamp(cutoff)


def apply_joint_filter(
    prediction: pd.Series,
    confidence: pd.Series,
    p_tp: pd.Series,
    *,
    primary_tau: float,
    path_tau: float,
) -> pd.Series:
    filtered = prediction.astype(int).copy()
    score = p_tp.reindex(filtered.index)
    keep = (
        filtered.isin((0, 2))
        & confidence.reindex(filtered.index).ge(float(primary_tau))
        & score.notna()
        & score.ge(float(path_tau))
    )
    filtered.loc[~keep] = 1
    return filtered


def select_joint_policy(
    grid: pd.DataFrame, *, n_folds: int
) -> pd.Series | None:
    return select_economic_candidate(grid, n_folds=n_folds)


def outer_audit_schedule() -> list[tuple[tuple[int, ...], int]]:
    return [
        (tuple(range(12)), 12),
        (tuple(range(13)), 13),
        (tuple(range(14)), 14),
    ]


def _geometry_key(row: pd.Series) -> tuple[int, int, int]:
    return int(row["tp_bps"]), int(row["sl_bps"]), int(row["max_hold"])


def _geometry_name(geometry: tuple[int, int, int]) -> str:
    tp_bps, sl_bps, max_hold = geometry
    return f"tp{tp_bps}_sl{sl_bps}_hold{max_hold}"


def _json_value(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _path_thresholds(
    probabilities: pd.Series,
    starts: pd.Series,
    fold_ids: tuple[int, ...],
) -> tuple[float, ...]:
    values = probabilities[starts.isin(fold_ids)].dropna().astype(float)
    if values.empty:
        return (0.0,)
    quantiles = np.quantile(values, np.linspace(0.0, 0.9, 10))
    return tuple(sorted({float(value) for value in quantiles}))


def main() -> int:
    started = time.time()
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    fee = float(cfg["instruments"]["btc"]["taker_fee_bps"])
    folds = monthly_development_folds()

    candidates_json = json.loads(
        (OUT_DIR / "candidates.json").read_text(encoding="utf-8")
    )
    params = candidates_json["candidates"][CANDIDATE]
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
    stage = build_market_stage_features(bars)
    features = pd.DataFrame(
        {
            "primary_conf": prediction_frame[f"{MODEL}_conf"].astype(float),
            "primary_p0": prediction_frame[f"{MODEL}_p0"].astype(float),
            "primary_p1": prediction_frame[f"{MODEL}_p1"].astype(float),
            "primary_p2": prediction_frame[f"{MODEL}_p2"].astype(float),
        },
        index=prediction_frame.index,
    ).join(intrabar).join(stage)
    features = features.dropna(subset=[*PRIMARY_FEATURES, *INTRABAR_FEATURES, *STAGE_FEATURES])
    all_signals = prediction_frame[f"{MODEL}_pred"].astype(int).reindex(features.index)

    geometry_grid = pd.read_parquet(GEOMETRY_GRID_PATH)
    scope_map = {
        f"outer_{folds[outer_id].validation_start:%Y-%m}_inner": inner_ids
        for inner_ids, outer_id in outer_audit_schedule()
    }
    scope_map["final_development"] = tuple(range(len(folds)))
    shortlists = {}
    geometry_union = set()
    for scope in scope_map:
        rows = shortlist_sortino(geometry_grid[geometry_grid["scope"].eq(scope)])
        geometries = tuple(_geometry_key(row) for _, row in rows.iterrows())
        shortlists[scope] = geometries
        geometry_union.update(geometries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PROBABILITY_DIR.mkdir(parents=True, exist_ok=True)
    path_candidates = {}
    path_probabilities = {}
    diagnostic_rows = []
    for geometry in sorted(geometry_union):
        tp_bps, sl_bps, max_hold = geometry
        candidate_frame = build_first_touch_candidates(
            minute,
            all_signals,
            features,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            max_hold=max_hold,
        )
        candidate_frame["fold_id"] = prediction_frame["fold_id"].reindex(
            candidate_frame.index
        ).astype(int)
        path_candidates[geometry] = candidate_frame

        probability_path = PROBABILITY_DIR / f"{_geometry_name(geometry)}.parquet"
        if probability_path.exists():
            probability = pd.read_parquet(probability_path)
            probability.index = pd.to_datetime(probability.index, utc=True)
            probability = probability.sort_index()
        else:
            probability = pd.DataFrame(
                np.nan,
                index=candidate_frame.index,
                columns=["path_p_sl", "path_p_timeout", "path_p_tp"],
            )
            probability["path_trained"] = False
            for fold_id, fold in enumerate(folds):
                predict_mask = candidate_frame["fold_id"].eq(fold_id)
                train_mask = causal_training_mask(
                    candidate_frame, fold.validation_start
                )
                y_train = candidate_frame.loc[train_mask, "outcome"].astype(int)
                if (
                    int(train_mask.sum()) < MIN_PATH_TRAIN
                    or int(predict_mask.sum()) == 0
                    or y_train.nunique() < 2
                ):
                    continue
                model = make_catboost(PATH_PARAMS)
                model.fit(
                    candidate_frame.loc[train_mask, PATH_FEATURES].astype(float),
                    y_train,
                )
                raw = np.asarray(
                    model.predict_proba(
                        candidate_frame.loc[predict_mask, PATH_FEATURES].astype(float)
                    ),
                    dtype=float,
                )
                mapped = np.zeros((int(predict_mask.sum()), 3), dtype=float)
                for source, label in enumerate(model.classes_):
                    mapped[:, int(label)] = raw[:, source]
                probability.loc[
                    predict_mask, ["path_p_sl", "path_p_timeout", "path_p_tp"]
                ] = mapped
                probability.loc[predict_mask, "path_trained"] = True
            probability.to_parquet(probability_path)
        path_probabilities[geometry] = probability

        trained = probability["path_trained"].fillna(False).astype(bool)
        actual_tp = candidate_frame.loc[trained, "outcome"].eq(TP_FIRST).astype(int)
        p_tp = probability.loc[trained, "path_p_tp"].astype(float)
        diagnostic_rows.append(
            {
                "tp_bps": tp_bps,
                "sl_bps": sl_bps,
                "max_hold": max_hold,
                "candidates": len(candidate_frame),
                "trained_predictions": int(trained.sum()),
                "tp_rate": float(candidate_frame["outcome"].eq(TP_FIRST).mean()),
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

    simulations = {}

    def simulate(
        geometry: tuple[int, int, int], fold_id: int, path_tau: float
    ):
        key = (*geometry, fold_id, path_tau)
        if key in simulations:
            return simulations[key]
        tp_bps, sl_bps, max_hold = geometry
        fold = folds[fold_id]
        frame = predictions[fold_id]
        filtered = apply_joint_filter(
            frame[f"{MODEL}_pred"].astype(int),
            frame[f"{MODEL}_conf"].astype(float),
            path_probabilities[geometry]["path_p_tp"],
            primary_tau=PRIMARY_TAU,
            path_tau=path_tau,
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

    def economic_scope(
        scope: str,
        fold_ids: tuple[int, ...],
        geometries: tuple[tuple[int, int, int], ...],
    ) -> pd.DataFrame:
        rows = []
        for geometry in geometries:
            candidates = path_candidates[geometry]
            probabilities = path_probabilities[geometry]["path_p_tp"]
            thresholds = _path_thresholds(
                probabilities, candidates["fold_id"], fold_ids
            )
            for path_tau in thresholds:
                ledgers = []
                returns = []
                fold_nets = []
                for fold_id in fold_ids:
                    ledger, per_bar = simulate(geometry, fold_id, path_tau)
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
                tp_bps, sl_bps, max_hold = geometry
                rows.append(
                    {
                        "scope": scope,
                        "width": WIDTH,
                        "candidate": CANDIDATE,
                        "tau": PRIMARY_TAU,
                        "tp_bps": tp_bps,
                        "sl_bps": sl_bps,
                        "max_hold": max_hold,
                        "path_tau": path_tau,
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
        scope = f"outer_{month}_inner"
        inner = economic_scope(scope, inner_ids, shortlists[scope])
        grid_parts.append(inner)
        selected = select_joint_policy(inner, n_folds=len(inner_ids))
        if selected is None:
            audit_rows.append(
                {
                    "outer_month": month,
                    "decision": "no_trade",
                    "tp_bps": None,
                    "sl_bps": None,
                    "max_hold": None,
                    "path_tau": None,
                    "outer_trades": 0,
                    "outer_net": 0.0,
                    "outer_sortino": 0.0,
                    "outer_sharpe": 0.0,
                }
            )
            continue
        geometry = _geometry_key(selected)
        path_tau = float(selected["path_tau"])
        ledger, per_bar = simulate(geometry, outer_id, path_tau)
        summary = economics_summary(per_bar)
        audit_rows.append(
            {
                "outer_month": month,
                "decision": "trade",
                "tp_bps": geometry[0],
                "sl_bps": geometry[1],
                "max_hold": geometry[2],
                "path_tau": path_tau,
                "outer_trades": len(ledger),
                "outer_net": float(ledger["net_return"].sum()),
                "outer_sortino": summary["sortino"],
                "outer_sharpe": summary["sharpe"],
            }
        )

    final = economic_scope(
        "final_development",
        tuple(range(len(folds))),
        shortlists["final_development"],
    )
    grid_parts.append(final)
    grid = pd.concat(grid_parts, ignore_index=True)
    audit = pd.DataFrame(audit_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    selected = select_joint_policy(final, n_folds=len(folds))

    grid.to_parquet(GRID_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)
    diagnostics.to_parquet(DIAGNOSTIC_PATH, index=False)
    payload = {
        "method": "top-three earlier-month Sortino geometries plus causal TP-first CatBoost gate",
        "top_k": TOP_K,
        "primary": {
            "width": WIDTH,
            "candidate": CANDIDATE,
            "tau": PRIMARY_TAU,
        },
        "shortlists": {
            scope: [list(geometry) for geometry in geometries]
            for scope, geometries in shortlists.items()
        },
        "decision": "no_trade" if selected is None else "trade",
        "selected_policy": (
            None
            if selected is None
            else {key: _json_value(value) for key, value in selected.items()}
        ),
        "outer_audit": audit.to_dict(orient="records"),
        "elapsed_s": round(time.time() - started, 1),
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(diagnostics.to_string(index=False))
    print(audit.to_string(index=False))
    if selected is None:
        print("FINAL JOINT DECISION: NO TRADE")
    else:
        print(
            "FINAL JOINT DECISION: "
            f"TP{int(selected['tp_bps'])}/SL{int(selected['sl_bps'])}/"
            f"H{int(selected['max_hold'])}, pTP>={float(selected['path_tau']):.4f}"
        )
    print(f"wrote -> {RESULT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
