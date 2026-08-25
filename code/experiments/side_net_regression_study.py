"""Execution core for the causal side-specific expected-net study."""
from __future__ import annotations

import json
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.intrabar_candidates import build_first_touch_candidates
from experiments.run_joint_path_selection import (
    CANDIDATE,
    PATH_PARAMS,
    PRIMARY_FEATURES,
    PRIMARY_TAU,
    WIDTH,
)
from experiments.run_side_net_regression import (
    GEOMETRY,
    OUTPUT_DIR,
    apply_net_filter,
    net_training_mask,
    realized_net_target,
    select_net_policy,
)
from experiments.run_side_path_selection import (
    MIN_PATH_TRAIN,
    PATH_FEATURES,
    THRESHOLD_QUANTILES,
    outer_audit_schedule,
)
from experiments.run_tune_antibull_widths import (
    CONFIG,
    DEVELOPMENT_END,
    MINUTE_PATH,
    MODEL,
    OUT_DIR,
    monthly_development_folds,
    prediction_cache_path,
)
from experiments.run_tune_dz75_regime import (
    REGIMES,
    past_regime_labels,
    regime_balanced_weights,
)
from features.intrabar import (
    INTRABAR_FEATURES,
    STAGE_FEATURES,
    build_intrabar_features,
    build_market_stage_features,
)
from models.zoo import make_catboost_regressor

GRID_PATH = OUTPUT_DIR / "economic_grid.parquet"
AUDIT_PATH = OUTPUT_DIR / "outer_audit.parquet"
DIAGNOSTIC_PATH = OUTPUT_DIR / "regression_diagnostics.parquet"
PREDICTION_PATH = OUTPUT_DIR / "expected_net.parquet"
RESULT_PATH = OUTPUT_DIR / "result.json"


def _thresholds(
    expected_net: pd.Series,
    candidates: pd.DataFrame,
    fold_ids: tuple[int, ...],
    *,
    side: int,
) -> tuple[float, ...]:
    mask = candidates["fold_id"].isin(fold_ids) & candidates["side"].eq(side)
    values = expected_net.loc[mask].dropna().astype(float)
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


def _best_net_control(path: Path) -> dict | None:
    if not path.exists():
        return None
    grid = pd.read_parquet(path)
    final = grid.loc[grid["scope"].eq("final_development")]
    if final.empty:
        return None
    row = final.sort_values("pooled_net", ascending=False).iloc[0]
    return {key: _json_value(value) for key, value in row.items()}


def run_study() -> int:
    started = time.time()
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
    minute = minute.sort_index().loc[
        lambda frame: frame.index < DEVELOPMENT_END
    ]
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
    features = features.dropna(
        subset=[*PRIMARY_FEATURES, *INTRABAR_FEATURES, *STAGE_FEATURES]
    )
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
    target = realized_net_target(candidates, fee_bps=fee)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if PREDICTION_PATH.exists():
        expected = pd.read_parquet(PREDICTION_PATH)
        expected.index = pd.to_datetime(expected.index, utc=True)
        expected = expected.sort_index()
    else:
        expected = pd.DataFrame(
            {
                "expected_net": np.nan,
                "net_trained": False,
                "model_side": 0,
            },
            index=candidates.index,
        )
        for fold_id, fold in enumerate(folds):
            for side in (1, -1):
                predict_mask = candidates["fold_id"].eq(fold_id) & candidates[
                    "side"
                ].eq(side)
                train_mask = net_training_mask(
                    candidates, fold.validation_start, side=side
                )
                if (
                    int(train_mask.sum()) < MIN_PATH_TRAIN
                    or int(predict_mask.sum()) == 0
                ):
                    continue
                model = make_catboost_regressor(PATH_PARAMS)
                weights = regime_balanced_weights(
                    candidates.loc[train_mask, "regime"]
                )
                model.fit(
                    candidates.loc[train_mask, list(PATH_FEATURES)].astype(float),
                    target.loc[train_mask],
                    sample_weight=weights,
                )
                expected.loc[predict_mask, "expected_net"] = np.asarray(
                    model.predict(
                        candidates.loc[
                            predict_mask, list(PATH_FEATURES)
                        ].astype(float)
                    ),
                    dtype=float,
                ).reshape(-1)
                expected.loc[predict_mask, "net_trained"] = True
                expected.loc[predict_mask, "model_side"] = side
        expected.to_parquet(PREDICTION_PATH)

    diagnostics = []
    for side, label in ((1, "long"), (-1, "short")):
        trained = (
            expected["net_trained"].fillna(False).astype(bool)
            & candidates["side"].eq(side)
        )
        actual = target.loc[trained].astype(float)
        predicted = expected.loc[trained, "expected_net"].astype(float)
        diagnostics.append(
            {
                "side": label,
                "candidates": int(candidates["side"].eq(side).sum()),
                "trained_predictions": int(trained.sum()),
                "positive_net_rate": float(
                    target.loc[candidates["side"].eq(side)].gt(0.0).mean()
                ),
                "target_mean": float(actual.mean()),
                "prediction_mean": float(predicted.mean()),
                "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae": float(mean_absolute_error(actual, predicted)),
                "rank_correlation": float(
                    actual.rank().corr(predicted.rank())
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
        filtered = apply_net_filter(
            frame[f"{MODEL}_pred"].astype(int),
            frame[f"{MODEL}_conf"].astype(float),
            expected["expected_net"],
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

    def economic_scope(
        scope: str, fold_ids: tuple[int, ...]
    ) -> pd.DataFrame:
        long_thresholds = _thresholds(
            expected["expected_net"], candidates, fold_ids, side=1
        )
        short_thresholds = _thresholds(
            expected["expected_net"], candidates, fold_ids, side=-1
        )
        rows = []
        for long_tau, short_tau in product(
            long_thresholds, short_thresholds
        ):
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
                    "robust_score": min(
                        summary["sortino"],
                        summary["sharpe"],
                        *regime_sortino.values(),
                    ),
                }
            )
        return pd.DataFrame(rows)

    grid_parts = []
    audit_rows = []
    for inner_ids, outer_id in outer_audit_schedule():
        month = f"{folds[outer_id].validation_start:%Y-%m}"
        inner = economic_scope(f"outer_{month}_inner", inner_ids)
        grid_parts.append(inner)
        selected = select_net_policy(inner, n_folds=len(inner_ids))
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
    selected = select_net_policy(final, n_folds=len(folds))
    best_net = final.sort_values("pooled_net", ascending=False).iloc[0]

    grid.to_parquet(GRID_PATH, index=False)
    audit.to_parquet(AUDIT_PATH, index=False)
    diagnostics.to_parquet(DIAGNOSTIC_PATH, index=False)
    controls = {
        "tp_first": _best_net_control(
            OUT_DIR / "side_path_selection" / "economic_grid.parquet"
        ),
        "stage7d": _best_net_control(
            OUT_DIR / "side_path_stage7d_selection" / "economic_grid.parquet"
        ),
    }
    payload = {
        "method": "causal side-specific regime-balanced expected-net CatBoost regressors",
        "target": "gross return minus fixed round-trip fees",
        "geometry": list(GEOMETRY),
        "primary": {
            "width": WIDTH,
            "candidate": CANDIDATE,
            "tau": PRIMARY_TAU,
        },
        "path_features": list(PATH_FEATURES),
        "regressor_params": PATH_PARAMS,
        "controls": controls,
        "best_net_policy": {
            key: _json_value(value) for key, value in best_net.items()
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
    print(
        "BEST NET: "
        f"{int(best_net['trades'])} trades, "
        f"net={float(best_net['pooled_net']):.2%}, "
        f"Sortino={float(best_net['pooled_sortino']):.2f}"
    )
    if selected is None:
        print("FINAL EXPECTED-NET DECISION: NO TRADE")
    else:
        print(
            "FINAL EXPECTED-NET DECISION: "
            f"long>={float(selected['long_tau']):.6f}, "
            f"short>={float(selected['short_tau']):.6f}"
        )
    print(f"wrote -> {RESULT_PATH}")
    return 0
