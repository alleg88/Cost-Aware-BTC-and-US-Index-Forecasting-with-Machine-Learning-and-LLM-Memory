"""Regime-robust Optuna tuning for BTC Balanced CatBoost at dz75.

Only 2024 development rows are used. Training samples are weighted so bull,
sideways, and bear regimes contribute equal total weight inside every fold.
The Optuna objective maximises the mean weakest-regime macro-F1 across five
non-overlapping blocked time-series folds. Forward evaluation remains untouched.

Run:  python -m experiments.run_tune_dz75_regime --trials 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from evaluation.splits import BlockingTimeSeriesSplit
from experiments.run_walkforward import build_walkforward_xy
from features.build import make_label
from models.zoo import MODELS

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
OUTPUT = (
    CODE_ROOT / "experiments" / "cache" / "tuning"
    / "btc_bothofpos_dz75_regime_catboost_balanced.json"
)

WIDTH = 75
REGIME_LOOKBACK = 672  # seven days of M15 bars
REGIME_THRESHOLD = 0.02
REGIMES = ("bull", "sideways", "bear")
MIN_REGIME_VALIDATION = 50
BASELINE_PARAMS = {
    "iterations": 500,
    "depth": 7,
    "learning_rate": 0.039579251400866544,
    "l2_leaf_reg": 18.821242414979505,
}


def past_regime_labels(
    close: pd.Series,
    *,
    lookback: int = REGIME_LOOKBACK,
    threshold: float = REGIME_THRESHOLD,
) -> pd.Series:
    """Classify market state using trailing returns available at that timestamp."""
    trailing = close.astype(float) / close.astype(float).shift(lookback) - 1.0
    regime = pd.Series("sideways", index=close.index, dtype="object", name="regime")
    regime.loc[trailing > threshold] = "bull"
    regime.loc[trailing < -threshold] = "bear"
    regime.loc[trailing.isna()] = "unknown"
    return regime


def regime_balanced_weights(regimes: pd.Series) -> pd.Series:
    """Give every observed regime the same aggregate training influence."""
    counts = regimes.value_counts()
    if counts.empty:
        raise ValueError("cannot weight an empty regime series")
    weights = regimes.map(len(regimes) / (len(counts) * counts)).astype(float)
    return weights.rename("regime_weight")


def robust_regime_score(scores: dict[str, float]) -> float:
    """Score a fold by its weakest market regime, not its dominant regime."""
    if not scores:
        raise ValueError("at least one regime score is required")
    return float(min(scores.values()))


def sample_params(trial: optuna.Trial) -> dict:
    """Conservative CatBoost search space for noisy financial tabular data."""
    return {
        "iterations": trial.suggest_int("iterations", 300, 800, step=100),
        "depth": trial.suggest_int("depth", 4, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.10, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 5.0, 50.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 2.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "rsm": trial.suggest_float("rsm", 0.6, 1.0),
    }


def evaluate_params(
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    regimes: pd.Series,
    splitter: BlockingTimeSeriesSplit,
) -> dict:
    fold_worst, fold_overall, fold_details = [], [], []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(X)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        train_regime = regimes.iloc[train_idx]
        valid_regime = regimes.iloc[valid_idx]

        model = MODELS["catboost_balanced"](params)
        model.fit(
            X_train,
            y_train,
            sample_weight=regime_balanced_weights(train_regime),
        )
        prediction = np.asarray(model.predict(X_valid)).reshape(-1).astype(int)
        overall = float(
            f1_score(y_valid, prediction, labels=[0, 1, 2], average="macro", zero_division=0)
        )
        regime_scores = {}
        for regime in REGIMES:
            mask = valid_regime.to_numpy() == regime
            if int(mask.sum()) < MIN_REGIME_VALIDATION:
                raise ValueError(
                    f"fold {fold} has only {int(mask.sum())} {regime} validation rows"
                )
            regime_scores[regime] = float(
                f1_score(
                    y_valid.iloc[np.flatnonzero(mask)],
                    prediction[mask],
                    labels=[0, 1, 2],
                    average="macro",
                    zero_division=0,
                )
            )
        worst = robust_regime_score(regime_scores)
        fold_worst.append(worst)
        fold_overall.append(overall)
        fold_details.append(
            {"fold": fold, "overall_macro_f1": overall,
             "worst_regime_macro_f1": worst, **regime_scores}
        )
    return {
        "robust_score": float(np.mean(fold_worst)),
        "overall_macro_f1": float(np.mean(fold_overall)),
        "robust_std": float(np.std(fold_worst, ddof=0)),
        "folds": fold_details,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=30)
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    X, y, _ = build_walkforward_xy(
        "btc",
        cfg,
        horizon=1,
        sentiment="both",
        label_fn=lambda feat: make_label(feat, threshold_bps=WIDTH, horizon=1),
        orderflow=True,
        positioning=True,
    )
    bars = pd.read_parquet(CODE_ROOT / cfg["instruments"]["btc"]["working_parquet"])
    bars.index = pd.to_datetime(bars.index, utc=True)
    regimes = past_regime_labels(bars.sort_index()["close"]).reindex(X.index)
    dev_start = pd.Timestamp(cfg["dates"]["train"][0], tz="UTC")
    dev_end = pd.Timestamp(cfg["dates"]["train"][1], tz="UTC") + pd.Timedelta(days=1)
    dev = (X.index >= dev_start) & (X.index < dev_end) & regimes.isin(REGIMES)
    X, y, regimes = X.loc[dev], y.loc[dev], regimes.loc[dev]

    sp = cfg["split"]
    splitter = BlockingTimeSeriesSplit(
        sp["n_splits"], sp["train_frac"], max(sp["embargo_bars"], 1)
    )
    print(
        f"dz75 regime tuning: {len(X):,} 2024 rows | "
        f"regimes={regimes.value_counts().to_dict()} | trials={args.trials}"
    )
    baseline = evaluate_params(BASELINE_PARAMS, X, y, regimes, splitter)
    print(
        f"baseline robust={baseline['robust_score']:.4f} "
        f"overall={baseline['overall_macro_f1']:.4f}"
    )

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )

    def objective(trial: optuna.Trial) -> float:
        result = evaluate_params(sample_params(trial), X, y, regimes, splitter)
        trial.set_user_attr("overall_macro_f1", result["overall_macro_f1"])
        trial.set_user_attr("robust_std", result["robust_std"])
        trial.set_user_attr("folds", result["folds"])
        return result["robust_score"]

    started = time.time()
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
    best = study.best_trial
    payload = {
        "model": "catboost_balanced",
        "instrument": "btc",
        "features": "bothofpos",
        "label": "dz75",
        "development_window": "2024 only",
        "objective": "mean weakest-regime macro-F1 across blocked folds",
        "regime_definition": {
            "lookback_bars": REGIME_LOOKBACK,
            "bull": f"trailing 7d return > +{REGIME_THRESHOLD:.0%}",
            "bear": f"trailing 7d return < -{REGIME_THRESHOLD:.0%}",
            "sideways": "otherwise",
        },
        "regime_counts": regimes.value_counts().to_dict(),
        "trials": args.trials,
        "elapsed_s": round(time.time() - started, 1),
        "baseline_params": BASELINE_PARAMS,
        "baseline_robust_score": baseline["robust_score"],
        "baseline_overall_macro_f1": baseline["overall_macro_f1"],
        "best_params": dict(best.params),
        "best_robust_score": float(best.value),
        "best_overall_macro_f1": float(best.user_attrs["overall_macro_f1"]),
        "best_robust_std": float(best.user_attrs["robust_std"]),
        "best_folds": best.user_attrs["folds"],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"best robust={payload['best_robust_score']:.4f} "
        f"overall={payload['best_overall_macro_f1']:.4f}"
    )
    print(f"best params={payload['best_params']}")
    print(f"wrote -> {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
