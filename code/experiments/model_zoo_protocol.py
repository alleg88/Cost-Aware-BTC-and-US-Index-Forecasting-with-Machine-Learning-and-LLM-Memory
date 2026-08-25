"""Frozen constants and candidate pools for the nine-model BTC study."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import optuna

BASE_MODELS = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)
WIDTHS = (55, 65, 75)
TAUS = (0.0, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
LOOKBACK_DAYS = 90
SEED = 42
CANDIDATE_COUNT = 15
DEVELOPMENT_END_EXCLUSIVE = "2025-07-01"
DEVELOPMENT_MONTHS = tuple(
    [f"2024-{month:02d}" for month in range(4, 13)]
    + [f"2025-{month:02d}" for month in range(1, 7)]
)


def _space_logreg(trial: optuna.Trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}


def _space_decision_tree(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 20, 400, log=True
        ),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _space_random_forest(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "min_samples_leaf": trial.suggest_int(
            "min_samples_leaf", 20, 200, log=True
        ),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _space_svm(trial: optuna.Trial) -> dict:
    return {"C": trial.suggest_float("C", 1e-3, 10.0, log=True)}


def _space_xgboost(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.01, 0.15, log=True
        ),
        "min_child_weight": trial.suggest_int(
            "min_child_weight", 10, 500, log=True
        ),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
    }


def _space_catboost(trial: optuna.Trial) -> dict:
    return {
        "iterations": trial.suggest_int("iterations", 400, 800, step=100),
        "depth": trial.suggest_int("depth", 5, 7),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.03, 0.10, log=True
        ),
        "l2_leaf_reg": trial.suggest_float(
            "l2_leaf_reg", 10.0, 50.0, log=True
        ),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.5),
        "bagging_temperature": trial.suggest_float(
            "bagging_temperature", 0.0, 1.5
        ),
        "rsm": trial.suggest_float("rsm", 0.65, 1.0),
    }


def _space_mlp(trial: optuna.Trial) -> dict:
    width = trial.suggest_categorical("width", [64, 128, 256])
    layers = trial.suggest_int("layers", 1, 3)
    return {
        "hidden": tuple([width] * layers),
        "dropout": trial.suggest_float("dropout", 0.0, 0.4),
        "epochs": trial.suggest_int("epochs", 10, 40, step=5),
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
    }


def _space_sequence(trial: optuna.Trial) -> dict:
    return {
        "seq_len": trial.suggest_categorical("seq_len", [16, 32, 64]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 2),
        "epochs": trial.suggest_int("epochs", 5, 25, step=5),
        "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
    }


SPACES: dict[str, Callable[[optuna.Trial], dict]] = {
    "logreg": _space_logreg,
    "decision_tree": _space_decision_tree,
    "random_forest": _space_random_forest,
    "svm_linear": _space_svm,
    "xgboost_balanced": _space_xgboost,
    "catboost_balanced": _space_catboost,
    "mlp": _space_mlp,
    "lstm": _space_sequence,
    "gru": _space_sequence,
}


def candidate_pool(
    model: str,
    n_trials: int = CANDIDATE_COUNT,
    seed: int = SEED,
) -> list[dict]:
    """Return a baseline plus outcome-independent seeded random candidates."""
    if model not in BASE_MODELS:
        raise ValueError(f"unsupported base model: {model}")
    if n_trials < 2:
        raise ValueError("at least two candidates are required")

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.RandomSampler(seed=seed),
    )
    candidates: list[dict] = [{}]
    for _ in range(n_trials - 1):
        trial = study.ask()
        candidates.append(SPACES[model](trial))
        study.tell(trial, 0.0)
    return candidates


def protocol_payload() -> dict:
    """Return the outcome-independent identity of the frozen primary study."""
    return {
        "models": list(BASE_MODELS),
        "widths": list(WIDTHS),
        "taus": list(TAUS),
        "candidate_count": CANDIDATE_COUNT,
        "seed": SEED,
        "lookback_days": LOOKBACK_DAYS,
        "development_months": list(DEVELOPMENT_MONTHS),
        "fold_count": len(DEVELOPMENT_MONTHS),
        "development_end_exclusive": DEVELOPMENT_END_EXCLUSIVE,
        "train_tail_trim_bars": 1,
        "execution": {
            "tp_bps": 150.0,
            "sl_bps": 75.0,
            "max_hold_m15_bars": 1,
            "fee_bps_per_side": 5.0,
        },
        "guards": {
            "minimum_trades": 50,
            "minimum_trades_per_side": 15,
            "positive_fold_fraction": "at least 2/3",
            "positive_total_long_short_net": True,
            "positive_pooled_sortino_and_sharpe": True,
            "positive_bull_sideways_bear_sortino": True,
            "no_trade_allowed": True,
        },
    }


def protocol_fingerprint() -> str:
    """Return a short stable hash used to isolate resumable caches."""
    raw = json.dumps(protocol_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
