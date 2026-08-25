"""Optuna hyperparameter tuning on the blocking CV (PROJECT-PLAN §5).

Tunes one registry model at a time against mean macro-F1 over the same
BlockingTimeSeriesSplit folds the model zoo uses, on the 2024 train year only —
2025 and the lockbox stay untouched, so tuning cannot leak into evaluation.
Best parameters are cached as JSON; `run_model_zoo --tuned` picks them up.

Run:  python -m experiments.run_tuning --instrument btc --sentiment both --model catboost_balanced --trials 30
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import yaml

from evaluation.splits import BlockingTimeSeriesSplit
from experiments.horizons import DEFAULT_LABEL, horizon_label, parse_horizon
from experiments.run_model_zoo import load_xy
from models.zoo import MODELS, run_cv

CODE_ROOT = Path(__file__).resolve().parents[1]
CONFIG = CODE_ROOT / "configs" / "default.yaml"
TUNING_DIR = CODE_ROOT / "experiments" / "cache" / "tuning"


def _space_catboost(t: optuna.Trial) -> dict:
    return {
        "iterations": t.suggest_int("iterations", 200, 800, step=100),
        "depth": t.suggest_int("depth", 4, 8),
        "learning_rate": t.suggest_float("learning_rate", 0.03, 0.2, log=True),
        "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
    }


def _space_xgboost(t: optuna.Trial) -> dict:
    # Widened toward conservative, well-resolved probabilities: shallower trees,
    # heavier leaf/regularisation, and max_delta_step (XGBoost's documented
    # remedy for over-confident updates under class imbalance).
    return {
        "n_estimators": t.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": t.suggest_int("max_depth", 2, 6),
        "learning_rate": t.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "min_child_weight": t.suggest_int("min_child_weight", 10, 500, log=True),
        "subsample": t.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": t.suggest_float("reg_lambda", 1.0, 50.0, log=True),
        "reg_alpha": t.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "gamma": t.suggest_float("gamma", 1e-3, 5.0, log=True),
        "max_delta_step": t.suggest_int("max_delta_step", 0, 10),
    }


def _space_random_forest(t: optuna.Trial) -> dict:
    return {
        "n_estimators": t.suggest_int("n_estimators", 200, 600, step=100),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 20, 200, log=True),
        "max_features": t.suggest_float("max_features", 0.3, 1.0),
    }


def _space_mlp(t: optuna.Trial) -> dict:
    width = t.suggest_categorical("width", [64, 128, 256])
    depth = t.suggest_int("layers", 1, 3)
    return {
        "hidden": tuple([width] * depth),
        "dropout": t.suggest_float("dropout", 0.0, 0.4),
        "epochs": t.suggest_int("epochs", 10, 40, step=5),
        "lr": t.suggest_float("lr", 3e-4, 3e-3, log=True),
    }


def _space_sequence(t: optuna.Trial) -> dict:
    return {
        "seq_len": t.suggest_categorical("seq_len", [16, 32, 64]),
        "hidden_size": t.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": t.suggest_int("num_layers", 1, 2),
        "epochs": t.suggest_int("epochs", 5, 25, step=5),
        "lr": t.suggest_float("lr", 3e-4, 3e-3, log=True),
    }


SPACES = {
    "catboost_balanced": _space_catboost,
    "xgboost_balanced": _space_xgboost,
    "random_forest": _space_random_forest,
    "mlp": _space_mlp,
    "lstm": _space_sequence,
    "gru": _space_sequence,
}


def tuning_path(instrument: str, sentiment: str, horizon: int, model: str) -> Path:
    return TUNING_DIR / f"{instrument}_{sentiment}_{horizon_label(horizon)}_{model}.json"


def load_tuned_params(instrument: str, sentiment: str, horizon: int, model: str) -> dict | None:
    path = tuning_path(instrument, sentiment, horizon, model)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["best_params"]


def tune(model: str, X, y, splitter, *, trials: int, seed: int = 42,
         objective_kind: str = "macro_f1") -> optuna.Study:
    """Optuna study. objective_kind:
      macro_f1 — maximise CV macro-F1 (accuracy only; the historical default)
      logloss  — minimise class-weighted log-loss (probability quality only)
      blend    — maximise macro_f1 - 0.5*logloss (single scalar trade-off)
      multi    — maximise macro_f1 AND minimise logloss (Pareto; knee picked later)
    """
    space = SPACES[model]
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def evaluate(trial):
        result = run_cv(MODELS[model], X, y, splitter, space(trial))
        trial.set_user_attr("std_macro_f1", result["std_macro_f1"])
        trial.set_user_attr("mean_macro_f1", result["mean_macro_f1"])
        trial.set_user_attr("mean_log_loss", result["mean_log_loss"])
        return result

    sampler = optuna.samplers.TPESampler(seed=seed)
    if objective_kind == "multi":
        study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)
        study.optimize(lambda t: (lambda r: (r["mean_macro_f1"], r["mean_log_loss"]))(
            evaluate(t)), n_trials=trials, show_progress_bar=False)
        return study

    def scalar(trial):
        r = evaluate(trial)
        if objective_kind == "logloss":
            return r["mean_log_loss"]
        if objective_kind == "blend":
            return r["mean_macro_f1"] - 0.5 * r["mean_log_loss"]
        return r["mean_macro_f1"]

    direction = "minimize" if objective_kind == "logloss" else "maximize"
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.optimize(scalar, n_trials=trials, show_progress_bar=False)
    return study


def pick_knee(study: optuna.Study) -> optuna.trial.FrozenTrial:
    """From a 2-objective Pareto front pick the knee: the trial closest to the
    utopia point (best macro-F1, best log-loss) in min-max-normalised space."""
    front = study.best_trials
    f1 = np.array([t.values[0] for t in front])
    ll = np.array([t.values[1] for t in front])
    f1n = (f1 - f1.min()) / (np.ptp(f1) or 1.0)        # higher is better -> 1 ideal
    lln = (ll - ll.min()) / (np.ptp(ll) or 1.0)        # lower is better  -> 0 ideal
    dist = np.hypot(1.0 - f1n, lln)
    return front[int(dist.argmin())]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    ap.add_argument("--horizon", type=parse_horizon, default=DEFAULT_LABEL,
                    help="m15 (default) / h1 / h4, or a bar count")
    ap.add_argument("--model", required=True, choices=sorted(SPACES))
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--objective", default="macro_f1",
                    choices=["macro_f1", "logloss", "blend", "multi"],
                    help="tuning target; non-default writes a suffixed cache so "
                         "the macro_f1 baseline is preserved for A/B")
    args = ap.parse_args()

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    sp = cfg["split"]
    splitter = BlockingTimeSeriesSplit(
        sp["n_splits"], sp["train_frac"], max(sp["embargo_bars"], args.horizon)
    )
    X, y = load_xy(args.instrument, args.sentiment, args.horizon, cfg)
    print(f"[tune {args.model}] {args.instrument} sentiment={args.sentiment} "
          f"horizon={horizon_label(args.horizon)} | {len(y):,} rows | {args.trials} trials")

    started = time.time()
    study = tune(args.model, X, y, splitter, trials=args.trials,
                 objective_kind=args.objective)

    if args.objective == "multi":
        chosen = pick_knee(study)
        best_macro_f1 = float(chosen.values[0])
        best_log_loss = float(chosen.values[1])
        best_params = dict(chosen.params)
        print(f"Pareto front: {len(study.best_trials)} trials; "
              f"knee = macro-F1 {best_macro_f1:.4f}, log-loss {best_log_loss:.4f}")
    else:
        chosen = study.best_trial
        best_params = dict(study.best_params)
        best_macro_f1 = float(chosen.user_attrs.get("mean_macro_f1", float("nan")))
        best_log_loss = float(chosen.user_attrs.get("mean_log_loss", float("nan")))

    # convert hidden width/layers back into the constructor's tuple form
    if args.model == "mlp":
        best_params["hidden"] = [best_params.pop("width")] * best_params.pop("layers")

    payload = {
        "model": args.model,
        "instrument": args.instrument,
        "sentiment": args.sentiment,
        "horizon": args.horizon,
        "trials": args.trials,
        "objective": args.objective,
        "elapsed_s": round(time.time() - started, 1),
        "best_macro_f1": best_macro_f1,
        "best_log_loss": best_log_loss,
        "best_params": best_params,
    }
    path = tuning_path(args.instrument, args.sentiment, args.horizon, args.model)
    if args.objective != "macro_f1":   # keep the baseline cache untouched for A/B
        path = path.with_name(path.stem + f"__{args.objective}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"macro-F1 {best_macro_f1:.4f} | log-loss {best_log_loss:.4f} "
          f"in {payload['elapsed_s']}s")
    print(f"best params  {best_params}")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
