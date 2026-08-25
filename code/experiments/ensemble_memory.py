"""RQ3 ensemble-memory experiment: glm reweights a soft-voting ensemble.

The reflection agent's force_flat lessons proved redundant with a confidence
gate (notebook 05). This experiment gives the agent the one lever a per-model
gate cannot replicate: conditional PER-MODEL weights over a soft-voting
ensemble of the tuned contenders, plus conditional gate tightening.

Actions on the ensemble:
  * downweight/upweight <model>  - multiply that model's vote by `factor`
    wherever the condition holds (weights renormalise inside the average);
  * force_flat                   - override the ensemble to flat.

`widen_deadzone` is still supported by apply_ensemble_lessons for older artifacts,
but the GLM reflection run excludes it by default because a high global tau already
acts as a deadzone.

Subcommands:
  build    merge the contenders' walk-forward caches into one ensemble cache
  reflect  run the weekly glm reflection loop against the ensemble
  oos      apply accepted lessons to later windows only, gated, with DM test;
           --max-harmful-windows K retires lessons whose leave-one-out marginal
           hurts the retention metric K windows in a row (forgetting)

Run:  python -m experiments.ensemble_memory build
      python -m experiments.ensemble_memory reflect --model glm-5.2:cloud
      python -m experiments.ensemble_memory oos --tau auto
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from evaluation.economics import strategy_returns
from evaluation.metrics import classification_scores
from experiments.spans import CACHE_SUFFIX
from memory.gates import condition_mask
from memory.schema import Lesson

CODE_ROOT = Path(__file__).resolve().parents[1]
WALKFORWARD_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
REFLECTION_DIR = CODE_ROOT / "experiments" / "cache" / "reflection"

BASE_MODELS = ("catboost_balanced", "gru", "xgboost_balanced")
ENSEMBLE_TAG = "ensemble3"
BOOKKEEPING = ("window", "train_start", "train_end", "validation_start",
               "validation_end", "y_true", "forward_return")


def ensemble_cache_path(instrument: str, sentiment: str, label: str) -> Path:
    return WALKFORWARD_DIR / f"{instrument}_{sentiment}_{ENSEMBLE_TAG}_{label}_{CACHE_SUFFIX}.parquet"


def prob_cols(model: str) -> list[str]:
    return [f"{model}_p{i}" for i in range(3)]


def build_ensemble_cache(instrument: str, sentiment: str, label: str) -> pd.DataFrame:
    """Merge the base models' walk-forward caches into one frame.

    Features/bookkeeping come from the first cache (identical across caches by
    construction); each base contributes its three class probabilities.
    """
    frames = {}
    for model in BASE_MODELS:
        path = WALKFORWARD_DIR / f"{instrument}_{sentiment}_{model}_{label}_{CACHE_SUFFIX}.parquet"
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        frames[model] = df.sort_index()

    first = frames[BASE_MODELS[0]]
    feature_cols = [c for c in first.columns
                    if c not in BOOKKEEPING and not c.startswith(f"{BASE_MODELS[0]}_")]
    merged = first[list(BOOKKEEPING) + feature_cols].copy()
    for model, df in frames.items():
        if not df.index.equals(first.index):
            raise ValueError(f"{model} cache index differs from {BASE_MODELS[0]}")
        merged[prob_cols(model)] = df[prob_cols(model)]

    probs = ensemble_probabilities(merged)
    merged[f"{ENSEMBLE_TAG}_pred"] = probs.to_numpy().argmax(axis=1)
    merged[f"{ENSEMBLE_TAG}_conf"] = probs.max(axis=1)
    for i in range(3):
        merged[f"{ENSEMBLE_TAG}_p{i}"] = probs.iloc[:, i]
    out = ensemble_cache_path(instrument, sentiment, label)
    merged.to_parquet(out)
    print(f"wrote {len(merged):,} rows x {merged.shape[1]} cols -> {out}")
    return merged


def ensemble_probabilities(frame: pd.DataFrame,
                           weights: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """Weighted soft-vote probabilities over the base models.

    `weights` maps model -> per-bar weight Series (default 1.0 everywhere).
    Weights renormalise per bar, so only RELATIVE weights matter.
    """
    total = None
    weight_sum = None
    for model in BASE_MODELS:
        w = weights.get(model) if weights else None
        w = pd.Series(1.0, index=frame.index) if w is None else w.astype(float)
        block = frame[prob_cols(model)].to_numpy(dtype=float) * w.to_numpy()[:, None]
        total = block if total is None else total + block
        weight_sum = w if weight_sum is None else weight_sum + w
    total = total / np.maximum(weight_sum.to_numpy()[:, None], 1e-12)
    return pd.DataFrame(total, index=frame.index, columns=[f"p{i}" for i in range(3)])


def apply_ensemble_lessons(
    frame: pd.DataFrame,
    lessons: list[Lesson],
    feature_columns: tuple[str, ...],
    *,
    tau: float = 0.0,
) -> pd.Series:
    """Ensemble prediction after weight lessons, force_flat, and deadzone lessons."""
    features = frame[list(feature_columns)]
    weights = {m: pd.Series(1.0, index=frame.index) for m in BASE_MODELS}
    force_flat = pd.Series(False, index=frame.index)
    required_conf = pd.Series(float(tau), index=frame.index)

    for lesson in lessons:
        mask = condition_mask(features, lesson.condition).reindex(frame.index, fill_value=False)
        if lesson.action in {"downweight", "upweight"} and lesson.target in weights:
            weights[lesson.target] = weights[lesson.target].where(~mask,
                                                                  weights[lesson.target] * lesson.factor)
        elif lesson.action == "force_flat":
            force_flat = force_flat | mask
        elif lesson.action == "widen_deadzone":
            tightened = np.maximum(required_conf[mask], float(tau) * float(lesson.factor))
            required_conf = required_conf.where(~mask, tightened)

    probs = ensemble_probabilities(frame, weights)
    pred = pd.Series(probs.to_numpy().argmax(axis=1), index=frame.index)
    conf = probs.max(axis=1)
    if (required_conf > 0.0).any():          # tau gate, tightened where lessons say so
        pred = pred.where(conf >= required_conf, 1)
    pred.loc[force_flat] = 1
    return pred.astype(int)


def ensemble_acceptance_score(
    frame: pd.DataFrame,
    lessons: list[Lesson],
    feature_columns: tuple[str, ...],
    *,
    tau: float,
    metric: str,
    fee_bps: float,
) -> float:
    """Score one candidate set for the weekly keep-if-better gate."""
    pred = apply_ensemble_lessons(frame, lessons, feature_columns, tau=tau)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        warnings.filterwarnings("ignore", message="A single label was found")
        macro_f1 = float(classification_scores(frame["y_true"].astype(int), pred)["macro_f1"])
    if metric == "macro_f1":
        return macro_f1
    if "forward_return" not in frame.columns:
        raise ValueError("net_return acceptance requires forward_return")
    net_return = float(strategy_returns(pred, frame["forward_return"], fee_bps).sum())
    if metric == "net_return":
        return net_return
    if metric == "hybrid":
        return net_return + 1e-6 * macro_f1
    raise ValueError("acceptance metric must be macro_f1, net_return, or hybrid")


def run_reflect(args) -> int:
    from experiments.run_reflection import (lessons_to_frame, load_predictions,
                                            windows_from_predictions)
    from experiments.walkforward import make_report_fn
    from memory.loop import WeeklyReflectionLoop
    from memory.proposer import OllamaLessonProposer

    cache = ensemble_cache_path(args.instrument, args.sentiment, args.label)
    predictions = load_predictions(cache)
    feature_columns = tuple(
        c for c in predictions.columns
        if c not in BOOKKEEPING
        and not any(c.startswith(f"{m}_p") for m in BASE_MODELS)
        and not c.startswith(f"{ENSEMBLE_TAG}_p")
        and c != f"{ENSEMBLE_TAG}_pred"
    )
    windows = windows_from_predictions(predictions)
    if args.limit_windows:
        windows = windows[: args.limit_windows]

    # per-model error views in the report so glm can see WHICH model fails where
    pred_map = {}
    for model in BASE_MODELS:
        col = f"__{model}_pred"
        predictions[col] = predictions[prob_cols(model)].to_numpy().argmax(axis=1)
        pred_map[model] = col
    pred_map[ENSEMBLE_TAG] = f"{ENSEMBLE_TAG}_pred"

    report_fn = make_report_fn(
        predictions,
        condition_cols=tuple(c for c in ("vol_regime", "hour", "dayofweek")
                             if c in predictions.columns),
        model_pred_cols=pred_map,
        focus_model=ENSEMBLE_TAG,
        confidence_col=f"{ENSEMBLE_TAG}_conf",
    )

    # Discovery uses a lower gate than the final deployed policy so lessons can
    # affect enough validation bars to be judged. The final OOS command can still
    # evaluate those lessons at --tau auto.
    from experiments.run_memory_oos import default_fee_bps, resolve_tau

    tau_arg = args.discovery_tau if args.discovery_tau is not None else args.tau
    tau = resolve_tau(tau_arg, args.instrument, args.sentiment, 1, ENSEMBLE_TAG)
    fee_bps = default_fee_bps(args.instrument) if args.fee_bps is None else float(args.fee_bps)
    print(
        f"acceptance scoring uses {args.acceptance_metric} "
        f"on gated predictions at discovery tau={tau:g}"
    )

    def score_fn(lessons: list[Lesson], window) -> float:
        frame = predictions.sort_index().loc[window.validation_start:window.validation_end]
        return ensemble_acceptance_score(
            frame,
            lessons,
            feature_columns,
            tau=tau,
            metric=args.acceptance_metric,
            fee_bps=fee_bps,
        )

    allowed_actions = tuple(a.strip() for a in args.allowed_actions.split(",") if a.strip())
    proposer = OllamaLessonProposer(
        feature_columns=feature_columns,
        model_names=BASE_MODELS,
        model=args.model,
        allowed_actions=allowed_actions,
        think=args.think,
    )
    loop = WeeklyReflectionLoop(
        proposer=proposer,
        report_fn=report_fn,
        score_fn=score_fn,
        feature_columns=feature_columns,
        model_names=BASE_MODELS,
        epsilon=args.epsilon,
    )
    results = [loop.run_window(w) for w in windows]
    frame = lessons_to_frame(results)
    out = lessons_path(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    accepted = int((frame["status"] == "accepted").sum()) if "status" in frame else 0
    rejected = int((frame["status"] == "rejected").sum()) if "status" in frame else 0
    by_action = frame.loc[frame["status"] == "accepted", "action"].value_counts().to_dict() \
        if "action" in frame else {}
    print(f"wrote {len(frame):,} lesson decisions ({accepted} accepted, {rejected} rejected) -> {out}")
    print(f"accepted by action: {by_action}")
    print(f"allowed actions: {list(allowed_actions)}")
    return 0


def lessons_path(args) -> Path:
    safe = args.model.replace(":", "_").replace("/", "_")
    return REFLECTION_DIR / (f"{args.instrument}_{args.sentiment}_{args.label}_"
                             f"{ENSEMBLE_TAG}_{safe}_lessons_2025.parquet")


def _ensemble_retention_score(frame: pd.DataFrame, pred: pd.Series, *,
                              metric: str, fee_bps: float) -> float:
    if metric == "macro_f1":
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            warnings.filterwarnings("ignore", message="A single label was found")
            return float(classification_scores(frame["y_true"].astype(int), pred)["macro_f1"])
    return float(strategy_returns(pred, frame["forward_return"], fee_bps).sum())


def _lesson_key(lesson: Lesson) -> str:
    return f"{lesson.source_window}|{lesson.condition}|{lesson.action}|{lesson.target}|{lesson.factor}"


def apply_ensemble_lessons_oos(
    predictions: pd.DataFrame,
    accepted: list[Lesson],
    feature_columns: tuple[str, ...],
    *,
    tau: float,
    fee_bps: float,
    max_harmful_windows: int | None = None,
    retention_metric: str = "net_return",
) -> pd.DataFrame:
    """Lessons from window N gate windows N+1 onward, with optional forgetting.

    Forgetting mirrors the single-model path: a lesson is retired after K
    CONSECUTIVE windows in which its leave-one-out marginal hurts the retention
    metric (the counter resets on any non-harmful window).
    """
    from experiments.run_reflection import windows_from_predictions

    windows = windows_from_predictions(predictions)
    order = {w.name: i for i, w in enumerate(windows)}
    accepted = [l for l in accepted
                if l.status == "accepted" and l.source_window in order]

    retired: set[str] = set()
    harmful_counts: dict[str, int] = {}
    frames = []
    for window in windows:
        frame = predictions.loc[predictions["window"] == window.name]
        if frame.empty:
            continue
        active = [l for l in accepted
                  if order[str(l.source_window)] < order[window.name]
                  and _lesson_key(l) not in retired]
        baseline = frame[f"{ENSEMBLE_TAG}_pred"].astype(int)
        conf = frame[f"{ENSEMBLE_TAG}_conf"]
        base_gated = baseline.where(conf >= tau, 1) if tau > 0 else baseline
        memory = apply_ensemble_lessons(frame, active, feature_columns, tau=tau)
        frames.append(pd.DataFrame({
            "window": frame["window"],
            "y_true": frame["y_true"].astype(int),
            "baseline_pred": base_gated,
            "memory_pred": memory,
            "forward_return": frame["forward_return"],
            "active_lesson_count": len(active),
            "retired_lesson_count": len(retired),
        }, index=frame.index))

        if max_harmful_windows is not None and active:
            full = _ensemble_retention_score(frame, memory,
                                             metric=retention_metric, fee_bps=fee_bps)
            for lesson in active:
                key = _lesson_key(lesson)
                without = [l for l in active if _lesson_key(l) != key]
                without_pred = apply_ensemble_lessons(frame, without, feature_columns, tau=tau)
                without_score = _ensemble_retention_score(
                    frame, without_pred, metric=retention_metric, fee_bps=fee_bps)
                harmful_counts[key] = harmful_counts.get(key, 0) + 1 if full < without_score else 0
                if harmful_counts[key] >= max_harmful_windows:
                    retired.add(key)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def run_oos(args) -> int:
    from evaluation.economics import diebold_mariano, strategy_returns
    from experiments.run_memory_oos import default_fee_bps, resolve_tau
    from experiments.run_reflection import lessons_from_frame, load_predictions

    cache = ensemble_cache_path(args.instrument, args.sentiment, args.label)
    predictions = load_predictions(cache)
    lesson_frame = pd.read_parquet(lessons_path(args))
    fee_bps = default_fee_bps(args.instrument)
    tau = resolve_tau(args.tau, args.instrument, args.sentiment, 1, ENSEMBLE_TAG)
    forget_k = args.max_harmful_windows

    feature_columns = tuple(
        c for c in predictions.columns
        if c not in BOOKKEEPING
        and not any(c.startswith(f"{m}_p") for m in BASE_MODELS)
        and not c.startswith(f"{ENSEMBLE_TAG}_p")
        and c != f"{ENSEMBLE_TAG}_pred"
    )
    gated = apply_ensemble_lessons_oos(
        predictions,
        lessons_from_frame(lesson_frame),
        feature_columns,
        tau=tau,
        fee_bps=fee_bps,
        max_harmful_windows=forget_k,
        retention_metric=args.retention_metric,
    )

    suffix = f"_oos_tau{int(round(tau * 100)):02d}"
    if forget_k is not None:
        metric_tag = "netret" if args.retention_metric == "net_return" else "macrof1"
        suffix += f"_forget{forget_k}_{metric_tag}"
    out = lessons_path(args).with_name(
        lessons_path(args).stem.replace("_lessons_", f"{suffix}_") + ".parquet")
    gated.to_parquet(out)

    summary = {}
    for variant in ("baseline_pred", "memory_pred"):
        scores = classification_scores(gated["y_true"], gated[variant])
        returns = strategy_returns(gated[variant], gated["forward_return"], fee_bps)
        summary[variant] = {"macro_f1": scores["macro_f1"],
                            "net_return_sum": float(returns.sum())}
    r_base = strategy_returns(gated["baseline_pred"], gated["forward_return"], fee_bps)
    r_mem = strategy_returns(gated["memory_pred"], gated["forward_return"], fee_bps)
    dm = diebold_mariano(r_base, r_mem)

    print(f"wrote {len(gated):,} OOS rows -> {out}")
    print(f"tau={tau:g} fee_bps={fee_bps:g} active lessons (final window): "
          f"{int(gated['active_lesson_count'].iloc[-1])}")
    if forget_k is not None:
        print(f"forgetting: max_harmful_windows={forget_k} ({args.retention_metric}); "
              f"retired by year-end: {int(gated['retired_lesson_count'].iloc[-1])}")
    b, m = summary["baseline_pred"], summary["memory_pred"]
    print(f"macro_f1 gate-alone={b['macro_f1']:.6f} gate+memory={m['macro_f1']:.6f} "
          f"delta={m['macro_f1'] - b['macro_f1']:+.6f}")
    print(f"net_return gate-alone={b['net_return_sum']:+.6f} gate+memory={m['net_return_sum']:+.6f} "
          f"delta={m['net_return_sum'] - b['net_return_sum']:+.6f}")
    print(f"DM gate-alone vs gate+memory: stat={dm['dm_stat']:+.3f} p={dm['p_value']:.4f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build", "reflect", "oos"])
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    ap.add_argument("--label", default="m15")
    ap.add_argument("--model", default="glm-5.2:cloud")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--limit-windows", type=int, default=None)
    ap.add_argument("--tau", default="auto")
    ap.add_argument("--discovery-tau", default="0.60",
                    help="gate used during reflection acceptance; final OOS can still use --tau auto")
    ap.add_argument("--acceptance-metric", choices=["macro_f1", "net_return", "hybrid"],
                    default="net_return")
    ap.add_argument("--allowed-actions", default="downweight,upweight,force_flat")
    ap.add_argument("--max-harmful-windows", type=int, default=None,
                    help="oos: retire a lesson after K consecutive windows where its "
                         "leave-one-out marginal hurts the retention metric")
    ap.add_argument("--retention-metric", choices=["macro_f1", "net_return"],
                    default="net_return")
    ap.add_argument("--think", default="high",
                    help="Ollama thinking effort for supported models; use none to omit")
    ap.add_argument("--fee-bps", type=float, default=None)
    args = ap.parse_args()
    if str(args.think).lower() in {"", "none", "false"}:
        args.think = None

    if args.command == "build":
        build_ensemble_cache(args.instrument, args.sentiment, args.label)
        return 0
    if args.command == "reflect":
        return run_reflect(args)
    return run_oos(args)


if __name__ == "__main__":
    raise SystemExit(main())
