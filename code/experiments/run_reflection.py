"""Run the RQ3 weekly reflection loop over a walk-forward prediction cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from evaluation.metrics import classification_scores
from experiments.walkforward import _apply_lessons, make_report_fn, make_score_fn
from memory.loop import ReflectionWindow, WeeklyReflectionLoop, WindowResult
from memory.proposer import MODEL, OllamaLessonProposer
from memory.schema import Lesson

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = CODE_ROOT / "experiments" / "cache" / "walkforward"
DEFAULT_OUT_DIR = CODE_ROOT / "experiments" / "cache" / "reflection"
DEFAULT_FEATURE_COLUMNS = ("vol_regime", "hour", "dayofweek")
DEFAULT_CONDITION_COLUMNS = ("vol_regime", "hour", "dayofweek")
MODEL_NAMES = ("catboost",)
DEFAULT_WF_MODEL = "catboost"          # legacy untuned cache the RQ3 artifacts use


def wf_cols(wf_model: str) -> dict[str, str]:
    """Column names a walk-forward cache uses for one model tag."""
    return {"pred": f"{wf_model}_pred", "conf": f"{wf_model}_conf",
            "probs": [f"{wf_model}_p{i}" for i in range(3)]}


def wf_feature_columns(predictions: pd.DataFrame, wf_model: str) -> tuple[str, ...]:
    """Condition/feature columns = everything except bookkeeping and the model's
    class outputs. The model's CONFIDENCE stays available on purpose, so lessons
    can reference it (it is known at prediction time)."""
    cols = wf_cols(wf_model)
    excluded = {"window", "train_start", "train_end", "validation_start",
                "validation_end", "y_true", "forward_return",
                cols["pred"], *cols["probs"]}
    return tuple(c for c in predictions.columns if c not in excluded)


def load_predictions(path: str | Path) -> pd.DataFrame:
    """Load a walk-forward cache with a UTC DatetimeIndex."""
    out = pd.read_parquet(path)
    out.index = pd.to_datetime(out.index, utc=True)
    return out.sort_index()


def windows_from_predictions(predictions: pd.DataFrame) -> list[ReflectionWindow]:
    """Reconstruct weekly windows from cached prediction rows."""
    required = {"window", "train_start", "train_end", "validation_start", "validation_end"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"prediction cache missing window column(s): {sorted(missing)}")

    windows: list[ReflectionWindow] = []
    for name, group in predictions.groupby("window", sort=True):
        first = group.iloc[0]
        windows.append(
            ReflectionWindow(
                name=str(name),
                train_start=pd.Timestamp(first["train_start"]),
                train_end=pd.Timestamp(first["train_end"]),
                validation_start=pd.Timestamp(first["validation_start"]),
                validation_end=pd.Timestamp(first["validation_end"]),
            )
        )
    return windows


def lessons_to_frame(results: list[WindowResult]) -> pd.DataFrame:
    """Convert lesson decisions to a Parquet-ready DataFrame."""
    rows = []
    for result in results:
        for lesson in [*result.accepted, *result.rejected]:
            row = lesson.to_dict()
            row["metadata"] = json.dumps(row.get("metadata", {}), sort_keys=True, default=str)
            rows.append(row)
    return pd.DataFrame(rows)


def _present(columns: tuple[str, ...] | list[str], predictions: pd.DataFrame) -> tuple[str, ...]:
    return tuple(col for col in columns if col in predictions.columns)


def lessons_from_frame(frame: pd.DataFrame) -> list[Lesson]:
    """Rebuild lessons from a persisted decision table."""
    lessons: list[Lesson] = []
    if frame.empty:
        return lessons
    for _, row in frame.iterrows():
        data = row.to_dict()
        metadata = data.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata) if metadata else {}
        data["metadata"] = metadata
        lessons.append(Lesson.from_dict(data))
    return lessons


def apply_lessons_out_of_sample(
    predictions: pd.DataFrame,
    lesson_frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...] | list[str],
    pred_col: str = "catboost_pred",
    conf_col: str | None = "catboost_conf",
    y_true_col: str = "y_true",
    max_harmful_windows: int | None = None,
    retention_epsilon: float = 0.0,
    retention_metric: str = "macro_f1",
    fee_bps: float = 0.0,
) -> pd.DataFrame:
    """Apply accepted lessons only to windows after their source window."""
    if max_harmful_windows is not None and max_harmful_windows < 1:
        raise ValueError("max_harmful_windows must be at least 1")
    if retention_metric not in {"macro_f1", "net_return"}:
        raise ValueError("retention_metric must be 'macro_f1' or 'net_return'")

    predictions = predictions.sort_index()
    windows = windows_from_predictions(predictions)
    window_order = {window.name: i for i, window in enumerate(windows)}
    accepted = [
        lesson for lesson in lessons_from_frame(lesson_frame)
        if lesson.status == "accepted" and lesson.source_window in window_order
    ]
    retired: set[str] = set()
    harmful_counts: dict[str, int] = {}

    frames: list[pd.DataFrame] = []
    for window in windows:
        order = window_order[window.name]
        active = [
            lesson for lesson in accepted
            if window_order[str(lesson.source_window)] < order and _lesson_key(lesson) not in retired
        ]
        frame = predictions.loc[predictions["window"] == window.name].copy()
        if frame.empty:
            continue
        baseline = frame[pred_col].astype(int)
        memory = _apply_lessons(
            frame,
            active,
            pred_col=pred_col,
            feature_columns=feature_columns,
        )
        out = pd.DataFrame(
            {
                "window": frame["window"],
                y_true_col: frame[y_true_col].astype(int),
                "baseline_pred": baseline,
                "memory_pred": memory.astype(int),
                "active_lesson_count": len(active),
                "retired_lesson_count": len(retired),
                "changed_by_memory": baseline != memory,
            },
            index=frame.index,
        )
        if "forward_return" in frame.columns:
            out["forward_return"] = frame["forward_return"]
        if conf_col is not None and conf_col in frame.columns:
            out["conf"] = frame[conf_col].astype(float)
        frames.append(out)

        if max_harmful_windows is not None and active:
            full_score = _retention_score(
                frame,
                memory,
                metric=retention_metric,
                y_true_col=y_true_col,
                fee_bps=fee_bps,
            )
            for lesson in active:
                key = _lesson_key(lesson)
                without = [item for item in active if _lesson_key(item) != key]
                without_pred = _apply_lessons(
                    frame,
                    without,
                    pred_col=pred_col,
                    feature_columns=feature_columns,
                )
                without_score = _retention_score(
                    frame,
                    without_pred,
                    metric=retention_metric,
                    y_true_col=y_true_col,
                    fee_bps=fee_bps,
                )
                if full_score < without_score - retention_epsilon:
                    harmful_counts[key] = harmful_counts.get(key, 0) + 1
                else:
                    harmful_counts[key] = 0
                if harmful_counts[key] >= max_harmful_windows:
                    retired.add(key)
    return pd.concat(frames).sort_index() if frames else pd.DataFrame()


def _lesson_key(lesson: Lesson) -> str:
    return json.dumps(
        {
            "source_window": lesson.source_window,
            "condition": lesson.condition,
            "action": lesson.action,
            "target": lesson.target,
            "factor": lesson.factor,
        },
        sort_keys=True,
    )


def _retention_score(
    frame: pd.DataFrame,
    pred: pd.Series,
    *,
    metric: str,
    y_true_col: str,
    fee_bps: float,
) -> float:
    if metric == "macro_f1":
        return float(classification_scores(frame[y_true_col].astype(int), pred.astype(int))["macro_f1"])
    if "forward_return" not in frame.columns:
        raise ValueError("net_return retention requires forward_return")
    return float(_strategy_returns(pred, frame["forward_return"], fee_bps).sum())


def _strategy_returns(pred: pd.Series, forward_return: pd.Series, fee_bps: float) -> pd.Series:
    positions = pred.astype(int).map({0: -1.0, 1: 0.0, 2: 1.0}).fillna(0.0)
    returns = forward_return.astype(float).fillna(0.0)
    previous = positions.shift(1, fill_value=0.0)
    turnover = (positions - previous).abs()
    return positions * returns - turnover * (float(fee_bps) / 10_000.0)


def _annualized_sharpe(returns: pd.Series, bars_per_year: int) -> float:
    clean = returns.dropna().astype(float)
    if len(clean) < 2:
        return 0.0
    std = clean.std(ddof=1)
    if not np.isfinite(std) or std == 0:
        return 0.0
    return float(clean.mean() / std * np.sqrt(bars_per_year))


def _baseline_return_split(gated: pd.DataFrame) -> dict[str, float | int]:
    if "forward_return" not in gated.columns:
        return {}
    baseline_pos = gated["baseline_pred"].astype(int).map({0: -1.0, 1: 0.0, 2: 1.0}).fillna(0.0)
    baseline_bar_return_bps = baseline_pos * gated["forward_return"].astype(float) * 10_000.0
    flattened = gated["changed_by_memory"].astype(bool) & (baseline_pos != 0.0)
    kept = (~gated["changed_by_memory"].astype(bool)) & (baseline_pos != 0.0)
    return {
        "flattened_bar_count": int(flattened.sum()),
        "kept_directional_bar_count": int(kept.sum()),
        "flattened_baseline_return_mean_bps": (
            float(baseline_bar_return_bps.loc[flattened].mean()) if flattened.any() else np.nan
        ),
        "kept_directional_baseline_return_mean_bps": (
            float(baseline_bar_return_bps.loc[kept].mean()) if kept.any() else np.nan
        ),
    }


def _variant_metrics(
    frame: pd.DataFrame,
    *,
    variant: str,
    pred_col: str,
    y_true_col: str,
    fee_bps: float,
    bars_per_year: int,
    tau: float = 0.0,
) -> dict[str, float | int | str]:
    scores = classification_scores(frame[y_true_col].astype(int), frame[pred_col].astype(int))
    out: dict[str, float | int | str] = {
        "variant": variant,
        "rows": int(len(frame)),
        "macro_f1": float(scores["macro_f1"]),
        "balanced_accuracy": float(scores["balanced_accuracy"]),
        "f1_down": float(scores["per_class_f1"]["down"]),
        "f1_flat": float(scores["per_class_f1"]["flat"]),
        "f1_up": float(scores["per_class_f1"]["up"]),
        "changed_rows": int(frame["changed_by_memory"].sum()) if variant == "memory" else 0,
    }
    if "forward_return" in frame.columns:
        from evaluation.economics import positions_from_predictions, strategy_returns

        conf = frame["conf"] if (tau > 0.0 and "conf" in frame.columns) else None
        returns = strategy_returns(frame[pred_col], frame["forward_return"], fee_bps,
                                   conf, tau)
        positions = positions_from_predictions(frame[pred_col], conf, tau)
        turnover = (positions - positions.shift(1, fill_value=0.0)).abs()
        out.update({
            "net_return_sum": float(returns.sum()),
            "net_return_mean": float(returns.mean()),
            "sharpe": _annualized_sharpe(returns, bars_per_year),
            "trade_count": int((turnover > 0).sum()),
            "turnover_sides": float(turnover.sum()),
            "exposure": float((positions != 0).mean()),
        })
    return out


def summarize_oos_memory(
    gated: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    fee_bps: float = 0.0,
    bars_per_year: int = 35_040,
    tau: float = 0.0,
) -> pd.DataFrame:
    """Summarize baseline vs memory-gated out-of-sample predictions."""
    split = _baseline_return_split(gated)
    rows = [
        _variant_metrics(
            gated,
            variant="baseline",
            pred_col="baseline_pred",
            y_true_col=y_true_col,
            fee_bps=fee_bps,
            bars_per_year=bars_per_year,
            tau=tau,
        ),
        _variant_metrics(
            gated,
            variant="memory",
            pred_col="memory_pred",
            y_true_col=y_true_col,
            fee_bps=fee_bps,
            bars_per_year=bars_per_year,
            tau=tau,
        ),
    ]
    for row in rows:
        row.update(split)
    return pd.DataFrame(rows)


def summarize_oos_windows(
    gated: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
    fee_bps: float = 0.0,
    bars_per_year: int = 35_040,
    tau: float = 0.0,
) -> pd.DataFrame:
    """Return paired baseline/memory metrics for each validation window."""
    rows = []
    for window, frame in gated.groupby("window", sort=True):
        baseline = _variant_metrics(
            frame,
            variant="baseline",
            pred_col="baseline_pred",
            y_true_col=y_true_col,
            fee_bps=fee_bps,
            bars_per_year=bars_per_year,
            tau=tau,
        )
        memory = _variant_metrics(
            frame,
            variant="memory",
            pred_col="memory_pred",
            y_true_col=y_true_col,
            fee_bps=fee_bps,
            bars_per_year=bars_per_year,
            tau=tau,
        )
        rows.append({
            "window": window,
            "rows": int(len(frame)),
            "baseline_macro_f1": baseline["macro_f1"],
            "memory_macro_f1": memory["macro_f1"],
            "delta_macro_f1": float(memory["macro_f1"]) - float(baseline["macro_f1"]),
            "baseline_net_return_sum": baseline.get("net_return_sum", np.nan),
            "memory_net_return_sum": memory.get("net_return_sum", np.nan),
            "delta_net_return_sum": float(memory.get("net_return_sum", np.nan))
            - float(baseline.get("net_return_sum", np.nan)),
            "changed_rows": int(frame["changed_by_memory"].sum()),
            "active_lesson_count": int(frame["active_lesson_count"].max()),
            "retired_lesson_count": int(frame.get("retired_lesson_count", pd.Series([0])).max()),
        })
    return pd.DataFrame(rows)


def paired_oos_significance(
    window_metrics: pd.DataFrame,
    *,
    delta_col: str = "delta_macro_f1",
) -> dict[str, float | int | str]:
    """Wilcoxon signed-rank test over paired weekly OOS deltas."""
    deltas = window_metrics[delta_col].dropna().astype(float)
    nonzero = deltas[deltas != 0.0]
    result: dict[str, float | int] = {
        "n_windows": int(len(deltas)),
        "mean_delta": float(deltas.mean()) if len(deltas) else np.nan,
        "median_delta": float(deltas.median()) if len(deltas) else np.nan,
        "wilcoxon_statistic": np.nan,
        "wilcoxon_pvalue": np.nan,
    }
    if len(nonzero) == 0:
        return result
    test = wilcoxon(deltas, zero_method="wilcox", alternative="two-sided", method="auto")
    result["wilcoxon_statistic"] = float(test.statistic)
    result["wilcoxon_pvalue"] = float(test.pvalue)
    return result


def paired_oos_significance_frame(window_metrics: pd.DataFrame) -> pd.DataFrame:
    """Return paired significance rows for classification and economic deltas."""
    rows = []
    for metric, delta_col in (
        ("macro_f1", "delta_macro_f1"),
        ("net_return_sum", "delta_net_return_sum"),
    ):
        result = paired_oos_significance(window_metrics, delta_col=delta_col)
        result["metric"] = metric
        rows.append(result)
    cols = ["metric", "n_windows", "mean_delta", "median_delta", "wilcoxon_statistic", "wilcoxon_pvalue"]
    return pd.DataFrame(rows)[cols]


def run_reflection_from_cache(
    predictions: pd.DataFrame,
    *,
    proposer,
    feature_columns: tuple[str, ...] | list[str],
    condition_cols: tuple[str, ...] | list[str] = DEFAULT_CONDITION_COLUMNS,
    news_cols: tuple[str, ...] | list[str] = (),
    out_path: str | Path | None = None,
    epsilon: float = 0.0,
    limit_windows: int | None = None,
    wf_model: str = DEFAULT_WF_MODEL,
) -> list[WindowResult]:
    """Run the reflection loop and optionally persist lesson decisions."""
    predictions = predictions.sort_index()
    windows = windows_from_predictions(predictions)
    if limit_windows is not None:
        windows = windows[:limit_windows]

    condition_cols = _present(tuple(condition_cols), predictions)
    news_cols = _present(tuple(news_cols), predictions)
    cols = wf_cols(wf_model)
    report_fn = make_report_fn(
        predictions,
        condition_cols=condition_cols,
        news_cols=news_cols,
        model_pred_cols={wf_model: cols["pred"]},
        focus_model=wf_model,
        confidence_col=cols["conf"],
    )
    score_fn = make_score_fn(
        predictions,
        pred_col=cols["pred"],
        feature_columns=feature_columns,
    )
    loop = WeeklyReflectionLoop(
        proposer=proposer,
        report_fn=report_fn,
        score_fn=score_fn,
        feature_columns=feature_columns,
        model_names=(wf_model,),
        epsilon=epsilon,
    )

    results = [loop.run_window(window) for window in windows]
    frame = lessons_to_frame(results)
    if out_path is not None:
        path = Path(out_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
    return results


def default_cache_path(instrument: str, sentiment: str, horizon: int,
                       wf_model: str = DEFAULT_WF_MODEL) -> Path:
    from experiments.horizons import horizon_label

    return DEFAULT_CACHE_DIR / f"{instrument}_{sentiment}_{wf_model}_{horizon_label(horizon)}_2025.parquet"


def default_output_path(instrument: str, sentiment: str, horizon: int, model: str,
                        wf_model: str = DEFAULT_WF_MODEL) -> Path:
    safe_model = model.replace(":", "_").replace("/", "_")
    from experiments.horizons import horizon_label

    label = horizon_label(horizon)
    # the legacy catboost artifacts keep their original names (notebook 05 reads them)
    wf_tag = "" if wf_model == DEFAULT_WF_MODEL else f"_{wf_model}"
    return DEFAULT_OUT_DIR / f"{instrument}_{sentiment}_{label}{wf_tag}_{safe_model}_lessons_2025.parquet"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instrument", default="btc", choices=["btc", "usa500", "usatech"])
    ap.add_argument("--sentiment", default="both", choices=["none", "classic", "llm", "both"])
    from experiments.horizons import DEFAULT_LABEL, parse_horizon
    ap.add_argument("--horizon", type=parse_horizon, default=DEFAULT_LABEL,
                    help="m15 (default) / h1 / h4, or a bar count")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--wf-model", default=DEFAULT_WF_MODEL,
                    help="walk-forward cache/model tag (catboost = legacy untuned cache; "
                         "catboost_balanced / gru / ... = tuned contender caches)")
    ap.add_argument("--epsilon", type=float, default=0.0)
    ap.add_argument("--limit-windows", type=int, default=None)
    args = ap.parse_args()

    cache_path = Path(args.cache) if args.cache else default_cache_path(
        args.instrument, args.sentiment, args.horizon, args.wf_model)
    out_path = Path(args.out) if args.out else default_output_path(
        args.instrument,
        args.sentiment,
        args.horizon,
        args.model,
        args.wf_model,
    )
    predictions = load_predictions(cache_path)
    feature_columns = wf_feature_columns(predictions, args.wf_model)
    proposer = OllamaLessonProposer(
        feature_columns=feature_columns,
        model_names=(args.wf_model,),
        model=args.model,
    )
    results = run_reflection_from_cache(
        predictions,
        proposer=proposer,
        feature_columns=feature_columns,
        out_path=out_path,
        epsilon=args.epsilon,
        limit_windows=args.limit_windows,
        wf_model=args.wf_model,
    )
    frame = lessons_to_frame(results)
    accepted = int((frame["status"] == "accepted").sum()) if "status" in frame else 0
    rejected = int((frame["status"] == "rejected").sum()) if "status" in frame else 0
    print(f"wrote {len(frame):,} lesson decisions ({accepted} accepted, {rejected} rejected) -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
