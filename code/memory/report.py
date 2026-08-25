"""Build compact error reports for the RQ3 reflection agent.

Reports contain metrics, condition breakdowns, and selected mistakes.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from evaluation.metrics import classification_scores
from memory.loop import ReflectionWindow


def _as_model_map(model_pred_cols: Mapping[str, str] | Sequence[str]) -> dict[str, str]:
    if isinstance(model_pred_cols, Mapping):
        return dict(model_pred_cols)
    return {name: name for name in model_pred_cols}


def _class_counts(y: pd.Series) -> dict[str, int]:
    counts = y.value_counts(dropna=False).sort_index()
    return {str(k): int(v) for k, v in counts.items()}


def _annualized_sharpe(
    y_pred: pd.Series,
    forward_return: pd.Series,
    periods_per_year: int,
) -> float | None:
    """Compute Sharpe from class predictions mapped to trading positions."""
    positions = y_pred.map({0: -1.0, 1: 0.0, 2: 1.0}).astype(float)
    pnl = positions * forward_return.astype(float)
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pnl) < 2:
        return None
    sd = pnl.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return None
    return float(np.sqrt(periods_per_year) * pnl.mean() / sd)


def _metrics_for_frame(
    frame: pd.DataFrame,
    y_true_col: str,
    model_map: dict[str, str],
    forward_return_col: str | None,
    periods_per_year: int,
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for model, pred_col in model_map.items():
        valid = frame[[y_true_col, pred_col]].dropna()
        if valid.empty:
            out[model] = {"n": 0, "macro_f1": None, "balanced_accuracy": None, "sharpe": None}
            continue
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            warnings.filterwarnings("ignore", message="A single label was found")
            scores = classification_scores(valid[y_true_col].astype(int), valid[pred_col].astype(int))
        sharpe = None
        if forward_return_col and forward_return_col in frame.columns:
            aligned = frame.loc[valid.index, [pred_col, forward_return_col]].dropna()
            if not aligned.empty:
                sharpe = _annualized_sharpe(
                    aligned[pred_col].astype(int),
                    aligned[forward_return_col],
                    periods_per_year,
                )
        out[model] = {
            "n": int(len(valid)),
            "macro_f1": float(scores["macro_f1"]),
            "balanced_accuracy": float(scores["balanced_accuracy"]),
            "per_class_f1": {k: float(v) for k, v in scores["per_class_f1"].items()},
            "sharpe": sharpe,
        }
    return out


def _condition_breakdowns(
    frame: pd.DataFrame,
    y_true_col: str,
    model_map: dict[str, str],
    condition_cols: Sequence[str],
    forward_return_col: str | None,
    periods_per_year: int,
    min_group_size: int,
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    breakdowns: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for col in condition_cols:
        if col not in frame.columns:
            raise ValueError(f"condition column missing from predictions frame: {col}")
        col_report: dict[str, dict[str, dict[str, object]]] = {}
        for value, group in frame.groupby(col, dropna=False):
            if len(group) < min_group_size:
                continue
            col_report[str(value)] = _metrics_for_frame(
                group,
                y_true_col=y_true_col,
                model_map=model_map,
                forward_return_col=forward_return_col,
                periods_per_year=periods_per_year,
            )
        breakdowns[col] = col_report
    return breakdowns


def _worst_predictions(
    frame: pd.DataFrame,
    y_true_col: str,
    model_name: str,
    pred_col: str,
    condition_cols: Sequence[str],
    news_cols: Sequence[str],
    confidence_col: str | None,
    max_rows: int,
) -> list[dict[str, object]]:
    valid = frame[[y_true_col, pred_col]].dropna()
    misses = frame.loc[valid.index][valid[y_true_col].astype(int) != valid[pred_col].astype(int)]
    if misses.empty:
        return []

    sort_cols: list[str] = []
    ascending: list[bool] = []
    if confidence_col and confidence_col in misses.columns:
        sort_cols.append(confidence_col)
        ascending.append(False)
    misses = misses.assign(_timestamp=misses.index.astype(str))
    sort_cols.append("_timestamp")
    ascending.append(True)
    misses = misses.sort_values(sort_cols, ascending=ascending).head(max_rows)

    rows: list[dict[str, object]] = []
    for ts, row in misses.iterrows():
        item: dict[str, object] = {
            "timestamp": str(ts),
            "model": model_name,
            "y_true": int(row[y_true_col]),
            "y_pred": int(row[pred_col]),
        }
        if confidence_col and confidence_col in row and pd.notna(row[confidence_col]):
            item["confidence"] = float(row[confidence_col])
        conditions = {
            col: None if pd.isna(row[col]) else row[col]
            for col in condition_cols
            if col in row
        }
        if conditions:
            item["conditions"] = conditions
        news = {
            col: None if pd.isna(row[col]) else row[col]
            for col in news_cols
            if col in row
        }
        if news:
            item["news_context"] = news
        rows.append(item)
    return rows


def _model_disagreement(
    frame: pd.DataFrame,
    model_map: dict[str, str],
    condition_cols: Sequence[str],
    min_group_size: int,
    max_groups: int = 8,
) -> dict[str, object]:
    pred_cols = list(model_map.values())
    valid = frame[pred_cols].dropna()
    if valid.empty or len(pred_cols) < 2:
        return {"n": int(len(valid)), "overall_rate": 0.0, "top_conditions": []}

    disagreement = valid.nunique(axis=1) > 1
    top: list[dict[str, object]] = []
    scoped = frame.loc[valid.index]
    for col in condition_cols:
        if col not in scoped.columns:
            continue
        for value, group in scoped.groupby(col, dropna=False):
            if len(group) < min_group_size:
                continue
            rate = float(disagreement.loc[group.index].mean())
            top.append({
                "condition": col,
                "value": None if pd.isna(value) else value,
                "n": int(len(group)),
                "disagreement_rate": rate,
            })
    top = sorted(top, key=lambda row: (row["disagreement_rate"], row["n"]), reverse=True)[:max_groups]
    return {"n": int(len(valid)), "overall_rate": float(disagreement.mean()), "top_conditions": top}


def _single_model_weaknesses(
    frame: pd.DataFrame,
    y_true_col: str,
    model_map: dict[str, str],
    condition_cols: Sequence[str],
    min_group_size: int,
    max_rows: int = 12,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model, pred_col in model_map.items():
        valid = frame[[y_true_col, pred_col]].dropna()
        if valid.empty:
            continue
        scoped = frame.loc[valid.index]
        for col in condition_cols:
            if col not in scoped.columns:
                continue
            for value, group in scoped.groupby(col, dropna=False):
                if len(group) < min_group_size:
                    continue
                y_true = group[y_true_col].astype(int)
                y_pred = group[pred_col].astype(int)
                error_rate = float((y_true != y_pred).mean())
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
                    warnings.filterwarnings("ignore", message="A single label was found")
                    scores = classification_scores(y_true, y_pred)
                rows.append({
                    "model": model,
                    "condition": col,
                    "value": None if pd.isna(value) else value,
                    "n": int(len(group)),
                    "error_rate": error_rate,
                    "macro_f1": float(scores["macro_f1"]),
                })
    return sorted(rows, key=lambda row: (row["error_rate"], row["n"]), reverse=True)[:max_rows]


def build_error_report(
    window: ReflectionWindow,
    predictions: pd.DataFrame,
    model_pred_cols: Mapping[str, str] | Sequence[str],
    *,
    y_true_col: str = "y_true",
    condition_cols: Sequence[str] = (),
    news_cols: Sequence[str] = (),
    focus_model: str | None = None,
    confidence_col: str | None = None,
    forward_return_col: str | None = None,
    periods_per_year: int = 365 * 24 * 4,
    min_group_size: int = 10,
    max_worst: int = 10,
) -> dict[str, object]:
    """Build the compact weekly digest consumed by the reflection proposer."""
    if not isinstance(predictions.index, pd.DatetimeIndex):
        raise ValueError("predictions must use a DatetimeIndex")
    if y_true_col not in predictions.columns:
        raise ValueError(f"missing truth column: {y_true_col}")

    model_map = _as_model_map(model_pred_cols)
    missing = [col for col in model_map.values() if col not in predictions.columns]
    if missing:
        raise ValueError(f"missing prediction column(s): {missing}")

    frame = predictions.sort_index().loc[window.validation_start:window.validation_end]
    if frame.empty:
        raise ValueError(f"no rows inside validation window: {window.name}")

    focus = focus_model or next(iter(model_map))
    if focus not in model_map:
        raise ValueError(f"focus model not present in model_pred_cols: {focus}")

    worst = _worst_predictions(
        frame,
        y_true_col=y_true_col,
        model_name=focus,
        pred_col=model_map[focus],
        condition_cols=condition_cols,
        news_cols=news_cols,
        confidence_col=confidence_col,
        max_rows=max_worst,
    )

    return {
        "window": asdict(window),
        "n_bars": int(len(frame)),
        "label_distribution": _class_counts(frame[y_true_col]),
        "overall": _metrics_for_frame(
            frame,
            y_true_col=y_true_col,
            model_map=model_map,
            forward_return_col=forward_return_col,
            periods_per_year=periods_per_year,
        ),
        "by_condition": _condition_breakdowns(
            frame,
            y_true_col=y_true_col,
            model_map=model_map,
            condition_cols=condition_cols,
            forward_return_col=forward_return_col,
            periods_per_year=periods_per_year,
            min_group_size=min_group_size,
        ),
        "model_disagreement": _model_disagreement(
            frame,
            model_map=model_map,
            condition_cols=condition_cols,
            min_group_size=min_group_size,
        ),
        "single_model_weaknesses": _single_model_weaknesses(
            frame,
            y_true_col=y_true_col,
            model_map=model_map,
            condition_cols=condition_cols,
            min_group_size=min_group_size,
        ),
        "high_confidence_errors": worst,
        "worst_predictions": worst,
    }
