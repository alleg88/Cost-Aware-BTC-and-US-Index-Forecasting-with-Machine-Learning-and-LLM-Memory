"""Build walk-forward prediction caches for the RQ3 reflection loop.

The cache stores unseen validation predictions and optional feature columns.
Helper functions convert the cache into reflection reports and validation scores.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.metrics import classification_scores
from memory.gates import condition_mask
from memory.loop import ReflectionWindow
from memory.report import build_error_report
from memory.schema import Lesson

CLASS_LABELS = (0, 1, 2)


def _timestamp(value: str | pd.Timestamp, tz) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if tz is None:
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)


def weekly_walkforward_windows(
    index: pd.DatetimeIndex,
    walk_start: str,
    walk_end: str,
    *,
    train_lookback: str = "180D",
    step: str = "7D",
    bar_size: str = "15min",
) -> list[ReflectionWindow]:
    """Return full validation windows with past-only rolling training ranges."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a DatetimeIndex")
    tz = index.tz
    start = _timestamp(walk_start, tz)
    end = _timestamp(walk_end, tz)
    if end == end.normalize():
        final_exclusive = end + pd.Timedelta(days=1)
    else:
        final_exclusive = end + pd.Timedelta(bar_size)

    windows: list[ReflectionWindow] = []
    lookback = pd.Timedelta(train_lookback)
    bar = pd.Timedelta(bar_size)
    latest_start = final_exclusive - pd.Timedelta(step)
    if latest_start < start:
        return windows
    starts = pd.date_range(start, latest_start, freq=step, tz=start.tz)
    for i, validation_start in enumerate(starts):
        validation_exclusive = min(validation_start + pd.Timedelta(step), final_exclusive)
        validation_end = validation_exclusive - bar
        if validation_start > validation_end:
            continue
        windows.append(
            ReflectionWindow(
                name=f"wf_{i:03d}_{validation_start.date()}",
                train_start=validation_start - lookback,
                train_end=validation_start,
                validation_start=validation_start,
                validation_end=validation_end,
            )
        )
    return windows


def _make_model(model_factory: Callable, params: dict | None):
    if params is None:
        return model_factory()
    try:
        return model_factory(params)
    except TypeError:
        return model_factory()


def _predict_proba_3(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(X), dtype=float)
        classes = list(getattr(model, "classes_", CLASS_LABELS))
        out = np.zeros((len(X), len(CLASS_LABELS)), dtype=float)
        for src_col, cls in enumerate(classes):
            if int(cls) in CLASS_LABELS:
                out[:, CLASS_LABELS.index(int(cls))] = raw[:, src_col]
        row_sum = out.sum(axis=1)
        missing = row_sum == 0
        out[missing, 1] = 1.0
        out[~missing] = out[~missing] / row_sum[~missing, None]
        return out

    pred = np.asarray(model.predict(X)).ravel().astype(int)
    out = np.zeros((len(X), len(CLASS_LABELS)), dtype=float)
    for i, cls in enumerate(pred):
        out[i, CLASS_LABELS.index(int(cls)) if int(cls) in CLASS_LABELS else 1] = 1.0
    return out


def run_walkforward_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    windows: Sequence[ReflectionWindow],
    model_factory: Callable,
    model_name: str = "catboost",
    params: dict | None = None,
    cache_path: str | Path | None = None,
    min_train_rows: int = 1,
    min_validation_rows: int = 1,
    include_features: bool = False,
    progress_label: str | None = None,
    train_tail_trim: int = 0,
    bar_size: str = "15min",
    sample_weight_fn: Callable[[pd.DataFrame, pd.Series], Sequence[float]] | None = None,
) -> pd.DataFrame:
    """Fit one model per window and cache the unseen validation predictions.

    train_tail_trim drops the last N bars of every training window so that
    labels with an N-bar lookahead (horizon > 1, triple-barrier max_hold)
    cannot peek into the validation week.
    """
    X = X.sort_index()
    y = y.reindex(X.index).sort_index()
    frames: list[pd.DataFrame] = []
    window_iter: Iterable[ReflectionWindow] = windows
    if progress_label:
        from tqdm import tqdm

        window_iter = tqdm(windows, total=len(windows), desc=progress_label, unit="window")

    trim = pd.Timedelta(bar_size) * train_tail_trim
    for window in window_iter:
        train_mask = (X.index >= window.train_start) & (X.index < window.train_end - trim)
        valid_mask = (X.index >= window.validation_start) & (X.index <= window.validation_end)
        X_train, y_train = X.loc[train_mask], y.loc[train_mask]
        X_valid, y_valid = X.loc[valid_mask], y.loc[valid_mask]
        if len(X_train) < min_train_rows or len(X_valid) < min_validation_rows:
            continue
        if y_train.nunique(dropna=True) < 2:
            continue

        model = _make_model(model_factory, params)
        fit_kwargs = {}
        if sample_weight_fn is not None:
            fit_kwargs["sample_weight"] = sample_weight_fn(X_train, y_train)
        model.fit(X_train, y_train, **fit_kwargs)
        proba = _predict_proba_3(model, X_valid)
        pred = np.asarray(CLASS_LABELS, dtype=int)[proba.argmax(axis=1)]

        frame = pd.DataFrame(
            {
                "window": window.name,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "validation_start": window.validation_start,
                "validation_end": window.validation_end,
                "y_true": y_valid.astype(int),
                f"{model_name}_pred": pred.astype(int),
                f"{model_name}_conf": proba.max(axis=1),
                f"{model_name}_p0": proba[:, 0],
                f"{model_name}_p1": proba[:, 1],
                f"{model_name}_p2": proba[:, 2],
            },
            index=X_valid.index,
        )
        if include_features:
            frame = frame.join(X_valid)
        frames.append(frame)

    out = pd.concat(frames).sort_index() if frames else pd.DataFrame()
    if cache_path is not None:
        path = Path(cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path)
    return out


def _window_frame(predictions: pd.DataFrame, window: ReflectionWindow) -> pd.DataFrame:
    return predictions.sort_index().loc[window.validation_start:window.validation_end].copy()


def make_report_fn(
    predictions: pd.DataFrame,
    *,
    model_pred_cols: dict[str, str] | None = None,
    condition_cols: Sequence[str] = (),
    news_cols: Sequence[str] = (),
    focus_model: str = "catboost",
    y_true_col: str = "y_true",
    confidence_col: str = "catboost_conf",
    forward_return_col: str = "forward_return",
):
    """Return a report function for WeeklyReflectionLoop."""
    model_map = model_pred_cols or {"catboost": "catboost_pred"}

    def report_fn(window: ReflectionWindow) -> dict[str, object]:
        kwargs = {
            "condition_cols": condition_cols,
            "news_cols": news_cols,
            "focus_model": focus_model,
            "confidence_col": confidence_col,
        }
        if forward_return_col in predictions.columns:
            kwargs["forward_return_col"] = forward_return_col
        return build_error_report(
            window,
            predictions,
            model_map,
            y_true_col=y_true_col,
            **kwargs,
        )

    return report_fn


def _apply_lessons(
    frame: pd.DataFrame,
    lessons: Iterable[Lesson],
    *,
    pred_col: str,
    feature_columns: Sequence[str],
) -> pd.Series:
    pred = frame[pred_col].astype(int).copy()
    features = frame[list(feature_columns)] if feature_columns else frame
    for lesson in lessons:
        mask = condition_mask(features, lesson.condition).reindex(frame.index, fill_value=False)
        if lesson.action == "force_flat":
            pred.loc[mask] = 1
    return pred


def make_score_fn(
    predictions: pd.DataFrame,
    *,
    pred_col: str = "catboost_pred",
    y_true_col: str = "y_true",
    feature_columns: Sequence[str] = (),
):
    """Return a score function that applies lessons and reports macro-F1."""

    def score_fn(lessons: list[Lesson], window: ReflectionWindow) -> float:
        frame = _window_frame(predictions, window)
        if frame.empty:
            raise ValueError(f"no predictions inside validation window: {window.name}")
        y_pred = _apply_lessons(
            frame,
            lessons,
            pred_col=pred_col,
            feature_columns=feature_columns,
        )
        return float(classification_scores(frame[y_true_col].astype(int), y_pred)["macro_f1"])

    return score_fn
