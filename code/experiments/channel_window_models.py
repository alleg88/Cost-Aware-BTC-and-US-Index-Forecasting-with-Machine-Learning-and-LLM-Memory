"""Leak-resistant OOF LogReg and CatBoost models for Notebook B."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from evaluation.channel_window_validation import (
    PurgedFold,
    effective_sample_size,
    expanding_purged_folds,
    interval_uniqueness,
)
from experiments.channel_window_dataset import CHANNEL_WINDOW_FEATURES


@dataclass
class OOFResult:
    predictions: pd.DataFrame
    audit: pd.DataFrame
    feature_columns: Sequence[str]
    architecture: str
    model_kind: str


_MODEL_REQUIRED = frozenset(
    {
        *CHANNEL_WINDOW_FEATURES,
        "side",
        "channel_episode_id",
        "decision_time",
        "label_start",
        "label_end",
        "r_net",
        "label_net_positive",
        "filled",
    }
)


def _validate(events: pd.DataFrame, architecture: str) -> pd.DataFrame:
    if architecture not in ("pooled", "separate"):
        raise ValueError("architecture must be 'pooled' or 'separate'")
    missing = sorted(_MODEL_REQUIRED.difference(events.columns))
    if missing:
        raise ValueError(f"events missing model columns: {missing}")
    work = events.copy().reset_index(drop=True)
    if not work["side"].isin(["long", "short"]).all():
        raise ValueError("model side must be long or short")
    if work["r_net"].isna().any() or work["label_net_positive"].isna().any():
        raise ValueError("OOF targets cannot be missing")
    work["_event_position"] = np.arange(len(work), dtype=np.int64)
    return work


def _model_slices(
    events: pd.DataFrame, fold: PurgedFold, architecture: str
) -> list[tuple[str | None, np.ndarray, np.ndarray]]:
    if architecture == "pooled":
        return [(None, fold.train, fold.valid)]
    slices = []
    side_values = events["side"].to_numpy()
    for side in ("long", "short"):
        slices.append(
            (
                side,
                fold.train[side_values[fold.train] == side],
                fold.valid[side_values[fold.valid] == side],
            )
        )
    return slices


def _audit_row(
    events: pd.DataFrame,
    fold: PurgedFold,
    model_side: str | None,
    train: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    raw_uniqueness: np.ndarray,
    status: str,
) -> dict[str, object]:
    train_episodes = set(events.iloc[train]["channel_episode_id"])
    valid_episodes = set(events.iloc[valid]["channel_episode_id"])
    return {
        "fold_id": fold.fold_id,
        "model_side": model_side,
        "status": status,
        "n_train": int(len(train)),
        "n_valid": int(len(valid)),
        "train_episodes": int(len(train_episodes)),
        "valid_episodes": int(len(valid_episodes)),
        "train_valid_episode_overlap": int(len(train_episodes & valid_episodes)),
        "train_ess": effective_sample_size(weights) if len(weights) else 0.0,
        "raw_uniqueness_mean": (
            float(raw_uniqueness.mean()) if len(raw_uniqueness) else np.nan
        ),
    }


def _prediction_rows(
    events: pd.DataFrame,
    valid: np.ndarray,
    fold_id: str,
    scores: np.ndarray,
) -> pd.DataFrame:
    out = events.iloc[valid].copy()
    out.insert(
        0,
        "row_id",
        out["candidate_id"].astype(str).to_numpy()
        if "candidate_id" in out
        else out["_event_position"].to_numpy(),
    )
    out.insert(1, "fold_id", fold_id)
    out.insert(2, "score", np.asarray(scores, dtype=float))
    return out.drop(columns="_event_position")


def _balanced_weights(labels: np.ndarray, uniqueness: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int8)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) != 2:
        raise ValueError("LogReg training fold needs both target classes")
    class_weight = {
        int(cls): len(labels) / (len(classes) * int(count))
        for cls, count in zip(classes, counts, strict=True)
    }
    return uniqueness * np.array([class_weight[int(label)] for label in labels])


def _logreg_pipeline(architecture: str) -> tuple[Pipeline, list[str]]:
    numeric = list(CHANNEL_WINDOW_FEATURES)
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scaler", StandardScaler()),
        ]
    )
    transformers: list[tuple[str, object, list[str]]] = [("numeric", numeric_pipe, numeric)]
    columns = numeric.copy()
    if architecture == "pooled":
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("side", categorical_pipe, ["side"]))
        columns.append("side")
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "logisticregression",
                LogisticRegression(
                    C=1.0,
                    max_iter=2000,
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )
    return model, columns


def run_logreg_oof(events: pd.DataFrame, *, architecture: str) -> OOFResult:
    """Generate classification probabilities from purged expanding folds."""
    work = _validate(events, architecture)
    folds = expanding_purged_folds(work)
    predictions: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    for fold in folds:
        for model_side, train, valid in _model_slices(work, fold, architecture):
            uniqueness = interval_uniqueness(work, train) if len(train) else np.array([])
            raw = (
                interval_uniqueness(work, train, normalize=False)
                if len(train)
                else np.array([])
            )
            target = work.iloc[train]["label_net_positive"].to_numpy(dtype=np.int8)
            if not len(train) or not len(valid) or len(np.unique(target)) != 2:
                audit.append(
                    _audit_row(work, fold, model_side, train, valid, uniqueness, raw, "skipped")
                )
                continue
            weights = _balanced_weights(target, uniqueness)
            model, columns = _logreg_pipeline(architecture)
            model.fit(
                work.iloc[train][columns],
                target,
                logisticregression__sample_weight=weights,
            )
            score = model.predict_proba(work.iloc[valid][columns])[:, 1]
            predictions.append(_prediction_rows(work, valid, fold.fold_id, score))
            audit.append(
                _audit_row(work, fold, model_side, train, valid, weights, raw, "fitted")
            )
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=["row_id", "fold_id", "score"])
    )
    if not prediction_frame.empty and prediction_frame["row_id"].duplicated().any():
        raise AssertionError("an event received more than one LogReg OOF score")
    return OOFResult(
        predictions=prediction_frame.sort_values("decision_time", kind="stable").reset_index(
            drop=True
        ) if not prediction_frame.empty else prediction_frame,
        audit=pd.DataFrame(audit),
        feature_columns=CHANNEL_WINDOW_FEATURES,
        architecture=architecture,
        model_kind="logreg",
    )


def logreg_continuation(result: OOFResult) -> bool:
    """Predeclared gate for whether the nonlinear model is worth fitting."""
    scored = result.predictions.dropna(subset=["score", "r_net"]).copy()
    if scored.empty:
        return False
    threshold = float(scored["score"].quantile(0.70))
    selected = scored[scored["score"] >= threshold]
    selected_filled = selected[selected["filled"].astype(bool)]
    if selected["r_net"].mean() <= scored["r_net"].mean():
        return False
    if len(selected_filled) < 30:
        return False
    if selected_filled["channel_episode_id"].nunique() < 20:
        return False
    if result.architecture == "pooled" and set(selected_filled["side"]) != {"long", "short"}:
        return False
    return True


def _catboost_columns(architecture: str) -> tuple[list[str], list[str]]:
    columns = list(CHANNEL_WINDOW_FEATURES)
    categorical: list[str] = []
    if architecture == "pooled":
        columns.append("side")
        categorical.append("side")
    return columns, categorical


def run_catboost_oof(events: pd.DataFrame, *, architecture: str) -> OOFResult:
    """Generate direct expected-net-R scores with one fixed shallow CatBoost."""
    work = _validate(events, architecture)
    folds = expanding_purged_folds(work)
    predictions: list[pd.DataFrame] = []
    audit: list[dict[str, object]] = []
    columns, categorical = _catboost_columns(architecture)
    for fold in folds:
        for model_side, train, valid in _model_slices(work, fold, architecture):
            uniqueness = interval_uniqueness(work, train) if len(train) else np.array([])
            raw = (
                interval_uniqueness(work, train, normalize=False)
                if len(train)
                else np.array([])
            )
            if not len(train) or not len(valid):
                audit.append(
                    _audit_row(work, fold, model_side, train, valid, uniqueness, raw, "skipped")
                )
                continue
            model = CatBoostRegressor(
                iterations=400,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=10.0,
                loss_function="RMSE",
                random_seed=42,
                allow_writing_files=False,
                verbose=False,
                thread_count=1,
            )
            train_x = work.iloc[train][columns].copy()
            valid_x = work.iloc[valid][columns].copy()
            if categorical:
                train_x["side"] = train_x["side"].astype(str)
                valid_x["side"] = valid_x["side"].astype(str)
            model.fit(
                train_x,
                work.iloc[train]["r_net"].to_numpy(dtype=float),
                cat_features=categorical,
                sample_weight=uniqueness,
            )
            score = model.predict(valid_x)
            predictions.append(_prediction_rows(work, valid, fold.fold_id, score))
            audit.append(
                _audit_row(work, fold, model_side, train, valid, uniqueness, raw, "fitted")
            )
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=["row_id", "fold_id", "score"])
    )
    if not prediction_frame.empty and prediction_frame["row_id"].duplicated().any():
        raise AssertionError("an event received more than one CatBoost OOF score")
    return OOFResult(
        predictions=prediction_frame.sort_values("decision_time", kind="stable").reset_index(
            drop=True
        ) if not prediction_frame.empty else prediction_frame,
        audit=pd.DataFrame(audit),
        feature_columns=CHANNEL_WINDOW_FEATURES,
        architecture=architecture,
        model_kind="catboost_regressor",
    )


def score_frozen_model(
    train_events: pd.DataFrame,
    score_events: pd.DataFrame,
    *,
    architecture: str,
    model_kind: str,
) -> pd.DataFrame:
    """Refit one frozen specification on all dev rows and score the next stage."""
    train = _validate(train_events, architecture)
    score = _validate(score_events, architecture)
    if model_kind not in {"logreg", "catboost_regressor"}:
        raise ValueError(f"unknown frozen model kind: {model_kind!r}")
    output = score_events.copy().reset_index(drop=True)
    output["score"] = np.nan
    side_slices = [(None, np.arange(len(train)), np.arange(len(score)))]
    if architecture == "separate":
        side_slices = [
            (
                side,
                np.flatnonzero(train["side"].to_numpy() == side),
                np.flatnonzero(score["side"].to_numpy() == side),
            )
            for side in ("long", "short")
        ]
    for _, train_index, score_index in side_slices:
        if not len(train_index) or not len(score_index):
            continue
        uniqueness = interval_uniqueness(train, train_index)
        if model_kind == "logreg":
            target = train.iloc[train_index]["label_net_positive"].to_numpy(dtype=np.int8)
            weights = _balanced_weights(target, uniqueness)
            model, columns = _logreg_pipeline(architecture)
            model.fit(
                train.iloc[train_index][columns],
                target,
                logisticregression__sample_weight=weights,
            )
            prediction = model.predict_proba(score.iloc[score_index][columns])[:, 1]
        else:
            columns, categorical = _catboost_columns(architecture)
            model = CatBoostRegressor(
                iterations=400,
                depth=4,
                learning_rate=0.03,
                l2_leaf_reg=10.0,
                loss_function="RMSE",
                random_seed=42,
                allow_writing_files=False,
                verbose=False,
                thread_count=1,
            )
            train_x = train.iloc[train_index][columns].copy()
            score_x = score.iloc[score_index][columns].copy()
            if categorical:
                train_x["side"] = train_x["side"].astype(str)
                score_x["side"] = score_x["side"].astype(str)
            model.fit(
                train_x,
                train.iloc[train_index]["r_net"].to_numpy(dtype=float),
                cat_features=categorical,
                sample_weight=uniqueness,
            )
            prediction = model.predict(score_x)
        output.loc[score_index, "score"] = np.asarray(prediction, dtype=float)
    if output["score"].isna().any():
        raise ValueError("frozen model left score rows unpredicted")
    return output
