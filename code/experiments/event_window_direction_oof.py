"""Expanding second-stage OOF orchestration for Notebook V direction models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.event_window_direction_dataset import (
    DirectionDataset,
    interval_uniqueness,
)
from experiments.event_window_direction_models import (
    DirectionModelConfig,
    fit_predict_delta_xgboost,
    fit_predict_value_logreg,
)


WARMUP_FOLD = "2022H1"
SCORED_FOLDS = (
    "2022H2",
    "2023H1",
    "2023H2",
    "2024H1",
    "2024H2",
    "2025H1",
)
_FOLD_ORDER = (WARMUP_FOLD, *SCORED_FOLDS)
_FOLD_STARTS = {
    "2022H1": pd.Timestamp("2022-01-01", tz="UTC"),
    "2022H2": pd.Timestamp("2022-07-01", tz="UTC"),
    "2023H1": pd.Timestamp("2023-01-01", tz="UTC"),
    "2023H2": pd.Timestamp("2023-07-01", tz="UTC"),
    "2024H1": pd.Timestamp("2024-01-01", tz="UTC"),
    "2024H2": pd.Timestamp("2024-07-01", tz="UTC"),
    "2025H1": pd.Timestamp("2025-01-01", tz="UTC"),
}
_FOLD_ENDS = {
    fold_id: _FOLD_STARTS[_FOLD_ORDER[position + 1]]
    for position, fold_id in enumerate(_FOLD_ORDER[:-1])
}
_FOLD_ENDS["2025H1"] = pd.Timestamp("2025-07-01", tz="UTC")
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class DirectionOOFResult:
    predictions: pd.DataFrame
    fold_audit: pd.DataFrame


def _validated_inputs(dataset: DirectionDataset) -> tuple[pd.DataFrame, np.ndarray]:
    decisions = dataset.decisions.copy().reset_index(drop=True)
    required = {
        "activation_key",
        "channel_side",
        "fold_id",
        "channel_episode_id",
        "decision_time",
        "delta_r",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"direction decisions missing columns: {missing}")
    matrix = np.asarray(dataset.tabular, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != len(decisions):
        raise ValueError("direction feature rows must align with decisions")
    if matrix.shape[1] != len(dataset.tabular_features):
        raise ValueError("direction feature columns must align with feature names")
    if len(dataset.tabular_features) != len(set(dataset.tabular_features)):
        raise ValueError("direction feature names must be unique")
    if decisions["activation_key"].isna().any() or decisions["activation_key"].duplicated().any():
        raise ValueError("direction activation keys must be unique and present")
    if decisions["channel_episode_id"].isna().any():
        raise ValueError("direction episode identifiers must be present")
    decisions["channel_side"] = decisions["channel_side"].astype(str).str.lower()
    if not decisions["channel_side"].isin(("long", "short")).all():
        raise ValueError("channel_side must contain only long or short")
    decisions["fold_id"] = decisions["fold_id"].astype(str)
    unknown = sorted(set(decisions["fold_id"]).difference(_FOLD_ORDER))
    if unknown:
        raise ValueError(f"unsupported frozen direction folds: {unknown}")
    missing_folds = [fold for fold in _FOLD_ORDER if fold not in set(decisions["fold_id"])]
    if missing_folds:
        raise ValueError(f"direction dataset missing frozen folds: {missing_folds}")
    decisions["decision_time"] = pd.to_datetime(
        decisions["decision_time"], utc=True, errors="raise"
    )
    decisions["delta_r"] = pd.to_numeric(decisions["delta_r"], errors="raise")
    if decisions[["decision_time", "delta_r"]].isna().any().any() or not np.isfinite(
        decisions["delta_r"].to_numpy(dtype=float)
    ).all():
        raise ValueError("direction decision times and delta_r must be finite and present")
    for fold_id in _FOLD_ORDER:
        in_fold = decisions["fold_id"].eq(fold_id)
        times = decisions.loc[in_fold, "decision_time"]
        if (times < _FOLD_STARTS[fold_id]).any() or (times >= _FOLD_ENDS[fold_id]).any():
            raise ValueError(f"direction timestamps fall outside frozen fold {fold_id}")
    return decisions, matrix


def _prediction_rows(
    validation: pd.DataFrame,
    *,
    fold_id: str,
    logreg_score: np.ndarray,
    xgb_delta: np.ndarray,
) -> pd.DataFrame:
    identity = validation[
        ["activation_key", "channel_episode_id", "decision_time", "channel_side"]
    ].reset_index(drop=True)
    logreg = identity.copy()
    logreg.insert(0, "fold_id", fold_id)
    logreg.insert(0, "model", "logreg")
    logreg["direction_score"] = np.asarray(logreg_score, dtype=float) - 0.5
    logreg["p_long"] = np.asarray(logreg_score, dtype=float)
    logreg["predicted_delta_r"] = np.nan
    logreg["chosen_direction"] = np.where(
        np.asarray(logreg_score, dtype=float) >= 0.5, "long", "short"
    )

    xgboost = identity.copy()
    xgboost.insert(0, "fold_id", fold_id)
    xgboost.insert(0, "model", "xgboost")
    xgboost["direction_score"] = np.asarray(xgb_delta, dtype=float)
    xgboost["p_long"] = np.nan
    xgboost["predicted_delta_r"] = np.asarray(xgb_delta, dtype=float)
    xgboost["chosen_direction"] = np.where(
        np.asarray(xgb_delta, dtype=float) > 0.0,
        "long",
        np.where(
            np.asarray(xgb_delta, dtype=float) < 0.0,
            "short",
            identity["channel_side"],
        ),
    )
    return pd.concat([logreg, xgboost], ignore_index=True)


def run_direction_oof(
    dataset: DirectionDataset,
    *,
    config: DirectionModelConfig = DirectionModelConfig(),
) -> DirectionOOFResult:
    """Fit on strictly earlier frozen activations and score every later activation."""
    decisions, matrix = _validated_inputs(dataset)
    predictions: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []

    for fold_position, fold_id in enumerate(SCORED_FOLDS, start=1):
        validation_mask = decisions["fold_id"].eq(fold_id).to_numpy()
        validation_positions = np.flatnonzero(validation_mask)
        validation = decisions.iloc[validation_positions]
        validation_start = _FOLD_STARTS[fold_id]
        validation_episodes = set(validation["channel_episode_id"])

        earlier = set(_FOLD_ORDER[:fold_position])
        train_mask = decisions["fold_id"].isin(earlier)
        train_mask &= ~decisions["channel_episode_id"].isin(validation_episodes)
        label_end = decisions["decision_time"] + pd.Timedelta(minutes=120)
        train_mask &= label_end < validation_start
        train_positions = np.flatnonzero(train_mask.to_numpy())
        if not len(train_positions):
            raise ValueError(f"fold {fold_id} has no purged prior training rows")

        train_delta = decisions.iloc[train_positions]["delta_r"].to_numpy(dtype=float)
        non_tie = np.abs(train_delta) > _TIE_TOLERANCE
        fit_positions = train_positions[non_tie]
        if not len(fit_positions):
            raise ValueError(f"fold {fold_id} has no non-tie training rows")
        uniqueness = interval_uniqueness(
            decisions.iloc[fit_positions]["decision_time"],
            horizon_minutes=120,
            normalize=True,
        )
        fit_delta = decisions.iloc[fit_positions]["delta_r"].to_numpy(dtype=float)
        score_x = matrix[validation_positions]
        logreg = fit_predict_value_logreg(
            train_x=matrix[fit_positions],
            delta_r=fit_delta,
            uniqueness=uniqueness,
            score_x=score_x,
            config=config,
        )
        xgboost = fit_predict_delta_xgboost(
            train_x=matrix[fit_positions],
            delta_r=fit_delta,
            uniqueness=uniqueness,
            score_x=score_x,
            config=config,
        )
        fold_predictions = _prediction_rows(
            validation,
            fold_id=fold_id,
            logreg_score=logreg.score,
            xgb_delta=xgboost.predicted_delta_r,
        )
        if len(fold_predictions) != 2 * len(validation):
            raise AssertionError("direction models did not score every activation")
        predictions.append(fold_predictions)

        train = decisions.iloc[train_positions]
        audits.append(
            {
                "fold_id": fold_id,
                "train_rows_before_ties": len(train_positions),
                "train_ties_excluded": int((~non_tie).sum()),
                "train_fit_rows": len(fit_positions),
                "validation_rows": len(validation_positions),
                "train_episodes": train["channel_episode_id"].nunique(),
                "validation_episodes": validation["channel_episode_id"].nunique(),
                "episode_overlap": len(
                    set(train["channel_episode_id"]) & validation_episodes
                ),
                "train_label_end_max": (
                    train["decision_time"] + pd.Timedelta(minutes=120)
                ).max(),
                "validation_start": validation_start,
                "uniqueness_mean": float(np.mean(uniqueness)),
                "uniqueness_min": float(np.min(uniqueness)),
                "uniqueness_max": float(np.max(uniqueness)),
            }
        )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    expected_keys = set(
        decisions.loc[decisions["fold_id"].isin(SCORED_FOLDS), "activation_key"]
    )
    for model in ("logreg", "xgboost"):
        model_rows = prediction_frame.loc[prediction_frame["model"].eq(model)]
        if set(model_rows["activation_key"]) != expected_keys or model_rows[
            "activation_key"
        ].duplicated().any():
            raise AssertionError(f"{model} did not preserve every scored activation")
    if not np.isfinite(prediction_frame["direction_score"]).all():
        raise AssertionError("direction scores must be finite")
    return DirectionOOFResult(
        predictions=prediction_frame,
        fold_audit=pd.DataFrame(audits),
    )


__all__ = [
    "DirectionOOFResult",
    "SCORED_FOLDS",
    "WARMUP_FOLD",
    "run_direction_oof",
]
