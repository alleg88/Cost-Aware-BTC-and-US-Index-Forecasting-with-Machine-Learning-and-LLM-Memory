"""Calibrated two-head XGBoost admission primitives for qualified Union v1."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from experiments.qualified_union import (
    CODE_ROOT,
    FORWARD_ROOT,
    H1_ROOT,
    LOCKBOX_START,
)


MOVE_THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30)
DIRECTION_THRESHOLDS = (0.60, 0.65, 0.70, 0.75, 0.80)
XGB_MODEL = "xgboost_balanced"
XGB_WIDTH_BPS = 65
RAW_ROOT = (
    CODE_ROOT
    / "experiments"
    / "cache"
    / "tuning"
    / "all_model_sentiment_raw_180d_fixed15"
    / "none"
    / XGB_MODEL
)
OUTPUT_ROOT = CODE_ROOT / "experiments" / "cache" / "xgb_strong_move_admission"
PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")


@dataclass(frozen=True)
class BinaryLogitCalibrator:
    slope: float
    intercept: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=float), 1e-8, 1.0 - 1e-8)
    return np.log(values / (1.0 - values))


def fit_binary_logit_calibrator(
    probability: np.ndarray | pd.Series,
    target: np.ndarray | pd.Series,
) -> BinaryLogitCalibrator:
    values = np.asarray(probability, dtype=float)
    labels = np.asarray(target, dtype=int)
    if values.ndim != 1 or len(values) != len(labels) or len(values) == 0:
        raise ValueError("probability and target must be non-empty aligned vectors")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("probability must be finite and in [0, 1]")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("calibration target must contain both binary classes")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    model.fit(_logit(values).reshape(-1, 1), labels)
    return BinaryLogitCalibrator(
        slope=float(model.coef_[0, 0]),
        intercept=float(model.intercept_[0]),
    )


def apply_binary_logit_calibrator(
    probability: np.ndarray | pd.Series,
    calibrator: BinaryLogitCalibrator,
) -> np.ndarray:
    values = np.asarray(probability, dtype=float)
    score = calibrator.slope * _logit(values) + calibrator.intercept
    score = np.clip(score, -50.0, 50.0)
    calibrated = 1.0 / (1.0 + np.exp(-score))
    return np.clip(calibrated, 1e-12, 1.0 - 1e-12)


def decompose_xgb_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(PROBABILITY_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"XGBoost prediction frame is missing columns: {missing}")
    values = frame.loc[:, PROBABILITY_COLUMNS].to_numpy(float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("XGBoost probabilities must be finite and non-negative")
    row_sum = values.sum(axis=1, keepdims=True)
    values = np.divide(
        values,
        row_sum,
        out=np.full_like(values, 1.0 / 3.0),
        where=row_sum > 0.0,
    )
    short, flat, long = values.T
    move = short + long
    conditional_long = np.divide(
        long,
        move,
        out=np.full(len(values), 0.5, dtype=float),
        where=move > 0.0,
    )
    return pd.DataFrame(
        {
            "p_move_raw": move,
            "p_long_given_move_raw": conditional_long,
        },
        index=frame.index,
    )


def fit_calibrators(oof_2024: pd.DataFrame) -> dict[str, BinaryLogitCalibrator]:
    decomposed = decompose_xgb_probabilities(oof_2024)
    labels = oof_2024["y_true"].astype(int)
    move_target = labels.ne(1).astype(int)
    move = fit_binary_logit_calibrator(decomposed["p_move_raw"], move_target)
    directional = move_target.eq(1)
    direction = fit_binary_logit_calibrator(
        decomposed.loc[directional, "p_long_given_move_raw"],
        labels.loc[directional].eq(2).astype(int),
    )
    return {"move": move, "direction_given_move": direction}


def apply_calibrators(
    frame: pd.DataFrame,
    calibrators: dict[str, BinaryLogitCalibrator],
) -> pd.DataFrame:
    output = frame.copy()
    decomposed = decompose_xgb_probabilities(output)
    output = output.join(decomposed)
    output["p_move_cal"] = apply_binary_logit_calibrator(
        output["p_move_raw"], calibrators["move"]
    )
    output["p_long_given_move_cal"] = apply_binary_logit_calibrator(
        output["p_long_given_move_raw"], calibrators["direction_given_move"]
    )
    output["xgb_latent_side"] = np.where(
        output["p_long_given_move_cal"] >= 0.5, 1.0, -1.0
    )
    output["xgb_direction_confidence"] = np.maximum(
        output["p_long_given_move_cal"],
        1.0 - output["p_long_given_move_cal"],
    )
    return output


def _load_prediction_paths(stage: str) -> list[Path]:
    if stage == "2024_oof":
        paths = sorted(
            (RAW_ROOT / "predictions").glob(
                f"w{XGB_WIDTH_BPS}_candidate_00_fold_*.parquet"
            )
        )
        expected = 5
    elif stage == "h1":
        paths = sorted(
            (H1_ROOT / XGB_MODEL).glob(
                f"stage_predictions/calibration_2025_*/w{XGB_WIDTH_BPS}_*.parquet"
            )
        )
        expected = 6
    elif stage == "forward":
        paths = sorted(
            (FORWARD_ROOT / XGB_MODEL).glob(
                f"stage_predictions/raw_forward/w{XGB_WIDTH_BPS}_*.parquet"
            )
        )
        expected = 1
    else:
        raise ValueError("unknown XGBoost stage")
    if len(paths) != expected:
        raise ValueError(f"{stage}: expected {expected} XGBoost files, found {len(paths)}")
    return paths


def load_xgb_panel(stage: str) -> tuple[pd.DataFrame, list[Path]]:
    paths = _load_prediction_paths(stage)
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    frame = frame.drop_duplicates("timestamp").set_index("timestamp").sort_index()
    if stage == "2024_oof" and frame.index.max() >= pd.Timestamp("2025-01-01", tz="UTC"):
        raise ValueError("XGBoost calibrators must use 2024 OOF only")
    if stage == "h1" and frame.index.max() >= pd.Timestamp("2025-07-01", tz="UTC"):
        raise ValueError("H1 XGBoost panel crossed July 2025")
    if stage == "forward" and frame.index.max() >= LOCKBOX_START:
        raise ValueError("XGBoost forward panel crossed the Q2 lockbox")
    return frame, paths


def build_addon_signal(
    union_frame: pd.DataFrame,
    xgb_frame: pd.DataFrame,
    *,
    move_threshold: float,
    direction_threshold: float,
) -> pd.Series:
    required_union = {
        "union_signal",
        "member_conflict",
        "lstm_latent_side",
        "svm_linear_latent_side",
    }
    required_xgb = {
        "p_move_cal",
        "p_long_given_move_cal",
    }
    missing_union = sorted(required_union.difference(union_frame.columns))
    missing_xgb = sorted(required_xgb.difference(xgb_frame.columns))
    if missing_union or missing_xgb:
        raise ValueError(
            f"admission fields missing: union={missing_union}, xgb={missing_xgb}"
        )
    if move_threshold not in MOVE_THRESHOLDS and not 0.0 <= move_threshold <= 1.0:
        raise ValueError("move threshold must be a probability")
    if direction_threshold not in DIRECTION_THRESHOLDS and not 0.5 <= direction_threshold <= 1.0:
        raise ValueError("direction threshold must be in [0.5, 1]")
    joined = union_frame.join(
        xgb_frame[["p_move_cal", "p_long_given_move_cal"]], how="inner"
    )
    xgb_side = pd.Series(
        np.where(joined["p_long_given_move_cal"] >= 0.5, 1.0, -1.0),
        index=joined.index,
    )
    direction_confidence = np.maximum(
        joined["p_long_given_move_cal"],
        1.0 - joined["p_long_given_move_cal"],
    )
    supported = (
        joined["lstm_latent_side"].eq(joined["svm_linear_latent_side"])
        & xgb_side.eq(joined["lstm_latent_side"])
    )
    eligible = (
        joined["union_signal"].eq(0.0)
        & ~joined["member_conflict"].astype(bool)
        & supported
        & joined["p_move_cal"].ge(float(move_threshold))
        & direction_confidence.ge(float(direction_threshold))
    )
    return xgb_side.where(eligible, 0.0).rename("xgb_addon_signal")


def combine_with_addon(union: pd.Series, addon: pd.Series) -> pd.Series:
    aligned = pd.concat([union.rename("union_signal"), addon], axis=1).fillna(0.0)
    combined = aligned["union_signal"].where(
        aligned["union_signal"].ne(0.0), aligned["xgb_addon_signal"]
    )
    return combined.rename("combined_signal")


def binary_metrics(target: pd.Series, raw: pd.Series, calibrated: pd.Series) -> list[dict[str, float | str | int]]:
    labels = np.asarray(target, dtype=int)
    rows: list[dict[str, float | str | int]] = []
    for arm, probability in (("raw", raw), ("calibrated", calibrated)):
        values = np.asarray(probability, dtype=float)
        rows.append(
            {
                "arm": arm,
                "rows": int(len(labels)),
                "prevalence": float(labels.mean()),
                "log_loss": float(log_loss(labels, values, labels=[0, 1])),
                "brier": float(np.mean((values - labels) ** 2)),
                "mean_probability": float(values.mean()),
            }
        )
    return rows


def reliability_table(
    target: pd.Series,
    probability: pd.Series,
    *,
    period: str,
    target_name: str,
    arm: str,
) -> pd.DataFrame:
    values = np.asarray(probability, dtype=float)
    labels = np.asarray(target, dtype=int)
    edges = np.linspace(0.0, 1.0, 11)
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (values >= left) & ((values <= right) if right == 1.0 else (values < right))
        if mask.any():
            rows.append(
                {
                    "period": period,
                    "target": target_name,
                    "arm": arm,
                    "bin": index,
                    "left": left,
                    "right": right,
                    "rows": int(mask.sum()),
                    "mean_probability": float(values[mask].mean()),
                    "observed_rate": float(labels[mask].mean()),
                }
            )
    return pd.DataFrame(rows)


__all__ = [
    "BinaryLogitCalibrator",
    "DIRECTION_THRESHOLDS",
    "MOVE_THRESHOLDS",
    "OUTPUT_ROOT",
    "apply_binary_logit_calibrator",
    "apply_calibrators",
    "binary_metrics",
    "build_addon_signal",
    "combine_with_addon",
    "decompose_xgb_probabilities",
    "fit_binary_logit_calibrator",
    "fit_calibrators",
    "load_xgb_panel",
    "reliability_table",
]
