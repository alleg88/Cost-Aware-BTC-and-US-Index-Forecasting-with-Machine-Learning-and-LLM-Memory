"""Fresh purged OOF predictions for Notebook L cost-aware entry."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from evaluation.channel_window_validation import PurgedFold, VALID_BLOCKS
from experiments.event_window_cost_aware_dataset import (
    CostAwareDecisionDataset,
    MakerExecutionConfig,
)
from experiments.event_window_cost_aware_models import (
    CostAwareModelConfig,
    fit_predict_cost_aware_model,
)
from experiments.event_window_tail_oof import _half_open_uniqueness


@dataclass(frozen=True)
class CostAwareFoldConfig:
    fit_fraction: float = 0.70
    early_fraction: float = 0.15
    calibration_fraction: float = 0.15


@dataclass(frozen=True)
class CostAwareOOFConfig:
    fold: CostAwareFoldConfig = field(default_factory=CostAwareFoldConfig)
    model: CostAwareModelConfig = field(default_factory=CostAwareModelConfig)
    execution: MakerExecutionConfig = field(default_factory=MakerExecutionConfig)
    confidence_z: float = 1.645


@dataclass(frozen=True)
class CostAwareOOFResult:
    model_name: str
    scores: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame


def _outer_folds(decisions: pd.DataFrame) -> list[PurgedFold]:
    groups = decisions.groupby("channel_episode_id", sort=False).agg(
        group_start=("decision_time", "min"),
        group_decision_end=("decision_time", "max"),
    )
    valid = decisions["model_target_valid"].astype(bool)
    training_end = decisions.loc[valid].groupby("channel_episode_id")["label_end"].max()
    folds: list[PurgedFold] = []
    for raw_start, raw_end in VALID_BLOCKS:
        start, end = pd.Timestamp(raw_start), pd.Timestamp(raw_end)
        train_episodes = groups.index[
            groups.index.isin(training_end.index)
            & groups["group_decision_end"].lt(start)
            & training_end.reindex(groups.index).le(start).fillna(False)
        ]
        valid_episodes = groups.index[
            groups["group_start"].ge(start)
            & groups["group_decision_end"].lt(end)
            & training_end.reindex(groups.index).lt(end).fillna(False)
        ]
        train = np.flatnonzero(
            decisions["channel_episode_id"].isin(train_episodes).to_numpy()
            & decisions["decision_time"].lt(start).to_numpy()
        )
        outer = np.flatnonzero(
            decisions["channel_episode_id"].isin(valid_episodes).to_numpy()
            & decisions["decision_time"].ge(start).to_numpy()
            & decisions["decision_time"].lt(end).to_numpy()
        )
        if not len(train) or not len(outer):
            continue
        folds.append(
            PurgedFold(
                fold_id=f"{start.year}H{1 if start.month == 1 else 2}",
                train=train,
                valid=outer,
                train_end=start,
                valid_start=start,
                valid_end=end,
            )
        )
    return folds


def _partitions(
    decisions: pd.DataFrame,
    outer_train: np.ndarray,
    config: CostAwareFoldConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    positions = np.asarray(outer_train, dtype=np.int64)
    positions = positions[decisions.iloc[positions]["model_target_valid"].to_numpy(bool)]
    selected = decisions.iloc[positions]
    episodes = (
        selected.groupby("channel_episode_id", sort=False)["decision_time"]
        .max()
        .rename("last_decision")
        .reset_index()
    )
    episodes["sort_id"] = episodes["channel_episode_id"].astype(str)
    episodes = episodes.sort_values(["last_decision", "sort_id"], kind="stable")
    count = len(episodes)
    fit_count = max(1, int(np.floor(config.fit_fraction * count)))
    early_count = max(1, int(np.floor(config.early_fraction * count)))
    if fit_count + early_count >= count:
        fit_count = max(1, count - early_count - 1)
    fit_ids = set(episodes.iloc[:fit_count]["channel_episode_id"])
    early_ids = set(episodes.iloc[fit_count : fit_count + early_count]["channel_episode_id"])
    calibration_ids = set(episodes.iloc[fit_count + early_count :]["channel_episode_id"])
    fit = positions[selected["channel_episode_id"].isin(fit_ids).to_numpy()]
    early = positions[selected["channel_episode_id"].isin(early_ids).to_numpy()]
    calibration = positions[selected["channel_episode_id"].isin(calibration_ids).to_numpy()]
    if not len(fit) or not len(early) or not len(calibration):
        raise ValueError("fit, early and calibration partitions must be non-empty")
    early_start = decisions.iloc[early]["decision_time"].min()
    calibration_start = decisions.iloc[calibration]["decision_time"].min()
    fit = fit[decisions.iloc[fit]["label_end"].le(early_start).to_numpy()]
    early = early[decisions.iloc[early]["label_end"].le(calibration_start).to_numpy()]
    if not len(fit) or not len(early):
        raise ValueError("purging removed a complete inner partition")
    return fit, early, calibration


def _fit_temperature(logits: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=int)
    w = np.asarray(weights, dtype=float)

    def objective(log_temperature: float) -> float:
        scaled = values / np.exp(log_temperature)
        shifted = scaled - scaled.max(axis=1, keepdims=True)
        log_norm = np.log(np.exp(shifted).sum(axis=1))
        return float(np.average(log_norm - shifted[np.arange(len(y)), y], weights=w))

    result = minimize_scalar(objective, bounds=(-4.0, 4.0), method="bounded")
    if not result.success:
        raise RuntimeError("temperature calibration failed")
    return float(np.exp(result.x))


def _probabilities(logits: np.ndarray, temperature: float) -> np.ndarray:
    scaled = np.asarray(logits, dtype=float) / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _calibration_bias_and_buffer(
    predicted: np.ndarray,
    realised: np.ndarray,
    weights: np.ndarray,
    episodes: np.ndarray,
    confidence_z: float,
) -> tuple[float, float, int]:
    residual = np.asarray(realised, dtype=float) - np.asarray(predicted, dtype=float)
    weight = np.asarray(weights, dtype=float)
    bias = float(np.average(residual, weights=weight))
    frame = pd.DataFrame({"episode": episodes, "residual": residual, "weight": weight})
    episode_mean = frame.groupby("episode", sort=False).apply(
        lambda group: np.average(group["residual"], weights=group["weight"]),
        include_groups=False,
    )
    episode_count = int(len(episode_mean))
    standard_error = (
        float(episode_mean.std(ddof=1) / np.sqrt(episode_count))
        if episode_count > 1
        else 0.0
    )
    return bias, float(max(0.0, confidence_z * standard_error)), episode_count


def _ev_components(
    probabilities: np.ndarray,
    timeout_gross: np.ndarray,
    risk_bps: np.ndarray,
    execution: MakerExecutionConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_unfilled, p_sl, p_tp, p_timeout = probabilities.T
    gross = -p_sl + execution.rr_multiple * p_tp + p_timeout * timeout_gross
    fee_bps = (
        p_sl * execution.round_trip_bps("sl")
        + p_tp * execution.round_trip_bps("tp")
        + p_timeout * execution.round_trip_bps("timeout")
    )
    fee_r = fee_bps / np.asarray(risk_bps, dtype=float)
    return gross, fee_r, gross - fee_r


def _fit_hash(x: np.ndarray, y: np.ndarray, weights: np.ndarray, model: str) -> str:
    digest = hashlib.sha256(model.encode("utf-8"))
    digest.update(np.ascontiguousarray(x).view(np.uint8))
    digest.update(np.asarray(y, dtype=np.int8).tobytes())
    digest.update(np.asarray(weights, dtype=np.float64).tobytes())
    return digest.hexdigest()


def run_cost_aware_model_oof(
    model_name: str,
    dataset: CostAwareDecisionDataset,
    config: CostAwareOOFConfig = CostAwareOOFConfig(),
) -> CostAwareOOFResult:
    if model_name not in {"logreg", "xgboost"}:
        raise ValueError("model_name must be logreg or xgboost")
    decisions = dataset.decisions.copy().reset_index(drop=True)
    for column in ("decision_time", "label_start", "label_end"):
        decisions[column] = pd.to_datetime(decisions[column], utc=True, errors="coerce")
    score_frames: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    for fold in _outer_folds(decisions):
        fit, early, calibration = _partitions(decisions, fold.train, config.fold)
        outer = np.asarray(fold.valid, dtype=np.int64)
        if not len(outer):
            raise ValueError(f"fold {fold.fold_id} has no outer rows")
        fit_weights = _half_open_uniqueness(decisions, fit)
        combined = np.concatenate([calibration, outer])
        raw = fit_predict_cost_aware_model(
            model_name,
            train_x=dataset.tabular[fit],
            outcome=decisions.iloc[fit]["outcome_code"].to_numpy(dtype=int),
            timeout_gross_target=decisions.iloc[fit]["timeout_gross_r"].to_numpy(float),
            advantage_target=decisions.iloc[fit]["enter_advantage_target"].to_numpy(float),
            advantage_valid=decisions.iloc[fit]["advantage_valid"].to_numpy(bool),
            sample_weight=fit_weights,
            score_x=dataset.tabular[combined],
            config=config.model,
        )
        split = len(calibration)
        calibration_weights = _half_open_uniqueness(decisions, calibration)
        temperature = _fit_temperature(
            raw.logits[:split],
            decisions.iloc[calibration]["outcome_code"].to_numpy(int),
            calibration_weights,
        )
        calibration_prob = _probabilities(raw.logits[:split], temperature)
        outer_prob = _probabilities(raw.logits[split:], temperature)

        fit_timeout = decisions.iloc[fit]["outcome"].eq("timeout").to_numpy()
        fit_timeout_values = decisions.iloc[fit].loc[fit_timeout, "timeout_gross_r"].to_numpy(float)
        low, high = np.quantile(fit_timeout_values, [0.01, 0.99])
        calibration_timeout = decisions.iloc[calibration]["outcome"].eq("timeout").to_numpy()
        timeout_bias = (
            float(
                np.average(
                    decisions.iloc[calibration].loc[calibration_timeout, "timeout_gross_r"].to_numpy(float)
                    - raw.timeout_gross_r[:split][calibration_timeout],
                    weights=calibration_weights[calibration_timeout],
                )
            )
            if calibration_timeout.any()
            else 0.0
        )
        calibration_timeout_prediction = np.clip(
            raw.timeout_gross_r[:split] + timeout_bias, low, high
        )
        outer_timeout_prediction = np.clip(raw.timeout_gross_r[split:] + timeout_bias, low, high)
        calibration_decisions = decisions.iloc[calibration]
        calibration_gross, calibration_fee, calibration_net = _ev_components(
            calibration_prob,
            calibration_timeout_prediction,
            calibration_decisions["risk_bps"].to_numpy(float),
            config.execution,
        )
        net_bias, net_buffer, net_episodes = _calibration_bias_and_buffer(
            calibration_net,
            calibration_decisions["r_net"].to_numpy(float),
            calibration_weights,
            calibration_decisions["channel_episode_id"].to_numpy(),
            config.confidence_z,
        )
        advantage_mask = calibration_decisions["advantage_valid"].to_numpy(bool)
        advantage_bias = (
            float(
                np.average(
                    calibration_decisions.loc[advantage_mask, "enter_advantage_target"].to_numpy(float)
                    - raw.enter_advantage[:split][advantage_mask],
                    weights=calibration_weights[advantage_mask],
                )
            )
            if advantage_mask.any()
            else 0.0
        )
        outer_decisions = decisions.iloc[outer].reset_index(drop=True)
        gross_ev, expected_fee_r, raw_net_ev = _ev_components(
            outer_prob,
            outer_timeout_prediction,
            outer_decisions["risk_bps"].to_numpy(float),
            config.execution,
        )
        calibrated_net_ev = raw_net_ev + net_bias
        conservative_net_ev = calibrated_net_ev - net_buffer
        advantage = raw.enter_advantage[split:] + advantage_bias
        score_frames.append(
            pd.DataFrame(
                {
                    "model": model_name,
                    "fold_id": fold.fold_id,
                    "window_id": outer_decisions["window_id"],
                    "channel_episode_id": outer_decisions["channel_episode_id"],
                    "side": outer_decisions["side"],
                    "step": outer_decisions["step"].astype(int),
                    "decision_time": outer_decisions["decision_time"],
                    "p_unfilled": outer_prob[:, 0],
                    "p_sl": outer_prob[:, 1],
                    "p_tp": outer_prob[:, 2],
                    "p_timeout": outer_prob[:, 3],
                    "timeout_gross_r_pred": outer_timeout_prediction,
                    "gross_ev": gross_ev,
                    "expected_fee_r": expected_fee_r,
                    "raw_net_ev": raw_net_ev,
                    "calibrated_net_ev": calibrated_net_ev,
                    "calibration_buffer": net_buffer,
                    "conservative_net_ev": conservative_net_ev,
                    "enter_advantage_vs_wait": advantage,
                }
            )
        )
        train_episodes = set(decisions.iloc[np.concatenate([fit, early, calibration])]["channel_episode_id"])
        outer_episodes = set(outer_decisions["channel_episode_id"])
        fold_rows.append(
            {
                "model": model_name,
                "fold": fold.fold_id,
                "fit_rows": len(fit),
                "early_rows": len(early),
                "calibration_rows": len(calibration),
                "validation_rows": len(outer),
                "episode_overlap": len(train_episodes & outer_episodes),
                "fit_model_hash": _fit_hash(
                    dataset.tabular[fit],
                    decisions.iloc[fit]["outcome_code"].to_numpy(int),
                    fit_weights,
                    model_name,
                ),
            }
        )
        calibration_rows.append(
            {
                "model": model_name,
                "fold": fold.fold_id,
                "temperature": temperature,
                "timeout_bias": timeout_bias,
                "net_ev_bias": net_bias,
                "calibration_buffer": net_buffer,
                "calibration_episodes": net_episodes,
                "advantage_bias": advantage_bias,
                "calibration_gross_ev_mean": float(np.mean(calibration_gross)),
                "calibration_expected_fee_r_mean": float(np.mean(calibration_fee)),
            }
        )
    scores = pd.concat(score_frames, ignore_index=True)
    if scores.duplicated(["window_id", "step"]).any():
        raise AssertionError("OOF keys overlap between folds")
    return CostAwareOOFResult(
        model_name,
        scores,
        pd.DataFrame(fold_rows),
        pd.DataFrame(calibration_rows),
    )


def assert_identical_cost_aware_keys(results: list[CostAwareOOFResult]) -> None:
    if not results:
        raise ValueError("results cannot be empty")
    reference = set(results[0].scores[["window_id", "step"]].itertuples(index=False, name=None))
    for result in results[1:]:
        keys = set(result.scores[["window_id", "step"]].itertuples(index=False, name=None))
        if keys != reference:
            raise AssertionError("LogReg and XGBoost OOF keys differ")


__all__ = [
    "CostAwareOOFConfig",
    "CostAwareOOFResult",
    "assert_identical_cost_aware_keys",
    "run_cost_aware_model_oof",
]
