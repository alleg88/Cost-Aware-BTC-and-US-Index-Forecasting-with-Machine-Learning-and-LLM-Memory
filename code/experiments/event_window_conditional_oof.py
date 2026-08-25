"""Conditional timing and severity contracts for Notebook P."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_cost_aware_oof import CostAwareFoldConfig, _partitions
from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset
from experiments.event_window_large_move_models import (
    LargeMoveModelConfig,
    fit_predict_opportunity,
)
from experiments.event_window_large_move_oof import _fit_binary_platt
from experiments.event_window_opportunity_oof import _sigmoid
from experiments.event_window_tail_oof import _half_open_uniqueness


TIMING_HORIZONS_MINUTES = (15, 30, 60, 120)


@dataclass(frozen=True)
class ConditionalOOFConfig:
    fold: CostAwareFoldConfig = field(default_factory=CostAwareFoldConfig)
    model: LargeMoveModelConfig = field(default_factory=LargeMoveModelConfig)


@dataclass(frozen=True)
class ConditionalOOFResult:
    model_name: str
    scores: pd.DataFrame
    calibration_scores: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame


def _probability_arrays(**values: np.ndarray) -> dict[str, np.ndarray]:
    arrays = {name: np.asarray(value, dtype=float) for name, value in values.items()}
    lengths = {len(value) for value in arrays.values() if value.ndim == 1}
    if any(value.ndim != 1 for value in arrays.values()) or len(lengths) != 1:
        raise ValueError("probabilities must be aligned one-dimensional arrays")
    if any(
        not np.isfinite(value).all() or ((value < 0.0) | (value > 1.0)).any()
        for value in arrays.values()
    ):
        raise ValueError("probabilities must be finite values in [0, 1]")
    return arrays


def timing_interval_index(time_to_hit_minutes: np.ndarray) -> np.ndarray:
    """Map valid 1..120 minute hits to the preregistered four intervals."""
    values = np.asarray(time_to_hit_minutes, dtype=float)
    result = np.full(values.shape, -1, dtype=np.int8)
    valid = np.isfinite(values) & (values >= 1.0) & (values <= 120.0)
    result[valid] = np.searchsorted(
        np.asarray(TIMING_HORIZONS_MINUTES, dtype=float),
        values[valid],
        side="left",
    ).astype(np.int8)
    return result


def compose_timing_probabilities(
    p_hit: np.ndarray,
    *,
    h_15: np.ndarray,
    h_30: np.ndarray,
    h_60: np.ndarray,
) -> dict[str, np.ndarray]:
    """Anchor sequential conditional hazards to frozen N3 incidence."""
    values = _probability_arrays(p_hit=p_hit, h_15=h_15, h_30=h_30, h_60=h_60)
    q_15 = values["h_15"]
    q_30 = q_15 + (1.0 - q_15) * values["h_30"]
    q_60 = q_30 + (1.0 - q_30) * values["h_60"]
    return {
        "p_t_le_15": values["p_hit"] * q_15,
        "p_t_le_30": values["p_hit"] * q_30,
        "p_t_le_60": values["p_hit"] * q_60,
        "p_t_le_120": values["p_hit"].copy(),
    }


def compose_severity_probabilities(
    p_hit: np.ndarray,
    *,
    p_ge_150_given_100: np.ndarray,
    p_ge_200_given_150: np.ndarray,
) -> dict[str, np.ndarray]:
    """Create nested 1B, 1.5B and 2B probabilities by construction."""
    values = _probability_arrays(
        p_hit=p_hit,
        p_ge_150_given_100=p_ge_150_given_100,
        p_ge_200_given_150=p_ge_200_given_150,
    )
    p_150 = values["p_hit"] * values["p_ge_150_given_100"]
    return {
        "p_ge_100": values["p_hit"].copy(),
        "p_ge_150": p_150,
        "p_ge_200": p_150 * values["p_ge_200_given_150"],
    }


def _decisions(dataset: LargeMoveDecisionDataset) -> pd.DataFrame:
    work = dataset.decisions.copy().reset_index(drop=True)
    required = {
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "label_start",
        "label_end",
        "magnitude_class",
        "magnitude_target_valid",
        "tth_100_min",
    }
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"conditional decisions missing columns: {missing}")
    if len(work) != len(dataset.tabular):
        raise ValueError("conditional decisions and features do not align")
    for column in ("decision_time", "label_start", "label_end"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    work["model_target_valid"] = work["magnitude_target_valid"].astype(bool)
    valid = work["model_target_valid"]
    if not work.loc[valid, "magnitude_class"].isin(range(5)).all():
        raise ValueError("valid conditional rows require a registered magnitude class")
    expected_end = work["decision_time"] + pd.Timedelta(minutes=120)
    if not work["label_end"].eq(expected_end).all():
        raise ValueError("all conditional labels must use fixed t+120 minute intervals")
    return work


def _weighted_prevalence(
    decisions: pd.DataFrame,
    positions: np.ndarray,
    target: np.ndarray,
) -> float:
    if not len(positions):
        raise ValueError("conditional baseline needs eligible rows")
    weights = _half_open_uniqueness(decisions, positions)
    return float(np.clip(np.average(target[positions], weights=weights), 1e-6, 1.0 - 1e-6))


def _fit_head(
    name: str,
    model_name: str,
    dataset: LargeMoveDecisionDataset,
    decisions: pd.DataFrame,
    *,
    fit: np.ndarray,
    early: np.ndarray,
    score_positions: np.ndarray,
    eligible: np.ndarray,
    target: np.ndarray,
    config: ConditionalOOFConfig,
) -> tuple[np.ndarray, float, dict[str, object]]:
    fit_rows = fit[eligible[fit]]
    early_rows = early[eligible[early]]
    if not len(fit_rows) or not len(early_rows):
        raise ValueError(f"conditional head {name} has an empty fit or calibration block")
    fit_weights = _half_open_uniqueness(decisions, fit_rows)
    baseline_rows = np.concatenate([fit_rows, early_rows])
    baseline = _weighted_prevalence(decisions, baseline_rows, target)
    fit_target = target[fit_rows].astype(np.int8)
    if np.unique(fit_target).size == 1:
        probability = np.full(len(score_positions), baseline, dtype=float)
        slope, intercept, fallback = 1.0, 0.0, True
        architecture = "constant_fallback"
    else:
        raw = fit_predict_opportunity(
            model_name,
            train_x=dataset.tabular[fit_rows],
            labels=fit_target,
            sample_weight=fit_weights,
            score_x=dataset.tabular[np.concatenate([early_rows, score_positions])],
            config=config.model,
        )
        if raw.opportunity_logit is None:
            raise AssertionError(f"conditional head {name} did not expose a binary logit")
        early_weights = _half_open_uniqueness(decisions, early_rows)
        slope, intercept, fallback = _fit_binary_platt(
            raw.opportunity_logit[: len(early_rows)],
            target[early_rows].astype(np.int8),
            early_weights,
        )
        probability = _sigmoid(
            raw.opportunity_logit[len(early_rows) :], slope, intercept
        )
        architecture = model_name
    audit = {
        "head": name,
        "model": model_name,
        "architecture": architecture,
        "fit_rows": len(fit_rows),
        "early_rows": len(early_rows),
        "fit_positive_rate": float(np.average(fit_target, weights=fit_weights)),
        "anchored_baseline_probability": baseline,
        "platt_slope": slope,
        "platt_intercept": intercept,
        "identity_or_constant_fallback": fallback,
    }
    return np.clip(probability, 1e-8, 1.0 - 1e-8), baseline, audit


def _timing_bins(h_15: np.ndarray, h_30: np.ndarray, h_60: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            h_15,
            (1.0 - h_15) * h_30,
            (1.0 - h_15) * (1.0 - h_30) * h_60,
            (1.0 - h_15) * (1.0 - h_30) * (1.0 - h_60),
        ]
    )


def _score_frame(
    model_name: str,
    fold_id: str,
    decisions: pd.DataFrame,
    positions: np.ndarray,
    p_hit: np.ndarray,
    head_probability: dict[str, np.ndarray],
    baseline: dict[str, float],
) -> pd.DataFrame:
    rows = decisions.iloc[positions].reset_index(drop=True)
    model_timing = compose_timing_probabilities(
        p_hit,
        h_15=head_probability["h15"],
        h_30=head_probability["h30"],
        h_60=head_probability["h60"],
    )
    baseline_timing = compose_timing_probabilities(
        p_hit,
        h_15=np.full(len(rows), baseline["h15"]),
        h_30=np.full(len(rows), baseline["h30"]),
        h_60=np.full(len(rows), baseline["h60"]),
    )
    model_severity = compose_severity_probabilities(
        p_hit,
        p_ge_150_given_100=head_probability["s15"],
        p_ge_200_given_150=head_probability["s20"],
    )
    baseline_severity = compose_severity_probabilities(
        p_hit,
        p_ge_150_given_100=np.full(len(rows), baseline["s15"]),
        p_ge_200_given_150=np.full(len(rows), baseline["s20"]),
    )
    model_bins = _timing_bins(
        head_probability["h15"], head_probability["h30"], head_probability["h60"]
    )
    baseline_bins = _timing_bins(
        np.full(len(rows), baseline["h15"]),
        np.full(len(rows), baseline["h30"]),
        np.full(len(rows), baseline["h60"]),
    )
    target_class = rows["magnitude_class"].to_numpy(int)
    time_to_hit = pd.to_numeric(rows["tth_100_min"], errors="coerce").to_numpy(float)
    output = pd.DataFrame(
        {
            "model": model_name,
            "fold_id": fold_id,
            "window_id": rows["window_id"],
            "channel_episode_id": rows["channel_episode_id"],
            "step": rows["step"].astype(int),
            "decision_time": rows["decision_time"],
            "label_start": rows["label_start"],
            "label_end": rows["label_end"],
            "magnitude_class": target_class,
            "magnitude_ratio": pd.to_numeric(rows["magnitude_ratio"], errors="coerce"),
            "tth_100_min": time_to_hit,
            "timing_interval": timing_interval_index(time_to_hit),
            "y_hit": (target_class >= 2).astype(np.int8),
            "y_t_le_15": ((target_class >= 2) & (time_to_hit <= 15)).astype(np.int8),
            "y_t_le_30": ((target_class >= 2) & (time_to_hit <= 30)).astype(np.int8),
            "y_t_le_60": ((target_class >= 2) & (time_to_hit <= 60)).astype(np.int8),
            "y_ge_150": (target_class >= 3).astype(np.int8),
            "y_ge_200": (target_class >= 4).astype(np.int8),
            "sample_weight": _half_open_uniqueness(decisions, positions),
            **model_timing,
            **model_severity,
        }
    )
    for horizon in TIMING_HORIZONS_MINUTES:
        output[f"p0_t_le_{horizon}"] = baseline_timing[f"p_t_le_{horizon}"]
    output["p0_ge_100"] = baseline_severity["p_ge_100"]
    output["p0_ge_150"] = baseline_severity["p_ge_150"]
    output["p0_ge_200"] = baseline_severity["p_ge_200"]
    for index, label in enumerate(("15", "30", "60", "120")):
        output[f"q_t_{label}"] = model_bins[:, index]
        output[f"q0_t_{label}"] = baseline_bins[:, index]
    return output


def run_conditional_fold(
    model_name: str,
    fold: PurgedFold,
    dataset: LargeMoveDecisionDataset,
    *,
    p_hit_by_position: np.ndarray,
    config: ConditionalOOFConfig = ConditionalOOFConfig(),
) -> ConditionalOOFResult:
    """Fit conditional heads on past rows and score reserved plus outer rows."""
    if model_name not in {"logreg", "xgboost"}:
        raise ValueError("conditional model must be logreg or xgboost")
    decisions = _decisions(dataset)
    p_hit = np.asarray(p_hit_by_position, dtype=float)
    if p_hit.shape != (len(decisions),):
        raise ValueError("p_hit_by_position must align with the dataset")
    fit, early, reserved = _partitions(decisions, fold.train, config.fold)
    outer = np.asarray(fold.valid, dtype=np.int64)
    outer = outer[decisions.iloc[outer]["model_target_valid"].to_numpy(bool)]
    score_positions = np.concatenate([reserved, outer])
    if not np.isfinite(p_hit[score_positions]).all():
        raise ValueError("frozen N3 probabilities are required for reserved and outer rows")
    target_class = decisions["magnitude_class"].to_numpy(int)
    time_to_hit = pd.to_numeric(decisions["tth_100_min"], errors="coerce").to_numpy(float)
    hit = target_class >= 2
    specifications = {
        "h15": (hit, hit & (time_to_hit <= 15)),
        "h30": (hit & (time_to_hit > 15), hit & (time_to_hit > 15) & (time_to_hit <= 30)),
        "h60": (hit & (time_to_hit > 30), hit & (time_to_hit > 30) & (time_to_hit <= 60)),
        "s15": (hit, target_class >= 3),
        "s20": (target_class >= 3, target_class >= 4),
    }
    probabilities: dict[str, np.ndarray] = {}
    baselines: dict[str, float] = {}
    audits = []
    for name, (eligible, target) in specifications.items():
        probability, baseline, audit = _fit_head(
            name,
            model_name,
            dataset,
            decisions,
            fit=fit,
            early=early,
            score_positions=score_positions,
            eligible=np.asarray(eligible, dtype=bool),
            target=np.asarray(target, dtype=np.int8),
            config=config,
        )
        probabilities[name] = probability
        baselines[name] = baseline
        audits.append({"fold": fold.fold_id, **audit})
    split = len(reserved)
    calibration_scores = _score_frame(
        model_name,
        fold.fold_id,
        decisions,
        reserved,
        p_hit[reserved],
        {name: value[:split] for name, value in probabilities.items()},
        baselines,
    )
    scores = _score_frame(
        model_name,
        fold.fold_id,
        decisions,
        outer,
        p_hit[outer],
        {name: value[split:] for name, value in probabilities.items()},
        baselines,
    )
    train_positions = np.concatenate([fit, early, reserved])
    train_episodes = set(decisions.iloc[train_positions]["channel_episode_id"])
    outer_episodes = set(decisions.iloc[outer]["channel_episode_id"])
    timing_order = scores[[f"p_t_le_{h}" for h in TIMING_HORIZONS_MINUTES]].to_numpy(float)
    severity_order = scores[["p_ge_200", "p_ge_150", "p_ge_100"]].to_numpy(float)
    fold_audit = pd.DataFrame(
        [
            {
                "model": model_name,
                "fold": fold.fold_id,
                "feature_set": dataset.feature_set,
                "features": len(dataset.tabular_features),
                "fit_rows": len(fit),
                "early_rows": len(early),
                "reserved_calibration_rows": len(reserved),
                "validation_rows": len(outer),
                "fit_episodes": decisions.iloc[fit]["channel_episode_id"].nunique(),
                "early_episodes": decisions.iloc[early]["channel_episode_id"].nunique(),
                "reserved_calibration_episodes": decisions.iloc[reserved]["channel_episode_id"].nunique(),
                "validation_episodes": decisions.iloc[outer]["channel_episode_id"].nunique(),
                "episode_overlap": len(train_episodes & outer_episodes),
                "train_label_end_max": decisions.iloc[train_positions]["label_end"].max(),
                "validation_start": fold.valid_start,
                "p120_identity_max_abs": float(np.max(np.abs(scores["p_t_le_120"] - p_hit[outer]))),
                "timing_monotonic_violations": int((np.diff(timing_order, axis=1) < -1e-12).any(axis=1).sum()),
                "severity_monotonic_violations": int((np.diff(severity_order, axis=1) < -1e-12).any(axis=1).sum()),
            }
        ]
    )
    if fold_audit.iloc[0].episode_overlap != 0:
        raise AssertionError("conditional OOF split channel episodes")
    if fold_audit.iloc[0].train_label_end_max > fold.valid_start:
        raise AssertionError("conditional training label crosses validation")
    if fold_audit.iloc[0].timing_monotonic_violations:
        raise AssertionError("conditional timing probabilities are not nested")
    if fold_audit.iloc[0].severity_monotonic_violations:
        raise AssertionError("conditional severity probabilities are not nested")
    return ConditionalOOFResult(
        model_name=model_name,
        scores=scores,
        calibration_scores=calibration_scores,
        fold_audit=fold_audit,
        calibration_audit=pd.DataFrame(audits),
    )


__all__ = [
    "ConditionalOOFConfig",
    "ConditionalOOFResult",
    "TIMING_HORIZONS_MINUTES",
    "compose_severity_probabilities",
    "compose_timing_probabilities",
    "run_conditional_fold",
    "timing_interval_index",
]
