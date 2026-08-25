"""Common expanding purged OOF orchestration for event-window tail models."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json

import numpy as np
import pandas as pd

from evaluation.channel_window_validation import PurgedFold, VALID_BLOCKS
from experiments.event_window_tail_calibration import (
    TimeoutCalibration,
    apply_temperature,
    apply_timeout_bias,
    expected_net_r,
    fit_temperature,
    fit_timeout_bias,
)
from experiments.event_window_tail_dataset import TailDecisionDataset
from experiments.event_window_tail_neural import (
    NeuralTailPrediction,
    TailNeuralConfig,
    TailNeuralTensors,
    fit_predict_neural_tail_model,
)
from experiments.event_window_tail_tabular import (
    LogRegTailConfig,
    TabularTailPrediction,
    XGBoostTailConfig,
    fit_predict_tail_model,
)


_MODEL_NAMES = frozenset({"logreg", "xgboost", "tcn", "gru"})
_KEY_COLUMNS = ["window_id", "step"]
_TIME_COLUMNS = ("source_bar_time", "decision_time", "label_start", "label_end")
_REFERENCE_RATE = 1_448 / 1_277


@dataclass(frozen=True)
class TailFoldConfig:
    fit_fraction: float = 0.70
    early_fraction: float = 0.15
    calibration_fraction: float = 0.15

    def __post_init__(self) -> None:
        values = np.asarray(
            [self.fit_fraction, self.early_fraction, self.calibration_fraction],
            dtype=float,
        )
        if not np.isfinite(values).all() or (values <= 0.0).any():
            raise ValueError("fold fractions must be positive and finite")
        if not np.isclose(values.sum(), 1.0):
            raise ValueError("fold fractions must sum to one")


@dataclass(frozen=True)
class TailOOFConfig:
    fold: TailFoldConfig = field(default_factory=TailFoldConfig)
    logreg: LogRegTailConfig = field(default_factory=LogRegTailConfig)
    xgboost: XGBoostTailConfig = field(default_factory=XGBoostTailConfig)
    neural: TailNeuralConfig = field(default_factory=TailNeuralConfig)
    random_seed: int = 42


@dataclass(frozen=True)
class TailInnerPartitions:
    fit: np.ndarray
    early: np.ndarray
    calibration: np.ndarray


@dataclass(frozen=True)
class TailOOFResult:
    model_name: str
    scores: pd.DataFrame
    fold_audit: pd.DataFrame
    calibration_audit: pd.DataFrame


@dataclass(frozen=True)
class _NeuralPack:
    tensors: TailNeuralTensors
    local_windows: np.ndarray
    steps: np.ndarray


def _validate_model_name(model_name: str) -> None:
    if model_name not in _MODEL_NAMES:
        raise ValueError(
            f"unsupported model_name: {model_name}; expected one of {sorted(_MODEL_NAMES)}"
        )


def _validated_decisions(dataset: TailDecisionDataset) -> pd.DataFrame:
    decisions = dataset.decisions.copy().reset_index(drop=True)
    required = {
        "window_id",
        "channel_episode_id",
        "side",
        "step",
        "decision_time",
        "label_start",
        "label_end",
        "model_target_valid",
        "outcome_code",
        "timeout_net_r",
        "tp_net_r",
        "sl_net_r",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise ValueError(f"tail decisions missing OOF columns: {missing}")
    if len(decisions) != len(dataset.tabular):
        raise ValueError("tail decisions and tabular rows must align")
    if decisions.duplicated(_KEY_COLUMNS).any():
        raise ValueError("tail decision keys must be unique")
    decisions["step"] = pd.to_numeric(decisions["step"], errors="raise").astype(int)
    for column in _TIME_COLUMNS:
        if column in decisions:
            decisions[column] = pd.to_datetime(
                decisions[column], utc=True, errors="coerce"
            )
    if decisions["decision_time"].isna().any():
        raise ValueError("decision_time cannot be missing")
    target_valid = decisions["model_target_valid"].fillna(False).astype(bool)
    target_intervals = decisions.loc[target_valid, ["label_start", "label_end"]]
    if target_intervals.isna().any().any():
        raise ValueError("training label intervals cannot be missing")
    if (target_intervals["label_end"] < target_intervals["label_start"]).any():
        raise ValueError("label_end cannot precede label_start")
    if decisions[["window_id", "channel_episode_id", "side"]].isna().any().any():
        raise ValueError("OOF identity columns cannot be missing")
    return decisions


def _outer_folds(decisions: pd.DataFrame) -> list[PurgedFold]:
    groups = decisions.groupby("channel_episode_id", sort=False).agg(
        group_start=("decision_time", "min"),
        group_decision_end=("decision_time", "max"),
    )
    target_valid = decisions["model_target_valid"].fillna(False).astype(bool)
    training_groups = decisions.loc[target_valid].groupby(
        "channel_episode_id", sort=False
    )["label_end"].max()
    groups["group_label_end"] = training_groups.reindex(groups.index)
    folds: list[PurgedFold] = []
    for raw_start, raw_end in VALID_BLOCKS:
        valid_start = pd.Timestamp(raw_start)
        valid_end = pd.Timestamp(raw_end)
        train_episodes = groups.index[
            groups.index.isin(training_groups.index)
            & groups["group_decision_end"].lt(valid_start)
            & training_groups.reindex(groups.index).le(valid_start).fillna(False)
        ]
        valid_episodes = groups.index[
            groups["group_start"].ge(valid_start)
            & groups["group_decision_end"].lt(valid_end)
            & groups["group_label_end"].lt(valid_end).fillna(False)
        ]
        train = np.flatnonzero(
            decisions["channel_episode_id"].isin(train_episodes).to_numpy()
            & decisions["decision_time"].lt(valid_start).to_numpy()
        )
        valid = np.flatnonzero(
            decisions["channel_episode_id"].isin(valid_episodes).to_numpy()
            & decisions["decision_time"].ge(valid_start).to_numpy()
            & decisions["decision_time"].lt(valid_end).to_numpy()
        )
        fold_id = f"{valid_start.year}H{1 if valid_start.month == 1 else 2}"
        folds.append(
            PurgedFold(
                fold_id=fold_id,
                train=train,
                valid=valid,
                train_end=valid_start,
                valid_start=valid_start,
                valid_end=valid_end,
            )
        )
    return folds


def _partition_counts(episode_count: int, config: TailFoldConfig) -> tuple[int, int]:
    if episode_count < 3:
        raise ValueError("outer training needs at least three complete episodes")
    fit_count = max(1, int(np.floor(config.fit_fraction * episode_count)))
    early_count = max(1, int(np.floor(config.early_fraction * episode_count)))
    if fit_count + early_count >= episode_count:
        excess = fit_count + early_count - episode_count + 1
        fit_count = max(1, fit_count - excess)
    return fit_count, early_count


def _purge_live_labels(
    decisions: pd.DataFrame,
    positions: np.ndarray,
    next_start: pd.Timestamp,
) -> np.ndarray:
    if positions.size == 0:
        return positions
    safe = decisions.iloc[positions]["label_end"].le(next_start).to_numpy()
    return positions[safe]


def chronological_inner_partitions(
    decisions: pd.DataFrame,
    outer_train_ids: np.ndarray,
    config: TailFoldConfig,
) -> TailInnerPartitions:
    """Split complete episodes chronologically and purge half-open live labels."""
    positions = np.asarray(outer_train_ids, dtype=np.int64)
    if positions.ndim != 1:
        raise ValueError("outer_train_ids must be one-dimensional")
    if positions.size == 0:
        raise ValueError("outer training rows cannot be empty")
    if positions.min() < 0 or positions.max() >= len(decisions):
        raise IndexError("outer_train_ids contains an out-of-range row")
    selected = decisions.iloc[positions]
    target_valid = selected["model_target_valid"].fillna(False).astype(bool).to_numpy(copy=True)
    outcome = pd.to_numeric(selected["outcome_code"], errors="coerce").to_numpy()
    target_valid &= np.isin(outcome, (0, 1, 2))
    positions = positions[target_valid]
    selected = decisions.iloc[positions]
    if positions.size == 0:
        raise ValueError("outer training has no valid model targets")

    episode_order = (
        selected.groupby("channel_episode_id", sort=False)["decision_time"]
        .max()
        .rename("last_decision")
        .reset_index()
    )
    episode_order["episode_sort"] = episode_order["channel_episode_id"].astype(str)
    episode_order = episode_order.sort_values(
        ["last_decision", "episode_sort"], kind="stable"
    ).reset_index(drop=True)
    fit_count, early_count = _partition_counts(len(episode_order), config)
    fit_episodes = set(episode_order.iloc[:fit_count]["channel_episode_id"])
    early_episodes = set(
        episode_order.iloc[fit_count : fit_count + early_count]["channel_episode_id"]
    )
    calibration_episodes = set(
        episode_order.iloc[fit_count + early_count :]["channel_episode_id"]
    )
    fit = positions[selected["channel_episode_id"].isin(fit_episodes).to_numpy()]
    early = positions[selected["channel_episode_id"].isin(early_episodes).to_numpy()]
    calibration = positions[
        selected["channel_episode_id"].isin(calibration_episodes).to_numpy()
    ]
    if not fit.size or not early.size or not calibration.size:
        raise ValueError("fit, early, and calibration partitions must be non-empty")
    early_start = decisions.iloc[early]["decision_time"].min()
    calibration_start = decisions.iloc[calibration]["decision_time"].min()
    fit = _purge_live_labels(decisions, fit, early_start)
    early = _purge_live_labels(decisions, early, calibration_start)
    if not fit.size or not early.size:
        raise ValueError("purging removed an entire fit or early partition")
    return TailInnerPartitions(fit=fit, early=early, calibration=calibration)


def _half_open_uniqueness(decisions: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    selected = decisions.iloc[np.asarray(positions, dtype=np.int64)]
    start = pd.to_datetime(selected["label_start"], utc=True, errors="raise")
    end = pd.to_datetime(selected["label_end"], utc=True, errors="raise")
    if start.isna().any() or end.isna().any() or (end <= start).any():
        raise ValueError("training labels need positive half-open intervals")
    boundaries = np.unique(
        np.concatenate(
            [start.to_numpy(dtype="datetime64[ns]"), end.to_numpy(dtype="datetime64[ns]")]
        )
    )
    if len(boundaries) < 2:
        return np.ones(len(selected), dtype=float)
    left = np.searchsorted(boundaries, start.to_numpy(dtype="datetime64[ns]"))
    right = np.searchsorted(boundaries, end.to_numpy(dtype="datetime64[ns]"))
    difference = np.zeros(len(boundaries) + 1, dtype=np.int64)
    np.add.at(difference, left, 1)
    np.add.at(difference, right, -1)
    concurrency = np.cumsum(difference[:-1])
    widths = np.diff(boundaries).astype("timedelta64[ns]").astype(np.float64)
    weights = np.empty(len(selected), dtype=float)
    for row, (begin, finish) in enumerate(zip(left, right, strict=True)):
        active = concurrency[begin:finish]
        span = widths[begin:finish]
        if not active.size or (active <= 0).any() or span.sum() <= 0.0:
            raise AssertionError("invalid half-open label concurrency")
        weights[row] = float(np.average(1.0 / active, weights=span))
    return weights / weights.mean()


def _neural_pack(
    dataset: TailDecisionDataset,
    decisions: pd.DataFrame,
    positions: np.ndarray,
    weights: np.ndarray | None,
    *,
    expose_targets: bool,
) -> _NeuralPack:
    positions = np.asarray(positions, dtype=np.int64)
    selected = decisions.iloc[positions]
    metadata = dataset.sequences.metadata.reset_index(drop=True)
    metadata_lookup = {
        window_id: row for row, window_id in enumerate(metadata["window_id"].tolist())
    }
    window_ids = list(dict.fromkeys(selected["window_id"].tolist()))
    try:
        global_windows = np.asarray([metadata_lookup[value] for value in window_ids], dtype=np.int64)
    except KeyError as exc:
        raise ValueError("decision window_id is absent from sequence metadata") from exc
    local_lookup = {value: row for row, value in enumerate(window_ids)}
    local_windows = selected["window_id"].map(local_lookup).to_numpy(dtype=np.int64)
    steps = selected["step"].to_numpy(dtype=np.int64)
    active_bars = dataset.sequences.context.shape[1]
    if (steps < 0).any() or (steps >= active_bars).any():
        raise ValueError("decision step is outside the active sequence")
    shape = (len(window_ids), active_bars)
    decision_valid = np.zeros(shape, dtype=bool)
    outcome = np.full(shape, -1, dtype=np.int64)
    timeout_target = np.full(shape, np.nan, dtype=np.float32)
    uniqueness = np.zeros(shape, dtype=np.float32)
    decision_valid[local_windows, steps] = True
    if expose_targets:
        outcome[local_windows, steps] = selected["outcome_code"].to_numpy(dtype=np.int64)
        timeout_target[local_windows, steps] = selected["timeout_net_r"].to_numpy(
            dtype=np.float32
        )
    assigned_weights = np.ones(len(positions), dtype=np.float32)
    if weights is not None:
        assigned_weights = np.asarray(weights, dtype=np.float32)
        if assigned_weights.shape != (len(positions),):
            raise ValueError("neural uniqueness weights do not align")
    uniqueness[local_windows, steps] = assigned_weights
    tensors = TailNeuralTensors(
        sequence=np.asarray(dataset.sequences.sequence)[global_windows],
        context=np.asarray(dataset.sequences.context)[global_windows],
        sequence_valid=np.asarray(dataset.sequences.sequence_valid)[global_windows],
        decision_valid=decision_valid,
        outcome=outcome,
        timeout_target=timeout_target,
        uniqueness=uniqueness,
    )
    return _NeuralPack(tensors=tensors, local_windows=local_windows, steps=steps)


def _fit_hash(
    model_name: str,
    dataset: TailDecisionDataset,
    decisions: pd.DataFrame,
    fit: np.ndarray,
    weights: np.ndarray,
    config: TailOOFConfig,
) -> str:
    digest = hashlib.sha256()
    digest.update(model_name.encode("utf-8"))
    digest.update(np.ascontiguousarray(dataset.tabular[fit]).view(np.uint8))
    digest.update(decisions.iloc[fit]["outcome_code"].to_numpy(dtype=np.int8).tobytes())
    digest.update(np.asarray(weights, dtype=np.float64).tobytes())
    model_config = (
        config.logreg
        if model_name == "logreg"
        else config.xgboost
        if model_name == "xgboost"
        else config.neural
    )
    digest.update(json.dumps(asdict(model_config), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _fit_predict_raw(
    model_name: str,
    dataset: TailDecisionDataset,
    decisions: pd.DataFrame,
    partitions: TailInnerPartitions,
    outer: np.ndarray,
    config: TailOOFConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    fit = partitions.fit
    calibration = partitions.calibration
    combined = np.concatenate([calibration, outer])
    weights = _half_open_uniqueness(decisions, fit)
    fit_hash = _fit_hash(model_name, dataset, decisions, fit, weights, config)
    if model_name in {"logreg", "xgboost"}:
        prediction: TabularTailPrediction = fit_predict_tail_model(
            model_name,
            train_x=dataset.tabular[fit],
            outcome=decisions.iloc[fit]["outcome_code"].to_numpy(dtype=np.int64),
            timeout_target=decisions.iloc[fit]["timeout_net_r"].to_numpy(dtype=float),
            sample_weight=weights,
            score_x=dataset.tabular[combined],
            episode_ids=decisions.iloc[fit]["channel_episode_id"].to_numpy(),
            config=config.logreg if model_name == "logreg" else config.xgboost,
        )
        logits = np.asarray(prediction.logits, dtype=float)
        timeout_prediction = np.asarray(prediction.timeout_net_r, dtype=float)
    else:
        fit_pack = _neural_pack(
            dataset, decisions, fit, weights, expose_targets=True
        )
        early_pack = _neural_pack(
            dataset, decisions, partitions.early, None, expose_targets=True
        )
        score_pack = _neural_pack(
            dataset, decisions, combined, None, expose_targets=False
        )
        prediction: NeuralTailPrediction = fit_predict_neural_tail_model(
            model_name,
            fit_pack.tensors,
            early_pack.tensors,
            score_pack.tensors,
            config.neural,
        )
        logits = np.asarray(prediction.logits, dtype=float)[
            score_pack.local_windows, score_pack.steps
        ]
        timeout_prediction = np.asarray(prediction.timeout_net_r, dtype=float)[
            score_pack.local_windows, score_pack.steps
        ]
    if logits.shape != (len(combined), 3):
        raise AssertionError("tail model returned misaligned three-class logits")
    if timeout_prediction.shape != (len(combined),):
        raise AssertionError("tail model returned misaligned timeout predictions")
    if not np.isfinite(logits).all() or not np.isfinite(timeout_prediction).all():
        raise AssertionError("tail model returned non-finite predictions")
    calibration_rows = len(calibration)
    return (
        logits[:calibration_rows],
        timeout_prediction[:calibration_rows],
        logits[calibration_rows:],
        timeout_prediction[calibration_rows:],
        fit_hash,
    )


def _timeout_calibration(
    decisions: pd.DataFrame,
    partitions: TailInnerPartitions,
    calibration_prediction: np.ndarray,
) -> tuple[TimeoutCalibration, int, int]:
    fit_rows = decisions.iloc[partitions.fit]
    fit_timeout = fit_rows["outcome_code"].eq(2).to_numpy()
    timeout_rows = int(fit_timeout.sum())
    timeout_episodes = int(
        fit_rows.loc[fit_timeout, "channel_episode_id"].nunique()
    )
    fit_weights = _half_open_uniqueness(decisions, partitions.fit)
    if timeout_rows == 0:
        raise ValueError("fit partition needs observed timeout outcomes")
    timeout_targets = fit_rows.loc[fit_timeout, "timeout_net_r"].to_numpy(dtype=float)
    timeout_weights = fit_weights[fit_timeout]
    if not np.isfinite(timeout_targets).all():
        raise ValueError("fit timeout targets must be finite")
    fallback_mean = float(np.average(timeout_targets, weights=timeout_weights))
    bounds = tuple(np.quantile(timeout_targets, [0.01, 0.99]).astype(float))
    calibration_rows = decisions.iloc[partitions.calibration]
    calibration_timeout = calibration_rows["outcome_code"].eq(2).to_numpy()
    sufficient = timeout_rows >= 200 and timeout_episodes >= 20
    if not sufficient:
        return (
            TimeoutCalibration(
                bias=fallback_mean,
                used_fallback=True,
                rows=int(calibration_timeout.sum()),
                training_bounds=bounds,
            ),
            timeout_rows,
            timeout_episodes,
        )
    calibration_weights = _half_open_uniqueness(decisions, partitions.calibration)
    calibration = fit_timeout_bias(
        calibration_prediction[calibration_timeout],
        calibration_rows.loc[calibration_timeout, "timeout_net_r"].to_numpy(dtype=float),
        calibration_weights[calibration_timeout],
        fallback_mean=fallback_mean,
        training_bounds=bounds,
    )
    return calibration, timeout_rows, timeout_episodes


def _episode_overlap(decisions: pd.DataFrame, left: np.ndarray, right: np.ndarray) -> int:
    left_ids = set(decisions.iloc[left]["channel_episode_id"])
    right_ids = set(decisions.iloc[right]["channel_episode_id"])
    return len(left_ids.intersection(right_ids))


def _live_overlap(decisions: pd.DataFrame, left: np.ndarray, right: np.ndarray) -> int:
    if not len(left) or not len(right):
        return 0
    boundary = decisions.iloc[right]["decision_time"].min()
    return int(decisions.iloc[left]["label_end"].gt(boundary).sum())


def _validate_fold_scores(scores: pd.DataFrame, expected_keys: set[tuple[object, int]]) -> None:
    if scores.duplicated(["model", *_KEY_COLUMNS]).any():
        raise AssertionError("duplicate model/window/step OOF rows")
    actual_keys = set(scores[_KEY_COLUMNS].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        missing = len(expected_keys.difference(actual_keys))
        extra = len(actual_keys.difference(expected_keys))
        raise AssertionError(f"outer decision keys do not align: missing={missing}, extra={extra}")
    probabilities = scores[["p_sl", "p_tp", "p_timeout"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise AssertionError("OOF probabilities must be finite")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise AssertionError("OOF probabilities must sum to one")
    if not np.isfinite(scores["ev_score"].to_numpy(dtype=float)).all():
        raise AssertionError("OOF EV must be finite")


def _calibration_frequency_threshold(
    decisions: pd.DataFrame,
    ev_score: np.ndarray,
) -> tuple[float, int, int, int]:
    """Choose a non-negative count target using calibration rows only."""
    values = np.asarray(ev_score, dtype=float)
    if values.shape != (len(decisions),) or not np.isfinite(values).all():
        raise ValueError("calibration EV scores must be finite and aligned")
    if decisions.empty:
        raise ValueError("calibration decisions cannot be empty")
    days = pd.to_datetime(decisions["decision_time"], utc=True).dt.floor("D")
    calendar_days = int((days.max() - days.min()) / pd.Timedelta("1D")) + 1
    target_attempts = max(0, int(round(_REFERENCE_RATE * calendar_days)))
    candidates = sorted({0.0, *(float(value) for value in values if value >= 0.0)})
    window_max = (
        pd.DataFrame({"window_id": decisions["window_id"].to_numpy(), "score": values})
        .groupby("window_id", sort=False)["score"]
        .max()
        .to_numpy(dtype=float)
    )
    ranked = [
        (
            abs(int(np.count_nonzero(window_max >= threshold)) - target_attempts),
            -threshold,
        )
        for threshold in candidates
    ]
    threshold = float(candidates[min(range(len(candidates)), key=ranked.__getitem__)])
    achieved = int(np.count_nonzero(window_max >= threshold))
    return threshold, calendar_days, target_attempts, achieved


def run_tail_fold(
    model_name: str,
    fold: PurgedFold,
    dataset: TailDecisionDataset,
    config: TailOOFConfig = TailOOFConfig(),
) -> TailOOFResult:
    """Fit, calibrate, and score one untouched outer fold."""
    _validate_model_name(model_name)
    decisions = _validated_decisions(dataset)
    outer = np.asarray(fold.valid, dtype=np.int64)
    if outer.ndim != 1 or not outer.size:
        raise ValueError(f"fold {fold.fold_id} has no outer score rows")
    partitions = chronological_inner_partitions(decisions, fold.train, config.fold)
    (
        calibration_logits,
        calibration_timeout_prediction,
        outer_logits,
        outer_timeout_prediction,
        fit_hash,
    ) = _fit_predict_raw(model_name, dataset, decisions, partitions, outer, config)
    calibration_rows = decisions.iloc[partitions.calibration]
    calibration_weights = _half_open_uniqueness(decisions, partitions.calibration)
    temperature = fit_temperature(
        calibration_logits,
        calibration_rows["outcome_code"].to_numpy(dtype=np.int64),
        calibration_weights,
    )
    timeout_calibration, timeout_fit_rows, timeout_fit_episodes = _timeout_calibration(
        decisions, partitions, calibration_timeout_prediction
    )
    calibration_probabilities = apply_temperature(calibration_logits, temperature)
    calibration_timeout_net_r = apply_timeout_bias(
        calibration_timeout_prediction, timeout_calibration
    )
    calibration_ev = expected_net_r(
        calibration_probabilities,
        tp_net_r=calibration_rows["tp_net_r"].to_numpy(dtype=float),
        sl_net_r=calibration_rows["sl_net_r"].to_numpy(dtype=float),
        timeout_net_r=calibration_timeout_net_r,
    )
    (
        matched_rate_threshold,
        matched_rate_calendar_days,
        matched_rate_target_attempts,
        matched_rate_calibration_attempts,
    ) = _calibration_frequency_threshold(calibration_rows, calibration_ev)
    probabilities = apply_temperature(outer_logits, temperature)
    timeout_net_r = apply_timeout_bias(
        outer_timeout_prediction, timeout_calibration
    )
    outer_rows = decisions.iloc[outer].reset_index(drop=True)
    ev = expected_net_r(
        probabilities,
        tp_net_r=outer_rows["tp_net_r"].to_numpy(dtype=float),
        sl_net_r=outer_rows["sl_net_r"].to_numpy(dtype=float),
        timeout_net_r=timeout_net_r,
    )
    source_time = (
        outer_rows["source_bar_time"]
        if "source_bar_time" in outer_rows
        else outer_rows["decision_time"] - pd.Timedelta("5min")
    )
    scores = pd.DataFrame(
        {
            "model": model_name,
            "fold_id": fold.fold_id,
            "window_id": outer_rows["window_id"],
            "channel_episode_id": outer_rows["channel_episode_id"],
            "side": outer_rows["side"],
            "step": outer_rows["step"].astype(int),
            "source_bar_time": source_time,
            "decision_time": outer_rows["decision_time"],
            "p_sl": probabilities[:, 0],
            "p_tp": probabilities[:, 1],
            "p_timeout": probabilities[:, 2],
            "timeout_net_r_pred": timeout_net_r,
            "ev_score": ev,
            "matched_rate_threshold": matched_rate_threshold,
            "temperature": temperature.temperature,
            "timeout_bias": timeout_calibration.bias,
        }
    )
    expected_keys = set(outer_rows[_KEY_COLUMNS].itertuples(index=False, name=None))
    _validate_fold_scores(scores, expected_keys)

    fit, early, calibration = (
        partitions.fit,
        partitions.early,
        partitions.calibration,
    )
    fold_audit = pd.DataFrame(
        [
            {
                "model": model_name,
                "fold_id": fold.fold_id,
                "fit_rows": len(fit),
                "early_rows": len(early),
                "calibration_rows": len(calibration),
                "outer_rows": len(outer),
                "fit_early_episode_overlap": _episode_overlap(decisions, fit, early),
                "fit_calibration_episode_overlap": _episode_overlap(
                    decisions, fit, calibration
                ),
                "early_calibration_episode_overlap": _episode_overlap(
                    decisions, early, calibration
                ),
                "train_outer_episode_overlap": _episode_overlap(
                    decisions, np.concatenate([fit, early, calibration]), outer
                ),
                "fit_early_live_label_overlap": _live_overlap(decisions, fit, early),
                "fit_calibration_live_label_overlap": _live_overlap(
                    decisions, fit, calibration
                ),
                "early_calibration_live_label_overlap": _live_overlap(
                    decisions, early, calibration
                ),
                "calibration_outer_live_label_overlap": _live_overlap(
                    decisions, calibration, outer
                ),
                "fit_model_hash": fit_hash,
                "train_end": fold.train_end,
                "valid_start": fold.valid_start,
                "valid_end": fold.valid_end,
            }
        ]
    )
    overlap_columns = [column for column in fold_audit if "overlap" in column]
    if not fold_audit[overlap_columns].eq(0).all().all():
        raise AssertionError(f"fold {fold.fold_id} failed the leakage audit")
    calibration_audit = pd.DataFrame(
        [
            {
                "model": model_name,
                "fold_id": fold.fold_id,
                "temperature": temperature.temperature,
                "temperature_rows": temperature.rows,
                "timeout_bias": timeout_calibration.bias,
                "timeout_fallback": timeout_calibration.used_fallback,
                "timeout_calibration_rows": timeout_calibration.rows,
                "timeout_fit_rows": timeout_fit_rows,
                "timeout_fit_episodes": timeout_fit_episodes,
                "timeout_bound_low": timeout_calibration.training_bounds[0],
                "timeout_bound_high": timeout_calibration.training_bounds[1],
                "matched_rate_threshold": matched_rate_threshold,
                "matched_rate_calendar_days": matched_rate_calendar_days,
                "matched_rate_target_attempts": matched_rate_target_attempts,
                "matched_rate_calibration_attempts": matched_rate_calibration_attempts,
            }
        ]
    )
    return TailOOFResult(model_name, scores, fold_audit, calibration_audit)


def run_tail_model_oof(
    model_name: str,
    dataset: TailDecisionDataset,
    config: TailOOFConfig = TailOOFConfig(),
) -> TailOOFResult:
    """Run one registered model over the seven frozen Notebook J outer folds."""
    _validate_model_name(model_name)
    decisions = _validated_decisions(dataset)
    folds = _outer_folds(decisions)
    results = [run_tail_fold(model_name, fold, dataset, config) for fold in folds]
    scores = pd.concat([result.scores for result in results], ignore_index=True)
    fold_audit = pd.concat(
        [result.fold_audit for result in results], ignore_index=True
    )
    calibration_audit = pd.concat(
        [result.calibration_audit for result in results], ignore_index=True
    )
    expected_positions = np.concatenate([fold.valid for fold in folds])
    expected_keys = set(
        decisions.iloc[expected_positions][_KEY_COLUMNS].itertuples(
            index=False, name=None
        )
    )
    if len(expected_positions) != len(expected_keys):
        raise AssertionError("outer folds contain duplicate decision keys")
    _validate_fold_scores(scores, expected_keys)
    scores = scores.sort_values(
        ["decision_time", "window_id", "step"], kind="stable"
    ).reset_index(drop=True)
    return TailOOFResult(model_name, scores, fold_audit, calibration_audit)


def assert_identical_outer_score_keys(results: list[TailOOFResult]) -> None:
    """Hard-gate a multi-model contest onto one exact outer decision calendar."""
    if not results:
        raise ValueError("at least one OOF result is required")
    expected = set(
        results[0].scores[_KEY_COLUMNS].itertuples(index=False, name=None)
    )
    for result in results[1:]:
        actual = set(result.scores[_KEY_COLUMNS].itertuples(index=False, name=None))
        if actual != expected:
            raise AssertionError(
                f"unequal model calendars: {results[0].model_name} vs {result.model_name}"
            )


__all__ = [
    "TailFoldConfig",
    "TailInnerPartitions",
    "TailOOFConfig",
    "TailOOFResult",
    "assert_identical_outer_score_keys",
    "chronological_inner_partitions",
    "run_tail_fold",
    "run_tail_model_oof",
]
