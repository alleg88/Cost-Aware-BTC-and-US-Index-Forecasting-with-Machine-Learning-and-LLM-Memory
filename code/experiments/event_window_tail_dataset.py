"""Causal per-decision features and distributional labels for event windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from experiments.event_window_dataset import EventWindowSequences


TABULAR_HORIZONS = (3, 6, 12, 24)
_PRE_WINDOW_BARS = 24
_WINDOW_BATCH_SIZE = 512
_KEY_COLUMNS = ("window_id", "step")
_TIMESTAMP_COLUMNS = (
    "source_bar_time",
    "decision_time",
    "entry_time",
    "exit_time",
    "label_start",
    "label_end",
)


@dataclass(frozen=True)
class TailDecisionDataset:
    decisions: pd.DataFrame
    tabular: np.ndarray
    tabular_features: tuple[str, ...]
    sequences: EventWindowSequences


def tabular_feature_names(
    sequence_features: Sequence[str],
    context_features: Sequence[str],
) -> tuple[str, ...]:
    """Return the stable current/summary/context feature order."""
    sequence = tuple(str(name) for name in sequence_features)
    context = tuple(str(name) for name in context_features)
    summaries = tuple(
        f"{feature}_{statistic}_{horizon}"
        for horizon in TABULAR_HORIZONS
        for statistic in ("delta", "mean", "std")
        for feature in sequence
    )
    names = (*sequence, *summaries, *context)
    if len(names) != len(set(names)):
        raise ValueError("tabular feature names must be unique")
    return names


def _validate_sequences(sequences: EventWindowSequences) -> tuple[int, int, int]:
    if sequences.sequence.ndim != 3 or sequences.context.ndim != 3:
        raise ValueError("sequence and context must be three-dimensional")
    windows, sequence_bars, sequence_features = sequences.sequence.shape
    context_windows, active_bars, context_features = sequences.context.shape
    if context_windows != windows or len(sequences.metadata) != windows:
        raise ValueError("metadata, sequence, and context window counts must align")
    if sequence_bars != _PRE_WINDOW_BARS + active_bars:
        raise ValueError("sequence must contain 24 pre-window rows plus active rows")
    if sequence_features != len(sequences.sequence_features):
        raise ValueError("sequence feature names do not align with the tensor")
    if context_features != len(sequences.context_features):
        raise ValueError("context feature names do not align with the tensor")
    if sequences.sequence_valid.shape != (windows, sequence_bars):
        raise ValueError("sequence_valid does not align with sequence")
    expected_active = (windows, active_bars)
    if sequences.decision_valid.shape != expected_active:
        raise ValueError("decision_valid does not align with context")
    if sequences.source_bar_times.shape != expected_active:
        raise ValueError("source_bar_times does not align with context")
    if sequences.decision_times.shape != expected_active:
        raise ValueError("decision_times does not align with context")
    if "window_id" not in sequences.metadata:
        raise ValueError("sequence metadata missing window_id")
    if sequences.metadata["window_id"].duplicated().any():
        raise ValueError("sequence window_id must be unique")
    active_sequence_valid = sequences.sequence_valid[:, _PRE_WINDOW_BARS:]
    if np.any(sequences.decision_valid & ~active_sequence_valid):
        raise ValueError("valid decisions require a valid current sequence row")
    return windows, active_bars, sequence_features


def _expected_decisions(
    sequences: EventWindowSequences,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    window_rows, steps = np.nonzero(sequences.decision_valid)
    window_ids = sequences.metadata["window_id"].to_numpy()[window_rows]
    expected = pd.DataFrame({"window_id": window_ids, "step": steps.astype(int)})
    return expected, window_rows, steps


def _align_labels(
    sequences: EventWindowSequences,
    labels: pd.DataFrame,
    expected: pd.DataFrame,
    window_rows: np.ndarray,
    steps: np.ndarray,
) -> pd.DataFrame:
    required = {
        "window_id",
        "step",
        "decision_time",
        "risk_bps",
        "outcome",
        "r_net",
        "label_start",
        "label_end",
        "model_target_valid",
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"labels missing columns: {missing}")
    work = labels.copy().reset_index(drop=True)
    work["step"] = pd.to_numeric(work["step"], errors="raise").astype(int)
    if work.duplicated(list(_KEY_COLUMNS)).any():
        raise ValueError("duplicate label decision keys")

    expected_keys = set(expected.itertuples(index=False, name=None))
    actual_keys = set(work.loc[:, _KEY_COLUMNS].itertuples(index=False, name=None))
    if actual_keys != expected_keys:
        raise ValueError("labels must match all valid decision keys exactly")

    non_key_columns = [column for column in work.columns if column not in _KEY_COLUMNS]
    aligned = expected.merge(
        work,
        on=list(_KEY_COLUMNS),
        how="left",
        sort=False,
        validate="one_to_one",
    )[[*_KEY_COLUMNS, *non_key_columns]]
    for column in _TIMESTAMP_COLUMNS:
        if column in aligned:
            aligned[column] = pd.to_datetime(aligned[column], utc=True, errors="coerce")
    if aligned["decision_time"].isna().any():
        raise ValueError("every label needs a finite decision_time")

    expected_times = pd.to_datetime(
        sequences.decision_times[window_rows, steps], utc=True, errors="coerce"
    )
    if not np.array_equal(
        aligned["decision_time"].to_numpy(dtype="datetime64[ns]"),
        expected_times.to_numpy(dtype="datetime64[ns]"),
    ):
        raise ValueError("label decision_time disagrees with the sequence clock")
    if "source_bar_time" in aligned:
        expected_sources = pd.to_datetime(
            sequences.source_bar_times[window_rows, steps], utc=True, errors="coerce"
        )
        if not np.array_equal(
            aligned["source_bar_time"].to_numpy(dtype="datetime64[ns]"),
            expected_sources.to_numpy(dtype="datetime64[ns]"),
        ):
            raise ValueError("label source_bar_time disagrees with the sequence clock")

    for identity in ("channel_episode_id", "side"):
        if identity not in aligned or identity not in sequences.metadata:
            continue
        expected_identity = sequences.metadata[identity].to_numpy()[window_rows]
        if not np.array_equal(aligned[identity].to_numpy(), expected_identity):
            raise ValueError(f"label {identity} disagrees with sequence metadata")

    valid_target = aligned["model_target_valid"].fillna(False).astype(bool)
    aligned["model_target_valid"] = valid_target
    if aligned.loc[valid_target, ["label_start", "label_end"]].isna().any().any():
        raise ValueError("training labels need finite label intervals")
    if (
        aligned.loc[valid_target, "label_end"]
        < aligned.loc[valid_target, "label_start"]
    ).any():
        raise ValueError("label_end cannot precede label_start")
    outcomes = aligned["outcome"].astype(str).str.lower()
    if not outcomes.loc[valid_target].isin({"sl", "tp", "timeout"}).all():
        raise ValueError("model_target_valid outcomes must be sl, tp, or timeout")
    net_r = pd.to_numeric(aligned["r_net"], errors="coerce")
    if not np.isfinite(net_r.loc[valid_target]).all():
        raise ValueError("model_target_valid rows require finite r_net")
    aligned["outcome"] = outcomes
    aligned["r_net"] = net_r
    return aligned


def _add_distributional_targets(
    decisions: pd.DataFrame,
    *,
    rr: float,
    cost_bps: float,
) -> pd.DataFrame:
    out = decisions.copy()
    risk_bps = pd.to_numeric(out["risk_bps"], errors="coerce").to_numpy(dtype=float)
    valid_risk = np.isfinite(risk_bps) & (risk_bps > 0.0)
    cost_r = np.full(len(out), np.nan, dtype=float)
    np.divide(cost_bps, risk_bps, out=cost_r, where=valid_risk)
    out["risk_bps"] = risk_bps
    out["tp_net_r"] = np.where(valid_risk, rr - cost_r, np.nan)
    out["sl_net_r"] = np.where(valid_risk, -1.0 - cost_r, np.nan)

    target_valid = out["model_target_valid"].to_numpy(dtype=bool)
    outcome_code = np.full(len(out), -1, dtype=np.int8)
    outcome = out["outcome"].to_numpy(dtype=object)
    for name, code in (("sl", 0), ("tp", 1), ("timeout", 2)):
        outcome_code[target_valid & (outcome == name)] = code
    out["outcome_code"] = outcome_code
    observed_timeout = target_valid & (outcome_code == 2)
    out["timeout_net_r"] = np.where(observed_timeout, out["r_net"], np.nan)
    return out


def _tabular_batch(
    sequences: EventWindowSequences,
    start: int,
    stop: int,
) -> np.ndarray:
    values = np.asarray(sequences.sequence[start:stop], dtype=np.float64)
    sequence_valid = sequences.sequence_valid[start:stop, :, None]
    finite = sequence_valid & np.isfinite(values)
    observed = np.where(finite, values, 0.0)
    padded_shape = (len(values), values.shape[1] + 1, values.shape[2])
    prefix = np.zeros(padded_shape, dtype=np.float64)
    prefix_squared = np.zeros(padded_shape, dtype=np.float64)
    prefix_count = np.zeros(padded_shape, dtype=np.int32)
    np.cumsum(observed, axis=1, dtype=np.float64, out=prefix[:, 1:])
    np.cumsum(observed * observed, axis=1, dtype=np.float64, out=prefix_squared[:, 1:])
    np.cumsum(finite, axis=1, dtype=np.int32, out=prefix_count[:, 1:])

    active_bars = sequences.context.shape[1]
    active_positions = _PRE_WINDOW_BARS + np.arange(active_bars)
    right = active_positions + 1
    current = values[:, active_positions, :]
    columns: list[np.ndarray] = [current]
    for horizon in TABULAR_HORIZONS:
        left = right - horizon
        sums = prefix[:, right, :] - prefix[:, left, :]
        squared = prefix_squared[:, right, :] - prefix_squared[:, left, :]
        counts = prefix_count[:, right, :] - prefix_count[:, left, :]
        mean = np.full_like(sums, np.nan)
        np.divide(sums, counts, out=mean, where=counts > 0)
        variance = np.full_like(sums, np.nan)
        np.divide(squared, counts, out=variance, where=counts > 0)
        variance -= mean * mean
        variance = np.maximum(variance, 0.0)
        std = np.sqrt(variance)

        lag_positions = active_positions - horizon
        lagged = values[:, lag_positions, :]
        lag_valid = finite[:, lag_positions, :]
        delta = current - lagged
        delta[~(np.isfinite(current) & lag_valid)] = np.nan
        columns.extend((delta, mean, std))
    columns.append(np.asarray(sequences.context[start:stop], dtype=np.float64))
    return np.concatenate(columns, axis=2).astype(np.float32, copy=False)


def _build_tabular(
    sequences: EventWindowSequences,
    row_count: int,
    feature_count: int,
) -> np.ndarray:
    out = np.empty((row_count, feature_count), dtype=np.float32)
    cursor = 0
    for start in range(0, len(sequences.metadata), _WINDOW_BATCH_SIZE):
        stop = min(start + _WINDOW_BATCH_SIZE, len(sequences.metadata))
        batch = _tabular_batch(sequences, start, stop)
        valid = sequences.decision_valid[start:stop]
        selected = batch[valid]
        out[cursor : cursor + len(selected)] = selected
        cursor += len(selected)
    if cursor != row_count:
        raise AssertionError("tabular decision count changed during construction")
    return out


def build_tail_decision_dataset(
    sequences: EventWindowSequences,
    labels: pd.DataFrame,
    *,
    rr: float = 2.0,
    cost_bps: float = 10.0,
) -> TailDecisionDataset:
    """Align valid causal steps with RR labels and deterministic decision features."""
    if not np.isfinite(rr) or rr <= 0.0:
        raise ValueError("rr must be positive and finite")
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be non-negative and finite")
    _validate_sequences(sequences)
    features = tabular_feature_names(
        sequences.sequence_features, sequences.context_features
    )
    expected, window_rows, steps = _expected_decisions(sequences)
    decisions = _align_labels(
        sequences, labels, expected, window_rows, steps
    )
    decisions = _add_distributional_targets(decisions, rr=rr, cost_bps=cost_bps)
    tabular = _build_tabular(sequences, len(decisions), len(features))
    return TailDecisionDataset(
        decisions=decisions.reset_index(drop=True),
        tabular=tabular,
        tabular_features=features,
        sequences=sequences,
    )


__all__ = [
    "TABULAR_HORIZONS",
    "TailDecisionDataset",
    "build_tail_decision_dataset",
    "tabular_feature_names",
]
