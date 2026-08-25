"""Small causal TCN and purged expanding OOF scores for event windows."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from evaluation.channel_window_validation import (
    expanding_purged_folds,
)
from experiments.event_window_dataset import EventWindowSequences


@dataclass(frozen=True)
class TCNConfig:
    channels: int = 32
    pre_window_bars: int = 24
    active_bars: int = 12
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    rank_weight: float = 0.20
    batch_size: int = 128
    epochs: int = 40
    patience: int = 5
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.channels < 1 or self.pre_window_bars < 1 or self.active_bars < 1:
            raise ValueError("channel and window sizes must be positive")
        if self.kernel_size < 1 or not self.dilations or min(self.dilations) < 1:
            raise ValueError("kernel size and dilations must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.rank_weight < 0.0:
            raise ValueError("rank_weight must be non-negative")
        if self.batch_size < 1 or self.epochs < 1 or self.patience < 1:
            raise ValueError("training counts must be positive")


@dataclass(frozen=True)
class OOFSequenceResult:
    scores: pd.DataFrame
    fold_audit: pd.DataFrame
    sequence_scalers: list[dict[str, np.ndarray]]
    context_scalers: list[dict[str, np.ndarray]]


class CausalResidualBlock(nn.Module):
    """Residual Conv1d block with explicit left-only padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.left = (kernel - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel,
            dilation=dilation,
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        causal = self.conv(F.pad(inputs, (self.left, 0)))
        return F.gelu(self.skip(inputs) + self.dropout(causal))


class EventWindowTCN(nn.Module):
    """Emit one predicted net-R score at each active five-minute decision."""

    def __init__(
        self,
        n_sequence_features: int,
        n_context_features: int,
        config: TCNConfig,
    ) -> None:
        super().__init__()
        if n_sequence_features < 1 or n_context_features < 1:
            raise ValueError("feature counts must be positive")
        self.config = config
        blocks: list[nn.Module] = []
        in_channels = n_sequence_features
        for dilation in config.dilations:
            blocks.append(
                CausalResidualBlock(
                    in_channels,
                    config.channels,
                    kernel=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
            )
            in_channels = config.channels
        self.encoder = nn.Sequential(*blocks)
        self.pre_window = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                config.pre_window_bars * n_sequence_features,
                config.channels,
            ),
            nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * config.channels + n_context_features, config.channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.channels, 1),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        expected = self.config.pre_window_bars + self.config.active_bars
        if sequence.ndim != 3 or sequence.shape[1] != expected:
            raise ValueError(f"sequence must have shape [N, {expected}, F]")
        if context.ndim != 3 or context.shape[:2] != (
            sequence.shape[0],
            self.config.active_bars,
        ):
            raise ValueError(
                f"context must have shape [N, {self.config.active_bars}, C]"
            )
        hidden = self.encoder(sequence.transpose(1, 2)).transpose(1, 2)
        left = self.config.pre_window_bars
        active = hidden[:, left : left + self.config.active_bars, :]
        pre = self.pre_window(sequence[:, :left, :]).unsqueeze(1)
        pre = pre.expand(-1, self.config.active_bars, -1)
        return self.head(torch.cat([active, pre, context], dim=-1)).squeeze(-1)


def within_window_pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Logistic ranking loss over valid unequal step pairs within each window."""
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError("prediction, target and mask must have the same shape")
    target_diff = target.unsqueeze(2) - target.unsqueeze(1)
    prediction_diff = prediction.unsqueeze(2) - prediction.unsqueeze(1)
    pairs = mask.bool().unsqueeze(2) & mask.bool().unsqueeze(1)
    pairs &= target_diff.ne(0.0)
    pairs &= torch.triu(torch.ones_like(pairs, dtype=torch.bool), diagonal=1)
    if not bool(pairs.any()):
        return prediction.sum() * 0.0
    ordered = -target_diff.sign() * prediction_diff
    return F.softplus(ordered)[pairs].mean()


def weighted_event_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    uniqueness: torch.Tensor,
    mask: torch.Tensor,
    rank_weight: float = 0.20,
) -> torch.Tensor:
    """Uniqueness-weighted Smooth-L1 plus within-window ranking loss."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must align")
    if uniqueness.shape != target.shape or mask.shape != target.shape:
        raise ValueError("uniqueness and mask must align with target")
    point = F.smooth_l1_loss(prediction, target, reduction="none")
    weights = uniqueness * mask.to(dtype=uniqueness.dtype)
    point_loss = (point * weights).sum() / weights.sum().clamp_min(1.0)
    return point_loss + rank_weight * within_window_pairwise_loss(
        prediction,
        target,
        mask,
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fit_scaler(
    values: np.ndarray,
    valid: np.ndarray,
) -> dict[str, np.ndarray]:
    if values.ndim != 3 or valid.shape != values.shape[:2]:
        raise ValueError("scaler values and validity mask do not align")
    feature_count = values.shape[2]
    median = np.zeros(feature_count, dtype=np.float32)
    scale = np.ones(feature_count, dtype=np.float32)
    for feature in range(feature_count):
        observed = values[:, :, feature][valid]
        observed = observed[np.isfinite(observed)]
        if observed.size == 0:
            continue
        median[feature] = np.float32(np.median(observed))
        spread = float(np.std(observed))
        if np.isfinite(spread) and spread > 1e-8:
            scale[feature] = np.float32(spread)
    return {"median": median, "scale": scale}


def _apply_scaler(
    values: np.ndarray,
    valid: np.ndarray,
    scaler: dict[str, np.ndarray],
) -> np.ndarray:
    median = np.asarray(scaler["median"], dtype=np.float32)
    scale = np.asarray(scaler["scale"], dtype=np.float32)
    out = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(out)
    out = np.where(finite, out, median.reshape(1, 1, -1))
    out = (out - median.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    out[~valid] = 0.0
    return out.astype(np.float32, copy=False)


def _validate_contract(
    sequences: EventWindowSequences,
    labels: pd.DataFrame,
    config: TCNConfig,
) -> pd.DataFrame:
    n_windows = len(sequences.metadata)
    if sequences.sequence.shape[:2] != (
        n_windows,
        config.pre_window_bars + config.active_bars,
    ):
        raise ValueError("sequence tensor violates the frozen 24+12 contract")
    if sequences.context.shape[:2] != (n_windows, config.active_bars):
        raise ValueError("context tensor violates the frozen 12-step contract")
    if sequences.sequence_valid.shape != sequences.sequence.shape[:2]:
        raise ValueError("sequence_valid does not align with sequence")
    if sequences.decision_valid.shape != sequences.context.shape[:2]:
        raise ValueError("decision_valid does not align with context")
    required_meta = {
        "window_id",
        "channel_episode_id",
        "side",
        "window_start",
    }
    missing_meta = sorted(required_meta.difference(sequences.metadata.columns))
    if missing_meta:
        raise ValueError(f"sequence metadata missing columns: {missing_meta}")
    if sequences.metadata["window_id"].duplicated().any():
        raise ValueError("window_id must be unique")
    required_labels = {
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "label_start",
        "label_end",
        "geometry_valid",
        "model_target_valid",
        "r_net",
    }
    missing_labels = sorted(required_labels.difference(labels.columns))
    if missing_labels:
        raise ValueError(f"labels missing columns: {missing_labels}")
    work = labels.copy().reset_index(drop=True)
    work["step"] = pd.to_numeric(work["step"], errors="raise").astype(int)
    if work["step"].lt(0).any() or work["step"].ge(config.active_bars).any():
        raise ValueError("label step is outside the active window")
    if work.duplicated(["window_id", "step"]).any():
        raise ValueError("labels must be unique by window_id and step")
    known = set(sequences.metadata["window_id"])
    if not set(work["window_id"]).issubset(known):
        raise ValueError("labels contain unknown windows")
    for column in ("decision_time", "label_start", "label_end"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="coerce")
    if work["decision_time"].isna().any():
        raise ValueError("every label needs a finite decision_time")
    if work.loc[work["geometry_valid"].astype(bool), ["label_start", "label_end"]].isna().any().any():
        raise ValueError("geometry-valid labels need finite label intervals")
    if (
        work.loc[work["geometry_valid"].astype(bool), "label_end"]
        < work.loc[work["geometry_valid"].astype(bool), "label_start"]
    ).any():
        raise ValueError("label_end cannot precede label_start")
    row_lookup = {
        window_id: row
        for row, window_id in enumerate(sequences.metadata["window_id"].tolist())
    }
    expected_keys = {
        (sequences.metadata.iloc[row]["window_id"], int(step))
        for row in range(len(sequences.metadata))
        for step in np.flatnonzero(sequences.decision_valid[row])
    }
    actual_keys = set(zip(work["window_id"], work["step"], strict=True))
    if actual_keys != expected_keys:
        raise ValueError("labels must cover every valid decision exactly once")
    for label in work.itertuples(index=False):
        row = row_lookup[label.window_id]
        expected = pd.Timestamp(sequences.decision_times[row, int(label.step)])
        expected = (
            expected.tz_localize("UTC")
            if expected.tzinfo is None
            else expected.tz_convert("UTC")
        )
        if label.decision_time != expected:
            raise ValueError("label decision_time disagrees with the sequence clock")
    return work


def _window_fold_frame(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    geometry = labels.loc[labels["geometry_valid"].astype(bool)]
    grouped = {key: group for key, group in geometry.groupby("window_id", sort=False)}
    for sample_index, meta in metadata.reset_index(drop=True).iterrows():
        group = grouped.get(meta["window_id"])
        if group is None or group.empty:
            continue
        if not group["channel_episode_id"].eq(meta["channel_episode_id"]).all():
            raise ValueError("label and tensor channel_episode_id disagree")
        rows.append(
            {
                "sample_index": int(sample_index),
                "window_id": meta["window_id"],
                "decision_time": pd.Timestamp(meta["window_start"]),
                "label_start": group["label_start"].min(),
                "label_end": group["label_end"].max(),
                "channel_episode_id": meta["channel_episode_id"],
            }
        )
    return pd.DataFrame(rows)


def _target_matrices(
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
    active_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    row_lookup = {
        window_id: row
        for row, window_id in enumerate(metadata["window_id"].tolist())
    }
    target = np.zeros((len(metadata), active_bars), dtype=np.float32)
    mask = np.zeros((len(metadata), active_bars), dtype=bool)
    for label in labels.itertuples(index=False):
        row = row_lookup.get(label.window_id)
        if row is None or not bool(label.model_target_valid):
            continue
        value = float(label.r_net)
        if not np.isfinite(value):
            raise ValueError("model_target_valid requires finite r_net")
        target[row, int(label.step)] = np.float32(value)
        mask[row, int(label.step)] = True
    return target, mask


def _inner_episode_split(
    sample_indices: np.ndarray,
    metadata: pd.DataFrame,
    labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    selected = metadata.iloc[sample_indices].copy()
    selected["window_start"] = pd.to_datetime(selected["window_start"], utc=True)
    episode_order = (
        selected.groupby("channel_episode_id", sort=False)["window_start"]
        .max()
        .sort_values(kind="stable")
    )
    if len(episode_order) < 2:
        return sample_indices, np.array([], dtype=np.int64)
    early_count = max(1, int(math.ceil(0.10 * len(episode_order))))
    early_episodes = set(episode_order.index[-early_count:])
    early_mask = selected["channel_episode_id"].isin(early_episodes).to_numpy()
    early = sample_indices[early_mask]
    early_start = selected.loc[early_mask, "window_start"].min()
    candidate_episodes = set(selected.loc[~early_mask, "channel_episode_id"])
    candidate_windows = set(
        selected.loc[selected["channel_episode_id"].isin(candidate_episodes), "window_id"]
    )
    candidate_labels = labels.loc[
        labels["window_id"].isin(candidate_windows)
        & labels["geometry_valid"].astype(bool)
    ]
    episode_label_end = candidate_labels.groupby("channel_episode_id")["label_end"].max()
    safe_episodes = set(episode_label_end.index[episode_label_end.lt(early_start)])
    inner = sample_indices[
        selected["channel_episode_id"].isin(safe_episodes).to_numpy()
    ]
    return inner, early


def _uniqueness_matrix(
    labels: pd.DataFrame,
    metadata: pd.DataFrame,
    sample_indices: np.ndarray,
    active_bars: int,
) -> np.ndarray:
    weights = np.zeros((len(metadata), active_bars), dtype=np.float32)
    allowed = set(metadata.iloc[sample_indices]["window_id"])
    selected = labels["window_id"].isin(allowed) & labels["model_target_valid"].astype(bool)
    positions = np.flatnonzero(selected.to_numpy())
    if positions.size == 0:
        return weights
    values = _half_open_interval_uniqueness(labels, positions)
    lookup = {
        window_id: row
        for row, window_id in enumerate(metadata["window_id"].tolist())
    }
    for position, value in zip(positions, values, strict=True):
        label = labels.iloc[position]
        weights[lookup[label["window_id"]], int(label["step"])] = np.float32(value)
    return weights


def _half_open_interval_uniqueness(
    labels: pd.DataFrame,
    positions: np.ndarray,
) -> np.ndarray:
    """Average inverse concurrency for information intervals ``[start, end)``."""
    selected = labels.iloc[np.asarray(positions, dtype=np.int64)]
    start = pd.to_datetime(selected["label_start"], utc=True, errors="raise")
    end = pd.to_datetime(selected["label_end"], utc=True, errors="raise")
    if start.isna().any() or end.isna().any() or (end <= start).any():
        raise ValueError("training label intervals must have positive half-open length")
    origin = start.min().floor("min")
    left = np.floor((start - origin).dt.total_seconds().to_numpy() / 60.0).astype(
        np.int64
    )
    right = np.ceil((end - origin).dt.total_seconds().to_numpy() / 60.0).astype(
        np.int64
    )
    difference = np.zeros(int(right.max()) + 1, dtype=np.int64)
    np.add.at(difference, left, 1)
    np.add.at(difference, right, -1)
    concurrency = np.cumsum(difference[:-1])
    weights = np.empty(len(selected), dtype=float)
    for row, (begin, finish) in enumerate(zip(left, right, strict=True)):
        active = concurrency[begin:finish]
        if active.size == 0 or (active <= 0).any():
            raise AssertionError("invalid half-open label concurrency")
        weights[row] = float(np.mean(1.0 / active))
    return weights / weights.mean()


def _batch_loss(
    model: EventWindowTCN,
    sequence: torch.Tensor,
    context: torch.Tensor,
    target: torch.Tensor,
    uniqueness: torch.Tensor,
    mask: torch.Tensor,
    rows: np.ndarray,
    config: TCNConfig,
) -> torch.Tensor:
    index = torch.as_tensor(rows, dtype=torch.long)
    prediction = model(sequence[index], context[index])
    return weighted_event_loss(
        prediction,
        target[index],
        uniqueness[index],
        mask[index],
        config.rank_weight,
    )


def _train_fold(
    sequence: np.ndarray,
    context: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    uniqueness: np.ndarray,
    train_rows: np.ndarray,
    early_rows: np.ndarray,
    config: TCNConfig,
    seed: int,
) -> EventWindowTCN:
    _seed_everything(seed)
    model = EventWindowTCN(sequence.shape[2], context.shape[2], config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    sequence_tensor = torch.from_numpy(sequence)
    context_tensor = torch.from_numpy(context)
    target_tensor = torch.from_numpy(target)
    mask_tensor = torch.from_numpy(target_mask)
    uniqueness_tensor = torch.from_numpy(uniqueness)
    generator = np.random.default_rng(seed)
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0

    for _ in range(config.epochs):
        model.train()
        shuffled = generator.permutation(train_rows)
        for start in range(0, len(shuffled), config.batch_size):
            batch = shuffled[start : start + config.batch_size]
            if not target_mask[batch].any():
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                model,
                sequence_tensor,
                context_tensor,
                target_tensor,
                uniqueness_tensor,
                mask_tensor,
                batch,
                config,
            )
            loss.backward()
            optimizer.step()

        monitor_rows = early_rows if early_rows.size and target_mask[early_rows].any() else train_rows
        model.eval()
        with torch.no_grad():
            monitor_uniqueness = uniqueness_tensor
            if early_rows.size and np.array_equal(monitor_rows, early_rows):
                monitor_uniqueness = torch.ones_like(uniqueness_tensor)
            monitor = _batch_loss(
                model,
                sequence_tensor,
                context_tensor,
                target_tensor,
                monitor_uniqueness,
                mask_tensor,
                monitor_rows,
                config,
            )
        monitor_value = float(monitor.item())
        if monitor_value < best_loss - 1e-8:
            best_loss = monitor_value
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def run_event_window_oof(
    sequences: EventWindowSequences,
    labels: pd.DataFrame,
    config: TCNConfig = TCNConfig(),
) -> OOFSequenceResult:
    """Return purged expanding OOF scores for every valid active step."""
    work_labels = _validate_contract(sequences, labels, config)
    metadata = sequences.metadata.copy().reset_index(drop=True)
    metadata["window_start"] = pd.to_datetime(metadata["window_start"], utc=True)
    fold_frame = _window_fold_frame(metadata, work_labels)
    if fold_frame.empty:
        return OOFSequenceResult(
            scores=pd.DataFrame(),
            fold_audit=pd.DataFrame(),
            sequence_scalers=[],
            context_scalers=[],
        )
    folds = expanding_purged_folds(fold_frame)
    target, target_mask = _target_matrices(metadata, work_labels, config.active_bars)
    target_mask &= np.asarray(sequences.decision_valid, dtype=bool)
    score_rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []
    sequence_scalers: list[dict[str, np.ndarray]] = []
    context_scalers: list[dict[str, np.ndarray]] = []

    for fold_no, fold in enumerate(folds):
        train_rows = fold_frame.iloc[fold.train]["sample_index"].to_numpy(dtype=np.int64)
        valid_rows = fold_frame.iloc[fold.valid]["sample_index"].to_numpy(dtype=np.int64)
        if train_rows.size == 0 or valid_rows.size == 0:
            continue
        inner_rows, early_rows = _inner_episode_split(
            train_rows,
            metadata,
            work_labels,
        )
        if inner_rows.size == 0 or not target_mask[inner_rows].any():
            continue

        sequence_scaler = _fit_scaler(
            np.asarray(sequences.sequence)[inner_rows],
            np.asarray(sequences.sequence_valid, dtype=bool)[inner_rows],
        )
        context_scaler = _fit_scaler(
            np.asarray(sequences.context)[inner_rows],
            np.asarray(sequences.decision_valid, dtype=bool)[inner_rows],
        )
        scaled_sequence = _apply_scaler(
            np.asarray(sequences.sequence),
            np.asarray(sequences.sequence_valid, dtype=bool),
            sequence_scaler,
        )
        scaled_context = _apply_scaler(
            np.asarray(sequences.context),
            np.asarray(sequences.decision_valid, dtype=bool),
            context_scaler,
        )
        uniqueness = _uniqueness_matrix(
            work_labels,
            metadata,
            inner_rows,
            config.active_bars,
        )
        model = _train_fold(
            scaled_sequence,
            scaled_context,
            target,
            target_mask,
            uniqueness,
            inner_rows,
            early_rows,
            config,
            config.random_seed + fold_no,
        )
        with torch.no_grad():
            prediction = model(
                torch.from_numpy(scaled_sequence[valid_rows]),
                torch.from_numpy(scaled_context[valid_rows]),
            ).cpu().numpy()

        for local_row, sample_index in enumerate(valid_rows):
            valid_steps = np.flatnonzero(sequences.decision_valid[sample_index])
            for step in valid_steps:
                decision = pd.Timestamp(sequences.decision_times[sample_index, step])
                decision = decision.tz_localize("UTC") if decision.tzinfo is None else decision.tz_convert("UTC")
                score_rows.append(
                    {
                        "window_id": metadata.at[sample_index, "window_id"],
                        "channel_episode_id": metadata.at[sample_index, "channel_episode_id"],
                        "side": metadata.at[sample_index, "side"],
                        "step": int(step),
                        "decision_time": decision,
                        "fold_id": fold.fold_id,
                        "score": float(prediction[local_row, step]),
                    }
                )

        train_episodes = set(metadata.iloc[train_rows]["channel_episode_id"])
        valid_episodes = set(metadata.iloc[valid_rows]["channel_episode_id"])
        train_label_end = fold_frame.iloc[fold.train]["label_end"]
        live_overlap = int(pd.to_datetime(train_label_end, utc=True).ge(fold.valid_start).sum())
        inner_early_overlap = 0
        if early_rows.size and inner_rows.size:
            early_start = metadata.iloc[early_rows]["window_start"].min()
            inner_windows = set(metadata.iloc[inner_rows]["window_id"])
            inner_early_overlap = int(
                work_labels.loc[
                    work_labels["window_id"].isin(inner_windows)
                    & work_labels["geometry_valid"].astype(bool),
                    "label_end",
                ].ge(early_start).sum()
            )
        audit_rows.append(
            {
                "fold_id": fold.fold_id,
                "train_windows": int(len(train_rows)),
                "inner_train_windows": int(len(inner_rows)),
                "early_stop_windows": int(len(early_rows)),
                "valid_windows": int(len(valid_rows)),
                "train_valid_episode_overlap": int(len(train_episodes & valid_episodes)),
                "live_label_overlap": live_overlap,
                "inner_early_live_label_overlap": inner_early_overlap,
                "train_end": fold.train_end,
                "valid_start": fold.valid_start,
                "valid_end": fold.valid_end,
            }
        )
        sequence_scalers.append(sequence_scaler)
        context_scalers.append(context_scaler)

    scores = pd.DataFrame(score_rows)
    if not scores.empty:
        if scores.duplicated(["window_id", "step"]).any():
            raise AssertionError("OOF scores are duplicated across folds")
        scores = scores.sort_values(["decision_time", "window_id", "step"]).reset_index(drop=True)
    return OOFSequenceResult(
        scores=scores,
        fold_audit=pd.DataFrame(audit_rows),
        sequence_scalers=sequence_scalers,
        context_scalers=context_scalers,
    )
