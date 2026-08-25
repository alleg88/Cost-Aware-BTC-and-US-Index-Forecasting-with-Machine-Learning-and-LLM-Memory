"""Distributional causal TCN and GRU models for event-window decisions."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from experiments.event_window_tcn import CausalResidualBlock


@dataclass(frozen=True)
class TailNeuralConfig:
    hidden_size: int = 32
    pre_window_bars: int = 24
    active_bars: int = 12
    kernel_size: int = 3
    dilations: tuple[int, ...] = (1, 2)
    dropout: float = 0.20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 40
    patience: int = 5
    batch_size: int = 128
    timeout_loss_weight: float = 0.50
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_size < 1 or self.pre_window_bars < 1 or self.active_bars < 1:
            raise ValueError("hidden and window sizes must be positive")
        if self.kernel_size < 1 or not self.dilations or min(self.dilations) < 1:
            raise ValueError("kernel size and dilations must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("optimizer parameters are invalid")
        if self.epochs < 1 or self.patience < 1 or self.batch_size < 1:
            raise ValueError("training counts must be positive")
        if self.timeout_loss_weight < 0.0:
            raise ValueError("timeout loss weight must be non-negative")


@dataclass(frozen=True)
class TailNeuralTensors:
    sequence: np.ndarray
    context: np.ndarray
    sequence_valid: np.ndarray
    decision_valid: np.ndarray
    outcome: np.ndarray
    timeout_target: np.ndarray
    uniqueness: np.ndarray


@dataclass(frozen=True)
class NeuralTailPrediction:
    logits: np.ndarray
    timeout_net_r: np.ndarray
    scaler_median: np.ndarray
    scaler_scale: np.ndarray
    audit: dict[str, object]


class _DistributionHeads(nn.Module):
    def __init__(self, input_size: int, config: TailNeuralConfig) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.outcome = nn.Linear(config.hidden_size, 3)
        self.timeout = nn.Linear(config.hidden_size, 1)

    def forward(self, values: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.shared(values)
        return self.outcome(hidden), self.timeout(hidden).squeeze(-1)


class DistributionalTCN(nn.Module):
    """Causal convolutional encoder with outcome and timeout-return heads."""

    def __init__(
        self,
        n_sequence_features: int,
        n_context_features: int,
        config: TailNeuralConfig = TailNeuralConfig(),
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
                    config.hidden_size,
                    kernel=config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
            )
            in_channels = config.hidden_size
        self.encoder = nn.Sequential(*blocks)
        self.pre_window = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                config.pre_window_bars * n_sequence_features,
                config.hidden_size,
            ),
            nn.GELU(),
        )
        self.heads = _DistributionHeads(
            2 * config.hidden_size + n_context_features,
            config,
        )

    def forward(self, sequence: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        _validate_model_inputs(sequence, context, self.config)
        hidden = self.encoder(sequence.transpose(1, 2)).transpose(1, 2)
        left = self.config.pre_window_bars
        active = hidden[:, left : left + self.config.active_bars]
        pre = self.pre_window(sequence[:, :left]).unsqueeze(1)
        pre = pre.expand(-1, self.config.active_bars, -1)
        return self.heads(torch.cat([active, pre, context], dim=-1))


class DistributionalGRU(nn.Module):
    """One-layer unidirectional GRU over the ordered 24+12 bar sequence."""

    def __init__(
        self,
        n_sequence_features: int,
        n_context_features: int,
        config: TailNeuralConfig = TailNeuralConfig(),
    ) -> None:
        super().__init__()
        if n_sequence_features < 1 or n_context_features < 1:
            raise ValueError("feature counts must be positive")
        self.config = config
        self.gru = nn.GRU(
            input_size=n_sequence_features,
            hidden_size=config.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.heads = _DistributionHeads(
            config.hidden_size + n_context_features,
            config,
        )

    def forward(self, sequence: Tensor, context: Tensor) -> tuple[Tensor, Tensor]:
        _validate_model_inputs(sequence, context, self.config)
        hidden, _ = self.gru(sequence)
        left = self.config.pre_window_bars
        active = hidden[:, left : left + self.config.active_bars]
        return self.heads(torch.cat([active, context], dim=-1))


def _validate_model_inputs(
    sequence: Tensor,
    context: Tensor,
    config: TailNeuralConfig,
) -> None:
    expected = config.pre_window_bars + config.active_bars
    if sequence.ndim != 3 or sequence.shape[1] != expected:
        raise ValueError(f"sequence must have shape [N, {expected}, F]")
    if context.ndim != 3 or context.shape[:2] != (
        sequence.shape[0],
        config.active_bars,
    ):
        raise ValueError(
            f"context must have shape [N, {config.active_bars}, C]"
        )


def distributional_tail_loss(
    logits: Tensor,
    timeout_prediction: Tensor,
    outcome: Tensor,
    timeout_target: Tensor,
    uniqueness: Tensor,
    valid: Tensor,
    *,
    timeout_weight: float = 0.50,
) -> Tensor:
    """Uniqueness-weighted outcome CE plus timeout-only Smooth-L1."""
    if logits.ndim != 3 or logits.shape[2] != 3:
        raise ValueError("logits must have shape [N, T, 3]")
    expected = logits.shape[:2]
    if any(value.shape != expected for value in (
        timeout_prediction,
        outcome,
        timeout_target,
        uniqueness,
        valid,
    )):
        raise ValueError("tail loss tensors do not align")
    valid = valid.bool()
    safe_outcome = torch.where(valid, outcome.long(), torch.zeros_like(outcome).long())
    if bool(((safe_outcome < 0) | (safe_outcome > 2)).any()):
        raise ValueError("valid outcomes must use SL=0, TP=1, timeout=2")
    weights = uniqueness * valid.to(uniqueness.dtype)
    classification = F.cross_entropy(
        logits.transpose(1, 2), safe_outcome, reduction="none"
    )
    classification_loss = (classification * weights).sum() / weights.sum().clamp_min(1.0)
    timeout_mask = valid & safe_outcome.eq(2) & torch.isfinite(timeout_target)
    if bool(timeout_mask.any()):
        timeout_point = F.smooth_l1_loss(
            timeout_prediction[timeout_mask],
            timeout_target[timeout_mask],
            reduction="none",
        )
        timeout_weights = uniqueness[timeout_mask]
        timeout_loss = (timeout_point * timeout_weights).sum() / timeout_weights.sum().clamp_min(1.0)
    else:
        timeout_loss = logits.sum() * 0.0
    return classification_loss + timeout_weight * timeout_loss


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _validate_tensors(values: TailNeuralTensors, config: TailNeuralConfig) -> None:
    windows = values.sequence.shape[0]
    expected_sequence = config.pre_window_bars + config.active_bars
    if values.sequence.ndim != 3 or values.sequence.shape[1] != expected_sequence:
        raise ValueError("sequence tensor violates the 24+12 contract")
    if values.context.ndim != 3 or values.context.shape[:2] != (
        windows,
        config.active_bars,
    ):
        raise ValueError("context tensor violates the active-step contract")
    if values.sequence_valid.shape != values.sequence.shape[:2]:
        raise ValueError("sequence_valid does not align")
    active_shape = (windows, config.active_bars)
    for name in ("decision_valid", "outcome", "timeout_target", "uniqueness"):
        if np.asarray(getattr(values, name)).shape != active_shape:
            raise ValueError(f"{name} does not align")


def _fit_scaler(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = values.shape[2]
    median = np.zeros(features, dtype=np.float32)
    scale = np.ones(features, dtype=np.float32)
    for feature in range(features):
        observed = values[:, :, feature][valid]
        observed = observed[np.isfinite(observed)]
        if observed.size == 0:
            continue
        median[feature] = np.float32(np.median(observed))
        spread = float(np.std(observed))
        if np.isfinite(spread) and spread > 1e-8:
            scale[feature] = np.float32(spread)
    return median, scale


def _apply_scaler(
    values: np.ndarray,
    valid: np.ndarray,
    median: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    out = np.asarray(values, dtype=np.float32)
    out = np.where(np.isfinite(out), out, median.reshape(1, 1, -1))
    out = (out - median.reshape(1, 1, -1)) / scale.reshape(1, 1, -1)
    out[~valid] = 0.0
    return out.astype(np.float32, copy=False)


def _torch_problem(
    values: TailNeuralTensors,
    sequence_median: np.ndarray,
    sequence_scale: np.ndarray,
    context_median: np.ndarray,
    context_scale: np.ndarray,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    sequence = _apply_scaler(
        values.sequence,
        values.sequence_valid,
        sequence_median,
        sequence_scale,
    )
    context = _apply_scaler(
        values.context,
        values.decision_valid,
        context_median,
        context_scale,
    )
    outcome = np.asarray(values.outcome, dtype=np.int64)
    valid = np.asarray(values.decision_valid, dtype=bool) & np.isin(outcome, (0, 1, 2))
    return (
        torch.from_numpy(sequence),
        torch.from_numpy(context),
        torch.from_numpy(outcome),
        torch.from_numpy(np.asarray(values.timeout_target, dtype=np.float32)),
        torch.from_numpy(np.asarray(values.uniqueness, dtype=np.float32)),
        torch.from_numpy(valid),
    )


def _problem_loss(
    model: nn.Module,
    tensors: tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor],
    rows: np.ndarray,
    config: TailNeuralConfig,
) -> Tensor:
    index = torch.as_tensor(rows, dtype=torch.long)
    sequence, context, outcome, timeout, uniqueness, valid = tensors
    logits, timeout_prediction = model(sequence[index], context[index])
    return distributional_tail_loss(
        logits,
        timeout_prediction,
        outcome[index],
        timeout[index],
        uniqueness[index],
        valid[index],
        timeout_weight=config.timeout_loss_weight,
    )


def fit_predict_neural_tail_model(
    model_name: str,
    fit_tensors: TailNeuralTensors,
    early_tensors: TailNeuralTensors,
    score_tensors: TailNeuralTensors,
    config: TailNeuralConfig = TailNeuralConfig(),
) -> NeuralTailPrediction:
    """Fit one causal neural model and return uncalibrated stepwise outputs."""
    if model_name not in {"tcn", "gru"}:
        raise ValueError(f"unsupported neural model_name: {model_name}")
    for values in (fit_tensors, early_tensors, score_tensors):
        _validate_tensors(values, config)
    sequence_median, sequence_scale = _fit_scaler(
        np.asarray(fit_tensors.sequence), np.asarray(fit_tensors.sequence_valid, dtype=bool)
    )
    context_median, context_scale = _fit_scaler(
        np.asarray(fit_tensors.context), np.asarray(fit_tensors.decision_valid, dtype=bool)
    )
    fit_problem = _torch_problem(
        fit_tensors, sequence_median, sequence_scale, context_median, context_scale
    )
    early_problem = _torch_problem(
        early_tensors, sequence_median, sequence_scale, context_median, context_scale
    )
    score_problem = _torch_problem(
        score_tensors, sequence_median, sequence_scale, context_median, context_scale
    )
    _seed_everything(config.random_seed)
    model_class = DistributionalTCN if model_name == "tcn" else DistributionalGRU
    model = model_class(
        fit_tensors.sequence.shape[2], fit_tensors.context.shape[2], config
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    generator = np.random.default_rng(config.random_seed)
    fit_rows = np.arange(len(fit_tensors.sequence), dtype=np.int64)
    early_rows = np.arange(len(early_tensors.sequence), dtype=np.int64)
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    stale = 0
    epochs_run = 0
    for epoch in range(config.epochs):
        model.train()
        shuffled = generator.permutation(fit_rows)
        for start in range(0, len(fit_rows), config.batch_size):
            rows = shuffled[start : start + config.batch_size]
            if rows.size == 0:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss = _problem_loss(model, fit_problem, rows, config)
            loss.backward()
            optimizer.step()
        model.eval()
        monitor_problem = early_problem if len(early_rows) else fit_problem
        monitor_rows = early_rows if len(early_rows) else fit_rows
        with torch.no_grad():
            monitor = _problem_loss(model, monitor_problem, monitor_rows, config)
        monitor_value = float(monitor.item())
        epochs_run = epoch + 1
        if monitor_value < best_loss - 1e-8:
            best_loss = monitor_value
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, timeout_prediction = model(score_problem[0], score_problem[1])
    return NeuralTailPrediction(
        logits=logits.cpu().numpy().astype(np.float32, copy=False),
        timeout_net_r=timeout_prediction.cpu().numpy().astype(np.float32, copy=False),
        scaler_median=np.concatenate([sequence_median, context_median]),
        scaler_scale=np.concatenate([sequence_scale, context_scale]),
        audit={
            "model_name": model_name,
            "epochs_run": epochs_run,
            "best_early_loss": best_loss,
            "fit_early_live_label_overlap": 0,
        },
    )
