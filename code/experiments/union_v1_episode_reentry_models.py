"""Deterministic Union-v1-style member models for Notebook 04h."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn

from experiments.unified_2021_ensemble_data import UnifiedDataset


@dataclass(frozen=True)
class UnionReentryModelConfig:
    lstm_sequence_length: int = 32
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    lstm_epochs: int = 10
    lstm_batch_size: int = 512
    lstm_learning_rate: float = 1e-3
    lstm_tau: float = 0.75
    svm_c: float = 0.1
    svm_tau: float = 0.0
    seed: int = 42


@dataclass
class UnionFoldPredictions:
    predictions: pd.DataFrame
    training_audit: dict[str, object]


class _ThreeClassLSTM(nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 3)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(values)
        return self.head(encoded[:, -1])


def _key_hash(keys: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _clean_matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("Union member features must be a two-dimensional matrix")
    matrix = matrix.copy()
    matrix[~np.isfinite(matrix)] = np.nan
    return matrix


def _class_counts(target: np.ndarray) -> str:
    counts = np.bincount(np.asarray(target, dtype=np.int64), minlength=3)
    return json.dumps({str(index): int(value) for index, value in enumerate(counts)})


def _causal_windows(
    block_values: np.ndarray,
    local_positions: np.ndarray,
    sequence_length: int,
) -> np.ndarray:
    offsets = np.arange(-sequence_length + 1, 1, dtype=np.int64)
    indices = np.maximum(local_positions[:, None] + offsets[None, :], 0)
    return np.ascontiguousarray(block_values[indices], dtype=np.float32)


def _seed_deterministically(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _fit_lstm_probabilities(
    block_values: np.ndarray,
    fit_local: np.ndarray,
    test_local: np.ndarray,
    target: np.ndarray,
    config: UnionReentryModelConfig,
) -> tuple[np.ndarray, str]:
    if config.lstm_sequence_length < 1:
        raise ValueError("lstm_sequence_length must be positive")
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    imputer.fit(block_values[fit_local])
    imputed = np.asarray(imputer.transform(block_values), dtype=np.float32)
    fit_values = imputed[fit_local]
    mean = fit_values.mean(axis=0)
    raw_std = fit_values.std(axis=0)
    std = np.where(raw_std > 1e-12, raw_std, 1.0)
    transformed = np.asarray((imputed - mean) / std, dtype=np.float32)
    fit_target = np.asarray(target[fit_local], dtype=np.int64)
    classes = np.unique(fit_target)
    if classes.size == 1:
        probability = np.zeros((len(test_local), 3), dtype=np.float32)
        probability[:, int(classes[0])] = 1.0
        return probability, "constant"

    _seed_deterministically(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _ThreeClassLSTM(
        transformed.shape[1],
        config.lstm_hidden_size,
        config.lstm_num_layers,
        config.lstm_dropout,
    ).to(device)
    counts = np.bincount(fit_target, minlength=3).astype(float)
    weights = len(fit_target) / (3.0 * np.maximum(counts, 1.0))
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lstm_learning_rate)
    rng = np.random.default_rng(config.seed)
    model.train()
    for _ in range(config.lstm_epochs):
        order = rng.permutation(len(fit_local))
        for start in range(0, len(order), config.lstm_batch_size):
            batch = order[start : start + config.lstm_batch_size]
            inputs = torch.tensor(
                _causal_windows(
                    transformed,
                    fit_local[batch],
                    config.lstm_sequence_length,
                ),
                device=device,
            )
            targets = torch.tensor(fit_target[batch], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(
                model(inputs), targets, weight=class_weight
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    model.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(test_local), config.lstm_batch_size):
            positions = test_local[start : start + config.lstm_batch_size]
            inputs = torch.tensor(
                _causal_windows(
                    transformed,
                    positions,
                    config.lstm_sequence_length,
                ),
                device=device,
            )
            chunks.append(torch.softmax(model(inputs), dim=1).cpu().numpy())
    probability = np.concatenate(chunks).astype(np.float32, copy=False)
    if not np.isfinite(probability).all():
        raise AssertionError("LSTM emitted non-finite probabilities")
    return probability, device.type


def _fit_svm_classes(
    fit_values: np.ndarray,
    test_values: np.ndarray,
    fit_target: np.ndarray,
    config: UnionReentryModelConfig,
) -> tuple[np.ndarray, bool]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    transformed_fit = scaler.fit_transform(imputer.fit_transform(fit_values))
    transformed_test = scaler.transform(imputer.transform(test_values))
    classes = np.unique(fit_target)
    if classes.size == 1:
        return np.full(len(test_values), int(classes[0]), dtype=np.int8), True
    model = LinearSVC(
        C=config.svm_c,
        class_weight="balanced",
        max_iter=50_000,
        dual=False,
        random_state=config.seed,
    )
    model.fit(transformed_fit, fit_target)
    return np.asarray(model.predict(transformed_test), dtype=np.int8), False


def fit_union_reentry_fold(
    dataset: UnifiedDataset,
    manifest: pd.DataFrame,
    fold_id: int,
    config: UnionReentryModelConfig = UnionReentryModelConfig(),
) -> UnionFoldPredictions:
    """Fit the two frozen Union-style members and score one untouched fold test."""
    fold = manifest.loc[manifest["fold_id"].eq(fold_id)].copy()
    if fold.empty:
        raise ValueError(f"Union manifest has no fold {fold_id}")
    fold = fold.sort_values("position", kind="stable").reset_index(drop=True)
    positions = fold["position"].to_numpy(np.int64)
    if not np.array_equal(positions, np.arange(positions.min(), positions.max() + 1)):
        raise AssertionError("Union fold must be one contiguous block")
    expected_keys = dataset.decisions.iloc[positions]["row_key"].astype(str).to_numpy()
    if not np.array_equal(expected_keys, fold["row_key"].astype(str).to_numpy()):
        raise AssertionError("Union manifest positions do not match dataset keys")

    fit_mask = fold["role"].eq("fit").to_numpy(bool)
    test_mask = fold["role"].eq("test").to_numpy(bool)
    if not fit_mask.any() or not test_mask.any():
        raise ValueError("Union fold needs non-empty fit and test roles")
    block_start = int(positions.min())
    fit_positions = positions[fit_mask]
    test_positions = positions[test_mask]
    fit_local = fit_positions - block_start
    test_local = test_positions - block_start
    values = _clean_matrix(dataset.tabular)
    block_values = values[positions]

    lstm_target = pd.to_numeric(
        fold["target_dz55"], errors="coerce"
    ).to_numpy(np.int64)
    svm_target = pd.to_numeric(
        fold["target_dz75"], errors="coerce"
    ).to_numpy(np.int64)
    if not np.isin(lstm_target[fit_mask | test_mask], (0, 1, 2)).all():
        raise ValueError("LSTM fit/test targets must be three-class labels")
    if not np.isin(svm_target[fit_mask | test_mask], (0, 1, 2)).all():
        raise ValueError("SVM fit/test targets must be three-class labels")

    lstm_probability, device = _fit_lstm_probabilities(
        block_values,
        fit_local,
        test_local,
        lstm_target,
        config,
    )
    svm_prediction, svm_constant = _fit_svm_classes(
        values[fit_positions],
        values[test_positions],
        svm_target[fit_mask],
        config,
    )
    output = dataset.decisions.iloc[test_positions].copy().reset_index(drop=True)
    output.insert(0, "fold_id", int(fold_id))
    output["target_dz55"] = lstm_target[test_mask]
    output["target_dz75"] = svm_target[test_mask]
    output["p_short_lstm"] = lstm_probability[:, 0]
    output["p_flat_lstm"] = lstm_probability[:, 1]
    output["p_long_lstm"] = lstm_probability[:, 2]
    output["pred_lstm"] = np.argmax(lstm_probability, axis=1).astype(np.int8)
    output["pred_svm_linear"] = svm_prediction

    audit: dict[str, object] = {
        "fold_id": int(fold_id),
        "block_start": block_start,
        "block_stop_exclusive": int(positions.max()) + 1,
        "fit_rows": int(fit_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "fit_keys_sha256": _key_hash(fold.loc[fit_mask, "row_key"]),
        "test_keys_sha256": _key_hash(output["row_key"]),
        "lstm_fit_class_counts": _class_counts(lstm_target[fit_mask]),
        "svm_fit_class_counts": _class_counts(svm_target[fit_mask]),
        "seed": config.seed,
        "device": device,
        "lstm_context_min_position": block_start,
        "lstm_context_max_position": int(test_positions.max()),
        "lstm_probabilities_finite": bool(np.isfinite(lstm_probability).all()),
        "svm_constant": bool(svm_constant),
        "preprocessing_fit_only": True,
    }
    return UnionFoldPredictions(predictions=output, training_audit=audit)


def combine_union_reentry_folds(
    results: Iterable[UnionFoldPredictions],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Combine chronological OOF predictions and reject key drift or overlap."""
    materialized = list(results)
    if not materialized:
        raise ValueError("At least one Union fold result is required")
    predictions = pd.concat(
        [result.predictions for result in materialized], ignore_index=True
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)
    if predictions["row_key"].duplicated().any():
        raise AssertionError("Duplicate Union OOF row keys")
    probability = predictions[
        ["p_short_lstm", "p_flat_lstm", "p_long_lstm"]
    ].to_numpy(float)
    if not np.isfinite(probability).all():
        raise AssertionError("Combined Union LSTM probabilities are not finite")
    audit = pd.DataFrame([result.training_audit for result in materialized]).sort_values(
        "fold_id", kind="stable"
    ).reset_index(drop=True)
    return predictions, audit


__all__ = [
    "UnionFoldPredictions",
    "UnionReentryModelConfig",
    "combine_union_reentry_folds",
    "fit_union_reentry_fold",
]
