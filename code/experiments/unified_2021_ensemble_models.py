"""Cross-calibrated binary model heads for Notebook 04d.

Every model sees the same causal feature rows.  Raw heads are fitted only on
the fold's fit role, sigmoid calibration is fitted on the later calibration
role, and outer-test probabilities are therefore genuinely out of sample.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from xgboost import XGBClassifier

from experiments.unified_2021_ensemble_data import UnifiedDataset


MODEL_NAMES = ("xgboost", "lstm", "svm_linear")
HEAD_NAMES = ("opportunity", "side")


@dataclass(frozen=True)
class UnifiedModelConfig:
    sequence_length: int = 32
    lstm_hidden_size: int = 64
    lstm_num_layers: int = 1
    lstm_dropout: float = 0.0
    lstm_epochs: int = 10
    lstm_batch_size: int = 512
    lstm_learning_rate: float = 1e-3
    xgb_estimators: int = 300
    xgb_depth: int = 4
    xgb_learning_rate: float = 0.1
    xgb_min_child_weight: float = 50.0
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_reg_lambda: float = 10.0
    svm_c: float = 0.1
    seed: int = 42
    n_jobs: int = 1


@dataclass
class FoldPredictionResult:
    test_predictions: pd.DataFrame
    calibration_predictions: pd.DataFrame
    calibration_metrics: pd.DataFrame
    reliability_bins: pd.DataFrame
    fit_audit: pd.DataFrame


@dataclass
class OOFResult:
    predictions: pd.DataFrame
    calibration_predictions: pd.DataFrame
    calibration_metrics: pd.DataFrame
    reliability_bins: pd.DataFrame
    model_key_audit: pd.DataFrame
    fit_audit: pd.DataFrame


@dataclass
class HistoricalSnapshot:
    models: dict[tuple[str, str], object]
    calibrators: dict[tuple[str, str], "SigmoidCalibrator"]
    calibration_predictions: pd.DataFrame
    fit_audit: pd.DataFrame
    fit_max_label_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_max_label_end: pd.Timestamp
    config: UnifiedModelConfig
    history_start_position: int


def sha256_keys(keys: Iterable[object]) -> str:
    """Hash ordered row keys without relying on pandas object serialization."""
    digest = hashlib.sha256()
    for key in keys:
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class SigmoidCalibrator:
    """Natural-prevalence Platt scaling with an explicit intercept."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None

    def fit(self, raw_score, target) -> "SigmoidCalibrator":
        raw = np.asarray(raw_score, dtype=float).reshape(-1)
        y = np.asarray(target, dtype=np.int64).reshape(-1)
        if len(raw) != len(y) or not len(raw):
            raise ValueError("calibration needs equally sized non-empty arrays")
        if not np.isfinite(raw).all() or not np.isin(y, (0, 1)).all():
            raise ValueError("calibration scores and binary targets must be finite")
        if np.unique(y).size == 1:
            prevalence = (float(y.sum()) + 0.5) / (len(y) + 1.0)
            self.coef_ = np.zeros(1, dtype=float)
            self.intercept_ = np.array(
                [np.log(prevalence / (1.0 - prevalence))], dtype=float
            )
            return self
        model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2_000)
        model.fit(raw.reshape(-1, 1), y)
        self.coef_ = np.asarray(model.coef_[0], dtype=float)
        self.intercept_ = np.asarray(model.intercept_, dtype=float)
        return self

    def predict_proba(self, raw_score) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict_proba")
        raw = np.asarray(raw_score, dtype=float).reshape(-1)
        probability = _sigmoid(raw * self.coef_[0] + self.intercept_[0])
        return np.clip(probability, 1e-7, 1.0 - 1e-7)


def _clean_matrix(values) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    if output.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    output = output.copy()
    output[~np.isfinite(output)] = np.nan
    return output


def _selection(length: int, sample_mask) -> np.ndarray:
    if sample_mask is None:
        selected = np.ones(length, dtype=bool)
    else:
        selected = np.asarray(sample_mask, dtype=bool)
    if selected.shape != (length,) or not selected.any():
        raise ValueError("sample_mask must select at least one training row")
    return selected


def _selected_weight(sample_weight, selected: np.ndarray) -> np.ndarray | None:
    if sample_weight is None:
        return None
    weight = np.asarray(sample_weight, dtype=float).reshape(-1)
    if len(weight) == len(selected):
        weight = weight[selected]
    elif len(weight) != int(selected.sum()):
        raise ValueError("sample_weight must match all rows or selected rows")
    if not np.isfinite(weight).all() or (weight < 0.0).any():
        raise ValueError("sample_weight must be finite and non-negative")
    return weight


class _BinaryXGBHead:
    def __init__(self, config: UnifiedModelConfig):
        self.config = config
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.model: XGBClassifier | None = None
        self.constant_raw: float | None = None

    def fit(self, X, y, sample_mask=None, sample_weight=None):
        values = _clean_matrix(X)
        target = np.asarray(y, dtype=np.int64)
        selected = _selection(len(values), sample_mask)
        fit_x = self.imputer.fit_transform(values[selected])
        fit_y = target[selected]
        if np.unique(fit_y).size == 1:
            prevalence = (float(fit_y.sum()) + 0.5) / (len(fit_y) + 1.0)
            self.constant_raw = float(np.log(prevalence / (1.0 - prevalence)))
            return self
        weight = compute_sample_weight("balanced", fit_y)
        extra = _selected_weight(sample_weight, selected)
        if extra is not None:
            weight = weight * extra
        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=self.config.xgb_estimators,
            max_depth=self.config.xgb_depth,
            learning_rate=self.config.xgb_learning_rate,
            min_child_weight=self.config.xgb_min_child_weight,
            subsample=self.config.xgb_subsample,
            colsample_bytree=self.config.xgb_colsample_bytree,
            reg_lambda=self.config.xgb_reg_lambda,
            random_state=self.config.seed,
            tree_method="hist",
            n_jobs=self.config.n_jobs,
        )
        self.model.fit(fit_x, fit_y, sample_weight=weight)
        return self

    def predict_raw(self, X, context=None) -> np.ndarray:
        values = self.imputer.transform(_clean_matrix(X))
        if self.constant_raw is not None:
            return np.full(len(values), self.constant_raw, dtype=float)
        if self.model is None:
            raise RuntimeError("fit must be called before predict_raw")
        return np.asarray(self.model.predict(values, output_margin=True), dtype=float)


class _BinarySVMHead:
    def __init__(self, config: UnifiedModelConfig):
        self.config = config
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.model: LinearSVC | None = None
        self.constant_raw: float | None = None

    def fit(self, X, y, sample_mask=None, sample_weight=None):
        values = _clean_matrix(X)
        target = np.asarray(y, dtype=np.int64)
        selected = _selection(len(values), sample_mask)
        fit_x = self.imputer.fit_transform(values[selected])
        fit_x = self.scaler.fit_transform(fit_x)
        fit_y = target[selected]
        if np.unique(fit_y).size == 1:
            prevalence = (float(fit_y.sum()) + 0.5) / (len(fit_y) + 1.0)
            self.constant_raw = float(np.log(prevalence / (1.0 - prevalence)))
            return self
        self.model = LinearSVC(
            C=self.config.svm_c,
            class_weight="balanced",
            max_iter=50_000,
            dual=False,
            random_state=self.config.seed,
        )
        self.model.fit(
            fit_x,
            fit_y,
            sample_weight=_selected_weight(sample_weight, selected),
        )
        return self

    def predict_raw(self, X, context=None) -> np.ndarray:
        values = self.scaler.transform(self.imputer.transform(_clean_matrix(X)))
        if self.constant_raw is not None:
            return np.full(len(values), self.constant_raw, dtype=float)
        if self.model is None:
            raise RuntimeError("fit must be called before predict_raw")
        return np.asarray(self.model.decision_function(values), dtype=float)


class _BinaryLSTMNet(nn.Module):
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
        self.head = nn.Linear(hidden_size, 2)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.lstm(values)
        return self.head(encoded[:, -1])


class BinaryLSTMHead:
    """Binary LSTM whose window for row t contains no row after t."""

    def __init__(
        self,
        sequence_length: int = 32,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
        epochs: int = 10,
        batch_size: int = 512,
        learning_rate: float = 1e-3,
        seed: int = 42,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.net: _BinaryLSTMNet | None = None
        self.constant_raw: float | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _seed_all(self) -> None:
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _transform(self, values, *, fit_mask: np.ndarray | None = None) -> np.ndarray:
        matrix = _clean_matrix(values)
        if fit_mask is not None:
            self.imputer.fit(matrix[fit_mask])
            imputed = self.imputer.transform(matrix)
            selected = imputed[fit_mask]
            self.mean_ = selected.mean(axis=0)
            raw_std = selected.std(axis=0)
            self.std_ = np.where(raw_std > 1e-12, raw_std, 1.0)
        else:
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("fit must be called before transforming rows")
            imputed = self.imputer.transform(matrix)
        return np.asarray((imputed - self.mean_) / self.std_, dtype=np.float32)

    def _windows(self, values: np.ndarray, positions: np.ndarray) -> np.ndarray:
        offsets = np.arange(-self.sequence_length + 1, 1, dtype=np.int64)
        indices = np.maximum(positions[:, None] + offsets[None, :], 0)
        return np.ascontiguousarray(values[indices], dtype=np.float32)

    def fit(self, X, y, sample_mask=None, sample_weight=None):
        if self.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        values = _clean_matrix(X)
        target = np.asarray(y, dtype=np.int64)
        if target.shape != (len(values),) or not np.isin(target, (0, 1)).all():
            raise ValueError("y must contain one binary target per feature row")
        selected = _selection(len(values), sample_mask)
        transformed = self._transform(values, fit_mask=selected)
        fit_positions = np.flatnonzero(selected)
        fit_y = target[selected]
        extra_weight = _selected_weight(sample_weight, selected)
        if extra_weight is None:
            extra_weight = np.ones(len(fit_y), dtype=np.float32)
        else:
            extra_weight = np.asarray(extra_weight, dtype=np.float32)
        if np.unique(fit_y).size == 1:
            prevalence = (float(fit_y.sum()) + 0.5) / (len(fit_y) + 1.0)
            self.constant_raw = float(np.log(prevalence / (1.0 - prevalence)))
            return self

        counts = np.bincount(fit_y, minlength=2).astype(float)
        class_weight = len(fit_y) / (2.0 * np.maximum(counts, 1.0))
        self._seed_all()
        self.net = _BinaryLSTMNet(
            transformed.shape[1],
            self.hidden_size,
            self.num_layers,
            self.dropout,
        ).to(self.device)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        class_tensor = torch.tensor(
            class_weight, dtype=torch.float32, device=self.device
        )
        rng = np.random.default_rng(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            order = rng.permutation(len(fit_positions))
            for start in range(0, len(order), self.batch_size):
                batch = order[start : start + self.batch_size]
                positions = fit_positions[batch]
                inputs = torch.tensor(
                    self._windows(transformed, positions), device=self.device
                )
                targets = torch.tensor(fit_y[batch], device=self.device)
                row_weight = torch.tensor(extra_weight[batch], device=self.device)
                optimizer.zero_grad(set_to_none=True)
                per_row = nn.functional.cross_entropy(
                    self.net(inputs), targets, weight=class_tensor, reduction="none"
                )
                denominator = (class_tensor[targets] * row_weight).sum().clamp_min(
                    torch.finfo(per_row.dtype).eps
                )
                loss = (per_row * row_weight).sum() / denominator
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
        self.net.eval()
        return self

    @torch.no_grad()
    def predict_raw(self, X, context=None) -> np.ndarray:
        values = _clean_matrix(X)
        if not len(values):
            return np.empty(0, dtype=float)
        if context is None:
            context_values = np.empty((0, values.shape[1]), dtype=np.float32)
        else:
            context_values = _clean_matrix(context)[-(self.sequence_length - 1) :]
        combined = np.concatenate([context_values, values], axis=0)
        transformed = self._transform(combined)
        score_positions = np.arange(len(context_values), len(combined), dtype=np.int64)
        if self.constant_raw is not None:
            return np.full(len(score_positions), self.constant_raw, dtype=float)
        if self.net is None:
            raise RuntimeError("fit must be called before predict_raw")
        raw: list[np.ndarray] = []
        for start in range(0, len(score_positions), self.batch_size):
            positions = score_positions[start : start + self.batch_size]
            inputs = torch.tensor(
                self._windows(transformed, positions), device=self.device
            )
            logits = self.net(inputs)
            raw.append((logits[:, 1] - logits[:, 0]).cpu().numpy())
        return np.concatenate(raw).astype(float, copy=False)


def _make_model(name: str, config: UnifiedModelConfig):
    if name == "xgboost":
        return _BinaryXGBHead(config)
    if name == "svm_linear":
        return _BinarySVMHead(config)
    if name == "lstm":
        return BinaryLSTMHead(
            sequence_length=config.sequence_length,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout,
            epochs=config.lstm_epochs,
            batch_size=config.lstm_batch_size,
            learning_rate=config.lstm_learning_rate,
            seed=config.seed,
        )
    raise KeyError(f"unknown model: {name}")


def _targets(decisions: pd.DataFrame) -> dict[str, np.ndarray]:
    opportunity = pd.to_numeric(
        decisions["opportunity"], errors="coerce"
    ).fillna(0).to_numpy(np.int64)
    side = decisions["side"].eq("long").to_numpy(np.int64)
    return {"opportunity": opportunity, "side": side}


def _head_eligible(decisions: pd.DataFrame, head: str) -> np.ndarray:
    if head == "opportunity":
        return np.ones(len(decisions), dtype=bool)
    return (
        decisions["opportunity"].eq(1).fillna(False).to_numpy(bool)
        & decisions["side_eligible"].fillna(False).to_numpy(bool)
        & decisions["side"].isin(("long", "short")).to_numpy(bool)
    )


def _score_positions(
    model,
    dataset: UnifiedDataset,
    positions: np.ndarray,
    history_start: int,
    sequence_length: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    if not len(positions):
        return np.empty(0, dtype=float)
    if not isinstance(model, BinaryLSTMHead):
        return model.predict_raw(dataset.tabular[positions])
    score_start = int(positions.min())
    score_stop = int(positions.max()) + 1
    context_start = max(history_start, score_start - sequence_length + 1)
    context = dataset.tabular[context_start:score_start]
    continuous = dataset.tabular[score_start:score_stop]
    raw = model.predict_raw(continuous, context=context)
    return raw[positions - score_start]


def _ece(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    total = len(target)
    error = 0.0
    for bin_id in range(bins):
        mask = index == bin_id
        if mask.any():
            error += mask.sum() / total * abs(
                float(probability[mask].mean()) - float(target[mask].mean())
            )
    return float(error)


def _calibration_metric_row(
    fold_id: int,
    model: str,
    head: str,
    target: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
) -> dict[str, object]:
    raw_probability = np.clip(_sigmoid(raw), 1e-7, 1.0 - 1e-7)
    calibrated = np.clip(calibrated, 1e-7, 1.0 - 1e-7)
    raw_brier = brier_score_loss(target, raw_probability)
    calibrated_brier = brier_score_loss(target, calibrated)
    raw_logloss = log_loss(target, raw_probability, labels=[0, 1])
    calibrated_logloss = log_loss(target, calibrated, labels=[0, 1])
    return {
        "fold_id": fold_id,
        "model": model,
        "head": head,
        "rows": len(target),
        "prevalence": float(target.mean()),
        "raw_brier": float(raw_brier),
        "calibrated_brier": float(calibrated_brier),
        "raw_log_loss": float(raw_logloss),
        "calibrated_log_loss": float(calibrated_logloss),
        "raw_ece": _ece(target, raw_probability),
        "calibrated_ece": _ece(target, calibrated),
        "calibration_non_improving": bool(
            calibrated_brier > raw_brier + 1e-12
            or calibrated_logloss > raw_logloss + 1e-12
        ),
    }


def _reliability_rows(
    fold_id: int,
    model: str,
    head: str,
    target: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> list[dict[str, object]]:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.digitize(probability, edges[1:-1]), bins - 1)
    rows: list[dict[str, object]] = []
    for bin_id in range(bins):
        mask = index == bin_id
        rows.append(
            {
                "fold_id": fold_id,
                "model": model,
                "head": head,
                "bin": bin_id,
                "lower": edges[bin_id],
                "upper": edges[bin_id + 1],
                "rows": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()) if mask.any() else np.nan,
                "observed_rate": float(target[mask].mean()) if mask.any() else np.nan,
            }
        )
    return rows


def _fit_models(
    dataset: UnifiedDataset,
    fit_positions: np.ndarray,
    history_start: int,
    config: UnifiedModelConfig,
) -> tuple[dict[tuple[str, str], object], list[dict[str, object]]]:
    decisions = dataset.decisions
    targets = _targets(decisions)
    models: dict[tuple[str, str], object] = {}
    audits: list[dict[str, object]] = []
    history_stop = int(fit_positions.max()) + 1
    history_positions = np.arange(history_start, history_stop, dtype=np.int64)
    fit_role = np.isin(history_positions, fit_positions)
    for head in HEAD_NAMES:
        eligible = _head_eligible(decisions.iloc[history_positions], head)
        mask = fit_role & eligible
        head_positions = history_positions[mask]
        if not len(head_positions):
            raise ValueError(f"no eligible {head} fit rows")
        keys = decisions.iloc[head_positions]["row_key"]
        for model_name in MODEL_NAMES:
            model = _make_model(model_name, config)
            model.fit(
                dataset.tabular[history_positions],
                targets[head][history_positions],
                sample_mask=mask,
            )
            models[(model_name, head)] = model
            audits.append(
                {
                    "model": model_name,
                    "head": head,
                    "fit_rows": len(head_positions),
                    "fit_keys_sha256": sha256_keys(keys),
                    "fit_min_decision_time": decisions.iloc[head_positions][
                        "decision_time"
                    ].min(),
                    "fit_max_label_end": pd.to_datetime(
                        decisions.iloc[head_positions]["label_end"], utc=True
                    ).max(),
                }
            )
    return models, audits


def _fit_calibrators(
    dataset: UnifiedDataset,
    positions: np.ndarray,
    history_start: int,
    models: dict[tuple[str, str], object],
    config: UnifiedModelConfig,
    fold_id: int,
) -> tuple[
    dict[tuple[str, str], SigmoidCalibrator],
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    decisions = dataset.decisions
    base = decisions.iloc[positions].copy().reset_index(drop=True)
    targets = _targets(decisions)
    calibrators: dict[tuple[str, str], SigmoidCalibrator] = {}
    metrics: list[dict[str, object]] = []
    reliability: list[dict[str, object]] = []
    for head in HEAD_NAMES:
        eligible = _head_eligible(decisions.iloc[positions], head)
        if not eligible.any():
            raise ValueError(f"no eligible {head} calibration rows")
        for model_name in MODEL_NAMES:
            raw = _score_positions(
                models[(model_name, head)],
                dataset,
                positions,
                history_start,
                config.sequence_length,
            )
            target = targets[head][positions]
            calibrator = SigmoidCalibrator().fit(raw[eligible], target[eligible])
            probability = calibrator.predict_proba(raw)
            calibrators[(model_name, head)] = calibrator
            prefix = "opportunity" if head == "opportunity" else "long"
            base[f"raw_{prefix}_{model_name}"] = raw
            base[f"p_{prefix}_{model_name}"] = probability
            metrics.append(
                _calibration_metric_row(
                    fold_id,
                    model_name,
                    head,
                    target[eligible],
                    raw[eligible],
                    probability[eligible],
                )
            )
            reliability.extend(
                _reliability_rows(
                    fold_id,
                    model_name,
                    head,
                    target[eligible],
                    probability[eligible],
                )
            )
    base.insert(0, "fold_id", fold_id)
    return (
        calibrators,
        base,
        pd.DataFrame(metrics),
        pd.DataFrame(reliability),
    )


def _score_calibrated(
    dataset: UnifiedDataset,
    positions: np.ndarray,
    history_start: int,
    models: dict[tuple[str, str], object],
    calibrators: dict[tuple[str, str], SigmoidCalibrator],
    config: UnifiedModelConfig,
    fold_id: int,
) -> pd.DataFrame:
    output = dataset.decisions.iloc[positions].copy().reset_index(drop=True)
    output.insert(0, "fold_id", fold_id)
    for head in HEAD_NAMES:
        prefix = "opportunity" if head == "opportunity" else "long"
        for model_name in MODEL_NAMES:
            raw = _score_positions(
                models[(model_name, head)],
                dataset,
                positions,
                history_start,
                config.sequence_length,
            )
            output[f"raw_{prefix}_{model_name}"] = raw
            output[f"p_{prefix}_{model_name}"] = calibrators[
                (model_name, head)
            ].predict_proba(raw)
    return output


def fit_cross_calibrated_fold(
    dataset: UnifiedDataset,
    manifest: pd.DataFrame,
    fold_id: int,
    config: UnifiedModelConfig = UnifiedModelConfig(),
) -> FoldPredictionResult:
    """Fit raw heads, later calibrators, and score one untouched outer block."""
    fold = manifest.loc[manifest["fold_id"].eq(fold_id)].copy()
    if fold.empty:
        raise ValueError(f"manifest has no fold {fold_id}")
    positions = fold["position"].to_numpy(np.int64)
    expected_keys = dataset.decisions.iloc[positions]["row_key"].astype(str).to_numpy()
    if not np.array_equal(expected_keys, fold["row_key"].astype(str).to_numpy()):
        raise AssertionError("manifest positions do not match unified row keys")
    fit_positions = fold.loc[fold["role"].eq("fit"), "position"].to_numpy(np.int64)
    calibration_positions = fold.loc[
        fold["role"].eq("calibration"), "position"
    ].to_numpy(np.int64)
    test_positions = fold.loc[fold["role"].eq("test"), "position"].to_numpy(np.int64)
    if not len(fit_positions) or not len(calibration_positions) or not len(test_positions):
        raise ValueError("fold needs non-empty fit, calibration and test roles")
    history_start = int(positions.min())
    models, audit_rows = _fit_models(
        dataset, fit_positions, history_start, config
    )
    calibrators, calibration, metrics, reliability = _fit_calibrators(
        dataset,
        calibration_positions,
        history_start,
        models,
        config,
        fold_id,
    )
    test = _score_calibrated(
        dataset,
        test_positions,
        history_start,
        models,
        calibrators,
        config,
        fold_id,
    )
    calibration_keys = set(calibration["row_key"].astype(str))
    for row in audit_rows:
        row["fold_id"] = fold_id
        head = str(row["head"])
        eligible = _head_eligible(dataset.decisions.iloc[fit_positions], head)
        fit_keys = set(
            dataset.decisions.iloc[fit_positions[eligible]]["row_key"].astype(str)
        )
        row["calibration_overlap"] = bool(fit_keys.intersection(calibration_keys))
    return FoldPredictionResult(
        test_predictions=test,
        calibration_predictions=calibration,
        calibration_metrics=metrics,
        reliability_bins=reliability,
        fit_audit=pd.DataFrame(audit_rows),
    )


def run_cross_calibrated_oof(
    dataset: UnifiedDataset,
    manifest: pd.DataFrame,
    config: UnifiedModelConfig = UnifiedModelConfig(),
    checkpoint_dir: str | Path | None = None,
) -> OOFResult:
    """Run every manifest fold and return one aligned outer-test table."""
    del checkpoint_dir  # Checkpoint ownership belongs to the staged runner.
    results = [
        fit_cross_calibrated_fold(dataset, manifest, int(fold_id), config)
        for fold_id in sorted(manifest["fold_id"].unique())
    ]
    predictions = pd.concat(
        [result.test_predictions for result in results], ignore_index=True
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)
    if predictions["row_key"].duplicated().any():
        raise AssertionError("outer-test decisions overlap across folds")
    calibration = pd.concat(
        [result.calibration_predictions for result in results], ignore_index=True
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)
    metrics = pd.concat(
        [result.calibration_metrics for result in results], ignore_index=True
    )
    reliability = pd.concat(
        [result.reliability_bins for result in results], ignore_index=True
    )
    fit_audit = pd.concat(
        [result.fit_audit for result in results], ignore_index=True
    )
    key_hash = sha256_keys(predictions["row_key"])
    key_audit = pd.DataFrame(
        {
            "model": MODEL_NAMES,
            "rows": len(predictions),
            "keys_sha256": key_hash,
        }
    )
    return OOFResult(
        predictions=predictions,
        calibration_predictions=calibration,
        calibration_metrics=metrics,
        reliability_bins=reliability,
        model_key_audit=key_audit,
        fit_audit=fit_audit,
    )


def _utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


def fit_historical_snapshot(
    dataset: UnifiedDataset,
    cutoff,
    config: UnifiedModelConfig = UnifiedModelConfig(),
    calibration_fraction: float = 0.15,
    embargo_bars: int = 8,
) -> HistoricalSnapshot:
    """Fit a purged historical snapshot for a later forward-only stage."""
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    cutoff_time = _utc_timestamp(cutoff)
    decisions = dataset.decisions
    decision_time = pd.to_datetime(decisions["decision_time"], utc=True)
    label_end = pd.to_datetime(decisions["label_end"], utc=True, errors="coerce")
    eligible = (
        decision_time.lt(cutoff_time)
        & label_end.lt(cutoff_time)
        & decisions["path_complete"].fillna(False).astype(bool)
    ).to_numpy(bool)
    eligible_positions = np.flatnonzero(eligible)
    if len(eligible_positions) < 20:
        raise ValueError("historical snapshot needs at least 20 complete rows")
    calibration_rows = max(1, int(np.ceil(len(eligible_positions) * calibration_fraction)))
    calibration_start_position = int(eligible_positions[-calibration_rows])
    calibration_start = decision_time.iloc[calibration_start_position]
    inner_embargo_start = max(0, calibration_start_position - embargo_bars)
    fit_positions = np.flatnonzero(
        eligible
        & (np.arange(len(decisions)) < inner_embargo_start)
        & label_end.lt(calibration_start).to_numpy(bool)
    )
    calibration_positions = np.flatnonzero(
        eligible & (np.arange(len(decisions)) >= calibration_start_position)
    )
    if not len(fit_positions) or not len(calibration_positions):
        raise ValueError("purge left no fit or calibration rows")
    history_start = int(eligible_positions.min())
    models, audit_rows = _fit_models(dataset, fit_positions, history_start, config)
    calibrators, calibration, _, _ = _fit_calibrators(
        dataset,
        calibration_positions,
        history_start,
        models,
        config,
        fold_id=-1,
    )
    fit_key_set = set(dataset.decisions.iloc[fit_positions]["row_key"].astype(str))
    calibration_key_set = set(calibration["row_key"].astype(str))
    for row in audit_rows:
        row["fold_id"] = -1
        row["calibration_overlap"] = bool(
            fit_key_set.intersection(calibration_key_set)
        )
    fit_max_label_end = label_end.iloc[fit_positions].max()
    calibration_max_label_end = label_end.iloc[calibration_positions].max()
    if not fit_max_label_end < calibration_start:
        raise AssertionError("historical fit labels cross calibration start")
    if not calibration_max_label_end < cutoff_time:
        raise AssertionError("historical calibration labels cross cutoff")
    return HistoricalSnapshot(
        models=models,
        calibrators=calibrators,
        calibration_predictions=calibration,
        fit_audit=pd.DataFrame(audit_rows),
        fit_max_label_end=fit_max_label_end,
        calibration_start=calibration_start,
        calibration_max_label_end=calibration_max_label_end,
        config=config,
        history_start_position=history_start,
    )


def score_historical_snapshot(
    snapshot: HistoricalSnapshot,
    dataset: UnifiedDataset,
    positions: Iterable[int],
) -> pd.DataFrame:
    """Score later decisions without refitting the frozen historical snapshot."""
    score_positions = np.asarray(list(positions), dtype=np.int64)
    if not len(score_positions):
        return dataset.decisions.iloc[0:0].copy()
    return _score_calibrated(
        dataset,
        score_positions,
        snapshot.history_start_position,
        snapshot.models,
        snapshot.calibrators,
        snapshot.config,
        fold_id=-1,
    )


__all__ = [
    "BinaryLSTMHead",
    "FoldPredictionResult",
    "HistoricalSnapshot",
    "MODEL_NAMES",
    "OOFResult",
    "SigmoidCalibrator",
    "UnifiedModelConfig",
    "fit_cross_calibrated_fold",
    "fit_historical_snapshot",
    "run_cross_calibrated_oof",
    "score_historical_snapshot",
    "sha256_keys",
]
