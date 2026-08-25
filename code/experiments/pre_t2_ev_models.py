"""Matched XGBoost and GRU outcome-EV models for Notebook I."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


MODEL_NAMES = ("xgboost", "gru")
ENSEMBLE_NAME = "xgboost_gru_50_50"


@dataclass(frozen=True)
class PreT2Prediction:
    outcome_probabilities: np.ndarray
    timeout_gross_r: np.ndarray
    timeout_target_low: float
    timeout_target_high: float


def _validate(
    model_name: str,
    train_static: np.ndarray,
    train_sequence: np.ndarray,
    labels: np.ndarray,
    gross_r: np.ndarray,
    weights: np.ndarray,
    score_static: np.ndarray,
    score_sequence: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if model_name not in MODEL_NAMES:
        raise ValueError(f"unsupported pre-T2 model: {model_name}")
    train_static = np.asarray(train_static, dtype=float)
    score_static = np.asarray(score_static, dtype=float)
    train_sequence = np.asarray(train_sequence, dtype=np.float32)
    score_sequence = np.asarray(score_sequence, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    gross_r = np.asarray(gross_r, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if train_static.ndim != 2 or score_static.ndim != 2:
        raise ValueError("static features must be two-dimensional")
    if train_static.shape[1] != score_static.shape[1]:
        raise ValueError("static feature counts must match")
    if train_sequence.ndim != 3 or score_sequence.ndim != 3:
        raise ValueError("sequence features must be three-dimensional")
    if train_sequence.shape[1:] != score_sequence.shape[1:]:
        raise ValueError("sequence shapes must match")
    if not (len(train_static) == len(train_sequence) == len(labels) == len(gross_r) == len(weights)):
        raise ValueError("training arrays must align")
    if len(score_static) != len(score_sequence) or not len(score_static):
        raise ValueError("scoring arrays must be non-empty and aligned")
    if set(np.unique(labels)) != {0, 1, 2}:
        raise ValueError("training requires SL, TP, and timeout classes")
    if not np.isfinite(gross_r).all():
        raise ValueError("gross-R targets must be finite")
    if not np.isfinite(weights).all() or (weights <= 0.0).any():
        raise ValueError("sample weights must be positive and finite")
    return train_static, train_sequence, labels, gross_r, weights, score_static, score_sequence


def _timeout_bounds(labels: np.ndarray, gross_r: np.ndarray) -> tuple[np.ndarray, float, float]:
    timeout = labels == 2
    if int(timeout.sum()) < 3:
        raise ValueError("timeout regression needs at least three training outcomes")
    target = gross_r[timeout]
    low, high = (float(value) for value in np.quantile(target, [0.01, 0.99]))
    if low >= high:
        centre = float(target.mean())
        low, high = centre - 1e-9, centre + 1e-9
    return timeout, low, high


def _fit_xgboost(
    train_x: np.ndarray,
    labels: np.ndarray,
    gross_r: np.ndarray,
    weights: np.ndarray,
    score_x: np.ndarray,
) -> PreT2Prediction:
    from sklearn.impute import SimpleImputer
    from xgboost import XGBClassifier, XGBRegressor

    train_x = np.where(np.isfinite(train_x), train_x, np.nan)
    score_x = np.where(np.isfinite(score_x), score_x, np.nan)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    fit_x = imputer.fit_transform(train_x)
    predict_x = imputer.transform(score_x)
    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    classifier.fit(fit_x, labels, sample_weight=weights)
    probabilities = np.asarray(classifier.predict_proba(predict_x), dtype=float)
    timeout, low, high = _timeout_bounds(labels, gross_r)
    regressor = XGBRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        min_child_weight=20.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=1,
        verbosity=0,
    )
    regressor.fit(
        fit_x[timeout], np.clip(gross_r[timeout], low, high),
        sample_weight=weights[timeout],
    )
    timeout_prediction = np.clip(regressor.predict(predict_x), low, high)
    return PreT2Prediction(probabilities, np.asarray(timeout_prediction, float), low, high)


def _fit_gru(
    train_sequence: np.ndarray,
    labels: np.ndarray,
    gross_r: np.ndarray,
    weights: np.ndarray,
    score_sequence: np.ndarray,
    *,
    epochs: int,
) -> PreT2Prediction:
    import torch
    from torch import nn

    if epochs < 1:
        raise ValueError("GRU epochs must be positive")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    mean = train_sequence.mean(axis=(0, 1), keepdims=True)
    std = train_sequence.std(axis=(0, 1), keepdims=True)
    std = np.where(std > 1e-8, std, 1.0)
    train_x = ((train_sequence - mean) / std).astype(np.float32)
    score_x = ((score_sequence - mean) / std).astype(np.float32)
    timeout, low, high = _timeout_bounds(labels, gross_r)
    clipped_target = np.clip(gross_r, low, high).astype(np.float32)

    class OutcomeGRU(nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.gru = nn.GRU(channels, 16, batch_first=True)
            self.outcome = nn.Linear(16, 3)
            self.timeout = nn.Linear(16, 1)

        def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            output, _ = self.gru(values)
            state = output[:, -1]
            return self.outcome(state), self.timeout(state).squeeze(1)

    model = OutcomeGRU(train_x.shape[2])
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    x_tensor = torch.from_numpy(train_x.copy())
    y_tensor = torch.from_numpy(labels.copy())
    r_tensor = torch.from_numpy(clipped_target.copy())
    w_tensor = torch.from_numpy(weights.astype(np.float32, copy=True))
    timeout_tensor = torch.from_numpy(timeout.copy())
    batch_size = 256
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_tensor))
        for start in range(0, len(order), batch_size):
            index = order[start:start + batch_size]
            optimiser.zero_grad()
            logits, timeout_prediction = model(x_tensor[index])
            classification = nn.functional.cross_entropy(
                logits, y_tensor[index], reduction="none"
            )
            loss = (classification * w_tensor[index]).sum() / w_tensor[index].sum()
            timeout_index = timeout_tensor[index]
            if bool(timeout_index.any()):
                timeout_loss = nn.functional.smooth_l1_loss(
                    timeout_prediction[timeout_index], r_tensor[index][timeout_index],
                    reduction="none",
                )
                timeout_weights = w_tensor[index][timeout_index]
                loss = loss + 0.25 * (timeout_loss * timeout_weights).sum() / timeout_weights.sum()
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        logits, timeout_prediction = model(torch.from_numpy(score_x))
        probabilities = torch.softmax(logits, dim=1).numpy()
        timeout_values = np.clip(timeout_prediction.numpy(), low, high)
    return PreT2Prediction(probabilities, timeout_values, low, high)


def fit_predict_pre_t2_model(
    model_name: str,
    train_static: np.ndarray,
    train_sequence: np.ndarray,
    labels: np.ndarray,
    gross_r: np.ndarray,
    sample_weight: np.ndarray,
    score_static: np.ndarray,
    score_sequence: np.ndarray,
    *,
    epochs: int = 8,
) -> PreT2Prediction:
    """Fit one registered model and return matched three-class EV inputs."""
    values = _validate(
        model_name, train_static, train_sequence, labels, gross_r,
        sample_weight, score_static, score_sequence,
    )
    train_static, train_sequence, labels, gross_r, weights, score_static, score_sequence = values
    if model_name == "xgboost":
        return _fit_xgboost(train_static, labels, gross_r, weights, score_static)
    return _fit_gru(
        train_sequence, labels, gross_r, weights, score_sequence, epochs=epochs
    )


def combine_predictions(
    xgboost: PreT2Prediction, gru: PreT2Prediction
) -> PreT2Prediction:
    """Return the preregistered equal-weight probability/timeout ensemble."""
    if xgboost.outcome_probabilities.shape != gru.outcome_probabilities.shape:
        raise ValueError("ensemble outcome arrays must align")
    if xgboost.timeout_gross_r.shape != gru.timeout_gross_r.shape:
        raise ValueError("ensemble timeout arrays must align")
    return PreT2Prediction(
        outcome_probabilities=(xgboost.outcome_probabilities + gru.outcome_probabilities) / 2.0,
        timeout_gross_r=(xgboost.timeout_gross_r + gru.timeout_gross_r) / 2.0,
        timeout_target_low=min(xgboost.timeout_target_low, gru.timeout_target_low),
        timeout_target_high=max(xgboost.timeout_target_high, gru.timeout_target_high),
    )
