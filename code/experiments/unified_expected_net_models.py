"""Calibrated conditional expected-net regressors for Notebook 04f."""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR
from torch import nn
from xgboost import XGBRegressor

from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.unified_2021_ensemble_models import (
    MODEL_NAMES,
    UnifiedModelConfig,
    sha256_keys,
)
from experiments.unified_expected_net_data import EXPECTED_NET_TARGETS


SIDES = ("long", "short")


@dataclass(frozen=True)
class NonNegativeAffineCalibrator:
    """Least-squares affine map whose slope cannot reverse a model signal."""

    slope: float
    intercept: float

    @classmethod
    def fit(cls, raw_score, target) -> "NonNegativeAffineCalibrator":
        raw = np.asarray(raw_score, dtype=float).reshape(-1)
        truth = np.asarray(target, dtype=float).reshape(-1)
        if len(raw) != len(truth) or not len(raw):
            raise ValueError("calibration needs equally sized non-empty arrays")
        if not np.isfinite(raw).all() or not np.isfinite(truth).all():
            raise ValueError("calibration scores and targets must be finite")
        centered = raw - raw.mean()
        variance = float(np.mean(centered * centered))
        if variance <= np.finfo(float).eps:
            slope = 0.0
        else:
            covariance = float(np.mean(centered * (truth - truth.mean())))
            slope = max(covariance / variance, 0.0)
        intercept = float(truth.mean() - slope * raw.mean())
        return cls(slope=float(slope), intercept=intercept)

    def predict(self, raw_score) -> np.ndarray:
        raw = np.asarray(raw_score, dtype=float)
        if not np.isfinite(raw).all():
            raise ValueError("calibration input must be finite")
        return self.intercept + self.slope * raw


@dataclass
class ExpectedNetFoldResult:
    test_predictions: pd.DataFrame
    preflight_predictions: pd.DataFrame
    calibration_predictions: pd.DataFrame
    calibration_metrics: pd.DataFrame
    fit_audit: pd.DataFrame
    target_scale_bps: float
    lstm_training_audit: dict[str, object]


def shared_target_scale(y_long, y_short) -> float:
    """Return one robust fit-only target scale for both sides, with a 1-bp floor."""
    long = np.asarray(y_long, dtype=float).reshape(-1)
    short = np.asarray(y_short, dtype=float).reshape(-1)
    joined = np.concatenate([long, short])
    if not len(joined) or not np.isfinite(joined).all():
        raise ValueError("shared target scale needs finite non-empty targets")
    return max(float(np.median(np.abs(joined))), 1.0)


def _clean_matrix(values) -> np.ndarray:
    output = np.asarray(values, dtype=np.float32)
    if output.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    output = output.copy()
    output[~np.isfinite(output)] = np.nan
    return output


def _selection(length: int, sample_mask) -> np.ndarray:
    selected = (
        np.ones(length, dtype=bool)
        if sample_mask is None
        else np.asarray(sample_mask, dtype=bool)
    )
    if selected.shape != (length,) or not selected.any():
        raise ValueError("sample_mask must select at least one training row")
    return selected


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def epoch_orders_sha256(orders: Sequence[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for order in orders:
        values = np.asarray(order, dtype=np.int64)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


class _XGBExpectedNetHead:
    def __init__(self, config: UnifiedModelConfig) -> None:
        self.config = config
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.model: XGBRegressor | None = None
        self.constant: float | None = None

    def fit(self, X, y, sample_mask=None) -> "_XGBExpectedNetHead":
        values = _clean_matrix(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        selected = _selection(len(values), sample_mask)
        fit_x = self.imputer.fit_transform(values[selected])
        fit_y = target[selected]
        if not np.isfinite(fit_y).all():
            raise ValueError("XGBoost targets must be finite")
        if float(np.ptp(fit_y)) <= np.finfo(float).eps:
            self.constant = float(fit_y.mean())
            return self
        self.model = XGBRegressor(
            objective="reg:squarederror",
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
        self.model.fit(fit_x, fit_y)
        return self

    def predict_raw(self, X) -> np.ndarray:
        values = self.imputer.transform(_clean_matrix(X))
        if self.constant is not None:
            return np.full(len(values), self.constant, dtype=float)
        if self.model is None:
            raise RuntimeError("fit must be called before predict_raw")
        return np.asarray(self.model.predict(values), dtype=float)


class _LinearSVRExpectedNetHead:
    def __init__(self, config: UnifiedModelConfig) -> None:
        self.config = config
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.scaler = StandardScaler()
        self.model: LinearSVR | None = None
        self.constant: float | None = None

    def fit(self, X, y, sample_mask=None) -> "_LinearSVRExpectedNetHead":
        values = _clean_matrix(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        selected = _selection(len(values), sample_mask)
        fit_x = self.imputer.fit_transform(values[selected])
        fit_x = self.scaler.fit_transform(fit_x)
        fit_y = target[selected]
        if not np.isfinite(fit_y).all():
            raise ValueError("LinearSVR targets must be finite")
        if float(np.ptp(fit_y)) <= np.finfo(float).eps:
            self.constant = float(fit_y.mean())
            return self
        self.model = LinearSVR(
            C=self.config.svm_c,
            epsilon=0.0,
            loss="squared_epsilon_insensitive",
            dual="auto",
            random_state=self.config.seed,
            max_iter=20_000,
        )
        self.model.fit(fit_x, fit_y)
        return self

    def predict_raw(self, X) -> np.ndarray:
        values = self.scaler.transform(self.imputer.transform(_clean_matrix(X)))
        if self.constant is not None:
            return np.full(len(values), self.constant, dtype=float)
        if self.model is None:
            raise RuntimeError("fit must be called before predict_raw")
        return np.asarray(self.model.predict(values), dtype=float)


class TwoOutputLSTMNet(nn.Module):
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
        # values: (batch, causal_sequence, features) -> (batch, LONG/SHORT)
        encoded, _ = self.lstm(values)
        return self.head(encoded[:, -1, :])


def make_lstm_initial_state(
    n_features: int,
    config: UnifiedModelConfig,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    net = TwoOutputLSTMNet(
        n_features=n_features,
        hidden_size=config.lstm_hidden_size,
        num_layers=config.lstm_num_layers,
        dropout=config.lstm_dropout,
    )
    return clone_state_dict(net.state_dict())


class TwoOutputLSTMRegressor:
    """Causal two-output LSTM trained with uniform row weights."""

    def __init__(self, config: UnifiedModelConfig) -> None:
        self.config = config
        self.imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.net: TwoOutputLSTMNet | None = None
        self.constant: np.ndarray | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_audit: dict[str, object] = {}

    def _seed_all(self) -> None:
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
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
        offsets = np.arange(-self.config.sequence_length + 1, 1, dtype=np.int64)
        indices = np.maximum(positions[:, None] + offsets[None, :], 0)
        return np.ascontiguousarray(values[indices], dtype=np.float32)

    def fit(
        self,
        X,
        y,
        sample_mask=None,
        *,
        initial_state: dict[str, torch.Tensor] | None = None,
        epoch_orders: Sequence[np.ndarray] | None = None,
        extra_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> "TwoOutputLSTMRegressor":
        if self.config.sequence_length < 1:
            raise ValueError("sequence_length must be positive")
        values = _clean_matrix(X)
        target = np.asarray(y, dtype=np.float32)
        if target.shape != (len(values), 2):
            raise ValueError("y must contain two targets per feature row")
        selected = _selection(len(values), sample_mask)
        if not np.isfinite(target[selected]).all():
            raise ValueError("selected LSTM targets must be finite")
        transformed = self._transform(values, fit_mask=selected)
        fit_positions = np.flatnonzero(selected)
        fit_y = target[selected]
        if np.all(np.ptp(fit_y, axis=0) <= np.finfo(np.float32).eps):
            self.constant = fit_y.mean(axis=0).astype(float)
            self.training_audit = {
                "initial_state_sha256": "constant",
                "final_state_sha256": "constant",
                "batch_order_sha256": "constant",
                "epochs": 0,
            }
            return self

        self._seed_all()
        self.net = TwoOutputLSTMNet(
            n_features=transformed.shape[1],
            hidden_size=self.config.lstm_hidden_size,
            num_layers=self.config.lstm_num_layers,
            dropout=self.config.lstm_dropout,
        )
        if initial_state is not None:
            self.net.load_state_dict(clone_state_dict(initial_state), strict=True)
        initial_hash = state_dict_sha256(self.net.state_dict())
        self.net = self.net.to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(), lr=self.config.lstm_learning_rate
        )
        if epoch_orders is None:
            rng = np.random.default_rng(self.config.seed)
            orders = tuple(
                rng.permutation(len(fit_positions))
                for _ in range(self.config.lstm_epochs)
            )
        else:
            orders = tuple(np.asarray(order, dtype=np.int64) for order in epoch_orders)
            if len(orders) != self.config.lstm_epochs:
                raise ValueError("epoch_orders must contain one order per epoch")
            expected = np.arange(len(fit_positions), dtype=np.int64)
            if any(
                len(order) != len(expected)
                or not np.array_equal(np.sort(order), expected)
                for order in orders
            ):
                raise ValueError("each epoch order must permute every fit row once")

        self.net.train()
        for order in orders:
            for start in range(0, len(order), self.config.lstm_batch_size):
                batch = order[start : start + self.config.lstm_batch_size]
                positions = fit_positions[batch]
                inputs = torch.tensor(
                    self._windows(transformed, positions),
                    dtype=torch.float32,
                    device=self.device,
                )
                targets = torch.tensor(
                    fit_y[batch], dtype=torch.float32, device=self.device
                )
                optimizer.zero_grad(set_to_none=True)
                predictions = self.net(inputs)
                loss = nn.functional.mse_loss(
                    predictions[:, 0], targets[:, 0]
                ) + nn.functional.mse_loss(predictions[:, 1], targets[:, 1])
                if extra_loss is not None:
                    loss = loss + extra_loss(targets, predictions)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.net.parameters(), max_norm=1.0, error_if_nonfinite=True
                )
                optimizer.step()
        self.net.eval()
        self.training_audit = {
            "initial_state_sha256": initial_hash,
            "final_state_sha256": state_dict_sha256(self.net.state_dict()),
            "batch_order_sha256": epoch_orders_sha256(orders),
            "epochs": self.config.lstm_epochs,
            "batch_size": self.config.lstm_batch_size,
            "gradient_clip": 1.0,
        }
        return self

    @torch.no_grad()
    def predict_raw(self, X, context=None) -> np.ndarray:
        values = _clean_matrix(X)
        if not len(values):
            return np.empty((0, 2), dtype=float)
        if context is None or self.config.sequence_length == 1:
            context_values = np.empty((0, values.shape[1]), dtype=np.float32)
        else:
            context_values = _clean_matrix(context)[
                -(self.config.sequence_length - 1) :
            ]
        combined = np.concatenate([context_values, values], axis=0)
        transformed = self._transform(combined)
        positions = np.arange(len(context_values), len(combined), dtype=np.int64)
        if self.constant is not None:
            return np.tile(self.constant, (len(positions), 1)).astype(float)
        if self.net is None:
            raise RuntimeError("fit must be called before predict_raw")
        batches: list[np.ndarray] = []
        for start in range(0, len(positions), self.config.lstm_batch_size):
            batch = positions[start : start + self.config.lstm_batch_size]
            inputs = torch.tensor(
                self._windows(transformed, batch),
                dtype=torch.float32,
                device=self.device,
            )
            batches.append(self.net(inputs).cpu().numpy())
        return np.concatenate(batches).astype(float, copy=False)


def _role_positions(fold: pd.DataFrame, role: str) -> np.ndarray:
    positions = fold.loc[fold["role"].eq(role), "position"].to_numpy(np.int64)
    if not len(positions):
        raise ValueError(f"fold has no {role} rows")
    return positions


def _target_matrix(decisions: pd.DataFrame) -> np.ndarray:
    values = decisions.loc[:, list(EXPECTED_NET_TARGETS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    return values


def _score_lstm_positions(
    model: TwoOutputLSTMRegressor,
    dataset: UnifiedDataset,
    positions: np.ndarray,
    history_start: int,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=np.int64)
    score_start = int(positions.min())
    score_stop = int(positions.max()) + 1
    context_start = max(
        history_start, score_start - model.config.sequence_length + 1
    )
    context = dataset.tabular[context_start:score_start]
    continuous = dataset.tabular[score_start:score_stop]
    raw = model.predict_raw(continuous, context=context)
    return raw[positions - score_start]


def _raw_role_predictions(
    dataset: UnifiedDataset,
    positions: np.ndarray,
    history_start: int,
    models: dict[tuple[str, str], object],
    lstm: TwoOutputLSTMRegressor,
    target_scale_bps: float,
) -> dict[tuple[str, str], np.ndarray]:
    raw: dict[tuple[str, str], np.ndarray] = {}
    lstm_values = _score_lstm_positions(
        lstm, dataset, positions, history_start
    ) * target_scale_bps
    for side_index, side in enumerate(SIDES):
        raw[("lstm", side)] = lstm_values[:, side_index]
        for model_name in ("xgboost", "svm_linear"):
            model = models[(model_name, side)]
            raw[(model_name, side)] = (
                model.predict_raw(dataset.tabular[positions]) * target_scale_bps
            )
    return raw


def _regression_metric_row(
    fold_id: int,
    model_name: str,
    side: str,
    target: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    calibrator: NonNegativeAffineCalibrator,
) -> dict[str, object]:
    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        if np.ptp(left) <= np.finfo(float).eps or np.ptp(right) <= np.finfo(float).eps:
            return float("nan")
        return float(np.corrcoef(left, right)[0, 1])

    raw_error = raw - target
    calibrated_error = calibrated - target
    return {
        "fold_id": fold_id,
        "model": model_name,
        "target": f"target_{side}_bps",
        "side": side,
        "rows": len(target),
        "target_mean_bps": float(target.mean()),
        "raw_mae_bps": float(np.mean(np.abs(raw_error))),
        "calibrated_mae_bps": float(np.mean(np.abs(calibrated_error))),
        "raw_rmse_bps": float(np.sqrt(np.mean(raw_error * raw_error))),
        "calibrated_rmse_bps": float(
            np.sqrt(np.mean(calibrated_error * calibrated_error))
        ),
        "raw_correlation": correlation(raw, target),
        "calibrated_correlation": correlation(calibrated, target),
        "slope": calibrator.slope,
        "intercept_bps": calibrator.intercept,
    }


def _score_role(
    dataset: UnifiedDataset,
    positions: np.ndarray,
    fold_id: int,
    source_role: str,
    history_start: int,
    models: dict[tuple[str, str], object],
    lstm: TwoOutputLSTMRegressor,
    calibrators: dict[tuple[str, str], NonNegativeAffineCalibrator],
    target_scale_bps: float,
) -> pd.DataFrame:
    output = dataset.decisions.iloc[positions].copy().reset_index(drop=True)
    output.insert(0, "source_role", source_role)
    output.insert(0, "fold_id", fold_id)
    raw = _raw_role_predictions(
        dataset,
        positions,
        history_start,
        models,
        lstm,
        target_scale_bps,
    )
    for side in SIDES:
        for model_name in MODEL_NAMES:
            values = raw[(model_name, side)]
            output[f"raw_{side}_{model_name}"] = values
            output[f"pred_{side}_{model_name}"] = calibrators[
                (model_name, side)
            ].predict(values)
    return output


def fit_expected_net_fold(
    dataset: UnifiedDataset,
    manifest: pd.DataFrame,
    fold_id: int,
    config: UnifiedModelConfig = UnifiedModelConfig(),
    *,
    lstm_initial_state: dict[str, torch.Tensor] | None = None,
    lstm_epoch_orders: Sequence[np.ndarray] | None = None,
    lstm_extra_loss: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> ExpectedNetFoldResult:
    """Fit three model families, calibrate later, and score preflight/test once."""
    fold = manifest.loc[manifest["fold_id"].eq(fold_id)].sort_values(
        "position", kind="stable"
    )
    if fold.empty:
        raise ValueError(f"manifest has no fold {fold_id}")
    positions = fold["position"].to_numpy(np.int64)
    expected_keys = dataset.decisions.iloc[positions]["row_key"].astype(str).to_numpy()
    if not np.array_equal(expected_keys, fold["row_key"].astype(str).to_numpy()):
        raise AssertionError("manifest positions do not match expected-net row keys")

    fit_positions = _role_positions(fold, "fit")
    calibration_positions = _role_positions(fold, "probability_calibration")
    preflight_positions = _role_positions(fold, "fixed_policy_preflight")
    test_positions = _role_positions(fold, "test")
    history_start = int(fold["position"].min())
    history_stop = int(fit_positions.max()) + 1
    history_positions = np.arange(history_start, history_stop, dtype=np.int64)
    fit_mask = np.isin(history_positions, fit_positions)

    target = _target_matrix(dataset.decisions)
    if not np.isfinite(target[fit_positions]).all():
        raise ValueError("fit expected-net targets must be finite")
    target_scale_bps = shared_target_scale(
        target[fit_positions, 0], target[fit_positions, 1]
    )
    scaled_target = target / target_scale_bps

    models: dict[tuple[str, str], object] = {}
    for side_index, side in enumerate(SIDES):
        xgb = _XGBExpectedNetHead(config).fit(
            dataset.tabular[history_positions],
            scaled_target[history_positions, side_index],
            sample_mask=fit_mask,
        )
        svm = _LinearSVRExpectedNetHead(config).fit(
            dataset.tabular[history_positions],
            scaled_target[history_positions, side_index],
            sample_mask=fit_mask,
        )
        models[("xgboost", side)] = xgb
        models[("svm_linear", side)] = svm

    lstm = TwoOutputLSTMRegressor(config).fit(
        dataset.tabular[history_positions],
        scaled_target[history_positions],
        sample_mask=fit_mask,
        initial_state=lstm_initial_state,
        epoch_orders=lstm_epoch_orders,
        extra_loss=lstm_extra_loss,
    )

    decisions = dataset.decisions
    fit_keys = decisions.iloc[fit_positions]["row_key"].astype(str)
    fit_key_set = set(fit_keys)
    later = {
        "probability_calibration": set(
            decisions.iloc[calibration_positions]["row_key"].astype(str)
        ),
        "fixed_policy_preflight": set(
            decisions.iloc[preflight_positions]["row_key"].astype(str)
        ),
        "test": set(decisions.iloc[test_positions]["row_key"].astype(str)),
    }
    audit_rows: list[dict[str, object]] = []
    for side in SIDES:
        for model_name in MODEL_NAMES:
            audit_rows.append(
                {
                    "fold_id": fold_id,
                    "model": model_name,
                    "target": f"target_{side}_bps",
                    "fit_rows": len(fit_positions),
                    "fit_keys_sha256": sha256_keys(fit_keys),
                    "fit_min_decision_time": decisions.iloc[fit_positions][
                        "decision_time"
                    ].min(),
                    "fit_max_label_end": pd.to_datetime(
                        decisions.iloc[fit_positions]["label_end"], utc=True
                    ).max(),
                    "probability_calibration_overlap": bool(
                        fit_key_set.intersection(later["probability_calibration"])
                    ),
                    "fixed_policy_preflight_overlap": bool(
                        fit_key_set.intersection(later["fixed_policy_preflight"])
                    ),
                    "test_overlap": bool(fit_key_set.intersection(later["test"])),
                    "uniform_row_weights": True,
                    "target_scale_bps": target_scale_bps,
                }
            )

    calibration = decisions.iloc[calibration_positions].copy().reset_index(drop=True)
    calibration.insert(0, "source_role", "probability_calibration")
    calibration.insert(0, "fold_id", fold_id)
    raw = _raw_role_predictions(
        dataset,
        calibration_positions,
        history_start,
        models,
        lstm,
        target_scale_bps,
    )
    calibrators: dict[tuple[str, str], NonNegativeAffineCalibrator] = {}
    metric_rows: list[dict[str, object]] = []
    for side_index, side in enumerate(SIDES):
        truth = target[calibration_positions, side_index]
        if not np.isfinite(truth).all():
            raise ValueError("calibration expected-net targets must be finite")
        for model_name in MODEL_NAMES:
            values = raw[(model_name, side)]
            calibrator = NonNegativeAffineCalibrator.fit(values, truth)
            calibrated = calibrator.predict(values)
            calibrators[(model_name, side)] = calibrator
            calibration[f"raw_{side}_{model_name}"] = values
            calibration[f"pred_{side}_{model_name}"] = calibrated
            metric_rows.append(
                _regression_metric_row(
                    fold_id,
                    model_name,
                    side,
                    truth,
                    values,
                    calibrated,
                    calibrator,
                )
            )

    preflight = _score_role(
        dataset,
        preflight_positions,
        fold_id,
        "fixed_policy_preflight",
        history_start,
        models,
        lstm,
        calibrators,
        target_scale_bps,
    )
    test = _score_role(
        dataset,
        test_positions,
        fold_id,
        "test",
        history_start,
        models,
        lstm,
        calibrators,
        target_scale_bps,
    )
    return ExpectedNetFoldResult(
        test_predictions=test,
        preflight_predictions=preflight,
        calibration_predictions=calibration,
        calibration_metrics=pd.DataFrame(metric_rows),
        fit_audit=pd.DataFrame(audit_rows),
        target_scale_bps=target_scale_bps,
        lstm_training_audit=dict(lstm.training_audit),
    )


def combine_expected_net_folds(
    results: Iterable[ExpectedNetFoldResult],
) -> dict[str, pd.DataFrame]:
    """Combine fold outputs while preserving unique chronological outer keys."""
    fold_results = list(results)
    if not fold_results:
        raise ValueError("at least one fold result is required")

    def combine(attribute: str, *, sort: bool = False) -> pd.DataFrame:
        output = pd.concat(
            [getattr(result, attribute) for result in fold_results],
            ignore_index=True,
        )
        if sort and "decision_time" in output:
            output = output.sort_values("decision_time", kind="stable").reset_index(
                drop=True
            )
        return output

    test = combine("test_predictions", sort=True)
    if test["row_key"].duplicated().any():
        raise AssertionError("outer expected-net test keys overlap across folds")
    return {
        "test_predictions": test,
        "preflight_predictions": combine("preflight_predictions", sort=True),
        "calibration_predictions": combine("calibration_predictions", sort=True),
        "calibration_metrics": combine("calibration_metrics"),
        "fit_audit": combine("fit_audit"),
    }


__all__ = [
    "ExpectedNetFoldResult",
    "NonNegativeAffineCalibrator",
    "SIDES",
    "TwoOutputLSTMNet",
    "TwoOutputLSTMRegressor",
    "clone_state_dict",
    "combine_expected_net_folds",
    "epoch_orders_sha256",
    "fit_expected_net_fold",
    "make_lstm_initial_state",
    "shared_target_scale",
    "state_dict_sha256",
]
