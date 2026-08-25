"""Paired MSE-versus-GMADL LSTM shadow for Notebook 04g."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from experiments.unified_2021_ensemble_models import UnifiedModelConfig
from experiments.unified_expected_net_models import (
    TwoOutputLSTMRegressor,
    clone_state_dict,
    epoch_orders_sha256,
    make_lstm_initial_state,
    state_dict_sha256,
)


GMADL_ALPHA = 1.0
GMADL_BETA = 1.0
GMADL_LAMBDA = 0.25


@dataclass
class PairedLSTMResult:
    control: TwoOutputLSTMRegressor
    candidate: TwoOutputLSTMRegressor
    initial_state_sha256: str
    batch_order_sha256: str
    seed: int
    alpha: float
    beta: float
    lambda_gmadl: float


def gmadl_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    *,
    alpha: float = GMADL_ALPHA,
    beta: float = GMADL_BETA,
) -> torch.Tensor:
    """Return the registered differentiable side-spread agreement term."""
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        raise ValueError("GMADL needs equal two-dimensional target/prediction tensors")
    if y_true.shape[1] != 2 or y_true.shape[0] < 1:
        raise ValueError("GMADL needs non-empty LONG/SHORT pairs")
    if not np.isfinite(alpha) or alpha <= 0.0:
        raise ValueError("GMADL alpha must be finite and positive")
    if not np.isfinite(beta) or beta <= 0.0:
        raise ValueError("GMADL beta must be finite and positive")
    true_spread = y_true[:, 0] - y_true[:, 1]
    predicted_spread = y_pred[:, 0] - y_pred[:, 1]
    agreement = torch.sigmoid(float(alpha) * true_spread * predicted_spread) - 0.5
    magnitude = torch.abs(true_spread).pow(float(beta))
    return torch.mean(-agreement * magnitude)


def _paired_epoch_orders(
    selected_rows: int,
    config: UnifiedModelConfig,
) -> tuple[np.ndarray, ...]:
    if selected_rows < 1:
        raise ValueError("paired LSTM needs at least one selected row")
    rng = np.random.default_rng(config.seed)
    return tuple(
        rng.permutation(selected_rows) for _ in range(config.lstm_epochs)
    )


def fit_paired_lstm_shadow(
    features,
    scaled_targets,
    sample_mask,
    *,
    config: UnifiedModelConfig = UnifiedModelConfig(),
) -> PairedLSTMResult:
    """Fit control and candidate from identical state, rows, and batch order."""
    values = np.asarray(features, dtype=np.float32)
    selected = np.asarray(sample_mask, dtype=bool)
    if values.ndim != 2 or selected.shape != (len(values),) or not selected.any():
        raise ValueError("paired LSTM inputs need a non-empty one-dimensional fit mask")
    initial_state = make_lstm_initial_state(values.shape[1], config)
    initial_hash = state_dict_sha256(initial_state)
    orders = _paired_epoch_orders(int(selected.sum()), config)
    order_hash = epoch_orders_sha256(orders)

    control = TwoOutputLSTMRegressor(config).fit(
        values,
        scaled_targets,
        sample_mask=selected,
        initial_state=clone_state_dict(initial_state),
        epoch_orders=orders,
    )

    def fixed_candidate_term(
        truth: torch.Tensor,
        prediction: torch.Tensor,
    ) -> torch.Tensor:
        return GMADL_LAMBDA * gmadl_loss(
            truth,
            prediction,
            alpha=GMADL_ALPHA,
            beta=GMADL_BETA,
        )

    candidate = TwoOutputLSTMRegressor(config).fit(
        values,
        scaled_targets,
        sample_mask=selected,
        initial_state=clone_state_dict(initial_state),
        epoch_orders=orders,
        extra_loss=fixed_candidate_term,
    )
    for arm_name, arm in (("control", control), ("candidate", candidate)):
        if arm.training_audit.get("initial_state_sha256") != initial_hash:
            raise AssertionError(f"{arm_name} LSTM did not use the paired initial state")
        if arm.training_audit.get("batch_order_sha256") != order_hash:
            raise AssertionError(f"{arm_name} LSTM did not use the paired batch order")
    return PairedLSTMResult(
        control=control,
        candidate=candidate,
        initial_state_sha256=initial_hash,
        batch_order_sha256=order_hash,
        seed=config.seed,
        alpha=GMADL_ALPHA,
        beta=GMADL_BETA,
        lambda_gmadl=GMADL_LAMBDA,
    )


def evaluate_shadow_admission(
    control_summary: dict[str, object],
    candidate_summary: dict[str, object],
    fold_deltas: pd.DataFrame,
    *,
    control_top_quartile_accuracy: float,
    candidate_top_quartile_accuracy: float,
) -> dict[str, object]:
    """Apply the registered one-seed shadow-only absolute and paired gates."""
    required_delta = {"fold_id", "net_delta"}
    missing = sorted(required_delta.difference(fold_deltas.columns))
    if missing:
        raise ValueError(f"shadow fold deltas lack columns: {missing}")
    delta = pd.to_numeric(fold_deltas["net_delta"], errors="raise").to_numpy(float)
    if len(delta) != 5 or not np.isfinite(delta).all():
        raise ValueError("shadow admission needs five finite fold deltas")
    for value, name in (
        (control_top_quartile_accuracy, "control_top_quartile_accuracy"),
        (candidate_top_quartile_accuracy, "candidate_top_quartile_accuracy"),
    ):
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")

    clean_keys = (
        "leakage_clean",
        "reconciliation_clean",
        "path_contract_clean",
        "cost_contract_clean",
    )
    control_absolute = bool(control_summary.get("development_pass", False))
    candidate_absolute = bool(candidate_summary.get("development_pass", False))
    candidate_clean = all(bool(candidate_summary.get(key, False)) for key in clean_keys)
    control_clean = all(bool(control_summary.get(key, False)) for key in clean_keys)
    count_noninferior = int(candidate_summary.get("trades", -1)) >= int(
        control_summary.get("trades", 0)
    )
    total_noninferior = float(candidate_summary.get("net_return", -np.inf)) >= float(
        control_summary.get("net_return", np.inf)
    )
    long_noninferior = float(
        candidate_summary.get("long_net_return", -np.inf)
    ) >= float(control_summary.get("long_net_return", np.inf))
    short_noninferior = float(
        candidate_summary.get("short_net_return", -np.inf)
    ) >= float(control_summary.get("short_net_return", np.inf))
    nonnegative_delta_folds = int((delta >= 0.0).sum())
    side_choice_improved = (
        candidate_top_quartile_accuracy > control_top_quartile_accuracy
    )
    zero_solo = (
        int(control_summary.get("xgboost_solo_trades", -1)) == 0
        and int(candidate_summary.get("xgboost_solo_trades", -1)) == 0
    )
    gates = {
        "control_absolute_gate": control_absolute,
        "candidate_absolute_gate": candidate_absolute,
        "control_clean_gate": control_clean,
        "candidate_clean_gate": candidate_clean,
        "count_noninferiority_gate": count_noninferior,
        "total_net_noninferiority_gate": total_noninferior,
        "long_net_noninferiority_gate": long_noninferior,
        "short_net_noninferiority_gate": short_noninferior,
        "fold_delta_gate": nonnegative_delta_folds >= 3,
        "top_quartile_side_choice_gate": side_choice_improved,
        "xgboost_solo_zero_gate": zero_solo,
    }
    admitted = bool(all(gates.values()))
    return {
        **gates,
        "nonnegative_delta_folds": nonnegative_delta_folds,
        "control_top_quartile_accuracy": float(control_top_quartile_accuracy),
        "candidate_top_quartile_accuracy": float(candidate_top_quartile_accuracy),
        "top_quartile_accuracy_delta": float(
            candidate_top_quartile_accuracy - control_top_quartile_accuracy
        ),
        "shadow_admit_for_reflection": admitted,
        "shadow_only": True,
        "h1_loaded": False,
        "forward_loaded": False,
        "lockbox_2026_q2_used": False,
    }


__all__ = [
    "GMADL_ALPHA",
    "GMADL_BETA",
    "GMADL_LAMBDA",
    "PairedLSTMResult",
    "evaluate_shadow_admission",
    "fit_paired_lstm_shadow",
    "gmadl_loss",
]
