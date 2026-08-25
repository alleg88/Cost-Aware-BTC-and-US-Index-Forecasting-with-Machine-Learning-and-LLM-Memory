from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.unified_2021_ensemble_models import UnifiedModelConfig


def test_registered_gmadl_formula_is_finite_differentiable_and_exact():
    from experiments.lstm_gmadl_shadow import gmadl_loss

    truth = torch.tensor([[2.0, -1.0], [-1.0, 1.0]], dtype=torch.float64)
    prediction = torch.tensor(
        [[1.0, -1.0], [0.5, 0.1]], dtype=torch.float64, requires_grad=True
    )
    r = truth[:, 0] - truth[:, 1]
    p = prediction[:, 0] - prediction[:, 1]
    expected = torch.mean(-(torch.sigmoid(r * p) - 0.5) * torch.abs(r))

    actual = gmadl_loss(truth, prediction, alpha=1.0, beta=1.0)
    actual.backward()

    assert actual.item() == pytest.approx(expected.item())
    assert torch.isfinite(actual)
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_paired_lstm_arms_share_initial_state_and_batch_order():
    from experiments.lstm_gmadl_shadow import fit_paired_lstm_shadow

    rng = np.random.default_rng(31)
    features = rng.normal(size=(32, 4)).astype(np.float32)
    targets = rng.normal(size=(32, 2)).astype(np.float32)
    selected = np.ones(32, dtype=bool)
    selected[:3] = False
    targets[:3] = np.nan
    config = UnifiedModelConfig(
        sequence_length=4,
        lstm_hidden_size=4,
        lstm_epochs=1,
        lstm_batch_size=16,
    )

    paired = fit_paired_lstm_shadow(
        features,
        targets,
        selected,
        config=config,
    )

    assert paired.initial_state_sha256 == paired.control.training_audit[
        "initial_state_sha256"
    ]
    assert paired.initial_state_sha256 == paired.candidate.training_audit[
        "initial_state_sha256"
    ]
    assert paired.batch_order_sha256 == paired.control.training_audit[
        "batch_order_sha256"
    ]
    assert paired.batch_order_sha256 == paired.candidate.training_audit[
        "batch_order_sha256"
    ]
    assert paired.seed == 42
    assert paired.lambda_gmadl == pytest.approx(0.25)


def _summary(*, passed: bool = True) -> dict[str, object]:
    return {
        "development_pass": passed,
        "trades": 150,
        "net_return": 0.10,
        "long_net_return": 0.04,
        "short_net_return": 0.06,
        "xgboost_solo_trades": 0,
        "leakage_clean": True,
        "reconciliation_clean": True,
        "path_contract_clean": True,
        "cost_contract_clean": True,
    }


def test_shadow_admission_requires_absolute_and_paired_noninferiority_gates():
    from experiments.lstm_gmadl_shadow import evaluate_shadow_admission

    control = _summary()
    candidate = dict(
        _summary(),
        trades=151,
        net_return=0.11,
        long_net_return=0.04,
        short_net_return=0.07,
    )
    fold_deltas = pd.DataFrame(
        {"fold_id": range(5), "net_delta": [0.01, 0.0, 0.02, -0.01, -0.02]}
    )

    result = evaluate_shadow_admission(
        control,
        candidate,
        fold_deltas,
        control_top_quartile_accuracy=0.55,
        candidate_top_quartile_accuracy=0.58,
    )

    assert result["nonnegative_delta_folds"] == 3
    assert result["shadow_admit_for_reflection"]

    failed_control = dict(control, development_pass=False)
    rejected = evaluate_shadow_admission(
        failed_control,
        candidate,
        fold_deltas,
        control_top_quartile_accuracy=0.55,
        candidate_top_quartile_accuracy=0.58,
    )
    assert not rejected["control_absolute_gate"]
    assert not rejected["shadow_admit_for_reflection"]

