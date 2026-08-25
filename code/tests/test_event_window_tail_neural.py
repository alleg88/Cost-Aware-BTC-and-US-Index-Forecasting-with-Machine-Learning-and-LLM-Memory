from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.event_window_tail_neural import (
    DistributionalGRU,
    DistributionalTCN,
    TailNeuralConfig,
    TailNeuralTensors,
    distributional_tail_loss,
    fit_predict_neural_tail_model,
)


def _sequence(*, windows: int = 2, features: int = 5) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    return torch.randn(windows, 36, features, generator=generator)


def _context(*, windows: int = 2, features: int = 4) -> torch.Tensor:
    generator = torch.Generator().manual_seed(11)
    return torch.randn(windows, 12, features, generator=generator)


def _mutate_after_step(sequence: torch.Tensor, step: int) -> torch.Tensor:
    changed = sequence.clone()
    changed[:, 24 + step + 1 :, :] += 1000.0
    return changed


@pytest.mark.parametrize("model_cls", [DistributionalTCN, DistributionalGRU])
def test_neural_model_emits_stepwise_distribution_and_timeout(model_cls):
    logits, timeout = model_cls(5, 4, TailNeuralConfig()).eval()(
        _sequence(), _context()
    )
    assert logits.shape == (2, 12, 3)
    assert timeout.shape == (2, 12)


@pytest.mark.parametrize("model_cls", [DistributionalTCN, DistributionalGRU])
def test_future_active_mutation_cannot_change_prior_output(model_cls):
    torch.manual_seed(3)
    model = model_cls(5, 4, TailNeuralConfig(dropout=0.0)).eval()
    first = model(_sequence(), _context())[0]
    second = model(_mutate_after_step(_sequence(), 3), _context())[0]
    torch.testing.assert_close(first[:, :4], second[:, :4])


def _loss_problem() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(13)
    logits = torch.randn(2, 12, 3, generator=generator)
    timeout_prediction = torch.randn(2, 12, generator=generator)
    outcome = torch.tensor([[0, 1, 2] * 4, [2, 1, 0] * 4])
    timeout_target = torch.zeros(2, 12)
    uniqueness = torch.ones(2, 12)
    valid = torch.ones(2, 12, dtype=torch.bool)
    valid[:, 10:] = False
    return {
        "logits": logits,
        "timeout_prediction": timeout_prediction,
        "outcome": outcome,
        "timeout_target": timeout_target,
        "uniqueness": uniqueness,
        "valid": valid,
    }


def test_timeout_loss_ignores_tp_and_sl_rows():
    base = _loss_problem()
    first = distributional_tail_loss(**base)
    changed = {name: value.clone() for name, value in base.items()}
    changed["timeout_target"][changed["outcome"] != 2] = 999.0
    second = distributional_tail_loss(**changed)
    torch.testing.assert_close(first, second)


def test_invalid_future_steps_have_zero_loss_weight():
    base = _loss_problem()
    first = distributional_tail_loss(**base)
    changed = {name: value.clone() for name, value in base.items()}
    changed["logits"][~changed["valid"]] = 1e6
    changed["timeout_prediction"][~changed["valid"]] = 1e6
    torch.testing.assert_close(first, distributional_tail_loss(**changed))


def _tensors(*, score_fill: float | None = None) -> TailNeuralTensors:
    generator = np.random.default_rng(19)
    sequence = generator.normal(size=(8, 36, 5)).astype(np.float32)
    context = generator.normal(size=(8, 12, 4)).astype(np.float32)
    if score_fill is not None:
        sequence.fill(score_fill)
        context.fill(score_fill)
    outcome = np.tile(np.asarray([0, 1, 2] * 4, dtype=np.int64), (8, 1))
    timeout = generator.normal(scale=0.2, size=(8, 12)).astype(np.float32)
    return TailNeuralTensors(
        sequence=sequence,
        context=context,
        sequence_valid=np.ones((8, 36), dtype=bool),
        decision_valid=np.ones((8, 12), dtype=bool),
        outcome=outcome,
        timeout_target=timeout,
        uniqueness=np.ones((8, 12), dtype=np.float32),
    )


@pytest.mark.parametrize("model_name", ["tcn", "gru"])
def test_neural_scalers_fit_on_training_rows_only(model_name):
    config = TailNeuralConfig(epochs=1, patience=1, batch_size=4, dropout=0.0)
    fit = _tensors()
    first = fit_predict_neural_tail_model(
        model_name, fit, fit, _tensors(score_fill=0.0), config
    )
    second = fit_predict_neural_tail_model(
        model_name, fit, fit, _tensors(score_fill=1e6), config
    )
    np.testing.assert_allclose(first.scaler_median, second.scaler_median)


@pytest.mark.parametrize("model_name", ["tcn", "gru"])
def test_tcn_and_gru_repeat_with_seed_42(model_name):
    config = TailNeuralConfig(epochs=2, patience=1, batch_size=4, dropout=0.0)
    problem = _tensors()
    first = fit_predict_neural_tail_model(model_name, problem, problem, problem, config)
    second = fit_predict_neural_tail_model(model_name, problem, problem, problem, config)
    np.testing.assert_allclose(first.logits, second.logits, rtol=1e-6, atol=1e-6)
    assert first.audit["fit_early_live_label_overlap"] == 0


def test_unknown_neural_model_is_rejected():
    problem = _tensors()
    with pytest.raises(ValueError, match="model_name"):
        fit_predict_neural_tail_model(
            "lstm", problem, problem, problem, TailNeuralConfig(epochs=1)
        )
