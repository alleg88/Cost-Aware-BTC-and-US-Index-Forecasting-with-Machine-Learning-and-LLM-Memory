from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.causal_sharpe_ensemble import (
    apply_regime_gate,
    combine_probabilities,
    dynamic_thresholds,
    positive_sharpe_weights,
)
from experiments.raw_hold_control import MODEL_NAMES


def test_positive_sharpe_weights_disable_nonpositive_models(monkeypatch):
    scores = iter([2.0, -1.0, 1.0, 0.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    monkeypatch.setattr(
        "experiments.causal_sharpe_ensemble.economics_summary",
        lambda returns: {"sharpe": next(scores)},
    )
    history = {model: pd.Series([0.0]) for model in MODEL_NAMES}

    weights, sharpes = positive_sharpe_weights(history)

    assert np.isclose(sum(weights.values()), 1.0)
    assert np.isclose(weights[MODEL_NAMES[0]], 2.0 / 3.0)
    assert weights[MODEL_NAMES[1]] == 0.0
    assert np.isclose(weights[MODEL_NAMES[2]], 1.0 / 3.0)
    assert sharpes[MODEL_NAMES[1]] == -1.0


def test_all_nonpositive_sharpes_produce_explicit_flat_probabilities(monkeypatch):
    monkeypatch.setattr(
        "experiments.causal_sharpe_ensemble.economics_summary",
        lambda returns: {"sharpe": -1.0},
    )
    history = {model: pd.Series([0.0]) for model in MODEL_NAMES}
    weights, _ = positive_sharpe_weights(history)
    probabilities = {
        model: np.tile(np.array([[0.1, 0.2, 0.7]]), (3, 1))
        for model in MODEL_NAMES
    }

    combined = combine_probabilities(probabilities, weights)

    assert np.allclose(combined, np.tile(np.array([[0.0, 1.0, 0.0]]), (3, 1)))


def test_dynamic_gate_raises_tau_and_blocks_high_funding_long():
    index = pd.date_range("2025-01-01", periods=120, freq="15min", tz="UTC")
    context = pd.DataFrame(
        {
            "vol_20": np.r_[np.ones(119), 10.0],
            "funding_z": np.r_[np.zeros(119), 2.5],
        },
        index=index,
    )
    prediction = pd.DataFrame(
        {
            "timestamp": index,
            "p_short": 0.1,
            "p_flat": 0.2,
            "p_long": 0.7,
        }
    )

    threshold = dynamic_thresholds(context, tau_base=0.60, alpha=0.05)
    gated, diagnostics = apply_regime_gate(
        prediction, context, tau_base=0.60, alpha=0.05, funding_filter=True
    )

    assert threshold.iloc[-1] > threshold.iloc[0]
    assert gated.iloc[-1]["pred"] == 1
    assert bool(diagnostics.iloc[-1]["blocked_long_by_funding"])

