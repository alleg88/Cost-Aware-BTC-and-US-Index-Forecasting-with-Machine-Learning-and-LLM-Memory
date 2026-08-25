from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.utils.class_weight import compute_sample_weight

from experiments.model_zoo_protocol import BASE_MODELS
from models.deep import _weighted_cross_entropy, make_sequences
from models.zoo import MODELS, XGBBalanced, make_catboost

# Small overrides so every model trains in seconds on the synthetic set.
FAST_PARAMS = {
    "random_forest": {"n_estimators": 10},
    "xgboost_balanced": {"n_estimators": 10},
    "catboost_balanced": {"iterations": 10},
    "decision_tree": {"min_samples_leaf": 5},
    "mlp": {"epochs": 2, "hidden": (16,), "batch_size": 128},
    "lstm": {"epochs": 2, "seq_len": 4, "hidden_size": 8, "batch_size": 128},
    "gru": {"epochs": 2, "seq_len": 4, "hidden_size": 8, "batch_size": 128},
    "stack_lstm_catboost": {
        "catboost_balanced": {"iterations": 10},
        "lstm": {"epochs": 2, "seq_len": 4, "hidden_size": 8, "batch_size": 128},
    },
    "stack_all": {
        "random_forest": {"n_estimators": 10},
        "xgboost_balanced": {"n_estimators": 10},
        "catboost_balanced": {"iterations": 10},
        "lstm": {"epochs": 2, "seq_len": 4, "hidden_size": 8, "batch_size": 128},
    },
}


def _toy_data(n: int = 400, k: int = 5):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(n, k)), columns=[f"f{i}" for i in range(k)])
    signal = X["f0"] + 0.5 * X["f1"]
    y = pd.Series(np.where(signal > 0.8, 2, np.where(signal < -0.8, 0, 1)))
    return X, y



def test_catboost_regressor_predicts_continuous_values():
    from models.zoo import make_catboost_regressor

    X, _ = _toy_data(n=80)
    target = X["f0"] * 0.01 - X["f1"] * 0.005
    model = make_catboost_regressor({"iterations": 10})

    model.fit(X.iloc[:60], target.iloc[:60])
    prediction = np.asarray(model.predict(X.iloc[60:])).reshape(-1)

    assert model.get_params()["loss_function"] == "RMSE"
    assert prediction.shape == (20,)
    assert np.isfinite(prediction).all()

def test_catboost_accepts_sqrt_balanced_weight_override():
    """The controlled arm must alter only CatBoost's automatic class weighting."""
    model = make_catboost({"iterations": 10, "auto_class_weights": "SqrtBalanced"})
    assert model.get_params()["auto_class_weights"] == "SqrtBalanced"


@pytest.mark.parametrize("name", BASE_MODELS)
def test_every_base_model_accepts_regime_sample_weight(name):
    X, y = _toy_data(n=180)
    weight = np.linspace(0.5, 1.5, 140)
    model = MODELS[name](FAST_PARAMS.get(name))
    model.fit(X.iloc[:140], y.iloc[:140], sample_weight=weight)

    proba = np.asarray(model.predict_proba(X.iloc[140:148]))
    assert proba.shape == (8, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_xgboost_combines_class_and_regime_weights():
    X, y = _toy_data(n=30)
    regime_weight = np.linspace(0.5, 1.5, len(y))
    captured = {}

    class CaptureModel:
        classes_ = np.array([0, 1, 2])

        def fit(self, X_fit, y_fit, sample_weight=None):
            captured["weight"] = np.asarray(sample_weight)

    model = XGBBalanced({"n_estimators": 1})
    model.model = CaptureModel()
    model.fit(X, y, sample_weight=regime_weight)

    expected = compute_sample_weight("balanced", y) * regime_weight
    np.testing.assert_allclose(captured["weight"], expected)


def test_weighted_cross_entropy_ignores_zero_weight_rows():
    logits = torch.tensor([[2.0, 0.0, -1.0], [-9.0, 9.0, 0.0]])
    targets = torch.tensor([0, 0])
    class_weight = torch.tensor([1.0, 2.0, 3.0])
    sample_weight = torch.tensor([1.0, 0.0])

    actual = _weighted_cross_entropy(
        logits, targets, class_weight, sample_weight
    )
    expected = torch.nn.functional.cross_entropy(logits[:1], targets[:1])
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("name", list(MODELS))
def test_every_registry_model_fits_and_predicts_three_classes(name):
    X, y = _toy_data()
    model = MODELS[name](FAST_PARAMS.get(name))
    model.fit(X.iloc[:300], y.iloc[:300])

    pred = np.asarray(model.predict(X.iloc[300:])).ravel().astype(int)
    assert pred.shape == (100,)
    assert set(pred) <= {0, 1, 2}

    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X.iloc[300:]))
        assert proba.shape == (100, 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_make_sequences_windows_never_use_future_rows():
    X = np.arange(20, dtype=np.float32).reshape(10, 2)
    seqs = make_sequences(X, seq_len=4)

    assert seqs.shape == (10, 4, 2)
    for t in range(10):
        np.testing.assert_array_equal(seqs[t, -1], X[t])          # window ends at row t
        first = max(0, t - 3)
        np.testing.assert_array_equal(seqs[t, -1 - (t - first):], X[first:t + 1])
    np.testing.assert_array_equal(seqs[0], np.repeat(X[:1], 4, axis=0))  # head padding


def test_deep_models_are_deterministic_across_fits():
    from models.deep import TorchMLPClassifier

    X, y = _toy_data()
    proba = []
    for _ in range(2):
        model = TorchMLPClassifier(epochs=2, hidden=(16,), seed=7)
        model.fit(X.iloc[:300], y.iloc[:300])
        proba.append(model.predict_proba(X.iloc[300:]))
    np.testing.assert_allclose(proba[0], proba[1], atol=1e-6)
