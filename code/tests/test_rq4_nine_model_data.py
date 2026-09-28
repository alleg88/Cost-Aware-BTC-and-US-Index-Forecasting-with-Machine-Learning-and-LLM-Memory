"""Temporal and source-contract checks for the new RQ4 forecast extension."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from experiments import rq4_nine_model_data as data


def test_development_preprocessing_cannot_learn_from_2024(monkeypatch):
    index = pd.date_range(data.FIT_START, data.DEVELOPMENT_START + pd.Timedelta(days=2),
                          freq="15min", inclusive="left")
    X = pd.DataFrame(1.0, index=index, columns=data.FEATURES)
    X.loc[index >= data.DEVELOPMENT_START, "r1"] = 999.0
    X.loc[data.DEVELOPMENT_START, "r1"] = np.nan
    X.loc[index < data.DEVELOPMENT_START, "funding_rate"] = np.nan
    y = pd.Series(np.arange(len(index)) % 3, index=index)
    regimes = pd.Series("sideways", index=index)
    observed = {}

    class RecordingModel:
        classes_ = np.array([0, 1, 2])

        def fit(self, features, target, sample_weight=None):
            observed["train"] = features.copy()
            return self

        def predict_proba(self, features):
            observed["test"] = features.copy()
            return np.tile([0.2, 0.6, 0.2], (len(features), 1))

    monkeypatch.setitem(data.MODELS, "logreg", lambda _: RecordingModel())
    prediction, audit = data.fit_development_model(X, y, regimes, "logreg", 65)
    assert observed["train"].index.max() < data.DEVELOPMENT_START
    assert observed["train"]["r1"].eq(1.0).all()
    assert observed["train"]["funding_rate"].eq(0.0).all()
    assert observed["test"].loc[data.DEVELOPMENT_START, "r1"] == 1.0
    assert audit["latest_training_label_available"] < audit["first_test_decision"]
    assert prediction["timestamp"].min() == data.DEVELOPMENT_START


def test_prediction_validation_rejects_future_fit_and_wrong_label_width():
    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    frame = pd.DataFrame({
        "timestamp": [timestamp], "y_true": [1], "pred": [1], "confidence": [0.8],
        "width_bps": [65], "train_start": [timestamp - pd.Timedelta(days=180)],
        "train_end": [timestamp], "p_short": [0.1], "p_flat": [0.8], "p_long": [0.1],
    })
    with pytest.raises(ValueError, match="noncausal"):
        data.validate_prediction(frame, 65)
    frame["train_end"] = timestamp - pd.Timedelta(minutes=30)
    with pytest.raises(ValueError, match="width"):
        data.validate_prediction(frame, 55)
    data.validate_prediction(frame, 65)
