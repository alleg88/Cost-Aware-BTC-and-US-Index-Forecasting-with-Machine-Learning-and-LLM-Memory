from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from experiments.walkforward import (
    make_report_fn,
    make_score_fn,
    run_walkforward_predictions,
    weekly_walkforward_windows,
)
from memory.loop import ReflectionWindow
from memory.schema import Lesson


class ToyClassifier:
    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict_proba(self, X):
        probs = np.full((len(X), 3), 0.1)
        signal = X["signal"].to_numpy()
        probs[signal < 0] = [0.8, 0.1, 0.1]
        probs[signal == 0] = [0.1, 0.8, 0.1]
        probs[signal > 0] = [0.1, 0.1, 0.8]
        return probs


class WeightedToyClassifier(ToyClassifier):
    def __init__(self, fitted_weights):
        self.fitted_weights = fitted_weights

    def fit(self, X, y, sample_weight=None):
        self.fitted_weights.append(pd.Series(sample_weight, index=X.index))
        return super().fit(X, y)


def test_weekly_walkforward_windows_keep_2026_lockbox_sealed():
    idx = pd.date_range("2024-07-01", "2026-06-30", freq="15min", tz="UTC")

    windows = weekly_walkforward_windows(
        idx,
        walk_start="2025-01-01",
        walk_end="2026-03-31",
        train_lookback="180D",
    )

    assert windows[0].name == "wf_000_2025-01-01"
    assert windows[0].validation_start == pd.Timestamp("2025-01-01", tz="UTC")
    assert windows[0].train_end == windows[0].validation_start
    assert windows[0].train_start == pd.Timestamp("2024-07-05", tz="UTC")
    assert any(w.validation_start.year == 2026 for w in windows)
    assert all(w.validation_start < pd.Timestamp("2026-04-01", tz="UTC") for w in windows)
    assert max(w.validation_end for w in windows) < pd.Timestamp("2026-04-01", tz="UTC")


def test_run_walkforward_predictions_writes_validation_cache(tmp_path):
    idx = pd.date_range("2024-07-01", "2025-01-15 23:45", freq="15min", tz="UTC")
    X = pd.DataFrame({"signal": np.resize([-1.0, 0.0, 1.0], len(idx))}, index=idx)
    y = pd.Series(np.resize([0, 1, 2], len(idx)), index=idx, name="label")
    windows = weekly_walkforward_windows(idx, "2025-01-01", "2025-01-15", train_lookback="30D")
    cache_path = tmp_path / "wf.parquet"

    preds = run_walkforward_predictions(
        X,
        y,
        windows=windows,
        model_factory=lambda: ToyClassifier(),
        model_name="catboost",
        cache_path=cache_path,
    )

    assert cache_path.exists()
    assert preds.index.min() >= pd.Timestamp("2025-01-01", tz="UTC")
    assert preds.index.max() < pd.Timestamp("2025-01-16", tz="UTC")
    assert {"window", "y_true", "catboost_pred", "catboost_conf"} <= set(preds.columns)
    assert {"catboost_p0", "catboost_p1", "catboost_p2"} <= set(preds.columns)
    assert np.allclose(preds[["catboost_p0", "catboost_p1", "catboost_p2"]].sum(axis=1), 1.0)
    assert sorted(preds["window"].unique()) == ["wf_000_2025-01-01", "wf_001_2025-01-08"]


def test_run_walkforward_predictions_can_show_progress(capsys):
    idx = pd.date_range("2024-12-01", "2025-01-07 23:45", freq="15min", tz="UTC")
    X = pd.DataFrame({"signal": np.resize([-1.0, 0.0, 1.0], len(idx))}, index=idx)
    y = pd.Series(np.resize([0, 1, 2], len(idx)), index=idx, name="label")
    windows = weekly_walkforward_windows(idx, "2025-01-01", "2025-01-07", train_lookback="7D")

    run_walkforward_predictions(
        X,
        y,
        windows=windows,
        model_factory=lambda: ToyClassifier(),
        progress_label="wf-test",
    )

    captured = capsys.readouterr()
    assert "wf-test" in captured.err


def test_run_walkforward_predictions_passes_window_specific_sample_weights():
    idx = pd.date_range("2024-12-01", "2025-01-07 23:45", freq="15min", tz="UTC")
    X = pd.DataFrame({"signal": np.resize([-1.0, 0.0, 1.0], len(idx))}, index=idx)
    y = pd.Series(np.resize([0, 1, 2], len(idx)), index=idx, name="label")
    regimes = pd.Series(
        np.where(np.arange(len(idx)) % 3 == 0, "bull", "bear"), index=idx
    )
    fitted_weights = []
    windows = weekly_walkforward_windows(
        idx, "2025-01-01", "2025-01-07", train_lookback="7D"
    )

    run_walkforward_predictions(
        X,
        y,
        windows=windows,
        model_factory=lambda: WeightedToyClassifier(fitted_weights),
        sample_weight_fn=lambda X_train, _y_train: regimes.loc[X_train.index].map(
            lambda regime: 1.0 if regime == "bull" else 0.5
        ),
    )

    assert len(fitted_weights) == 1
    assert fitted_weights[0].index.equals(
        idx[(idx >= windows[0].train_start) & (idx < windows[0].train_end)]
    )
    assert set(fitted_weights[0].unique()) == {0.5, 1.0}


def test_cli_walkforward_trims_the_label_horizon(monkeypatch, tmp_path):
    """The generic CLI must trim each training tail by its label horizon."""
    from experiments import run_walkforward

    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    X = pd.DataFrame({"signal": np.arange(len(idx))}, index=idx)
    y = pd.Series(np.resize([0, 1, 2], len(idx)), index=idx, name="label")
    aux = pd.DataFrame(
        {"forward_return": 0.0, "vol_regime": "low"}, index=idx
    )
    captured: dict[str, int | None] = {}

    monkeypatch.setattr(
        run_walkforward.yaml,
        "safe_load",
        lambda _text: {
            "dates": {
                "train": ["2024-01-01", "2024-12-31"],
                "walkforward": ["2025-01-01", "2025-01-31"],
            }
        },
    )
    monkeypatch.setattr(
        run_walkforward,
        "build_walkforward_xy",
        lambda *args, **kwargs: (X, y, aux),
    )
    monkeypatch.setattr(
        run_walkforward,
        "weekly_walkforward_windows",
        lambda *args, **kwargs: [],
    )
    monkeypatch.setattr(run_walkforward, "resolve_params", lambda *args, **kwargs: None)

    def fake_predictions(*args, **kwargs):
        captured["train_tail_trim"] = kwargs.get("train_tail_trim")
        return pd.DataFrame({"window": ["wf_000"]}, index=idx[:1])

    monkeypatch.setattr(run_walkforward, "run_walkforward_predictions", fake_predictions)
    out = tmp_path / "walkforward.parquet"
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_walkforward.py", "--horizon", "4", "--out", str(out)],
    )

    assert run_walkforward.main() == 0
    assert captured["train_tail_trim"] == 4

def test_report_and_score_functions_read_cached_predictions():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    preds = pd.DataFrame(
        {
            "y_true": [1, 1, 2, 0],
            "catboost_pred": [2, 1, 2, 0],
            "catboost_conf": [0.95, 0.80, 0.90, 0.90],
            "vol_regime": ["high", "low", "low", "low"],
            "forward_return": [0.0, 0.0, 0.01, -0.01],
        },
        index=idx,
    )
    window = ReflectionWindow(
        name="wf_000_2025-01-01",
        train_start=pd.Timestamp("2024-12-31", tz="UTC"),
        train_end=pd.Timestamp("2025-01-01", tz="UTC"),
        validation_start=idx.min(),
        validation_end=idx.max(),
    )

    report = make_report_fn(preds, condition_cols=("vol_regime",))(window)
    baseline = make_score_fn(preds)([], window)
    lesson = Lesson(
        condition="vol_regime == 'high'",
        action="force_flat",
        target="ensemble",
        factor=1.0,
        evidence="high-vol directional calls were wrong",
        confidence=0.8,
    )
    gated = make_score_fn(preds, feature_columns=("vol_regime",))([lesson], window)

    assert report["window"]["name"] == "wf_000_2025-01-01"
    assert report["overall"]["catboost"]["n"] == 4
    assert gated > baseline
