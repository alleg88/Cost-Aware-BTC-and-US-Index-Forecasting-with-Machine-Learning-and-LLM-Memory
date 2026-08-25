"""Leakage and split-integrity sanity tests (cheap insurance the pipeline is honest).

Run:  cd code && ../code/.venv/Scripts/python -m pytest tests/ -v
(or just `pytest` from the code/ dir with the venv active).

The as-of news-join leakage test is stubbed until the later sentiment phase.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation.splits import BlockingTimeSeriesSplit
from features.build import build_dataset
from models.zoo import make_catboost, run_cv

CODE_ROOT = Path(__file__).resolve().parents[1]
WORKING_PARQUET = CODE_ROOT / "data" / "btcusdt_m15_2024_2025.parquet"


@pytest.fixture(scope="module")
def dataset():
    """A fast slice of the real M15 data for the leakage check."""
    if not WORKING_PARQUET.exists():
        pytest.skip("snapshot parquet missing; run data/load.py first")
    df = pd.read_parquet(WORKING_PARQUET).iloc[:25_000]
    X, y = build_dataset(df, threshold_bps=25.0, horizon=1)
    return X, y


def test_splits_dont_overlap_and_respect_embargo():
    """Every fold's train precedes its test with the embargo gap; folds never overlap."""
    X = np.zeros((10_000, 3))
    splitter = BlockingTimeSeriesSplit(n_splits=5, train_frac=0.8, embargo=4)
    seen_test = set()
    for train_idx, test_idx in splitter.split(X):
        assert train_idx.max() < test_idx.min(), "train must precede test"
        assert test_idx.min() - train_idx.max() > splitter.embargo, "embargo gap too small"
        assert not (seen_test & set(test_idx.tolist())), "test folds overlap"
        seen_test |= set(test_idx.tolist())


def test_shuffled_labels_collapse_to_chance(dataset):
    """With labels shuffled, a real model must score near 3-class chance (~0.33 macro-F1).

    If it stays high, features are leaking the target. A fast CatBoost is used here.
    """
    X, y = dataset
    rng = np.random.default_rng(0)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index, name="label")

    splitter = BlockingTimeSeriesSplit(n_splits=3, train_frac=0.8, embargo=4)
    fast = lambda params=None: make_catboost({"iterations": 60})
    result = run_cv(fast, X, y_shuffled, splitter)

    assert result["mean_macro_f1"] < 0.40, (
        f"shuffled-label macro-F1={result['mean_macro_f1']:.3f} is too high — "
        "the pipeline is leaking the label"
    )


def test_real_labels_beat_chance(dataset):
    """Sanity in the other direction: with real labels the model should clear chance."""
    X, y = dataset
    splitter = BlockingTimeSeriesSplit(n_splits=3, train_frac=0.8, embargo=4)
    fast = lambda params=None: make_catboost({"iterations": 120})
    result = run_cv(fast, X, y, splitter)
    assert result["mean_macro_f1"] > 0.34, "model fails to beat chance on real labels"
