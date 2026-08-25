"""Time-series cross-validation splitters.

BlockingTimeSeriesSplit gives leakage-resistant cross-validation for time series.
Unlike sklearn's TimeSeriesSplit — whose expanding folds overlap and reuse history —
this divides the series into contiguous, NON-overlapping blocks, each split internally
into train -> (embargo gap) -> test. Folds share no data, so fold scores are
independent and leakage-resistant.

    +-----------------------------------------------------------+  full series
    [ train .... | emb | test ]                                   fold 0
                  [ train .... | emb | test ]                     fold 1
                                [ train .... | emb | test ]       fold 2  ...

Compatible with sklearn (cross_val_score(cv=...)) and Optuna.
"""
from __future__ import annotations

import numpy as np


class BlockingTimeSeriesSplit:
    """Non-overlapping blocked time-series CV.

    Parameters
    ----------
    n_splits : int
        Number of contiguous blocks (folds).
    train_frac : float
        Fraction of each block used for training; the rest (minus the embargo) is test.
    embargo : int
        Number of bars dropped between a block's train and test segments. Must be
        >= the label horizon to stop next-bar labels leaking across the boundary.
    """

    def __init__(self, n_splits: int = 5, train_frac: float = 0.8, embargo: int = 4):
        if not 0.0 < train_frac < 1.0:
            raise ValueError("train_frac must be in (0, 1)")
        self.n_splits = n_splits
        self.train_frac = train_frac
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        block = n // self.n_splits
        if block == 0:
            raise ValueError("Not enough samples for the requested n_splits")
        for i in range(self.n_splits):
            start = i * block
            stop = start + block
            cut = start + int(block * self.train_frac)
            train_idx = np.arange(start, cut)
            test_idx = np.arange(cut + self.embargo, stop)
            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx
