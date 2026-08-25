"""Symmetric selectivity gates over a panel of calibrated models.

The economics notebooks establish that the only lever that monetises the weak M15
signal is *selectivity* — trading only the highest-conviction bars. The default
mechanism is a per-model confidence threshold (tau). This module provides two
alternative, still-**symmetric** (long and short treated identically) gates that
decide *when to trade* from a panel of models rather than from one probability:

  * **K-of-N agreement** — trade a direction only when at least K of the N models
    vote for it (and it is the plurality). Discrete, hard to overfit, and needs no
    probability calibration to be meaningful.

  * **Conformal singleton** — using calibrated class probabilities, build a
    split-conformal (LAC) prediction set per bar and trade only when the set is the
    singleton {up} or {down}. Distribution-free finite-sample coverage; the set
    collapses to one class exactly when the model is confident enough that the
    calibration data licenses excluding the other two.

Both keep short trades fully in play (fair in a trending year, unlike a long-only
model), and both plug into evaluation.economics via the 0/1/2 signal they return.
Class encoding: down=0, flat=1, up=2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def agreement_signal(votes_up: pd.Series, votes_down: pd.Series, k: int) -> pd.Series:
    """K-of-N directional agreement -> 0/1/2 signal.

    up (2)   if at least k models vote up AND up strictly outvotes down,
    down (0) if at least k models vote down AND down strictly outvotes up,
    flat (1) otherwise (too few agree, or the panel is split).
    """
    up = votes_up.astype(int)
    down = votes_down.astype(int)
    sig = pd.Series(1, index=up.index, dtype="int64")
    sig[(up >= k) & (up > down)] = 2
    sig[(down >= k) & (down > up)] = 0
    return sig


def conformal_qhat(p_true: np.ndarray, alpha: float) -> float:
    """Split-conformal (LAC) threshold from calibration nonconformity scores.

    Score s_i = 1 - p_model(true class_i). The threshold is the
    ceil((n+1)(1-alpha))/n empirical quantile of the calibration scores; a test
    class y is admitted to the prediction set iff p(y) >= 1 - qhat.
    """
    s = 1.0 - np.asarray(p_true, dtype=float)
    n = len(s)
    if n == 0:
        return 1.0
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    level = min(level, 1.0)
    return float(np.quantile(s, level, method="higher"))


def conformal_signal(probs: pd.DataFrame, qhat: float,
                     cols=("p0", "p1", "p2")) -> pd.Series:
    """Trade only on singleton conformal sets: {up}->2, {down}->0, else flat.

    probs: per-bar calibrated class probabilities (columns cols = down/flat/up).
    A class y is in the set iff probs[y] >= 1 - qhat. We trade only when the set
    is exactly one directional class (a {flat} singleton or any larger set = out).
    """
    thresh = 1.0 - qhat
    in_set = probs[list(cols)].to_numpy(dtype=float) >= thresh
    # An EMPTY set means every class was too improbable to license — i.e. the bar
    # is atypical / low-confidence, so we stay flat (never force the argmax; that
    # would trade exactly the most uncertain bars). Only a directional SINGLETON
    # {down} or {up} opens a position.
    size = in_set.sum(axis=1)
    down_only = in_set[:, 0] & (size == 1)
    up_only = in_set[:, 2] & (size == 1)
    sig = np.ones(len(probs), dtype=np.int64)   # flat
    sig[down_only] = 0
    sig[up_only] = 2
    return pd.Series(sig, index=probs.index, dtype="int64")
