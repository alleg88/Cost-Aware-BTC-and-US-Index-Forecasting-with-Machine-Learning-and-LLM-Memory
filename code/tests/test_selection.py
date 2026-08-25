"""Tests for the symmetric selectivity gates (agreement + conformal)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.selection import agreement_signal, conformal_qhat, conformal_signal


def test_agreement_is_symmetric_and_thresholds():
    idx = pd.RangeIndex(5)
    up = pd.Series([3, 1, 2, 0, 2], index=idx)
    down = pd.Series([1, 3, 2, 0, 0], index=idx)
    sig = agreement_signal(up, down, k=2)
    # row0: 3 up ->long; row1: 3 down ->short; row2: tie ->flat;
    # row3: none ->flat; row4: 2 up>0 down ->long
    assert sig.tolist() == [2, 0, 1, 1, 2]


def test_agreement_symmetry_under_direction_swap():
    idx = pd.RangeIndex(4)
    up = pd.Series([3, 2, 4, 1], index=idx)
    down = pd.Series([0, 1, 0, 2], index=idx)
    s1 = agreement_signal(up, down, k=3)
    s2 = agreement_signal(down, up, k=3)         # swap up<->down
    # swapping votes must swap long<->short exactly (2<->0), flats unchanged
    assert s2.replace({0: 2, 2: 0}).tolist() == s1.tolist()


def test_conformal_qhat_monotone_in_alpha():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0.2, 0.9, 500)
    # smaller alpha (higher coverage) -> larger qhat -> larger prediction sets
    assert conformal_qhat(p_true, 0.05) >= conformal_qhat(p_true, 0.30)


def test_conformal_singletons_trade_only_confident_bars():
    # three bars: confident-up, confident-down, ambiguous
    probs = pd.DataFrame({"p0": [0.05, 0.90, 0.34],
                          "p1": [0.05, 0.05, 0.33],
                          "p2": [0.90, 0.05, 0.33]})
    # qhat=0.2 -> admit classes with p >= 0.8: bar0={up}, bar1={down}, bar2={} -> argmax fallback, size3?
    sig = conformal_signal(probs, qhat=0.2)
    assert sig.tolist() == [2, 0, 1]   # ambiguous bar stays flat


def test_conformal_wide_set_is_flat():
    # low threshold admits every class -> set size 3 -> no trade
    probs = pd.DataFrame({"p0": [0.34], "p1": [0.33], "p2": [0.33]})
    sig = conformal_signal(probs, qhat=0.9)   # thresh=0.1, all admitted
    assert sig.tolist() == [1]
