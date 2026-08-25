from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.economics import (
    confidence_weights, strategy_returns, strategy_returns_sized,
)


def _series(vals):
    idx = pd.date_range("2025-01-01", periods=len(vals), freq="15min", tz="UTC")
    return pd.Series(vals, index=idx)


def test_confidence_weights_ramp_from_tau_to_one():
    conf = _series([0.5, 0.6, 0.8, 1.0])
    w = confidence_weights(conf, tau=0.6, cap=1.0, floor=0.0)
    assert w.iloc[0] == 0.0                       # below gate -> 0
    assert w.iloc[1] == 0.0                       # exactly at gate -> floor
    assert abs(w.iloc[2] - 0.5) < 1e-9            # halfway from 0.6 to 1.0
    assert abs(w.iloc[3] - 1.0) < 1e-9            # full confidence -> cap
    assert (w <= 1.0).all() and (w >= 0.0).all()


def test_floor_gives_minimum_size_above_gate():
    conf = _series([0.7, 0.85, 1.0])
    w = confidence_weights(conf, tau=0.7, cap=1.0, floor=0.25)
    assert abs(w.iloc[0] - 0.25) < 1e-9          # at gate -> floor, not zero
    assert w.iloc[1] > 0.25 and w.iloc[2] == 1.0


def test_sized_equals_flat_when_cap_equals_floor_equals_one():
    # cap=floor=1 -> every taken bar sized 1.0 -> identical to the unit-size rule
    pred = _series([2, 0, 2, 1, 2])
    conf = _series([0.9, 0.8, 0.95, 0.3, 0.85])
    fwd = _series([0.01, -0.02, 0.015, 0.0, -0.01])
    flat = strategy_returns(pred, fwd, fee_bps=5.0, conf=conf, tau=0.5)
    sized = strategy_returns_sized(pred, fwd, fee_bps=5.0, conf=conf, tau=0.5,
                                   cap=1.0, floor=1.0)
    pd.testing.assert_series_equal(flat, sized, check_names=False)


def test_sizing_scales_gross_pnl_by_weight():
    # one long bar, confidence exactly halfway -> weight 0.5 -> half the gross,
    # and half the turnover cost of the unit-size trade.
    pred = _series([2])
    conf = _series([0.75])                        # halfway between tau=0.5 and 1.0
    fwd = _series([0.02])
    sized = strategy_returns_sized(pred, fwd, fee_bps=10.0, conf=conf, tau=0.5)
    expected = 0.5 * 0.02 - 0.5 * (10.0 / 1e4)    # sized gross - sized entry cost
    assert abs(sized.iloc[0] - expected) < 1e-12
