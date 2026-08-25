from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.economics import (
    diebold_mariano,
    economics_summary,
    max_drawdown,
    positions_from_predictions,
    strategy_returns,
    sweep_tau,
)


def test_positions_map_and_confidence_gate():
    pred = pd.Series([0, 1, 2, 2, 0])
    conf = pd.Series([0.9, 0.9, 0.4, 0.8, 0.3])

    ungated = positions_from_predictions(pred)
    assert ungated.tolist() == [-1.0, 0.0, 1.0, 1.0, -1.0]

    gated = positions_from_predictions(pred, conf, tau=0.5)
    assert gated.tolist() == [-1.0, 0.0, 0.0, 1.0, 0.0]   # low-confidence bars stay out


def test_strategy_returns_charges_fee_per_side():
    pred = pd.Series([2, 2, 0, 1])                        # long, hold, flip short, exit
    fwd = pd.Series([0.01, 0.01, 0.01, 0.01])
    r = strategy_returns(pred, fwd, fee_bps=10.0)

    assert r.iloc[0] == pytest.approx(0.01 - 0.001)       # enter long: 1 side
    assert r.iloc[1] == pytest.approx(0.01)               # hold: no fee
    assert r.iloc[2] == pytest.approx(-0.01 - 0.002)      # flip long->short: 2 sides
    assert r.iloc[3] == pytest.approx(0.0 - 0.001)        # exit: 1 side


def test_max_drawdown_on_known_curve():
    # cumulative: 1, 3, 2, 0, 1  -> peak 3, trough 0 -> drawdown 3
    returns = pd.Series([1.0, 2.0, -1.0, -2.0, 1.0])
    assert max_drawdown(returns) == pytest.approx(3.0)


def test_sweep_tau_monotone_exposure():
    rng = np.random.default_rng(0)
    pred = pd.Series(rng.integers(0, 3, size=500))
    conf = pd.Series(rng.uniform(0.34, 1.0, size=500))
    fwd = pd.Series(rng.normal(0, 0.001, size=500))

    sweep = sweep_tau(pred, conf, fwd, fee_bps=5.0, taus=(0.0, 0.5, 0.9))
    exposures = sweep["exposure"].tolist()
    assert exposures[0] >= exposures[1] >= exposures[2]   # higher tau -> less exposure
    assert set(sweep["tau"]) == {0.0, 0.5, 0.9}



def test_global_tau_requires_fifty_turnover_events():
    """Calibration cannot choose a threshold below the strengthened floor."""
    from experiments.run_arms_economics import calibrate_global

    def candidate(n: int):
        pred = pd.Series(np.resize([2, 1], n))
        conf = pd.Series(0.9, index=pred.index)
        fwd = pd.Series(0.01, index=pred.index)
        return calibrate_global(pred, conf, fwd, fee=5.0)

    assert candidate(49) is None
    assert candidate(50) is not None


def test_bracket_dm_uses_the_maximum_holding_lag():
    """Bracket P&L requires a Newey-West lag covering the maximum hold."""
    from experiments.run_arms_economics import MAX_HOLD, dm_lag_for_engine

    assert dm_lag_for_engine("per-bar") == 1
    assert dm_lag_for_engine("m15 brackets") == MAX_HOLD
    assert dm_lag_for_engine("1m brackets+trail") == MAX_HOLD
    with pytest.raises(ValueError, match="unknown engine"):
        dm_lag_for_engine("unknown")

def test_diebold_mariano_detects_direction_and_null():
    rng = np.random.default_rng(1)
    base = pd.Series(rng.normal(0, 0.001, size=2000))
    better = base + 0.0005

    dm = diebold_mariano(base, better)
    assert dm["dm_stat"] > 2.0 and dm["p_value"] < 0.05   # b clearly outperforms

    null = diebold_mariano(base, base.copy())
    assert null["p_value"] == pytest.approx(1.0)          # identical series: no difference


def test_economics_summary_reports_trades_and_exposure():
    pred = pd.Series([1, 2, 2, 1, 0, 1])
    fwd = pd.Series([0.0, 0.001, -0.001, 0.0, 0.002, 0.0])
    r = strategy_returns(pred, fwd, fee_bps=0.0)
    s = economics_summary(r, pred)

    assert s["trade_count"] == 4                          # enter, exit, enter, exit
    assert s["exposure"] == pytest.approx(3 / 6)
    assert "sortino" in s and "max_drawdown" in s
