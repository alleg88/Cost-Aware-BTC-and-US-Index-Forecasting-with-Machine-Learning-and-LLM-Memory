import pandas as pd


def _row(**overrides):
    row = {
        "tp_bps": 100,
        "sl_bps": 75,
        "max_hold": 1,
        "trades": 80,
        "pooled_net": 0.04,
        "pooled_sortino": 1.2,
        "pooled_sharpe": 0.9,
        "positive_folds": 10,
        "n_long": 50,
        "n_short": 30,
        "long_net": 0.02,
        "short_net": 0.02,
        "bull_sortino": 0.8,
        "sideways_sortino": 0.7,
        "bear_sortino": 0.6,
        "robust_score": 0.6,
    }
    row.update(overrides)
    return row


def test_geometry_grid_is_small_and_predeclared():
    from experiments.run_geometry_selection import GEOMETRIES

    assert GEOMETRIES == tuple(
        (tp, sl, hold)
        for tp, sl in ((100, 75), (150, 75), (150, 100), (200, 100))
        for hold in (1, 4, 8)
    )


def test_geometry_selection_uses_existing_economic_guards():
    from experiments.run_geometry_selection import select_geometry

    eligible = _row(tp_bps=150, sl_bps=100, max_hold=4, robust_score=0.8)
    thin = _row(tp_bps=200, sl_bps=100, max_hold=8, trades=49, robust_score=9)

    selected = select_geometry(pd.DataFrame([eligible, thin]), n_folds=15)

    assert selected is not None
    assert tuple(selected[["tp_bps", "sl_bps", "max_hold"]]) == (150, 100, 4)
    assert select_geometry(pd.DataFrame([thin]), n_folds=15) is None


def test_geometry_outer_audits_use_only_earlier_folds():
    from experiments.run_geometry_selection import outer_audit_schedule

    assert outer_audit_schedule() == [
        (tuple(range(12)), 12),
        (tuple(range(13)), 13),
        (tuple(range(14)), 14),
    ]
