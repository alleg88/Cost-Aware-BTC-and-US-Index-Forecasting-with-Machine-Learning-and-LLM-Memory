import pandas as pd
import pytest


def test_dz75_regime_run_contract_is_frozen():
    from experiments.run_walkforward_dz75_regime import (
        LOOKBACK_DAYS,
        MAX_HOLD,
        SL_BPS,
        TP_BPS,
        WIDTH,
        load_tuned_params,
        prediction_cache_path,
    )

    assert WIDTH == 75
    assert LOOKBACK_DAYS == 90
    assert (TP_BPS, SL_BPS, MAX_HOLD) == (150.0, 75.0, 1)
    assert prediction_cache_path().name == (
        "btc_bothofpos_cb-regime_dz75_lb90d_to2026.parquet"
    )
    assert load_tuned_params() == {
        "iterations": 700,
        "depth": 6,
        "learning_rate": 0.09340506545140258,
        "l2_leaf_reg": 32.50064360669027,
        "random_strength": 0.4518921136865075,
        "bagging_temperature": 0.8073593889083576,
        "rsm": 0.6908556046603513,
    }


def test_tau_selection_enforces_50_trade_floor():
    from experiments.run_walkforward_dz75_regime import select_tau

    grid = pd.DataFrame(
        {
            "tau": [0.40, 0.50, 0.60],
            "cal_sortino": [9.0, 1.0, 2.0],
            "cal_trades": [49, 60, 50],
        }
    )

    selected = select_tau(grid, floor=50)
    assert selected["tau"] == 0.60
    with pytest.raises(ValueError, match="50-trade floor"):
        select_tau(grid.iloc[:1], floor=50)
