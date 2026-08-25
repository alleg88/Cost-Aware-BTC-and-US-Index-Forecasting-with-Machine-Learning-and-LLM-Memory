import pandas as pd

from experiments.run_intrabar_holding import (
    HOLD_GRID,
    SL_GRID,
    TP_GRID,
    path_safe_mask,
    select_best_geometry,
)


def test_holding_experiment_scope_is_one_two_and_four_hours():
    assert HOLD_GRID == (4, 8, 16)
    assert TP_GRID == (100.0, 150.0, 200.0, 300.0)
    assert SL_GRID == (50.0, 75.0, 100.0)


def test_path_safe_mask_embargoes_full_next_open_holding_path():
    index = pd.DatetimeIndex(
        ["2026-03-31 22:45", "2026-03-31 23:00"], tz="UTC"
    )
    boundary = pd.Timestamp("2026-04-01", tz="UTC")

    safe = path_safe_mask(index, hold=4, boundary=boundary)

    assert safe.tolist() == [True, False]


def test_geometry_selection_enforces_fifty_trade_floor():
    grid = pd.DataFrame(
        [
            {"tp_bps": 300.0, "sl_bps": 50.0, "max_hold": 16,
             "cal_sortino": 9.0, "cal_trades": 49},
            {"tp_bps": 150.0, "sl_bps": 75.0, "max_hold": 8,
             "cal_sortino": 1.5, "cal_trades": 50},
        ]
    )

    selected = select_best_geometry(grid, floor=50)

    assert selected["tp_bps"] == 150.0
    assert selected["cal_trades"] == 50
