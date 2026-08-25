import numpy as np
import pandas as pd

from evaluation.trades_intrabar import simulate_bracket_trades_intrabar


def _m15(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")


def _seconds(start: pd.Timestamp, n_m15_bars: int) -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_m15_bars * 15 * 60, freq="1s", tz="UTC")


def _bars(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": 100.0, "high": 103.0, "low": 97.0, "close": 100.0},
        index=index,
    )


def _one_second(index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=index,
    )


def test_one_second_path_uses_first_touch_and_records_audit_fields():
    index = _m15(4)
    one_second = _one_second(_seconds(index[0], 4))
    one_second.loc[index[1] + pd.Timedelta(seconds=1), "high"] = 102.5
    one_second.loc[index[1] + pd.Timedelta(seconds=2), "low"] = 97.5

    ledger, _ = simulate_bracket_trades_intrabar(
        _bars(index),
        one_second,
        pd.Series({index[0]: 2}),
        tp_bps=200,
        sl_bps=200,
        max_hold=1,
        fee_bps=0,
        expected_interval=pd.Timedelta(seconds=1),
        include_audit=True,
    )

    assert ledger.iloc[0]["exit_reason"] == "take_profit"
    assert not bool(ledger.iloc[0]["ambiguous_touch"])
    assert ledger.iloc[0]["execution_interval_seconds"] == 1.0
    assert ledger.iloc[0]["intrabar_exit_time"] == index[1] + pd.Timedelta(seconds=1)


def test_same_second_touch_is_stop_first_and_ambiguous_for_long_and_short():
    index = _m15(3)
    one_second = _one_second(_seconds(index[0], 3))
    touch = index[1] + pd.Timedelta(seconds=3)
    one_second.loc[touch, ["high", "low"]] = [102.5, 97.5]

    for prediction in (0, 2):
        ledger, _ = simulate_bracket_trades_intrabar(
            _bars(index),
            one_second,
            pd.Series({index[0]: prediction}),
            tp_bps=200,
            sl_bps=200,
            max_hold=1,
            fee_bps=0,
            expected_interval=pd.Timedelta(seconds=1),
            include_audit=True,
        )
        assert ledger.iloc[0]["exit_reason"] == "stop_loss"
        assert bool(ledger.iloc[0]["ambiguous_touch"])
        assert ledger.iloc[0]["gross_return"] < 0


def test_two_bar_hold_excludes_touch_at_the_30_minute_boundary():
    index = _m15(5)
    one_second = _one_second(_seconds(index[0], 5))
    one_second.loc[index[3], "high"] = 102.0

    ledger, _ = simulate_bracket_trades_intrabar(
        _bars(index),
        one_second,
        pd.Series({index[0]: 2}),
        tp_bps=100,
        sl_bps=100,
        max_hold=2,
        fee_bps=0,
        expected_interval=pd.Timedelta(seconds=1),
    )

    assert ledger.iloc[0]["exit_reason"] == "timeout"
    assert ledger.iloc[0]["bars_held"] == 2
    assert ledger.iloc[0]["exit_time"] == index[2]


def test_strict_one_second_execution_rejects_a_gap_inside_the_hold_window():
    index = _m15(3)
    seconds = _seconds(index[0], 3).delete(15 * 60 + 5)

    with np.testing.assert_raises_regex(ValueError, "missing 1s execution data"):
        simulate_bracket_trades_intrabar(
            _bars(index),
            _one_second(seconds),
            pd.Series({index[0]: 2}),
            tp_bps=100,
            sl_bps=100,
            max_hold=1,
            fee_bps=0,
            expected_interval=pd.Timedelta(seconds=1),
        )
