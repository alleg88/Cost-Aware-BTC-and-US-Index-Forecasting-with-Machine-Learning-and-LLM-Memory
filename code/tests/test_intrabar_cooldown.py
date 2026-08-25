import pandas as pd

from evaluation.trades_intrabar import simulate_bracket_trades_intrabar


def test_cooldown_skips_full_bars_after_each_exit():
    index = pd.date_range("2025-01-01", periods=12, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
        index=index,
    )
    minute_index = pd.date_range(
        index[0], periods=len(index) * 15, freq="1min", tz="UTC"
    )
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
        index=minute_index,
    )
    prediction = pd.Series(2, index=index)

    ledger, _ = simulate_bracket_trades_intrabar(
        bars,
        minute,
        prediction,
        tp_bps=1_000,
        sl_bps=1_000,
        max_hold=1,
        fee_bps=0,
        cooldown_bars=4,
    )

    assert ledger["entry_time"].tolist() == [index[1], index[6], index[11]]
