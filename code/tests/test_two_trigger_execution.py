"""Fixed execution behavior shared by Fast-T2 studies."""

import numpy as np
import pandas as pd

from evaluation.two_trigger_execution import resolve_fixed_trade


def _minute_bars(periods: int = 140) -> pd.DataFrame:
    index = pd.date_range("2024-01-01 00:00", periods=periods, freq="1min", tz="UTC")
    close = 100.0 + np.linspace(0.0, 0.2, periods)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
        },
        index=index,
    )


def test_fixed_trade_is_stop_first_when_one_bar_touches_both():
    bars = _minute_bars()
    entry_time = pd.Timestamp("2024-01-01 00:10", tz="UTC")
    bars.loc[entry_time, ["low", "high"]] = [99.0, 102.0]

    result = resolve_fixed_trade(
        bars,
        entry_time=entry_time,
        side="long",
        stop=99.5,
        target=101.5,
        max_hold_minutes=120,
        round_trip_cost_bps=10.0,
    )

    assert result is not None
    assert result.outcome == "sl"
    assert result.exit_price == 99.5
    assert result.r_net < -1.0


def test_fixed_trade_keeps_levels_after_delayed_entry():
    bars = _minute_bars()
    t2 = pd.Timestamp("2024-01-01 00:10", tz="UTC")

    first = resolve_fixed_trade(
        bars, entry_time=t2, side="long", stop=99.5, target=103.0
    )
    delayed = resolve_fixed_trade(
        bars,
        entry_time=t2 + pd.Timedelta(minutes=4),
        side="long",
        stop=99.5,
        target=103.0,
    )

    assert first is not None and delayed is not None
    assert first.stop_price == delayed.stop_price == 99.5
    assert first.target_price == delayed.target_price == 103.0


def test_fixed_trade_returns_none_for_an_incomplete_path():
    bars = _minute_bars(periods=30)

    result = resolve_fixed_trade(
        bars,
        entry_time=pd.Timestamp("2024-01-01 00:10", tz="UTC"),
        side="short",
        stop=101.0,
        target=99.0,
        max_hold_minutes=120,
    )

    assert result is None
