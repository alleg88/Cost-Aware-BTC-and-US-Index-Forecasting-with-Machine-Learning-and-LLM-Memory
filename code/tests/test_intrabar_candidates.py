import pandas as pd
import pytest

from experiments.intrabar_candidates import (
    SL_FIRST,
    TIMEOUT,
    TP_FIRST,
    build_first_touch_candidates,
)


def _future_minutes(*, high: float, low: float, close: float = 100.0) -> pd.DataFrame:
    index = pd.date_range("2025-01-01 00:15", periods=15, freq="1min", tz="UTC")
    frame = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": close},
        index=index,
    )
    frame.iloc[0, frame.columns.get_loc("high")] = high
    frame.iloc[0, frame.columns.get_loc("low")] = low
    return frame


@pytest.mark.parametrize(
    ("side_class", "high", "low", "expected"),
    [
        (2, 101.5, 99.5, TP_FIRST),
        (2, 100.5, 98.5, SL_FIRST),
        (0, 100.5, 98.5, TP_FIRST),
        (2, 101.5, 98.5, SL_FIRST),
    ],
)
def test_first_touch_labels_long_short_and_stop_first_tie(
    side_class: int, high: float, low: float, expected: int
):
    signal_time = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    signals = pd.Series([side_class], index=[signal_time])
    features = pd.DataFrame({"known_feature": [7.0]}, index=[signal_time])

    result = build_first_touch_candidates(
        _future_minutes(high=high, low=low),
        signals,
        features,
        tp_bps=100.0,
        sl_bps=100.0,
        max_hold=1,
    )

    assert result.iloc[0]["outcome"] == expected
    assert result.iloc[0]["known_feature"] == 7.0
    assert result.iloc[0]["outcome_close_time"] == pd.Timestamp(
        "2025-01-01 00:16", tz="UTC"
    )


def test_first_touch_timeout_and_next_m15_entry_alignment():
    signal_time = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    index = pd.date_range(signal_time, periods=30, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        index=index,
    )
    minute.loc[index[:15], "high"] = 110.0
    signals = pd.Series([2], index=[signal_time])
    features = pd.DataFrame({"known_feature": [3.0]}, index=[signal_time])

    result = build_first_touch_candidates(
        minute,
        signals,
        features,
        tp_bps=100.0,
        sl_bps=100.0,
        max_hold=1,
    )

    assert result.iloc[0]["outcome"] == TIMEOUT
    assert result.iloc[0]["entry_time"] == pd.Timestamp("2025-01-01 00:15", tz="UTC")
    assert result.iloc[0]["outcome_close_time"] == pd.Timestamp(
        "2025-01-01 00:30", tz="UTC"
    )
