import numpy as np
import pandas as pd
import pytest

from features.intrabar import (
    INTRABAR_FEATURES,
    STAGE_FEATURES,
    build_intrabar_features,
    build_market_stage_features,
)


def _minute_frame(periods: int = 30) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="1min", tz="UTC")
    close = np.full(periods, 100.0)
    if periods >= 30:
        close[25:] = np.linspace(100.0, 105.0, 5)
    return pd.DataFrame(
        {
            "open": np.r_[100.0, close[:-1]],
            "high": np.maximum(np.r_[100.0, close[:-1]], close) + 0.1,
            "low": np.minimum(np.r_[100.0, close[:-1]], close) - 0.1,
            "close": close,
            "volume": 10.0,
            "taker_buy_base": np.resize([4.0, 6.0], periods),
        },
        index=index,
    )


def test_intrabar_features_use_each_completed_m15_group_only():
    result = build_intrabar_features(_minute_frame())

    assert tuple(result.columns) == INTRABAR_FEATURES
    assert result.index.tolist() == [
        pd.Timestamp("2025-01-01 00:00", tz="UTC"),
        pd.Timestamp("2025-01-01 00:15", tz="UTC"),
    ]
    assert result.iloc[0]["m1_final5_return"] == pytest.approx(0.0)
    assert result.iloc[1]["m1_final5_return"] > 0.0
    assert result.iloc[0]["m1_late_volume_share"] == pytest.approx(1 / 3)


def test_intrabar_features_reject_incomplete_m15_group():
    with pytest.raises(ValueError, match="complete 15-minute groups"):
        build_intrabar_features(_minute_frame(periods=29))


def test_market_stage_features_are_past_only():
    index = pd.date_range("2025-01-01", periods=140, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": np.linspace(100.0, 120.0, len(index))}, index=index)
    changed = bars.copy()
    changed.iloc[-1, 0] = 1_000.0

    original_features = build_market_stage_features(bars)
    changed_features = build_market_stage_features(changed)

    assert tuple(original_features.columns) == STAGE_FEATURES
    pd.testing.assert_frame_equal(original_features.iloc[:-1], changed_features.iloc[:-1])

def test_regime_stage_features_match_the_causal_seven_day_window():
    from features import intrabar as intrabar_module

    regime_stage_features = intrabar_module.REGIME_STAGE_FEATURES
    index = pd.date_range("2025-01-01", periods=700, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 140.0, len(index)), index=index)
    bars = pd.DataFrame({"close": close})

    result = build_market_stage_features(bars, include_regime_stage=True)
    last = result.iloc[-1]
    window = close.iloc[-673:]

    assert tuple(result.loc[:, regime_stage_features].columns) == regime_stage_features
    assert last["stage_return_7d"] == pytest.approx(close.iloc[-1] / close.iloc[-673] - 1.0)
    assert last["stage_drawdown_7d"] == pytest.approx(close.iloc[-1] / window.max() - 1.0)
    assert last["stage_rebound_7d"] == pytest.approx(close.iloc[-1] / window.min() - 1.0)



