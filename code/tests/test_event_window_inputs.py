"""Causality contracts for the broad event-window input frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.event_window_inputs import (
    POSITIONING_FEATURES,
    RAW_FIVE_MINUTE_FEATURES,
    build_five_minute_feature_frame,
    build_positioning_feature_frame,
    known_structural_stop,
    merge_positioning_asof,
)


def _projected_five_minute_fixture(periods: int = 420) -> pd.DataFrame:
    index = pd.date_range("2024-01-02", periods=periods, freq="5min", tz="UTC")
    step = np.arange(periods, dtype=float)
    close = 100.0 * np.exp(0.0002 * step + 0.001 * np.sin(step / 7.0))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * 1.001
    low = np.minimum(open_, close) * 0.999
    volume = 100.0 + step % 17
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trade_count": 50.0 + step % 11,
            "taker_buy_base": volume * (0.45 + 0.05 * np.sin(step / 9.0)),
            "minute_count": 5,
            "activity_ratio": 1.0 + 0.1 * np.sin(step / 13.0),
            "channel_pos": 0.5,
            "channel_lower": close * 0.98,
            "channel_mid": close,
            "channel_upper": close * 1.02,
        },
        index=index,
    )


def _five_minute_fixture_with_gap() -> pd.DataFrame:
    raw = _projected_five_minute_fixture(periods=40)
    return raw.drop(pd.Timestamp("2024-01-02 01:00", tz="UTC"))


def _positioning_fixture(
    *,
    index: pd.DatetimeIndex | list[pd.Timestamp] | None = None,
    include_availability: bool = True,
    source_age_min: float = 0.0,
) -> pd.DataFrame:
    if index is None:
        index = pd.date_range("2024-01-01", periods=700, freq="15min", tz="UTC")
    index = pd.DatetimeIndex(index)
    step = np.arange(len(index), dtype=float)
    frame = pd.DataFrame(
        {
            "funding_rate": 0.0001 + step * 1e-8,
            "sum_open_interest": 1_000_000.0 * np.exp(step * 0.0001),
            "toptrader_ls": 1.1 + step * 1e-5,
            "taker_ls": 0.9 + step * 1e-5,
            "positioning_stale": False,
            "positioning_age_min": source_age_min,
        },
        index=index,
    )
    if include_availability:
        frame["availability_time"] = index + pd.Timedelta("15min")
    return frame


def _decisions_at(*times: str) -> pd.DataFrame:
    return pd.DataFrame({"decision_time": pd.to_datetime(list(times), utc=True)})


def test_feature_contracts_are_frozen():
    assert len(RAW_FIVE_MINUTE_FEATURES) == 19
    assert len(POSITIONING_FEATURES) == 16
    assert {"activity_ratio", "channel_pos", "cadence_gap"} <= set(
        RAW_FIVE_MINUTE_FEATURES
    )
    assert {
        "oi_chg_15m",
        "oi_chg_1h",
        "oi_chg_4h",
        "oi_accel_1h",
        "oi_z_7d",
        "positioning_age_min",
    } <= set(POSITIONING_FEATURES)


def test_positioning_features_include_multi_horizon_oi_and_age():
    raw = _positioning_fixture()
    out = build_positioning_feature_frame(raw)
    required = {
        "oi_chg_15m",
        "oi_chg_1h",
        "oi_chg_4h",
        "oi_accel_1h",
        "oi_z_7d",
        "funding_z",
        "toptrader_log_ratio",
        "taker_log_ratio",
        "positioning_stale",
        "positioning_age_min",
    }
    assert required <= set(out.columns)
    assert out.iloc[-1]["oi_chg_1h"] == pytest.approx(0.0004)
    trailing_oi = np.log(raw["sum_open_interest"].iloc[-672:])
    expected_oi_z = (trailing_oi.iloc[-1] - trailing_oi.mean()) / trailing_oi.std()
    assert out.iloc[-1]["oi_z_7d"] == pytest.approx(expected_oi_z)


def test_asof_join_never_uses_positioning_available_after_decision():
    features = build_positioning_feature_frame(
        _positioning_fixture(
            index=pd.to_datetime(
                ["2024-01-02 10:00", "2024-01-02 10:15"], utc=True
            )
        )
    )
    merged = merge_positioning_asof(
        _decisions_at("2024-01-02 10:10", "2024-01-02 10:15", "2024-01-02 10:29"),
        features,
    )
    matched = merged["positioning_availability_time"].notna()
    assert (
        merged.loc[matched, "positioning_availability_time"]
        <= merged.loc[matched, "decision_time"]
    ).all()
    assert merged.iloc[0]["positioning_missing"] == 1


def test_missing_availability_defaults_to_source_open_plus_15_minutes():
    raw = _positioning_fixture(
        index=[pd.Timestamp("2024-01-02 10:00", tz="UTC")],
        include_availability=False,
    )
    features = build_positioning_feature_frame(raw)
    decisions = _decisions_at("2024-01-02 10:10", "2024-01-02 10:15")
    merged = merge_positioning_asof(decisions, features)
    assert merged.iloc[0]["positioning_missing"] == 1
    assert merged.iloc[1]["positioning_availability_time"] == pd.Timestamp(
        "2024-01-02 10:15", tz="UTC"
    )


def test_nonmonotone_positioning_availability_is_rejected_before_transforms():
    raw = _positioning_fixture(
        index=pd.date_range("2024-01-02 10:00", periods=3, freq="15min", tz="UTC")
    )
    raw["availability_time"] = pd.to_datetime(
        ["2024-01-02 10:45", "2024-01-02 10:30", "2024-01-02 11:00"],
        utc=True,
    )
    with pytest.raises(ValueError, match="availability.*monotone"):
        build_positioning_feature_frame(raw)


def test_carried_positioning_age_increases_on_the_5m_grid():
    features = build_positioning_feature_frame(
        _positioning_fixture(
            index=[pd.Timestamp("2024-01-02 10:00", tz="UTC")],
            source_age_min=5.0,
        )
    )
    merged = merge_positioning_asof(
        _decisions_at("2024-01-02 10:15", "2024-01-02 10:25"), features
    )
    assert merged["positioning_age_min"].tolist() == [5.0, 15.0]


def test_carried_positioning_becomes_stale_after_sixty_minutes():
    features = build_positioning_feature_frame(
        _positioning_fixture(index=[pd.Timestamp("2024-01-02 10:00", tz="UTC")])
    )
    merged = merge_positioning_asof(
        _decisions_at("2024-01-02 11:15", "2024-01-02 11:20"), features
    )
    assert merged["positioning_age_min"].tolist() == [60.0, 65.0]
    assert merged["positioning_stale"].tolist() == [0, 1]


def test_unknown_positioning_age_is_not_presented_as_fresh():
    features = build_positioning_feature_frame(
        _positioning_fixture(
            index=[pd.Timestamp("2024-01-02 10:00", tz="UTC")],
            source_age_min=np.nan,
        )
    )
    merged = merge_positioning_asof(
        _decisions_at("2024-01-02 10:15"), features
    )
    assert np.isnan(merged.iloc[0]["positioning_age_min"])
    assert merged.iloc[0]["positioning_stale"] == 1
    assert merged.iloc[0]["positioning_missing"] == 0


def test_future_5m_mutation_cannot_change_prior_features():
    raw = _projected_five_minute_fixture()
    cutoff = raw.index[400]
    first = build_five_minute_feature_frame(raw)
    changed = raw.copy()
    changed.loc[changed.index > cutoff, "volume"] *= 100.0
    second = build_five_minute_feature_frame(changed)
    pd.testing.assert_frame_equal(first.loc[:cutoff], second.loc[:cutoff])


def test_gap_invalidates_trailing_features_and_sets_explicit_mask():
    out = build_five_minute_feature_frame(_five_minute_fixture_with_gap())
    after_gap = pd.Timestamp("2024-01-02 01:05", tz="UTC")
    assert out.loc[after_gap, "cadence_gap"] == 1
    assert pd.isna(out.loc[after_gap, "volume_log_ratio_24"])
    assert pd.isna(out.loc[after_gap, "realized_vol_12"])


def test_raw_feature_formulas_use_only_completed_trailing_values():
    raw = _projected_five_minute_fixture(periods=40)
    out = build_five_minute_feature_frame(raw)
    at = raw.index[30]
    previous = np.log1p(raw.loc[:at, "volume"].iloc[-25:-1]).median()
    expected_surprise = np.log1p(raw.loc[at, "volume"]) - previous
    assert out.loc[at, "volume_log_ratio_24"] == pytest.approx(expected_surprise)
    assert out.loc[at, "distance_lower_bps"] == pytest.approx(200.0)
    assert out.loc[at, "distance_upper_bps"] == pytest.approx(-200.0)


def test_native_binance_count_column_populates_trade_count_feature():
    raw = _projected_five_minute_fixture(periods=40).rename(
        columns={"trade_count": "count"}
    )
    out = build_five_minute_feature_frame(raw)
    assert np.isfinite(out.iloc[-1]["trade_count_log_ratio_24"])


def test_activity_ratio_is_built_when_projection_does_not_attach_it():
    raw = _projected_five_minute_fixture().drop(columns="activity_ratio")
    out = build_five_minute_feature_frame(raw)
    assert np.isfinite(out.iloc[-1]["activity_ratio"])


def test_structural_stop_uses_source_and_eleven_preceding_bars_only():
    raw = _projected_five_minute_fixture(periods=20)
    source = raw.index[15]
    history = raw.loc[:source].iloc[-12:]
    long_expected = history["low"].min() * (1.0 - 5.0 / 1e4)
    short_expected = history["high"].max() * (1.0 + 5.0 / 1e4)

    long_stop = known_structural_stop(raw, side="long", source_bar_time=source)
    short_stop = known_structural_stop(raw, side="short", source_bar_time=source)
    changed = raw.copy()
    changed.loc[changed.index > source, "low"] = 1.0

    assert long_stop == pytest.approx(long_expected)
    assert short_stop == pytest.approx(short_expected)
    assert known_structural_stop(
        changed, side="long", source_bar_time=source
    ) == pytest.approx(long_stop)


def test_structural_stop_is_missing_when_history_crosses_a_gap():
    raw = _five_minute_fixture_with_gap()
    stop = known_structural_stop(
        raw,
        side="long",
        source_bar_time=pd.Timestamp("2024-01-02 01:10", tz="UTC"),
    )
    assert np.isnan(stop)
