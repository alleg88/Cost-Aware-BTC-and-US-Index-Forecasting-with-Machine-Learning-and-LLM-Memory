from __future__ import annotations

import numpy as np
import pandas as pd

import experiments.event_window_dataset as dataset_module
from experiments.event_window_dataset import build_event_window_sequences
from features.event_windows import EventWindowConfig


UTC = "UTC"


def _five_features(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="5min", tz=UTC)
    x = np.arange(periods, dtype=float)
    close = 100.0 + 0.02 * x + 0.3 * np.sin(x / 7.0)
    open_ = close - 0.02 * np.cos(x / 5.0)
    high = np.maximum(open_, close) + 0.10
    low = np.minimum(open_, close) - 0.10
    lower = close - 1.0
    upper = close + 1.0
    mid = (lower + upper) / 2.0
    log_close = np.log(close)
    returns = pd.Series(log_close, index=index).diff()
    out = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "minute_count": 5,
            "log_return": returns,
            "range_bps": (high - low) / open_ * 1e4,
            "body_bps": (close - open_) / open_ * 1e4,
            "lower_wick_fraction": (np.minimum(open_, close) - low) / (high - low),
            "upper_wick_fraction": (high - np.maximum(open_, close)) / (high - low),
            "volume_log_ratio_24": 0.1,
            "quote_volume_log_ratio_24": 0.2,
            "trade_count_log_ratio_24": 0.05,
            "taker_imbalance": np.sin(x / 9.0) * 0.2,
            "taker_imbalance_mean_3": np.sin(x / 9.0) * 0.15,
            "taker_imbalance_delta": 0.01,
            "realized_vol_12": returns.rolling(12).std(),
            "activity_ratio": 1.0,
            "channel_pos": 0.25,
            "channel_lower": lower,
            "channel_mid": mid,
            "channel_upper": upper,
            "distance_lower_bps": (close - lower) / close * 1e4,
            "distance_mid_bps": (close - mid) / close * 1e4,
            "distance_upper_bps": (close - upper) / close * 1e4,
            "channel_slope_bps": 6.0,
            "channel_r2": 0.45,
            "channel_width": upper - lower,
            "channel_confluence": 1,
            "channel_confluence_count": 2,
            "channel_sign_90": 1,
            "channel_sign_120": 1,
            "channel_regime_age_hours": 10.0,
            "rsi_channel": 45.0,
            "bar_complete": 1,
            "cadence_gap": 0,
        },
        index=index,
    )
    return out


def _positioning() -> pd.DataFrame:
    index = pd.date_range("2023-12-31 23:00", periods=80, freq="15min", tz=UTC)
    x = np.arange(len(index), dtype=float)
    return pd.DataFrame(
        {
            "positioning_availability_time": index,
            "oi_chg_15m": 0.001 + x * 0.0,
            "oi_chg_1h": 0.002 + x * 0.0,
            "oi_chg_4h": 0.004 + x * 0.0,
            "oi_accel_1h": 0.0002,
            "oi_z_7d": 0.1,
            "funding_rate": 0.0001,
            "funding_z": 0.2,
            "toptrader_log_ratio": 0.05,
            "taker_log_ratio": -0.03,
            "oi_missing": 0,
            "funding_missing": 0,
            "toptrader_missing": 0,
            "taker_ratio_missing": 0,
            "positioning_missing": 0,
            "positioning_stale": 0,
            "positioning_age_min": 0.0,
        },
        index=index,
    )


def _manifest(*, early_steps: int | None = None) -> pd.DataFrame:
    starts = [pd.Timestamp("2024-01-01 05:00", tz=UTC), pd.Timestamp("2024-01-01 10:00", tz=UTC)]
    sides = ["long", "short"]
    rows = []
    for number, (start, side) in enumerate(zip(starts, sides, strict=True), start=1):
        steps = early_steps if early_steps is not None and number == 1 else 12
        rows.append(
            {
                "window_id": f"w{number}",
                "channel_episode_id": f"ep{number}",
                "side": side,
                "window_start": start,
                "window_end": start + pd.Timedelta(minutes=5 * steps),
                "source_bar_time": start - pd.Timedelta("5min"),
                "end_reason": "slope_reversal" if steps < 12 else "timeout",
            }
        )
    return pd.DataFrame(rows)


def test_one_sample_is_one_window_with_24_pre_and_12_active_steps():
    data = build_event_window_sequences(
        _manifest(), _five_features(), _positioning(), EventWindowConfig()
    )
    assert data.sequence.shape[:2] == (len(data.metadata), 36)
    assert data.context.shape[:2] == (len(data.metadata), 12)
    assert data.source_bar_times.shape == data.decision_times.shape
    assert data.decision_times.shape == data.decision_valid.shape
    assert set(data.metadata["side"]) == {"long", "short"}
    assert data.sequence.dtype == np.float32
    assert data.context.dtype == np.float32


def test_active_timestamps_are_source_close_boundaries_before_window_end():
    data = build_event_window_sequences(
        _manifest(), _five_features(), _positioning(), EventWindowConfig()
    )
    valid = data.decision_valid
    np.testing.assert_array_equal(
        data.decision_times[valid],
        data.source_bar_times[valid] + np.timedelta64(5, "m"),
    )
    for row, end in enumerate(pd.to_datetime(data.metadata["window_end"], utc=True)):
        decisions = pd.to_datetime(data.decision_times[row][valid[row]], utc=True)
        assert (decisions < end).all()


def test_incomplete_or_gapped_precontext_invalidates_window_sample():
    five = _five_features()
    five = five.drop(pd.Timestamp("2024-01-01 03:30", tz=UTC))
    data = build_event_window_sequences(
        _manifest().iloc[[0]], five, _positioning(), EventWindowConfig()
    )
    assert data.metadata.empty


def test_early_closed_window_masks_unobserved_active_steps():
    data = build_event_window_sequences(
        _manifest(early_steps=4).iloc[[0]],
        _five_features(),
        _positioning(),
        EventWindowConfig(),
    )
    assert data.decision_valid[0].sum() == 4
    assert (~data.sequence_valid[0, 28:]).all()


def test_active_step_tensor_does_not_change_when_later_bars_change():
    manifest = _manifest().iloc[[0]]
    first_five = _five_features()
    second_five = first_five.copy()
    later = pd.Timestamp("2024-01-01 05:25", tz=UTC)
    second_five.loc[second_five.index >= later, ["close", "high", "low"]] *= 3.0
    first = build_event_window_sequences(
        manifest, first_five, _positioning(), EventWindowConfig()
    )
    second = build_event_window_sequences(
        manifest, second_five, _positioning(), EventWindowConfig()
    )
    np.testing.assert_allclose(first.sequence[:, :29], second.sequence[:, :29], equal_nan=True)
    np.testing.assert_allclose(first.context[:, :5], second.context[:, :5], equal_nan=True)


def test_short_side_flips_directional_features_without_splitting_dataset():
    data = build_event_window_sequences(
        _manifest(), _five_features(), _positioning(), EventWindowConfig()
    )
    side_feature = data.context_features.index("side_sign")
    slope_feature = data.context_features.index("channel_slope_side")
    assert data.context[0, 0, side_feature] == 1.0
    assert data.context[1, 0, side_feature] == -1.0
    assert data.context[0, 0, slope_feature] == 6.0
    assert data.context[1, 0, slope_feature] == -6.0


def test_shared_inputs_are_not_rebuilt_for_every_window_or_step(monkeypatch):
    merge_calls = 0
    largest_stop_frame = 0
    original_merge = dataset_module.merge_positioning_asof
    original_stop = dataset_module.known_structural_stop

    def counted_merge(*args, **kwargs):
        nonlocal merge_calls
        merge_calls += 1
        return original_merge(*args, **kwargs)

    def bounded_stop(frame, **kwargs):
        nonlocal largest_stop_frame
        largest_stop_frame = max(largest_stop_frame, len(frame))
        return original_stop(frame, **kwargs)

    monkeypatch.setattr(dataset_module, "merge_positioning_asof", counted_merge)
    monkeypatch.setattr(dataset_module, "known_structural_stop", bounded_stop)
    build_event_window_sequences(
        _manifest(), _five_features(), _positioning(), EventWindowConfig()
    )
    assert merge_calls == 1
    assert largest_stop_frame == 12
