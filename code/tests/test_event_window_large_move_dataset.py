from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.event_window_large_move_dataset import (
    AdaptiveMoveConfig,
    MOVE_CODES,
    VOLATILITY_FEATURE_COLUMNS,
    build_large_move_dataset,
    build_opportunity_dataset,
    label_adaptive_large_moves,
)


def _minute(*, future: str = "up", volatile: bool = False) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=260, freq="1min", tz="UTC")
    close = np.full(len(index), 100.0)
    if volatile:
        close[:122] = 100.0 + np.sin(np.arange(122)) * 0.8
    frame = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": np.arange(len(index), dtype=float) + 10.0,
            "trade_count": np.arange(len(index), dtype=float) + 100.0,
            "taker_buy_base": (np.arange(len(index), dtype=float) + 10.0) * 0.55,
        },
        index=index,
    )
    decision = 122
    if future == "up":
        frame.iloc[decision + 3, frame.columns.get_loc("high")] = 101.0
    elif future == "down":
        frame.iloc[decision + 2, frame.columns.get_loc("low")] = 99.0
    elif future == "tie":
        frame.iloc[decision + 1, frame.columns.get_loc("high")] = 101.0
        frame.iloc[decision + 1, frame.columns.get_loc("low")] = 99.0
    return frame


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1"],
            "channel_episode_id": ["e1"],
            "side": ["long"],
            "step": [0],
            "decision_time": [pd.Timestamp("2024-01-01 02:02", tz="UTC")],
        }
    )


def test_adaptive_label_uses_past_floor_and_first_touch():
    labels = label_adaptive_large_moves(_decisions(), _minute(future="up"))
    row = labels.iloc[0]
    assert row.move_label == "up_big"
    assert row.move_code == MOVE_CODES["up_big"]
    assert row.model_target_valid
    assert row.adaptive_barrier_bps == pytest.approx(75.0)
    assert row.label_end == pd.Timestamp("2024-01-01 02:06", tz="UTC")


def test_barrier_is_dynamic_but_bounded():
    minute = _minute(future="none", volatile=True)
    labels = label_adaptive_large_moves(_decisions(), minute)
    row = labels.iloc[0]
    history_close = minute["close"].iloc[1:122].to_numpy(float)
    expected_sigma = np.std(
        np.diff(np.log(history_close[::-5][:13][::-1])), ddof=1
    ) * 1e4
    assert row.past_sigma_5m_bps == pytest.approx(expected_sigma)
    assert 75.0 < row.adaptive_barrier_bps <= 250.0
    assert row.adaptive_barrier_bps == pytest.approx(
        min(250.0, 75.0 + 0.5 * row.past_sigma_5m_bps * np.sqrt(24.0))
    )


def test_same_minute_double_touch_is_opportunity_valid_but_direction_invalid():
    labels = label_adaptive_large_moves(_decisions(), _minute(future="tie"))
    assert labels.iloc[0].move_label == "ambiguous"
    assert labels.iloc[0].move_code == -1
    assert labels.iloc[0].opportunity_code == 1
    assert labels.iloc[0].opportunity_target_valid
    assert not labels.iloc[0].direction_target_valid
    assert not labels.iloc[0].model_target_valid


def test_future_gap_without_prior_touch_is_censored():
    minute = _minute(future="none").drop(pd.Timestamp("2024-01-01 02:20", tz="UTC"))
    labels = label_adaptive_large_moves(_decisions(), minute)
    assert labels.iloc[0].move_label == "censored"
    assert not labels.iloc[0].opportunity_target_valid
    assert not labels.iloc[0].model_target_valid


def test_future_changes_do_not_change_the_causal_barrier():
    up = label_adaptive_large_moves(_decisions(), _minute(future="up")).iloc[0]
    down = label_adaptive_large_moves(_decisions(), _minute(future="down")).iloc[0]
    assert up.adaptive_barrier_bps == down.adaptive_barrier_bps
    assert up.past_rv_60_bps == down.past_rv_60_bps
    assert up.move_label != down.move_label


def test_decision_open_anchors_the_barriers_but_not_past_volatility():
    plain = _minute(future="none")
    gapped = plain.copy()
    decision = gapped.index[122]
    gapped.loc[decision, ["open", "high", "low", "close"]] = [105.0, 105.02, 104.98, 105.0]
    original = label_adaptive_large_moves(_decisions(), plain).iloc[0]
    changed = label_adaptive_large_moves(_decisions(), gapped).iloc[0]
    assert original.reference_price == pytest.approx(100.0)
    assert changed.reference_price == pytest.approx(105.0)
    assert changed.adaptive_barrier_bps == original.adaptive_barrier_bps
    assert changed.past_sigma_5m_bps == original.past_sigma_5m_bps
    assert changed.move_label != "up_big"


def test_one_censored_decision_excludes_the_complete_window():
    decisions = pd.concat(
        [
            _decisions(),
            _decisions().assign(
                step=1,
                decision_time=pd.Timestamp("2024-01-01 03:30", tz="UTC"),
            ),
        ],
        ignore_index=True,
    )
    labels = label_adaptive_large_moves(decisions, _minute(future="up"))
    assert labels["row_target_valid"].tolist() == [True, False]
    assert labels["incomplete_window"].all()
    assert not labels["model_target_valid"].any()
    assert labels["opportunity_incomplete_window"].all()
    assert not labels["opportunity_target_valid"].any()


def test_feature_sets_prune_masks_and_add_raw_directional_features():
    labels = label_adaptive_large_moves(_decisions(), _minute(future="up"))
    base = SimpleNamespace(
        decisions=_decisions()[["window_id", "step"]],
        tabular=np.array([[1.0, -2.0, -1.0, 7.0]], dtype=np.float32),
        tabular_features=(
            "log_return_side",
            "channel_slope_side",
            "side_sign",
            "active_window_mask",
        ),
    )
    simple = build_large_move_dataset(base, labels, feature_set="base")
    assert simple.tabular_features == (
        "log_return_side",
        "channel_slope_side",
        "side_sign",
        "adaptive_barrier_bps",
    )
    directional = build_large_move_dataset(base, labels, feature_set="directional")
    assert "raw_log_return" in directional.tabular_features
    assert "past_rv_60_bps" not in directional.tabular_features
    expanded = build_large_move_dataset(base, labels, feature_set="volatility")
    assert set(VOLATILITY_FEATURE_COLUMNS).issubset(expanded.tabular_features)
    assert "raw_log_return" in expanded.tabular_features
    assert "raw_channel_slope" in expanded.tabular_features
    raw = dict(zip(expanded.tabular_features, expanded.tabular[0], strict=True))
    assert raw["raw_log_return"] == pytest.approx(-1.0)
    assert raw["raw_channel_slope"] == pytest.approx(2.0)
    assert "active_window_mask" in expanded.dropped_features


def test_directional_features_reconstruct_physical_channel_and_positioning_values():
    labels = label_adaptive_large_moves(_decisions(), _minute(future="up"))
    names = (
        "side_sign",
        "channel_position_side",
        "distance_adverse_rail_bps",
        "distance_favourable_rail_bps",
        "distance_mid_bps_side",
        "rsi_channel_side",
        "taker_imbalance_delta_side",
        "oi_accel_1h_side",
        "funding_z_side",
        "toptrader_log_ratio_side",
        "taker_log_ratio_side",
        "price_oi_interaction",
        "log_return_side_delta_3",
        "log_return_side_mean_3",
        "log_return_side_std_3",
        "channel_position_side_delta_3",
        "channel_position_side_mean_3",
        "distance_adverse_rail_bps_delta_3",
        "distance_favourable_rail_bps_delta_3",
        "distance_adverse_rail_bps_std_3",
        "distance_favourable_rail_bps_std_3",
    )
    values = np.array(
        [[
            -1.0, 0.2, 10.0, 20.0, 5.0, 30.0, 0.3, 0.4, -1.2, 0.2,
            -0.3, 0.5, 2.0, -3.0, 4.0, 0.1, 0.3, 1.0, 2.0, 3.0, 4.0,
        ]],
        dtype=np.float32,
    )
    base = SimpleNamespace(
        decisions=_decisions()[["window_id", "step"]],
        tabular=values,
        tabular_features=names,
    )
    dataset = build_large_move_dataset(base, labels, feature_set="directional")
    raw = dict(zip(dataset.tabular_features, dataset.tabular[0], strict=True))
    assert raw["raw_channel_position"] == pytest.approx(0.8)
    assert raw["raw_distance_lower_bps"] == pytest.approx(20.0)
    assert raw["raw_distance_upper_bps"] == pytest.approx(-10.0)
    assert raw["raw_distance_mid_bps"] == pytest.approx(-5.0)
    assert raw["raw_rsi_channel"] == pytest.approx(70.0)
    assert raw["raw_taker_imbalance_delta"] == pytest.approx(-0.3)
    assert raw["raw_oi_accel_1h"] == pytest.approx(-0.4)
    assert raw["raw_funding_z"] == pytest.approx(1.2)
    assert raw["raw_toptrader_log_ratio"] == pytest.approx(-0.2)
    assert raw["raw_taker_log_ratio"] == pytest.approx(0.3)
    assert raw["raw_price_oi_interaction"] == pytest.approx(-0.5)
    assert raw["raw_log_return_delta_3"] == pytest.approx(-2.0)
    assert raw["raw_log_return_mean_3"] == pytest.approx(3.0)
    assert "raw_log_return_std_3" not in raw
    assert raw["raw_channel_position_delta_3"] == pytest.approx(-0.1)
    assert raw["raw_channel_position_mean_3"] == pytest.approx(0.7)
    assert raw["raw_distance_lower_bps_delta_3"] == pytest.approx(2.0)
    assert raw["raw_distance_upper_bps_delta_3"] == pytest.approx(-1.0)
    assert raw["raw_distance_lower_bps_std_3"] == pytest.approx(4.0)
    assert raw["raw_distance_upper_bps_std_3"] == pytest.approx(3.0)


def test_opportunity_features_are_invariant_to_equivalent_side_encoding():
    labels = label_adaptive_large_moves(_decisions(), _minute(future="up"))
    names = (
        "side_sign",
        "log_return_side",
        "channel_slope_side",
        "channel_position_side",
        "distance_adverse_rail_bps",
        "distance_favourable_rail_bps",
        "rsi_channel_side",
        "funding_rate_side",
        "range_bps",
        "realized_vol_12",
        "window_age_fraction",
        "retest_count",
        "structural_risk_bps_known",
    )
    long_values = np.array(
        [[1.0, 2.0, 3.0, 0.25, 10.0, 20.0, 40.0, 0.1, 8.0, 5.0, 0.5, 2.0, 25.0]],
        dtype=np.float32,
    )
    short_values = np.array(
        [[-1.0, -2.0, -3.0, 0.75, 20.0, 10.0, 60.0, -0.1, 8.0, 5.0, 0.5, 2.0, 25.0]],
        dtype=np.float32,
    )
    long_base = SimpleNamespace(
        decisions=_decisions()[["window_id", "step"]],
        tabular=long_values,
        tabular_features=names,
    )
    short_base = SimpleNamespace(
        decisions=_decisions()[["window_id", "step"]],
        tabular=short_values,
        tabular_features=names,
    )
    long = build_opportunity_dataset(long_base, labels, include_volatility=False)
    short = build_opportunity_dataset(short_base, labels, include_volatility=False)
    assert long.tabular_features == short.tabular_features
    assert np.allclose(long.tabular, short.tabular, equal_nan=True)
    assert "raw_log_return" in long.tabular_features
    assert "raw_channel_position" in long.tabular_features
    assert "side_sign" not in long.tabular_features
    assert not any("_side" in name for name in long.tabular_features)
    assert "retest_count" not in long.tabular_features
    assert "structural_risk_bps_known" not in long.tabular_features


def test_opportunity_volatility_arm_adds_only_registered_causal_block():
    labels = label_adaptive_large_moves(_decisions(), _minute(future="up"))
    base = SimpleNamespace(
        decisions=_decisions()[["window_id", "step"]],
        tabular=np.array([[1.0, 4.0]], dtype=np.float32),
        tabular_features=("side_sign", "range_bps"),
    )
    isolated = build_opportunity_dataset(base, labels, include_volatility=False)
    expanded = build_opportunity_dataset(base, labels, include_volatility=True)
    added = set(expanded.tabular_features).difference(isolated.tabular_features)
    assert added == set(VOLATILITY_FEATURE_COLUMNS).difference({"adaptive_barrier_bps"})
    assert expanded.decisions["opportunity_target_valid"].all()


def test_adaptive_config_rejects_inconsistent_history_contract():
    with pytest.raises(ValueError, match="divisible by five"):
        AdaptiveMoveConfig(volatility_lookback_minutes=61)
    with pytest.raises(ValueError, match="120-minute"):
        AdaptiveMoveConfig(feature_history_minutes=119)
