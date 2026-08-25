import pandas as pd
import pytest

from evaluation.event_window_large_move_policy import (
    LargeMovePolicyConfig,
    policy_summary,
    replay_large_move_policy,
    select_calibration_threshold,
)


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1", "w1", "w2"],
            "channel_episode_id": ["e1", "e1", "e2"],
            "step": [0, 1, 0],
            "decision_time": pd.to_datetime(
                ["2024-01-01 00:00Z", "2024-01-01 00:05Z", "2024-01-02 00:00Z"]
            ),
            "p_no_big": [0.50, 0.20, 0.20],
            "p_up_big": [0.30, 0.65, 0.15],
            "p_down_big": [0.20, 0.15, 0.65],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1", "w1", "w2"],
            "step": [0, 1, 0],
            "move_code": [0, 1, 2],
            "move_label": ["no_big_move", "up_big", "down_big"],
            "adaptive_barrier_bps": [100.0, 100.0, 120.0],
            "terminal_return_bps": [5.0, 100.0, -120.0],
            "model_target_valid": [True, True, True],
        }
    )


def test_policy_waits_then_selects_its_own_direction_once_per_window():
    selected = replay_large_move_policy(_scores(), _labels(), threshold=0.50)
    assert selected[["window_id", "step"]].values.tolist() == [["w1", 1], ["w2", 0]]
    assert selected["predicted_direction"].tolist() == ["long", "short"]
    assert selected["direction_correct"].all()
    assert selected["net_bps"].tolist() == pytest.approx([93.0, 113.0])


def test_positive_ev_direction_can_trade_even_when_no_big_is_largest_class():
    scores = _scores().iloc[[0]].copy()
    labels = _labels().iloc[[0]].copy()
    selected = replay_large_move_policy(scores, labels, threshold=0.25)
    assert len(selected) == 1
    assert selected.iloc[0].neutral_timeout_ev_proxy_bps > 0.0


def test_negative_ev_direction_is_not_forced_by_a_low_threshold():
    scores = _scores().iloc[[0]].copy()
    scores[["p_no_big", "p_up_big", "p_down_big"]] = [0.50, 0.27, 0.23]
    labels = _labels().iloc[[0]].copy()
    selected = replay_large_move_policy(scores, labels, threshold=0.20)
    assert selected.empty
    for column in ("window_id", "step", "net_bps", "net_r", "predicted_direction"):
        assert column in selected.columns


def test_no_big_trade_uses_terminal_return_and_taker_cost():
    scores = _scores().iloc[[2]].copy()
    scores[["p_no_big", "p_up_big", "p_down_big"]] = [0.20, 0.65, 0.15]
    labels = _labels().iloc[[2]].copy()
    labels[["move_code", "move_label", "terminal_return_bps"]] = [0, "no_big_move", 20.0]
    selected = replay_large_move_policy(scores, labels, threshold=0.50)
    assert selected.iloc[0].trade_outcome == "no_big_timeout"
    assert selected.iloc[0].net_bps == pytest.approx(10.0)


def test_threshold_selection_reports_frontier_and_frequency_band():
    config = LargeMovePolicyConfig(threshold_grid=(0.50, 0.90))
    threshold, frontier = select_calibration_threshold(
        _scores(), _labels(), config=config
    )
    assert threshold == 0.50
    assert frontier["selected"].sum() == 1
    summary = policy_summary(
        replay_large_move_policy(_scores(), _labels(), threshold=threshold), _scores()
    )
    assert summary["trades_per_day"] == pytest.approx(1.0)
    assert summary["directional_precision"] == pytest.approx(1.0)
