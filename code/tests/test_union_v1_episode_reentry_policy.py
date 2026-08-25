from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.union_v1_episode_reentry_policy import (
    build_union_v1_style_signals,
    evaluate_reentry_development,
    identify_same_side_episodes,
    replay_union_control_and_reentry,
    select_episode_reentries,
)


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_key": ["lstm_074_svm_flat", "lstm_075_svm_flat", "lstm_flat_svm_short"],
            "decision_time": pd.date_range(
                "2021-01-01", periods=3, freq="15min", tz="UTC"
            ),
            "p_short_lstm": [0.13, 0.10, 0.10],
            "p_flat_lstm": [0.13, 0.15, 0.80],
            "p_long_lstm": [0.74, 0.75, 0.10],
            "pred_lstm": [2, 2, 1],
            "pred_svm_linear": [1, 1, 0],
        }
    )


def _signal_episode() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "row_key": [f"row-{index}" for index in range(7)],
            "decision_time": pd.to_datetime(
                [
                    "2021-01-01 00:00Z",
                    "2021-01-01 00:15Z",
                    "2021-01-01 00:30Z",
                    "2021-01-01 00:45Z",
                    "2021-01-01 01:00Z",
                    "2021-01-01 01:15Z",
                    "2021-01-01 01:30Z",
                ],
                utc=True,
            ),
            "union_signal": [1, 1, 1, 0, -1, -1, -1],
        }
    )
    return identify_same_side_episodes(frame)


def _control_ledger(net: float = 0.02) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_time": pd.to_datetime(
                ["2021-01-01 00:00Z", "2021-01-01 00:30Z", "2021-01-01 01:00Z"],
                utc=True,
            ),
            "entry_time": pd.to_datetime(
                ["2021-01-01 00:15Z", "2021-01-01 00:45Z", "2021-01-01 01:15Z"],
                utc=True,
            ),
            "exit_time": pd.to_datetime(
                ["2021-01-01 00:15Z", "2021-01-01 00:45Z", "2021-01-01 01:15Z"],
                utc=True,
            ),
            "side": [1, 1, -1],
            "net_return": [net, net, net],
        }
    )


def _summary(
    *,
    trades: int,
    net: float,
    long_net: float,
    short_net: float,
    long_trades: int,
    short_trades: int,
) -> dict[str, object]:
    return {
        "trades": trades,
        "net_return": net,
        "long_net_return": long_net,
        "short_net_return": short_net,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "audit_clean": True,
    }


def _passing_folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "fold_id": range(5),
            "control_net_return": [0.01, 0.01, 0.01, -0.001, -0.001],
            "incremental_net_return": [0.001, 0.0, 0.002, -0.001, -0.001],
        }
    )


def test_union_uses_frozen_lstm_tau_and_svm_class_only() -> None:
    signals = build_union_v1_style_signals(_predictions()).set_index("row_key")

    assert signals.loc["lstm_074_svm_flat", "union_signal"] == 0
    assert signals.loc["lstm_075_svm_flat", "union_signal"] == 1
    assert signals.loc["lstm_flat_svm_short", "union_signal"] == -1


def test_opposite_active_members_veto_the_trade() -> None:
    opposite = _predictions().iloc[[1]].copy()
    opposite["pred_svm_linear"] = 0
    signals = build_union_v1_style_signals(opposite)

    assert bool(signals.iloc[0]["member_conflict"])
    assert signals.iloc[0]["union_signal"] == 0


def test_candidate_adds_only_earliest_skipped_bar_once_per_episode() -> None:
    selected = select_episode_reentries(_signal_episode(), _control_ledger())

    assert selected["signal_time"].tolist() == [
        pd.Timestamp("2021-01-01 00:15", tz="UTC"),
        pd.Timestamp("2021-01-01 01:15", tz="UTC"),
    ]
    assert selected["episode_id"].nunique() == len(selected)


def test_outcomes_cannot_change_reentry_selection() -> None:
    left = select_episode_reentries(_signal_episode(), _control_ledger(net=0.02))
    right = select_episode_reentries(_signal_episode(), _control_ledger(net=-0.02))

    pd.testing.assert_frame_equal(left, right)


def test_replay_preserves_control_and_adds_one_nonoverlapping_trade() -> None:
    index = pd.date_range("2021-01-01", periods=7, freq="15min", tz="UTC")
    signals = identify_same_side_episodes(
        pd.DataFrame(
            {
                "row_key": [f"row-{index}" for index in range(7)],
                "decision_time": index,
                "union_signal": [1, 1, 1, 1, 0, 0, 0],
            }
        )
    )
    m15 = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=index,
    )
    minute_index = pd.date_range(
        "2021-01-01", periods=7 * 15, freq="1min", tz="UTC"
    )
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=minute_index,
    )

    replay = replay_union_control_and_reentry(signals, m15, minute)

    assert len(replay.control_ledger) == 2
    assert len(replay.reentry_ledger) == 1
    assert len(replay.candidate_ledger) == 3
    control_columns = replay.control_ledger.columns.tolist()
    pd.testing.assert_frame_equal(
        replay.candidate_ledger.loc[
            replay.candidate_ledger["route"].eq("union_control"), control_columns
        ].reset_index(drop=True),
        replay.control_ledger.reset_index(drop=True),
    )
    assert replay.candidate_ledger["entry_time"].is_unique
    assert np.isclose(
        replay.candidate_returns.sum(),
        replay.candidate_ledger["net_return"].sum(),
        atol=1e-12,
    )


def test_development_gate_requires_frequency_and_noninferior_both_sides() -> None:
    result = evaluate_reentry_development(
        control=_summary(
            trades=100,
            net=0.05,
            long_net=0.03,
            short_net=0.02,
            long_trades=50,
            short_trades=50,
        ),
        candidate=_summary(
            trades=115,
            net=0.05,
            long_net=0.03,
            short_net=0.02,
            long_trades=60,
            short_trades=55,
        ),
        incremental=_summary(
            trades=15,
            net=0.0,
            long_net=0.0,
            short_net=0.0,
            long_trades=10,
            short_trades=5,
        ),
        fold_metrics=_passing_folds(),
    )

    assert result["required_candidate_trades"] == 115
    assert result["development_pass"]


def test_one_losing_incremental_side_fails_closed() -> None:
    control = _summary(
        trades=100,
        net=0.05,
        long_net=0.03,
        short_net=0.02,
        long_trades=50,
        short_trades=50,
    )
    candidate = _summary(
        trades=115,
        net=0.05,
        long_net=0.031,
        short_net=0.019,
        long_trades=60,
        short_trades=55,
    )
    incremental = _summary(
        trades=15,
        net=0.0,
        long_net=0.001,
        short_net=-0.001,
        long_trades=10,
        short_trades=5,
    )

    result = evaluate_reentry_development(
        control, candidate, incremental, _passing_folds()
    )

    assert not result["development_pass"]
    assert not result["incremental_short_nonnegative"]
