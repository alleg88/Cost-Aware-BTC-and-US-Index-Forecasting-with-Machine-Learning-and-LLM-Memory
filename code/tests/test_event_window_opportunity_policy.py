import pandas as pd

from evaluation.event_window_opportunity_policy import (
    causal_crossing_alerts,
    causal_level_rearm_alerts,
    collapse_episode_time,
    select_causal_threshold,
)


def _frame(scores: list[float]) -> pd.DataFrame:
    start = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "channel_episode_id": ["episode"] * len(scores),
            "decision_time": [start + pd.Timedelta(minutes=5 * i) for i in range(len(scores))],
            "score": scores,
        }
    )


def test_causal_alert_requires_a_fresh_crossing_after_forward_cooldown():
    frame = _frame(
        [
            0.40,
            0.60,
            0.70,
            0.40,
            0.60,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.70,
            0.40,
            0.60,
        ]
    )
    selected = causal_crossing_alerts(frame, threshold=0.50, cooldown_minutes=60)
    assert selected.loc[selected["alert"], "decision_time"].tolist() == [
        pd.Timestamp("2024-01-01 00:05:00", tz="UTC"),
        pd.Timestamp("2024-01-01 01:15:00", tz="UTC"),
    ]


def test_future_scores_cannot_change_an_already_emitted_alert():
    original = causal_crossing_alerts(
        _frame([0.40, 0.60, 0.70]), threshold=0.50, cooldown_minutes=60
    )
    changed = causal_crossing_alerts(
        _frame([0.40, 0.60, 0.01]), threshold=0.50, cooldown_minutes=60
    )
    past = pd.Timestamp("2024-01-01 00:05:00", tz="UTC")
    assert bool(original.loc[original["decision_time"].eq(past), "alert"].iloc[0])
    assert bool(changed.loc[changed["decision_time"].eq(past), "alert"].iloc[0])


def test_level_rearm_fires_after_cooldown_without_a_new_crossing():
    frame = _frame([0.80] * 13)
    selected = causal_level_rearm_alerts(
        frame,
        threshold=0.50,
        cooldown_minutes=60,
    )
    assert selected.loc[selected["alert"], "decision_time"].tolist() == [
        pd.Timestamp("2024-01-01 00:00:00", tz="UTC"),
        pd.Timestamp("2024-01-01 01:00:00", tz="UTC"),
    ]


def test_crossing_does_not_rearm_while_the_score_stays_above():
    selected = causal_crossing_alerts(
        _frame([0.80] * 13),
        threshold=0.50,
        cooldown_minutes=60,
    )
    assert int(selected["alert"].sum()) == 1


def test_duplicate_episode_time_rows_keep_one_maximum_score():
    frame = _frame([0.40, 0.60])
    duplicate = frame.iloc[[1]].copy()
    duplicate["score"] = 0.90
    collapsed = collapse_episode_time(pd.concat([frame, duplicate], ignore_index=True))
    assert len(collapsed) == 2
    assert collapsed.loc[collapsed.decision_time.eq(frame.iloc[1].decision_time), "score"].iloc[0] == 0.90


def test_threshold_is_selected_only_from_past_calibration_at_target_rate():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    calibration = pd.DataFrame(
        {
            "channel_episode_id": ["e0", "e0", "e1", "e1"],
            "decision_time": [
                start,
                start + pd.Timedelta(minutes=5),
                start + pd.Timedelta(days=1),
                start + pd.Timedelta(days=1, minutes=5),
            ],
            "score": [0.20, 0.80, 0.20, 0.90],
        }
    )
    selected = select_causal_threshold(
        calibration,
        target_activations_per_day=1.0,
        cooldown_minutes=60,
    )
    assert selected.actual_activations_per_day == 1.0
    assert selected.calibration_rows == 4
    assert selected.threshold >= 0.20


def test_threshold_selection_uses_the_requested_level_rearm_policy():
    calibration = _frame([0.80] * 13)
    crossing = select_causal_threshold(
        calibration,
        target_activations_per_day=2.0,
        cooldown_minutes=60,
    )
    rearm = select_causal_threshold(
        calibration,
        target_activations_per_day=2.0,
        cooldown_minutes=60,
        alert_policy="level_rearm",
    )
    replay = causal_level_rearm_alerts(
        calibration,
        threshold=rearm.threshold,
        cooldown_minutes=60,
    )
    assert crossing.actual_activations_per_day == 1.0
    assert rearm.actual_activations_per_day == 2.0
    assert int(replay["alert"].sum()) == rearm.activations
