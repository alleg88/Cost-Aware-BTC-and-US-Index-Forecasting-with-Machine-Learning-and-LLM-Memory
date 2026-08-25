from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from reflection_agent.index_v2.policy import (
    ScorePolicy,
    apply_score_policy,
    calibrate_score_policy,
    seeded_hash_scores,
    select_h1_policy,
    uncertainty_scores,
)


def _probability_frame() -> pd.DataFrame:
    rows: dict[str, list[float]] = {"opportunity_id": ["A", "B", "C"]}
    margins = (0.8, 0.2, 0.0)
    for model_index in range(9):
        rows[f"m{model_index:02d}_p_short"] = [
            (1.0 - margin) / 2.0 for margin in margins
        ]
        rows[f"m{model_index:02d}_p_flat"] = [0.0, 0.0, 0.0]
        rows[f"m{model_index:02d}_p_long"] = [
            (1.0 + margin) / 2.0 for margin in margins
        ]
    return pd.DataFrame(rows)


def test_calibration_hits_literal_h1_target_without_outcomes():
    scores = [900, 800, 800, 700, 600, 500, 400, 300, 200, 100]
    ids = [f"H1-{index}" for index in range(10)]

    policy = calibrate_score_policy(scores, ids, target_rate=0.20, seed=42)
    actions = apply_score_policy(scores, ids, policy)

    assert actions.dtype == np.bool_
    assert int(actions.sum()) == 2
    assert policy.threshold == 800
    assert policy.target_count == 2
    assert policy.calibration_size == 10


def test_tie_hashing_is_deterministic_and_preserves_input_order():
    scores = [500] * 20
    ids = [f"ID-{index:02d}" for index in range(20)]

    first = calibrate_score_policy(scores, ids, target_rate=0.25, seed=42)
    second = calibrate_score_policy(scores, list(reversed(ids)), target_rate=0.25, seed=42)
    first_actions = apply_score_policy(scores, ids, first)
    second_actions = apply_score_policy(
        list(reversed(scores)), list(reversed(ids)), second
    )[::-1]

    assert first == second
    assert first_actions.tolist() == second_actions.tolist()
    assert int(first_actions.sum()) == 5


@pytest.mark.parametrize(
    ("scores", "ids", "rate"),
    [
        ([], [], 0.20),
        ([1], [], 0.20),
        ([1, 2], ["A", "A"], 0.20),
        ([1.0], ["A"], 0.20),
        ([True], ["A"], 0.20),
        ([-1], ["A"], 0.20),
        ([1001], ["A"], 0.20),
        ([1], ["A"], 0.0),
        ([1], ["A"], 1.0),
    ],
)
def test_calibration_rejects_invalid_scores_ids_or_rate(scores, ids, rate):
    with pytest.raises(ValueError):
        calibrate_score_policy(scores, ids, target_rate=rate, seed=42)


def test_apply_rejects_policy_or_input_drift():
    policy = calibrate_score_policy([900, 100], ["A", "B"], 0.5, 42)

    with pytest.raises(ValueError):
        apply_score_policy([900], ["A", "B"], policy)
    with pytest.raises(ValueError):
        apply_score_policy([900, 100], ["A", "A"], policy)
    with pytest.raises(ValueError):
        apply_score_policy([900, 100], ["A", "B"], replace(policy, threshold=1001))


def test_uncertainty_scores_use_only_current_probability_margin():
    frame = _probability_frame()

    scores = uncertainty_scores(frame)

    assert scores.tolist() == [200, 800, 1000]
    assert scores.dtype.kind in "iu"


def test_uncertainty_scores_reject_missing_nonfinite_or_out_of_range_probabilities():
    missing = _probability_frame().drop(columns="m08_p_long")
    with pytest.raises(ValueError, match="miss"):
        uncertainty_scores(missing)

    nonfinite = _probability_frame()
    nonfinite.loc[0, "m00_p_long"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        uncertainty_scores(nonfinite)

    invalid = _probability_frame()
    invalid.loc[0, "m00_p_long"] = 1.1
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        uncertainty_scores(invalid)


def test_seeded_hash_scores_are_reproducible_bounded_and_identity_only():
    ids = ["A", "B", "C"]

    first = seeded_hash_scores(ids, seed=42)
    second = seeded_hash_scores(ids, seed=42)
    changed_seed = seeded_hash_scores(ids, seed=43)

    assert first.tolist() == second.tolist()
    assert not np.array_equal(first, changed_seed)
    assert ((first >= 0) & (first <= 1000)).all()
    assert first.dtype.kind in "iu"


def test_h1_selection_uses_sortino_then_net_drawdown_and_lower_rate():
    candidates = pd.DataFrame(
        [
            {
                "target_rate": 0.15,
                "daily_sortino": 0.20,
                "net_return": 0.01,
                "max_drawdown": -0.02,
            },
            {
                "target_rate": 0.20,
                "daily_sortino": 0.30,
                "net_return": 0.00,
                "max_drawdown": -0.01,
            },
            {
                "target_rate": 0.25,
                "daily_sortino": 0.30,
                "net_return": -0.01,
                "max_drawdown": -0.01,
            },
        ]
    )

    selected = select_h1_policy(candidates)

    assert selected["target_rate"] == 0.20


def test_h1_selection_uses_drawdown_then_lower_rate_after_metric_ties():
    candidates = pd.DataFrame(
        [
            {
                "target_rate": 0.15,
                "daily_sortino": -0.10,
                "net_return": -0.01,
                "max_drawdown": -0.03,
            },
            {
                "target_rate": 0.20,
                "daily_sortino": -0.10,
                "net_return": -0.01,
                "max_drawdown": -0.02,
            },
            {
                "target_rate": 0.25,
                "daily_sortino": -0.10,
                "net_return": -0.01,
                "max_drawdown": -0.02,
            },
        ]
    )

    assert select_h1_policy(candidates)["target_rate"] == 0.20


@pytest.mark.parametrize(
    "mutation",
    [
        pd.DataFrame(),
        pd.DataFrame([{"target_rate": 0.20}]),
        pd.DataFrame(
            [
                {
                    "target_rate": 0.20,
                    "daily_sortino": np.nan,
                    "net_return": 0.0,
                    "max_drawdown": -0.1,
                }
            ]
        ),
    ],
)
def test_h1_selection_rejects_empty_incomplete_or_nonfinite_candidates(mutation):
    with pytest.raises(ValueError):
        select_h1_policy(mutation)


def test_score_policy_rejects_invalid_manual_construction():
    with pytest.raises(ValueError):
        ScorePolicy(
            target_rate=0.20,
            threshold=-1,
            tie_hash_cutoff=0.5,
            seed=42,
            calibration_size=10,
            target_count=2,
        )
