from __future__ import annotations

import pandas as pd
import pytest

from reflection_agent.v3.opportunities import (
    OUTPUT_COLUMNS,
    assign_observation_episodes,
    build_development_opportunities,
    build_exact_opportunities,
    coverage_candidate_mask,
)


Q2_START = pd.Timestamp("2026-04-01", tz="UTC")


@pytest.fixture(scope="module")
def development() -> pd.DataFrame:
    return build_development_opportunities()


@pytest.fixture(scope="module")
def h1() -> pd.DataFrame:
    return build_exact_opportunities("h1")


@pytest.fixture(scope="module")
def forward() -> pd.DataFrame:
    return build_exact_opportunities("forward")


@pytest.mark.parametrize(
    ("tier", "total", "long_count", "short_count"),
    [
        ("HIGH_EXTRA", 524, 239, 285),
        ("MID_EXTRA", 834, 404, 430),
        ("LOW_EXTRA", 1150, 500, 650),
    ],
)
def test_development_candidate_supply_is_frozen(
    development: pd.DataFrame,
    tier: str,
    total: int,
    long_count: int,
    short_count: int,
) -> None:
    candidates = development.loc[
        development["route"].eq("COVERAGE_CANDIDATE")
        & development["confidence_tier"].eq(tier)
    ]
    assert len(candidates) == total
    assert int(candidates["side"].eq("LONG").sum()) == long_count
    assert int(candidates["side"].eq("SHORT").sum()) == short_count


def _route_side_counts(frame: pd.DataFrame, route: str) -> dict[str, int]:
    selected = frame.loc[frame["route"].eq(route), "side"]
    return {
        "LONG": int(selected.eq("LONG").sum()),
        "SHORT": int(selected.eq("SHORT").sum()),
    }


def test_union_and_exact_candidate_supplies_are_frozen(
    development: pd.DataFrame,
    h1: pd.DataFrame,
    forward: pd.DataFrame,
) -> None:
    assert int(development["route"].eq("UNION_BASE").sum()) == 948
    assert int(h1["route"].eq("UNION_BASE").sum()) == 88
    assert int(forward["route"].eq("UNION_BASE").sum()) == 74
    assert _route_side_counts(h1, "COVERAGE_CANDIDATE") == {
        "LONG": 462,
        "SHORT": 474,
    }
    assert _route_side_counts(forward, "COVERAGE_CANDIDATE") == {
        "LONG": 377,
        "SHORT": 551,
    }


@pytest.mark.parametrize(
    ("fixture_name", "start", "end"),
    [
        ("development", "2021-01-01", "2025-01-01"),
        ("h1", "2025-01-01", "2025-07-01"),
        ("forward", "2025-07-01", "2026-04-01"),
    ],
)
def test_opportunities_are_unique_causal_and_bounded(
    request: pytest.FixtureRequest,
    fixture_name: str,
    start: str,
    end: str,
) -> None:
    frame = request.getfixturevalue(fixture_name)
    lower = pd.Timestamp(start, tz="UTC")
    upper = pd.Timestamp(end, tz="UTC")
    assert tuple(frame.columns) == OUTPUT_COLUMNS
    assert frame["opportunity_id"].is_unique
    assert frame["row_key"].is_unique
    assert frame["decision_time"].ge(lower).all()
    assert frame["decision_time"].lt(upper).all()
    assert frame["feature_available_time"].le(frame["decision_time"]).all()
    assert frame["decision_time"].lt(frame["outcome_available_time"]).all()
    assert frame["outcome_available_time"].lt(Q2_START).all()
    assert frame["path_complete"].all()
    assert set(frame["side"]) == {"LONG", "SHORT"}
    assert set(frame["route"]) == {"UNION_BASE", "COVERAGE_CANDIDATE"}
    cost = frame["gross_return"] - frame["net_return"]
    assert cost.to_numpy() == pytest.approx(frame["round_trip_cost"].to_numpy())


def test_candidate_mask_is_union_flat_and_vetoes_opposition() -> None:
    frame = pd.DataFrame(
        {
            "union_signal": [0, 0, 0, 1],
            "pred_lstm": [2, 2, 0, 2],
            "pred_svm_linear": [1, 0, 1, 1],
            "p_short_lstm": [0.1, 0.1, 0.7, 0.1],
            "p_flat_lstm": [0.2, 0.2, 0.2, 0.2],
            "p_long_lstm": [0.7, 0.7, 0.1, 0.7],
            "path_complete": [True, True, True, True],
        }
    )
    assert coverage_candidate_mask(frame).tolist() == [True, False, True, False]


def test_candidate_episode_closure_counts_only_candidates(
    development: pd.DataFrame,
) -> None:
    assigned = assign_observation_episodes(development)
    assert assigned["observation_episode_id"].notna().all()
    assert assigned.groupby("observation_episode_id")["fold_id"].nunique().le(1).all()
    complete = assigned.loc[assigned["episode_can_propose"]]
    summary = complete.groupby("observation_episode_id", sort=False).first()
    assert summary["episode_candidate_count"].between(20, 30).all()
    assert summary["episode_long_candidate_count"].ge(6).all()
    assert summary["episode_short_candidate_count"].ge(6).all()
    counted = assigned.groupby("observation_episode_id")["route"].apply(
        lambda values: int(values.eq("COVERAGE_CANDIDATE").sum())
    )
    reported = assigned.groupby("observation_episode_id")[
        "episode_candidate_count"
    ].first()
    assert counted.to_dict() == reported.to_dict()
    assert set(assigned["episode_status"]).issubset(
        {"COMPLETE", "INSUFFICIENT_SIDE_SUPPORT"}
    )

