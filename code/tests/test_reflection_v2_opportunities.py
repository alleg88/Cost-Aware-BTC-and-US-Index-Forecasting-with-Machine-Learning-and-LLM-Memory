from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from reflection_agent.v2.opportunities import (
    assign_observation_episodes,
    build_development_opportunities,
    build_exact_opportunities,
    regime_tags,
)


CODE_ROOT = Path(__file__).parents[1]
DEVELOPMENT_CACHE = CODE_ROOT / "experiments" / "cache" / "union_v1_episode_reentry"


@pytest.fixture(scope="module")
def development() -> pd.DataFrame:
    return build_development_opportunities(DEVELOPMENT_CACHE)


@pytest.fixture(scope="module")
def exact_h1() -> pd.DataFrame:
    return build_exact_opportunities("h1")


@pytest.fixture(scope="module")
def exact_forward() -> pd.DataFrame:
    return build_exact_opportunities("forward")


def test_development_opportunity_universe_reconciles(development: pd.DataFrame) -> None:
    assert development["route"].value_counts().to_dict() == {
        "UNION_BASE": 948,
        "REENTRY": 283,
    }
    reentries = development.loc[development["route"].eq("REENTRY")]
    assert reentries.groupby("fold_id", sort=True).size().tolist() == [9, 94, 18, 99, 63]
    assert development["opportunity_id"].is_unique
    assert development["row_key"].is_unique
    assert development["decision_time"].lt(development["outcome_available_time"]).all()
    assert development["feature_available_time"].le(development["decision_time"]).all()
    assert development["source_role"].eq("OOF_TEST").all()
    assert development["source_artifact_hash"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_regime_mapping_is_fixed_and_missing_safe() -> None:
    assert regime_tags(
        pd.Series(
            {
                "vol_z": -0.5,
                "channel_slope_20": 2.0,
                "funding_z": float("nan"),
                "oi_z": 0.5,
            }
        )
    ) == {
        "vol_regime": "LOW",
        "trend_regime": "UP",
        "funding_regime": "MISSING",
        "oi_regime": "RISING",
    }
    with pytest.raises(ValueError, match="vol_z"):
        regime_tags(
            pd.Series(
                {
                    "vol_z": float("nan"),
                    "channel_slope_20": 0.0,
                    "funding_z": 0.0,
                    "oi_z": 0.0,
                }
            )
        )


def test_observation_episodes_never_mix_folds(development: pd.DataFrame) -> None:
    assigned = assign_observation_episodes(development)
    folds_per_episode = assigned.groupby("observation_episode_id")["fold_id"].nunique()
    assert folds_per_episode.eq(1).all()
    complete = assigned.drop_duplicates("observation_episode_id").loc[
        lambda frame: frame["episode_can_propose"]
    ]
    assert complete["episode_opportunity_count"].between(60, 90).all()
    assert complete["episode_reentry_count"].ge(10).all()
    assert complete["episode_reentry_long_count"].ge(3).all()
    assert complete["episode_reentry_short_count"].ge(3).all()
    assert assigned["outcome_available_time"].le(assigned["episode_cutoff_utc"]).all()


def test_partial_fold_remainder_is_ineligible() -> None:
    rows = []
    start = pd.Timestamp("2024-01-01", tz="UTC")
    for fold_id, count in ((0, 59), (1, 60), (2, 60)):
        for offset in range(count):
            is_reentry = offset < (10 if fold_id != 2 else 9)
            rows.append(
                {
                    "opportunity_id": f"f{fold_id}-{offset}",
                    "fold_id": fold_id,
                    "route": "REENTRY" if is_reentry else "UNION_BASE",
                    "side": "LONG" if offset % 2 == 0 else "SHORT",
                    "decision_time": start + pd.Timedelta(days=fold_id, minutes=offset),
                    "outcome_available_time": start
                    + pd.Timedelta(days=fold_id, minutes=offset + 1),
                }
            )
    assigned = assign_observation_episodes(pd.DataFrame(rows))
    summary = assigned.drop_duplicates("observation_episode_id").set_index("fold_id")
    assert not bool(summary.loc[0, "episode_can_propose"])
    assert bool(summary.loc[1, "episode_can_propose"])
    assert not bool(summary.loc[2, "episode_can_propose"])


def test_exact_counts_and_sides_stay_frozen(
    exact_h1: pd.DataFrame, exact_forward: pd.DataFrame
) -> None:
    assert exact_h1["route"].value_counts().to_dict() == {
        "UNION_BASE": 88,
        "REENTRY": 22,
    }
    assert exact_forward["route"].value_counts().to_dict() == {
        "UNION_BASE": 74,
        "REENTRY": 14,
    }
    forward_reentries = exact_forward.loc[exact_forward["route"].eq("REENTRY")]
    assert forward_reentries["side"].value_counts().to_dict() == {"SHORT": 9, "LONG": 5}
    assert exact_h1["stage"].eq("h1").all()
    assert exact_forward["stage"].eq("forward").all()
    assert exact_forward["outcome_available_time"].max() < pd.Timestamp(
        "2026-04-01", tz="UTC"
    )


def test_reentry_context_is_complete(
    development: pd.DataFrame, exact_h1: pd.DataFrame, exact_forward: pd.DataFrame
) -> None:
    for frame in (development, exact_h1, exact_forward):
        reentries = frame.loc[frame["route"].eq("REENTRY")]
        assert reentries["previous_exit_reason"].isin(
            {"TAKE_PROFIT", "STOP_LOSS", "TIMEOUT"}
        ).all()
        assert reentries["previous_exit_reason_available_time"].gt(
            reentries["decision_time"]
        ).all()
        assert reentries["previous_exit_reason_available_time"].lt(
            reentries["entry_time"]
        ).all()
        assert reentries["episode_bar_bucket"].isin({"SECOND", "THIRD_PLUS"}).all()
        assert reentries["member_pattern"].isin(
            {"LSTM_ONLY", "SVM_ONLY", "BOTH_AGREE"}
        ).all()
        assert reentries[["vol_regime", "trend_regime", "funding_regime", "oi_regime"]].notna().all().all()
