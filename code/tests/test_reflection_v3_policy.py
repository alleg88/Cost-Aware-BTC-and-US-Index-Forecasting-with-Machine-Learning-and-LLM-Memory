from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from reflection_agent.v3.contracts import AllowRule, Predicate
from reflection_agent.v3.policy import ActiveAllowRule, apply_policy, compile_allow_mask


UTC = timezone.utc
BASE = datetime(2024, 1, 1, tzinfo=UTC)


def _rule(side: str, **conditions: str) -> AllowRule:
    predicates = [Predicate(field="side", operator="EQ", value=side)]
    predicates.extend(
        Predicate(field=field, operator="EQ", value=value)
        for field, value in conditions.items()
    )
    return AllowRule(action="ALLOW_CANDIDATE", predicates=predicates)


def _active(rule_id: str, rule: AllowRule) -> ActiveAllowRule:
    return ActiveAllowRule(
        rule_id=rule_id,
        source_candidate_id=f"candidate_{rule_id}",
        source_stage="development",
        source_fold_id=0,
        activates_at_utc=BASE,
        rule=rule,
    )


def _opportunities() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "opportunity_id": "union_a",
                "route": "UNION_BASE",
                "side": "LONG",
                "decision_time": BASE + timedelta(minutes=15),
                "entry_time": BASE + timedelta(minutes=30),
                "outcome_available_time": BASE + timedelta(minutes=35),
                "confidence_tier": pd.NA,
                "signal_run_bucket": pd.NA,
                "vol_regime": "NORMAL",
                "trend_regime": "UP",
                "funding_regime": "NEUTRAL",
                "oi_regime": "RISING",
            },
            {
                "opportunity_id": "short_high",
                "route": "COVERAGE_CANDIDATE",
                "side": "SHORT",
                "decision_time": BASE + timedelta(minutes=45),
                "entry_time": BASE + timedelta(minutes=60),
                "outcome_available_time": BASE + timedelta(minutes=70),
                "confidence_tier": "HIGH_EXTRA",
                "signal_run_bucket": "FIRST",
                "vol_regime": "NORMAL",
                "trend_regime": "DOWN",
                "funding_regime": "NEGATIVE",
                "oi_regime": "FALLING",
            },
            {
                "opportunity_id": "long_mid",
                "route": "COVERAGE_CANDIDATE",
                "side": "LONG",
                "decision_time": BASE + timedelta(minutes=75),
                "entry_time": BASE + timedelta(minutes=90),
                "outcome_available_time": BASE + timedelta(minutes=100),
                "confidence_tier": "MID_EXTRA",
                "signal_run_bucket": "SECOND",
                "vol_regime": "HIGH",
                "trend_regime": "UP",
                "funding_regime": "POSITIVE",
                "oi_regime": "RISING",
            },
        ]
    )


def test_compiled_actions_preserve_union_and_candidate_side() -> None:
    short_high = _active(
        "short_high_rule", _rule("SHORT", confidence_tier="HIGH_EXTRA")
    )
    result = apply_policy(_opportunities(), [short_high])
    assert result.query("route == 'UNION_BASE'")["selected"].all()
    admitted = result.query("route == 'COVERAGE_CANDIDATE' and selected")
    assert admitted["side"].eq("SHORT").all()
    assert admitted["action"].eq("OPEN_SHORT").all()
    skipped = result.query("route == 'COVERAGE_CANDIDATE' and not selected")
    assert skipped["action"].eq("SKIP").all()
    assert compile_allow_mask(_opportunities(), short_high.rule).sum() == 1


def test_static_controls_are_frozen_and_direction_preserving() -> None:
    high = apply_policy(_opportunities(), [], static_variant="static_high_extra")
    assert high.set_index("opportunity_id")["selected"].to_dict() == {
        "union_a": True,
        "short_high": True,
        "long_mid": False,
    }
    all_extra = apply_policy(_opportunities(), [], static_variant="static_all_extra")
    assert all_extra["selected"].all()
    union = apply_policy(_opportunities(), [], static_variant="union_baseline")
    assert union["selected"].tolist() == [True, False, False]
    assert set(all_extra.loc[all_extra["side"].eq("LONG"), "action"]) == {
        "OPEN_LONG"
    }
    assert set(all_extra.loc[all_extra["side"].eq("SHORT"), "action"]) == {
        "OPEN_SHORT"
    }


def test_rule_capacity_is_three_per_side() -> None:
    rules = [
        _active(f"long_{index}", _rule("LONG"))
        for index in range(4)
    ]
    with pytest.raises(ValueError, match="three active LONG rules"):
        apply_policy(_opportunities(), rules)


def test_scheduler_never_overlaps_union_or_an_admitted_candidate() -> None:
    frame = _opportunities()
    overlapping = frame.iloc[[1]].copy()
    overlapping["opportunity_id"] = "overlap_union"
    overlapping["side"] = "LONG"
    overlapping["decision_time"] = BASE + timedelta(minutes=16)
    overlapping["entry_time"] = BASE + timedelta(minutes=31)
    overlapping["outcome_available_time"] = BASE + timedelta(minutes=34)
    overlapping["confidence_tier"] = "HIGH_EXTRA"
    frame = pd.concat([frame, overlapping], ignore_index=True)
    result = apply_policy(frame, [], static_variant="static_all_extra").set_index(
        "opportunity_id"
    )
    assert result.at["union_a", "selected"]
    assert not result.at["overlap_union", "selected"]
    assert result.at["overlap_union", "skip_reason"] == "OVERLAP_UNION"


def test_rule_activation_is_strictly_before_decision() -> None:
    rule = _active("late", _rule("SHORT"))
    frame = _opportunities()
    frame.loc[frame["opportunity_id"].eq("short_high"), "decision_time"] = BASE
    result = apply_policy(frame, [rule]).set_index("opportunity_id")
    assert not result.at["short_high", "selected"]

