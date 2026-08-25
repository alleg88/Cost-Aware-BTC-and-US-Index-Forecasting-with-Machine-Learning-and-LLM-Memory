from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from pydantic import ValidationError

from reflection_agent.v2.contracts import AllowRule
from reflection_agent.v2.policy import ActiveAllowRule, apply_policy, compile_allow_mask


T0 = datetime(2024, 1, 1, tzinfo=UTC)


def opportunities() -> pd.DataFrame:
    rows = []
    for index in range(4):
        decision = T0 + timedelta(hours=index)
        rows.append(
            {
                "opportunity_id": f"base-{index}",
                "route": "UNION_BASE",
                "side": "LONG" if index % 2 else "SHORT",
                "vol_regime": "HIGH",
                "decision_time": decision,
                "entry_time": decision + timedelta(minutes=15),
                "gross_return": 0.002,
                "net_return": 0.001,
            }
        )
        rows.append(
            {
                "opportunity_id": f"reentry-{index}",
                "route": "REENTRY",
                "side": "SHORT" if index != 2 else "LONG",
                "vol_regime": "HIGH" if index != 3 else "NORMAL",
                "decision_time": decision + timedelta(minutes=30),
                "entry_time": decision + timedelta(minutes=45),
                "gross_return": 0.003,
                "net_return": 0.002,
            }
        )
    return pd.DataFrame(rows)


def short_high_rule() -> AllowRule:
    return AllowRule.model_validate(
        {
            "action": "ALLOW_REENTRY",
            "predicates": [
                {"field": "side", "operator": "EQ", "value": "SHORT"},
                {"field": "vol_regime", "operator": "EQ", "value": "HIGH"},
            ],
        }
    )


def test_rule_can_only_add_matching_reentries_and_never_touch_base() -> None:
    frame = opportunities()
    active = ActiveAllowRule(
        rule_id="rule-1",
        source_candidate_id="candidate-1",
        source_fold_id=0,
        activates_at_utc=T0 - timedelta(minutes=1),
        deactivates_at_utc=None,
        rule=short_high_rule(),
    )
    result = apply_policy(frame, [active])
    assert result.loc[result["route"].eq("UNION_BASE"), "selected"].all()
    selected = result.loc[result["route"].eq("REENTRY") & result["selected"]]
    assert set(selected["opportunity_id"]) == {"reentry-0", "reentry-1"}
    assert selected["side"].eq("SHORT").all()
    assert selected["vol_regime"].eq("HIGH").all()
    assert set(result.loc[result["route"].eq("UNION_BASE"), "opportunity_id"]) == set(
        frame.loc[frame["route"].eq("UNION_BASE"), "opportunity_id"]
    )


def test_activation_is_strictly_before_each_selected_opportunity() -> None:
    frame = opportunities()
    first = frame.loc[frame["opportunity_id"].eq("reentry-0"), "decision_time"].iloc[0]
    active = ActiveAllowRule(
        rule_id="rule-1",
        source_candidate_id="candidate-1",
        source_fold_id=0,
        activates_at_utc=first,
        deactivates_at_utc=None,
        rule=short_high_rule(),
    )
    result = apply_policy(frame, [active]).set_index("opportunity_id")
    assert not bool(result.loc["reentry-0", "selected"])
    assert bool(result.loc["reentry-1", "selected"])


def test_static_add_all_is_explicit_and_does_not_need_a_rule() -> None:
    result = apply_policy(opportunities(), [], static_add_all=True)
    assert result["selected"].all()
    assert result.loc[result["route"].eq("REENTRY"), "selected_rule_ids"].eq(
        "STATIC_ADD_ALL"
    ).all()


def test_unknown_field_or_non_equality_tree_is_impossible() -> None:
    with pytest.raises(ValidationError):
        AllowRule.model_validate(
            {
                "action": "ALLOW_REENTRY",
                "predicates": [
                    {"field": "future_return", "operator": "EQ", "value": "POSITIVE"}
                ],
            }
        )
    with pytest.raises(ValidationError):
        AllowRule.model_validate(
            {
                "action": "ALLOW_REENTRY",
                "predicates": [
                    {"field": "side", "operator": "OR", "value": "SHORT"}
                ],
            }
        )


def test_compiler_fails_closed_if_registered_column_is_missing() -> None:
    with pytest.raises(ValueError, match="missing condition column"):
        compile_allow_mask(opportunities().drop(columns="vol_regime"), short_high_rule())


def test_policy_history_may_exceed_three_versions_but_not_three_concurrent_rules() -> None:
    frame = opportunities()
    history = []
    for index in range(4):
        start = T0 - timedelta(hours=4 - index)
        history.append(
            ActiveAllowRule(
                rule_id=f"rule-{index}",
                source_candidate_id=f"candidate-{index}",
                source_fold_id=0,
                activates_at_utc=start,
                deactivates_at_utc=start + timedelta(minutes=30),
                rule=short_high_rule(),
            )
        )
    assert len(apply_policy(frame, history)) == len(frame)

    overlapping = [
        rule.model_copy(update={"deactivates_at_utc": None}) for rule in history
    ]
    with pytest.raises(ValueError, match="three allow rules can be active"):
        apply_policy(frame, overlapping)
