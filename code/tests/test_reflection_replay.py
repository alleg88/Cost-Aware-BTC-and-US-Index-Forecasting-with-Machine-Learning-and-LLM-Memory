from datetime import UTC, datetime, timedelta

from reflection_agent.contracts import Candidate, ExpectedEffect, MemorySnippet, PolicyEdit, ShadowState
from reflection_agent.replay import REGISTERED_VARIANTS, shadow_window_ids, shuffle_time_eligible_memories
from reflection_agent.store import AgentStore
from experiments.run_reflection_agent import _active_policy_rules, _active_policy_state


def _shadow():
    source = datetime(2025, 7, 13, 23, 59, 59, 999999, tzinfo=UTC)
    candidate = Candidate(
        candidate_id="c1", hypothesis="Use a bounded confidence gate in shadow.",
        edits=[PolicyEdit(edit_id="e1", action="set_confidence_threshold", value=0.75)],
        mechanism="The deterministic gate changes only eligible consensus rows.",
        expected_effect=ExpectedEffect(net_return="increase", turnover="decrease"),
        falsifiers=["delta_net_return_lte_0"], confidence=0.5,
    )
    return ShadowState(
        shadow_id="s1", candidate=candidate, source_window_id="2025-W28",
        source_cutoff_utc=source, eligible_after_utc=source + timedelta(microseconds=1),
    )


def test_registered_replay_matrix_covers_memory_and_news_controls():
    assert set(REGISTERED_VARIANTS) == {
        "reflection_no_memory", "reflection_real_memory", "reflection_shuffled_memory",
        "news_none", "news_aggregate",
    }
    assert {variant.news_mode for variant in REGISTERED_VARIANTS.values()} == {"none", "aggregate", "bounded_text"}


def test_shadow_windows_begin_after_source_and_cap_is_applied_by_caller():
    shadow = _shadow()
    assert shadow_window_ids(
        shadow, close_end_exclusive=datetime(2025, 7, 28, tzinfo=UTC)
    ) == ["2025-W29", "2025-W30"]


def test_shuffled_memory_is_deterministic_and_preserves_membership():
    memories = [
        MemorySnippet(
            memory_id=f"m{index}", memory_type="episodic",
            cutoff_utc=datetime(2025, 7, index + 1, tzinfo=UTC),
            lesson=f"Supported lesson number {index}.", evidence_status="supported", tags=["vol:high"],
        )
        for index in range(4)
    ]
    first = shuffle_time_eligible_memories(memories, window_id="2025-W40")
    second = shuffle_time_eligible_memories(memories, window_id="2025-W40")
    assert [item.memory_id for item in first] == [item.memory_id for item in second]
    assert {item.memory_id for item in first} == {item.memory_id for item in memories}


def test_active_policy_is_used_only_after_activation(tmp_path):
    from reflection_agent.contracts import PolicyRule, PolicyVersion

    store = AgentStore(tmp_path / "state.sqlite")
    activates = datetime(2025, 8, 1, tzinfo=UTC)
    policy = PolicyVersion(
        policy_id="p1", parent_policy_id="p0", source_candidate_id="c1",
        rule=PolicyRule(
            rule_id="r1", edits=[PolicyEdit(edit_id="tau", action="set_confidence_threshold", value=0.75)]
        ),
        activates_at_utc=activates, evaluation_id="ev1",
    )
    store.save_record("policies", "p1", "protocol", policy.model_dump(mode="json"))
    assert _active_policy_state(
        store, protocol_hash="protocol", cutoff_utc=activates - timedelta(microseconds=1)
    )[0] == "policy-unanimity-consensus-v1"
    assert _active_policy_state(
        store, protocol_hash="protocol", cutoff_utc=activates
    ) == ("p1", ["tau"])
    assert [rule.rule_id for rule in _active_policy_rules(
        store, protocol_hash="protocol", cutoff_utc=activates
    )] == ["frozen-unanimity-consensus", "r1"]
