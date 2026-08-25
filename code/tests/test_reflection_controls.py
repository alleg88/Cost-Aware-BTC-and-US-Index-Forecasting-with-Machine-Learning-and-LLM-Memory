from reflection_agent.controls import deterministic_router_rules, materialize_candidate_rules
from reflection_agent.contracts import Candidate, ExpectedEffect, PolicyEdit


def _candidate(edit):
    return Candidate(
        candidate_id="c1",
        hypothesis="Use one bounded edit against the frozen consensus policy.",
        edits=[edit],
        mechanism="The deterministic compiler applies the edit before evaluation.",
        expected_effect=ExpectedEffect(net_return="increase", turnover="increase"),
        falsifiers=["delta_net_return_lte_0"],
        confidence=0.5,
    )


def test_deterministic_router_uses_registered_two_edit_budget():
    base, router = deterministic_router_rules()
    assert base.rule_id == "frozen-unanimity-consensus"
    assert len(router.edits) == 2
    assert router.conditions.all[0].field == "model_disagreement"
    assert {edit.action for edit in router.edits} == {"select_frozen_expert", "require_minimum_agreement"}


def test_candidate_materialization_starts_from_consensus_and_resolves_meta_edits():
    ordinary = materialize_candidate_rules(_candidate(
        PolicyEdit(edit_id="e1", action="set_confidence_threshold", value=0.75)
    ))
    assert ordinary[0].rule_id == "frozen-unanimity-consensus"
    assert ordinary[1].rule_id == "c1"

    removed = materialize_candidate_rules(_candidate(
        PolicyEdit(edit_id="e2", action="remove_active_edit", target="consensus-agreement")
    ))
    assert removed == ()
