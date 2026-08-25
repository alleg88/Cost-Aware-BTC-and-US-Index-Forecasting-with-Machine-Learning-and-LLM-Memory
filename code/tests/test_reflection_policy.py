import numpy as np
import pandas as pd
import pytest

from reflection_agent.contracts import ConditionPredicate, ConditionTree, PolicyEdit, PolicyRule
from reflection_agent.policy import compile_policy, condition_mask, resolve_edits, validate_edit
from reflection_agent.search import candidate_rule
from reflection_agent.contracts import Candidate, ExpectedEffect


def _panel() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    index = pd.date_range("2025-07-01", periods=3, freq="15min", tz="UTC")
    models = (
        "logreg", "decision_tree", "random_forest", "svm_linear", "xgboost_balanced",
        "catboost_balanced", "mlp", "lstm", "gru",
    )
    panel = {}
    for model in models:
        values = np.array([[0.2, 0.2, 0.6], [0.6, 0.2, 0.2], [0.2, 0.6, 0.2]])
        if model == "lstm":
            values = np.array([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8], [0.1, 0.1, 0.8]])
        panel[model] = pd.DataFrame(values, index=index, columns=["p_short", "p_flat", "p_long"])
    context = pd.DataFrame(
        {
            "vol_regime": ["high", "normal", "high"],
            "trend_regime": ["up", "down", "flat"],
            "model_disagreement": [0.8, 0.2, 0.9],
            "ensemble_confidence": [0.6, 0.6, 0.6],
            "news_impact": [2.0, 0.0, 1.0],
            "news_dispersion": [0.5, 0.1, 0.2],
            "data_quality_state": ["ok", "ok", "ok"],
            "hour_block": ["asia", "europe", "us"],
            "day_of_week": ["Tue", "Tue", "Tue"],
        },
        index=index,
    )
    return panel, context


def test_condition_compiler_has_no_eval_and_uses_two_predicates():
    _, context = _panel()
    tree = ConditionTree(all=[
        ConditionPredicate(field="vol_regime", operator="eq", value="high"),
        ConditionPredicate(field="news_impact", operator="gte", value=1.5),
    ])
    assert condition_mask(context, tree).tolist() == [True, False, False]


def test_select_frozen_lstm_only_applies_inside_condition():
    panel, context = _panel()
    rule = PolicyRule(
        rule_id="r1",
        conditions=ConditionTree(all=[ConditionPredicate(field="vol_regime", operator="eq", value="high")]),
        edits=[PolicyEdit(edit_id="e1", action="select_frozen_expert", target="lstm")],
    )
    result = compile_policy(panel, context, [rule])
    assert result.predictions.tolist() == [0, 0, 2]
    assert np.allclose(result.probabilities.iloc[0], panel["lstm"].iloc[0])
    assert result.weights.loc[result.weights.index[0], "lstm"] == 1.0
    assert result.weights.loc[result.weights.index[0]].drop("lstm").eq(0.0).all()


def test_confidence_and_agreement_gates_are_bounded_and_deterministic():
    panel, context = _panel()
    rule = PolicyRule(rule_id="r1", edits=[
        PolicyEdit(edit_id="tau", action="set_confidence_threshold", value=0.75),
        PolicyEdit(edit_id="agree", action="require_minimum_agreement", value=1.0),
    ])
    result = compile_policy(panel, context, [rule])
    assert result.predictions.tolist() == [1, 1, 1]
    assert not result.eligible.any()


def test_semantic_grid_rejects_arbitrary_values():
    with pytest.raises(ValueError, match="outside fixed grid"):
        validate_edit(PolicyEdit(edit_id="e1", action="set_confidence_threshold", value=0.73))


def test_meta_edits_remove_or_reduce_only_known_active_edits():
    active = [PolicyEdit(edit_id="tau", action="set_confidence_threshold", value=0.80)]
    reduced = resolve_edits(active, [PolicyEdit(edit_id="r", action="reduce_active_edit", target="tau")])
    assert reduced[0].value == 0.75
    assert resolve_edits(active, [PolicyEdit(edit_id="x", action="remove_active_edit", target="tau")]) == ()
    with pytest.raises(ValueError, match="unknown active edit"):
        resolve_edits(active, [PolicyEdit(edit_id="x", action="remove_active_edit", target="missing")])


def test_meta_edits_cannot_be_conditionally_applied():
    candidate = Candidate(
        candidate_id="c1",
        hypothesis="Remove the active gate only in the supplied high volatility regime.",
        conditions=ConditionTree(all=[ConditionPredicate(field="vol_regime", operator="eq", value="high")]),
        edits=[PolicyEdit(edit_id="e1", action="remove_active_edit", target="consensus-agreement")],
        mechanism="This would otherwise make policy state depend on row-level conditions.",
        expected_effect=ExpectedEffect(net_return="increase", turnover="increase"),
        falsifiers=["delta_net_return_lte_0"],
        confidence=0.5,
    )
    with pytest.raises(ValueError, match="unconditional"):
        candidate_rule(candidate)
