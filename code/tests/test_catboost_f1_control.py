import pandas as pd


def test_select_f1_candidate_prefers_robust_then_overall_then_lower_id():
    from experiments.catboost_f1_control import select_f1_candidate

    grid = pd.DataFrame(
        [
            {"candidate": 4, "robust_f1": 0.40, "overall_f1": 0.44},
            {"candidate": 2, "robust_f1": 0.41, "overall_f1": 0.42},
            {"candidate": 1, "robust_f1": 0.41, "overall_f1": 0.42},
        ]
    )

    assert int(select_f1_candidate(grid)["candidate"]) == 1


def test_combined_table_keeps_both_objectives_for_every_width_and_mode():
    from experiments.catboost_f1_control import combine_objective_tables

    rows = [
        {"width": width, "mode": mode, "trades": 10}
        for width in (55, 65, 75)
        for mode in ("frozen_2025_h1", "frozen_apr_jun", "rolling_apr_jun")
    ]
    combined = combine_objective_tables(pd.DataFrame(rows), pd.DataFrame(rows))

    assert len(combined) == 18
    assert set(combined["objective"]) == {"Economic-tuned", "F1-tuned"}
    assert not combined.duplicated(["objective", "width", "mode"]).any()


def test_f1_control_uses_identical_scopes_and_33_policy_choices():
    from experiments.catboost_economic_optuna import policy_choices, study_scopes
    from experiments.catboost_f1_control import control_scopes

    assert control_scopes() == study_scopes()
    assert len(policy_choices()) == 33