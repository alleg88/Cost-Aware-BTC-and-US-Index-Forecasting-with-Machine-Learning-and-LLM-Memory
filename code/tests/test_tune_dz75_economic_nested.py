import pandas as pd


def _candidate(**overrides):
    row = {
        "candidate": 0,
        "tau": 0.60,
        "trades": 80,
        "pooled_net": 0.04,
        "pooled_sortino": 1.2,
        "positive_folds": 6,
        "n_long": 40,
        "n_short": 40,
        "long_net": 0.02,
        "short_net": 0.02,
        "bull_sortino": 0.8,
        "sideways_sortino": 0.6,
        "bear_sortino": 0.7,
        "robust_score": 0.6,
    }
    row.update(overrides)
    return row


def test_monthly_nested_folds_are_past_only_and_inside_2024():
    from experiments.run_tune_dz75_economic_nested import monthly_study_folds

    folds = monthly_study_folds()

    assert len(folds) == 9
    assert folds[0].validation_start == pd.Timestamp("2024-04-01", tz="UTC")
    assert folds[-1].validation_end == pd.Timestamp("2024-12-31 23:45", tz="UTC")
    for fold in folds:
        assert fold.train_end == fold.validation_start
        assert fold.train_end - fold.train_start == pd.Timedelta("90D")
        assert fold.validation_start > fold.train_start
        assert fold.validation_end < pd.Timestamp("2025-01-01", tz="UTC")


def test_outer_audits_select_only_from_earlier_folds():
    from experiments.run_tune_dz75_economic_nested import outer_audit_schedule

    schedule = outer_audit_schedule()

    assert schedule == [((0, 1, 2, 3, 4, 5), 6),
                        ((0, 1, 2, 3, 4, 5, 6), 7),
                        ((0, 1, 2, 3, 4, 5, 6, 7), 8)]
    assert all(max(inner) < outer for inner, outer in schedule)


def test_candidate_selection_enforces_economic_and_stability_guardrails():
    from experiments.run_tune_dz75_economic_nested import select_candidate

    grid = pd.DataFrame([
        _candidate(candidate=0, robust_score=9.0, trades=49),
        _candidate(candidate=1, robust_score=8.0, pooled_net=-0.01),
        _candidate(candidate=2, robust_score=7.0, short_net=-0.001),
        _candidate(candidate=3, robust_score=6.0, bear_sortino=-0.01),
        _candidate(candidate=4, robust_score=0.5),
        _candidate(candidate=5, robust_score=0.9),
    ])

    selected = select_candidate(grid, n_folds=9)
    assert selected is not None
    assert int(selected["candidate"]) == 5


def test_candidate_selection_can_choose_no_trade():
    from experiments.run_tune_dz75_economic_nested import select_candidate

    rejected = pd.DataFrame([_candidate(trades=49), _candidate(pooled_sortino=0.0)])

    assert select_candidate(rejected, n_folds=9) is None
