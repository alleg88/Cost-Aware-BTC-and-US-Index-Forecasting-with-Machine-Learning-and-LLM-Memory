import pandas as pd


def _row(**overrides):
    row = {
        "cooldown_bars": 4,
        "width": 75,
        "candidate": 6,
        "tau": 0.75,
        "trades": 80,
        "pooled_net": 0.04,
        "pooled_sortino": 1.2,
        "pooled_sharpe": 0.9,
        "positive_folds": 10,
        "n_long": 50,
        "n_short": 30,
        "long_net": 0.02,
        "short_net": 0.02,
        "bull_sortino": 0.8,
        "sideways_sortino": 0.7,
        "bear_sortino": 0.6,
        "robust_score": 0.6,
    }
    row.update(overrides)
    return row


def test_turnover_study_freezes_model_and_only_tests_requested_cooldowns():
    from experiments.run_turnover_control import (
        CANDIDATE,
        COOLDOWNS,
        TAU,
        WIDTH,
    )

    assert (WIDTH, CANDIDATE, TAU) == (75, 6, 0.75)
    assert COOLDOWNS == (4, 8, 16)


def test_turnover_selection_excludes_control_and_keeps_economic_guards():
    from experiments.run_turnover_control import select_cooldown

    control = _row(cooldown_bars=0, robust_score=9.0)
    eligible = _row(cooldown_bars=8, robust_score=0.8)
    thin = _row(cooldown_bars=16, robust_score=2.0, trades=49)

    selected = select_cooldown(
        pd.DataFrame([control, eligible, thin]), n_folds=15
    )

    assert selected is not None
    assert int(selected["cooldown_bars"]) == 8


def test_turnover_nested_audits_use_only_earlier_folds():
    from experiments.run_turnover_control import outer_audit_schedule

    assert outer_audit_schedule() == [
        (tuple(range(12)), 12),
        (tuple(range(13)), 13),
        (tuple(range(14)), 14),
    ]
