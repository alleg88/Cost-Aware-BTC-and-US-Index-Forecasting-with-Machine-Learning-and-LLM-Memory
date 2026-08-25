from pathlib import Path
from types import SimpleNamespace

import optuna
import pandas as pd


def _policy(**overrides):
    row = {
        "trades": 80,
        "n_long": 40,
        "n_short": 40,
        "positive_folds": 6,
        "pooled_net": 0.04,
        "pooled_sortino": 1.2,
        "pooled_sharpe": 0.9,
        "bull_sortino": 0.8,
        "sideways_sortino": 0.7,
        "bear_sortino": 0.6,
        "robust_score": 0.6,
    }
    row.update(overrides)
    return row


def test_protocol_uses_sortino_selected_widths_and_equal_policy_grid():
    from experiments.catboost_economic_optuna import GEOMETRIES, TAUS, WIDTHS

    assert WIDTHS == (55, 65, 75)
    assert GEOMETRIES == ((150, 75, 1), (150, 100, 1), (200, 100, 1))
    assert len(TAUS) == 11
    assert len(GEOMETRIES) * len(TAUS) == 33


def test_primary_folds_tune_on_2024_and_evaluate_2025_h1():
    from experiments.catboost_economic_optuna import (
        monthly_folds,
        primary_evaluation_fold_ids,
        primary_tuning_fold_ids,
    )

    folds = monthly_folds()
    tuning = primary_tuning_fold_ids()
    evaluation = primary_evaluation_fold_ids()

    assert tuning == tuple(range(9))
    assert evaluation == tuple(range(9, 15))
    assert folds[tuning[0]].validation_start == pd.Timestamp(
        "2024-04-01", tz="UTC"
    )
    assert folds[tuning[-1]].validation_start == pd.Timestamp(
        "2024-12-01", tz="UTC"
    )
    assert folds[evaluation[0]].validation_start == pd.Timestamp(
        "2025-01-01", tz="UTC"
    )
    assert folds[evaluation[-1]].validation_start == pd.Timestamp(
        "2025-06-01", tz="UTC"
    )


def test_rolling_schedule_uses_nine_earlier_months_per_outer_month():
    from experiments.catboost_economic_optuna import monthly_folds, rolling_schedule

    folds = monthly_folds()
    schedule = rolling_schedule()

    assert schedule == [
        (tuple(range(3, 12)), 12),
        (tuple(range(4, 13)), 13),
        (tuple(range(5, 14)), 14),
    ]
    assert [folds[outer].validation_start.strftime("%Y-%m") for _, outer in schedule] == [
        "2025-04",
        "2025-05",
        "2025-06",
    ]
    assert all(len(inner) == 9 and max(inner) < outer for inner, outer in schedule)


def test_robust_score_is_the_weakest_pooled_or_regime_metric():
    from experiments.catboost_economic_optuna import robust_score

    assert robust_score(
        pooled_sortino=1.1,
        pooled_sharpe=0.8,
        bull_sortino=0.7,
        sideways_sortino=0.9,
        bear_sortino=-0.2,
    ) == -0.2


def test_constraints_cover_sample_adequacy_not_profitability():
    from experiments.catboost_economic_optuna import constraint_values

    feasible = constraint_values(_policy(pooled_net=-0.2), n_folds=9)
    sparse = constraint_values(
        _policy(trades=49, n_long=14, n_short=10, positive_folds=5),
        n_folds=9,
    )

    assert all(value <= 0 for value in feasible)
    assert sparse == (1.0, 1.0, 5.0, 1.0)


def test_rank_policy_keeps_all_rows_and_prefers_feasible_robust_result():
    from experiments.catboost_economic_optuna import rank_policy

    grid = pd.DataFrame(
        [
            _policy(policy="sparse", trades=12, robust_score=9.0),
            _policy(policy="stable", robust_score=-0.5, pooled_sortino=-0.2),
            _policy(policy="weaker", robust_score=-0.8, pooled_sortino=0.5),
        ]
    )

    selected = rank_policy(grid, n_folds=9)

    assert selected["policy"] == "stable"
    assert len(grid) == 3


def test_prediction_cache_is_content_keyed_and_scope_independent(tmp_path: Path):
    from experiments.catboost_economic_optuna import prediction_cache_path

    params = {"iterations": 300, "depth": 6, "learning_rate": 0.1}
    first = prediction_cache_path(
        tmp_path, width=55, params=params, fold_id=0, month="2024-04"
    )
    second = prediction_cache_path(
        tmp_path, width=55, params=params, fold_id=0, month="2024-04"
    )
    changed = prediction_cache_path(
        tmp_path,
        width=55,
        params={**params, "depth": 7},
        fold_id=0,
        month="2024-04",
    )

    assert first == second
    assert first != changed
    assert "w55" in first.name and "2024-04" in first.name

def test_policy_choices_cover_every_tau_and_geometry_once():
    from experiments.catboost_economic_optuna import GEOMETRIES, TAUS, policy_choices

    choices = policy_choices()

    assert len(choices) == 33
    assert len(set(choices)) == 33
    assert {choice[0] for choice in choices} == set(TAUS)
    assert {choice[1] for choice in choices} == set(GEOMETRIES)


def test_constraints_adapter_reads_saved_trial_constraints():
    from experiments.catboost_economic_optuna import constraints_from_trial

    trial = SimpleNamespace(user_attrs={"constraints": (1.0, -2.0, 3.0, 0.0)})

    assert constraints_from_trial(trial) == (1.0, -2.0, 3.0, 0.0)


def test_completed_trial_count_ignores_failed_and_running_trials():
    from experiments.catboost_economic_optuna import completed_trial_count

    study = SimpleNamespace(
        trials=[
            SimpleNamespace(state=optuna.trial.TrialState.COMPLETE),
            SimpleNamespace(state=optuna.trial.TrialState.FAIL),
            SimpleNamespace(state=optuna.trial.TrialState.RUNNING),
        ]
    )

    assert completed_trial_count(study) == 1


def test_study_scopes_define_primary_plus_three_rolling_audits():
    from experiments.catboost_economic_optuna import study_scopes

    scopes = study_scopes()

    assert [scope.name for scope in scopes] == [
        "primary_2024",
        "rolling_pre_2025_04",
        "rolling_pre_2025_05",
        "rolling_pre_2025_06",
    ]
    assert scopes[0].outer_fold_id is None
    assert [scope.outer_fold_id for scope in scopes[1:]] == [12, 13, 14]
    assert all(len(scope.inner_fold_ids) == 9 for scope in scopes)
