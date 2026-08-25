import pandas as pd


def _economic_row(**overrides):
    row = {
        "width": 55,
        "candidate": 0,
        "tau": 0.60,
        "trades": 80,
        "pooled_net": 0.04,
        "pooled_sortino": 1.2,
        "pooled_sharpe": 0.9,
        "positive_folds": 6,
        "n_long": 40,
        "n_short": 40,
        "long_net": 0.02,
        "short_net": 0.02,
        "bull_sortino": 0.8,
        "sideways_sortino": 0.7,
        "bear_sortino": 0.6,
        "robust_score": 0.6,
    }
    row.update(overrides)
    return row


def test_monthly_development_folds_are_90d_and_end_before_july_2025():
    from experiments.run_tune_antibull_widths import monthly_development_folds

    folds = monthly_development_folds()

    assert len(folds) == 15
    assert folds[0].validation_start == pd.Timestamp("2024-04-01", tz="UTC")
    assert folds[-1].validation_end == pd.Timestamp("2025-06-30 23:45", tz="UTC")
    assert all(fold.train_end == fold.validation_start for fold in folds)
    assert all(
        fold.train_end - fold.train_start == pd.Timedelta("90D") for fold in folds
    )


def test_candidate_pool_is_seeded_and_starts_with_project_baseline():
    from experiments.run_tune_antibull_widths import candidate_pool

    first = candidate_pool(15)
    second = candidate_pool(15)

    assert first == second
    assert first[0] == {
        "iterations": 300,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3.0,
    }


def test_f1_selection_uses_weakest_regime_score():
    from experiments.run_tune_antibull_widths import select_f1_candidate

    grid = pd.DataFrame(
        {
            "candidate": [0, 1, 2],
            "robust_f1": [0.31, 0.35, 0.34],
            "overall_f1": [0.45, 0.36, 0.40],
        }
    )

    selected = select_f1_candidate(grid)
    assert int(selected["candidate"]) == 1


def test_economic_selection_is_joint_across_width_and_tau():
    from experiments.run_tune_antibull_widths import select_economic_candidate

    eligible = _economic_row(width=55, candidate=4, tau=0.65, robust_score=0.8)
    better_but_thin = _economic_row(
        width=75, candidate=7, tau=0.75, robust_score=9.0, trades=49
    )

    selected = select_economic_candidate(
        pd.DataFrame([eligible, better_but_thin]), n_folds=9
    )

    assert selected is not None
    assert int(selected["width"]) == 55
    assert select_economic_candidate(
        pd.DataFrame([better_but_thin]), n_folds=9
    ) is None


def test_selected_width_grid_comes_from_notebook_01_net_ranking():
    from experiments.run_tune_antibull_widths import WIDTHS

    assert WIDTHS == (55, 65, 75)



def test_outer_audits_use_only_earlier_development_folds():
    from experiments.run_tune_antibull_widths import outer_audit_schedule

    assert outer_audit_schedule() == [
        (tuple(range(12)), 12),
        (tuple(range(13)), 13),
        (tuple(range(14)), 14),
    ]


def test_prediction_cache_identity_includes_width_candidate_hash_and_fold():
    from experiments.run_tune_antibull_widths import prediction_cache_path

    path = prediction_cache_path(
        65, 3, {"iterations": 500, "depth": 6}, 7, "2024-11"
    )

    assert "w65" in path.name
    assert "candidate_03" in path.name
    assert "fold_07_2024-11" in path.name
    assert path.suffix == ".parquet"

def test_configured_model_zoo_scope_is_isolated_and_bounded():
    from experiments.run_tune_antibull_widths import configure_study

    study = configure_study(
        model="gru",
        model_zoo=True,
        n_trials=15,
        fold_limit=1,
        candidate_limit=2,
    )

    assert study.model == "gru"
    assert study.smoke
    assert study.paths.root.parts[-2:] == ("smoke", "gru")
    assert len(study.folds) == 1
    assert len(study.candidates) == 2
    assert study.candidates[0] == {}


def test_legacy_catboost_scope_preserves_existing_artifacts():
    from experiments.run_tune_antibull_widths import OUT_DIR, configure_study

    study = configure_study(
        model="catboost_balanced",
        model_zoo=False,
        n_trials=2,
    )

    assert not study.smoke
    assert study.paths.root == OUT_DIR
    assert study.candidates[0]["iterations"] == 300
    assert len(study.folds) == 15
