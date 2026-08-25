import pandas as pd


def _economic_row(**overrides):
    row = {
        "long_tau": 0.1,
        "short_tau": 0.1,
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


def test_side_study_freezes_the_66_trade_geometry():
    from experiments.run_side_path_selection import GEOMETRY

    assert GEOMETRY == (200, 100, 1)


def test_side_training_mask_is_causal_and_side_specific():
    from experiments.run_side_path_selection import side_training_mask

    candidates = pd.DataFrame(
        {
            "outcome_close_time": pd.to_datetime(
                ["2025-03-31 23:59Z", "2025-03-31 23:59Z", "2025-04-01 00:00Z"],
                utc=True,
            ),
            "side": [1, -1, 1],
        }
    )

    mask = side_training_mask(
        candidates, pd.Timestamp("2025-04-01", tz="UTC"), side=1
    )

    assert mask.tolist() == [True, False, False]


def test_side_filter_uses_independent_long_and_short_thresholds():
    from experiments.run_side_path_selection import apply_side_filter

    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    prediction = pd.Series([2, 0, 2, 0], index=index)
    confidence = pd.Series([0.8, 0.8, 0.7, 0.9], index=index)
    p_tp = pd.Series([0.4, 0.4, 0.9, 0.6], index=index)

    filtered = apply_side_filter(
        prediction,
        confidence,
        p_tp,
        primary_tau=0.75,
        long_tau=0.3,
        short_tau=0.5,
    )

    assert filtered.tolist() == [2, 1, 1, 0]


def test_side_policy_keeps_the_existing_guards():
    from experiments.run_side_path_selection import select_side_policy

    eligible = _economic_row(long_tau=0.2, short_tau=0.1, robust_score=0.8)
    thin_short = _economic_row(
        long_tau=0.3, short_tau=0.05, n_short=14, robust_score=9.0
    )

    selected = select_side_policy(
        pd.DataFrame([eligible, thin_short]), n_folds=15
    )

    assert selected is not None
    assert float(selected["long_tau"]) == 0.2
    assert select_side_policy(pd.DataFrame([thin_short]), n_folds=15) is None

def test_stage7d_variant_uses_a_distinct_cache_and_features():
    from experiments.run_side_path_selection import variant_settings
    from features.intrabar import REGIME_STAGE_FEATURES

    output_dir, path_features = variant_settings("stage7d")

    assert output_dir.name == "side_path_stage7d_selection"
    assert set(REGIME_STAGE_FEATURES).issubset(path_features)

