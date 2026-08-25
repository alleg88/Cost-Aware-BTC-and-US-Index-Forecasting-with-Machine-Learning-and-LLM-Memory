import pandas as pd


def _economic_row(**overrides):
    row = {
        "tp_bps": 150,
        "sl_bps": 75,
        "max_hold": 4,
        "path_tau": 0.5,
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


def test_shortlist_uses_only_top_three_sortino_rows():
    from experiments.run_joint_path_selection import shortlist_sortino

    grid = pd.DataFrame(
        {
            "tp_bps": [100, 150, 200, 250],
            "pooled_sortino": [0.2, 1.1, 0.8, 0.5],
            "pooled_net": [0.1, 0.0, 0.2, 0.3],
        }
    )

    selected = shortlist_sortino(grid)

    assert selected["tp_bps"].tolist() == [150, 200, 250]


def test_training_rows_must_close_before_prediction_month():
    from experiments.run_joint_path_selection import causal_training_mask

    candidates = pd.DataFrame(
        {
            "outcome_close_time": pd.to_datetime(
                ["2025-03-31 23:59Z", "2025-04-01 00:00Z"], utc=True
            )
        }
    )

    mask = causal_training_mask(
        candidates, pd.Timestamp("2025-04-01", tz="UTC")
    )

    assert mask.tolist() == [True, False]


def test_joint_filter_requires_primary_and_path_confidence():
    from experiments.run_joint_path_selection import apply_joint_filter

    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    prediction = pd.Series([2, 0, 2, 1], index=index)
    confidence = pd.Series([0.8, 0.7, 0.9, 0.9], index=index)
    p_tp = pd.Series([0.6, 0.8, 0.4, 0.9], index=index)

    filtered = apply_joint_filter(
        prediction,
        confidence,
        p_tp,
        primary_tau=0.75,
        path_tau=0.5,
    )

    assert filtered.tolist() == [2, 1, 1, 1]


def test_joint_policy_keeps_no_trade_guard():
    from experiments.run_joint_path_selection import select_joint_policy

    eligible = _economic_row(path_tau=0.6, robust_score=0.8)
    thin = _economic_row(path_tau=0.7, trades=49, robust_score=9.0)

    selected = select_joint_policy(
        pd.DataFrame([eligible, thin]), n_folds=15
    )

    assert selected is not None
    assert float(selected["path_tau"]) == 0.6
    assert select_joint_policy(pd.DataFrame([thin]), n_folds=15) is None
