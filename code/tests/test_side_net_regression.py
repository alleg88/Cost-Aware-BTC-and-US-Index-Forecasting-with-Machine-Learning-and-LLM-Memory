import pandas as pd
import pytest


def _economic_row(**overrides):
    row = {
        "long_tau": 0.001,
        "short_tau": 0.001,
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


def test_realized_net_target_subtracts_round_trip_fee():
    from experiments.run_side_net_regression import realized_net_target

    candidates = pd.DataFrame({"gross_return": [0.02, -0.01]})

    result = realized_net_target(candidates, fee_bps=5.0)

    assert result.tolist() == pytest.approx([0.019, -0.011])


def test_net_filter_uses_independent_side_thresholds():
    from experiments.run_side_net_regression import apply_net_filter

    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    prediction = pd.Series([2, 0, 2, 0], index=index)
    confidence = pd.Series([0.8, 0.8, 0.7, 0.9], index=index)
    expected_net = pd.Series([0.003, 0.0005, 0.01, 0.002], index=index)

    result = apply_net_filter(
        prediction,
        confidence,
        expected_net,
        primary_tau=0.75,
        long_tau=0.002,
        short_tau=0.001,
    )

    assert result.tolist() == [2, 1, 1, 0]


def test_net_study_freezes_geometry_and_output():
    from experiments.run_side_net_regression import GEOMETRY, OUTPUT_DIR

    assert GEOMETRY == (200, 100, 1)
    assert OUTPUT_DIR.name == "side_net_regression_selection"


def test_net_training_mask_is_causal_and_side_specific():
    from experiments.run_side_net_regression import net_training_mask

    candidates = pd.DataFrame(
        {
            "outcome_close_time": pd.to_datetime(
                ["2025-03-31 23:59Z", "2025-03-31 23:59Z", "2025-04-01 00:00Z"],
                utc=True,
            ),
            "side": [1, -1, 1],
        }
    )

    result = net_training_mask(
        candidates, pd.Timestamp("2025-04-01", tz="UTC"), side=1
    )

    assert result.tolist() == [True, False, False]


def test_net_policy_keeps_the_existing_guards():
    from experiments.run_side_net_regression import select_net_policy

    eligible = _economic_row(long_tau=0.002, short_tau=0.001, robust_score=0.8)
    thin_short = _economic_row(n_short=14, robust_score=9.0)

    selected = select_net_policy(pd.DataFrame([eligible, thin_short]), n_folds=15)

    assert selected is not None
    assert float(selected["long_tau"]) == 0.002
    assert select_net_policy(pd.DataFrame([thin_short]), n_folds=15) is None
