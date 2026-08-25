from __future__ import annotations

from experiments.run_walkforward_arms import arm_params, arms_for_width


def test_sqrtbalanced_arm_is_limited_to_the_preregistered_dz40_comparison():
    """The new arm changes only class weighting at the nominated BTC width."""
    assert arms_for_width(40) == ("legacy", "f1", "econ", "sqrtbalanced")
    assert arms_for_width(55) == ("legacy", "f1", "econ")

    params = arm_params("sqrtbalanced", 40)
    assert params["auto_class_weights"] == "SqrtBalanced"
