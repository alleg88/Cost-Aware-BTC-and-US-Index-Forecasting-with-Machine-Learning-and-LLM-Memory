"""Fold-local feature quality and correlation filtering."""

from importlib import import_module

import numpy as np
import pandas as pd
import pytest


def _module():
    try:
        return import_module("experiments.pre_t2_feature_engineering")
    except ModuleNotFoundError:
        pytest.fail("pre-T2 feature engineering module is not implemented")


def test_near_duplicate_is_removed_using_registered_priority():
    module = _module()
    rng = np.random.default_rng(42)
    keep = rng.normal(size=200)
    frame = pd.DataFrame(
        {
            "keep_first": keep,
            "drop_second": keep,
            "independent": rng.normal(size=200),
        }
    )
    selection = module.fold_correlation_filter(
        frame,
        feature_order=("keep_first", "drop_second", "independent"),
    )

    assert selection.selected_features == ("keep_first", "independent")
    assert selection.removed_features == ("drop_second",)
    row = selection.audit.query(
        "feature_a == 'keep_first' and feature_b == 'drop_second'"
    ).iloc[0]
    assert np.isclose(row["abs_spearman"], 1.0)
    assert row["action"] == "drop_second"


def test_moderate_correlation_is_reported_but_not_removed():
    module = _module()
    rng = np.random.default_rng(7)
    first = rng.normal(size=1000)
    second = 0.85 * first + 0.55 * rng.normal(size=1000)
    frame = pd.DataFrame({"first": first, "second": second})
    selection = module.fold_correlation_filter(
        frame, feature_order=("first", "second")
    )

    assert selection.selected_features == ("first", "second")
    assert selection.removed_features == ()
    assert selection.audit.iloc[0]["action"] == "report_only"


def test_profile_reports_missing_constant_and_non_finite_rates():
    module = _module()
    frame = pd.DataFrame(
        {
            "good": [1.0, 2.0, 3.0],
            "constant": [4.0, 4.0, 4.0],
            "bad": [1.0, np.nan, np.inf],
        }
    )
    profile = module.profile_features(frame, ("good", "constant", "bad"))
    indexed = profile.set_index("feature")

    assert indexed.loc["constant", "unique_values"] == 1
    assert np.isclose(indexed.loc["bad", "missing_pct"], 100 / 3)
    assert np.isclose(indexed.loc["bad", "non_finite_pct"], 100 / 3)

