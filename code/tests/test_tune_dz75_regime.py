import numpy as np
import pandas as pd

from experiments.run_tune_dz75_regime import (
    past_regime_labels,
    regime_balanced_weights,
    robust_regime_score,
)


def test_past_regime_labels_do_not_change_when_future_price_changes():
    index = pd.date_range("2024-01-01", periods=800, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 120.0, len(index)), index=index)
    changed = close.copy()
    changed.iloc[-1] = 1_000.0

    original = past_regime_labels(close)
    revised = past_regime_labels(changed)

    pd.testing.assert_series_equal(original.iloc[:-1], revised.iloc[:-1])


def test_regime_balanced_weights_give_each_regime_equal_total_weight():
    regimes = pd.Series(
        ["bull"] * 6 + ["sideways"] * 3 + ["bear"],
        index=pd.RangeIndex(10),
    )

    weights = regime_balanced_weights(regimes)
    totals = weights.groupby(regimes).sum()

    assert totals.max() == totals.min()
    assert weights.mean() == 1.0


def test_robust_score_is_controlled_by_weakest_market_regime():
    assert robust_regime_score({"bull": 0.70, "sideways": 0.55, "bear": 0.40}) == 0.40
