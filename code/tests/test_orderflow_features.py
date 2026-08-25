from __future__ import annotations

import numpy as np
import pandas as pd

from features.build import FEATURE_COLS, ORDERFLOW_FEATURE_COLS, add_features, build_dataset


def _bars(n: int, *, orderflow: bool) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    df = pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1, "close": close,
        "volume": rng.uniform(100, 200, n),
    }, index=idx)
    if orderflow:
        df["count"] = rng.integers(50, 500, n)
        df["taker_buy_base"] = df["volume"] * rng.uniform(0.2, 0.8, n)
        df["taker_buy_quote"] = df["taker_buy_base"] * close
    return df


def test_orderflow_features_added_when_columns_present():
    feat = add_features(_bars(120, orderflow=True))
    assert set(ORDERFLOW_FEATURE_COLS).issubset(feat.columns)


def test_orderflow_features_skipped_when_columns_absent():
    feat = add_features(_bars(120, orderflow=False))     # index-CFD-like frame
    assert not any(c in feat.columns for c in ORDERFLOW_FEATURE_COLS)


def test_ofi_matches_definition_and_is_bounded():
    df = _bars(120, orderflow=True)
    feat = add_features(df)
    expected = 2.0 * (df["taker_buy_base"] / df["volume"]) - 1.0
    assert np.allclose(feat["ofi"], expected)
    assert feat["ofi"].between(-1.0, 1.0).all()


def test_build_dataset_includes_orderflow_by_default():
    X, _ = build_dataset(_bars(200, orderflow=True), threshold_bps=25.0)
    assert set(ORDERFLOW_FEATURE_COLS).issubset(X.columns)
    # price-only ablation and sources without taker columns stay unchanged
    Xp, _ = build_dataset(_bars(200, orderflow=True), threshold_bps=25.0, orderflow=False)
    assert list(Xp.columns) == FEATURE_COLS
    Xi, _ = build_dataset(_bars(200, orderflow=False), threshold_bps=25.0)
    assert list(Xi.columns) == FEATURE_COLS


def test_orderflow_features_are_leak_free():
    df = _bars(200, orderflow=True)
    base = add_features(df)["ofi_z20"].iloc[:150].copy()
    # perturb only FUTURE bars; a leak-free feature at t<150 must not move
    df2 = df.copy()
    df2.iloc[160:, df2.columns.get_loc("taker_buy_base")] *= 0.1
    after = add_features(df2)["ofi_z20"].iloc[:150]
    pd.testing.assert_series_equal(base, after)
