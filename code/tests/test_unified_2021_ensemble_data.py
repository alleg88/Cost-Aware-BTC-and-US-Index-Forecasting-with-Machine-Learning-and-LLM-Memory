from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import (
    UNIFIED_FEATURES,
    UnifiedDataConfig,
    build_causal_sequences,
    build_economic_labels,
    build_unified_dataset,
    make_blocking_fold_manifest,
)


def _m15_fixture(periods: int = 96) -> pd.DataFrame:
    index = pd.date_range("2021-01-01", periods=periods, freq="15min", tz="UTC")
    base = 100.0 * np.exp(np.arange(periods, dtype=float) * 0.0003)
    close = base * 1.0001
    volume = 1000.0 + np.arange(periods, dtype=float)
    return pd.DataFrame(
        {
            "open": base,
            "high": np.maximum(base, close) * 1.001,
            "low": np.minimum(base, close) * 0.999,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "count": 100 + np.arange(periods),
            "taker_buy_base": volume * 0.55,
            "taker_buy_quote": volume * close * 0.55,
            "minute_count": 15,
        },
        index=index,
    )


def _minute_fixture(m15: pd.DataFrame) -> pd.DataFrame:
    periods = len(m15) * 15 + 181
    index = pd.date_range(m15.index.min(), periods=periods, freq="1min", tz="UTC")
    open_price = 100.0 * np.exp(np.arange(periods, dtype=float) * 0.00002)
    close = open_price * np.exp(0.00002)
    volume = 50.0 + (np.arange(periods) % 17)
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.00001,
            "low": np.minimum(open_price, close) * 0.99999,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "count": 10 + (np.arange(periods) % 5),
            "taker_buy_base": volume * 0.53,
            "taker_buy_quote": volume * close * 0.53,
        },
        index=index,
    )


def _positioning_fixture(m15: pd.DataFrame) -> pd.DataFrame:
    rows = len(m15)
    return pd.DataFrame(
        {
            "funding_rate": np.sin(np.arange(rows) / 13.0) * 1e-5,
            "sum_open_interest": 1_000_000.0 + np.arange(rows) * 100.0,
            "toptrader_ls": 1.05 + np.sin(np.arange(rows) / 19.0) * 0.01,
            "taker_ls": 0.98 + np.cos(np.arange(rows) / 17.0) * 0.01,
            "positioning_stale": False,
            "positioning_age_min": 0.0,
        },
        index=m15.index,
    )


def test_decision_and_positioning_availability_are_causal():
    m15 = _m15_fixture()
    dataset = build_unified_dataset(m15, _minute_fixture(m15), _positioning_fixture(m15))

    decisions = dataset.decisions
    assert (decisions["decision_time"] == decisions["bar_time"] + pd.Timedelta(minutes=15)).all()
    assert (decisions["positioning_availability_time"] <= decisions["decision_time"]).all()
    assert (decisions["entry_time"] > decisions["decision_time"]).all()
    assert tuple(dataset.feature_names) == UNIFIED_FEATURES
    assert not {"hour", "dayofweek"}.intersection(dataset.feature_names)
    assert not any(
        token in name
        for name in dataset.feature_names
        for token in ("future", "label", "target", "outcome", "net_r")
    )
    assert dataset.tabular.shape == (len(decisions), len(UNIFIED_FEATURES))


def test_lstm_window_ends_on_same_row_as_tabular_decision():
    values = np.arange(30, dtype=np.float32).reshape(10, 3)

    sequences = build_causal_sequences(values, sequence_length=4)

    assert sequences.shape == (10, 4, 3)
    np.testing.assert_array_equal(sequences[:, -1, :], values)
    np.testing.assert_array_equal(sequences[0], np.repeat(values[:1], 4, axis=0))


def test_economic_labels_pair_identical_complete_half_open_paths():
    m15 = _m15_fixture(48)
    decisions = build_unified_dataset(
        m15, _minute_fixture(m15), _positioning_fixture(m15)
    ).decisions
    labels, paths = build_economic_labels(decisions, _minute_fixture(m15))

    complete_paths = paths.loc[paths["path_complete"]].copy()
    paired_signatures = complete_paths.pivot(
        index="row_key", columns="direction", values="path_signature"
    )
    assert (paired_signatures["long"] == paired_signatures["short"]).all()
    assert set(complete_paths["target_multiple_b"]) == {2.0}
    assert set(complete_paths["hold_minutes"]) == {120}
    assert set(complete_paths["cost_bps"]) == {10.0}
    complete_labels = labels.loc[labels["path_complete"]]
    expected_opportunity = complete_labels[["net_r_long", "net_r_short"]].max(axis=1).gt(0.0)
    assert (complete_labels["opportunity"].astype(bool) == expected_opportunity).all()
    assert complete_labels.loc[complete_labels["side"].eq("tie"), "side_eligible"].eq(False).all()
    assert (
        complete_labels["label_end"]
        == complete_labels["entry_time"] + pd.Timedelta(minutes=120)
    ).all()


def test_missing_minute_censors_both_hypothetical_sides():
    m15 = _m15_fixture(48)
    minute = _minute_fixture(m15)
    dataset = build_unified_dataset(m15, minute, _positioning_fixture(m15))
    target = dataset.decisions.iloc[20]
    missing_time = target["decision_time"] + pd.Timedelta(minutes=30)
    minute = minute.drop(index=missing_time)

    labels, paths = build_economic_labels(dataset.decisions, minute)

    selected_label = labels.loc[labels["row_key"].eq(target["row_key"])].iloc[0]
    selected_paths = paths.loc[paths["row_key"].eq(target["row_key"])]
    assert not bool(selected_label["path_complete"])
    assert selected_label["side"] == "censored"
    assert len(selected_paths) == 2
    assert selected_paths["censored"].astype(bool).all()


def test_fold_manifest_is_non_overlapping_embargoed_and_label_purged():
    index = pd.date_range("2021-01-01 00:15", periods=1000, freq="15min", tz="UTC")
    decisions = pd.DataFrame(
        {
            "row_key": [f"row-{position}" for position in range(len(index))],
            "decision_time": index,
            "adaptive_barrier_bps": 100.0,
        }
    )
    labels = pd.DataFrame(
        {
            "row_key": decisions["row_key"],
            "path_complete": True,
            "label_end": index + pd.Timedelta(minutes=121),
        }
    )

    manifest = make_blocking_fold_manifest(
        decisions, labels, UnifiedDataConfig(n_splits=5, embargo_bars=8)
    )

    outer_test_keys = manifest.loc[manifest["role"].eq("test"), "row_key"]
    assert not outer_test_keys.duplicated().any()
    assert set(manifest["fold_id"]) == {0, 1, 2, 3, 4}
    for _, fold in manifest.groupby("fold_id", sort=True):
        fit = fold.loc[fold["role"].eq("fit")]
        calibration = fold.loc[fold["role"].eq("calibration")]
        test = fold.loc[fold["role"].eq("test")]
        assert len(fit) and len(calibration) and len(test)
        assert fit["label_end"].max() < calibration["decision_time"].min()
        assert calibration["label_end"].max() < test["decision_time"].min()
        assert set(fit["row_key"]).isdisjoint(set(calibration["row_key"]))
        assert set(calibration["row_key"]).isdisjoint(set(test["row_key"]))
        assert fold["outer_train_fraction"].dropna().unique().tolist() == [0.8]

