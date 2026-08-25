import itertools

import numpy as np
import pandas as pd
import pytest

from experiments.raw_hold_control import (
    MODEL_NAMES,
    build_raw_hold_summary,
    load_raw_hold_summary,
    validate_raw_hold_summary,
    simulate_fixed_hold,
)


def _bars() -> pd.DataFrame:
    index = pd.date_range("2025-07-01", periods=6, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0, 100.0, 105.0, 110.0, 108.0, 106.0],
            "high": [101.0, 105.0, 111.0, 111.0, 109.0, 107.0],
            "low": [99.0, 99.0, 104.0, 107.0, 105.0, 99.0],
            "close": [100.0, 104.0, 110.0, 108.0, 106.0, 100.0],
        },
        index=index,
    )


def test_fixed_hold_enters_next_open_exits_fixed_close_and_skips_overlap():
    bars = _bars()
    pred = pd.Series(
        [2, 0, 0, 2, 1, 1],
        index=bars.index,
        dtype=int,
    )

    ledger, per_bar = simulate_fixed_hold(
        bars,
        pred,
        hold_bars=2,
        fee_bps=5.0,
    )

    assert ledger["side"].tolist() == [1, -1]
    assert ledger["entry_time"].tolist() == [bars.index[1], bars.index[3]]
    assert ledger["exit_time"].tolist() == [bars.index[2], bars.index[4]]
    assert ledger["bars_held"].tolist() == [2, 2]
    assert ledger["exit_reason"].tolist() == ["fixed_hold", "fixed_hold"]
    assert ledger.iloc[0]["gross_return"] == pytest.approx(0.10)
    assert ledger.iloc[0]["net_return"] == pytest.approx(0.099)
    assert ledger.iloc[1]["gross_return"] == pytest.approx(
        -(106.0 / 110.0 - 1.0)
    )
    assert float(per_bar.sum()) == pytest.approx(float(ledger["net_return"].sum()))


def test_fixed_hold_one_bar_charges_entry_and_exit_costs():
    bars = _bars()
    pred = pd.Series(1, index=bars.index, dtype=int)
    pred.iloc[0] = 2

    ledger, per_bar = simulate_fixed_hold(
        bars,
        pred,
        hold_bars=1,
        fee_bps=5.0,
    )

    assert len(ledger) == 1
    assert ledger.iloc[0]["entry_time"] == bars.index[1]
    assert ledger.iloc[0]["exit_time"] == bars.index[1]
    assert ledger.iloc[0]["gross_return"] == pytest.approx(0.04)
    assert ledger.iloc[0]["net_return"] == pytest.approx(0.039)
    assert per_bar.loc[bars.index[1]] == pytest.approx(0.039)


def test_fixed_hold_rejects_invalid_inputs():
    bars = _bars()
    pred = pd.Series(1, index=bars.index, dtype=int)

    with pytest.raises(ValueError, match="hold_bars"):
        simulate_fixed_hold(bars, pred, hold_bars=0, fee_bps=5.0)
    with pytest.raises(ValueError, match="timezone"):
        simulate_fixed_hold(
            bars.tz_localize(None),
            pred.tz_localize(None),
            hold_bars=1,
            fee_bps=5.0,
        )


def _valid_summary() -> pd.DataFrame:
    rows = []
    for model_name, width_bps, hold_minutes in itertools.product(
        MODEL_NAMES,
        (55, 65, 75),
        (15, 30),
    ):
        rows.append(
            {
                "model_name": model_name,
                "model": model_name,
                "width_bps": width_bps,
                "candidate_id": 0,
                "hold_minutes": hold_minutes,
                "fee_bps_per_side": 5.0,
                "fit_end": pd.Timestamp("2025-01-01", tz="UTC"),
                "period_start": pd.Timestamp("2025-07-01", tz="UTC"),
                "period_end": pd.Timestamp("2026-04-01", tz="UTC"),
                "trades": 100,
                "n_long": 50,
                "n_short": 50,
                "gross_return": 0.02,
                "net_return": 0.01,
                "sortino": 0.5,
                "sharpe": 0.3,
                "positive_months": 5,
            }
        )
    return pd.DataFrame(rows)


def test_raw_hold_summary_contract_has_exact_cartesian_grid_and_sealed_boundary():
    frame = _valid_summary()

    validate_raw_hold_summary(frame)

    with pytest.raises(ValueError, match="54"):
        validate_raw_hold_summary(frame.iloc[:-1])
    duplicated = pd.concat([frame.iloc[:-1], frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="unique"):
        validate_raw_hold_summary(duplicated)
    leaked = frame.copy()
    leaked.loc[0, "period_end"] = pd.Timestamp("2026-04-02", tz="UTC")
    with pytest.raises(ValueError, match="sealed"):
        validate_raw_hold_summary(leaked)
    refitted = frame.copy()
    refitted.loc[0, "fit_end"] = pd.Timestamp("2025-06-30", tz="UTC")
    with pytest.raises(ValueError, match="2024-only"):
        validate_raw_hold_summary(refitted)


def test_raw_hold_summary_metrics_are_finite_and_trade_sides_reconcile():
    frame = _valid_summary()
    frame.loc[0, "sortino"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_raw_hold_summary(frame)

    frame = _valid_summary()
    frame.loc[0, "n_long"] = 49
    with pytest.raises(ValueError, match="trade sides"):
        validate_raw_hold_summary(frame)


def test_build_raw_hold_summary_reuses_2024_frozen_predictions(tmp_path):
    output = tmp_path / "raw_hold_summary.parquet"

    built = build_raw_hold_summary(output_path=output)
    loaded = load_raw_hold_summary(output)

    assert output.exists()
    assert len(built) == 54
    assert loaded.equals(built)
    assert set(loaded["hold_minutes"]) == {15, 30}
    assert pd.to_datetime(loaded["fit_end"], utc=True).max() < pd.Timestamp(
        "2025-01-01", tz="UTC"
    )
    assert pd.to_datetime(loaded["period_end"], utc=True).max() == pd.Timestamp(
        "2026-04-01", tz="UTC"
    )


def test_load_raw_hold_summary_requires_an_existing_valid_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_raw_hold_summary(tmp_path / "missing.parquet")

    invalid = _valid_summary().iloc[:-1]
    path = tmp_path / "invalid.parquet"
    invalid.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="54"):
        load_raw_hold_summary(path)
