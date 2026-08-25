from __future__ import annotations

from pathlib import Path

import pandas as pd

from data.dukascopy import load_dukascopy_instrument, write_snapshots


def _write_export(path: Path, rows: list[tuple[str, float, float]]) -> None:
    lines = ["Time (EET),Open,High,Low,Close,Volume "]
    for timestamp, open_, close in rows:
        high = max(open_, close) + 1.0
        low = min(open_, close) - 1.0
        lines.append(f"{timestamp},{open_},{high},{low},{close},1.0")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_load_dukascopy_instrument_merges_bid_ask_and_converts_to_utc(tmp_path):
    rows_bid = [
        ("2024.01.02 01:00:00", 100.0, 101.0),
        ("2024.01.02 01:15:00", 101.0, 102.0),
    ]
    rows_ask = [
        ("2024.01.02 01:00:00", 100.5, 101.5),
        ("2024.01.02 01:15:00", 101.5, 102.5),
    ]
    _write_export(tmp_path / "TESTIDXUSD_15 Mins_Bid_2024.01.01_2026.03.31.csv", rows_bid)
    _write_export(tmp_path / "TESTIDXUSD_15 Mins_Ask_2024.01.01_2026.03.31.csv", rows_ask)

    df = load_dukascopy_instrument(tmp_path, "TESTIDXUSD")

    assert df.index.tz is not None
    assert df.index[0] == pd.Timestamp("2024-01-01 23:00:00", tz="UTC")
    assert df.loc[df.index[0], "close"] == 101.25
    assert df.loc[df.index[0], "spread_close"] == 0.5
    assert df.loc[df.index[0], "instrument"] == "TESTIDXUSD"


def test_write_snapshots_splits_working_and_lockbox(tmp_path):
    idx = pd.to_datetime(
        ["2024-01-02 00:00:00", "2025-12-31 23:45:00", "2026-01-02 00:00:00"],
        utc=True,
    )
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0, 3.0],
            "high": [1.0, 2.0, 3.0],
            "low": [1.0, 2.0, 3.0],
            "close": [1.0, 2.0, 3.0],
            "volume": [1.0, 1.0, 1.0],
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )

    working_path, lockbox_path = write_snapshots(df, "TESTIDXUSD", tmp_path)

    working = pd.read_parquet(working_path)
    lockbox = pd.read_parquet(lockbox_path)
    assert len(working) == 2
    assert len(lockbox) == 1
