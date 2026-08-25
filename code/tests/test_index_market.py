from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.index_market import (
    LOCKBOX_START,
    load_index_minutes,
    resample_index_minutes,
)


def _write_export(
    path: Path,
    rows: list[tuple[str, float, float, float, float, float]],
) -> None:
    lines = ["Time (EET),Open,High,Low,Close,Volume"]
    lines.extend(
        f"{timestamp},{open_},{high},{low},{close},{volume}"
        for timestamp, open_, high, low, close, volume in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _export_pair(
    root: Path,
    bid: list[tuple[str, float, float, float, float, float]],
    ask: list[tuple[str, float, float, float, float, float]],
) -> None:
    suffix = "2024.01.01_2026.08.01.csv"
    _write_export(root / f"TESTIDXUSD_1 Min_Bid_{suffix}", bid)
    _write_export(root / f"TESTIDXUSD_1 Min_Ask_{suffix}", ask)


def test_load_index_minutes_is_utc_cutoff_first_and_flags_crosses(tmp_path: Path):
    bid = [
        ("2024.01.02 01:00:00", 100.0, 101.0, 99.0, 100.0, 1.0),
        ("2024.01.02 01:01:00", 101.0, 102.0, 100.0, 101.0, 2.0),
        ("2026.04.01 03:00:00", 200.0, 201.0, 199.0, 200.0, 1.0),
        ("2026.04.01 03:00:00", 200.0, 201.0, 199.0, 200.0, 1.0),
    ]
    ask = [
        ("2024.01.02 01:00:00", 100.5, 101.5, 99.5, 100.5, 1.0),
        ("2024.01.02 01:01:00", 100.5, 101.5, 99.5, 100.5, 2.0),
        ("2026.04.01 03:00:00", 200.5, 201.5, 199.5, 200.5, 1.0),
        ("2026.04.01 03:00:00", 200.5, 201.5, 199.5, 200.5, 1.0),
    ]
    _export_pair(tmp_path, bid, ask)

    minute, audit = load_index_minutes(
        tmp_path, "TESTIDXUSD", end_exclusive=LOCKBOX_START
    )

    assert str(minute.index.tz) == "UTC"
    assert minute.index.max() < LOCKBOX_START
    assert minute.index[0] == pd.Timestamp("2024-01-01 23:00:00", tz="UTC")
    assert int(minute["crossed_close"].sum()) == 1
    assert (minute["execution_spread_bps"] >= 0).all()
    assert minute.loc[minute.index[1], "raw_spread_close"] == pytest.approx(-0.5)
    assert audit["rows_after_cutoff"] == 2
    assert audit["rows_removed_by_cutoff"] == 2
    assert audit["duplicate_timestamps"] == 0


def test_load_index_minutes_requires_exact_bid_ask_timestamp_alignment(tmp_path: Path):
    bid = [("2024.01.02 01:00:00", 100.0, 101.0, 99.0, 100.0, 1.0)]
    ask = [("2024.01.02 01:01:00", 100.5, 101.5, 99.5, 100.5, 1.0)]
    _export_pair(tmp_path, bid, ask)

    with pytest.raises(ValueError, match="timestamp grids differ"):
        load_index_minutes(tmp_path, "TESTIDXUSD", end_exclusive=LOCKBOX_START)


def test_resampled_bar_records_open_availability_and_minute_count():
    index = pd.date_range("2024-01-01", periods=29, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {
            "open": range(100, 129),
            "high": range(101, 130),
            "low": range(99, 128),
            "close": range(100, 129),
            "volume": 1.0,
            "raw_spread_close_bps": 2.0,
            "crossed_close": False,
        },
        index=index,
    )

    m15 = resample_index_minutes(minute, "15min")

    assert m15["minute_count"].tolist() == [15, 14]
    assert m15["complete_bar"].tolist() == [True, False]
    assert m15["bar_open"].equals(pd.Series(m15.index, index=m15.index))
    assert (
        m15["available_at"] == m15.index + pd.Timedelta(minutes=15)
    ).all()
