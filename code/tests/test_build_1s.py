import json
from pathlib import Path

import pandas as pd
import pytest

from data.build_1s import build_all, build_month


RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


def _kline(open_time: int, price: float, *, unit: str) -> str:
    step = {"ms": 1_000, "us": 1_000_000}[unit]
    values = [
        open_time,
        price,
        price + 1.0,
        price - 1.0,
        price + 0.5,
        10.0,
        open_time + step - 1,
        1000.0,
        20,
        6.0,
        600.0,
        0,
    ]
    return ",".join(map(str, values))


def _write_month(
    raw: Path,
    month: str,
    start: int,
    *,
    unit: str,
    header: bool = False,
) -> Path:
    lines = [
        _kline(start, 100.0, unit=unit),
        _kline(start + {"ms": 1_000, "us": 1_000_000}[unit], 101.0, unit=unit),
    ]
    if header:
        lines.insert(0, ",".join(RAW_COLUMNS))
    path = raw / f"BTCUSDT-1s-{month}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_build_all_writes_monthly_partitions_and_manifest_for_ms_and_us(tmp_path: Path):
    raw = tmp_path / "raw"
    out = tmp_path / "partitions"
    raw.mkdir()
    _write_month(raw, "2024-01", 1_704_067_200_000, unit="ms")
    _write_month(
        raw,
        "2024-02",
        1_706_745_600_000_000,
        unit="us",
        header=True,
    )

    manifest = build_all(
        raw,
        out,
        start_month="2024-01",
        end_month="2024-02",
    )

    saved = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert saved == manifest
    assert [row["month"] for row in manifest["partitions"]] == ["2024-01", "2024-02"]
    assert manifest["sealed_start"] == "2026-04-01T00:00:00+00:00"
    for row in manifest["partitions"]:
        partition = out / row["file"]
        frame = pd.read_parquet(partition)
        assert list(frame.columns) == ["open", "high", "low", "close"]
        assert len(frame) == 2
        assert frame.index.tz is not None
        assert frame.index.is_unique
        assert frame.index.is_monotonic_increasing
        assert row["rows"] == 2
        assert row["missing_seconds"] == 0
        assert len(row["source_csv_sha256"]) == 64
        assert len(row["output_sha256"]) == 64


def test_build_month_rejects_unsorted_timestamps(tmp_path: Path):
    raw = tmp_path / "BTCUSDT-1s-2024-01.csv"
    first = 1_704_067_200_000
    raw.write_text(
        _kline(first + 1_000, 101.0, unit="ms") + "\n"
        + _kline(first, 100.0, unit="ms") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sorted and unique"):
        build_month(raw, tmp_path / "out")


def test_build_month_rejects_invalid_ohlc(tmp_path: Path):
    raw = tmp_path / "BTCUSDT-1s-2024-01.csv"
    values = _kline(1_704_067_200_000, 100.0, unit="ms").split(",")
    values[2] = "99.0"
    raw.write_text(",".join(values) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid OHLC"):
        build_month(raw, tmp_path / "out")


def test_build_month_rejects_sealed_2026_q2_data(tmp_path: Path):
    raw = tmp_path / "BTCUSDT-1s-2026-04.csv"
    raw.write_text(
        _kline(1_775_001_600_000, 100.0, unit="ms") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sealed boundary"):
        build_month(raw, tmp_path / "out")
