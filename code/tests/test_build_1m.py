from pathlib import Path

import pandas as pd

from data.build_1m import build_snapshot


def _kline(open_time: int, price: float) -> str:
    values = [
        open_time,
        price,
        price + 1.0,
        price - 1.0,
        price + 0.5,
        10.0,
        open_time + 59_999,
        1000.0,
        20,
        6.0,
        600.0,
        0,
    ]
    return ",".join(map(str, values))


def test_build_snapshot_writes_requested_full_span_path(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    first = 1_704_067_200_000
    (raw / "BTCUSDT-1m-2024-01.csv").write_text(
        _kline(first, 100.0) + "\n" + _kline(first + 60_000, 101.0) + "\n",
        encoding="utf-8",
    )
    (raw / "BTCUSDT-1m-2024-02.csv").write_text(
        _kline(first + 60_000, 101.0) + "\n" + _kline(first + 120_000, 102.0) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "nested" / "btcusdt_1m_2024_2026.parquet"

    built = build_snapshot(raw, out)

    assert out.exists()
    assert len(built) == 3
    assert built.index.is_unique
    assert built.index.is_monotonic_increasing
    assert built.index.tz is not None
    pd.testing.assert_frame_equal(pd.read_parquet(out), built)
