"""Build validated monthly BTCUSDT 1-second execution partitions.

The 1s candles are execution-only: CatBoost signals and features remain M15.
Each source month is processed independently so the full history is never held
in memory at once.

Run:  python code/data/build_1s.py
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

try:
    from .load import _parse_open_time
except ImportError:  # direct script execution: python code/data/build_1s.py
    from load import _parse_open_time


RAW_DIR = Path(__file__).resolve().parent / "raw" / "binance"
OUT_DIR = Path(__file__).resolve().parent / "btcusdt_1s_2024_2026"
SEALED_START = pd.Timestamp("2026-04-01", tz="UTC")
_RAW_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_base",
    "taker_buy_quote", "ignore",
]
_PRICE_COLUMNS = ["open", "high", "low", "close"]
_MONTH_PATTERN = re.compile(r"BTCUSDT-1s-(\d{4}-\d{2})\.csv$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_prices(csv_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        csv_path,
        header=None,
        names=_RAW_COLUMNS,
        usecols=range(5),
        memory_map=True,
    )
    if frame.empty:
        raise ValueError(f"empty 1s source: {csv_path}")
    numeric_time = pd.to_numeric(frame["open_time"], errors="coerce")
    if pd.isna(numeric_time.iloc[0]):
        frame = frame.iloc[1:].reset_index(drop=True)
        numeric_time = pd.to_numeric(frame["open_time"], errors="coerce")
    if frame.empty or numeric_time.isna().any():
        raise ValueError(f"invalid open_time in {csv_path}")
    for column in _PRICE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[_PRICE_COLUMNS].isna().any().any():
        raise ValueError(f"invalid OHLC in {csv_path}")
    frame.index = _parse_open_time(numeric_time)
    frame.index.name = "timestamp"
    return frame[_PRICE_COLUMNS]


def _validate(frame: pd.DataFrame, month: str, sealed_start: pd.Timestamp) -> None:
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("1s source timestamps must be sorted and unique")
    if frame.index.tz is None:
        raise ValueError("1s source timestamps must be timezone-aware")
    if (frame.index >= sealed_start).any():
        raise ValueError(f"1s source crosses sealed boundary {sealed_start.isoformat()}")
    month_start = pd.Timestamp(f"{month}-01", tz="UTC")
    month_end = month_start + pd.offsets.MonthBegin(1)
    if frame.index[0] < month_start or frame.index[-1] >= month_end:
        raise ValueError(f"1s timestamps do not match source month {month}")
    prices = frame[_PRICE_COLUMNS]
    valid = (
        (prices > 0).all(axis=1)
        & (prices["high"] >= prices["low"])
        & (prices["high"] >= prices[["open", "close"]].max(axis=1))
        & (prices["low"] <= prices[["open", "close"]].min(axis=1))
    )
    if not valid.all():
        raise ValueError("invalid OHLC relationship in 1s source")


def build_month(
    csv_path: Path,
    output_dir: Path,
    *,
    sealed_start: pd.Timestamp = SEALED_START,
) -> dict:
    """Validate one Binance 1s CSV and write its deterministic parquet partition."""
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    match = _MONTH_PATTERN.fullmatch(csv_path.name)
    if match is None:
        raise ValueError(f"unexpected 1s filename: {csv_path.name}")
    month = match.group(1)
    sealed_start = pd.Timestamp(sealed_start)
    if sealed_start.tzinfo is None:
        sealed_start = sealed_start.tz_localize("UTC")
    else:
        sealed_start = sealed_start.tz_convert("UTC")

    frame = _read_prices(csv_path)
    _validate(frame, month, sealed_start)
    expected_rows = int((frame.index[-1] - frame.index[0]) / pd.Timedelta(seconds=1)) + 1

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{month}.parquet"
    temporary = output_path.with_suffix(".parquet.part")
    try:
        frame.to_parquet(temporary)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)

    return {
        "month": month,
        "file": output_path.name,
        "rows": int(len(frame)),
        "start": frame.index[0].isoformat(),
        "end": frame.index[-1].isoformat(),
        "missing_seconds": expected_rows - len(frame),
        "source_csv_sha256": _sha256(csv_path),
        "output_sha256": _sha256(output_path),
    }


def build_all(
    raw_dir: Path,
    output_dir: Path,
    *,
    start_month: str = "2024-01",
    end_month: str = "2026-03",
    sealed_start: pd.Timestamp = SEALED_START,
) -> dict:
    """Build every requested month and atomically write the dataset manifest."""
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    months = [str(period) for period in pd.period_range(start_month, end_month, freq="M")]
    partitions = []
    for month in months:
        source = raw_dir / f"BTCUSDT-1s-{month}.csv"
        if not source.exists():
            raise FileNotFoundError(f"missing 1s source month: {source}")
        partitions.append(
            build_month(source, output_dir, sealed_start=sealed_start)
        )

    manifest = {
        "schema_version": 1,
        "symbol": "BTCUSDT",
        "interval": "1s",
        "sealed_start": pd.Timestamp(sealed_start).isoformat(),
        "partitions": partitions,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    temporary = manifest_path.with_suffix(".json.part")
    try:
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--start-month", default="2024-01")
    parser.add_argument("--end-month", default="2026-03")
    args = parser.parse_args()

    manifest = build_all(
        args.raw_dir,
        args.out_dir,
        start_month=args.start_month,
        end_month=args.end_month,
    )
    rows = sum(partition["rows"] for partition in manifest["partitions"])
    print(f"wrote {len(manifest['partitions'])} monthly 1s partitions ({rows:,} rows)")
    print(f"manifest: {args.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
