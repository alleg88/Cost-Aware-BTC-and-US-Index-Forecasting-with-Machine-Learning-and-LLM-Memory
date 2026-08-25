"""Load JForex/Dukascopy M15 index CSV exports and snapshot clean parquet files.

The raw CSVs exported by JForex Historical Data Manager are timestamped as
``Time (EET)``. Dukascopy uses Eastern European time with daylight saving, so the
loader localizes those naive timestamps to Europe/Helsinki and converts them to
UTC before any downstream news joins or labels.

Public API:
    load_dukascopy_instrument(raw_dir, instrument) -> DataFrame
    build_dukascopy_snapshots(raw_dir, output_dir)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

RAW_TIME_COL = "Time (EET)"
OHLC = ["open", "high", "low", "close"]
SIDES = ("bid", "ask")
DEFAULT_INSTRUMENTS = ("USA500IDXUSD", "USATECHIDXUSD")
DEFAULT_START = "2024-01-01"
DEFAULT_WORKING_END = "2025-12-31"
DEFAULT_LOCKBOX_START = "2026-01-01"
DEFAULT_LOCKBOX_END = "2026-03-31"


@dataclass(frozen=True)
class ValidationReport:
    instrument: str
    rows: int
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    duplicate_timestamps: int
    missing_grid_bars: int
    negative_spread_rows: int


def _find_export(raw_dir: Path, instrument: str, side: str) -> Path:
    matches = sorted(raw_dir.glob(f"{instrument}_15 Mins_{side.title()}_*.csv"))
    if len(matches) != 1:
        found = ", ".join(p.name for p in matches) or "none"
        raise FileNotFoundError(
            f"Expected one {instrument} {side} export in {raw_dir}; found {found}"
        )
    return matches[0]


def _read_side(csv_path: Path, side: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "time (eet)" not in df.columns:
        raise ValueError(f"{csv_path.name}: missing Time (EET) column")

    timestamp_local = pd.to_datetime(
        df["time (eet)"], format="%Y.%m.%d %H:%M:%S", errors="raise"
    )
    timestamp_utc = (
        timestamp_local.dt.tz_localize(
            "Europe/Helsinki", ambiguous="infer", nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
    )

    out = pd.DataFrame(index=pd.DatetimeIndex(timestamp_utc, name="timestamp"))
    for col in OHLC + ["volume"]:
        if col not in df.columns:
            raise ValueError(f"{csv_path.name}: missing {col} column")
        values = pd.to_numeric(df[col], errors="raise").to_numpy()
        out[f"{col}_{side}" if col in OHLC else f"volume_{side}"] = values
    return out.sort_index()


def _validate_ohlc(df: pd.DataFrame, side: str) -> None:
    high = df[f"high_{side}"]
    low = df[f"low_{side}"]
    open_ = df[f"open_{side}"]
    close = df[f"close_{side}"]
    volume = df[f"volume_{side}"]

    if not (high >= low).all():
        raise ValueError(f"{side}: high < low")
    if not (high >= pd.concat([open_, close], axis=1).max(axis=1)).all():
        raise ValueError(f"{side}: high < max(open, close)")
    if not (low <= pd.concat([open_, close], axis=1).min(axis=1)).all():
        raise ValueError(f"{side}: low > min(open, close)")
    if not (pd.concat([open_, high, low, close], axis=1) > 0).all().all():
        raise ValueError(f"{side}: non-positive price")
    if not (volume >= 0).all():
        raise ValueError(f"{side}: negative volume")


def _validation_report(df: pd.DataFrame, instrument: str) -> ValidationReport:
    full_grid = pd.date_range(df.index.min(), df.index.max(), freq="15min", tz="UTC")
    trading_gap_count = len(full_grid.difference(df.index))
    duplicate_count = int(df.index.duplicated().sum())
    negative_spread_count = int((df["spread_close"] < 0).sum())
    return ValidationReport(
        instrument=instrument,
        rows=len(df),
        start_utc=df.index.min(),
        end_utc=df.index.max(),
        duplicate_timestamps=duplicate_count,
        missing_grid_bars=trading_gap_count,
        negative_spread_rows=negative_spread_count,
    )


def load_dukascopy_instrument(
    raw_dir: str | Path, instrument: str, start: str | None = None, end: str | None = None
) -> pd.DataFrame:
    """Return one instrument with BID/ASK OHLCV, mid prices, and close spread in UTC."""
    raw_dir = Path(raw_dir)
    bid = _read_side(_find_export(raw_dir, instrument, "bid"), "bid")
    ask = _read_side(_find_export(raw_dir, instrument, "ask"), "ask")

    df = bid.join(ask, how="inner")
    if df.empty:
        raise ValueError(f"{instrument}: bid/ask join produced no rows")
    df = df[~df.index.duplicated(keep="first")].sort_index()

    for side in SIDES:
        _validate_ohlc(df, side)

    for col in OHLC:
        df[f"{col}_mid"] = (df[f"{col}_bid"] + df[f"{col}_ask"]) / 2.0
    df["spread_close"] = df["close_ask"] - df["close_bid"]
    df["spread_close_bps"] = df["spread_close"] / df["close_mid"] * 10_000.0
    df["volume"] = df[["volume_bid", "volume_ask"]].max(axis=1)
    df["instrument"] = instrument

    # Downstream price-only code expects unsuffixed OHLCV columns.
    for col in OHLC:
        df[col] = df[f"{col}_mid"]

    if (df["spread_close"] < 0).any():
        raise ValueError(f"{instrument}: negative close spread after bid/ask merge")

    if start is not None:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        df = df[df.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return df


def write_snapshots(
    df: pd.DataFrame,
    instrument: str,
    output_dir: str | Path,
    working_end: str = DEFAULT_WORKING_END,
    lockbox_start: str = DEFAULT_LOCKBOX_START,
    lockbox_end: str = DEFAULT_LOCKBOX_END,
) -> tuple[Path, Path]:
    """Write 2024-2025 working and Q1-2026 lockbox parquet snapshots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    working = df[df.index < pd.Timestamp(working_end, tz="UTC") + pd.Timedelta(days=1)]
    lockbox = df[
        (df.index >= pd.Timestamp(lockbox_start, tz="UTC"))
        & (df.index < pd.Timestamp(lockbox_end, tz="UTC") + pd.Timedelta(days=1))
    ]
    if working.empty:
        raise ValueError(f"{instrument}: working snapshot is empty")
    if lockbox.empty:
        raise ValueError(f"{instrument}: lockbox snapshot is empty")

    stem = instrument.lower().replace("idxusd", "")
    working_path = output_dir / f"{stem}_m15_2024_2025.parquet"
    lockbox_path = output_dir / f"{stem}_m15_lockbox_2026Q1.parquet"
    working.to_parquet(working_path, engine="pyarrow", index=True)
    lockbox.to_parquet(lockbox_path, engine="pyarrow", index=True)
    return working_path, lockbox_path


def build_dukascopy_snapshots(
    raw_dir: str | Path,
    output_dir: str | Path,
    instruments: tuple[str, ...] = DEFAULT_INSTRUMENTS,
) -> list[ValidationReport]:
    """Load, validate, and write snapshots for all configured Dukascopy instruments."""
    reports: list[ValidationReport] = []
    for instrument in instruments:
        df = load_dukascopy_instrument(
            raw_dir, instrument, start=DEFAULT_START, end=DEFAULT_LOCKBOX_END
        )
        report = _validation_report(df, instrument)
        reports.append(report)
        working_path, lockbox_path = write_snapshots(df, instrument, output_dir)
        print(
            f"{instrument}: {report.rows:,} rows, "
            f"{report.start_utc} -> {report.end_utc}, "
            f"spread<0={report.negative_spread_rows}, "
            f"grid gaps={report.missing_grid_bars:,}"
        )
        print(f"  wrote {working_path}")
        print(f"  wrote {lockbox_path}")
    return reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        default=Path(__file__).resolve().parent / "raw" / "dukascopy",
        type=Path,
    )
    parser.add_argument(
        "--output-dir",
        default=Path(__file__).resolve().parent,
        type=Path,
    )
    args = parser.parse_args()
    build_dukascopy_snapshots(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
