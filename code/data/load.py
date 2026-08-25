"""Load raw Binance klines, clean them, and snapshot to parquet.

15m is a native Binance interval, so bars are used directly — no aggregation. The
resample path is kept only for the case where a coarser target than the downloaded base
is ever requested; when base == target (the M15 case) it is a no-op and skipped.

Two Binance gotchas handled here:
  * open_time switched from milliseconds (2024 files) to microseconds (2025+).
  * some monthly CSVs ship with a header row, others do not.

Public API:
    load_bars(raw_dir, base_interval, target_interval, start, end) -> DataFrame
    build_snapshots(config_path)   # write working + lockbox parquet from the config
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import yaml

# Binance monthly kline column order (12 columns, no reliable header).
_RAW_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_base", "taker_buy_quote", "ignore",
]
_OHLCV = ["open", "high", "low", "close", "volume"]
# Order-flow columns from the Binance kline schema: aggressor-side volume and
# trade count. Kept alongside OHLCV so microstructure features can be derived.
# Other vendors (e.g. Dukascopy index CFDs) do not carry these, which the
# feature builder handles by simply skipping the order-flow block.
_ORDERFLOW = ["quote_volume", "count", "taker_buy_base", "taker_buy_quote"]


def _pandas_freq(interval: str) -> str:
    """Binance interval ('15m', '1h') -> pandas offset alias ('15min', '1h')."""
    m = re.fullmatch(r"(\d+)([mhdw])", interval)
    if not m:
        return interval
    n, unit = m.groups()
    return f"{n}{'min' if unit == 'm' else unit}"


def _parse_open_time(series: pd.Series) -> pd.DatetimeIndex:
    """Parse Binance epoch timestamps, auto-detecting ms vs microseconds by magnitude."""
    vals = pd.to_numeric(series, errors="coerce")
    mx = vals.max()
    if mx > 1e17:        # nanoseconds
        unit = "ns"
    elif mx > 1e14:      # microseconds (2025+ files)
        unit = "us"
    elif mx > 1e11:      # milliseconds (2024 files)
        unit = "ms"
    else:                # seconds
        unit = "s"
    return pd.to_datetime(vals, unit=unit, utc=True)


def _read_one(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None, names=_RAW_COLS)
    # Drop a header row if the file shipped with one (open_time not numeric).
    if not str(df.iloc[0]["open_time"]).replace(".", "").isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    df.index = _parse_open_time(df["open_time"])
    df.index.name = "timestamp"
    for col in _OHLCV + _ORDERFLOW:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[_OHLCV + _ORDERFLOW]


def load_raw(raw_dir: str | Path, base_interval: str = "15m",
             symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Read and concatenate all monthly CSVs for the base interval into one UTC frame."""
    raw_dir = Path(raw_dir)
    pattern = f"{symbol}-{base_interval}-*.csv"
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No {pattern} in {raw_dir}. Run download_binance.py first."
        )
    return pd.concat([_read_one(f) for f in files]).sort_index()


def clean(df: pd.DataFrame, base_interval: str = "15m") -> pd.DataFrame:
    """Deduplicate, drop NaNs, assert OHLC integrity, and report grid gaps."""
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = df.dropna(subset=_OHLCV)

    # Integrity checks — a hard stop beats silently training on corrupt bars.
    assert (df["high"] >= df["low"]).all(), "high < low"
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), "high < max(open,close)"
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), "low > min(open,close)"
    assert (df[["open", "high", "low", "close"]] > 0).all().all(), "non-positive price"
    assert (df["volume"] >= 0).all(), "negative volume"

    freq = _pandas_freq(base_interval)
    full = pd.date_range(df.index[0], df.index[-1], freq=freq, tz="UTC")
    missing = len(full) - len(df)
    if missing:
        print(f"  [clean] {missing:,} missing {base_interval} bars "
              f"({100 * missing / len(full):.3f}% of the grid)")
    return df


def _resample(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Resample to a coarser target. Bar stamped by OPEN time; contents in [t, t+target)."""
    spec = dict(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    for col in _ORDERFLOW:                       # order-flow is additive over the bar
        if col in df.columns:
            spec[col] = (col, "sum")
    agg = df.resample(target, label="left", closed="left").agg(**spec)
    return agg.dropna(subset=["open", "high", "low", "close"])


def load_bars(
    raw_dir: str | Path,
    base_interval: str = "15m",
    target_interval: str = "15min",
    start: str | None = None,
    end: str | None = None,
    symbol: str = "BTCUSDT",
) -> pd.DataFrame:
    """Raw -> clean (-> resample only if target is coarser than base) -> date slice."""
    bars = clean(load_raw(raw_dir, base_interval, symbol), base_interval)
    if _pandas_freq(base_interval) != _pandas_freq(target_interval):
        bars = _resample(bars, target_interval)
    if start is not None:
        bars = bars[bars.index >= pd.Timestamp(start, tz="UTC")]
    if end is not None:
        # `end` names the last day to keep, so the slice runs to the final bar of
        # that day. The bound is exclusive: an inclusive one admits the first bar
        # of the following day, which is how a single bar came to sit in both the
        # working snapshot and the sealed lockbox.
        bars = bars[bars.index < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]
    return bars


def build_snapshots(config_path: str | Path) -> None:
    """Write the configured working and sealed lockbox parquet snapshots."""
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text())
    code_root = config_path.resolve().parents[1]  # .../code
    raw_dir = code_root / cfg["paths"]["raw_binance"]
    base = cfg["base_interval"]
    target = cfg["target_interval"]

    train_start = cfg["dates"]["train"][0]
    wf_end = cfg["dates"]["walkforward"][1]
    lb_start, lb_end = cfg["dates"]["lockbox"]

    print(f"Building working snapshot ({train_start}..{wf_end}) at {target}...")
    working = load_bars(raw_dir, base, target, start=train_start, end=wf_end)
    working_path = code_root / cfg["paths"]["working_parquet"]
    working.to_parquet(working_path)
    print(f"  wrote {len(working):,} bars -> {working_path}")

    print(f"Building lockbox snapshot ({lb_start}..{lb_end}, sealed) at {target}...")
    lockbox = load_bars(raw_dir, base, target, start=lb_start, end=lb_end)
    overlap = working.index.intersection(lockbox.index)
    if len(overlap):
        raise ValueError(
            f"working and lockbox snapshots share {len(overlap)} bar(s), "
            f"first {overlap[0]} — the sealed slice must not touch the working one"
        )
    lockbox_path = code_root / cfg["paths"]["lockbox_parquet"]
    lockbox.to_parquet(lockbox_path)
    print(f"  wrote {len(lockbox):,} bars -> {lockbox_path}")


if __name__ == "__main__":
    import sys

    default_cfg = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    build_snapshots(sys.argv[1] if len(sys.argv) > 1 else default_cfg)
