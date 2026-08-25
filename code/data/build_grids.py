"""Snapshot the derived bar grids used by the channel study to parquet.

The channel study reads several grids at once: an execution grid (5m/15m/30m) and
a channel grid (1h/2h/4h), all of which are ablation dimensions. Deriving them from
the 1-minute snapshot on every run costs minutes per script, so they are built once
here and read directly afterwards.

Aggregation is exact: over the 2024-2025 span where native Binance 15m bars are also
available, the 1m -> 15m aggregate reproduces all 78,817 bars to the cent, with volume
agreeing to machine epsilon. The derived grids are therefore the same object a native
download would give, not an approximation of it.

Bars are stamped by OPEN time and contain [t, t+interval), matching load.py.

Run:  python code/data/build_grids.py
      python code/data/build_grids.py --grids 15min,1h
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent
SOURCE = DATA_DIR / "btcusdt_1m_2021_2026.parquet"
GRIDS = ("5min", "15min", "30min", "1h", "2h", "4h")

# Order-flow columns are additive over the bar; OHLC use first/max/min/last.
_ORDERFLOW = ["quote_volume", "count", "taker_buy_base", "taker_buy_quote"]


def build_grid(bars1m: pd.DataFrame, grid: str) -> pd.DataFrame:
    spec = dict(
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    )
    for col in _ORDERFLOW:
        if col in bars1m.columns:
            spec[col] = (col, "sum")
    out = bars1m.resample(grid, label="left", closed="left").agg(**spec)
    # Minutes actually present in each bar. The 1m source has 1,073 missing minutes
    # in seven gaps, so a handful of bars are built from fewer minutes than they
    # should be. Without this column such a bar is indistinguishable from a complete
    # one, and a channel window containing it looks clean when it is not.
    out["minute_count"] = bars1m["close"].resample(grid, label="left", closed="left").count()
    out = out.dropna(subset=["open", "high", "low", "close"])
    if not out.index.is_unique or not out.index.is_monotonic_increasing:
        raise ValueError(f"{grid}: index must be unique and sorted")
    if (out["high"] < out["low"]).any():
        raise ValueError(f"{grid}: high < low")
    if (out["high"] < out[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{grid}: high below open/close")
    if (out["low"] > out[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{grid}: low above open/close")
    return out


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--grids", default=",".join(GRIDS),
                    help="comma-separated pandas offset aliases")
    args = ap.parse_args()

    sym = args.symbol.lower()
    source = args.source or (DATA_DIR / f"{sym}_1m_2021_2026.parquet")
    bars1m = pd.read_parquet(source)
    print(f"source: {len(bars1m):,} 1m bars  "
          f"{bars1m.index.min()} .. {bars1m.index.max()}")
    for grid in (g.strip() for g in args.grids.split(",")):
        frame = build_grid(bars1m, grid)
        out = DATA_DIR / f"{sym}_{grid}_2021_2026.parquet"
        frame.to_parquet(out)
        print(f"  {grid:>6}: {len(frame):>9,} bars -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
