"""Snapshot the 1-minute BTCUSDT klines to parquet for the intrabar exit engine.

The 1-minute bars are used only by evaluation.trades_intrabar to resolve bracket
exits inside each M15 signal bar; signals stay on M15. This mirrors the M15
snapshot step in load.py but for the 1m base interval, writing a single parquet
spanning every 1m CSV currently in data/raw/binance (2025 + 2026).

Run:  python code/data/build_1m.py
      python code/data/build_1m.py --out code/data/btcusdt_1m_2024_2026.parquet
"""
from __future__ import annotations

from pathlib import Path

try:
    from .load import load_bars
except ImportError:  # direct script execution: python code/data/build_1m.py
    from load import load_bars

RAW_DIR = Path(__file__).resolve().parent / "raw" / "binance"
OUT = Path(__file__).resolve().parent / "btcusdt_1m_2025_2026.parquet"


def build_snapshot(raw_dir: Path, out: Path, symbol: str = "BTCUSDT"):
    """Build one sorted, unique UTC 1m parquet from downloaded monthly CSVs."""
    bars = load_bars(raw_dir, base_interval="1m", target_interval="1min", symbol=symbol)
    if not bars.index.is_unique or not bars.index.is_monotonic_increasing:
        raise ValueError("1m snapshot index must be unique and sorted")
    if bars.index.tz is None:
        raise ValueError("1m snapshot index must be timezone-aware")
    out.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(out)
    return bars


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    out = args.out or (RAW_DIR.parent / f"{args.symbol.lower()}_1m_2021_2026.parquet")
    bars = build_snapshot(RAW_DIR, out, args.symbol)
    args.out = out
    print(f"wrote {len(bars):,} 1m bars -> {args.out}")
    print(f"span: {bars.index.min()} .. {bars.index.max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
