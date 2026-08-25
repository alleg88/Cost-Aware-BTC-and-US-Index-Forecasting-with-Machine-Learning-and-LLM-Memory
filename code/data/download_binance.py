"""Download BTCUSDT market data from the Binance public bulk archive.

Source: https://data.binance.vision (free, no API key). Three datasets:

  klines       spot 15m (default), 1m, or 1s candles — monthly zips. 15m is a native
               interval, so bars are used directly (no aggregation; see load.py).
  metrics      USDⓈ-M futures positioning snapshots (open interest, top-trader
               and taker long/short ratios) — DAILY zips with 5-minute rows.
               The REST endpoint only serves the last 30 days; this archive is
               the only source of history. See build_positioning.py.
  fundingRate  USDⓈ-M perp funding events (one row / 8h) — monthly zips.

Every file is verified against its published SHA256 .CHECKSUM and unzipped to
data/raw/binance/ (futures data under futures/). Idempotent: existing,
checksum-valid files are skipped.

Run:  python code/data/download_binance.py
      python code/data/download_binance.py --interval 1s
      python code/data/download_binance.py --dataset metrics
      python code/data/download_binance.py --dataset fundingRate
"""
from __future__ import annotations

import hashlib
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"
SYMBOL = "BTCUSDT"
INTERVAL = "15m"

# Study period (2024-01 .. 2025-12) + out-of-time lockbox (2026-01 .. 2026-03).
MONTHS = [
    f"{year}-{month:02d}"
    for year in (2024, 2025)
    for month in range(1, 13)
] + [f"2026-{month:02d}" for month in range(1, 4)]

# 1-minute klines are used only as the intrabar EXIT engine for bracket trades
# (evaluation.trades_intrabar), so the default 1m pull covers the 2025 walk-forward
# year only (Q1 calibration + Q2-Q4 evaluation). Signals stay on M15.
MONTHS_1M = [f"2025-{month:02d}" for month in range(1, 13)]

# Futures positioning data (metrics + fundingRate) covers the full study span
# incl. the lockbox quarter: features must exist wherever bars exist.
FUTURES_MONTHS = MONTHS + [f"2026-{month:02d}" for month in range(4, 7)]
FUTURES_BASE = "https://data.binance.vision/data/futures/um"

RAW_DIR = Path(__file__).resolve().parent / "raw" / "binance"
FUT_DIR = RAW_DIR / "futures"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> bool:
    """Stream a URL to dest. Returns False on 404 (month not yet published)."""
    resp = requests.get(url, stream=True, timeout=60)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
    ) as bar:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))
    tmp.replace(dest)
    return True


def _expected_sha(checksum_path: Path) -> str:
    # Binance .CHECKSUM format:  "<sha256>  <filename>"
    return checksum_path.read_text().split()[0]


def fetch_month(month: str, interval: str = INTERVAL) -> str:
    """Download + verify + unzip one month. Returns a short status string."""
    fname = f"{SYMBOL}-{interval}-{month}.zip"
    url = f"{BASE_URL}/{SYMBOL}/{interval}/{fname}"
    zip_path = RAW_DIR / fname
    csv_name = fname.replace(".zip", ".csv")
    csv_path = RAW_DIR / csv_name

    if csv_path.exists():
        return f"skip (csv exists): {month}"

    # Checksum first so the zip can be verified.
    csum_path = RAW_DIR / (fname + ".CHECKSUM")
    if not csum_path.exists():
        if not _download(url + ".CHECKSUM", csum_path):
            return f"not published yet: {month}"

    if not zip_path.exists():
        if not _download(url, zip_path):
            return f"not published yet: {month}"

    expected = _expected_sha(csum_path)
    actual = _sha256(zip_path)
    if actual != expected:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {fname}: {actual} != {expected}")

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(RAW_DIR)

    return f"ok: {month}"


def _fetch_verified(url: str, dest_dir: Path, fname: str) -> str:
    """Download fname (+ .CHECKSUM), verify, unzip into dest_dir. Idempotent."""
    zip_path = dest_dir / fname
    csv_path = dest_dir / fname.replace(".zip", ".csv")
    if csv_path.exists():
        return f"skip (csv exists): {fname}"
    csum_path = dest_dir / (fname + ".CHECKSUM")
    if not csum_path.exists() and not _download(url + ".CHECKSUM", csum_path):
        return f"not published yet: {fname}"
    if not zip_path.exists() and not _download(url, zip_path):
        return f"not published yet: {fname}"
    expected = _expected_sha(csum_path)
    actual = _sha256(zip_path)
    if actual != expected:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(f"checksum mismatch for {fname}: {actual} != {expected}")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest_dir)
    return f"ok: {fname}"


def _month_days(month: str) -> list[str]:
    """All YYYY-MM-DD dates of one YYYY-MM month."""
    import calendar
    y, m = int(month[:4]), int(month[5:7])
    return [f"{month}-{d:02d}" for d in range(1, calendar.monthrange(y, m)[1] + 1)]


def fetch_metrics_day(day: str) -> str:
    """One daily futures-metrics file (5-minute positioning snapshots)."""
    fname = f"{SYMBOL}-metrics-{day}.zip"
    url = f"{FUTURES_BASE}/daily/metrics/{SYMBOL}/{fname}"
    return _fetch_verified(url, FUT_DIR / "metrics", fname)


def fetch_funding_month(month: str) -> str:
    """One monthly funding-rate file (8h funding events)."""
    fname = f"{SYMBOL}-fundingRate-{month}.zip"
    url = f"{FUTURES_BASE}/monthly/fundingRate/{SYMBOL}/{fname}"
    return _fetch_verified(url, FUT_DIR / "fundingRate", fname)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="klines",
                    choices=["klines", "metrics", "fundingRate"],
                    help="klines (spot candles), metrics (futures positioning, "
                         "daily 5m snapshots), or fundingRate (8h funding events)")
    ap.add_argument("--interval", default=INTERVAL,
                    help="Binance kline interval (klines only; default 15m, "
                         "use 1m or 1s for the intrabar exit engine)")
    ap.add_argument("--months", default=None,
                    help="comma-separated YYYY-MM list (default: full study period "
                         "for 15m/1s/metrics/funding, the 2025 walk-forward year for 1m)")
    args = ap.parse_args()

    if args.months:
        months = [m.strip() for m in args.months.split(",")]
    elif args.dataset != "klines":
        months = FUTURES_MONTHS
    elif args.interval == "1m":
        months = MONTHS_1M
    else:
        months = MONTHS

    if args.dataset == "metrics":
        days = [d for m in months for d in _month_days(m)]
        (FUT_DIR / "metrics").mkdir(parents=True, exist_ok=True)
        print(f"Downloading {SYMBOL} futures metrics ({len(days)} days) -> "
              f"{FUT_DIR / 'metrics'}")
        results = [fetch_metrics_day(d) for d in days]
        ok = sum(r.startswith(("ok", "skip")) for r in results)
        print(f"Done. {ok}/{len(days)} daily metrics files present.")
        for r in results:
            if "not published" in r:
                print("  -", r)
        return 0

    if args.dataset == "fundingRate":
        (FUT_DIR / "fundingRate").mkdir(parents=True, exist_ok=True)
        print(f"Downloading {SYMBOL} funding rates ({len(months)} months) -> "
              f"{FUT_DIR / 'fundingRate'}")
        for month in months:
            print(" ", fetch_funding_month(month))
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {SYMBOL} {args.interval} ({len(months)} months) -> {RAW_DIR}")
    results = []
    for month in months:
        status = fetch_month(month, args.interval)
        results.append(status)
        print(" ", status)

    csvs = sorted(RAW_DIR.glob(f"{SYMBOL}-{args.interval}-*.csv"))
    print(f"\nDone. {len(csvs)} monthly {args.interval} CSV file(s) present in {RAW_DIR}.")
    missing = [r for r in results if "not published" in r]
    if missing:
        print(f"NOTE: {len(missing)} month(s) not yet published by Binance:")
        for m in missing:
            print("  -", m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
