"""Collect the Crypto Fear & Greed index (alternative.me) — daily 0-100 sentiment.

Free, no API key. The API returns the full history in one request; this keeps only the
**study window** (config dates, plus a short warm-up buffer for rolling features) and
snapshots it to parquet with a UTC daily index. Downstream forward-fills onto the bar grid.

Run:  python code/sentiment/fear_greed.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests
import yaml

URL = "https://api.alternative.me/fng/"
CODE_ROOT = Path(__file__).resolve().parents[1]
OUT = CODE_ROOT / "sentiment" / "raw" / "crypto_fear_greed.parquet"
WARMUP_DAYS = 60   # keep a little history before the train start for rolling features


def fetch() -> pd.DataFrame:
    """Return the full Fear & Greed history as a UTC-indexed DataFrame."""
    resp = requests.get(URL, params={"limit": 0, "format": "json"}, timeout=30)
    resp.raise_for_status()
    data = resp.json()["data"]
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
    df["value"] = df["value"].astype(int)
    df = (
        df.rename(columns={"value_classification": "classification"})
          [["timestamp", "value", "classification"]]
          .set_index("timestamp")
          .sort_index()
    )
    return df


def _window() -> tuple[pd.Timestamp, pd.Timestamp]:
    cfg = yaml.safe_load((CODE_ROOT / "configs" / "default.yaml").read_text())
    start = pd.Timestamp(cfg["dates"]["train"][0], tz="UTC") - pd.Timedelta(days=WARMUP_DAYS)
    end = pd.Timestamp(cfg["dates"]["lockbox"][1], tz="UTC") + pd.Timedelta(days=1)
    return start, end


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df = fetch()
    start, end = _window()
    df = df[(df.index >= start) & (df.index <= end)]
    df.to_parquet(OUT)
    print(f"wrote {len(df):,} daily values "
          f"({df.index[0].date()} -> {df.index[-1].date()}) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
