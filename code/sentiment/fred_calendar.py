"""Collect US macro releases from the FRED API — actual values + first-release timestamps.

FRED gives the *initial release* value of each series (output_type=4) together with the
date it was first published (the observation's realtime_start). That release date — not the
reference period — is what the market reacts to, so it is the event time used for the
leak-free as-of join later. FRED does not provide consensus/forecast, so "surprise" is left
to a later phase; this collects actuals + release timestamps (the strongest index signal).

Release clock time is not in FRED, so each series is stamped at its well-known US release
time (mostly 08:30 America/New_York) and converted to UTC.

API key: set the FRED_API_KEY environment variable, or put the key on one line in
code/sentiment/secrets/fred_key.txt (gitignored). Get a free key at
https://fredaccount.stlouisfed.org/apikeys

Run:  python code/sentiment/fred_calendar.py
"""
from __future__ import annotations

import os
from datetime import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.stlouisfed.org/fred/series/observations"
CODE_ROOT = Path(__file__).resolve().parents[1]
KEY_FILE = CODE_ROOT / "sentiment" / "secrets" / "fred_key.txt"
OUT = CODE_ROOT / "sentiment" / "raw" / "fred_calendar.parquet"

# series_id -> (label, US release time of day, ET)
SERIES = {
    "CPIAUCSL": ("CPI", time(8, 30)),
    "CPILFESL": ("Core CPI", time(8, 30)),
    "PAYEMS":   ("Nonfarm Payrolls", time(8, 30)),
    "UNRATE":   ("Unemployment Rate", time(8, 30)),
    "ICSA":     ("Initial Jobless Claims", time(8, 30)),
    "PCEPI":    ("PCE Price Index", time(8, 30)),
    "PCEPILFE": ("Core PCE", time(8, 30)),
    "RSAFS":    ("Retail Sales", time(8, 30)),
    "INDPRO":   ("Industrial Production", time(9, 15)),
    "GDPC1":    ("Real GDP", time(8, 30)),
}

# Pull releases that occurred in the study window; reach back a little so a Jan-2024
# release about Nov/Dec-2023 data is captured.
REALTIME_START = "2024-01-01"
REALTIME_END = "2026-04-01"
OBSERVATION_START = "2023-09-01"


def _api_key() -> str:
    key = os.environ.get("FRED_API_KEY")
    if key:
        return key.strip()
    if KEY_FILE.exists():
        return KEY_FILE.read_text().strip()
    raise SystemExit(
        f"No FRED key. Set FRED_API_KEY or put the key in {KEY_FILE}\n"
        "Get one free at https://fredaccount.stlouisfed.org/apikeys"
    )


def fetch_series(series_id: str, label: str, rel_time: time, key: str) -> pd.DataFrame:
    """Initial-release observations for one series, stamped at the US release time (UTC)."""
    params = {
        "series_id": series_id, "api_key": key, "file_type": "json",
        "output_type": 4,                       # initial release only
        "realtime_start": REALTIME_START, "realtime_end": REALTIME_END,
        "observation_start": OBSERVATION_START, "observation_end": REALTIME_END,
    }
    resp = requests.get(BASE, params=params, timeout=60)
    resp.raise_for_status()
    obs = resp.json().get("observations", [])
    rows = []
    for o in obs:
        if o.get("value") in (".", "", None):
            continue
        # realtime_start = the date this value was first published (release date)
        rel_dt = pd.Timestamp(f"{o['realtime_start']} {rel_time}", tz="America/New_York")
        rows.append({
            "release_time": rel_dt.tz_convert("UTC"),
            "series": series_id,
            "label": label,
            "ref_date": pd.Timestamp(o["date"]),
            "value": float(o["value"]),
        })
    return pd.DataFrame(rows)


def main() -> int:
    key = _api_key()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for sid, (label, rel_time) in SERIES.items():
        df = fetch_series(sid, label, rel_time, key)
        print(f"  {sid:10s} {label:24s}: {len(df):3d} releases")
        frames.append(df)
    cal = pd.concat(frames, ignore_index=True).sort_values("release_time")
    cal.to_parquet(OUT)
    print(f"\nwrote {len(cal):,} release events across {len(SERIES)} series "
          f"({cal['release_time'].min().date()} -> {cal['release_time'].max().date()}) -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
