"""Load GDELT GKG exports from BigQuery into clean per-stream parquet.

The GKG dataset is pulled via the free BigQuery sandbox (no per-request throttle, unlike
the DOC API). Drop the exported CSVs in code/sentiment/raw/gdelt_bq/. Two naming schemes both work:
  * per-stream files  -> btc_2024.csv, usmarket_2025.csv   (stream taken from the filename)
  * combined files    -> gdelt_2024.csv with a `stream` column
This script normalises them: parses the GKG DATE to a UTC timestamp, extracts the tone
score from V2Tone, dedupes by URL, and writes one parquet per stream.

Expected CSV columns: DATE, url, domain, V2Tone, V2Themes (+ optional `stream`).
V2Tone is comma-separated; its first field is the article tone (avg sentiment).

Run:  python code/sentiment/gdelt_bigquery.py
"""
from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = CODE_ROOT / "sentiment" / "raw" / "gdelt_bq"
OUT_DIR = CODE_ROOT / "sentiment" / "raw"


def _tone(v2tone: pd.Series) -> pd.Series:
    """First field of the comma-separated V2Tone string = article tone (avg sentiment)."""
    return pd.to_numeric(v2tone.astype(str).str.split(",").str[0], errors="coerce")


# Tech-index (NASDAQ-100) classifier. The BigQuery CASE tag for usatech matched only a few
# URL substrings, so most tech headlines fell into the broad usa500 bucket. This widens the
# tech stream by matching the headline TEXT (now exported) against index/theme terms and the
# megacap constituents. usatech is an OVERLAPPING subset of the market stream — the same
# megacaps are top S&P 500 holdings, so these articles legitimately belong to usa500 too and
# are never removed from it. Non-capturing groups; case-insensitive applied at call site.
_TECH_RE = re.compile(
    r"nasdaq|tech stock|big tech|magnificent (?:seven|7)|semiconductor|chipmaker|chip stock|"
    r"ai stock|cloud computing|software stock|"
    r"\b(?:apple|microsoft|nvidia|amazon|alphabet|google|meta platforms|tesla|broadcom|"
    r"netflix|adobe|qualcomm|intel|cisco|oracle|salesforce|palantir|micron|asml|tsmc|"
    r"advanced micro|amd|arm holdings)\b"
)


def _is_tech(df: pd.DataFrame) -> pd.Series:
    """True where the headline (or, as fallback, the URL) is NASDAQ-100/tech relevant."""
    text = (df["title"].fillna("").astype(str) + " " + df["url"].fillna("").astype(str)).str.lower()
    return text.str.contains(_TECH_RE)


def _read_members(path: Path) -> list[pd.DataFrame]:
    """Read CSV(s) from a .zip (BigQuery Drive export), .csv, or .csv.gz file."""
    frames = []
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                low = name.lower()
                if low.endswith(".csv"):
                    with z.open(name) as fh:
                        frames.append(pd.read_csv(fh))
                elif low.endswith(".gz"):
                    with z.open(name) as fh:
                        frames.append(pd.read_csv(fh, compression="gzip"))
    elif path.suffix == ".csv":
        frames.append(pd.read_csv(path))
    elif path.suffix == ".gz":
        frames.append(pd.read_csv(path, compression="gzip"))
    return frames


def load_all() -> pd.DataFrame | None:
    """Concatenate + normalise all exports (.zip/.csv/.gz). Index files carry a `stream`
    column; Bitcoin files do not, so any stream-less file is tagged 'btc'."""
    files = sorted(p for p in IN_DIR.iterdir() if p.suffix in (".zip", ".csv", ".gz"))
    if not files:
        print(f"  no exports in {IN_DIR} — drop the BigQuery results there first")
        return None
    parts = []
    for f in files:
        for d in _read_members(f):
            if "stream" not in d.columns:
                d["stream"] = "btc"
            parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["seendate"] = pd.to_datetime(df["DATE"].astype("int64").astype(str),
                                    format="%Y%m%d%H%M%S", utc=True)
    df["tone"] = _tone(df["V2Tone"])
    # title is the headline DeBERTa/Gemma score; HTML-unescape entity-encoded text.
    if "title" in df.columns:
        df["title"] = df["title"].map(lambda x: html.unescape(x) if isinstance(x, str) else x)
    else:
        df["title"] = pd.NA
    # V2Themes is dropped from the export to keep file sizes exportable; keep it if present.
    df["themes"] = df["V2Themes"] if "V2Themes" in df.columns else pd.NA
    return df


def _write(df: pd.DataFrame, stream: str) -> None:
    df = (
        df[["seendate", "url", "domain", "tone", "title", "themes"]]
        .dropna(subset=["url"])
        .drop_duplicates("url")
        .sort_values("seendate")
        .reset_index(drop=True)
    )
    out = OUT_DIR / f"gdelt_{stream}.parquet"
    df.to_parquet(out)
    print(f"  [{stream}] {len(df):,} unique articles "
          f"({df['seendate'].min().date()} -> {df['seendate'].max().date()}) -> {out}")


def main() -> int:
    IN_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_all()
    if raw is None:
        return 0
    # Three feature streams, derived from content (not the raw BigQuery CASE tag):
    #   btc      = crypto articles (the BQ btc tag is keyword-clean, kept as-is)
    #   usa500   = the full non-crypto US-market base (everything not btc)
    #   usatech  = the tech-relevant SUBSET of usa500 (overlaps it; see _is_tech)
    btc = raw[raw["stream"] == "btc"]
    market = raw[raw["stream"] != "btc"]
    usatech = market[_is_tech(market)]
    _write(btc, "btc")
    _write(market, "usa500")
    _write(usatech, "usatech")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
