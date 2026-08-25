"""Prepare @realDonaldTrump Truth Social posts for financial-market sentiment scoring.

This script fetches Trump's Truth Social posts from Hugging Face, filters them to the
study and lockbox period (2024-01-01 to 2026-03-31) and to posts relevant to financial
markets, and writes them to sentiment/raw/trump_truth_posts.csv.
"""
from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

# Define paths
CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"
OUT_CSV = RAW_DIR / "trump_truth_posts.csv"

# Config window
START_DATE = "2024-01-01 00:00:00+00:00"
END_DATE = "2026-03-31 23:59:59+00:00"

# Refined filters for financial-market relevance
TARIFF_RE = re.compile(r"\b(tariff|tariffs|china|trade war)\b", re.I)

# Fed/Reserve: Match fed, federal reserve, powell, interest rates.
# Exclude lowercase 'fed' when part of 'fed up' or used as a verb (e.g. 'he fed').
FED_RAW_RE = re.compile(r"\b(fed|federal reserve|powell|interest rates?)\b", re.I)
FED_EXCLUDE_RE = re.compile(r"\b(fed\s+up)\b|\b(he|she|they|we|i|who|that)\s+fed\b", re.I)

def matches_fed(text: str) -> bool:
    for m in FED_RAW_RE.finditer(text):
        matched_word = m.group(1).lower()
        if matched_word == 'fed':
            # Check context around the 'fed' match to filter out 'fed up' / 'he fed' verbs
            start_idx = max(0, m.start() - 10)
            end_idx = min(len(text), m.end() + 5)
            snippet = text[start_idx:end_idx]
            if FED_EXCLUDE_RE.search(snippet):
                continue
            return True
        else:
            return True
    return False

# Inflation & Jobs: Exclude 'Steve Jobs' unless inflation/unemployment or other jobs are mentioned.
INFLATION_RE = re.compile(r"\b(inflation|jobs|unemployment|cpi)\b", re.I)
JOBS_EXCLUDE_RE = re.compile(r"\bsteve\s+jobs\b", re.I)

def matches_jobs(text: str) -> bool:
    if INFLATION_RE.search(text):
        if JOBS_EXCLUDE_RE.search(text):
            jobs_count = len(re.findall(r"\bjobs\b", text, re.I))
            steve_jobs_count = len(re.findall(r"\bsteve\s+jobs\b", text, re.I))
            other_inflation = bool(re.search(r"\b(inflation|unemployment|cpi)\b", text, re.I))
            if jobs_count > steve_jobs_count or other_inflation:
                return True
            return False
        return True
    return False

CRYPTO_RE = re.compile(r"\b(bitcoin|btc|crypto|digital asset|digital assets|bitcoin reserve)\b", re.I)
TREASURY_RE = re.compile(r"\b(treasury|dollar|debt|deficit|sanctions)\b", re.I)

# Rates: Exclude non-market contexts like third-rate, crime rate, etc.
RATE_RAW_RE = re.compile(r"\brates?\b", re.I)
EXCLUDE_RATE_RE = re.compile(
    r"\b(third[-|\s]rate|3rd[-|\s]rate|second[-|\s]rate|2nd[-|\s]rate|first[-|\s]rate|1st[-|\s]rate"
    r"|murder[-|\s]rate|crime[-|\s]rate|homicide[-|\s]rate|success[-|\s]rate|approval[-|\s]rate|death[-|\s]rate|ratings?)\b"
    r"|\bat\s+a\s+(\w+\s+)?rate\b"
    r"|\bat\s+any\s+rate\b"
    r"|\bwhich\s+rate\b",
    re.I
)

def matches_rates(text: str) -> bool:
    if RATE_RAW_RE.search(text):
        if not EXCLUDE_RATE_RE.search(text):
            return True
    return False

# Oil/Energy: Must contain oil/energy AND at least one market/price/economic keyword
OIL_ENERGY_RAW_RE = re.compile(r"\b(oil|energy)\b", re.I)
MARKET_KEYWORDS = [
    "price", "prices", "cost", "costs", "market", "inflation", "gdp", "production", 
    "refinery", "refining", "reserves", "spr", "pipeline", "drilling", "drill", 
    "unleash", "supply", "gasoline", "gas", "fuel", "tariffs", "tariff", "trade", 
    "dollar", "tax", "taxes", "strait", "hormuz", "liquid gold", "refineries"
]
MARKET_KEYWORDS_RE = re.compile(r"\b(" + "|".join(MARKET_KEYWORDS) + r")\b", re.I)

def is_market_moving_oil_energy(text: str) -> bool:
    if not OIL_ENERGY_RAW_RE.search(text):
        return False
    return bool(MARKET_KEYWORDS_RE.search(text))

def matches_financial_market(text: str) -> bool:
    # 1. Tariffs & China
    if TARIFF_RE.search(text):
        return True
    # 2. Fed, Federal Reserve, Powell, interest rates
    if matches_fed(text):
        return True
    # Check general "rates" excluding non-market ones
    if matches_rates(text):
        return True
    # 3. Inflation & Jobs
    if matches_jobs(text):
        return True
    # 4. Crypto
    if CRYPTO_RE.search(text):
        return True
    # 5. Treasury & Deficit & Dollar & Sanctions
    if TREASURY_RE.search(text):
        return True
    # 6. Oil/Energy (if market-moving)
    if is_market_moving_oil_energy(text):
        return True
    return False

def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    url = "https://huggingface.co/datasets/chrissoria/trump-truth-social/resolve/main/data/train-00000-of-00001.parquet"
    print(f"Fetching Truth Social posts from Hugging Face: {url}...")
    df = pd.read_parquet(url, columns=["datetime", "text", "url"])
    
    print("Parsing timestamps...")
    df["dt"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["dt"])
    
    print(f"Filtering to date window: {START_DATE} -> {END_DATE}...")
    window_df = df[(df["dt"] >= START_DATE) & (df["dt"] <= END_DATE)].copy()
    print(f"Total posts in window: {len(window_df)}")
    
    print("Filtering for financial-market relevant posts...")
    window_df["is_relevant"] = window_df["text"].apply(matches_financial_market)
    filtered_df = window_df[window_df["is_relevant"]].copy()
    
    # Sort chronologically
    filtered_df = filtered_df.sort_values("dt")
    
    # Format created_at to strict UTC ISO-8601 string: YYYY-MM-DDTHH:MM:SSZ
    # Convert dt to UTC timezone first
    filtered_df["created_at"] = filtered_df["dt"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Output columns: created_at, url, content
    # content is plain text
    filtered_df["content"] = filtered_df["text"]
    out_df = filtered_df[["created_at", "url", "content"]]
    
    # Save to CSV
    out_df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Successfully wrote {len(out_df):,} relevant posts to {OUT_CSV}")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
