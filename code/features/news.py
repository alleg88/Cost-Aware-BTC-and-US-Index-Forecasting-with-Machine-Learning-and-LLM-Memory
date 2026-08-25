"""Turn scored headlines into leak-free per-bar sentiment features.

Step 3 of the pipeline: aggregate article-level sentiment (from sentiment/score.py) onto the
M15 bar grid. This is the leakage-critical step (papers/wiki/design/news-scoring-howto.md,
risk 1): a bar may only use news that is public by its own **close**.

Convention: a bar stamped at open time ``t`` covers ``[t, t+15min)`` and its close is at
``t+15min`` — the same instant its OHLCV (and therefore the price features and the decision)
are known. News published within ``[t, t+15min)`` is public by that close, so it is included
for bar ``t``; the label predicts bar ``t+1``, so there is no look-ahead.

Implementation buckets articles into 15-min bins, lays them on a *continuous* grid (so news
during market-closed gaps still accumulates and reaches the next open bar), computes trailing
windows + an exponentially-decayed sentiment on that grid, then aligns to the price bars.

Public API:
    load_scores(stream) -> DataFrame[seendate, sent]
    build_news_features(scores, bar_index, ...) -> DataFrame aligned to bar_index
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"
_FREQ = "15min"
_BARS_PER_HOUR = 4

NEWS_FEATURE_COLS = [
    "news_cnt_6h", "news_cnt_24h",      # article volume (attention proxy)
    "news_sent_6h", "news_sent_24h",    # mean sentiment over trailing windows
    "news_sent_decay",                  # EWM-decayed net sentiment (the main signal)
]


def load_scores(stream: str) -> pd.DataFrame:
    """Read the cached article-level scores for a stream: [seendate (UTC), sent].

    Echo-suppressed: only each story's FIRST appearance is kept (see dedup.first_seen_only),
    so syndicated reprints don't re-inject stale sentiment into the trailing windows.
    """
    from sentiment.dedup import first_seen_only

    path = RAW_DIR / f"scores_{stream}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run sentiment/score.py --stream {stream}")
    df = pd.read_parquet(path)[["seendate", "title", "title_norm", "sent"]].dropna(subset=["sent"])
    df["seendate"] = pd.to_datetime(df["seendate"], utc=True)
    df = first_seen_only(df)
    return df[["seendate", "sent"]].sort_values("seendate")


def build_news_features(
    scores: pd.DataFrame,
    bar_index: pd.DatetimeIndex,
    decay_halflife_hours: float = 6.0,
) -> pd.DataFrame:
    """Per-bar, leak-free news features aligned to ``bar_index`` (UTC bar open times).

    Each bar ``t`` sees only articles with ``seendate < t + 15min`` (its close). Bars with no
    news in a window get sentiment 0 (neutral) and count 0.
    """
    if bar_index.tz is None:
        raise ValueError("bar_index must be tz-aware (UTC)")

    # 1. bucket articles to 15-min bins (floor). A bin [b, b+15min) is complete at b+15min.
    binned = scores.copy()
    binned["bin"] = binned["seendate"].dt.floor(_FREQ)
    agg = binned.groupby("bin")["sent"].agg(sum_sent="sum", cnt="count")

    if agg.empty:
        return pd.DataFrame(0.0, index=bar_index, columns=NEWS_FEATURE_COLS)

    # 2. lay on a continuous grid spanning news + bars (accumulate across market-closed gaps).
    start = min(agg.index.min(), bar_index.min())
    end = max(agg.index.max(), bar_index.max())
    grid = pd.date_range(start, end, freq=_FREQ, tz="UTC")
    g = agg.reindex(grid, fill_value=0.0)

    # 3. trailing aggregates on the grid (right-closed → bin b included at index b = its close).
    w6, w24 = 6 * _BARS_PER_HOUR, 24 * _BARS_PER_HOUR
    cnt6 = g["cnt"].rolling(w6, min_periods=1).sum()
    cnt24 = g["cnt"].rolling(w24, min_periods=1).sum()
    sum6 = g["sum_sent"].rolling(w6, min_periods=1).sum()
    sum24 = g["sum_sent"].rolling(w24, min_periods=1).sum()
    # mean sentiment = summed sentiment / article count; 0 (neutral) when no articles.
    sent6 = np.where(cnt6 > 0, sum6 / cnt6.replace(0, np.nan), 0.0)
    sent24 = np.where(cnt24 > 0, sum24 / cnt24.replace(0, np.nan), 0.0)
    # EWM-decayed net sentiment over the per-bin summed sentiment (recent news weighs more).
    halflife_bars = decay_halflife_hours * _BARS_PER_HOUR
    decay = g["sum_sent"].ewm(halflife=halflife_bars, adjust=False).mean()

    feats = pd.DataFrame({
        "news_cnt_6h": cnt6, "news_cnt_24h": cnt24,
        "news_sent_6h": sent6, "news_sent_24h": sent24,
        "news_sent_decay": decay,
    }, index=grid)

    # 4. align to the price bars (bar t <- grid value at t, i.e. as of bar t's close).
    return feats.reindex(bar_index).fillna(0.0)[NEWS_FEATURE_COLS]
