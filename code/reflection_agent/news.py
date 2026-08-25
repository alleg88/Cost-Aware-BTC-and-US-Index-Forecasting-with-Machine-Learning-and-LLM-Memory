"""Causal sanitization and balanced selection of news/event context."""
from __future__ import annotations

import hashlib
import html
import re
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

from reflection_agent.contracts import NewsContext, NewsItem

SOURCE_FAMILIES = ("gdelt_news", "direct_policy_event", "fred_macro", "fear_greed")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML_TAG = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+", flags=re.IGNORECASE)
_WHITESPACE = re.compile(r"\s+")


def sanitize_text(value: object, *, max_characters: int = 500) -> str:
    text = html.unescape(str(value or ""))
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _CONTROL.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()[:max_characters]


def stable_event_id(source_family: str, available_at: object, summary: object) -> str:
    payload = f"{source_family}|{pd.Timestamp(available_at).isoformat()}|{sanitize_text(summary)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def normalize_events(
    frame: pd.DataFrame,
    *,
    source_family: Literal["gdelt_news", "direct_policy_event", "fred_macro", "fear_greed"],
    available_column: str,
    summary_column: str,
    publisher_category: str,
    impact_column: str | None = None,
    sentiment_column: str | None = None,
    max_summary_characters: int = 500,
) -> pd.DataFrame:
    required = {available_column, summary_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"event frame misses columns: {sorted(missing)}")
    out = pd.DataFrame(index=frame.index)
    out["available_at_utc"] = pd.to_datetime(frame[available_column], utc=True, errors="raise")
    out["source_family"] = source_family
    out["publisher_category"] = publisher_category
    out["summary"] = frame[summary_column].map(
        lambda value: sanitize_text(value, max_characters=max_summary_characters)
    )
    if impact_column and impact_column in frame:
        out["impact"] = pd.to_numeric(frame[impact_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    else:
        out["impact"] = 0.0
    if sentiment_column and sentiment_column in frame:
        out["sentiment"] = pd.to_numeric(frame[sentiment_column], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    else:
        out["sentiment"] = 0.0
    out["event_id"] = [
        stable_event_id(source_family, available_at, summary)
        for available_at, summary in zip(out["available_at_utc"], out["summary"], strict=True)
    ]
    return out.reset_index(drop=True)


def select_balanced_events(
    events: pd.DataFrame,
    *,
    cutoff_utc: datetime,
    lookback: timedelta = timedelta(days=7),
    max_items: int = 12,
    max_per_source_family: int = 3,
) -> NewsContext:
    required = {
        "event_id", "available_at_utc", "source_family", "publisher_category",
        "summary", "impact", "sentiment",
    }
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"normalized events miss columns: {sorted(missing)}")
    cutoff = pd.Timestamp(cutoff_utc)
    if cutoff.tzinfo is None:
        raise ValueError("cutoff_utc must be timezone-aware")
    available = pd.to_datetime(events["available_at_utc"], utc=True, errors="raise")
    eligible = events.loc[(available <= cutoff) & (available > cutoff - lookback)].copy()
    eligible["available_at_utc"] = available.loc[eligible.index]
    unknown = set(eligible["source_family"]) - set(SOURCE_FAMILIES)
    if unknown:
        raise ValueError(f"unknown source families: {sorted(unknown)}")
    source_counts = {family: int((eligible["source_family"] == family).sum()) for family in SOURCE_FAMILIES}
    eligible = eligible.sort_values(
        ["impact", "available_at_utc", "event_id"], ascending=[False, False, True], kind="mergesort"
    )
    balanced = eligible.groupby("source_family", sort=True, group_keys=False).head(max_per_source_family)
    balanced = balanced.sort_values(
        ["impact", "available_at_utc", "event_id"], ascending=[False, False, True], kind="mergesort"
    ).head(max_items)
    items = [NewsItem.model_validate(row) for row in balanced[list(required)].to_dict("records")]
    sentiments = eligible["sentiment"].astype(float)
    aggregate_features = {
        "event_count": float(len(eligible)),
        "source_diversity": float(sum(count > 0 for count in source_counts.values())),
        "mean_sentiment": float(sentiments.mean()) if len(sentiments) else 0.0,
        "news_dispersion": float(sentiments.std(ddof=0)) if len(sentiments) else 0.0,
        "max_impact": float(eligible["impact"].max()) if len(eligible) else 0.0,
    }
    return NewsContext(source_counts=source_counts, aggregate_features=aggregate_features, top_items=items)

