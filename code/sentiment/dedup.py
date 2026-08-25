"""Causal near-duplicate handling for news sentiment.

Feature aggregation keeps the first story occurrence inside a fixed 72-hour window.
Automatic fuzzy removal requires 95% similarity; 90-95% matches are retained and
labelled for audit. Decisions use only already-observed stories, so appending future
rows cannot change the treatment of earlier rows.
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import re
from urllib.parse import unquote, urlsplit, urlunsplit

import pandas as pd
from rapidfuzz import fuzz


_SUFFIX = re.compile(r"\s*[-\u2013\u2014|]\s*[a-z0-9&.'\u2019 ]{2,35}$")
_BYLINE = re.compile(r"\s+by\s+[a-z0-9&.'\u2019 ]{2,30}$")
_PUNCT = re.compile(r"[^a-z0-9 ]+")
_WS = re.compile(r"\s+")
_PROTECTED = re.compile(
    r"(?<![a-z])(?:[$\u00a3\u20ac]?\d+(?:[.,]\d+)?(?:%|[kmbt])?|"
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
    r"january|february|march|april|may|june|july|august|september|"
    r"october|november|december)(?![a-z])"
)

DEDUP_WINDOW_HOURS = 72
FUZZ_THRESHOLD = 95
REVIEW_THRESHOLD = 90
_FUZZ_THRESHOLD = FUZZ_THRESHOLD
_MAX_BLOCK = 400


def strip_outlet(norm_title: str) -> str:
    """Reduce a normalised title to a syndication-robust comparison key."""
    title = norm_title
    for pattern in (_SUFFIX, _BYLINE, _SUFFIX):
        match = pattern.search(title)
        if match and len(title[: match.start()].split()) >= 4:
            title = title[: match.start()]
    title = _PUNCT.sub(" ", title)
    return _WS.sub(" ", title).strip()


def _protected_tokens(title: str) -> tuple[str, ...]:
    """Numbers and calendar words that fuzzy matching must preserve."""
    return tuple(sorted(_PROTECTED.findall(title.lower())))


def _canonical_url(value: object) -> str:
    """Remove query/fragment tracking while preserving host and article path."""
    raw = str(value or "").strip()
    if not raw or raw.lower() == "nan":
        return ""
    parsed = urlsplit(raw if "://" in raw else f"https://{raw}")
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    path = unquote(parsed.path or "/").rstrip("/") or "/"
    return urlunsplit(("https", host, path, "", "")) if host else ""


def _domain(canonical_url: str) -> str:
    return urlsplit(canonical_url).hostname or ""


def _stable_story_id(timestamp: pd.Timestamp, canonical_url: str, title_key: str) -> str:
    identity = f"{timestamp.isoformat()}|{canonical_url}|{title_key}"
    return "story_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _fuzzy_merge(keys: list[str]) -> dict[str, str]:
    """Scoring-cache helper: precision-first global title clustering."""
    blocks: dict[str, list[str]] = {}
    for key in keys:
        prefix = " ".join(key.split()[:2])
        blocks.setdefault(prefix, []).append(key)

    remap: dict[str, str] = {}
    for members in blocks.values():
        if len(members) < 2 or len(members) > _MAX_BLOCK:
            continue
        canonical: list[str] = []
        for key in sorted(members, key=len):
            for candidate in canonical:
                if (
                    _protected_tokens(key) == _protected_tokens(candidate)
                    and fuzz.token_sort_ratio(key, candidate) >= FUZZ_THRESHOLD
                ):
                    remap[key] = candidate
                    break
            else:
                canonical.append(key)
    return remap


def add_cluster_keys(unique: pd.DataFrame) -> pd.DataFrame:
    """Add precision-first cluster keys used only by the scoring cache."""
    out = unique.copy()
    out["cluster_key"] = out["title_norm"].map(strip_outlet)
    out.loc[out["cluster_key"] == "", "cluster_key"] = out["title_norm"]
    remap = _fuzzy_merge(out["cluster_key"].unique().tolist())
    if remap:
        out["cluster_key"] = out["cluster_key"].map(lambda key: remap.get(key, key))
    return out


def add_story_ids(
    df: pd.DataFrame,
    *,
    time_col: str = "seendate",
    title_col: str = "title",
    url_col: str = "url",
    window_hours: int = DEDUP_WINDOW_HOURS,
    fuzzy_threshold: int = FUZZ_THRESHOLD,
    review_threshold: int = REVIEW_THRESHOLD,
) -> pd.DataFrame:
    """Add causal story IDs and keep/audit decisions to article rows.

    Exact canonical URLs and outlet-stripped titles are suppressed inside the
    window. Fuzzy matching is cross-domain only and requires protected numeric
    and calendar tokens to match. Only retained primary stories become future
    references, preventing echoes from extending a story indefinitely.
    """
    missing = {time_col, title_col}.difference(df.columns)
    if missing:
        raise ValueError(f"dedup input missing columns: {sorted(missing)}")

    out = df.copy()
    out["_input_order"] = range(len(out))
    out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="raise")
    if "title_norm" not in out.columns:
        from sentiment.score import _norm

        out["title_norm"] = _norm(out[title_col])
    out["_title_key"] = out["title_norm"].map(strip_outlet)
    out.loc[out["_title_key"] == "", "_title_key"] = out["title_norm"]
    urls = out[url_col] if url_col in out.columns else pd.Series("", index=out.index)
    out["_canonical_url"] = urls.map(_canonical_url)
    out["_domain"] = out["_canonical_url"].map(_domain)
    out["_protected"] = out["title_norm"].map(_protected_tokens)
    ordered = out.sort_values([time_col, "_input_order"], kind="stable")

    window = pd.Timedelta(hours=window_hours)
    exact_urls: dict[str, tuple[pd.Timestamp, str]] = {}
    exact_titles: dict[str, tuple[pd.Timestamp, str]] = {}
    blocks: dict[str, deque] = defaultdict(deque)
    story_ids: dict[int, str] = {}
    keep: dict[int, bool] = {}
    reasons: dict[int, str] = {}
    similarities: dict[int, float | None] = {}

    for idx, row in ordered.iterrows():
        timestamp = row[time_col]
        title_key = row["_title_key"]
        canonical_url = row["_canonical_url"]
        domain = row["_domain"]
        protected = row["_protected"]
        prefix = " ".join(title_key.split()[:2])

        match_story = None
        reason = "unique"
        similarity = None

        prior_url = exact_urls.get(canonical_url) if canonical_url else None
        if prior_url and timestamp - prior_url[0] <= window:
            match_story = prior_url[1]
            reason = "exact_url"
            similarity = 100.0

        prior_title = exact_titles.get(title_key) if title_key else None
        if match_story is None and prior_title and timestamp - prior_title[0] <= window:
            match_story = prior_title[1]
            reason = "exact_title"
            similarity = 100.0

        queue = blocks[prefix]
        while queue and timestamp - queue[0][0] > window:
            queue.popleft()

        review_score = None
        if match_story is None and title_key:
            for _, prior_key, prior_protected, prior_domain, prior_story in reversed(queue):
                if domain and prior_domain and domain == prior_domain:
                    continue
                if protected != prior_protected:
                    continue
                score = float(fuzz.token_sort_ratio(title_key, prior_key))
                if score >= fuzzy_threshold:
                    match_story = prior_story
                    reason = "fuzzy_95"
                    similarity = score
                    break
                if review_threshold <= score < fuzzy_threshold:
                    review_score = max(review_score or score, score)

        if match_story is not None:
            story_id = match_story
            keep_story = False
        else:
            story_id = _stable_story_id(timestamp, canonical_url, title_key)
            keep_story = True
            if review_score is not None:
                reason = "review_90_95"
                similarity = review_score
            if canonical_url:
                exact_urls[canonical_url] = (timestamp, story_id)
            if title_key:
                exact_titles[title_key] = (timestamp, story_id)
                queue.append((timestamp, title_key, protected, domain, story_id))

        story_ids[idx] = story_id
        keep[idx] = keep_story
        reasons[idx] = reason
        similarities[idx] = similarity

    out["story_id"] = pd.Series(story_ids)
    out["keep_story"] = pd.Series(keep, dtype=bool)
    out["dedup_reason"] = pd.Series(reasons)
    out["dedup_similarity"] = pd.Series(similarities, dtype="float64")
    return out.sort_values("_input_order", kind="stable").drop(
        columns=["_input_order", "_title_key", "_canonical_url", "_domain", "_protected"]
    )


def first_seen_only(df: pd.DataFrame, time_col: str = "seendate") -> pd.DataFrame:
    """Return retained primary stories under the causal 72-hour/95% rule."""
    enriched = add_story_ids(df, time_col=time_col)
    return enriched.loc[enriched["keep_story"]].drop(
        columns=["story_id", "keep_story", "dedup_reason", "dedup_similarity"]
    )
