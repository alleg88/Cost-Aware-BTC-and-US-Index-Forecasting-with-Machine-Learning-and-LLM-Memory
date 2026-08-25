"""Assemble ALL leak-free sentiment/event features for a bar grid.

Combines four collected sources into per-bar features, each joined so a bar only ever sees
information public by its own close (see features/news.py for the convention):

  * DeBERTa headline sentiment  (features/news.py)          -> news_*  (all instruments)
  * GDELT V2Tone                (gdelt_<stream>.parquet)     -> tone_*  (all instruments)
  * Fed/Trump direct events     (scores_direct_events_*)     -> de_*    (if scored)
  * Crypto Fear & Greed         (crypto_fear_greed.parquet)  -> fng_*   (BTC only)
  * FRED macro releases         (fred_calendar.parquet)      -> macro_* (all instruments)

The LLM arm (build_llm_features) adds llm_* (GDELT) and de_llm_* (direct events).

Public API:
    assemble_sentiment_features(instrument, bar_index) -> DataFrame aligned to bar_index
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from features.news import build_news_features, load_scores

CODE_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = CODE_ROOT / "sentiment" / "raw"
_FREQ = "15min"
_BPH = 4  # M15 bars per hour

MATCHED_SENTIMENT_FEATURES_COMMON = [
    "sent_news_decay",
    "sent_news_count_24h",
    "sent_tone_decay",
    "sent_macro_decay",
]
MATCHED_SENTIMENT_FEATURES_BTC = [
    *MATCHED_SENTIMENT_FEATURES_COMMON,
    "sent_fng_change_7d",
]
# Schema variant that additionally carries the Fed/Trump direct-event column. The
# matched arms do not use it: the two scorers agree on its sign at chance level
# (r = -0.24 over 64 events), so it cannot carry information in both arms at once
# and is studied on its own instead.
MATCHED_FEATURES_WITH_DIRECT_COMMON = [
    "sent_news_decay", "sent_news_count_24h", "sent_direct_decay",
    "sent_tone_decay", "sent_macro_decay",
]
MATCHED_FEATURES_WITH_DIRECT_BTC = [
    *MATCHED_FEATURES_WITH_DIRECT_COMMON, "sent_fng_change_7d",
]
# The LLM arm's structured output, exposed as its own channels. These describe the
# news flow itself rather than its direction, and none of them has a DeBERTa
# equivalent: the encoder emits a single scalar and nothing else.
LLM_STRUCTURED_FEATURES = [
    "sent_llm_relevance_decay",      # how relevant the recent flow is
    "sent_llm_hi_impact_decay",      # direction of high-impact on-topic stories only
    "sent_llm_topic_share_24h",      # share of the 24h flow that is on-topic
]

# Federal Reserve releases and Trump Truth Social posts. Each arm scores and weights
# them by its own means, exactly as it does headlines, so neither borrows from the
# other. One column, not several: there are only ~65 events in the whole span, and a
# family of channels would claim resolution that sample cannot support. The signed
# decayed pulse carries both roles at once — magnitude is how much event mass is
# still live, sign is what the recent events said, and zero means none are recent.
DIRECT_EVENT_FEATURES = ["sent_direct_pulse"]
# Policy events are digested over days, not hours, so they persist far longer than a
# headline; a headline half-life would leave this block at zero on almost every bar.
DIRECT_EVENT_HALFLIFE_H = 48.0

LLM_FULL_FEATURES_BTC = [
    *MATCHED_SENTIMENT_FEATURES_BTC, *LLM_STRUCTURED_FEATURES, *DIRECT_EVENT_FEATURES,
]

# On-topic lexical test for the DeBERTa arm. Deliberately keyword-based: the
# DeBERTa arm must stay reproducible from DeBERTa outputs plus the headline text
# alone, with no field borrowed from the LLM scorer.
_BTC_PATTERN = r"\b(?:bitcoin|btc|crypto|cryptocurrenc\w*|digital asset\w*|satoshi)\b"
_ASSET_PATTERNS = {"btc": _BTC_PATTERN}
_IMPACT_WEIGHT = {0: 1.0, 1: 2.0, 2: 3.0}       # LLM low / medium / high


def _grid_agg(times, values, bar_index):
    """Per-15min-bin (sum, count) on a continuous grid spanning the events and the bars."""
    ev = pd.DataFrame({"t": pd.to_datetime(pd.Series(times), utc=True),
                       "v": np.asarray(values, dtype=float)}).dropna(subset=["t"])
    if ev.empty:
        grid = pd.date_range(bar_index.min(), bar_index.max(), freq=_FREQ, tz="UTC")
        return pd.DataFrame({"sum": 0.0, "cnt": 0.0}, index=grid), grid
    ev["bin"] = ev["t"].dt.floor(_FREQ)
    agg = ev.groupby("bin")["v"].agg(**{"sum": "sum", "cnt": "count"})
    start, end = min(agg.index.min(), bar_index.min()), max(agg.index.max(), bar_index.max())
    grid = pd.date_range(start, end, freq=_FREQ, tz="UTC")
    return agg.reindex(grid, fill_value=0.0), grid


def _first_seen_with_echoes(frame: pd.DataFrame, time_col: str = "seendate") -> pd.DataFrame:
    """`first_seen_only` plus how many later echoes each retained story suppressed.

    The echo count is a scorer-independent importance proxy: a story carried by
    many outlets is, on average, a bigger story. It comes from the same causal
    72-hour ledger, so it cannot see the future.
    """
    from sentiment.dedup import add_story_ids

    enriched = add_story_ids(frame, time_col=time_col)
    echoes = enriched.groupby("story_id").size().rename("echo_count")
    kept = enriched.loc[enriched["keep_story"]].join(echoes, on="story_id")
    return kept.drop(columns=["story_id", "keep_story", "dedup_reason", "dedup_similarity"])


def _decayed_weighted_mean(times, values, weights, bar_index, *, halflife_h=6.0):
    """Exponentially decayed WEIGHTED MEAN of an event stream, aligned leak-free.

    Normalising by the decayed weight mass keeps this column purely directional: an
    unnormalised decayed sum would rise with story volume as well as with story
    direction, and volume is already carried by the separate count column.
    """
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    num, grid = _grid_agg(times, values * weights, bar_index)
    den, _ = _grid_agg(times, weights, bar_index)
    span = halflife_h * _BPH
    num_d = num["sum"].ewm(halflife=span, adjust=False).mean()
    den_d = den["sum"].ewm(halflife=span, adjust=False).mean()
    out = np.where(den_d.to_numpy() > 1e-12, num_d.to_numpy() / den_d.replace(0.0, np.nan).to_numpy(), 0.0)
    return pd.Series(out, index=grid).reindex(bar_index).fillna(0.0)


def _window_features(times, values, bar_index, prefix, windows_h=(24,),
                     decay_halflife_h=6.0, mean=True, count=True):
    """Trailing rolling windows + EWM decay on the grid, aligned to bar_index (leak-free)."""
    g, grid = _grid_agg(times, values, bar_index)
    out = {}
    for wh in windows_h:
        w = wh * _BPH
        c = g["cnt"].rolling(w, min_periods=1).sum()
        s = g["sum"].rolling(w, min_periods=1).sum()
        if count:
            out[f"{prefix}_cnt_{wh}h"] = c
        if mean:
            out[f"{prefix}_mean_{wh}h"] = np.where(c > 0, s / c.replace(0, np.nan), 0.0)
    out[f"{prefix}_decay"] = g["sum"].ewm(halflife=decay_halflife_h * _BPH, adjust=False).mean()
    return pd.DataFrame(out, index=grid).reindex(bar_index).fillna(0.0)


def build_tone_features(stream: str, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """GDELT V2Tone aggregated per bar (mean over 24h + EWM decay). Echo-suppressed."""
    from sentiment.dedup import first_seen_only

    if len(bar_index) == 0:
        return pd.DataFrame(index=bar_index)
    available_end = bar_index.max() + pd.Timedelta(_FREQ)
    g = pd.read_parquet(
        RAW_DIR / f"gdelt_{stream}.parquet",
        columns=["seendate", "title", "tone"],
        filters=[("seendate", "<", available_end)],
    ).dropna(subset=["tone"])
    g["seendate"] = pd.to_datetime(g["seendate"], utc=True)
    if len(g) and g["seendate"].ge(available_end).any():
        raise AssertionError("GDELT tone predicate crossed the available boundary")
    g = first_seen_only(g)
    return _window_features(g["seendate"], g["tone"], bar_index, "tone",
                            windows_h=(24,), mean=True, count=False)


def build_macro_features(bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """FRED macro-release pulse: release count in 24h + an EWM-decayed release impulse."""
    if len(bar_index) == 0:
        return pd.DataFrame(index=bar_index)
    available_end = bar_index.max() + pd.Timedelta(_FREQ)
    fred = pd.read_parquet(
        RAW_DIR / "fred_calendar.parquet",
        columns=["release_time"],
        filters=[("release_time", "<", available_end)],
    ).dropna()
    return _window_features(fred["release_time"], np.ones(len(fred)), bar_index, "macro",
                            windows_h=(24,), decay_halflife_h=6.0, mean=False, count=True)


def build_fear_greed_features(bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Crypto Fear & Greed: most recent daily index value + its 7-day change (BTC only)."""
    fg = pd.read_parquet(RAW_DIR / "crypto_fear_greed.parquet").copy()
    fg.index = pd.to_datetime(fg.index, utc=True)
    fg = fg.sort_index()
    ns = "datetime64[ns, UTC]"
    right = pd.DataFrame({"ts": fg.index.astype(ns), "fng_value": fg["value"].astype(float)})
    # as-of: each bar sees the latest F&G value published at/before its close.
    close = pd.DataFrame({"close": (bar_index + pd.Timedelta(_FREQ)).astype(ns)})
    merged = pd.merge_asof(close.sort_values("close"), right, left_on="close", right_on="ts",
                           direction="backward")
    merged.index = bar_index
    val = merged["fng_value"]
    chg7 = val - val.shift(7 * 24 * _BPH)   # change vs ~7 days ago on the bar grid
    return pd.DataFrame({"fng_value": val.fillna(50.0),        # 50 = neutral if unknown
                         "fng_chg7": chg7.fillna(0.0)}, index=bar_index)


_ASSET = {"btc": "BTC", "usa500": "US500", "usatech": "USTECH"}


def build_direct_event_features(stream: str, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build leak-free direct-event count and DeBERTa sentiment features."""
    from sentiment.dedup import first_seen_only

    path = RAW_DIR / f"scores_direct_events_{stream}.parquet"
    if not path.exists():
        return pd.DataFrame(index=bar_index)                 # skip streams not yet scored
    s = pd.read_parquet(path)[["seendate", "title", "title_norm", "sent"]].dropna(subset=["sent"])
    s["seendate"] = pd.to_datetime(s["seendate"], utc=True)
    s = first_seen_only(s)
    return _window_features(s["seendate"], s["sent"], bar_index, "de",
                            windows_h=(24,), mean=True, count=True)


def build_llm_features(stream: str, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Per-bar features from the LLM scorer (sentiment/score_llm.py), leak-free.

    Uses the structured fields to separate signal from noise — the hypothesis the plain
    DeBERTa tone failed (see the sentiment-ablation notebook):
      * <p>_sent_* — relevance-WEIGHTED sentiment over all items (chatter ~0 weight);
      * <p>_hi_*   — sentiment/volume of HIGH-impact, on-topic items only.
    Built for GDELT news (prefix `llm`) and, if scored, the Fed/Trump direct events (`de_llm`).
    """
    from sentiment.dedup import first_seen_only

    cols = ["seendate", "title", "title_norm", "llm_sent", "llm_relevance", "llm_impact", "llm_asset"]
    on_topic = [_ASSET[stream], "macro"]
    sources = [("llm", RAW_DIR / f"scores_llm_{stream}.parquet", True),
               ("de_llm", RAW_DIR / f"scores_llm_direct_events_{stream}.parquet", False)]
    parts = []
    for prefix, path, required in sources:
        if not path.exists():
            if required:
                raise FileNotFoundError(f"{path} not found — run sentiment/score_llm.py first")
            continue
        s = pd.read_parquet(path)[cols].dropna(subset=["llm_sent"])
        s["seendate"] = pd.to_datetime(s["seendate"], utc=True)
        s = first_seen_only(s)                                 # echo suppression
        weighted = s["llm_sent"] * s["llm_relevance"]
        hi = s[(s["llm_impact"] >= 2) & s["llm_asset"].isin(on_topic)]
        parts.append(_window_features(s["seendate"], weighted, bar_index, f"{prefix}_sent",
                                      windows_h=(24,), mean=True, count=False))
        parts.append(_window_features(hi["seendate"], hi["llm_sent"], bar_index, f"{prefix}_hi",
                                      windows_h=(24,), mean=True, count=True))
    return pd.concat(parts, axis=1)


def _load_matched_scores(stream: str, scorer: str, *, direct: bool,
                         with_metadata: bool = False) -> pd.DataFrame:
    """Load the common successfully scored rows for either scorer.

    Both arms always see the identical story set. `with_metadata=True` also
    returns the per-story fields each arm may weight by; which of them an arm is
    allowed to touch is decided by the caller, never here — the DeBERTa arm must
    never read an `llm_*` column.
    """
    if scorer not in {"classic", "llm"}:
        raise ValueError("scorer must be 'classic' or 'llm'")
    direct_tag = "_direct_events" if direct else ""
    classic_path = RAW_DIR / f"scores{direct_tag}_{stream}.parquet"
    llm_path = RAW_DIR / f"scores_llm{direct_tag}_{stream}.parquet"
    for path in (classic_path, llm_path):
        if not path.exists():
            raise FileNotFoundError(path)
    keys = ["seendate", "url", "title"]
    llm_extra = ["llm_relevance", "llm_impact", "llm_asset"] if with_metadata else []
    classic = pd.read_parquet(classic_path)[[*keys, "sent"]].dropna(subset=["sent"])
    llm_cols = pd.read_parquet(llm_path)
    llm = llm_cols[[*keys, "llm_sent", *[c for c in llm_extra if c in llm_cols]]].dropna(
        subset=["llm_sent"])
    scored = classic.merge(llm, on=keys, how="inner", validate="one_to_one")
    scored["seendate"] = pd.to_datetime(scored["seendate"], utc=True)

    if with_metadata:
        scored = _first_seen_with_echoes(scored)
    else:
        from sentiment.dedup import first_seen_only

        scored = first_seen_only(scored)

    score_col = "llm_sent" if scorer == "llm" else "sent"
    keep = [*keys, score_col]
    if with_metadata:
        keep += [c for c in ("echo_count", *llm_extra) if c in scored.columns]
        if scorer == "classic":
            keep.append("sent")                       # |sent| is DeBERTa's own confidence
    out = scored[list(dict.fromkeys(keep))].copy()
    return out.rename(columns={score_col: "matched_sentiment"})


def _arm_weights(scored: pd.DataFrame, *, scorer: str, stream: str):
    """Per-story (keep-mask, weight) derived ONLY from that arm's own means.

    DeBERTa arm  — lexical on-topic test on the headline text, weighted by the
                   model's own confidence |sent| and by the echo count.
    LLM arm      — the scorer's own `asset` tag, weighted by `relevance` and
                   `impact`.
    Neither arm can see the other's fields, so each is reproducible from its own
    pipeline and the comparison stays layer-versus-layer.
    """
    if scorer == "classic":
        pattern = _ASSET_PATTERNS.get(stream)
        on_topic = (scored["title"].astype(str).str.contains(pattern, case=False, regex=True)
                    if pattern else pd.Series(True, index=scored.index))
        confidence = scored.get("sent", scored["matched_sentiment"]).abs().clip(0.0, 1.0)
        echoes = np.log1p(scored.get("echo_count", pd.Series(1.0, index=scored.index)))
        weight = confidence * (1.0 + echoes)
    else:
        on_topic = scored["llm_asset"].isin([_ASSET.get(stream, stream.upper()), "macro"])
        relevance = scored["llm_relevance"].astype(float).clip(0.0, 1.0)
        impact = scored["llm_impact"].map(_IMPACT_WEIGHT).fillna(1.0)
        weight = relevance * impact
    weight = weight.astype(float).fillna(0.0)
    return on_topic.fillna(False).to_numpy(), weight.to_numpy()


def build_matched_sentiment_features(
    stream: str,
    bar_index: pd.DatetimeIndex,
    *,
    scorer: str,
    weighting: str = "plain",
    halflife_h: float = 6.0,
    include_direct_events: bool = False,
) -> pd.DataFrame:
    """Build the compact sentiment schema for the DeBERTa or the LLM arm.

    Both arms always use the same story set, the same causal 72-hour ledger, the
    same windows and the same output columns; only the scalar sentiment differs.

    `weighting`:
      * ``"plain"`` — every retained story counts equally and the decay column is
        an EWM of the per-bar sum.
      * ``"own"``   — each arm filters and weights using **only its own means**
        (see `_arm_weights`) and the decay column is a weighted mean, so direction
        is separated from volume.

    `include_direct_events` adds the Fed/Trump `sent_direct_decay` column. The
    matched arms leave it out (see `MATCHED_FEATURES_WITH_DIRECT_COMMON`).
    """
    if bar_index.tz is None:
        raise ValueError("bar_index must be tz-aware (UTC)")
    if weighting not in {"plain", "own"}:
        raise ValueError("weighting must be 'plain' or 'own'")

    news = _load_matched_scores(stream, scorer, direct=False,
                                with_metadata=weighting == "own")
    if weighting == "own":
        keep, weight = _arm_weights(news, scorer=scorer, stream=stream)
        kept = news.loc[keep]
        decay = _decayed_weighted_mean(kept["seendate"], kept["matched_sentiment"],
                                       weight[keep], bar_index, halflife_h=halflife_h)
        counts = _window_features(kept["seendate"], kept["matched_sentiment"], bar_index,
                                  "sent_news", windows_h=(24,), decay_halflife_h=halflife_h,
                                  mean=False, count=True)
        news_features = pd.DataFrame(
            {"sent_news_decay": decay,
             "sent_news_count_24h": counts["sent_news_cnt_24h"]}, index=bar_index)
    else:
        news_features = _window_features(
            news["seendate"], news["matched_sentiment"], bar_index, "sent_news",
            windows_h=(24,), decay_halflife_h=halflife_h, mean=False, count=True,
        ).rename(columns={"sent_news_cnt_24h": "sent_news_count_24h"})

    tone = build_tone_features(stream, bar_index)[["tone_decay"]].rename(
        columns={"tone_decay": "sent_tone_decay"}
    )
    macro = build_macro_features(bar_index)[["macro_decay"]].rename(
        columns={"macro_decay": "sent_macro_decay"}
    )
    parts = [news_features, tone, macro]
    columns = list(MATCHED_SENTIMENT_FEATURES_COMMON)

    if include_direct_events:
        direct = _load_matched_scores(stream, scorer, direct=True)
        parts.append(_window_features(
            direct["seendate"], direct["matched_sentiment"], bar_index, "sent_direct",
            windows_h=(), decay_halflife_h=halflife_h, mean=False, count=False))
        columns = list(MATCHED_FEATURES_WITH_DIRECT_COMMON)

    if stream == "btc":
        fear_greed = build_fear_greed_features(bar_index)[["fng_chg7"]].rename(
            columns={"fng_chg7": "sent_fng_change_7d"}
        )
        parts.append(fear_greed)
        columns = (list(MATCHED_FEATURES_WITH_DIRECT_BTC) if include_direct_events
                   else list(MATCHED_SENTIMENT_FEATURES_BTC))
    return pd.concat(parts, axis=1).reindex(columns=columns).fillna(0.0)


def build_direct_event_block(
    stream: str,
    bar_index: pd.DatetimeIndex,
    *,
    scorer: str,
    weighting: str = "own",
    halflife_h: float = DIRECT_EVENT_HALFLIFE_H,
) -> pd.DataFrame:
    """Fed / Trump direct events as one signed decayed pulse, weighted by the arm's
    own means.

    `weighting` follows the arm it belongs to: ``"own"`` filters and weights by that
    scorer's own fields, ``"plain"`` counts every event equally, so an unweighted arm
    stays unweighted throughout.

    Unlike the headline column this is a decayed *sum*, not a mean, and that is the
    point: a mean would hold the last event's direction indefinitely, whereas the sum
    falls back to zero as the event ages. One column therefore states three things at
    once — whether a policy event is still live (magnitude), what it said (sign), and
    how heavily its own arm rated it (weight). The two scorers disagree on the sign of
    a policy event at chance level, so the magnitude is the robust part of the signal
    and the sign is the fragile part; keeping them in one column lets a model use the
    reliable component without being forced to trust the other.
    """
    events = _load_matched_scores(stream, scorer, direct=True, with_metadata=True)
    if events.empty:
        return pd.DataFrame(0.0, index=bar_index, columns=DIRECT_EVENT_FEATURES)
    if weighting == "plain":
        on_topic = np.ones(len(events), dtype=bool)
        weight = np.ones(len(events))
    else:
        on_topic, weight = _arm_weights(events, scorer=scorer, stream=stream)
        if not on_topic.any():                 # never strand the block on an empty mask
            on_topic = np.ones(len(events), dtype=bool)
    kept = events.loc[on_topic]
    # Rescaling by 1/alpha puts the pulse in units of "one unit-weight event at its own
    # bar"; alpha is a constant of the half-life and uses no information from the data.
    alpha = 1.0 - 0.5 ** (1.0 / (halflife_h * _BPH))
    signed = _window_features(
        kept["seendate"], kept["matched_sentiment"].to_numpy() * weight[on_topic],
        bar_index, "sent_direct", windows_h=(), decay_halflife_h=halflife_h,
        mean=False, count=False)["sent_direct_decay"]
    return pd.DataFrame({"sent_direct_pulse": signed / alpha}, index=bar_index)


def build_llm_full_features(
    stream: str,
    bar_index: pd.DatetimeIndex,
    *,
    halflife_h: float = 6.0,
) -> pd.DataFrame:
    """The LLM arm with its structured output kept, not reduced to one scalar.

    Returns the five matched columns (LLM scalar, own filter and weights) plus the
    three channels in `LLM_STRUCTURED_FEATURES`. These describe the news flow —
    how relevant it is, what the high-impact subset is saying, and how much of it
    is on-topic — which a single sentiment scalar cannot express and which the
    DeBERTa arm has no way to produce.
    """
    base = build_matched_sentiment_features(stream, bar_index, scorer="llm",
                                            weighting="own", halflife_h=halflife_h)
    scored = _load_matched_scores(stream, "llm", direct=False, with_metadata=True)
    on_topic, weight = _arm_weights(scored, scorer="llm", stream=stream)
    relevance = scored["llm_relevance"].astype(float).clip(0.0, 1.0).to_numpy()
    impact = scored["llm_impact"].map(_IMPACT_WEIGHT).fillna(1.0).to_numpy()
    ones = np.ones(len(scored))

    kept = scored.loc[on_topic]
    relevance_decay = _decayed_weighted_mean(
        kept["seendate"], relevance[on_topic], ones[on_topic], bar_index,
        halflife_h=halflife_h)

    sharp = on_topic & (impact >= _IMPACT_WEIGHT[2])
    hi = scored.loc[sharp]
    hi_decay = (_decayed_weighted_mean(hi["seendate"], hi["matched_sentiment"],
                                       weight[sharp], bar_index, halflife_h=halflife_h)
                if len(hi) else pd.Series(0.0, index=bar_index))

    on_counts = _window_features(kept["seendate"], ones[on_topic], bar_index, "on",
                                 windows_h=(24,), mean=False, count=True)["on_cnt_24h"]
    all_counts = _window_features(scored["seendate"], ones, bar_index, "all",
                                  windows_h=(24,), mean=False, count=True)["all_cnt_24h"]
    share = (on_counts / all_counts.where(all_counts > 0)).fillna(0.0)

    extra = pd.DataFrame({
        "sent_llm_relevance_decay": relevance_decay,
        "sent_llm_hi_impact_decay": hi_decay,
        "sent_llm_topic_share_24h": share,
    }, index=bar_index)
    direct = build_direct_event_block(stream, bar_index, scorer="llm")
    columns = (LLM_FULL_FEATURES_BTC if stream == "btc"
               else [*MATCHED_SENTIMENT_FEATURES_COMMON, *LLM_STRUCTURED_FEATURES,
                     *DIRECT_EVENT_FEATURES])
    return (pd.concat([base, extra, direct], axis=1)
            .reindex(columns=columns).fillna(0.0))


def assemble_sentiment_features(instrument: str, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
    """All leak-free sentiment/event features for an instrument, aligned to bar_index."""
    parts = [
        build_news_features(load_scores(instrument), bar_index),  # DeBERTa headline sentiment
        build_tone_features(instrument, bar_index),               # GDELT tone
        build_macro_features(bar_index),                          # FRED macro pulse
        build_direct_event_features(instrument, bar_index),       # Fed/Trump events (DeBERTa)
    ]
    if instrument == "btc":
        parts.append(build_fear_greed_features(bar_index))        # crypto Fear & Greed
    return pd.concat(parts, axis=1)
