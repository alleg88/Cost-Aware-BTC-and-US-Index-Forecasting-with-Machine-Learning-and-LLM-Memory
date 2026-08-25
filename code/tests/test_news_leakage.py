"""Leak-free as-of join test: a bar must never see news published after its close.

This guards the highest-stakes step in the sentiment layer — a look-ahead here would
silently inflate results (papers/wiki/design/news-scoring-howto.md, risk 1).
"""
import numpy as np
import pandas as pd
import pytest

from features.news import build_news_features, NEWS_FEATURE_COLS


def _bars(n=12, start="2024-01-01 00:00"):
    return pd.date_range(start, periods=n, freq="15min", tz="UTC")


def test_bar_never_sees_future_news():
    bars = _bars()
    # one strongly-positive article at 00:40 -> 15min bin [00:30, 00:45), public at 00:45.
    scores = pd.DataFrame({
        "seendate": [pd.Timestamp("2024-01-01 00:40", tz="UTC")],
        "sent": [1.0],
    })
    feats = build_news_features(scores, bars)

    close = bars + pd.Timedelta("15min")
    article_time = pd.Timestamp("2024-01-01 00:45", tz="UTC")  # bin completion / public time

    # bars closing before the article is public must see zero news.
    before = feats.loc[close <= article_time - pd.Timedelta("15min")]
    assert (before[["news_cnt_6h", "news_cnt_24h"]].to_numpy() == 0).all()
    assert (before["news_sent_decay"].to_numpy() == 0).all()

    # from the bar whose close is the article's public time onward, the news is visible.
    after = feats.loc[bars >= pd.Timestamp("2024-01-01 00:30", tz="UTC")]
    assert (after["news_cnt_24h"].to_numpy() >= 1).all()
    assert (after["news_sent_decay"].to_numpy() > 0).all()


def test_no_news_is_neutral_zero():
    bars = _bars()
    empty = pd.DataFrame({"seendate": pd.to_datetime([], utc=True), "sent": []})
    feats = build_news_features(empty, bars)
    assert list(feats.columns) == NEWS_FEATURE_COLS
    assert (feats.to_numpy() == 0).all()


def test_requires_utc_index():
    naive = pd.date_range("2024-01-01", periods=4, freq="15min")  # tz-naive
    scores = pd.DataFrame({"seendate": pd.to_datetime([], utc=True), "sent": []})
    with pytest.raises(ValueError):
        build_news_features(scores, naive)


def test_first_seen_only_keeps_earliest_and_drops_echoes():
    from sentiment.dedup import first_seen_only

    df = pd.DataFrame({
        "title": [
            "Tesla Deliveries Fall Sharply - MarketWatch",   # original
            "Tesla Deliveries Fall Sharply By Reuters",      # echo, 19h later
            "Completely Different Story",
        ],
        "title_norm": [
            "tesla deliveries fall sharply - marketwatch",
            "tesla deliveries fall sharply by reuters",
            "completely different story",
        ],
        "seendate": pd.to_datetime(
            ["2024-01-01 08:00", "2024-01-02 03:00", "2024-01-01 12:00"], utc=True),
        "sent": [-0.8, -0.8, 0.1],
    })
    out = first_seen_only(df)
    assert len(out) == 2                                          # echo dropped
    kept = out[out["title"].str.startswith("Tesla")]
    assert kept["seendate"].iloc[0] == pd.Timestamp("2024-01-01 08:00", tz="UTC")


def test_temporal_dedup_keeps_recurrence_after_72_hours():
    from sentiment.dedup import first_seen_only

    df = pd.DataFrame({
        "title": ["Bitcoin market update", "Bitcoin market update"],
        "url": ["https://a.test/one", "https://b.test/two"],
        "seendate": pd.to_datetime(
            ["2024-01-01 00:00", "2024-01-04 00:01"], utc=True
        ),
    })
    assert len(first_seen_only(df)) == 2


def test_temporal_dedup_drops_95_percent_cross_domain_echo():
    from sentiment.dedup import first_seen_only

    df = pd.DataFrame({
        "title": [
            "Mt. Gox begins repayments in Bitcoin and Bitcoin Cash",
            "Mt Gox begins repayments in Bitcoin and Bitcoin Cash",
        ],
        "url": ["https://a.test/one", "https://b.test/two"],
        "seendate": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"], utc=True),
    })
    assert len(first_seen_only(df)) == 1


def test_temporal_dedup_keeps_sub_95_percent_match_for_audit():
    from sentiment.dedup import add_story_ids

    df = pd.DataFrame({
        "title": [
            "Morgan Stanley Files For Bitcoin, Solana ETF With SEC",
            "Morgan Stanley Files With SEC For Spot Bitcoin ETF",
        ],
        "url": ["https://a.test/one", "https://b.test/two"],
        "seendate": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"], utc=True),
    })
    out = add_story_ids(df)
    assert out["keep_story"].tolist() == [True, True]
    assert out.loc[1, "dedup_reason"] == "review_90_95"


def test_temporal_dedup_protects_changed_numbers():
    from sentiment.dedup import first_seen_only

    df = pd.DataFrame({
        "title": ["Bitcoin falls below $40k", "Bitcoin falls below $60k"],
        "url": ["https://a.test/one", "https://b.test/two"],
        "seendate": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"], utc=True),
    })
    assert len(first_seen_only(df)) == 2


def test_temporal_dedup_is_prefix_stable_when_future_rows_are_appended():
    from sentiment.dedup import add_story_ids

    base = pd.DataFrame({
        "title": ["Bitcoin ETF approved", "Bitcoin ETF approved - Reuters"],
        "url": ["https://a.test/one", "https://b.test/two"],
        "seendate": pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"], utc=True),
    })
    future = pd.concat([base, pd.DataFrame({
        "title": ["Bitcoin ETF approved after market close"],
        "url": ["https://c.test/three"],
        "seendate": pd.to_datetime(["2024-02-01"], utc=True),
    })], ignore_index=True)
    expected = add_story_ids(base)[["story_id", "keep_story", "dedup_reason"]]
    actual = add_story_ids(future).iloc[:2][["story_id", "keep_story", "dedup_reason"]]
    pd.testing.assert_frame_equal(expected.reset_index(drop=True), actual.reset_index(drop=True))
