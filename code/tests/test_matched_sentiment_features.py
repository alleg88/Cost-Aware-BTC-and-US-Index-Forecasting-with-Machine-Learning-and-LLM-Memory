import pandas as pd
import pytest


def _fixture(bars, *, relevance=0.9, impact=2, asset="BTC"):
    """Minimal matched news/direct/tone/macro/fear tables for both scorers."""
    news_classic = pd.DataFrame({
        "seendate": [bars[0]], "url": ["https://a.test/n"],
        "title": ["Bitcoin rises"], "sent": [0.4],
    })
    news_llm = pd.DataFrame({
        "seendate": [bars[0]], "url": ["https://a.test/n"],
        "title": ["Bitcoin rises"], "llm_sent": [0.7],
        "llm_relevance": [relevance], "llm_impact": [impact], "llm_asset": [asset],
    })
    direct_classic = pd.DataFrame({
        "seendate": [bars[1]], "url": ["https://fed.test/e"],
        "title": ["Fed statement"], "sent": [-0.2],
    })
    direct_llm = pd.DataFrame({
        "seendate": [bars[1]], "url": ["https://fed.test/e"],
        "title": ["Fed statement"], "llm_sent": [-0.5],
        "llm_relevance": [0.8], "llm_impact": [2], "llm_asset": ["macro"],
    })
    tone = pd.DataFrame({
        "seendate": [bars[0]], "url": ["https://a.test/n"],
        "title": ["Bitcoin rises"], "tone": [1.0],
    })
    macro = pd.DataFrame({"release_time": [bars[2]]})
    fear = pd.DataFrame(
        {"value": [40, 45]},
        index=pd.to_datetime(["2023-12-25", "2024-01-01"], utc=True),
    )
    return {
        "scores_btc.parquet": news_classic,
        "scores_llm_btc.parquet": news_llm,
        "scores_direct_events_btc.parquet": direct_classic,
        "scores_llm_direct_events_btc.parquet": direct_llm,
        "gdelt_btc.parquet": tone,
        "fred_calendar.parquet": macro,
        "crypto_fear_greed.parquet": fear,
    }


def _patch_reads(monkeypatch, tables):
    def fake_read(path, *args, **kwargs):
        name = getattr(path, "name", str(path).split("/")[-1])
        return tables[name].copy()

    monkeypatch.setattr(pd, "read_parquet", fake_read)


@pytest.mark.parametrize("weighting", ["plain", "own"])
def test_matched_sentiment_schema_is_identical_for_classic_and_llm(monkeypatch, weighting):
    import features.sentiment as sentiment

    bars = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    _patch_reads(monkeypatch, _fixture(bars))
    classic = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="classic", weighting=weighting)
    llm = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="llm", weighting=weighting)

    assert list(classic.columns) == sentiment.MATCHED_SENTIMENT_FEATURES_BTC
    assert list(llm.columns) == sentiment.MATCHED_SENTIMENT_FEATURES_BTC
    assert "sent_direct_decay" not in classic.columns      # not part of the matched schema
    assert classic.shape == llm.shape
    for common in ["sent_news_count_24h", "sent_tone_decay", "sent_macro_decay",
                   "sent_fng_change_7d"]:
        pd.testing.assert_series_equal(classic[common], llm[common])
    assert not classic["sent_news_decay"].equals(llm["sent_news_decay"])


def test_direct_event_column_is_buildable_outside_the_matched_schema(monkeypatch):
    """The direct-event variant stays available for the §5 case study."""
    import features.sentiment as sentiment

    bars = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    _patch_reads(monkeypatch, _fixture(bars))
    classic = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="classic", include_direct_events=True)
    llm = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="llm", include_direct_events=True)

    assert list(classic.columns) == sentiment.MATCHED_FEATURES_WITH_DIRECT_BTC
    assert not classic["sent_direct_decay"].equals(llm["sent_direct_decay"])


def test_deberta_arm_never_reads_llm_fields(monkeypatch):
    """The whole point of the split: LLM metadata must not leak into the DeBERTa arm.

    Behavioural, not textual — the LLM's relevance, impact and asset are perturbed
    to values that would visibly change any feature weighted by them, and the
    DeBERTa arm must come out bit-for-bit identical while the LLM arm moves.
    """
    import features.sentiment as sentiment

    bars = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")

    _patch_reads(monkeypatch, _fixture(bars, relevance=0.9, impact=2, asset="BTC"))
    classic_a = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="classic", weighting="own")
    llm_a = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="llm", weighting="own")

    _patch_reads(monkeypatch, _fixture(bars, relevance=0.05, impact=0, asset="other"))
    classic_b = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="classic", weighting="own")
    llm_b = sentiment.build_matched_sentiment_features(
        "btc", bars, scorer="llm", weighting="own")

    pd.testing.assert_frame_equal(classic_a, classic_b)       # untouched by LLM fields
    assert not llm_a["sent_news_decay"].equals(llm_b["sent_news_decay"])


def test_own_weighting_drops_off_topic_stories_in_both_arms(monkeypatch):
    """Each arm filters on-topic by its own means: keywords for DeBERTa, tag for LLM."""
    import features.sentiment as sentiment

    bars = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")
    tables = _fixture(bars, asset="other")
    tables["scores_btc.parquet"] = tables["scores_btc.parquet"].assign(
        title=["Soybean futures ease"])
    tables["scores_llm_btc.parquet"] = tables["scores_llm_btc.parquet"].assign(
        title=["Soybean futures ease"])
    tables["gdelt_btc.parquet"] = tables["gdelt_btc.parquet"].assign(
        title=["Soybean futures ease"])
    _patch_reads(monkeypatch, tables)

    for scorer in ("classic", "llm"):
        built = sentiment.build_matched_sentiment_features(
            "btc", bars, scorer=scorer, weighting="own")
        assert (built["sent_news_decay"] == 0.0).all()
        assert (built["sent_news_count_24h"] == 0.0).all()
