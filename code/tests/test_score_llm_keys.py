from __future__ import annotations

from sentiment.score_llm import VERSION, _batch_schema, _parse, _parse_text


def test_batch_schema_uses_news_item_keys():
    schema = _batch_schema(3)

    assert list(schema["properties"].keys()) == ["n0", "n1", "n2"]
    assert schema["required"] == ["n0", "n1", "n2"]
    assert VERSION == "4"


def test_plain_text_fallback_parses_news_item_keys_only():
    parsed = _parse_text(
        "n0: sentiment=0.5, relevance=0.8, impact=medium, asset=BTC\n"
        "h1: sentiment=-0.5, relevance=0.8, impact=high, asset=US500",
        2,
    )

    assert parsed[0] == {
        "llm_sent": 0.5,
        "llm_relevance": 0.8,
        "llm_impact": 1,
        "llm_asset": "BTC",
    }
    assert parsed[1] is None



def test_parser_normalizes_invalid_assets_and_bounds_scores():
    parsed = _parse(
        {
            "sentiment": 2.0,
            "relevance": -0.2,
            "impact": "high",
            "asset": "ETH",
        }
    )

    assert parsed == {
        "llm_sent": 1.0,
        "llm_relevance": 0.0,
        "llm_impact": 2,
        "llm_asset": "other",
    }
