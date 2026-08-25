from __future__ import annotations

from pathlib import Path


QUERY_PATH = (
    Path(__file__).resolve().parents[1]
    / "sentiment"
    / "queries"
    / "gdelt_q2_2026_english_whitelist.sql"
)


def test_gdelt_q2_query_freezes_window_language_sources_and_topics() -> None:
    assert QUERY_PATH.is_file()
    sql = QUERY_PATH.read_text(encoding="utf-8")

    assert "_PARTITIONTIME >= TIMESTAMP('2026-04-01')" in sql
    assert "_PARTITIONTIME < TIMESTAMP('2026-07-01')" in sql
    assert "COALESCE(TranslationInfo, '') = ''" in sql
    assert "NET.REG_DOMAIN(DocumentIdentifier)" in sql
    assert "bitcoin|\\bbtc\\b|cryptocurrency" in sql
    assert "wall street|federal reserve" in sql
    assert "nasdaq|tech stocks?|big tech" in sql

    expected_domains = {
        "apnews.com",
        "barrons.com",
        "bitcoinmagazine.com",
        "bloomberg.com",
        "businessinsider.com",
        "cnbc.com",
        "coindesk.com",
        "cointelegraph.com",
        "decrypt.co",
        "forbes.com",
        "fortune.com",
        "ft.com",
        "investing.com",
        "marketwatch.com",
        "reuters.com",
        "theblock.co",
        "wsj.com",
    }
    assert all(f"'{domain}'" in sql for domain in expected_domains)

