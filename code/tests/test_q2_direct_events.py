from __future__ import annotations

import importlib.util
import json

import pandas as pd


def test_truth_archive_rows_are_filtered_routed_and_q2_bounded() -> None:
    assert importlib.util.find_spec("sentiment.q2_direct_events") is not None
    from sentiment.q2_direct_events import truth_rows_from_archive

    posts = pd.DataFrame(
        [
            {
                "id": "before",
                "created_at": "2026-03-31T23:59:59Z",
                "content": "Federal Reserve policy update",
                "url": "https://truthsocial.com/@realDonaldTrump/before",
            },
            {
                "id": "macro",
                "created_at": "2026-04-01T00:00:00Z",
                "content": "Federal Reserve and inflation policy update",
                "url": "https://truthsocial.com/@realDonaldTrump/macro",
            },
            {
                "id": "irrelevant",
                "created_at": "2026-05-01T12:00:00Z",
                "content": "Thank you to everyone at tonight's event",
                "url": "https://truthsocial.com/@realDonaldTrump/irrelevant",
            },
            {
                "id": "crypto",
                "created_at": "2026-06-30T23:59:59Z",
                "content": "Bitcoin and digital assets are important",
                "url": "https://truthsocial.com/@realDonaldTrump/crypto",
            },
            {
                "id": "at_end",
                "created_at": "2026-07-01T00:00:00Z",
                "content": "Federal Reserve policy update",
                "url": "https://truthsocial.com/@realDonaldTrump/at_end",
            },
        ]
    )

    rows = truth_rows_from_archive(
        posts,
        start=pd.Timestamp("2026-04-01", tz="UTC"),
        end=pd.Timestamp("2026-07-01", tz="UTC"),
    )

    assert len(rows) == 5
    assert {(row["url"], row["stream"]) for row in rows} == {
        ("https://truthsocial.com/@realDonaldTrump/macro", "usa500"),
        ("https://truthsocial.com/@realDonaldTrump/macro", "usatech"),
        ("https://truthsocial.com/@realDonaldTrump/crypto", "btc"),
        ("https://truthsocial.com/@realDonaldTrump/crypto", "usa500"),
        ("https://truthsocial.com/@realDonaldTrump/crypto", "usatech"),
    }
    assert all(row["source"] == "truth_social" for row in rows)
    assert min(row["available_at"] for row in rows) == pd.Timestamp(
        "2026-04-01", tz="UTC"
    )
    assert max(row["available_at"] for row in rows) == pd.Timestamp(
        "2026-06-30 23:59:59", tz="UTC"
    )


def test_stage_q2_combines_sources_and_writes_only_the_requested_window(tmp_path) -> None:
    from sentiment.direct_events import _make_rows
    from sentiment.q2_direct_events import stage_q2_direct_events

    start = pd.Timestamp("2026-04-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    posts = pd.DataFrame(
        [
            {
                "id": "macro",
                "created_at": "2026-04-02T12:00:00Z",
                "content": "Federal Reserve and inflation policy update",
                "url": "https://truthsocial.com/@realDonaldTrump/macro",
            }
        ]
    )
    fed_rows = _make_rows(
        event_time=pd.Timestamp("2026-06-17 18:00:00", tz="UTC"),
        source="fed",
        event_type="fed_fomc_statement",
        streams=("btc", "usa500", "usatech"),
        title="Federal Reserve issues FOMC statement",
        text="Inflation remains elevated.",
        url="https://www.federalreserve.gov/statement.htm",
        start=start,
        end=end,
    )
    fed_rows.append(
        {
            **fed_rows[0],
            "seendate": pd.Timestamp("2026-03-01", tz="UTC"),
            "event_time": pd.Timestamp("2026-03-01", tz="UTC"),
            "available_at": pd.Timestamp("2026-03-01", tz="UTC"),
            "url": "https://www.federalreserve.gov/outside.htm",
        }
    )

    staged = stage_q2_direct_events(
        posts,
        fed_rows=fed_rows,
        output_dir=tmp_path,
        start=start,
        end=end,
    )

    assert len(staged) == 5
    assert set(staged["source"]) == {"fed", "truth_social"}
    assert staged["available_at"].min() == pd.Timestamp("2026-04-02 12:00:00", tz="UTC")
    assert staged["available_at"].max() == pd.Timestamp("2026-06-17 18:00:00", tz="UTC")
    assert (tmp_path / "truth_social_q2_all.parquet").is_file()
    assert len(pd.read_parquet(tmp_path / "direct_events_usa500.parquet")) == 2


def test_collect_q2_writes_a_hash_bound_validation_manifest(tmp_path) -> None:
    from sentiment.direct_events import _make_rows
    from sentiment.q2_direct_events import collect_q2_to_directory

    start = pd.Timestamp("2026-04-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    payload = json.dumps(
        [
            {
                "id": "macro",
                "created_at": "2026-04-02T12:00:00Z",
                "content": "Federal Reserve and inflation policy update",
                "url": "https://truthsocial.com/@realDonaldTrump/macro",
            }
        ]
    ).encode("utf-8")
    fed_rows = _make_rows(
        event_time=pd.Timestamp("2026-06-17 18:00:00", tz="UTC"),
        source="fed",
        event_type="fed_fomc_statement",
        streams=("btc", "usa500", "usatech"),
        title="Federal Reserve issues FOMC statement",
        text="Inflation remains elevated.",
        url="https://www.federalreserve.gov/statement.htm",
        start=start,
        end=end,
    )

    staged, manifest = collect_q2_to_directory(
        truth_payload=payload,
        fed_rows=fed_rows,
        output_dir=tmp_path,
        start=start,
        end=end,
        truth_source_url="https://example.test/truth_archive.json",
    )

    saved = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert len(staged) == 5
    assert manifest == saved
    assert saved["truth_archive_q2_rows"] == 1
    assert saved["truth_relevant_posts"] == 1
    assert saved["fed_documents"] == 1
    assert saved["stream_rows"] == {"btc": 1, "usa500": 2, "usatech": 2}
    assert saved["duplicate_stream_events"] == 0
    assert set(saved["artifact_sha256"]) == {
        "direct_events.parquet",
        "direct_events_btc.parquet",
        "direct_events_usa500.parquet",
        "direct_events_usatech.parquet",
        "truth_social_q2_all.parquet",
    }
