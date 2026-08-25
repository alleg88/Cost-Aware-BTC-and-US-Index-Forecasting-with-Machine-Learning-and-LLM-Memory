from __future__ import annotations

import pandas as pd

import sentiment.direct_events as direct_events
from sentiment.direct_events import (
    _make_rows,
    _streams_for_text,
    rebuild_direct_streams,
    validate_canonical_direct_coverage,
)


def test_make_rows_duplicates_event_to_each_stream_with_score_text():
    rows = _make_rows(
        event_time=pd.Timestamp("2024-01-31 19:00:00", tz="UTC"),
        source="fed",
        event_type="fed_fomc_statement",
        streams=("btc", "usa500"),
        title="Federal Reserve issues FOMC statement",
        text="Inflation remains elevated.",
        url="https://www.federalreserve.gov/example.htm",
    )

    assert [r["stream"] for r in rows] == ["btc", "usa500"]
    assert rows[0]["seendate"] == pd.Timestamp("2024-01-31 19:00:00", tz="UTC")
    assert "Inflation remains elevated" in rows[0]["score_text"]


def test_streams_for_text_routes_crypto_to_btc_and_macro_to_indices():
    assert _streams_for_text("Treasury discusses Bitcoin and digital assets") == (
        "btc",
        "usa500",
        "usatech",
    )
    assert _streams_for_text("Treasury market bond auction update") == ("usa500", "usatech")


def test_make_rows_honours_an_explicit_half_open_window():
    start = pd.Timestamp("2026-04-01", tz="UTC")
    end = pd.Timestamp("2026-07-01", tz="UTC")
    common = {
        "source": "fed",
        "event_type": "fed_fomc_statement",
        "streams": ("usa500",),
        "title": "Federal Reserve issues FOMC statement",
        "text": "Inflation remains elevated.",
        "url": "https://www.federalreserve.gov/example.htm",
        "start": start,
        "end": end,
    }

    inside = _make_rows(event_time=start, **common)
    before = _make_rows(event_time=start - pd.Timedelta(seconds=1), **common)
    at_end = _make_rows(event_time=end, **common)

    assert len(inside) == 1
    assert before == []
    assert at_end == []


def test_fed_collector_fetches_details_only_inside_the_requested_window(monkeypatch):
    calendar = """
    <a href="/newsevents/pressreleases/monetary20260318a.htm">March statement</a>
    <a href="/newsevents/pressreleases/monetary20260507a.htm">May statement</a>
    """
    detail = """
    <html><head><title>Federal Reserve issues FOMC statement</title></head>
    <main>Recent indicators suggest that economic activity remains solid.
    For media inquiries, call the press office.</main></html>
    """
    calls: list[str] = []

    def fake_get(url: str) -> str:
        calls.append(url)
        return calendar if url == direct_events.FED_FOMC else detail

    monkeypatch.setattr(direct_events, "_get", fake_get)
    rows = direct_events.collect_fed_fomc(
        start=pd.Timestamp("2026-04-01", tz="UTC"),
        end=pd.Timestamp("2026-07-01", tz="UTC"),
    )

    assert calls == [
        direct_events.FED_FOMC,
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20260507a.htm",
    ]
    assert len(rows) == 3
    assert {row["stream"] for row in rows} == {"btc", "usa500", "usatech"}
    assert {row["seendate"] for row in rows} == {
        pd.Timestamp("2026-05-07 18:00:00", tz="UTC")
    }


def test_write_outputs_can_target_a_staging_directory(tmp_path):
    rows = _make_rows(
        event_time=pd.Timestamp("2026-05-07 18:00:00", tz="UTC"),
        source="fed",
        event_type="fed_fomc_statement",
        streams=("usa500", "usatech"),
        title="Federal Reserve issues FOMC statement",
        text="Inflation remains elevated.",
        url="https://www.federalreserve.gov/example.htm",
        start=pd.Timestamp("2026-04-01", tz="UTC"),
        end=pd.Timestamp("2026-07-01", tz="UTC"),
    )

    frame = direct_events.write_outputs(rows, output_dir=tmp_path)

    assert len(frame) == 2
    assert (tmp_path / "direct_events.parquet").is_file()
    assert (tmp_path / "direct_events_btc.parquet").is_file()
    assert len(pd.read_parquet(tmp_path / "direct_events_usa500.parquet")) == 1
    assert len(pd.read_parquet(tmp_path / "direct_events_usatech.parquet")) == 1


def test_direct_streams_rebuild_from_frozen_master_in_causal_order(tmp_path):
    master = tmp_path / "direct_events.parquet"
    rows = []
    for timestamp, suffix in (("2025-01-02", "b"), ("2025-01-01", "a")):
        rows.extend(
            _make_rows(
                event_time=pd.Timestamp(timestamp, tz="UTC"),
                source="fed",
                event_type="fed_fomc_statement",
                streams=direct_events.STREAMS,
                title=f"Event {suffix}",
                text="Policy event.",
                url=f"https://www.federalreserve.gov/{suffix}",
            )
        )
    pd.DataFrame(rows).to_parquet(master, index=False)
    original = master.read_bytes()
    output = tmp_path

    outputs = rebuild_direct_streams(master, output)

    assert set(outputs) == {"btc", "usa500", "usatech"}
    assert all(path.is_file() for path in outputs.values())
    assert master.read_bytes() == original
    for path in outputs.values():
        frame = pd.read_parquet(path)
        assert frame["available_at"].is_monotonic_increasing


def test_canonical_direct_coverage_checks_truth_social_and_fed_counts(tmp_path):
    master = pd.DataFrame(
        {
            "source": ["fed"] * 111 + ["truth_social"] * 3162,
            "url": [f"fed-{i // 3}" for i in range(111)]
            + [f"truth-{i // 2}" for i in range(3162)],
        }
    )
    master_path = tmp_path / "direct_events.parquet"
    master.to_parquet(master_path, index=False)
    truth_path = tmp_path / "trump_truth_posts.csv"
    pd.DataFrame(
        {
            "created_at": ["2025-01-01T00:00:00Z"] * 1660,
            "url": [f"truth-{i}" for i in range(1660)],
            "content": ["tariff"] * 1660,
        }
    ).to_csv(truth_path, index=False)

    report = validate_canonical_direct_coverage(master_path, truth_path)

    assert report == {"fed_stream_rows": 111, "truth_social_posts": 1660, "truth_social_stream_rows": 3162}


def test_fed_minutes_use_public_release_date_not_meeting_date(monkeypatch):
    calendar = """
    <a href="/monetarypolicy/fomcminutes20260429.htm">HTML</a>
    <br> (Released May 20, 2026)
    <a href="/monetarypolicy/fomcminutes20260617.htm">HTML</a>
    <br> (Released July 08, 2026)
    """
    detail = """
    <html><head><title>FOMC Minutes</title></head>
    <main>Minutes of the Federal Open Market Committee. Inflation discussion.
    Last Update:</main></html>
    """
    calls: list[str] = []

    def fake_get(url: str) -> str:
        calls.append(url)
        return calendar if url == direct_events.FED_FOMC else detail

    monkeypatch.setattr(direct_events, "_get", fake_get)
    rows = direct_events.collect_fed_fomc(
        start=pd.Timestamp("2026-04-01", tz="UTC"),
        end=pd.Timestamp("2026-07-01", tz="UTC"),
    )

    assert calls == [
        direct_events.FED_FOMC,
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20260429.htm",
    ]
    assert len(rows) == 3
    assert {row["event_time"] for row in rows} == {
        pd.Timestamp("2026-05-20 18:00:00", tz="UTC")
    }
