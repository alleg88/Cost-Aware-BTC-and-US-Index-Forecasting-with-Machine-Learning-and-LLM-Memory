from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


IDENTITY_FIELDS = {
    "tag": "deepseek-v4-flash:0731-cloud",
    "digest": "d" * 64,
    "remote_model": "deepseek-v4-flash:0731",
    "remote_host": "https://ollama.com",
    "registry_digest_prefix": "031ce2a95446",
    "prompt_hash": "p" * 64,
    "schema_hash": "s" * 64,
    "temperature": 0.0,
    "think": "low",
    "num_ctx": 8192,
    "batch_size": 10,
    "batch_protocol": "indexed-json-object-v1",
    "scorer_implementation_hash": "i" * 64,
}


def _install_fixture(tmp_path: Path, monkeypatch):
    from features import index_sentiment

    timestamp = pd.Timestamp("2024-01-01", tz="UTC")
    tables = {
        "scores_usa500.parquet": pd.DataFrame(
            {"seendate": [timestamp], "url": ["n"], "title": ["US stocks rise"], "sent": [0.2]}
        ),
        "scores_llm_usa500.parquet": pd.DataFrame(
            {
                "seendate": [timestamp], "url": ["n"], "title": ["US stocks rise"],
                "llm_sent": [0.4], "llm_relevance": [0.9], "llm_impact": [2],
                "llm_asset": ["US500"],
            }
        ),
        "scores_direct_events_usa500.parquet": pd.DataFrame(
            {"seendate": [timestamp], "url": ["d"], "title": ["Fed update"], "sent": [-0.1]}
        ),
        "scores_llm_direct_events_usa500.parquet": pd.DataFrame(
            {
                "seendate": [timestamp], "url": ["d"], "title": ["Fed update"],
                "llm_sent": [-0.3], "llm_relevance": [0.8], "llm_impact": [2],
                "llm_asset": ["macro"],
            }
        ),
        "gdelt_usa500.parquet": pd.DataFrame(
            {"seendate": [timestamp], "title": ["US stocks rise"], "tone": [1.0]}
        ),
        "fred_calendar.parquet": pd.DataFrame({"release_time": [timestamp]}),
    }
    for name in tables:
        (tmp_path / name).touch()
    (tmp_path / "index_deepseek_identity.json").write_text(
        json.dumps({"identity": IDENTITY_FIELDS}), encoding="utf-8"
    )
    for prefix in ("", "_direct_events"):
        classic = tmp_path / f"scores{prefix}_usa500.manifest.json"
        deepseek = tmp_path / f"scores_llm{prefix}_usa500.manifest.json"
        common = {
            "stream": "usa500",
            "source_prefix": "gdelt" if not prefix else "direct_events",
            "cutoff_exclusive": "2026-04-01T00:00:00+00:00",
            "source_sha256_scope": "canonical_pre_cutoff_scoring_rows",
            "source_sha256": "h" * 64,
            "complete": True,
        }
        classic.write_text(json.dumps(common), encoding="utf-8")
        deepseek.write_text(json.dumps({**common, "identity": IDENTITY_FIELDS}), encoding="utf-8")

    def fake_read(path, *args, **kwargs):
        return tables[Path(path).name].copy()

    monkeypatch.setattr(index_sentiment, "RAW_DIR", tmp_path)
    monkeypatch.setattr(index_sentiment.pd, "read_parquet", fake_read)
    monkeypatch.setattr(
        index_sentiment,
        "_first_seen_with_echoes",
        lambda frame: frame.assign(echo_count=1),
    )
    monkeypatch.setattr("sentiment.dedup.first_seen_only", lambda frame: frame)
    return index_sentiment


def test_deepseek_full_adds_only_structured_fields_to_exact_matched_base(
    tmp_path: Path, monkeypatch
):
    sentiment = _install_fixture(tmp_path, monkeypatch)
    bars = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")

    matched = sentiment.build_matched_index_features("usa500", bars, scorer="llm")
    full = sentiment.build_deepseek_full_features("usa500", bars)

    pd.testing.assert_frame_equal(full[list(sentiment.MATCHED_FEATURES)], matched)
    assert list(full.columns) == list(sentiment.DEEPSEEK_FULL_FEATURES)
    assert "sent_direct_pulse" not in full.columns


def test_index_sentiment_consumers_reject_incomplete_score_manifest(tmp_path: Path, monkeypatch):
    sentiment = _install_fixture(tmp_path, monkeypatch)
    path = tmp_path / "scores_llm_usa500.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["complete"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    bars = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")

    try:
        sentiment.build_matched_index_features("usa500", bars, scorer="llm")
    except ValueError as error:
        assert "complete" in str(error)
    else:
        raise AssertionError("incomplete scorer manifest was accepted")


def test_future_echoes_never_change_first_publication_weight():
    from features.index_sentiment import _first_seen_with_echoes

    first = pd.DataFrame(
        {
            "seendate": pd.to_datetime(["2024-01-01 00:00"], utc=True),
            "url": ["a"],
            "title": ["US stocks rise after Fed decision"],
        }
    )
    echoed = pd.concat(
        [
            first,
            pd.DataFrame(
                {
                    "seendate": pd.to_datetime(["2024-01-01 01:00"], utc=True),
                    "url": ["b"],
                    "title": ["US stocks rise after Fed decision"],
                }
            ),
        ],
        ignore_index=True,
    )

    before = _first_seen_with_echoes(first)
    after = _first_seen_with_echoes(echoed)

    assert before.iloc[0]["echo_count"] == after.iloc[0]["echo_count"] == 1


def test_index_sentiment_rejects_different_raw_snapshot_hashes(tmp_path: Path, monkeypatch):
    sentiment = _install_fixture(tmp_path, monkeypatch)
    classic_path = tmp_path / "scores_usa500.manifest.json"
    llm_path = tmp_path / "scores_llm_usa500.manifest.json"
    classic = json.loads(classic_path.read_text(encoding="utf-8"))
    llm = json.loads(llm_path.read_text(encoding="utf-8"))
    classic["source_sha256"] = "a" * 64
    llm["source_sha256"] = "b" * 64
    classic_path.write_text(json.dumps(classic), encoding="utf-8")
    llm_path.write_text(json.dumps(llm), encoding="utf-8")
    bars = pd.date_range("2024-01-01", periods=8, freq="15min", tz="UTC")

    try:
        sentiment.build_matched_index_features("usa500", bars, scorer="llm")
    except ValueError as error:
        assert "snapshot" in str(error)
    else:
        raise AssertionError("different scoring snapshots were accepted")
