from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

import experiments.final_q2_lockbox_state as state_module
from experiments.final_q2_lockbox_contract import Q2_END, Q2_START
from experiments.final_q2_lockbox_state import OpeningIdentity, begin_global_open
from sentiment.index_scoring import (
    DEEPSEEK_TAG,
    ScorerIdentity,
    prepare_scoring_rows,
    score_deberta_index,
    score_deepseek_index,
)


def _opening_identity(**changes) -> OpeningIdentity:
    base = OpeningIdentity(
        implementation_commit="a" * 40,
        manifest_commit="b" * 40,
        protocol_hash="c" * 64,
        manifest_sha256="d" * 64,
        q2_source_hashes={"sentiment": "e" * 64},
    )
    return replace(base, **changes)


def _scorer_identity() -> ScorerIdentity:
    return ScorerIdentity(
        tag=DEEPSEEK_TAG,
        digest="d3f1c8744721" + "a" * 52,
        remote_model="deepseek-v4-flash:0731",
        remote_host="https://ollama.com",
        registry_digest_prefix="031ce2a95446",
        ollama_version="0.32.9",
        pulled_at_utc="2026-08-12T00:00:00+00:00",
        prompt_hash="p" * 64,
        schema_hash="s" * 64,
        temperature=0.0,
        think="low",
        num_ctx=8192,
        batch_size=10,
        batch_protocol="indexed-json-object-v1",
        scorer_implementation_hash="i" * 64,
    )


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seendate": pd.to_datetime(
                [
                    "2026-03-31 23:59:00Z",
                    "2026-04-01 00:00:00Z",
                    "2026-06-30 23:59:00Z",
                    "2026-07-01 00:00:00Z",
                ]
            ),
            "url": ["pre", "q2-first", "q2-last", "post"],
            "title": ["Pre", "First", "Last", "Post"],
        }
    )


def _open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> OpeningIdentity:
    identity = _opening_identity()
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")
    begin_global_open(identity, root=tmp_path)
    return identity


def test_scoring_rows_use_exact_half_open_q2_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _open(tmp_path, monkeypatch)

    rows = prepare_scoring_rows(
        _source(),
        start_inclusive=Q2_START,
        end_exclusive=Q2_END,
        opening_identity=identity,
    )

    assert rows["url"].tolist() == ["q2-first", "q2-last"]
    assert rows["seendate"].ge(Q2_START).all()
    assert rows["seendate"].lt(Q2_END).all()


def test_q2_scoring_rejects_identity_without_physical_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", tmp_path / "OPENED.json")

    with pytest.raises(PermissionError, match="OPENED"):
        prepare_scoring_rows(
            _source(),
            start_inclusive=Q2_START,
            end_exclusive=Q2_END,
            opening_identity=_opening_identity(),
        )


def test_default_pre_q2_scoring_contract_is_unchanged() -> None:
    rows = prepare_scoring_rows(_source())

    assert rows["url"].tolist() == ["pre"]
    assert rows["seendate"].lt(Q2_START).all()


def test_path_scoring_pushes_both_q2_predicates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _open(tmp_path, monkeypatch)
    path = tmp_path / "source.parquet"
    _source().to_parquet(path, index=False)
    observed = {}
    original = pd.read_parquet

    def capture(path, *args, **kwargs):
        observed["filters"] = kwargs.get("filters")
        return original(path, *args, **kwargs)

    monkeypatch.setattr("sentiment.index_scoring.pd.read_parquet", capture)
    prepare_scoring_rows(
        path,
        start_inclusive=Q2_START,
        end_exclusive=Q2_END,
        opening_identity=identity,
    )

    assert ("seendate", ">=", Q2_START.to_pydatetime()) in observed["filters"]
    assert ("seendate", "<", Q2_END.to_pydatetime()) in observed["filters"]


def test_deberta_q2_output_is_isolated_and_interval_stamped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _open(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    output = tmp_path / "q2_scores"
    raw.mkdir()
    _source().to_parquet(raw / "gdelt_usa500.parquet", index=False)

    path = score_deberta_index(
        "usa500",
        raw_dir=raw,
        output_dir=output,
        start_inclusive=Q2_START,
        end_exclusive=Q2_END,
        opening_identity=identity,
        scorer=lambda titles: [0.25 for _ in titles],
    )
    scored = pd.read_parquet(path)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert path.parent == output
    assert len(scored) == 2
    assert manifest["start_inclusive"] == Q2_START.isoformat()
    assert manifest["cutoff_exclusive"] == Q2_END.isoformat()
    assert not (raw / path.name).exists()


def test_deberta_q2_default_scorer_is_local_and_revision_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _open(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    output = tmp_path / "q2_scores"
    raw.mkdir()
    _source().to_parquet(raw / "gdelt_usa500.parquet", index=False)
    observed = {}

    def builder(batch_size, **kwargs):
        observed.update(kwargs)
        return lambda titles: [0.25 for _ in titles]

    monkeypatch.setattr("sentiment.index_scoring._build_scorer", builder)
    path = score_deberta_index(
        "usa500",
        raw_dir=raw,
        output_dir=output,
        start_inclusive=Q2_START,
        end_exclusive=Q2_END,
        opening_identity=identity,
    )
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert observed == {
        "revision": "9e10915c245a80a89b18d1ac51350e093c7bb35a",
        "local_files_only": True,
    }
    assert manifest["revision"] == observed["revision"]
    assert manifest["local_files_only"] is True


def test_deepseek_q2_state_and_output_are_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _open(tmp_path, monkeypatch)
    raw = tmp_path / "raw"
    output = tmp_path / "q2_scores"
    raw.mkdir()
    _source().to_parquet(raw / "gdelt_usatech.parquet", index=False)

    path = score_deepseek_index(
        "usatech",
        identity=_scorer_identity(),
        raw_dir=raw,
        output_dir=output,
        start_inclusive=Q2_START,
        end_exclusive=Q2_END,
        opening_identity=identity,
        workers=1,
        batch_scorer=lambda titles, scorer_identity: [
            {
                "llm_sent": 0.1,
                "llm_relevance": 0.8,
                "llm_impact": 1,
                "llm_asset": "USTECH",
            }
            for _ in titles
        ],
        verify_identity=lambda scorer_identity: None,
    )
    manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert path.parent == output
    assert manifest["start_inclusive"] == Q2_START.isoformat()
    assert manifest["identity"]["remote_model"] == "deepseek-v4-flash:0731"
    assert list(output.glob("index_deepseek_state_*.parquet"))
    assert not list(raw.glob("index_deepseek_state_*.parquet"))


def test_feature_builder_forwards_explicit_q2_score_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from features import index_sentiment

    identity = _opening_identity()
    bars = pd.date_range(Q2_START, periods=2, freq="15min")
    calls = []

    def fake_decay(stream, bar_index, **kwargs):
        calls.append(kwargs)
        zeros = pd.Series(0.0, index=bar_index)
        return zeros, zeros

    monkeypatch.setattr(index_sentiment, "_matched_decay", fake_decay)
    monkeypatch.setattr(
        index_sentiment,
        "_tone_and_macro",
        lambda stream, bar_index, **kwargs: (
            pd.Series(0.0, index=bar_index),
            pd.Series(0.0, index=bar_index),
        ),
    )

    index_sentiment.build_matched_index_features(
        "usa500",
        bars,
        scorer="classic",
        score_root=tmp_path,
        available_start=Q2_START,
        available_end=Q2_END,
        opening_identity=identity,
    )

    assert len(calls) == 2
    assert all(call["score_root"] == tmp_path for call in calls)
    assert all(call["available_start"] == Q2_START for call in calls)
    assert all(call["opening_identity"] == identity for call in calls)


def test_q2_matched_decay_includes_bound_pre_q2_score_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from features import index_sentiment

    warmup = pd.DataFrame(
        {
            "seendate": [Q2_START - pd.Timedelta(minutes=15)],
            "url": ["https://example.com/story"],
            "title": ["Markets enter the quarter with momentum"],
            "title_norm": ["markets enter the quarter with momentum"],
            "matched_sentiment": [1.0],
            "echo_count": [1],
            "llm_relevance": [1.0],
            "llm_impact": [1],
            "llm_asset": ["US500"],
        }
    )
    empty = warmup.iloc[0:0].copy()
    observed = []

    def load_scores(stream, scorer, **kwargs):
        observed.append(kwargs["score_root"])
        return warmup.copy() if kwargs["score_root"] == "warm" else empty.copy()

    monkeypatch.setattr(index_sentiment, "_load_matched_scores", load_scores)
    bars = pd.date_range(Q2_START, periods=2, freq="15min")
    decay, count = index_sentiment._matched_decay(
        "usa500",
        bars,
        scorer="classic",
        direct=False,
        halflife_h=6.0,
        score_root="q2",
        warmup_score_root="warm",
        available_start=Q2_START,
        available_end=Q2_END,
        opening_identity=_opening_identity(),
    )

    assert observed == ["q2", "warm"]
    assert decay.iloc[0] > 0.0
    assert count.iloc[0] == 1.0


def test_q2_tone_and_macro_continuous_context_has_no_lower_cutoff(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from features import index_sentiment

    identity = _open(tmp_path, monkeypatch)
    observed = []

    def read(path, *args, **kwargs):
        observed.append(kwargs["filters"])
        if Path(path).name == "fred_calendar.parquet":
            return pd.DataFrame(
                {"release_time": [Q2_START - pd.Timedelta(hours=1)]}
            )
        return pd.DataFrame(
            {
                "seendate": [Q2_START - pd.Timedelta(hours=1)],
                "title": ["Pre-quarter market context"],
                "tone": [1.0],
            }
        )

    monkeypatch.setattr(index_sentiment.pd, "read_parquet", read)
    bars = pd.date_range(Q2_START, periods=2, freq="15min")
    tone, macro = index_sentiment._tone_and_macro(
        "usa500",
        bars,
        source_root=tmp_path,
        available_start=Q2_START,
        available_end=Q2_END,
        opening_identity=identity,
        continuous_context=True,
    )

    assert all(
        not any(predicate[1] == ">=" for predicate in filters)
        for filters in observed
    )
    assert tone.iloc[0] > 0.0
    assert macro.iloc[0] > 0.0
