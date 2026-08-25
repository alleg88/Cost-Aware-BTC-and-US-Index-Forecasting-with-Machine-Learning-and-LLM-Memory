from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sentiment.index_scoring import (
    CUTOFF,
    DEBERTA_REVISION,
    DEEPSEEK_TAG,
    ScorerIdentity,
    cache_key,
    canonical_scoring_source_hash,
    load_saved_deepseek_identity,
    prepare_scoring_rows,
    resolve_deepseek_identity,
    score_deberta_index,
    score_deepseek_index,
)


FULL_DIGEST = "d3f1c8744721" + "a" * 52


def _identity() -> ScorerIdentity:
    return ScorerIdentity(
        tag=DEEPSEEK_TAG,
        digest=FULL_DIGEST,
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


def test_scoring_cutoff_precedes_dedup_and_todo():
    source = pd.DataFrame(
        {
            "seendate": pd.to_datetime(
                ["2024-01-01", "2026-04-02", "2025-02-01"], utc=True
            ),
            "url": ["pre", "q2", "second"],
            "title": ["Same headline", "Same headline", "Another headline"],
        }
    )

    rows = prepare_scoring_rows(source, end_exclusive=CUTOFF)

    assert rows["seendate"].max() < CUTOFF
    assert rows["url"].tolist() == ["pre", "second"]
    assert "q2" not in rows["url"].tolist()


def test_path_scoring_uses_predicate_pushdown_and_hash_excludes_q2(
    tmp_path: Path, monkeypatch
):
    source = pd.DataFrame(
        {
            "seendate": pd.to_datetime(["2025-01-01", "2026-04-02"], utc=True),
            "url": ["pre", "q2"],
            "title": ["Known headline", "Future version one"],
        }
    )
    path = tmp_path / "source.parquet"
    source.to_parquet(path)
    observed = {}
    original = pd.read_parquet

    def capture(path, *args, **kwargs):
        observed["filters"] = kwargs.get("filters")
        return original(path, *args, **kwargs)

    monkeypatch.setattr("sentiment.index_scoring.pd.read_parquet", capture)
    first = prepare_scoring_rows(path)
    first_hash = canonical_scoring_source_hash(first)
    source.loc[1, "title"] = "Future version two"
    source.to_parquet(path)
    second = prepare_scoring_rows(path)

    assert observed["filters"]
    assert any(term[0] == "seendate" and term[1] == "<" for term in observed["filters"])
    assert first_hash == canonical_scoring_source_hash(second)
    assert first["seendate"].max() < CUTOFF


def test_deberta_pre_q2_output_and_manifest_stamp_model_revision(tmp_path: Path):
    pd.DataFrame(
        {
            "seendate": pd.to_datetime(["2025-01-01"], utc=True),
            "url": ["pre"],
            "title": ["Stocks rise"],
        }
    ).to_parquet(tmp_path / "gdelt_usa500.parquet", index=False)

    output = score_deberta_index(
        "usa500",
        raw_dir=tmp_path,
        scorer=lambda texts: [0.2] * len(texts),
    )
    scored = pd.read_parquet(output)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text("utf-8"))

    assert scored["revision"].tolist() == [DEBERTA_REVISION]
    assert manifest["revision"] == DEBERTA_REVISION


def test_deepseek_requires_0731_full_digest_and_complete_cache_key():
    tags = {
        "models": [
            {
                "name": DEEPSEEK_TAG,
                "model": DEEPSEEK_TAG,
                "digest": FULL_DIGEST,
                "remote_model": "deepseek-v4-flash:0731",
                "remote_host": "https://ollama.com",
                "capabilities": ["completion", "thinking"],
            }
        ]
    }

    identity = resolve_deepseek_identity(tags, ollama_version="0.32.9")

    assert identity.tag == "deepseek-v4-flash:0731-cloud"
    assert identity.digest == FULL_DIGEST
    assert identity.registry_digest_prefix == "031ce2a95446"
    assert identity.num_ctx == 8192
    assert identity.batch_size == 10
    assert identity.batch_protocol == "indexed-json-object-v1"
    assert len(identity.scorer_implementation_hash) == 64
    assert cache_key("usa500", "headline", identity) != cache_key(
        "usatech", "headline", identity
    )


def test_saved_identity_resume_preserves_timestamp_and_state_path(tmp_path: Path):
    identity = _identity()
    (tmp_path / "index_deepseek_identity.json").write_text(
        json.dumps({"identity": identity.to_dict()}), encoding="utf-8"
    )
    verified = []

    loaded = load_saved_deepseek_identity(
        tmp_path, verify_identity=lambda value: verified.append(value)
    )

    from sentiment.index_scoring import _state_path

    assert loaded == identity
    assert loaded.pulled_at_utc == "2026-08-12T00:00:00+00:00"
    assert _state_path(tmp_path, "usa500", "gdelt", loaded) == _state_path(
        tmp_path, "usa500", "gdelt", identity
    )
    assert verified == [identity]


def test_mutable_cloud_alias_cannot_satisfy_0731_identity():
    tags = {
        "models": [
            {
                "name": "deepseek-v4-flash:cloud",
                "digest": FULL_DIGEST,
                "remote_model": "deepseek-v4-flash:preview",
                "remote_host": "https://ollama.com",
            }
        ]
    }

    with pytest.raises(RuntimeError, match="0731-cloud"):
        resolve_deepseek_identity(tags, ollama_version="0.32.9")


def test_deepseek_scoring_is_cutoff_safe_resumable_and_identity_stamped(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "seendate": pd.to_datetime(
                ["2024-01-01", "2025-01-01", "2026-04-02"], utc=True
            ),
            "url": ["a", "b", "q2"],
            "title": ["Stocks rise", "Stocks fall", "Future headline"],
        }
    )
    raw.to_parquet(tmp_path / "gdelt_usa500.parquet")

    def fake_batch(titles, identity):
        assert identity.digest == FULL_DIGEST
        return [
            {
                "llm_sent": 0.25,
                "llm_relevance": 0.8,
                "llm_impact": 1,
                "llm_asset": "US500",
            }
            for _ in titles
        ]

    out = score_deepseek_index(
        "usa500",
        identity=_identity(),
        raw_dir=tmp_path,
        workers=1,
        batch_scorer=fake_batch,
        verify_identity=lambda identity: None,
    )
    scored = pd.read_parquet(out)
    manifest = json.loads(out.with_suffix(".manifest.json").read_text(encoding="utf-8"))

    assert len(scored) == 2
    assert scored["seendate"].max() < CUTOFF
    assert scored["model_digest"].eq(FULL_DIGEST).all()
    assert scored["prompt_hash"].eq("p" * 64).all()
    assert scored["cache_key"].nunique() == 2
    assert manifest["identity"]["tag"] == DEEPSEEK_TAG
    assert manifest["rows_after_cutoff"] == 2
    assert manifest["source_sha256_scope"] == "canonical_pre_cutoff_scoring_rows"
    assert manifest["complete"] is True


def test_limited_deepseek_run_writes_only_identity_scoped_state(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "seendate": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "url": ["a", "b"],
            "title": ["Stocks rise", "Stocks fall"],
        }
    )
    raw.to_parquet(tmp_path / "gdelt_usa500.parquet")

    state = score_deepseek_index(
        "usa500",
        identity=_identity(),
        raw_dir=tmp_path,
        workers=1,
        limit=1,
        batch_scorer=lambda titles, identity: [
            {
                "llm_sent": 0.1,
                "llm_relevance": 0.8,
                "llm_impact": 1,
                "llm_asset": "US500",
            }
            for _ in titles
        ],
        verify_identity=lambda identity: None,
    )

    assert "_" + state.stem.rsplit("_", 1)[-1] in state.name
    assert not (tmp_path / "scores_llm_usa500.parquet").exists()
    assert not (tmp_path / "scores_llm_usa500.manifest.json").exists()


def test_deepseek_state_rejects_recomputed_cache_key_mismatch(tmp_path: Path):
    identity = _identity()
    state_name = "index_deepseek_state_gdelt_usa500_" + "x" * 16 + ".parquet"
    legacy = pd.DataFrame(
        {
            "instrument": ["usa500"],
            "title_norm": ["stocks rise"],
            "cache_key": ["tampered"],
            "llm_sent": [0.1],
            "llm_relevance": [0.8],
            "llm_impact": [1],
            "llm_asset": ["US500"],
            "model": [identity.tag],
            "model_digest": [identity.digest],
            "remote_model": [identity.remote_model],
            "remote_host": [identity.remote_host],
            "registry_digest_prefix": [identity.registry_digest_prefix],
            "prompt_hash": [identity.prompt_hash],
            "schema_hash": [identity.schema_hash],
            "think": [identity.think],
            "temperature": [identity.temperature],
            "num_ctx": [identity.num_ctx],
            "batch_size": [identity.batch_size],
            "batch_protocol": [identity.batch_protocol],
            "scorer_implementation_hash": [identity.scorer_implementation_hash],
        }
    ).to_parquet(tmp_path / state_name)

    from sentiment.index_scoring import _load_deepseek_state

    with pytest.raises(ValueError, match="cache key"):
        _load_deepseek_state(tmp_path / state_name, identity, "usa500")


def test_incomplete_existing_deepseek_cache_is_rejected(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "seendate": pd.to_datetime(["2024-01-01"], utc=True),
            "url": ["a"],
            "title": ["Stocks rise"],
        }
    )
    raw.to_parquet(tmp_path / "gdelt_usa500.parquet")
    legacy = pd.DataFrame(
        {
            "title_norm": ["stocks rise"],
            "llm_sent": [0.1],
            "model": [DEEPSEEK_TAG],
        }
    )
    from sentiment.index_scoring import _state_path

    legacy.to_parquet(_state_path(tmp_path, "usa500", "gdelt", _identity()))

    with pytest.raises(ValueError, match="identity columns"):
        score_deepseek_index(
            "usa500",
            identity=_identity(),
            raw_dir=tmp_path,
            workers=1,
            batch_scorer=lambda titles, identity: [],
            verify_identity=lambda identity: None,
        )
