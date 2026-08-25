from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.external_evidence import sha256_file
from experiments.frozen_evidence import FrozenEvidenceError, stage_frozen_evidence


def _bundle(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    evidence = tmp_path / "evidence"
    source = evidence / "files" / "sentiment" / "raw" / "scores_llm_btc.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen cloud output")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "files": [
            {
                "id": "llm_scores",
                "class": "frozen_nondeterministic",
                "provider": "Ollama Cloud",
                "path": "sentiment/raw/scores_llm_btc.parquet",
                "stage_path": "sentiment/raw/scores_llm_btc.parquet",
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        ],
    }
    return evidence, manifest, source


def test_llm_stage_requires_hash_match_and_never_calls_provider(tmp_path, monkeypatch):
    evidence, manifest, _ = _bundle(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("OLLAMA_API_KEY", "must-not-be-used")
    monkeypatch.setattr(
        "requests.get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("network used")),
    )

    report = stage_frozen_evidence(evidence, work, manifest, kind="llm_scores")

    assert report.files == 1
    assert report.bytes == len(b"frozen cloud output")
    assert (work / "sentiment" / "raw" / "scores_llm_btc.parquet").read_bytes() == b"frozen cloud output"


def test_frozen_stage_fails_closed_on_hash_mismatch(tmp_path):
    evidence, manifest, source = _bundle(tmp_path)
    source.write_bytes(b"FROZEN CLOUD OUTPUT")
    work = tmp_path / "work"
    work.mkdir()

    with pytest.raises(FrozenEvidenceError, match="hash mismatch"):
        stage_frozen_evidence(evidence, work, manifest, kind="llm_scores")


def test_frozen_stage_rejects_manifest_path_escape(tmp_path):
    evidence, manifest, _ = _bundle(tmp_path)
    row = manifest["files"][0]
    assert isinstance(row, dict)
    row["stage_path"] = "../outside.json"
    work = tmp_path / "work"
    work.mkdir()

    with pytest.raises(FrozenEvidenceError, match="escapes"):
        stage_frozen_evidence(evidence, work, manifest, kind="llm_scores")


def test_frozen_stage_accepts_manifest_path_and_is_idempotent(tmp_path):
    evidence, manifest, _ = _bundle(tmp_path)
    manifest_path = evidence / "source_evidence_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()

    first = stage_frozen_evidence(evidence, work, manifest_path, kind="llm_scores")
    second = stage_frozen_evidence(evidence, work, manifest_path, kind="llm_scores")

    assert first == second


def test_frozen_stage_restores_registered_scientific_handoffs(tmp_path):
    evidence = tmp_path / "evidence"
    source = evidence / "files" / "experiments" / "cache" / "handoff" / "state.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"frozen handoff")
    work = tmp_path / "work"
    work.mkdir()
    manifest = {
        "files": [
            {
                "id": "channel_handoff_n",
                "class": "frozen_handoff",
                "path": "experiments/cache/handoff/state.json",
                "stage_path": "experiments/cache/handoff/state.json",
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        ]
    }

    report = stage_frozen_evidence(
        evidence, work, manifest, kind="channel_handoffs"
    )

    assert report.paths == ("experiments/cache/handoff/state.json",)
    assert (work / report.paths[0]).read_bytes() == b"frozen handoff"


def test_snapshot_class_stages_every_registered_snapshot_record(tmp_path):
    evidence, manifest, _ = _bundle(tmp_path)
    source = evidence / "files" / "data" / "raw" / "index.csv"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"index source")
    files = manifest["files"]
    assert isinstance(files, list)
    files.append(
        {
            "id": "jforex_index_minutes",
            "class": "snapshot_raw",
            "provider": "JForex/Dukascopy",
            "path": "data/raw/index.csv",
            "stage_path": "data/raw/index.csv",
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    )
    work = tmp_path / "work"
    work.mkdir()

    report = stage_frozen_evidence(evidence, work, manifest, kind="snapshot_raw")

    assert report.paths == ("data/raw/index.csv",)
