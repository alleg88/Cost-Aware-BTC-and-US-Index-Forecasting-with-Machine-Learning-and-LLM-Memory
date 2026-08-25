from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.source_evidence import (
    EvidenceError,
    build_manifest,
    export_bundle,
    restore_bundle,
    verify_manifest,
)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write(path: Path, content: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return {
        "path": path.as_posix(),
        "bytes": len(content),
        "sha256": _sha256(content),
    }


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    code_root = tmp_path / "code"
    rows = []
    for relative, content in (
        ("data/raw/binance/BTCUSDT-15m-2024-01.zip", b"public-zip"),
        ("data/raw/binance/BTCUSDT-15m-2024-01.zip.CHECKSUM", b"checksum"),
        ("data/raw/binance/BTCUSDT-15m-2024-01.csv", b"derived-csv"),
        ("sentiment/raw/trump_truth_posts.csv", b"truth"),
        ("sentiment/raw/scores_llm_btc.parquet", b"cloud-score"),
        ("experiments/cache/tuning/prediction.parquet", b"derived-model"),
        ("sentiment/secrets/provider.key", b"never-copy"),
    ):
        row = _write(code_root / relative, content)
        row["path"] = relative
        rows.append(row)

    policy = {
        "schema_version": 1,
        "rules": [
            {
                "id": "public",
                "class": "public_raw",
                "glob": "data/raw/binance/**/*.{zip,zip.CHECKSUM}",
                "provider": "Binance",
                "stage": "data/raw/binance",
            },
            {
                "id": "truth",
                "class": "snapshot_raw",
                "glob": "sentiment/raw/trump_truth_posts.csv",
                "provider": "Truth Social archive",
                "stage": "sentiment/raw",
            },
            {
                "id": "llm",
                "class": "frozen_nondeterministic",
                "glob": "sentiment/raw/scores_llm_*.parquet",
                "provider": "Cloud LLM",
                "stage": "sentiment/raw",
            },
        ],
    }
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy), encoding="utf-8")
    full_manifest = tmp_path / "full.json"
    full_manifest.write_text(json.dumps({"files": rows}), encoding="utf-8")
    return code_root, policy_path, full_manifest


def test_source_manifest_excludes_derived_files_extracted_csvs_and_secrets(
    tmp_path: Path,
) -> None:
    code_root, policy_path, full_manifest = _fixture(tmp_path)

    payload = build_manifest(code_root, policy_path, full_manifest)

    assert [row["path"] for row in payload["files"]] == [
        "data/raw/binance/BTCUSDT-15m-2024-01.zip",
        "data/raw/binance/BTCUSDT-15m-2024-01.zip.CHECKSUM",
        "sentiment/raw/scores_llm_btc.parquet",
        "sentiment/raw/trump_truth_posts.csv",
    ]
    assert payload["class_summary"] == {
        "frozen_nondeterministic": {"bytes": 11, "files": 1},
        "public_raw": {"bytes": 18, "files": 2},
        "snapshot_raw": {"bytes": 5, "files": 1},
    }
    assert len(payload["manifest_identity"]) == 64


def test_manifest_verification_reports_missing_and_changed_files(tmp_path: Path) -> None:
    code_root, policy_path, full_manifest = _fixture(tmp_path)
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(build_manifest(code_root, policy_path, full_manifest)),
        encoding="utf-8",
    )

    ready = verify_manifest(code_root, manifest)
    assert ready["status"] == "READY"
    (code_root / "sentiment/raw/trump_truth_posts.csv").write_bytes(b"changed")
    (code_root / "sentiment/raw/scores_llm_btc.parquet").unlink()

    report = verify_manifest(code_root, manifest)

    assert report["status"] == "NOT_READY"
    assert report["missing"] == ["sentiment/raw/scores_llm_btc.parquet"]
    assert report["mismatched"] == ["sentiment/raw/trump_truth_posts.csv"]


def test_snapshot_only_export_omits_public_downloads_and_verifies(tmp_path: Path) -> None:
    code_root, policy_path, full_manifest = _fixture(tmp_path)
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(build_manifest(code_root, policy_path, full_manifest)),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"

    report = export_bundle(code_root, bundle, manifest, include_public=False)

    assert report == {
        "status": "READY",
        "file_count": 2,
        "total_bytes": 16,
        "public_files_omitted": 2,
    }
    assert not (
        bundle / "files/data/raw/binance/BTCUSDT-15m-2024-01.zip"
    ).exists()
    assert (bundle / "files/sentiment/raw/trump_truth_posts.csv").is_file()
    assert (bundle / "source_evidence_manifest.json").is_file()
    assert verify_manifest(bundle, bundle / "source_evidence_manifest.json")[
        "status"
    ] == "READY"


def test_restore_copies_only_bundle_members_into_an_evidence_root(tmp_path: Path) -> None:
    code_root, policy_path, full_manifest = _fixture(tmp_path)
    manifest = tmp_path / "source.json"
    manifest.write_text(
        json.dumps(build_manifest(code_root, policy_path, full_manifest)),
        encoding="utf-8",
    )
    bundle = tmp_path / "bundle"
    export_bundle(code_root, bundle, manifest, include_public=False)
    restored = tmp_path / "restored"

    report = restore_bundle(bundle, restored)

    assert report["status"] == "READY"
    assert (restored / "files/sentiment/raw/scores_llm_btc.parquet").is_file()
    assert not (restored / "files/data/raw/binance").exists()
    assert verify_manifest(restored, restored / "source_evidence_manifest.json")[
        "status"
    ] == "READY"


def test_bundle_rejects_nonempty_targets_and_escaping_manifest_paths(
    tmp_path: Path,
) -> None:
    code_root, policy_path, full_manifest = _fixture(tmp_path)
    manifest = tmp_path / "source.json"
    payload = build_manifest(code_root, policy_path, full_manifest)
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(EvidenceError, match="empty"):
        export_bundle(code_root, occupied, manifest)

    payload["files"][0]["path"] = "../escape.zip"
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(EvidenceError, match="escapes"):
        verify_manifest(code_root, bad_manifest)
