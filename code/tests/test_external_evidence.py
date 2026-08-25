from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path(__file__).parents[1]


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "experiments.external_evidence", *arguments],
        cwd=CODE_ROOT,
        text=True,
        capture_output=True,
    )


def test_external_evidence_snapshot_verifies_content_and_excludes_secrets(tmp_path):
    source = tmp_path / "code"
    files = {
        "data/raw/index.csv": b"time,price\n1,2\n",
        "sentiment/raw/news.parquet": b"parquet-placeholder",
        "experiments/cache/run/result.json": b'{"net": 1.0}\n',
        "sentiment/secrets/provider.key": b"never-publish",
        "data/helper.py": b"print('not evidence')\n",
    }
    for relative, content in files.items():
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    manifest = tmp_path / "external-evidence.json"

    snapshot = _run(
        "snapshot",
        "--code-root",
        str(source),
        "--manifest",
        str(manifest),
    )

    assert snapshot.returncode == 0, snapshot.stderr
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [entry["path"] for entry in payload["files"]] == [
        "data/raw/index.csv",
        "experiments/cache/run/result.json",
        "sentiment/raw/news.parquet",
    ]
    assert payload["total_bytes"] == sum(
        len(files[path])
        for path in (
            "data/raw/index.csv",
            "experiments/cache/run/result.json",
            "sentiment/raw/news.parquet",
        )
    )

    verified = _run(
        "verify",
        "--code-root",
        str(source),
        "--manifest",
        str(manifest),
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["status"] == "READY"

    bundle = tmp_path / "bundle"
    exported = _run(
        "export",
        "--code-root",
        str(source),
        "--manifest",
        str(manifest),
        "--target",
        str(bundle),
    )
    assert exported.returncode == 0, exported.stderr
    assert (bundle / "external_evidence_manifest.json").is_file()
    assert not (bundle / "sentiment/secrets/provider.key").exists()

    restored_root = tmp_path / "restored-code"
    restored_root.mkdir()
    restored = _run(
        "restore",
        "--code-root",
        str(restored_root),
        "--source",
        str(bundle),
        "--manifest",
        str(bundle / "external_evidence_manifest.json"),
    )
    assert restored.returncode == 0, restored.stderr
    restored_check = _run(
        "verify",
        "--code-root",
        str(restored_root),
        "--manifest",
        str(bundle / "external_evidence_manifest.json"),
    )
    assert restored_check.returncode == 0, restored_check.stderr

    (source / "sentiment/raw/news.parquet").write_bytes(b"changed")
    rejected = _run(
        "verify",
        "--code-root",
        str(source),
        "--manifest",
        str(manifest),
    )
    assert rejected.returncode == 1
    report = json.loads(rejected.stdout)
    assert report["status"] == "NOT_READY"
    assert report["mismatched"] == ["sentiment/raw/news.parquet"]
