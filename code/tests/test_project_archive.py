"""Project archives preserve references without transferring local state."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


def bootstrap_module():
    path = Path(__file__).parents[2] / "prepare_project.py"
    assert path.is_file(), "ZIP exports need a standalone preparation entry point"
    spec = importlib.util.spec_from_file_location("prepare_project", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def snapshot(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("Research artifact\n", encoding="utf-8")
    paths = ["README.md", "code/pyproject.toml"]
    manifest = {
        path: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest()
        for path in paths
    }
    (tmp_path / "release_files.json").write_text(json.dumps(manifest), encoding="utf-8")
    return paths


def test_zip_preparation_tracks_only_verified_files_and_is_idempotent(tmp_path):
    expected = snapshot(tmp_path)
    (tmp_path / ".env").write_text("private-local-setting", encoding="utf-8")
    prepare = bootstrap_module().prepare
    prepare(tmp_path)
    files = subprocess.check_output(["git", "ls-files"], cwd=tmp_path, text=True).splitlines()
    assert files == expected
    before = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path)
    (tmp_path / "README.md").write_text("User edit\n", encoding="utf-8")
    prepare(tmp_path)
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tmp_path) == before
    assert (tmp_path / "README.md").read_text("utf-8") == "User edit\n"


@pytest.mark.parametrize("relative", [
    "code/.source_evidence/files/input.csv",
    "code/data/btcusdt_m15_2024_2025.parquet",
    "code/data/btcusdt_positioning_m15_2024_2026.parquet",
])
def test_embedded_rebuild_inputs_are_verified_but_not_git_references(tmp_path, relative):
    expected = snapshot(tmp_path)
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"value\n1\n")
    manifest_path = tmp_path / "release_files.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    bootstrap_module().prepare(tmp_path)
    assert subprocess.check_output(["git", "ls-files"], cwd=tmp_path, text=True).splitlines() == expected
    assert source.read_bytes() == b"value\n1\n"


def test_corrupt_partial_input_is_rejected_before_git_initialisation(tmp_path):
    snapshot(tmp_path)
    source = tmp_path / "code/data/btcusdt_m15_2024_2025.parquet"
    source.parent.mkdir()
    source.write_bytes(b"supplied input")
    manifest_path = tmp_path / "release_files.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["code/data/btcusdt_m15_2024_2025.parquet"] = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    source.write_bytes(b"modified input")
    with pytest.raises(ValueError, match="Archive checksum mismatch"):
        bootstrap_module().prepare(tmp_path)
    assert not (tmp_path / ".git").exists()


@pytest.mark.parametrize("kind", ["changed", "missing", "escape", "git_path", "empty"])
def test_zip_preparation_rejects_bad_manifest_before_creating_git(tmp_path, kind):
    snapshot(tmp_path)
    if kind == "changed":
        (tmp_path / "README.md").write_text("corrupt", encoding="utf-8")
    elif kind == "missing":
        (tmp_path / "README.md").unlink()
    else:
        bad = {"escape": {"../outside": "0" * 64}, "git_path": {".git/config": "0" * 64}, "empty": {}}
        (tmp_path / "release_files.json").write_text(json.dumps(bad[kind]), encoding="utf-8")
    with pytest.raises(ValueError):
        bootstrap_module().prepare(tmp_path)
    assert not (tmp_path / ".git").exists()
