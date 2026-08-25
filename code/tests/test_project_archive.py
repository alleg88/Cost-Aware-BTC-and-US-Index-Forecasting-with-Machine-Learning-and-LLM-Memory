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
