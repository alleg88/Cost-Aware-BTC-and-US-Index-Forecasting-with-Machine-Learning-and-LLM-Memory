from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pandas as pd


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    _git(tmp_path, "config", "core.autocrlf", "false")
    return tmp_path


def _task(output: str, comparison):
    from experiments.rebuild_graph import RebuildTask

    return RebuildTask.from_dict(
        {
            "id": "result",
            "depends_on": [],
            "module": "fixture.result",
            "args": [],
            "inputs": [],
            "outputs": [output],
            "profile": "canonical",
            "comparison": comparison,
        }
    )


def test_exact_comparator_detects_a_changed_tracked_artifact(tmp_path):
    assert importlib.util.find_spec("experiments.rebuild_compare") is not None
    from experiments.rebuild_compare import compare_task_to_git

    root = _repository(tmp_path)
    output = root / "result.json"
    (root / ".gitattributes").write_text(
        "result.json text eol=crlf\n", encoding="utf-8"
    )
    output.write_bytes(b'{"value": 1}\r\n')
    _git(root, "add", ".gitattributes", "result.json")
    _git(root, "commit", "-m", "reference")
    task = _task("result.json", "exact")

    matched = compare_task_to_git(task, code_root=root, repository_root=root)
    output.write_bytes(b'{"value": 2}\r\n')
    changed = compare_task_to_git(task, code_root=root, repository_root=root)

    assert matched.status == "matched" and matched.compared_files == 1
    assert changed.status == "different"
    assert changed.issues[0].path == "result.json"


def test_numeric_comparator_honours_tolerance_and_excluded_columns(tmp_path):
    assert importlib.util.find_spec("experiments.rebuild_compare") is not None
    from experiments.rebuild_compare import compare_task_to_git

    root = _repository(tmp_path)
    output = root / "result.parquet"
    pd.DataFrame({"metric": [1.0, 2.0], "scored_at": ["old", "old"]}).to_parquet(
        output, index=False
    )
    _git(root, "add", "result.parquet")
    _git(root, "commit", "-m", "reference")
    task = _task(
        "result.parquet",
        {"mode": "numeric", "exclude_columns": ["scored_at"], "atol": 1e-6, "rtol": 0.0},
    )
    pd.DataFrame(
        {"metric": [1.0 + 5e-7, 2.0], "scored_at": ["new", "new"]}
    ).to_parquet(output, index=False)

    within = compare_task_to_git(task, code_root=root, repository_root=root)
    pd.DataFrame({"metric": [1.01, 2.0], "scored_at": ["new", "new"]}).to_parquet(
        output, index=False
    )
    outside = compare_task_to_git(task, code_root=root, repository_root=root)

    assert within.status == "matched"
    assert outside.status == "different"


def test_directory_comparator_reports_a_missing_tracked_file(tmp_path):
    assert importlib.util.find_spec("experiments.rebuild_compare") is not None
    from experiments.rebuild_compare import compare_task_to_git

    root = _repository(tmp_path)
    output = root / "results"
    output.mkdir()
    (output / "one.csv").write_text("value\n1\n", encoding="utf-8")
    (output / "two.json").write_text('{"value": 2}\n', encoding="utf-8")
    _git(root, "add", "results")
    _git(root, "commit", "-m", "reference")
    (output / "two.json").unlink()

    report = compare_task_to_git(
        _task("results", "row_exact"),
        code_root=root,
        repository_root=root,
    )

    assert report.status == "different"
    assert report.compared_files == 2
    assert report.issues[0].reason == "missing rebuilt artifact"
