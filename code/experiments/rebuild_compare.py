"""Compare rebuilt task outputs with the tracked Git reference artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import subprocess
from typing import Any, Mapping

import numpy as np
import pandas as pd

from experiments.rebuild_graph import RebuildTask


_VOLATILE_JSON_KEYS = frozenset(
    {"created_at_utc", "completed_at_utc", "recorded_at_utc", "updated_at_utc"}
)


@dataclass(frozen=True)
class ComparisonIssue:
    path: str
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "reason": self.reason}


@dataclass(frozen=True)
class TaskComparison:
    task_id: str
    status: str
    compared_files: int
    issues: tuple[ComparisonIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "compared_files": self.compared_files,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _git(repository_root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=False,
        capture_output=True,
        shell=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"git {' '.join(arguments)} failed: {message}")
    return result.stdout


def _code_prefix(code_root: Path, repository_root: Path) -> PurePosixPath:
    relative = code_root.resolve(strict=True).relative_to(
        repository_root.resolve(strict=True)
    )
    return PurePosixPath() if str(relative) == "." else PurePosixPath(relative.as_posix())


def _tracked_files(
    repository_root: Path,
    code_root: Path,
    output: str,
    revision: str,
) -> tuple[str, ...]:
    prefix = _code_prefix(code_root, repository_root)
    pathspec = (prefix / PurePosixPath(output)).as_posix()
    rows = _git(
        repository_root,
        "ls-tree",
        "-r",
        "--name-only",
        "-z",
        revision,
        "--",
        pathspec,
    )
    return tuple(
        value.decode("utf-8") for value in rows.split(b"\0") if value
    )


def _working_path(
    repository_path: str,
    code_root: Path,
    repository_root: Path,
) -> tuple[str, Path]:
    prefix = _code_prefix(code_root, repository_root)
    path = PurePosixPath(repository_path)
    relative = path.relative_to(prefix) if prefix.parts else path
    relative_text = relative.as_posix()
    return relative_text, code_root / Path(*relative.parts)


def _comparison_spec(task: RebuildTask) -> tuple[str, float, float, frozenset[str]]:
    if isinstance(task.comparison, str):
        mode = task.comparison
        payload: Mapping[str, object] = {}
    elif isinstance(task.comparison, Mapping):
        mode = str(task.comparison.get("mode", ""))
        payload = task.comparison
    else:
        raise ValueError(f"task {task.id} has an invalid comparison specification")
    if mode not in {"exact", "row_exact", "numeric"}:
        raise ValueError(f"task {task.id} has an unknown comparison mode: {mode!r}")
    raw_excluded = payload.get("exclude_columns", [])
    if not isinstance(raw_excluded, list) or not all(
        isinstance(value, str) for value in raw_excluded
    ):
        raise ValueError(f"task {task.id} exclude_columns must be a string list")
    return (
        mode,
        float(payload.get("atol", 0.0)),
        float(payload.get("rtol", 0.0)),
        frozenset(raw_excluded),
    )


def _table(reference: bytes, working: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    suffix = working.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(BytesIO(reference)), pd.read_parquet(working)
    if suffix == ".csv":
        return pd.read_csv(BytesIO(reference)), pd.read_csv(working)
    raise ValueError(f"unsupported table format: {working.suffix}")


def _compare_frames(
    reference: pd.DataFrame,
    working: pd.DataFrame,
    *,
    mode: str,
    atol: float,
    rtol: float,
    excluded: frozenset[str],
) -> None:
    reference = reference.drop(columns=[name for name in excluded if name in reference])
    working = working.drop(columns=[name for name in excluded if name in working])
    if mode == "row_exact":
        pd.testing.assert_frame_equal(reference, working, check_exact=True)
        return
    if list(reference.columns) != list(working.columns):
        raise AssertionError("column order differs")
    if reference.shape != working.shape:
        raise AssertionError("table shape differs")
    pd.testing.assert_index_equal(reference.index, working.index, exact=True)
    for column in reference.columns:
        left = reference[column]
        right = working[column]
        if pd.api.types.is_numeric_dtype(left.dtype) and pd.api.types.is_numeric_dtype(
            right.dtype
        ):
            if not np.allclose(
                left.to_numpy(dtype=float),
                right.to_numpy(dtype=float),
                atol=atol,
                rtol=rtol,
                equal_nan=True,
            ):
                raise AssertionError(f"numeric column differs: {column}")
        else:
            pd.testing.assert_series_equal(left, right, check_dtype=False)


def _compare_json(
    reference: Any,
    working: Any,
    *,
    numeric: bool,
    atol: float,
    rtol: float,
    excluded: frozenset[str],
) -> bool:
    if isinstance(reference, dict) and isinstance(working, dict):
        omitted = excluded | (_VOLATILE_JSON_KEYS if numeric else frozenset())
        left_keys = set(reference).difference(omitted)
        right_keys = set(working).difference(omitted)
        return left_keys == right_keys and all(
            _compare_json(
                reference[key],
                working[key],
                numeric=numeric,
                atol=atol,
                rtol=rtol,
                excluded=excluded,
            )
            for key in left_keys
        )
    if isinstance(reference, list) and isinstance(working, list):
        return len(reference) == len(working) and all(
            _compare_json(
                left,
                right,
                numeric=numeric,
                atol=atol,
                rtol=rtol,
                excluded=excluded,
            )
            for left, right in zip(reference, working)
        )
    if (
        numeric
        and isinstance(reference, (int, float))
        and not isinstance(reference, bool)
        and isinstance(working, (int, float))
        and not isinstance(working, bool)
    ):
        return bool(np.isclose(reference, working, atol=atol, rtol=rtol, equal_nan=True))
    return reference == working


def _compare_file(
    reference: bytes,
    working: Path,
    *,
    mode: str,
    atol: float,
    rtol: float,
    excluded: frozenset[str],
) -> str | None:
    if mode == "exact":
        return None if reference == working.read_bytes() else "byte mismatch"
    if working.suffix.lower() in {".csv", ".parquet"}:
        try:
            left, right = _table(reference, working)
            _compare_frames(
                left,
                right,
                mode=mode,
                atol=atol,
                rtol=rtol,
                excluded=excluded,
            )
        except Exception as error:
            return str(error).splitlines()[0]
        return None
    if working.suffix.lower() == ".json":
        try:
            left = json.loads(reference.decode("utf-8"))
            right = json.loads(working.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return f"invalid JSON: {error}"
        return None if _compare_json(
            left,
            right,
            numeric=mode == "numeric",
            atol=atol,
            rtol=rtol,
            excluded=excluded,
        ) else "JSON value mismatch"
    return None if reference == working.read_bytes() else "byte mismatch"


def compare_task_to_git(
    task: RebuildTask,
    *,
    code_root: Path,
    repository_root: Path,
    revision: str = "HEAD",
) -> TaskComparison:
    """Compare every tracked reference beneath a task's declared outputs."""
    code_root = Path(code_root).resolve(strict=True)
    repository_root = Path(repository_root).resolve(strict=True)
    mode, atol, rtol, excluded = _comparison_spec(task)
    tracked = sorted(
        {
            path
            for output in task.outputs
            for path in _tracked_files(repository_root, code_root, output, revision)
        }
    )
    if not tracked:
        return TaskComparison(task.id, "unreferenced", 0, ())
    issues: list[ComparisonIssue] = []
    for repository_path in tracked:
        relative, working = _working_path(repository_path, code_root, repository_root)
        if not working.is_file():
            issues.append(ComparisonIssue(relative, "missing rebuilt artifact"))
            continue
        if mode == "exact":
            reference = _git(
                repository_root,
                "cat-file",
                "--filters",
                f"--path={repository_path}",
                f"{revision}:{repository_path}",
            )
        else:
            reference = _git(
                repository_root,
                "cat-file",
                "-p",
                f"{revision}:{repository_path}",
            )
        reason = _compare_file(
            reference,
            working,
            mode=mode,
            atol=atol,
            rtol=rtol,
            excluded=excluded,
        )
        if reason is not None:
            issues.append(ComparisonIssue(relative, reason))
    return TaskComparison(
        task.id,
        "different" if issues else "matched",
        len(tracked),
        tuple(issues),
    )


__all__ = [
    "ComparisonIssue",
    "TaskComparison",
    "compare_task_to_git",
]
