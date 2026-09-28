"""Validated, hash-resumable task graph for source-to-results reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Mapping, Sequence
from uuid import uuid4

from experiments.external_evidence import sha256_file


_TASK_ID = re.compile(r"[A-Za-z0-9_.-]+")


class GraphError(ValueError):
    """Raised when a graph is ambiguous, cyclic or references unsafe artifacts."""


class TaskExecutionError(RuntimeError):
    """Raised when a registered task cannot produce its declared outputs."""


def _relative_path(value: object) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise GraphError(f"artifact path escapes code root: {text!r}")
    return path.as_posix()


def _canonical_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_hash(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        rows = [
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": sha256_file(child),
                "bytes": child.stat().st_size,
            }
            for child in sorted(path.rglob("*"))
            if child.is_file()
        ]
        return _canonical_hash(rows)
    raise FileNotFoundError(path)


@dataclass(frozen=True)
class RebuildTask:
    id: str
    depends_on: tuple[str, ...]
    module: str
    args: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    profile: str
    comparison: object
    environment: tuple[tuple[str, str], ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RebuildTask":
        task_id = str(payload.get("id", ""))
        module = str(payload.get("module", ""))
        if _TASK_ID.fullmatch(task_id) is None or not module:
            raise GraphError(f"invalid task id or module: {task_id!r}")
        profile = str(payload.get("profile", ""))
        if profile not in {"canonical", "live", "all"}:
            raise GraphError(f"invalid task profile for {task_id}: {profile!r}")

        def strings(field: str) -> tuple[str, ...]:
            value = payload.get(field)
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise GraphError(f"task {task_id} requires a string list for {field}")
            return tuple(value)

        depends_on = strings("depends_on")
        if any(_TASK_ID.fullmatch(item) is None for item in depends_on):
            raise GraphError(f"task {task_id} has an invalid dependency")
        raw_environment = payload.get("environment", {})
        if not isinstance(raw_environment, dict) or not all(
            isinstance(key, str) and key and isinstance(value, str)
            for key, value in raw_environment.items()
        ):
            raise GraphError(f"task {task_id} environment must map strings to strings")
        return cls(
            id=task_id,
            depends_on=depends_on,
            module=module,
            args=strings("args"),
            inputs=tuple(_relative_path(item) for item in strings("inputs")),
            outputs=tuple(_relative_path(item) for item in strings("outputs")),
            profile=profile,
            comparison=payload.get("comparison", "exact"),
            environment=tuple(sorted(raw_environment.items())),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "depends_on": list(self.depends_on),
            "module": self.module,
            "args": list(self.args),
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "profile": self.profile,
            "comparison": self.comparison,
            "environment": dict(self.environment),
        }


@dataclass(frozen=True)
class RebuildGraph:
    sources: tuple[str, ...]
    tasks: tuple[RebuildTask, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "RebuildGraph":
        if payload.get("schema_version") != 1:
            raise GraphError("rebuild graph schema_version must be 1")
        raw_sources = payload.get("sources", [])
        raw_tasks = payload.get("tasks")
        if not isinstance(raw_sources, list) or not all(
            isinstance(value, str) for value in raw_sources
        ):
            raise GraphError("rebuild graph sources must be a string list")
        if not isinstance(raw_tasks, list):
            raise GraphError("rebuild graph tasks must be a list")
        tasks = tuple(RebuildTask.from_dict(task) for task in raw_tasks)
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            raise GraphError("rebuild task ids must be unique")
        return cls(
            sources=tuple(_relative_path(value) for value in raw_sources),
            tasks=tasks,
        )

    def task_map(self) -> dict[str, RebuildTask]:
        return {task.id: task for task in self.tasks}

    def topological_order(self) -> tuple[str, ...]:
        tasks = self.task_map()
        for task in self.tasks:
            unknown = set(task.depends_on) - set(tasks)
            if unknown:
                raise GraphError(f"task {task.id} has unknown dependencies: {sorted(unknown)}")
        pending = {task.id: set(task.depends_on) for task in self.tasks}
        ordered: list[str] = []
        declaration_order = [task.id for task in self.tasks]
        while pending:
            ready = [task_id for task_id in declaration_order if task_id in pending and not pending[task_id]]
            if not ready:
                raise GraphError(f"rebuild graph contains a cycle: {sorted(pending)}")
            for task_id in ready:
                ordered.append(task_id)
                del pending[task_id]
                for dependencies in pending.values():
                    dependencies.discard(task_id)
        return tuple(ordered)

    def registered_outputs(self) -> frozenset[str]:
        return frozenset(output for task in self.tasks for output in task.outputs)

    def validate_artifacts(self) -> None:
        self.topological_order()
        tasks = self.task_map()

        def ancestors(task_id: str) -> set[str]:
            found: set[str] = set()
            pending = list(tasks[task_id].depends_on)
            while pending:
                dependency = pending.pop()
                if dependency in found:
                    continue
                found.add(dependency)
                pending.extend(tasks[dependency].depends_on)
            return found

        producers: dict[str, list[str]] = {}
        for task in self.tasks:
            for output in task.outputs:
                producers.setdefault(output, []).append(task.id)
        duplicates = {path: ids for path, ids in producers.items() if len(ids) != 1}
        if duplicates:
            raise GraphError(f"artifact has more than one producer: {duplicates}")
        source_set = set(self.sources)
        for task in self.tasks:
            for input_path in task.inputs:
                if input_path in source_set:
                    continue
                owners = producers.get(input_path, [])
                if len(owners) != 1:
                    raise GraphError(
                        f"input {input_path!r} for task {task.id} has no registered producer"
                    )
                if owners[0] not in ancestors(task.id):
                    raise GraphError(
                        f"producer {owners[0]} for {input_path!r} is not a dependency of {task.id}"
                    )


@dataclass(frozen=True)
class RebuildContext:
    code_root: Path
    repository_root: Path
    state_root: Path
    profile: str = "canonical"
    env: Mapping[str, str] | None = None
    runtime_identity: str = ""


@dataclass(frozen=True)
class RebuildReport:
    executed: tuple[str, ...]
    skipped: tuple[str, ...]


def load_graph(path: Path) -> RebuildGraph:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise GraphError("rebuild graph root must be an object")
    graph = RebuildGraph.from_dict(payload)
    graph.validate_artifacts()
    return graph


def _git_head(repository_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
        shell=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNVERSIONED"


def _task_path(code_root: Path, relative: str) -> Path:
    code_root = code_root.resolve(strict=True)
    target = (code_root / Path(*PurePosixPath(relative).parts)).resolve()
    if code_root not in target.parents:
        raise GraphError(f"artifact path escapes code root: {relative!r}")
    return target


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _selected_ids(
    graph: RebuildGraph,
    profile: str,
    selected_tasks: Sequence[str] | None,
) -> set[str]:
    tasks = graph.task_map()
    if selected_tasks is None:
        selected = {task.id for task in graph.tasks if task.profile in {profile, "all"}}
    else:
        selected = set(selected_tasks)
        unknown = selected - set(tasks)
        if unknown:
            raise GraphError(f"unknown selected tasks: {sorted(unknown)}")
    pending = list(selected)
    while pending:
        task_id = pending.pop()
        task = tasks[task_id]
        if task.profile not in {profile, "all"}:
            raise GraphError(f"task {task_id} is not available in profile {profile}")
        for dependency in task.depends_on:
            if dependency not in selected:
                selected.add(dependency)
                pending.append(dependency)
    return selected


def _state_matches(
    state_path: Path,
    task_identity: str,
    output_paths: Mapping[str, Path],
) -> bool:
    if not state_path.is_file():
        return False
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if state.get("status") != "complete" or state.get("task_identity") != task_identity:
        return False
    expected = state.get("outputs")
    if not isinstance(expected, dict):
        return False
    try:
        actual = {relative: _path_hash(path) for relative, path in output_paths.items()}
    except FileNotFoundError:
        return False
    return actual == expected


def execute_graph(
    graph: RebuildGraph,
    context: RebuildContext,
    selected_tasks: Sequence[str] | None = None,
    *,
    force_tasks: Sequence[str] = (),
    stream: bool = False,
) -> RebuildReport:
    graph.validate_artifacts()
    code_root = Path(context.code_root).resolve(strict=True)
    state_root = Path(context.state_root).resolve()
    selected = _selected_ids(graph, context.profile, selected_tasks)
    forced = set(force_tasks)
    if not forced.issubset(selected):
        raise GraphError(f"forced tasks are outside the selected graph: {sorted(forced - selected)}")
    tasks = graph.task_map()
    executed: list[str] = []
    skipped: list[str] = []

    for task_id in graph.topological_order():
        if task_id not in selected:
            continue
        task = tasks[task_id]
        input_paths = {relative: _task_path(code_root, relative) for relative in task.inputs}
        missing = [relative for relative, path in input_paths.items() if not path.exists()]
        if missing:
            raise TaskExecutionError(f"task {task.id} missing input: {missing}")
        input_hashes = {relative: _path_hash(path) for relative, path in input_paths.items()}
        task_identity = _canonical_hash(
            {
                "task": task.to_dict(),
                "inputs": input_hashes,
                "commit": _git_head(Path(context.repository_root)),
                "profile": context.profile,
                "runtime": context.runtime_identity,
            }
        )
        output_paths = {relative: _task_path(code_root, relative) for relative in task.outputs}
        state_path = state_root / f"{task.id}.json"
        if task.id not in forced and _state_matches(state_path, task_identity, output_paths):
            skipped.append(task.id)
            if stream:
                print(f"REUSE {task.id} (matching inputs and output hashes)", flush=True)
            continue

        command = [sys.executable, "-m", task.module, *task.args]
        task_environment = None
        if context.env is not None or task.environment:
            task_environment = dict(os.environ if context.env is None else context.env)
            task_environment.update(task.environment)
        if stream:
            print(f"RUN {task.id}", flush=True)
            state_root.mkdir(parents=True, exist_ok=True)
            task_environment = dict(os.environ if task_environment is None else task_environment)
            task_environment.update(PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            with (state_root / f"{task.id}.log").open("w", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=code_root, env=task_environment,
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                      text=True, encoding="utf-8", errors="replace", bufsize=1) as process:
                    try:
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        returncode = process.wait()
                    except BaseException:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
                        raise
        else:
            result = subprocess.run(command, cwd=code_root, env=task_environment,
                                    check=False, capture_output=True, text=True, shell=False)
            returncode = result.returncode
        if returncode != 0:
            _atomic_json(
                state_path,
                {
                    "status": "failed",
                    "task_id": task.id,
                    "task_identity": task_identity,
                    "returncode": int(returncode),
                },
            )
            raise TaskExecutionError(f"task {task.id} failed with exit code {returncode}")
        missing_outputs = [relative for relative, path in output_paths.items() if not path.exists()]
        if missing_outputs:
            _atomic_json(
                state_path,
                {
                    "status": "failed",
                    "task_id": task.id,
                    "task_identity": task_identity,
                    "missing_outputs": missing_outputs,
                },
            )
            raise TaskExecutionError(f"task {task.id} missing output: {missing_outputs}")
        output_hashes = {relative: _path_hash(path) for relative, path in output_paths.items()}
        _atomic_json(
            state_path,
            {
                "status": "complete",
                "task_id": task.id,
                "task_identity": task_identity,
                "outputs": output_hashes,
            },
        )
        executed.append(task.id)
        if stream:
            print(f"DONE {task.id}", flush=True)
    return RebuildReport(executed=tuple(executed), skipped=tuple(skipped))


__all__ = [
    "GraphError",
    "RebuildContext",
    "RebuildGraph",
    "RebuildReport",
    "RebuildTask",
    "TaskExecutionError",
    "execute_graph",
    "load_graph",
]
