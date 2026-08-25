"""Verify Notebook V from a clean checkout without opening sealed periods."""
from __future__ import annotations

import argparse
import ast
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]


def _module_path(module: str, code_root: Path) -> Path | None:
    parts = tuple(part for part in module.split(".") if part)
    if not parts:
        return None
    module_file = code_root.joinpath(*parts).with_suffix(".py")
    if module_file.is_file():
        return module_file.resolve()
    package_file = code_root.joinpath(*parts, "__init__.py")
    return package_file.resolve() if package_file.is_file() else None


def _module_names(path: Path, code_root: Path) -> set[str]:
    relative = path.resolve().relative_to(code_root.resolve())
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    package = parts if path.name == "__init__.py" else parts[:-1]
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            keep = len(package) - (node.level - 1)
            if keep < 0:
                continue
            prefix = package[:keep]
            base = ".".join([*prefix, *(node.module or "").split(".")])
        else:
            base = node.module or ""
        if base:
            names.add(base)
        for alias in node.names:
            if alias.name != "*" and base:
                names.add(f"{base}.{alias.name}")
    return names


def local_import_closure(
    entrypoints: Iterable[Path], *, code_root: Path = CODE_ROOT
) -> set[Path]:
    """Return every statically imported local Python source, including roots."""
    root = Path(code_root).resolve()
    pending = [Path(path).resolve() for path in entrypoints]
    closure: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in closure:
            continue
        if not path.is_file() or not path.is_relative_to(root):
            raise ValueError(f"local import entry is unavailable: {path}")
        closure.add(path)
        for name in _module_names(path, root):
            imported = _module_path(name, root)
            if imported is not None and imported not in closure:
                pending.append(imported)
    return closure


def read_environment_lock(path: Path) -> dict[str, str]:
    """Parse an exact ``distribution==version`` lock without resolving ranges."""
    locked: dict[str, str] = {}
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.count("==") != 1:
            raise ValueError(f"Notebook V lock is not exact: {raw_line}")
        name, version = (part.strip() for part in line.split("==", 1))
        canonical = name.lower().replace("_", "-")
        if not canonical or not version or canonical in locked:
            raise ValueError(f"invalid Notebook V lock entry: {raw_line}")
        locked[canonical] = version
    return dict(sorted(locked.items()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload is not an object: {path}")
    return payload


def _artifact_hashes_valid(
    run_dir: Path, state: dict[str, object], names: Iterable[str]
) -> bool:
    records = state.get("artifacts")
    if not isinstance(records, dict):
        return False
    for name in names:
        path = run_dir / name
        record = records.get(name)
        if (
            not path.is_file()
            or not isinstance(record, dict)
            or int(record.get("size", -1)) != path.stat().st_size
            or str(record.get("sha256", "")) != _sha256(path)
        ):
            return False
    return True


def _git_tracked(repo_root: Path) -> set[str]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=repo_root, text=True, encoding="utf-8"
    )
    return {line.replace("\\", "/") for line in output.splitlines() if line}


def _relative(path: Path, repo_root: Path) -> str:
    return path.resolve().relative_to(repo_root.resolve()).as_posix()


def audit_repository(
    *,
    code_root: Path = CODE_ROOT,
    verify_inputs: bool = False,
    data_root: Path | None = None,
) -> dict[str, object]:
    """Audit the published V result, frozen handoffs, code and environment."""
    from experiments import run_event_window_direction_head as runner
    from experiments import run_event_window_tail_models as tail_runner

    root = Path(code_root).resolve()
    repo_root = Path(
        subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            text=True,
            encoding="utf-8",
        ).strip()
    ).resolve()

    pointer_path = runner.RUN_ROOT / "latest_dev.json"
    pointer = _read_json(pointer_path)
    run_hash = str(pointer.get("run_hash", ""))
    relative_path = Path(str(pointer.get("relative_path", "")))
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Notebook V latest pointer path is unsafe")
    run_dir = (runner.RUN_ROOT / relative_path).resolve()
    expected_run_dir = (runner.RUN_ROOT / run_hash / "full").resolve()
    if run_dir != expected_run_dir:
        raise ValueError("Notebook V latest pointer does not select its full run")
    state = _read_json(run_dir / "run_state.json")
    summary = _read_json(run_dir / "summary.json")
    protocol = _read_json(run_dir / "protocol.json")
    artifact_hashes_valid = bool(
        state.get("status") == "complete"
        and state.get("run_hash") == run_hash
        and state.get("protocol_hash") == pointer.get("protocol_hash")
        and state.get("summary") == summary
        and all(
            protocol.get(name) == state.get(name)
            for name in ("run_hash", "protocol_hash", "source_hash", "input_hash")
        )
        and _artifact_hashes_valid(run_dir, state, runner.READER_ARTIFACTS)
    )

    frozen_u_valid = True
    frozen_j_valid = True
    try:
        frozen_u = runner.load_frozen_u_artifacts(runner.FROZEN_U_ROOT)
    except Exception:
        frozen_u_valid = False
        frozen_u = None
    try:
        frozen_j = tail_runner.load_frozen_j_artifacts(tail_runner.FROZEN_J_ROOT)
    except Exception:
        frozen_j_valid = False
        frozen_j = None

    closure = local_import_closure((Path(runner.__file__),), code_root=root)
    registered = {Path(path).resolve() for path in runner._SOURCE_DEPENDENCIES}
    transitive_sources_covered = closure.issubset(registered)
    source_hash_matches = runner._source_hash() == state.get("source_hash")

    lock_path = root / "requirements-v-repro.txt"
    locked = read_environment_lock(lock_path)
    installed: dict[str, str | None] = {}
    for name in locked:
        try:
            installed[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed[name] = None
    python_expected = (root / ".python-version").read_text(encoding="utf-8").strip()
    python_actual = ".".join(map(str, sys.version_info[:3]))
    environment_matches = python_actual == python_expected and all(
        installed[name] == version for name, version in locked.items()
    )

    required_paths: set[Path] = {
        *registered,
        Path(__file__).resolve(),
        root / "experiments" / "build_notebook_v.py",
        root / "notebooks" / "V_economic_direction_head.ipynb",
        root / "notebooks" / "V_REPRODUCIBILITY.md",
        root / "requirements-v-repro.txt",
        root / ".python-version",
        root / "tests" / "test_notebook_v_reproducibility.py",
        pointer_path,
        run_dir / "run_state.json",
        *(run_dir / name for name in runner.READER_ARTIFACTS),
        runner.FROZEN_U_ROOT / "latest_dev.json",
        runner.FROZEN_U_ROOT
        / runner.FROZEN_U_RUN_HASH
        / "full"
        / "run_state.json",
        *(
            runner.FROZEN_U_ROOT / runner.FROZEN_U_RUN_HASH / "full" / name
            for name in runner.FROZEN_U_ARTIFACTS
        ),
        tail_runner.FROZEN_J_ROOT / "latest_dev.json",
        tail_runner.FROZEN_J_ROOT
        / tail_runner.FROZEN_J_RUN_HASH
        / "full"
        / "run_state.json",
        *(
            tail_runner.FROZEN_J_ROOT
            / tail_runner.FROZEN_J_RUN_HASH
            / "full"
            / name
            for name in tail_runner.FROZEN_J_ARTIFACTS
        ),
    }
    missing_paths = sorted(
        _relative(path, repo_root) for path in required_paths if not path.is_file()
    )
    tracked = _git_tracked(repo_root)
    untracked_paths = sorted(
        relative
        for path in required_paths
        if path.is_file()
        for relative in (_relative(path, repo_root),)
        if relative not in tracked
    )

    input_identity_matches: bool | None = None
    if verify_inputs:
        if data_root is None:
            raise ValueError("--verify-inputs requires --data-root")
        actual_identity = runner._bounded_development_raw_identity(
            Path(data_root), runner.DirectionHeadConfig()
        )
        expected_identity = summary.get("bounded_development_input_identity")
        input_identity_matches = bool(
            isinstance(expected_identity, dict)
            and expected_identity.get("applicable") is True
            and expected_identity.get("passed") is True
            and actual_identity.get("aggregate_sha256")
            == expected_identity.get("aggregate_sha256")
            and actual_identity.get("source_fingerprints")
            == expected_identity.get("source_fingerprints")
        )

    artifact_passes = all(
        (
            not missing_paths,
            not untracked_paths,
            artifact_hashes_valid,
            frozen_u_valid,
            frozen_j_valid,
            transitive_sources_covered,
            environment_matches,
            input_identity_matches is not False,
        )
    )
    artifact_status = (
        "REPRODUCIBLE_FULL_INPUTS"
        if artifact_passes and verify_inputs
        else "REPRODUCIBLE_ARTIFACT"
        if artifact_passes
        else "FAILED"
    )
    status = (
        f"{artifact_status}_SOURCE_DRIFT"
        if artifact_passes and not source_hash_matches
        else artifact_status
    )
    return {
        "status": status,
        "artifact_status": artifact_status,
        "exact_rerun_required": bool(artifact_passes and not source_hash_matches),
        "published_run_hash": run_hash,
        "published_source_hash": state.get("source_hash"),
        "computed_source_hash": runner._source_hash(),
        "source_hash_matches": source_hash_matches,
        "transitive_sources_covered": transitive_sources_covered,
        "artifact_hashes_valid": artifact_hashes_valid,
        "frozen_u_valid": frozen_u_valid and frozen_u is not None,
        "frozen_j_valid": frozen_j_valid and frozen_j is not None,
        "environment_matches": environment_matches,
        "python_expected": python_expected,
        "python_actual": python_actual,
        "locked_packages": locked,
        "installed_packages": installed,
        "input_identity_matches": input_identity_matches,
        "missing_paths": missing_paths,
        "untracked_paths": untracked_paths,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    parser.add_argument("--verify-inputs", action="store_true")
    parser.add_argument("--data-root", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = audit_repository(
        code_root=args.code_root,
        verify_inputs=args.verify_inputs,
        data_root=args.data_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if str(report["artifact_status"]).startswith("REPRODUCIBLE") else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "audit_repository",
    "local_import_closure",
    "read_environment_lock",
]
