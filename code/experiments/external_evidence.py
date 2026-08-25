"""Snapshot and verify the untracked evidence needed for exact local reruns."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = CODE_ROOT / "external_evidence_manifest.json"
EVIDENCE_ROOTS = ("data", "sentiment", "experiments/cache")
EVIDENCE_SUFFIXES = {
    ".checksum",
    ".csv",
    ".db",
    ".joblib",
    ".json",
    ".jsonl",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pt",
    ".pth",
    ".sqlite",
    ".txt",
    ".zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_paths(code_root: Path) -> set[str]:
    repo_root = code_root.parent
    if not (repo_root / ".git").exists():
        return set()
    result = subprocess.run(
        ["git", "ls-files", "-z", "--", "code"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    prefix = "code/"
    return {
        path[len(prefix) :]
        for path in result.stdout.decode("utf-8").split("\0")
        if path.startswith(prefix)
    }


def evidence_files(code_root: Path) -> Iterable[Path]:
    code_root = Path(code_root).resolve(strict=True)
    tracked = tracked_paths(code_root)
    selected: dict[str, Path] = {}
    for root_name in EVIDENCE_ROOTS:
        root = code_root / Path(root_name)
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(code_root).as_posix()
            parts = {part.lower() for part in path.relative_to(code_root).parts}
            if "secrets" in parts or "__pycache__" in parts:
                continue
            if path.suffix.lower() not in EVIDENCE_SUFFIXES:
                continue
            if relative in tracked:
                continue
            selected[relative] = path
    for relative in sorted(selected):
        yield selected[relative]


def build_manifest(code_root: Path) -> dict[str, object]:
    code_root = Path(code_root).resolve(strict=True)
    files = [
        {
            "path": path.relative_to(code_root).as_posix(),
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        for path in evidence_files(code_root)
    ]
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "file_count": len(files),
        "total_bytes": sum(int(entry["bytes"]) for entry in files),
        "files": files,
    }


def verify_manifest(code_root: Path, manifest_path: Path) -> dict[str, object]:
    code_root = Path(code_root).resolve(strict=True)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    missing: list[str] = []
    mismatched: list[str] = []
    for entry in payload["files"]:
        relative = str(entry["path"])
        path = (code_root / relative).resolve()
        if code_root not in path.parents:
            raise ValueError(f"manifest path escapes code root: {relative}")
        if not path.is_file():
            missing.append(relative)
            continue
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            mismatched.append(relative)
    status = "READY" if not missing and not mismatched else "NOT_READY"
    return {
        "status": status,
        "file_count": int(payload["file_count"]),
        "total_bytes": int(payload["total_bytes"]),
        "missing": missing,
        "mismatched": mismatched,
    }


def _manifest_path(root: Path, relative: str) -> Path:
    root = root.resolve(strict=True)
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError(f"manifest path escapes transfer root: {relative}")
    return path


def copy_manifest_files(
    source_root: Path,
    target_root: Path,
    manifest_path: Path,
) -> dict[str, int]:
    source_root = Path(source_root).resolve(strict=True)
    target_root = Path(target_root).resolve(strict=True)
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    copied_bytes = 0
    for entry in payload["files"]:
        relative = str(entry["path"])
        source = _manifest_path(source_root, relative)
        if not source.is_file():
            raise FileNotFoundError(relative)
        if source.stat().st_size != int(entry["bytes"]) or sha256_file(source) != entry["sha256"]:
            raise ValueError(f"source evidence mismatch: {relative}")
        destination = _manifest_path(target_root, relative)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied_bytes += int(entry["bytes"])
    return {"file_count": int(payload["file_count"]), "total_bytes": copied_bytes}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("snapshot", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--code-root", type=Path, default=CODE_ROOT)
        child.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    export = subparsers.add_parser("export")
    export.add_argument("--code-root", type=Path, default=CODE_ROOT)
    export.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    export.add_argument("--target", type=Path, required=True)
    restore = subparsers.add_parser("restore")
    restore.add_argument("--code-root", type=Path, default=CODE_ROOT)
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "snapshot":
        payload = build_manifest(args.code_root)
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": "READY",
                    "manifest": str(args.manifest),
                    "file_count": payload["file_count"],
                    "total_bytes": payload["total_bytes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.command == "export":
        target = args.target.resolve()
        if target.exists() and any(target.iterdir()):
            raise RuntimeError(f"export target must be empty: {target}")
        target.mkdir(parents=True, exist_ok=True)
        report = copy_manifest_files(args.code_root, target, args.manifest)
        shutil.copy2(args.manifest, target / "external_evidence_manifest.json")
        print(json.dumps({"status": "READY", **report}, indent=2, sort_keys=True))
        return 0

    if args.command == "restore":
        code_root = args.code_root.resolve(strict=True)
        report = copy_manifest_files(args.source, code_root, args.manifest)
        print(json.dumps({"status": "READY", **report}, indent=2, sort_keys=True))
        return 0

    report = verify_manifest(args.code_root, args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
