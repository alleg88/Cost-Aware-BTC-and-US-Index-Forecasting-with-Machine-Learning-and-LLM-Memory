"""Build, verify and transfer the dissertation source-evidence package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Mapping, Sequence

from experiments.external_evidence import sha256_file
from experiments.source_registry import (
    EVIDENCE_CLASSES,
    PolicyError,
    SourcePolicy,
    build_source_records,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = CODE_ROOT / "configs" / "source_evidence_policy.json"
DEFAULT_FULL_MANIFEST = CODE_ROOT / "external_evidence_manifest.json"
DEFAULT_MANIFEST = CODE_ROOT / "source_evidence_manifest.json"
MANIFEST_NAME = "source_evidence_manifest.json"
BUNDLE_METADATA_NAME = "bundle_metadata.json"


class EvidenceError(ValueError):
    """Raised when evidence is unsafe, missing, changed or internally inconsistent."""


def _safe_relative(value: object, *, field: str = "manifest path") -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise EvidenceError(f"{field} escapes its root: {text!r}")
    return path.as_posix()


def _safe_join(root: Path, relative: object) -> Path:
    root = Path(root).resolve()
    path = (root / _safe_relative(relative)).resolve()
    if path != root and root not in path.parents:
        raise EvidenceError(f"manifest path escapes its root: {relative!r}")
    return path


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def manifest_identity(files: Sequence[Mapping[str, object]]) -> str:
    canonical = json.dumps(
        list(files),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EvidenceError(f"JSON root must be an object: {path}")
    return payload


def _validate_row(row: Mapping[str, object]) -> dict[str, object]:
    path = _safe_relative(row.get("path", ""))
    stage_path = _safe_relative(row.get("stage_path", ""), field="stage path")
    evidence_class = str(row.get("class", ""))
    if evidence_class not in EVIDENCE_CLASSES:
        raise EvidenceError(f"invalid evidence class for {path}: {evidence_class!r}")
    byte_count = row.get("bytes")
    if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
        raise EvidenceError(f"invalid byte count for {path}")
    sha256 = str(row.get("sha256", "")).lower()
    if len(sha256) != 64 or any(character not in "0123456789abcdef" for character in sha256):
        raise EvidenceError(f"invalid SHA-256 for {path}")
    provider = str(row.get("provider", "")).strip()
    source_id = str(row.get("id", "")).strip()
    if not provider or not source_id:
        raise EvidenceError(f"missing source identity for {path}")
    return {
        "id": source_id,
        "class": evidence_class,
        "provider": provider,
        "path": path,
        "stage_path": stage_path,
        "bytes": byte_count,
        "sha256": sha256,
    }


def load_manifest(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    if payload.get("schema_version") != 1:
        raise EvidenceError("source manifest schema_version must be 1")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise EvidenceError("source manifest files must be a list")
    files = [_validate_row(row) for row in raw_files if isinstance(row, dict)]
    if len(files) != len(raw_files):
        raise EvidenceError("source manifest file rows must be objects")
    if len({str(row["path"]) for row in files}) != len(files):
        raise EvidenceError("source manifest paths must be unique")
    expected_identity = manifest_identity(files)
    if payload.get("manifest_identity") != expected_identity:
        raise EvidenceError("source manifest identity mismatch")
    if payload.get("file_count") != len(files):
        raise EvidenceError("source manifest file_count mismatch")
    if payload.get("total_bytes") != sum(int(row["bytes"]) for row in files):
        raise EvidenceError("source manifest total_bytes mismatch")
    return {**payload, "files": files}


def _verify_file(path: Path, row: Mapping[str, object]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == int(row["bytes"])
        and sha256_file(path) == row["sha256"]
    )


def build_manifest(
    code_root: Path,
    policy_path: Path,
    full_manifest_path: Path,
) -> dict[str, object]:
    code_root = Path(code_root).resolve(strict=True)
    try:
        policy = SourcePolicy.load(policy_path)
        full_manifest = _read_json(full_manifest_path)
        records = build_source_records(policy, full_manifest)
    except PolicyError as error:
        raise EvidenceError(str(error)) from error
    files = [record.to_dict() for record in records]
    for row in files:
        path = _safe_join(code_root, row["path"])
        if not _verify_file(path, row):
            raise EvidenceError(f"source bytes differ from full manifest: {row['path']}")
        lowered_parts = {part.lower() for part in PurePosixPath(str(row["path"])).parts}
        if "secrets" in lowered_parts:
            raise EvidenceError(f"secret path entered source policy: {row['path']}")

    class_summary: dict[str, dict[str, int]] = {}
    for evidence_class in sorted(EVIDENCE_CLASSES):
        selected = [row for row in files if row["class"] == evidence_class]
        if selected:
            class_summary[evidence_class] = {
                "bytes": sum(int(row["bytes"]) for row in selected),
                "files": len(selected),
            }
    policy_bytes = Path(policy_path).read_bytes()
    return {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "policy_sha256": _sha256_bytes(policy_bytes),
        "manifest_identity": manifest_identity(files),
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "class_summary": class_summary,
        "files": files,
    }


def _bundle_metadata(source_root: Path) -> dict[str, object] | None:
    path = Path(source_root) / BUNDLE_METADATA_NAME
    return _read_json(path) if path.is_file() else None


def _selected_rows(
    payload: Mapping[str, object],
    *,
    include_public: bool,
) -> list[dict[str, object]]:
    files = payload["files"]
    assert isinstance(files, list)
    return [
        row
        for row in files
        if isinstance(row, dict)
        and (include_public or row["class"] != "public_raw")
    ]


def verify_manifest(source_root: Path, manifest_path: Path) -> dict[str, object]:
    source_root = Path(source_root).resolve(strict=True)
    payload = load_manifest(manifest_path)
    metadata = _bundle_metadata(source_root)
    if metadata is not None:
        if metadata.get("manifest_identity") != payload["manifest_identity"]:
            raise EvidenceError("bundle metadata manifest identity mismatch")
        include_public = bool(metadata.get("include_public"))
        file_root = source_root / "files"
    else:
        include_public = True
        file_root = source_root
    rows = _selected_rows(payload, include_public=include_public)
    missing: list[str] = []
    mismatched: list[str] = []
    for row in rows:
        relative = str(row["path"])
        path = _safe_join(file_root, relative)
        if not path.is_file():
            missing.append(relative)
        elif not _verify_file(path, row):
            mismatched.append(relative)
    return {
        "status": "READY" if not missing and not mismatched else "NOT_READY",
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "manifest_identity": payload["manifest_identity"],
        "missing": missing,
        "mismatched": mismatched,
    }


def _require_empty_directory(path: Path) -> Path:
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise EvidenceError(f"target directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".part")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def export_bundle(
    source_root: Path,
    target_root: Path,
    manifest_path: Path,
    *,
    include_public: bool = False,
) -> dict[str, object]:
    source_root = Path(source_root).resolve(strict=True)
    payload = load_manifest(manifest_path)
    full_report = verify_manifest(source_root, manifest_path)
    if full_report["status"] != "READY":
        raise EvidenceError("source root does not satisfy the source manifest")
    target_root = _require_empty_directory(target_root)
    rows = _selected_rows(payload, include_public=include_public)
    file_root = target_root / "files"
    for row in rows:
        _atomic_copy(
            _safe_join(source_root, row["path"]),
            _safe_join(file_root, row["path"]),
        )
    _atomic_copy(Path(manifest_path).resolve(strict=True), target_root / MANIFEST_NAME)
    metadata = {
        "schema_version": 1,
        "manifest_identity": payload["manifest_identity"],
        "include_public": bool(include_public),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
    }
    (target_root / BUNDLE_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = verify_manifest(target_root, target_root / MANIFEST_NAME)
    if report["status"] != "READY":
        raise EvidenceError("exported source bundle failed verification")
    return {
        "status": "READY",
        "file_count": len(rows),
        "total_bytes": metadata["total_bytes"],
        "public_files_omitted": int(payload["file_count"]) - len(rows),
    }


def restore_bundle(bundle_root: Path, target_root: Path) -> dict[str, object]:
    bundle_root = Path(bundle_root).resolve(strict=True)
    manifest_path = bundle_root / MANIFEST_NAME
    report = verify_manifest(bundle_root, manifest_path)
    if report["status"] != "READY":
        raise EvidenceError("source bundle failed verification")
    target_root = _require_empty_directory(target_root)
    payload = load_manifest(manifest_path)
    metadata = _bundle_metadata(bundle_root)
    if metadata is None:
        raise EvidenceError("source bundle metadata is missing")
    rows = _selected_rows(payload, include_public=bool(metadata.get("include_public")))
    for row in rows:
        _atomic_copy(
            _safe_join(bundle_root / "files", row["path"]),
            _safe_join(target_root / "files", row["path"]),
        )
    _atomic_copy(manifest_path, target_root / MANIFEST_NAME)
    _atomic_copy(
        bundle_root / BUNDLE_METADATA_NAME,
        target_root / BUNDLE_METADATA_NAME,
    )
    restored = verify_manifest(target_root, target_root / MANIFEST_NAME)
    if restored["status"] != "READY":
        raise EvidenceError("restored source bundle failed verification")
    return restored


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("--code-root", type=Path, default=CODE_ROOT)
    snapshot.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    snapshot.add_argument("--full-manifest", type=Path, default=DEFAULT_FULL_MANIFEST)
    snapshot.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--source", type=Path, default=CODE_ROOT)
    verify.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)

    export = subparsers.add_parser("export")
    export.add_argument("--source", type=Path, default=CODE_ROOT)
    export.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    export.add_argument("--target", type=Path, required=True)
    export.add_argument("--include-public", action="store_true")

    restore = subparsers.add_parser("restore")
    restore.add_argument("--source", type=Path, required=True)
    restore.add_argument("--target", type=Path, required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "snapshot":
            payload = build_manifest(args.code_root, args.policy, args.full_manifest)
            _write_json(args.manifest, payload)
            output: Mapping[str, object] = {
                "status": "READY",
                "manifest": str(args.manifest),
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
                "manifest_identity": payload["manifest_identity"],
            }
        elif args.command == "verify":
            output = verify_manifest(args.source, args.manifest)
        elif args.command == "report":
            payload = load_manifest(args.manifest)
            output = {
                "status": "READY",
                "file_count": payload["file_count"],
                "total_bytes": payload["total_bytes"],
                "class_summary": payload["class_summary"],
                "manifest_identity": payload["manifest_identity"],
            }
        elif args.command == "export":
            output = export_bundle(
                args.source,
                args.target,
                args.manifest,
                include_public=args.include_public,
            )
        else:
            output = restore_bundle(args.source, args.target)
    except (EvidenceError, OSError, json.JSONDecodeError) as error:
        print(json.dumps({"status": "NOT_READY", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output.get("status") == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_METADATA_NAME",
    "DEFAULT_MANIFEST",
    "EvidenceError",
    "MANIFEST_NAME",
    "build_manifest",
    "export_bundle",
    "load_manifest",
    "manifest_identity",
    "restore_bundle",
    "verify_manifest",
]
