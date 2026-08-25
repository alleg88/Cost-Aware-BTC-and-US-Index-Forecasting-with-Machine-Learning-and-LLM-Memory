"""Restore verified snapshot and non-deterministic evidence without provider calls."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from typing import Mapping
from uuid import uuid4

from experiments.external_evidence import sha256_file


KIND_IDS = {
    "llm_scores": frozenset({"llm_scores", "q2_llm_scores"}),
    "agent_calls": frozenset({"btc_agent_calls", "btc_agent_preflight"}),
    "direct_events": frozenset(
        {"truth_social_pre_q2", "direct_events_pre_q2", "direct_events_q2"}
    ),
    "macro": frozenset({"fred", "fear_greed"}),
    "channel_handoffs": frozenset(
        {
            "channel_handoff_n",
            "channel_handoff_o",
            "channel_handoff_p",
            "channel_handoff_q",
            "channel_handoff_r",
        }
    ),
}
ALLOWED_CLASSES = frozenset(
    {"snapshot_raw", "frozen_nondeterministic", "frozen_handoff"}
)


class FrozenEvidenceError(ValueError):
    """Raised when offline evidence is missing, unsafe or not byte-identical."""


@dataclass(frozen=True)
class StageReport:
    kind: str
    files: int
    bytes: int
    paths: tuple[str, ...]


def _safe_join(root: Path, relative: object) -> Path:
    text = str(relative).replace("\\", "/")
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts:
        raise FrozenEvidenceError(f"evidence path escapes root: {text!r}")
    root = root.resolve(strict=True)
    target = (root / Path(*pure.parts)).resolve()
    if root not in target.parents:
        raise FrozenEvidenceError(f"evidence path escapes root: {text!r}")
    return target


def _load_manifest(manifest: Path | Mapping[str, object]) -> Mapping[str, object]:
    if isinstance(manifest, Mapping):
        return manifest
    payload = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FrozenEvidenceError("source manifest root must be an object")
    return payload


def _verify_file(path: Path, row: Mapping[str, object]) -> None:
    if not path.is_file():
        raise FrozenEvidenceError(f"missing frozen evidence: {row.get('path')}")
    expected_bytes = row.get("bytes")
    expected_hash = str(row.get("sha256", ""))
    if path.stat().st_size != expected_bytes:
        raise FrozenEvidenceError(f"size mismatch: {row.get('path')}")
    if sha256_file(path) != expected_hash:
        raise FrozenEvidenceError(f"hash mismatch: {row.get('path')}")


def _atomic_copy(source: Path, destination: Path, row: Mapping[str, object]) -> None:
    if destination.exists():
        _verify_file(destination, row)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copy2(source, temporary)
        _verify_file(temporary, row)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def stage_frozen_evidence(
    evidence_root: Path,
    code_root: Path,
    manifest: Path | Mapping[str, object],
    kind: str,
) -> StageReport:
    """Restore one registered evidence family from a source-only bundle."""
    evidence_root = Path(evidence_root).resolve(strict=True)
    code_root = Path(code_root).resolve(strict=True)
    payload = _load_manifest(manifest)
    rows = payload.get("files")
    if not isinstance(rows, list):
        raise FrozenEvidenceError("source manifest files must be a list")
    ids = KIND_IDS.get(kind, frozenset({kind}))
    selected = [
        row
        for row in rows
        if isinstance(row, dict)
        and (
            row.get("class") == kind
            if kind in ALLOWED_CLASSES
            else row.get("id") in ids
        )
    ]
    if not selected:
        raise FrozenEvidenceError(f"no source records for evidence kind: {kind}")

    staged: list[str] = []
    total_bytes = 0
    for row in sorted(selected, key=lambda value: str(value["path"])):
        if row.get("class") not in ALLOWED_CLASSES:
            raise FrozenEvidenceError(f"unsupported evidence class for {row.get('path')}")
        source = _safe_join(evidence_root / "files", row.get("path"))
        destination = _safe_join(code_root, row.get("stage_path"))
        _verify_file(source, row)
        _atomic_copy(source, destination, row)
        staged.append(str(row["stage_path"]))
        total_bytes += int(row["bytes"])
    return StageReport(kind=kind, files=len(staged), bytes=total_bytes, paths=tuple(staged))


__all__ = [
    "FrozenEvidenceError",
    "KIND_IDS",
    "StageReport",
    "stage_frozen_evidence",
]
