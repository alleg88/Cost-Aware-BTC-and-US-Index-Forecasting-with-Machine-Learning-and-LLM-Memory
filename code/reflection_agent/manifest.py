"""Canonical hashing for the frozen agent protocol and its inputs."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Literal

from reflection_agent.config import ProtocolConfig
from reflection_agent.contracts import (
    CandidateBatch,
    EvaluationRecord,
    ObservationReport,
    ReflectionRecord,
    RefinerBatch,
    SemanticBelief,
)


def canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def sha256_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_revision(code_root: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(code_root),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def build_manifest(
    config: ProtocolConfig,
    *,
    code_root: str | Path,
    input_paths: Iterable[str | Path] = (),
    output_mode: Literal["probe", "schema", "json"] | None = None,
) -> dict[str, Any]:
    paths = sorted((Path(path).resolve() for path in input_paths), key=lambda path: str(path).lower())
    inputs = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in paths]
    schemas = {
        model.__name__: sha256_payload(model.model_json_schema())
        for model in (ObservationReport, CandidateBatch, RefinerBatch, EvaluationRecord, ReflectionRecord, SemanticBelief)
    }
    body = {
        "protocol_version": config.protocol_version,
        "config": config.model_dump(mode="json") | {"output_mode": output_mode or config.output_mode},
        "code_revision": current_revision(code_root),
        "schemas": schemas,
        "inputs": inputs,
    }
    return body | {"protocol_hash": sha256_payload(body)}


def write_manifest(path: str | Path, manifest: dict[str, Any], *, allow_replace: bool = False) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if destination.exists() and destination.read_text(encoding="utf-8") != encoded and not allow_replace:
        raise ValueError(f"refusing to overwrite a different protocol manifest: {destination}")
    destination.write_text(encoded, encoding="utf-8")
    return destination
