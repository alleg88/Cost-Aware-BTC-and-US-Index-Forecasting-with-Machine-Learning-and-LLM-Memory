"""Append-only state and global opening sentinel for the final Q2 lockbox."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping


CODE_ROOT = Path(__file__).resolve().parents[1]
GLOBAL_STATE_ROOT = CODE_ROOT / "experiments" / "cache" / "final_q2_lockbox"
GLOBAL_SENTINEL_PATH = GLOBAL_STATE_ROOT / "OPENED.json"


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(character in "0123456789abcdef" for character in value)


@dataclass(frozen=True)
class OpeningIdentity:
    implementation_commit: str
    manifest_commit: str
    protocol_hash: str
    manifest_sha256: str
    q2_source_hashes: Mapping[str, str]

    def validate(self) -> "OpeningIdentity":
        for label, value in (
            ("implementation_commit", self.implementation_commit),
            ("manifest_commit", self.manifest_commit),
        ):
            if not _is_hex(str(value).lower(), 40):
                raise ValueError(f"{label} must be a 40-character commit hash")
        for label, value in (
            ("protocol_hash", self.protocol_hash),
            ("manifest_sha256", self.manifest_sha256),
        ):
            if not _is_hex(str(value).lower(), 64):
                raise ValueError(f"{label} must be a 64-character sha256")
        if not self.q2_source_hashes:
            raise ValueError("q2_source_hashes must not be empty")
        for key, value in self.q2_source_hashes.items():
            if not key or not _is_hex(str(value).lower(), 64):
                raise ValueError("every Q2 source must have a named 64-character sha256")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "implementation_commit": self.implementation_commit.lower(),
            "manifest_commit": self.manifest_commit.lower(),
            "protocol_hash": self.protocol_hash.lower(),
            "manifest_sha256": self.manifest_sha256.lower(),
            "q2_source_hashes": {
                str(key): str(value).lower()
                for key, value in sorted(self.q2_source_hashes.items())
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "OpeningIdentity":
        expected = {
            "implementation_commit",
            "manifest_commit",
            "protocol_hash",
            "manifest_sha256",
            "q2_source_hashes",
        }
        if set(payload) != expected:
            raise ValueError("opening identity keys changed")
        identity = cls(
            implementation_commit=str(payload["implementation_commit"]).lower(),
            manifest_commit=str(payload["manifest_commit"]).lower(),
            protocol_hash=str(payload["protocol_hash"]).lower(),
            manifest_sha256=str(payload["manifest_sha256"]).lower(),
            q2_source_hashes={
                str(key): str(value).lower()
                for key, value in dict(payload["q2_source_hashes"]).items()
            },
        )
        return identity.validate()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_root(root: str | Path | None) -> Path:
    return GLOBAL_STATE_ROOT if root is None else Path(root)


def _protocol_root(identity: OpeningIdentity, root: str | Path | None) -> Path:
    return _state_root(root) / identity.validate().protocol_hash


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)
    return path


def begin_global_open(
    identity: OpeningIdentity, *, root: str | Path | None = None
) -> OpeningIdentity:
    identity.validate()
    state_root = _state_root(root)
    state_root.mkdir(parents=True, exist_ok=True)
    sentinel = state_root / "OPENED.json"
    payload = {
        "state": "OPENING",
        "opened_at_utc": _now(),
        "identity": identity.to_dict(),
    }
    with sentinel.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return identity


def load_opening_identity(path: str | Path = GLOBAL_SENTINEL_PATH) -> OpeningIdentity:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(payload) != {"state", "opened_at_utc", "identity"} or payload["state"] != "OPENING":
        raise ValueError("global opening sentinel is malformed")
    return OpeningIdentity.from_dict(payload["identity"])


def assert_exact_resume(
    identity: OpeningIdentity, *, root: str | Path | None = None
) -> OpeningIdentity:
    sentinel = _state_root(root) / "OPENED.json"
    try:
        saved = load_opening_identity(sentinel)
    except FileNotFoundError as exc:
        raise PermissionError("global OPENED sentinel is absent") from exc
    if saved != identity.validate():
        raise PermissionError("opening identity differs from the immutable global sentinel")
    complete = _protocol_root(identity, root) / "COMPLETE.json"
    if complete.exists():
        raise PermissionError("lockbox run is already COMPLETE and cannot resume")
    return saved


def require_global_opening(identity: OpeningIdentity) -> OpeningIdentity:
    try:
        saved = load_opening_identity(GLOBAL_SENTINEL_PATH)
    except FileNotFoundError as exc:
        raise PermissionError("global OPENED sentinel is absent") from exc
    if saved != identity.validate():
        raise PermissionError("opening identity differs from the repository-global sentinel")
    return saved


def write_aborted_before_open(
    identity: OpeningIdentity,
    reason: str,
    *,
    root: str | Path | None = None,
) -> Path:
    if not str(reason).strip():
        raise ValueError("abort reason must not be empty")
    path = _protocol_root(identity, root) / "ABORTED_BEFORE_OPEN.json"
    return _write_json_atomic(
        path,
        {
            "state": "ABORTED_BEFORE_OPEN",
            "recorded_at_utc": _now(),
            "reason": str(reason),
            "identity": identity.to_dict(),
        },
    )


def mark_failed_after_open(
    identity: OpeningIdentity,
    reason: str,
    *,
    root: str | Path | None = None,
) -> Path:
    assert_exact_resume(identity, root=root)
    if not str(reason).strip():
        raise ValueError("failure reason must not be empty")
    return _write_json_atomic(
        _protocol_root(identity, root) / "FAILED_AFTER_OPEN.json",
        {
            "state": "FAILED_AFTER_OPEN",
            "recorded_at_utc": _now(),
            "reason": str(reason),
            "identity": identity.to_dict(),
        },
    )


def mark_complete(
    identity: OpeningIdentity,
    result_hashes: Mapping[str, str],
    *,
    root: str | Path | None = None,
) -> Path:
    assert_exact_resume(identity, root=root)
    if not result_hashes or any(
        not key or not _is_hex(str(value).lower(), 64)
        for key, value in result_hashes.items()
    ):
        raise ValueError("result_hashes must contain named sha256 values")
    return _write_json_atomic(
        _protocol_root(identity, root) / "COMPLETE.json",
        {
            "state": "COMPLETE",
            "recorded_at_utc": _now(),
            "identity": identity.to_dict(),
            "result_hashes": {
                str(key): str(value).lower()
                for key, value in sorted(result_hashes.items())
            },
        },
    )


__all__ = [
    "GLOBAL_SENTINEL_PATH",
    "GLOBAL_STATE_ROOT",
    "OpeningIdentity",
    "assert_exact_resume",
    "begin_global_open",
    "load_opening_identity",
    "mark_complete",
    "mark_failed_after_open",
    "require_global_opening",
    "write_aborted_before_open",
]
