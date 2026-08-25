"""Typed, fail-closed classification of dissertation source evidence."""

from __future__ import annotations

from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path, PurePosixPath
import re
from typing import Literal, Mapping, Sequence


EvidenceClass = Literal[
    "public_raw",
    "snapshot_raw",
    "frozen_nondeterministic",
    "frozen_handoff",
]
EVIDENCE_CLASSES = frozenset(
    {"public_raw", "snapshot_raw", "frozen_nondeterministic", "frozen_handoff"}
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_BRACES = re.compile(r"\{([^{}]+)\}")


class PolicyError(ValueError):
    """Raised when a source policy or manifest row is ambiguous or unsafe."""


def _relative_posix(value: object, *, field: str) -> str:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or ".." in path.parts:
        raise PolicyError(f"{field} escapes the code root: {text!r}")
    return path.as_posix()


def _expand_braces(pattern: str) -> tuple[str, ...]:
    match = _BRACES.search(pattern)
    if match is None:
        return (pattern,)
    choices = match.group(1).split(",")
    if any(not choice for choice in choices):
        raise PolicyError(f"empty glob alternative: {pattern!r}")
    expanded: list[str] = []
    for choice in choices:
        replacement = pattern[: match.start()] + choice + pattern[match.end() :]
        expanded.extend(_expand_braces(replacement))
    return tuple(expanded)


def _glob_matches(path: str, pattern: str) -> bool:
    candidates = {pattern}
    pending = [pattern]
    while pending:
        current = pending.pop()
        if "**/" not in current:
            continue
        collapsed = current.replace("**/", "", 1)
        if collapsed not in candidates:
            candidates.add(collapsed)
            pending.append(collapsed)
    return any(fnmatch.fnmatchcase(path, candidate) for candidate in candidates)


@dataclass(frozen=True)
class SourceRule:
    id: str
    evidence_class: EvidenceClass
    glob: str
    patterns: tuple[str, ...]
    provider: str
    stage: str
    timestamp_column: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourceRule":
        rule_id = str(payload.get("id", "")).strip()
        provider = str(payload.get("provider", "")).strip()
        if not rule_id or not provider:
            raise PolicyError("source rules require non-empty id and provider")
        evidence_class = str(payload.get("class", ""))
        if evidence_class not in EVIDENCE_CLASSES:
            raise PolicyError(f"unknown evidence class: {evidence_class!r}")
        glob = _relative_posix(payload.get("glob", ""), field="glob")
        stage = _relative_posix(payload.get("stage", ""), field="stage")
        patterns = tuple(
            _relative_posix(pattern, field="glob")
            for pattern in _expand_braces(glob)
        )
        timestamp = payload.get("timestamp_column")
        return cls(
            id=rule_id,
            evidence_class=evidence_class,  # type: ignore[arg-type]
            glob=glob,
            patterns=patterns,
            provider=provider,
            stage=stage,
            timestamp_column=None if timestamp is None else str(timestamp),
        )

    def matches(self, relative_path: str) -> bool:
        return any(_glob_matches(relative_path, pattern) for pattern in self.patterns)


@dataclass(frozen=True)
class SourceRecord:
    id: str
    evidence_class: EvidenceClass
    provider: str
    path: str
    stage_path: str
    bytes: int
    sha256: str

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "class": self.evidence_class,
            "provider": self.provider,
            "path": self.path,
            "stage_path": self.stage_path,
            "bytes": self.bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class SourcePolicy:
    schema_version: int
    rules: tuple[SourceRule, ...]

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "SourcePolicy":
        if payload.get("schema_version") != 1:
            raise PolicyError("source policy schema_version must be 1")
        raw_rules = payload.get("rules")
        if not isinstance(raw_rules, list) or not raw_rules:
            raise PolicyError("source policy requires at least one rule")
        rules = tuple(SourceRule.from_dict(rule) for rule in raw_rules)
        ids = [rule.id for rule in rules]
        if len(ids) != len(set(ids)):
            raise PolicyError("source rule ids must be unique")
        return cls(schema_version=1, rules=rules)

    @classmethod
    def load(cls, path: Path) -> "SourcePolicy":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise PolicyError("source policy root must be an object")
        return cls.from_dict(payload)

    def classify(self, relative_path: str) -> SourceRule | None:
        normalized = _relative_posix(relative_path, field="manifest path")
        matches = [rule for rule in self.rules if rule.matches(normalized)]
        if len(matches) > 1:
            ids = ", ".join(rule.id for rule in matches)
            raise PolicyError(
                f"source {normalized!r} matched more than one rule: {ids}"
            )
        return matches[0] if matches else None


def build_source_records(
    policy: SourcePolicy,
    full_manifest: Mapping[str, object],
) -> tuple[SourceRecord, ...]:
    raw_files = full_manifest.get("files")
    if not isinstance(raw_files, list):
        raise PolicyError("full manifest files must be a list")
    records: list[SourceRecord] = []
    seen: set[str] = set()
    for row in raw_files:
        if not isinstance(row, dict):
            raise PolicyError("full manifest file rows must be objects")
        path = _relative_posix(row.get("path", ""), field="manifest path")
        rule = policy.classify(path)
        if rule is None:
            continue
        if path in seen:
            raise PolicyError(f"duplicate source path: {path}")
        seen.add(path)
        byte_count = row.get("bytes")
        if not isinstance(byte_count, int) or isinstance(byte_count, bool) or byte_count < 0:
            raise PolicyError(f"invalid byte count for {path}")
        sha256 = str(row.get("sha256", "")).lower()
        if _SHA256.fullmatch(sha256) is None:
            raise PolicyError(f"invalid SHA-256 for {path}")
        stage_root = PurePosixPath(rule.stage)
        source_path = PurePosixPath(path)
        if source_path != stage_root and stage_root not in source_path.parents:
            raise PolicyError(
                f"source path {path!r} is outside its stage root {rule.stage!r}"
            )
        records.append(
            SourceRecord(
                id=rule.id,
                evidence_class=rule.evidence_class,
                provider=rule.provider,
                path=path,
                stage_path=path,
                bytes=byte_count,
                sha256=sha256,
            )
        )
    return tuple(sorted(records, key=lambda record: record.path))


__all__ = [
    "EVIDENCE_CLASSES",
    "PolicyError",
    "SourcePolicy",
    "SourceRecord",
    "SourceRule",
    "build_source_records",
]
