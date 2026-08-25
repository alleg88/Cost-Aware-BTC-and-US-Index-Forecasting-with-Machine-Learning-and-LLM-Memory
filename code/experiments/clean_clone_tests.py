"""Explicit boundary between tracked-only and local-evidence regression tests."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_EVIDENCE_MANIFEST = CODE_ROOT / "tests" / "local_evidence_tests.txt"


class CleanCloneManifestError(ValueError):
    """Raised when the local-evidence test manifest is invalid or stale."""


def load_local_evidence_nodeids(path: Path = DEFAULT_LOCAL_EVIDENCE_MANIFEST) -> tuple[str, ...]:
    """Load exact pytest node IDs, rejecting malformed or duplicate entries."""
    path = Path(path)
    nodeids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    invalid = [nodeid for nodeid in nodeids if not nodeid.startswith("tests/") or "::" not in nodeid]
    if invalid:
        raise CleanCloneManifestError(f"invalid pytest node ID: {invalid[0]}")
    duplicates = sorted({nodeid for nodeid in nodeids if nodeids.count(nodeid) > 1})
    if duplicates:
        raise CleanCloneManifestError(f"duplicate pytest node ID: {duplicates[0]}")
    return nodeids


def partition_clean_clone_nodes(
    collected_nodeids: Iterable[str],
    local_evidence_nodeids: Iterable[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Partition collected tests and fail if an evidence registration is stale."""
    collected = tuple(collected_nodeids)
    evidence = set(local_evidence_nodeids)
    stale = sorted(evidence.difference(collected))
    if stale:
        raise CleanCloneManifestError(
            f"local-evidence test was not collected: {stale[0]}"
        )
    runnable = tuple(nodeid for nodeid in collected if nodeid not in evidence)
    skipped = tuple(nodeid for nodeid in collected if nodeid in evidence)
    return runnable, skipped
