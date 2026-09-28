"""Opt-in identities for refitted development handoffs, never raw data or Q2."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Mapping


MODE_ENV = "MSC_REBUILD_CHANNEL_HANDOFFS"


def recomputed_handoffs() -> bool:
    """Enabled explicitly by the compact Rebuild task recipes only."""
    return os.environ.get(MODE_ENV) == "recomputed"


def selected_run_hash(pointer: Mapping[str, object], reference: str) -> str:
    if not recomputed_handoffs():
        return reference
    value = str(pointer.get("run_hash", ""))
    if re.fullmatch(r"[0-9a-f]{20}", value) is None or pointer.get("relative_path") != f"{value}/full":
        raise ValueError("recomputed development handoff identity is invalid")
    return value


def handoff_run_hash(run_root: Path, reference: str) -> str:
    """Select a new full-run identity; callers still verify every artifact."""
    if not recomputed_handoffs():
        return reference
    pointer = json.loads((Path(run_root) / "latest_dev.json").read_text(encoding="utf-8"))
    return selected_run_hash(pointer, reference)


def validate_recomputed_development(protocol: Mapping[str, object]) -> None:
    if recomputed_handoffs() and (
        protocol.get("stage") != "dev"
        or bool(protocol.get("smoke", False))
        or protocol.get("development_end_exclusive") != "2025-07-01"
        or protocol.get("forward_or_lockbox_loaded") is not False
    ):
        raise ValueError("recomputed handoff must be a full bounded development run")
