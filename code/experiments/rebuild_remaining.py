"""Stage immutable channel handoffs and verify completed Q2 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.external_evidence import sha256_file
from experiments.frozen_evidence import stage_frozen_evidence


CODE_ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def stage_channel_handoffs(
    evidence_root: Path,
    manifest_path: Path,
    code_root: Path,
    receipt: Path,
) -> dict[str, object]:
    manifest = _read_json(manifest_path)
    identity = str(manifest.get("manifest_identity", ""))
    if len(identity) != 64:
        raise ValueError("source manifest identity must be a 64-character hash")
    report = stage_frozen_evidence(
        evidence_root,
        code_root,
        manifest,
        kind="channel_handoffs",
    )
    payload = {
        "bytes": report.bytes,
        "files": report.files,
        "kind": report.kind,
        "manifest_identity": identity,
    }
    _atomic_json(receipt, payload)
    return payload


def verify_completed_q2(receipt: Path) -> dict[str, object]:
    from experiments.final_q2_lockbox_runner import (
        DEFAULT_MANIFEST_PATH,
        _completed_result_hashes,
    )
    from experiments.final_q2_lockbox_state import GLOBAL_STATE_ROOT, load_opening_identity

    identity = load_opening_identity()
    if sha256_file(DEFAULT_MANIFEST_PATH) != identity.manifest_sha256:
        raise ValueError("pre-lockbox manifest bytes differ from the opening identity")
    result_root = GLOBAL_STATE_ROOT / identity.protocol_hash
    complete = _read_json(result_root / "COMPLETE.json")
    if complete.get("state") != "COMPLETE" or complete.get("identity") != identity.to_dict():
        raise ValueError("completed Q2 identity changed")
    result_hashes = _completed_result_hashes(result_root, identity)
    if result_hashes is None or complete.get("result_hashes") != result_hashes:
        raise ValueError("completed Q2 result hash registry changed")
    payload = {
        "artifacts": len(result_hashes),
        "protocol_hash": identity.protocol_hash,
        "state": "COMPLETE",
    }
    _atomic_json(receipt, payload)
    return payload


def verify_q2_sentiment_sensitivity(receipt: Path) -> dict[str, object]:
    from experiments.final_q2_lockbox_state import load_opening_identity

    identity = load_opening_identity()
    root = CODE_ROOT / "experiments" / "cache" / "q2_sentiment_sensitivity"
    manifest_path = root / "results" / "manifest.json"
    manifest = _read_json(manifest_path)
    if manifest.get("opening_identity") != identity.to_dict():
        raise ValueError("Q2 sentiment sensitivity opening identity changed")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or not artifact_hashes:
        raise ValueError("Q2 sentiment sensitivity artifact registry is missing")
    for name, expected in artifact_hashes.items():
        path = manifest_path.parent / str(name)
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"Q2 sentiment sensitivity artifact changed: {name}")
    payload = {
        "artifacts": len(artifact_hashes),
        "manifest_sha256": sha256_file(manifest_path),
        "protocol_hash": identity.protocol_hash,
        "state": "VERIFIED",
    }
    _atomic_json(receipt, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    stage = commands.add_parser("stage_channel")
    stage.add_argument("--evidence-root", type=Path, default=Path(".source_evidence"))
    stage.add_argument(
        "--manifest", type=Path, default=Path("source_evidence_manifest.json")
    )
    stage.add_argument("--code-root", type=Path, default=CODE_ROOT)
    stage.add_argument(
        "--receipt", type=Path, default=Path(".rebuild/channel_handoffs_staged.json")
    )

    verify = commands.add_parser("verify_q2")
    verify.add_argument(
        "--receipt", type=Path, default=Path(".rebuild/final_q2_verified.json")
    )

    sensitivity = commands.add_parser("verify_q2_sensitivity")
    sensitivity.add_argument(
        "--receipt",
        type=Path,
        default=Path(".rebuild/q2_sentiment_sensitivity_verified.json"),
    )

    args = parser.parse_args(argv)
    if args.action == "stage_channel":
        payload = stage_channel_handoffs(
            args.evidence_root,
            args.manifest,
            args.code_root,
            args.receipt,
        )
    elif args.action == "verify_q2":
        payload = verify_completed_q2(args.receipt)
    else:
        payload = verify_q2_sentiment_sensitivity(args.receipt)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "stage_channel_handoffs",
    "verify_completed_q2",
    "verify_q2_sentiment_sensitivity",
]
