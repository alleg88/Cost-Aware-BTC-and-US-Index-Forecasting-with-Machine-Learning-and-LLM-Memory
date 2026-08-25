"""Materialise the immutable qualified-Union v1 handoff."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.qualified_union import (
    CODE_ROOT,
    FORWARD_START,
    H1_START,
    LOCKBOX_START,
    MEMBERS,
    build_union_frame,
    monthly_summary,
    prediction_paths,
    protocol,
    simulate_union,
    summarize,
)


OUTPUT_ROOT = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _store_phase(stage: str, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, object], pd.DataFrame]:
    frame = build_union_frame(stage)
    ledger, per_bar = simulate_union(frame["union_signal"], start=start, end=end)
    frame.rename_axis("timestamp").reset_index().to_parquet(
        OUTPUT_ROOT / f"{stage}_signals.parquet", index=False
    )
    ledger.to_parquet(OUTPUT_ROOT / f"{stage}_ledger.parquet", index=False)
    per_bar.rename_axis("timestamp").reset_index().to_parquet(
        OUTPUT_ROOT / f"{stage}_per_bar.parquet", index=False
    )
    return summarize(ledger, per_bar, phase=stage), monthly_summary(per_bar, phase=stage)


def run() -> dict[str, object]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frozen_protocol = protocol()
    _write_json(OUTPUT_ROOT / "protocol.json", frozen_protocol)

    h1_summary, h1_monthly = _store_phase("h1", H1_START, FORWARD_START)
    forward_summary, forward_monthly = _store_phase(
        "forward", FORWARD_START, LOCKBOX_START
    )
    summary = pd.DataFrame([h1_summary, forward_summary])
    summary.to_csv(OUTPUT_ROOT / "summary.csv", index=False)
    pd.concat([h1_monthly, forward_monthly], ignore_index=True).to_csv(
        OUTPUT_ROOT / "monthly.csv", index=False
    )

    source_paths = sorted(
        {
            path
            for stage in ("h1", "forward")
            for member in MEMBERS
            for path in prediction_paths(member, stage)
        }
    )
    source_hashes = {
        str(path.relative_to(CODE_ROOT)).replace("\\", "/"): _sha256(path)
        for path in source_paths
    }
    artifact_paths = sorted(
        path
        for path in OUTPUT_ROOT.iterdir()
        if path.is_file() and path.name != "manifest.json"
    )
    artifact_hashes = {path.name: _sha256(path) for path in artifact_paths}
    manifest = {
        "protocol_version": frozen_protocol["protocol_version"],
        "protocol_sha256": artifact_hashes["protocol.json"],
        "source_hashes": source_hashes,
        "artifact_hashes": artifact_hashes,
        "forward_replay_count": 1,
        "lockbox_2026_q2_used": False,
        "max_scored_timestamp": str(
            pd.read_parquet(OUTPUT_ROOT / "forward_signals.parquet")["timestamp"].max()
        ),
        "decision": "freeze_union_v1",
    }
    _write_json(OUTPUT_ROOT / "manifest.json", manifest)
    return manifest


def main() -> int:
    manifest = run()
    summary = pd.read_csv(OUTPUT_ROOT / "summary.csv")
    print(summary.to_string(index=False))
    print(f"manifest: {manifest['protocol_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
