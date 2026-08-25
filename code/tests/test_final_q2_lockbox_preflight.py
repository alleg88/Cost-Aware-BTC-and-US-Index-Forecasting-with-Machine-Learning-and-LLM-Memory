from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from experiments.final_q2_lockbox_preflight import (
    build_prelockbox_manifest,
    q2_source_registry,
    validate_manifest_commit_binding,
    verify_provider_checksums,
    write_manifest,
)


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_q2_source_registry_is_complete_and_deterministic() -> None:
    registry = q2_source_registry(Path("C:/fixture/code"))

    assert len(registry) == 214
    assert len(set(registry)) == len(registry)
    assert sum(key.startswith("btc_metrics_") and key.endswith("_zip") for key in registry) == 91
    assert sum(key.startswith("btc_metrics_") and key.endswith("_checksum") for key in registry) == 91
    assert {
        "btc_m15_q2",
        "btc_m1_q2_source",
        "btc_positioning_q2_source",
        "usa500_bid_q2_source",
        "usa500_ask_q2_source",
        "usatech_bid_q2_source",
        "usatech_ask_q2_source",
        "volidx_bid_q2_source",
        "volidx_ask_q2_source",
        "gdelt_usa500_q2_source",
        "direct_events_usa500_q2_source",
        "gdelt_usatech_q2_source",
        "direct_events_usatech_q2_source",
        "fred_q2_source",
    }.issubset(registry)


def test_provider_checksum_verification_is_byte_exact(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    checksum = tmp_path / "source.zip.CHECKSUM"
    digest = _write(archive, b"provider bytes")
    checksum.write_text(f"{digest}  source.zip\n", encoding="utf-8")

    verified = verify_provider_checksums(
        {"fixture_zip": archive, "fixture_checksum": checksum}
    )

    assert verified == {"fixture_zip": digest}
    checksum.write_text(f"{'0' * 64}  source.zip\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_provider_checksums(
            {"fixture_zip": archive, "fixture_checksum": checksum}
        )


def test_manifest_builder_binds_every_input_without_decoding_q2(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "protocol_version": "fixture",
                "start_utc": "2026-04-01T00:00:00+00:00",
                "end_utc": "2026-07-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    q2 = tmp_path / "q2.bin"
    warmup = tmp_path / "warmup.bin"
    critical = tmp_path / "runner.py"
    estimator = tmp_path / "estimator.joblib"
    rebuilt = tmp_path / "rebuilt.parquet"
    _write(q2, b"opaque q2")
    _write(warmup, b"pre q2")
    _write(critical, b"code")
    estimator_hash = _write(estimator, b"model")
    rebuilt_hash = _write(rebuilt, b"panel")
    reconstruction_manifest = tmp_path / "reconstruction_manifest.json"
    reconstruction_manifest.write_text(
        json.dumps(
            {
                "fit_key": "fixture:model",
                "q2_decoded": False,
                "serialized_estimator": estimator.as_posix(),
                "serialized_estimator_sha256": estimator_hash,
                "rebuilt_panel": rebuilt.as_posix(),
                "rebuilt_panel_sha256": rebuilt_hash,
            }
        ),
        encoding="utf-8",
    )

    manifest = build_prelockbox_manifest(
        implementation_commit="a" * 40,
        protocol_path=protocol,
        protocol_hash="b" * 64,
        candidates=[{"candidate_id": "fixture", "stream": "btcusdt"}],
        costs={"btcusdt": {"per_side_bps": 5.0, "round_trip_bps": 10.0}},
        q2_sources={"q2": q2},
        warmups={"warmup": warmup},
        reconstruction_manifests=[reconstruction_manifest],
        critical_sources={"runner": critical},
        model_identities={"fixture": {"digest": "c" * 64}},
    )

    assert manifest["q2_decoded"] is False
    assert manifest["implementation_commit"] == "a" * 40
    assert manifest["q2_sources"]["q2"]["sha256"] == hashlib.sha256(b"opaque q2").hexdigest()
    assert manifest["warmups"]["warmup"]["sha256"] == hashlib.sha256(b"pre q2").hexdigest()
    assert manifest["reconstructed_estimators"][0]["fit_key"] == "fixture:model"


def test_manifest_commit_binding_requires_manifest_as_only_child_commit(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"implementation_commit": "a" * 40}), encoding="utf-8"
    )

    binding = validate_manifest_commit_binding(
        manifest,
        manifest_commit="b" * 40,
        parent_commit="a" * 40,
        committed_manifest_bytes=manifest.read_bytes(),
    )

    assert binding.implementation_commit == "a" * 40
    assert binding.manifest_commit == "b" * 40
    with pytest.raises(ValueError, match="parent"):
        validate_manifest_commit_binding(
            manifest,
            manifest_commit="b" * 40,
            parent_commit="c" * 40,
            committed_manifest_bytes=manifest.read_bytes(),
        )


def test_manifest_writer_uses_git_stable_utf8_lf_bytes(tmp_path: Path) -> None:
    path = write_manifest(tmp_path / "manifest.json", {"value": "line"})

    assert path.read_bytes().endswith(b"\n")
    assert b"\r\n" not in path.read_bytes()
