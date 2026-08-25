from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest

import experiments.final_q2_lockbox_state as state_module
from experiments.final_q2_lockbox_state import (
    OpeningIdentity,
    assert_exact_resume,
    begin_global_open,
    load_opening_identity,
    mark_complete,
    mark_failed_after_open,
    require_global_opening,
    write_aborted_before_open,
)


def _identity(**changes) -> OpeningIdentity:
    base = OpeningIdentity(
        implementation_commit="a" * 40,
        manifest_commit="b" * 40,
        protocol_hash="c" * 64,
        manifest_sha256="d" * 64,
        q2_source_hashes={"btc_m15": "e" * 64},
    )
    return replace(base, **changes)


def test_global_sentinel_allows_exactly_one_protocol(tmp_path: Path) -> None:
    identity = _identity()
    begin_global_open(identity, root=tmp_path)

    with pytest.raises(FileExistsError):
        begin_global_open(
            _identity(protocol_hash="f" * 64, manifest_commit="1" * 40),
            root=tmp_path,
        )

    saved = load_opening_identity(tmp_path / "OPENED.json")
    assert saved == identity


def test_resume_requires_exact_identity(tmp_path: Path) -> None:
    identity = _identity()
    begin_global_open(identity, root=tmp_path)

    assert_exact_resume(identity, root=tmp_path)
    with pytest.raises(PermissionError, match="identity"):
        assert_exact_resume(replace(identity, implementation_commit="9" * 40), root=tmp_path)


def test_require_global_opening_reads_the_physical_repository_sentinel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "OPENED.json"
    monkeypatch.setattr(state_module, "GLOBAL_SENTINEL_PATH", sentinel)
    identity = _identity()

    with pytest.raises(PermissionError, match="OPENED"):
        require_global_opening(identity)

    begin_global_open(identity, root=tmp_path)
    assert require_global_opening(identity) == identity


def test_aborted_before_open_does_not_create_global_sentinel(tmp_path: Path) -> None:
    identity = _identity()
    path = write_aborted_before_open(identity, "digest mismatch", root=tmp_path)

    assert path.name == "ABORTED_BEFORE_OPEN.json"
    assert json.loads(path.read_text(encoding="utf-8"))["state"] == "ABORTED_BEFORE_OPEN"
    assert not (tmp_path / "OPENED.json").exists()


def test_post_open_terminal_files_are_protocol_bound(tmp_path: Path) -> None:
    identity = _identity()
    begin_global_open(identity, root=tmp_path)

    failed = mark_failed_after_open(identity, "worker interrupted", root=tmp_path)
    complete = mark_complete(identity, {"manifest": "f" * 64}, root=tmp_path)

    assert json.loads(failed.read_text(encoding="utf-8"))["state"] == "FAILED_AFTER_OPEN"
    assert json.loads(complete.read_text(encoding="utf-8"))["state"] == "COMPLETE"
    assert failed.parent.name == complete.parent.name == identity.protocol_hash
    with pytest.raises(PermissionError, match="already COMPLETE"):
        assert_exact_resume(identity, root=tmp_path)
    assert (tmp_path / "OPENED.json").exists()


def test_opening_identity_rejects_malformed_hashes() -> None:
    with pytest.raises(ValueError, match="commit"):
        _identity(implementation_commit="short").validate()
    with pytest.raises(ValueError, match="sha256"):
        _identity(manifest_sha256="bad").validate()
