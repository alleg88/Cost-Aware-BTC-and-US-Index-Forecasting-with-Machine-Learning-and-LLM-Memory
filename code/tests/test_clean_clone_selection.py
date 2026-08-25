import subprocess
import sys

import pytest

from experiments.clean_clone_tests import (
    CleanCloneManifestError,
    DEFAULT_LOCAL_EVIDENCE_MANIFEST,
    load_local_evidence_nodeids,
    partition_clean_clone_nodes,
)


def test_manifest_loader_accepts_comments_and_rejects_duplicates(tmp_path):
    manifest = tmp_path / "local-evidence.txt"
    manifest.write_text(
        "# requires local data\n"
        "tests/test_alpha.py::test_evidence\n"
        "\n"
        "tests/test_beta.py::test_evidence\n",
        encoding="utf-8",
    )

    assert load_local_evidence_nodeids(manifest) == (
        "tests/test_alpha.py::test_evidence",
        "tests/test_beta.py::test_evidence",
    )

    manifest.write_text(
        "tests/test_alpha.py::test_evidence\n"
        "tests/test_alpha.py::test_evidence\n",
        encoding="utf-8",
    )
    with pytest.raises(CleanCloneManifestError, match="duplicate"):
        load_local_evidence_nodeids(manifest)


def test_clean_clone_partition_skips_only_exact_registered_nodes():
    runnable, skipped = partition_clean_clone_nodes(
        (
            "tests/test_alpha.py::test_unit",
            "tests/test_alpha.py::test_evidence",
            "tests/test_beta.py::test_unit",
        ),
        ("tests/test_alpha.py::test_evidence",),
    )

    assert runnable == (
        "tests/test_alpha.py::test_unit",
        "tests/test_beta.py::test_unit",
    )
    assert skipped == ("tests/test_alpha.py::test_evidence",)


def test_clean_clone_partition_rejects_a_stale_manifest_entry():
    with pytest.raises(CleanCloneManifestError, match="not collected"):
        partition_clean_clone_nodes(
            ("tests/test_alpha.py::test_unit",),
            ("tests/test_removed.py::test_old",),
        )


def test_repository_manifest_names_the_99_local_evidence_checks():
    nodeids = load_local_evidence_nodeids(DEFAULT_LOCAL_EVIDENCE_MANIFEST)

    assert len(nodeids) == 99
    assert "tests/test_raw_hold_control.py::test_build_raw_hold_summary_reuses_2024_frozen_predictions" in nodeids
    assert "tests/test_reflection_v3_reconciliation.py::test_final_experiment_reconciles_registered_continuous_protocol" in nodeids


def test_pytest_exposes_the_clean_clone_mode():
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "--help"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert "--clean-clone" in completed.stdout
