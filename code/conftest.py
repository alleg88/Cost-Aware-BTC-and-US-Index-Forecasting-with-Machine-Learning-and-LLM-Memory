"""Put the code/ root on sys.path so modules import as `data.load`, `evaluation.splits`, etc.

pytest imports this file from the rootdir and prepends its directory to sys.path, which
makes the namespace packages (data/, features/, evaluation/, models/) importable without
an install step. The notebook does the same via a small sys.path bootstrap cell.
"""
import sys
from pathlib import Path

import pytest

from experiments.clean_clone_tests import (
    CleanCloneManifestError,
    load_local_evidence_nodeids,
    partition_clean_clone_nodes,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def pytest_addoption(parser):
    group = parser.getgroup("reproducibility")
    group.addoption(
        "--clean-clone",
        action="store_true",
        help="skip only registered tests that require ignored local evidence",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--clean-clone"):
        return
    try:
        _, skipped = partition_clean_clone_nodes(
            (item.nodeid for item in items),
            load_local_evidence_nodeids(),
        )
    except CleanCloneManifestError as error:
        raise pytest.UsageError(str(error)) from error
    skipped_set = set(skipped)
    marker = pytest.mark.skip(reason="requires ignored local scientific evidence")
    for item in items:
        if item.nodeid in skipped_set:
            item.add_marker(marker)
