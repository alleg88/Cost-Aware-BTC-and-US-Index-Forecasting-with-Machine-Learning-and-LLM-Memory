import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
from zipfile import ZipFile

import pytest

from experiments.repository_release import (
    ReleaseAuditError,
    audit_repository,
    validate_binance_csv_archive,
    validated_binance_csv_candidates,
)


REPO_ROOT = Path(__file__).parents[2]


def _write_verified_archive(root: Path, name: str = "BTCUSDT-15m-2025-01.csv") -> Path:
    raw_dir = root / "binance"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / name
    csv_path.write_text("open_time,open\n1,2\n", encoding="utf-8")
    zip_path = csv_path.with_suffix(".zip")
    with ZipFile(zip_path, "w") as archive:
        archive.write(csv_path, arcname=csv_path.name)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    Path(str(zip_path) + ".CHECKSUM").write_text(
        f"{digest}  {zip_path.name}\n",
        encoding="utf-8",
    )
    return csv_path


def test_validate_binance_csv_requires_verified_matching_archive(tmp_path):
    csv_path = _write_verified_archive(tmp_path)

    result = validate_binance_csv_archive(csv_path)

    assert result["symbol"] == "BTCUSDT"
    assert result["archive_sha256"] == hashlib.sha256(
        csv_path.with_suffix(".zip").read_bytes()
    ).hexdigest()
    assert result["csv_bytes"] == csv_path.stat().st_size


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_symbol", "ineligible Binance CSV"),
        ("wrong_checksum", "checksum mismatch"),
        ("wrong_member", "CSV member absent"),
    ],
)
def test_validate_binance_csv_fails_closed(tmp_path, mutation, message):
    csv_path = _write_verified_archive(tmp_path)
    if mutation == "wrong_symbol":
        csv_path = csv_path.rename(csv_path.with_name("ETHUSDT-15m-2025-01.csv"))
    elif mutation == "wrong_checksum":
        Path(str(csv_path.with_suffix(".zip")) + ".CHECKSUM").write_text(
            f"{'0' * 64}  {csv_path.with_suffix('.zip').name}\n",
            encoding="utf-8",
        )
    else:
        zip_path = csv_path.with_suffix(".zip")
        with ZipFile(zip_path, "w") as archive:
            archive.writestr("different.csv", "x")
        digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        Path(str(zip_path) + ".CHECKSUM").write_text(
            f"{digest}  {zip_path.name}\n",
            encoding="utf-8",
        )

    with pytest.raises(ReleaseAuditError, match=message):
        validate_binance_csv_archive(csv_path)


def test_candidate_scan_is_top_level_btcusdt_only(tmp_path):
    first = _write_verified_archive(tmp_path, "BTCUSDT-1m-2025-02.csv")
    second = _write_verified_archive(tmp_path, "BTCUSDT-15m-2025-01.csv")
    nested = tmp_path / "binance" / "futures" / "BTCUSDT-15m-2025-01.csv"
    nested.parent.mkdir()
    nested.write_text("unique", encoding="utf-8")

    rows = validated_binance_csv_candidates(tmp_path / "binance")

    assert [row["csv"] for row in rows] == sorted([first.name, second.name])
    assert nested.exists()


def test_live_repository_release_audit_is_green():
    report = audit_repository(REPO_ROOT)

    assert report["status"] == "READY"
    assert report["canonical_notebooks"] == 22
    assert report["q2_market_rows_opened"] is True
    assert report["launcher_notebooks"] == 0
    assert report["notebook_errors"] == 0
    assert report["unexecuted_canonical_code_cells"] == 0
    assert report["tracked_forbidden"] == []
    assert report["tracked_cache_artifacts"] == 211
    assert report["local_evidence_tests"] == 99
    json.dumps(report)


def test_final_direct_dependencies_are_exactly_pinned():
    pyproject = tomllib.loads((REPO_ROOT / "code" / "pyproject.toml").read_text("utf-8"))
    project = pyproject["project"]
    dependencies = set(project["dependencies"])
    extras = set(project["optional-dependencies"]["extras"])
    acquisition = set(project["optional-dependencies"]["acquisition"])

    assert "torch==2.13.0" in dependencies
    assert {
        "optuna==4.9.0",
        "transformers==5.14.1",
        "sentencepiece==0.2.2",
        "protobuf==7.35.1",
        "sentence-transformers==5.7.0",
        "ollama==0.6.2",
        "nbformat==5.10.4",
        "nbclient==0.11.0",
        "ipykernel==7.3.0",
    }.issubset(extras)
    assert acquisition == {"google-cloud-bigquery==3.44.0"}
    repro = (REPO_ROOT / "code" / "requirements-repro.txt").read_text("utf-8")
    assert repro.splitlines()[-1] == "-e .[extras]"
    assert ".[extras,acquisition]" in repro


def test_built_wheel_contains_nested_reflection_runtime(tmp_path):
    """Catch package discovery that drops the v2/v3/v4 runtime from installs."""
    code_root = REPO_ROOT / "code"
    source_root = tmp_path / "source"
    wheel_root = tmp_path / "wheel"
    source_root.mkdir()
    wheel_root.mkdir()
    shutil.copy2(code_root / "pyproject.toml", source_root / "pyproject.toml")
    shutil.copy2(code_root / "conftest.py", source_root / "conftest.py")
    for package in (
        "data",
        "features",
        "evaluation",
        "models",
        "sentiment",
        "experiments",
        "memory",
        "ensemble",
        "reflection_agent",
    ):
        for source in (code_root / package).rglob("*.py"):
            destination = source_root / source.relative_to(code_root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    script = (
        "import os; "
        "from setuptools.build_meta import build_wheel; "
        f"os.chdir({str(source_root)!r}); "
        f"build_wheel({str(wheel_root)!r})"
    )
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True)
    wheel_path = next(wheel_root.glob("*.whl"))
    with ZipFile(wheel_path) as wheel:
        members = set(wheel.namelist())

    assert {
        "reflection_agent/v2/transport.py",
        "reflection_agent/v3/opportunities.py",
        "reflection_agent/v4/router.py",
    }.issubset(members)


def test_github_workflow_verifies_a_clean_tracked_checkout():
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "reproducibility.yml"
    ).read_text("utf-8")

    for required in (
        "name: Clean clone verification",
        "actions/checkout@v7",
        "actions/setup-python@v7",
        "python-version: '3.12.10'",
        "cache: 'pip'",
        "requirements-repro.txt",
        "https://download.pytorch.org/whl/cpu",
        "torch==2.13.0",
        "python -m pip check",
        "python -m experiments.repository_release --repo-root ..",
        "python -m experiments.reproduce_source --audit-only",
        "python -m experiments.reproduce_notebooks --help",
        "python -m pytest -q -p no:cacheprovider --clean-clone",
    ):
        assert required in workflow


def test_reproducibility_guide_defines_source_to_results_boundary():
    guide = (REPO_ROOT / "REPRODUCIBILITY.md").read_text("utf-8")

    for required in (
        "Tracked code, installation and tests",
        "Saved executed notebooks",
        "Public, checksum-verified downloads",
        "Portable source-evidence bundle",
        "Generated calculations",
        "Truth Social",
        "GDELT GKG via BigQuery",
        "--clean-clone",
        "Q2 2026 was opened once",
        "22 canonical readers",
        "python -m experiments.source_evidence restore",
        "python -m experiments.reproduce_source",
        "python -m experiments.reproduce_notebooks",
        "OLLAMA_API_KEY",
        "code/notebooks/README.md",
    ):
        assert required in guide


def test_readme_navigation_opens_the_notebook_and_input_instructions():
    import re

    for readme_path in (REPO_ROOT / "README.md", REPO_ROOT / "code/README.md"):
        text = readme_path.read_text(encoding="utf-8")
        links = re.findall(r"\]\(([^()]+)\)", text)
        assert links
        local_links = [link.split("#", 1)[0] for link in links
                       if not re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", link)
                       and not link.startswith("#")]
        targets = [(readme_path.parent / link).resolve() for link in local_links]
        assert all(target.is_file() for target in targets)
        assert REPO_ROOT / "code/notebooks/README.md" in targets
        assert REPO_ROOT / "code/notebooks/DATA.md" in targets


def test_reproduction_requirements_document_the_cloud_key_without_storing_it():
    requirements = (REPO_ROOT / "code" / "requirements-repro.txt").read_text("utf-8")
    assert "ollama" in requirements.lower()
    assert "OLLAMA_API_KEY" in requirements
    assert "Bearer " not in requirements


def test_local_rebuild_and_source_bundle_roots_are_ignored():
    ignore = (REPO_ROOT / "code" / ".gitignore").read_text("utf-8").splitlines()

    assert {".pytest_tmp/", ".rebuild/", ".source_evidence/"}.issubset(ignore)
