"""Read-only release audit for the dissertation repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import nbformat

from experiments.notebook_hygiene import NOTEBOOK_SEQUENCES
from experiments.clean_clone_tests import load_local_evidence_nodeids


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO_ROOT = CODE_ROOT.parent
_BINANCE_CSV = re.compile(r"BTCUSDT-[A-Za-z0-9]+-\d{4}-\d{2}\.csv")


class ReleaseAuditError(RuntimeError):
    """Raised when release evidence fails closed validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_binance_csv_archive(csv_path: Path) -> dict[str, object]:
    """Prove that one top-level BTCUSDT CSV has an exact restorable ZIP."""
    csv_path = Path(csv_path).resolve(strict=True)
    if csv_path.parent.name != "binance" or _BINANCE_CSV.fullmatch(csv_path.name) is None:
        raise ReleaseAuditError(f"ineligible Binance CSV: {csv_path}")

    zip_path = csv_path.with_suffix(".zip")
    checksum_path = Path(str(zip_path) + ".CHECKSUM")
    if not zip_path.is_file() or not checksum_path.is_file():
        raise ReleaseAuditError(f"archive pair absent: {csv_path.name}")

    tokens = checksum_path.read_text(encoding="utf-8").split()
    if len(tokens) < 2 or Path(tokens[1]).name != zip_path.name:
        raise ReleaseAuditError(f"invalid checksum record: {checksum_path.name}")
    expected = tokens[0].lower()
    actual = sha256_file(zip_path)
    if actual != expected:
        raise ReleaseAuditError(f"checksum mismatch: {zip_path.name}")

    try:
        with ZipFile(zip_path) as archive:
            if archive.testzip() is not None:
                raise ReleaseAuditError(f"corrupt ZIP member: {zip_path.name}")
            members = {Path(name).name for name in archive.namelist()}
    except BadZipFile as error:
        raise ReleaseAuditError(f"invalid ZIP: {zip_path.name}") from error
    if csv_path.name not in members:
        raise ReleaseAuditError(f"CSV member absent: {zip_path.name}")

    return {
        "csv": csv_path.name,
        "zip": zip_path.name,
        "symbol": "BTCUSDT",
        "archive_sha256": actual,
        "csv_bytes": int(csv_path.stat().st_size),
    }


def validated_binance_csv_candidates(raw_dir: Path) -> list[dict[str, object]]:
    """Return only verified top-level BTCUSDT CSV duplicates, sorted by name."""
    raw_dir = Path(raw_dir).resolve(strict=True)
    if raw_dir.name != "binance" or not raw_dir.is_dir():
        raise ReleaseAuditError(f"unexpected Binance raw directory: {raw_dir}")
    return [
        validate_binance_csv_archive(path)
        for path in sorted(raw_dir.glob("BTCUSDT-*.csv"), key=lambda item: item.name)
    ]


def _tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    return sorted(path for path in result.stdout.decode("utf-8").split("\0") if path)


def _forbidden_tracked(paths: list[str]) -> list[str]:
    forbidden: list[str] = []
    for path in paths:
        normalized = path.replace("\\", "/")
        lower = normalized.lower()
        basename = Path(lower).name
        if lower.startswith(
            ("code/data/raw/", "code/sentiment/raw/", "code/sentiment/secrets/")
        ):
            forbidden.append(normalized)
        elif lower.startswith("code/data/") and Path(lower).suffix in {
            ".csv",
            ".parquet",
            ".zip",
        }:
            forbidden.append(normalized)
        elif basename == ".env" or basename.endswith(".key"):
            forbidden.append(normalized)
        elif "token" in basename and basename.endswith(".txt"):
            forbidden.append(normalized)
    return forbidden


def _notebook_audit(code_root: Path) -> dict[str, int]:
    notebook_root = code_root / "notebooks"
    canonical = [name for names in NOTEBOOK_SEQUENCES.values() for name in names]
    actual = {
        path.name
        for path in notebook_root.glob("*.ipynb")
        if path.name != "00_run_in_colab.ipynb"
    }
    if actual != set(canonical) or len(canonical) != len(set(canonical)):
        raise ReleaseAuditError("canonical notebook catalog mismatch")
    launcher = notebook_root / "00_run_in_colab.ipynb"
    if not launcher.is_file():
        raise ReleaseAuditError("00_run_in_colab.ipynb is missing")

    errors = 0
    unexecuted = 0
    for name in canonical:
        notebook = nbformat.read(notebook_root / name, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            if cell.execution_count is None:
                unexecuted += 1
            errors += sum(
                output.get("output_type") == "error" for output in cell.get("outputs", [])
            )
    return {
        "canonical_notebooks": len(canonical),
        "launcher_notebooks": 1,
        "notebook_errors": int(errors),
        "unexecuted_canonical_code_cells": int(unexecuted),
    }


def audit_repository(repo_root: Path) -> dict[str, object]:
    """Audit only tracked code and saved notebook outputs; never load market rows."""
    repo_root = Path(repo_root).resolve(strict=True)
    code_root = repo_root / "code"
    if not (repo_root / ".git").exists() or not (code_root / "pyproject.toml").is_file():
        raise ReleaseAuditError(f"unexpected repository root: {repo_root}")

    tracked = _tracked_files(repo_root)
    forbidden = _forbidden_tracked(tracked)
    notebook_report = _notebook_audit(code_root)
    tracked_cache = sum(
        path.replace("\\", "/").startswith("code/experiments/cache/")
        for path in tracked
    )
    local_evidence_tests = load_local_evidence_nodeids(
        code_root / "tests" / "local_evidence_tests.txt"
    )
    problems: list[str] = []
    if forbidden:
        problems.append("forbidden raw or secret files are tracked")
    if notebook_report["notebook_errors"]:
        problems.append("saved notebook errors are present")
    if notebook_report["unexecuted_canonical_code_cells"]:
        problems.append("canonical notebook code cells are unexecuted")
    if tracked_cache == 0:
        problems.append("compact tracked experiment evidence is absent")

    return {
        "status": "READY" if not problems else "NOT_READY",
        **notebook_report,
        "tracked_files": len(tracked),
        "tracked_cache_artifacts": int(tracked_cache),
        "local_evidence_tests": len(local_evidence_tests),
        "tracked_forbidden": forbidden,
        "problems": problems,
        "q2_market_rows_opened": bool(
            nbformat.read(code_root / "notebooks" / "07_final_q2_lockbox.ipynb", as_version=4)
            .metadata.get("final_q2_lockbox")
            == "COMPLETE"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--binance-raw", type=Path)
    args = parser.parse_args()
    try:
        report = audit_repository(args.repo_root)
        if args.binance_raw is not None:
            candidates = validated_binance_csv_candidates(args.binance_raw)
            report["validated_binance_csv_candidates"] = len(candidates)
            report["validated_binance_csv_bytes"] = sum(
                int(row["csv_bytes"]) for row in candidates
            )
    except (OSError, ReleaseAuditError, subprocess.CalledProcessError) as error:
        print(json.dumps({"status": "NOT_READY", "error": str(error)}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
