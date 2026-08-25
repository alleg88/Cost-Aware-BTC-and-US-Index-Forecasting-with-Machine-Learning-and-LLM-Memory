"""Run checks from the code ZIP; add the source ZIP to rebuild results."""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

PREFIX = "cost-aware-market-forecasting"
ACTIONS = ("Check installation", "Rebuild results")


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def extract_archives(code_zip: Path, data_zip: Path | None, destination: Path) -> Path:
    """Validate members before writing; preserve an existing matching run."""
    destination = Path(destination).resolve()
    identity = {"code": _sha(code_zip), "data": _sha(data_zip) if data_zip else None}
    marker = destination / ".archives.json"
    root = destination / PREFIX
    add_sources = False
    if destination.exists():
        if marker.is_file() and root.is_dir():
            previous = json.loads(marker.read_text())
            if previous == identity:
                return root
            add_sources = (previous == {"code": identity["code"], "data": None}
                           and data_zip is not None)
        if not add_sources:
            raise ValueError("This folder contains another run. Use a new folder or restart the Colab runtime.")
    with ZipFile(code_zip) as code, (ZipFile(data_zip) if data_zip else nullcontext()) as data:
        packages = [(code, PREFIX + "/")]
        if data is not None:
            packages.append((data, PREFIX + "/code/.source_evidence/"))
        for archive, prefix in packages:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("Duplicate ZIP entries; download the archive again.")
            for item in archive.infolist():
                path = PurePosixPath(item.filename)
                if (not item.filename.startswith(prefix) or path.is_absolute()
                        or ".." in path.parts or ".git" in path.parts
                        or "\\" in item.filename or ":" in item.filename
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError(f"Unsafe ZIP entry: {item.filename}")
            if archive.testzip() is not None:
                raise ValueError("Damaged ZIP; download the archive again.")
        manifest = json.loads(code.read(PREFIX + "/release_files.json"))
        if not isinstance(manifest, dict) or not {"prepare_project.py", "code/pyproject.toml"} <= manifest.keys():
            raise ValueError("The code ZIP is missing its file manifest.")
        expected_names = {PREFIX + "/" + name for name in manifest} | {PREFIX + "/release_files.json"}
        if {item.filename for item in code.infolist() if not item.is_dir()} != expected_names:
            raise ValueError("The code ZIP does not match its file manifest.")
        for name, expected in manifest.items():
            with code.open(PREFIX + "/" + name) as handle:
                if hashlib.file_digest(handle, "sha256").hexdigest() != expected:
                    raise ValueError(f"Code checksum mismatch: {name}")
        reuse_sources = add_sources and (root / "code/.source_evidence").exists()
        if reuse_sources:
            # A previous run may have finished the rename but not the marker update.
            for item in data.infolist():
                if item.is_dir():
                    continue
                existing = destination / item.filename
                with data.open(item) as handle:
                    expected = hashlib.file_digest(handle, "sha256").hexdigest()
                if not existing.is_file() or _sha(existing) != expected:
                    raise ValueError("Existing source files differ from this ZIP; use a new output folder.")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".zip-extract-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "contents"
            staging.mkdir()
            if not add_sources:
                code.extractall(staging)
            if data is not None and not reuse_sources:
                data.extractall(staging)
            staged_marker = staging / ".archives.json"
            staged_marker.write_text(json.dumps(identity), encoding="utf-8")
            if add_sources:
                if not reuse_sources:
                    (staging / PREFIX / "code/.source_evidence").rename(root / "code/.source_evidence")
                staged_marker.replace(marker)
            else:
                staging.rename(destination)
    return root


def _run(command: list[str], cwd: Path, title: str) -> None:
    print(title, flush=True)
    with (cwd / "run.log").open("a", encoding="utf-8") as log:
        result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        with (cwd / "run.log").open(encoding="utf-8", errors="replace") as log:
            print("".join(deque(log, maxlen=30)), flush=True)
        raise RuntimeError(f"{title} failed. See {cwd / 'run.log'} for details.")


def run(action: str = "Check installation", *, archive_dir: Path = Path.cwd(),
        destination: Path = Path("/content/market-forecasting"), native: bool = False) -> Path:
    """Use Colab's Python when requested; preserve the pinned local setup."""
    if action not in ACTIONS:
        raise ValueError(f"Choose one of: {', '.join(ACTIONS)}")
    if shutil.which("git") is None:
        raise RuntimeError("Git is required. Colab includes it; locally, install Git and reopen the terminal.")
    code_zip = Path(archive_dir) / f"{PREFIX}-code.zip"
    data_zip = Path(archive_dir) / f"{PREFIX}-sources.zip"
    if not code_zip.is_file():
        raise ValueError(f"Upload {PREFIX}-code.zip before running this cell.")
    if not data_zip.is_file():
        if action == "Rebuild results":
            raise ValueError(f"Rebuild results also requires {PREFIX}-sources.zip. Upload both ZIPs.")
        data_zip = None
    print("Checking and unpacking ZIPs...", flush=True)
    root = extract_archives(code_zip, data_zip, destination)
    code = root / "code"
    if native:
        python = Path(sys.executable)
        print(f"Using Colab Python {sys.version.split()[0]} (no separate environment).", flush=True)
        _run([str(python), "-m", "experiments.colab_runtime", "--install"], code, "Installing missing packages...")
    else:
        uv = shutil.which("uv")
        if uv is None:
            _run([sys.executable, "-m", "pip", "install", "-q", "uv"], root, "Installing the environment manager...")
        uv_command = [uv] if uv else [sys.executable, "-m", "uv"]
        environment = code / ".venv"
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            _run([*uv_command, "venv", "--python", "3.12.10", str(environment)], root, "Preparing Python 3.12.10...")
        _run([*uv_command, "pip", "install", "--python", str(python), "--index-url",
              "https://download.pytorch.org/whl/cpu", "torch==2.13.0"], root, "Installing CPU PyTorch...")
        _run([*uv_command, "pip", "install", "--python", str(python), "-r", "requirements-repro.txt"], code,
             "Installing project dependencies...")
    _run([str(python), str(root / "prepare_project.py")], root, "Preparing the project...")
    if data_zip is not None:
        _run([str(python), "-m", "experiments.source_evidence", "verify", "--source", ".source_evidence",
              "--manifest", ".source_evidence/source_evidence_manifest.json"], code, "Checking source data...")
    else:
        print("Code-only check: source data are not loaded or verified.", flush=True)
    _run([str(python), "-m", "experiments.reproduce_tracked"], code, "Checking installation and running tests (several minutes)...")
    _run([str(python), "-m", "experiments.reproduce_source", "--audit-only"], code, "Checking the calculation sequence...")
    if action == "Rebuild results":
        _run([str(python), "-m", "experiments.reproduce_source"], code, "Rebuilding results (this can take many hours)...")
        _run([str(python), "-m", "experiments.reproduce_notebooks"], code, "Updating the result notebooks...")
    print("Results rebuilt." if action == "Rebuild results" else "Installation checks passed; calculations were not rerun.", flush=True)
    print(f"Notebooks: {code / 'notebooks'}", flush=True)
    return root


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true", help="also refit models and rebuild all results")
    parser.add_argument("--archives", type=Path, default=Path.cwd(), help="folder containing the code ZIP and optional source ZIP")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "market-forecasting-run")
    args = parser.parse_args()
    run("Rebuild results" if args.rebuild else "Check installation", archive_dir=args.archives, destination=args.output)
