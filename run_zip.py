"""Start Release-Client.zip or rebuild from the self-contained Release-Rebuild.zip."""
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
ACTIONS = ("Check installation", "Partial check", "Rebuild results")


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
        if not isinstance(manifest, dict) or "code/pyproject.toml" not in manifest:
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


def prepare_reader(notebook_name: str, source: Path,
                   extra_dependencies: tuple[str, ...] = ()) -> Path:
    """Use the same reader preparation from an extracted project or its ZIP."""
    source = Path(source).resolve()
    root = extract_archives(source, None, source.parent / "Release-Client") if source.is_file() else source
    code_root = root / "code"
    if not (code_root / "pyproject.toml").is_file():
        raise FileNotFoundError("Open code/notebooks from the extracted Release-Client.zip.")
    sys.path.insert(0, str(code_root))
    from experiments.notebook_runtime import prepare_notebook

    return prepare_notebook(notebook_name, code_root, extra_dependencies)


def prepare_rebuild(notebook_name: str, source: Path,
                    extra_dependencies: tuple[str, ...] = ()) -> Path:
    """Run the selected experiment's producers, then continue ordinary notebook cells."""
    source = Path(source).resolve()
    if source.is_file():
        with ZipFile(source) as archive:
            if PREFIX + "/code/.source_evidence/source_evidence_manifest.json" not in archive.namelist():
                raise ValueError("Choose Release-Rebuild.zip: the Client archive has no rebuild inputs.")
        print("Checking and unpacking Release-Rebuild.zip...", flush=True)
        root = extract_archives(source, None, source.parent / "Release-Rebuild")
    else:
        root = source
    code_root = root / "code"
    if not (code_root / ".source_evidence/source_evidence_manifest.json").is_file():
        raise ValueError("Open a notebook from Release-Rebuild.zip; its source inputs are required.")
    if not (code_root / "pyproject.toml").is_file() or not (root / "prepare_project.py").is_file():
        raise ValueError("Incomplete Release-Rebuild.zip; extract or upload the full archive.")
    _run([sys.executable, str(root / "prepare_project.py")], root, "Verifying the extracted project...")
    sys.path.insert(0, str(code_root))
    from experiments.notebook_runtime import prepare_notebook
    from experiments.notebook_rebuild import rebuild_notebook_inputs

    print("Preparing packages in this notebook's Python kernel (no developer test suite)...", flush=True)
    prepare_notebook(None, code_root, extra_dependencies)
    rebuild_notebook_inputs(notebook_name, code_root)
    return code_root


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
    archive_dir = Path(archive_dir)
    needs_rebuild = action in {"Partial check", "Rebuild results"}
    release_name = "Release-Rebuild.zip" if needs_rebuild else "Release-Client.zip"
    code_zip = archive_dir / release_name
    data_zip = None
    embedded_sources = False
    if code_zip.is_file():
        with ZipFile(code_zip) as package:
            embedded_sources = PREFIX + "/code/.source_evidence/source_evidence_manifest.json" in package.namelist()
    else:
        # Existing two-archive exports remain usable from the command line.
        code_zip = archive_dir / f"{PREFIX}-code.zip"
        if not code_zip.is_file():
            raise ValueError(f"Place {release_name} beside the starter notebook before running it.")
        legacy_sources = archive_dir / f"{PREFIX}-sources.zip"
        if needs_rebuild and legacy_sources.is_file():
            data_zip = legacy_sources
    if needs_rebuild and not embedded_sources and data_zip is None:
        raise ValueError(f"{action} requires sources. Use Release-Rebuild.zip, not the client ZIP.")
    print(f"Checking and unpacking {code_zip.name}...", flush=True)
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
    if data_zip is not None or embedded_sources:
        _run([str(python), "-m", "experiments.source_evidence", "verify", "--source", ".source_evidence",
              "--manifest", ".source_evidence/source_evidence_manifest.json"], code, "Checking source data...")
    else:
        print("Code-only check: source data are not loaded or verified.", flush=True)
    checks = [str(python), "-m", "experiments.reproduce_tracked"]
    _run([*checks, "--audit-only"], code, "Checking packages and project files (no full test suite)...")
    _run([str(python), "-m", "experiments.reproduce_source", "--audit-only"], code, "Checking the calculation sequence...")
    if action == "Rebuild results":
        _run([str(python), "-m", "experiments.reproduce_source"], code, "Rebuilding results (this can take many hours)...")
        _run([str(python), "-m", "experiments.reproduce_notebooks"], code, "Updating the result notebooks...")
    if action == "Partial check":
        _run([str(python), "-m", "experiments.reproduce_partial"], code,
             "Executing the complete first notebook from supplied inputs (not the full project)...")
    if action == "Rebuild results":
        print("Results rebuilt.", flush=True)
    elif action == "Partial check":
        print("Partial check passed. Full rebuild was not run.", flush=True)
        print(f"New notebook and diagnostic report: {root / 'partial-check'}", flush=True)
    else:
        print("Installation ready. Full tests and model fitting were not run.", flush=True)
    print(f"Notebooks: {code / 'notebooks'}", flush=True)
    return root


if __name__ == "__main__":
    if not Path(__file__).with_name("prepare_project.py").is_file():
        print("Open a notebook in code/notebooks. This helper unpacks the ZIP automatically in Colab; do not run it separately.")
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--rebuild", action="store_true", help="use Release-Rebuild.zip to refit models and rebuild results")
    modes.add_argument("--partial", action="store_true", help="use Release-Rebuild.zip to execute only the first notebook")
    parser.add_argument("--archives", type=Path, default=Path.cwd(), help="folder containing the release ZIP")
    parser.add_argument("--output", type=Path, default=Path.cwd() / "market-forecasting-run")
    args = parser.parse_args()
    action = "Rebuild results" if args.rebuild else "Partial check" if args.partial else "Check installation"
    run(action, archive_dir=args.archives, destination=args.output)
