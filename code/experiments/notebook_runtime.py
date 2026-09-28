"""Prepare a reader without private paths, automatic research runs or test output."""
from __future__ import annotations

from importlib import metadata
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from experiments.colab_runtime import _name, _requirements
from experiments.rebuild_notebook_dependencies import NOTEBOOK_INPUTS


def notebook_inputs(notebook_name: str) -> tuple[str, ...]:
    """Reader inputs, excluding receipts only the full rebuild needs."""
    contracts = {name: paths for group in NOTEBOOK_INPUTS.values() for name, paths in group.items()}
    if notebook_name not in contracts:
        raise ValueError(f"Unknown notebook: {notebook_name}")
    paths = [path for path in contracts[notebook_name] if not path.startswith(".rebuild/")]
    if notebook_name in {"15_RQ3_D_indices_DeBERTa_sentiment.ipynb", "16_RQ3_E_indices_LLM_sentiment.ipynb"}:
        llm = "llm_" if notebook_name == "16_RQ3_E_indices_LLM_sentiment.ipynb" else ""
        for stream in ("usa500", "usatech"):
            paths.append(f"experiments/cache/index_all_model_forward/{stream}")
            for prefix in ("", "direct_events_"):
                paths.append(f"sentiment/raw/scores_{llm}{prefix}{stream}.manifest.json")
        if llm:
            paths.append("sentiment/raw/index_*_identity.json")
    if notebook_name == "10_RQ2_H_indices_all_model_ensemble.ipynb":
        paths.extend(f"experiments/cache/index_replication/{stream}/vix_admission.json"
                     for stream in ("usa500", "usatech"))
    if notebook_name == "11_RQ2_I_indices_policy_comparison.ipynb":
        paths.extend(f"experiments/cache/index_replication/{stream}"
                     for stream in ("usa500", "usatech"))
    if notebook_name == "22_Lockbox_Q2_2026.ipynb":
        paths.append("experiments/cache/final_q2_lockbox/OPENED.json")
    return tuple(dict.fromkeys(paths))


class MissingNotebookInputs(FileNotFoundError):
    """A short, actionable data request in Jupyter and Colab."""

    def _render_traceback_(self):
        return str(self).splitlines()


def _missing_inputs(code_root: Path, notebook_name: str) -> list[str]:
    return [relative for relative in notebook_inputs(notebook_name)
            if not any(path.is_file() or (path.is_dir() and any(p.is_file() for p in path.rglob("*")))
                       for path in code_root.glob(relative))]


def _upload_inputs(code_root: Path, notebook_name: str, missing: list[str]) -> None:
    """Request named inputs once; stage uploads before accepting any user bytes."""
    from google.colab import files

    by_name = {PurePosixPath(relative).name: relative for relative in missing}
    zip_mode = (len(by_name) != len(missing)
                or any(not PurePosixPath(path).suffix or any(c in path for c in "*?[") for path in missing))
    print(f"Required inputs for {notebook_name} (data/results, not project code):", flush=True)
    for relative in missing:
        print(f"  {relative}", flush=True)
    if zip_mode:
        print("Choose files: select Notebook-inputs.zip, containing the paths above relative to code/.", flush=True)
        print("Use the corresponding calculated files from a completed Rebuild run; see notebooks/DATA.md.", flush=True)
    else:
        print("Choose files: select " + ", ".join(by_name) + ". You may select them together.", flush=True)
        print("For the first notebook these files are in the extracted Rebuild archive's code/data/ folder.", flush=True)
    print("Execution waits here for your selection; after upload, the input check continues.", flush=True)
    with TemporaryDirectory(prefix="notebook-input-upload-") as temporary:
        staging = Path(temporary)
        saved = files.upload(target_dir=str(staging))
        if not saved:
            print("No file was uploaded. Supply the named inputs and run this cell again.", flush=True)
            return
        for saved_name in saved:
            saved_path = Path(saved_name)
            if not saved_path.is_absolute():
                saved_path = staging / saved_path
            if saved_path.resolve().parent != staging.resolve():
                raise ValueError("The upload returned a file outside its staging folder.")
        uploaded = {Path(name).name for name in saved}
        del saved
        expected = {"Notebook-inputs.zip"} if zip_mode else set(by_name)
        unexpected = set(uploaded) - expected
        if unexpected:
            raise ValueError(f"Wrong file(s): {', '.join(sorted(unexpected))}. Expected: {', '.join(sorted(expected))}.")
        if zip_mode:
            with ZipFile(staging / "Notebook-inputs.zip") as archive:
                planned = []
                seen = set()
                for item in archive.infolist():
                    path = PurePosixPath(item.filename)
                    if (path.is_absolute() or ".." in path.parts or ".git" in path.parts
                            or "\\" in item.filename or ":" in item.filename
                            or stat.S_ISLNK(item.external_attr >> 16)):
                        raise ValueError(f"Unsafe input ZIP member: {item.filename}")
                    if item.is_dir():
                        continue
                    relative = item.filename.removeprefix("code/")
                    allowed = any(PurePosixPath(relative).match(required)
                                  or (not PurePosixPath(required).suffix and relative.startswith(required + "/"))
                                  for required in missing)
                    target = (code_root / relative).resolve()
                    if not allowed or not target.is_relative_to(code_root) or target in seen:
                        raise ValueError(f"Input ZIP contains an unrequested or duplicate path: {item.filename}")
                    if target.exists():
                        raise ValueError(f"Input ZIP would overwrite an existing file: {relative}")
                    seen.add(target)
                    planned.append((item, target))
                for target in seen:
                    if any(parent in seen or parent.is_file() for parent in target.parents
                           if parent != code_root and parent.is_relative_to(code_root)):
                        raise ValueError(f"Input ZIP has a file/directory conflict: {target.relative_to(code_root)}")
                if archive.testzip() is not None:
                    raise ValueError("Damaged Notebook-inputs.zip; create or download it again.")
                for item, target in planned:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("xb") as destination:
                        shutil.copyfileobj(source, destination)
        else:
            for name in uploaded:
                target = (code_root / by_name[name]).resolve()
                if not target.is_relative_to(code_root) or target.exists():
                    raise ValueError(f"Cannot replace existing input: {by_name[name]}")
            for name in uploaded:
                target = code_root / by_name[name]
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(staging / name, target)
    print("Upload accepted; continuing the input check.", flush=True)


def prepare_notebook(
    notebook_name: str | None, code_root: Path, extra_dependencies: tuple[str, ...] = ()
) -> Path:
    code_root = Path(code_root).resolve(strict=True)
    if notebook_name is not None:
        missing = _missing_inputs(code_root, notebook_name)
        if missing and "google.colab" in sys.modules:
            try:
                _upload_inputs(code_root, notebook_name, missing)
            except Exception as error:
                print(f"Input upload failed: {error}", flush=True)
                raise
            missing = _missing_inputs(code_root, notebook_name)
        if missing:
            error = MissingNotebookInputs(
                "To rerun this notebook, supply these inputs under code/:\n"
                + "\n".join(f"  {path}" for path in missing)
                + "\nSources and commands: notebooks/DATA.md."
                + "\nSaved results can be read without running cells."
            )
            print(str(error), flush=True)
            raise error
    missing_packages = []
    for requirement in [*_requirements(code_root), *extra_dependencies]:
        if _name(requirement) == "pytest":
            continue
        try:
            metadata.version(_name(requirement))
        except metadata.PackageNotFoundError:
            missing_packages.append(requirement)
    if missing_packages:
        command = [sys.executable, "-m", "experiments.colab_runtime", "--install"]
        result = subprocess.run(command, cwd=code_root, capture_output=True, text=True)
        if result.returncode:
            log = code_root / "notebook-setup.log"
            log.write_text(result.stdout + result.stderr, encoding="utf-8")
            raise RuntimeError("Could not install Python packages; see code/notebook-setup.log.")
    os.chdir(code_root)
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))
    return code_root
