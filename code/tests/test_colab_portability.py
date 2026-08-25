import json
import subprocess
import sys
import types
from pathlib import Path
from zipfile import ZipFile

import pytest


NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def test_colab_launcher_is_first_by_name():
    assert sorted(NOTEBOOKS.glob("*.ipynb"))[0].name == "00_run_in_colab.ipynb"


@pytest.mark.parametrize("mode,existing,selected,expected_uploads", [
    ("Check installation", (), ("code",), 1),
    ("Check installation", ("code",), (), 0),
    ("Rebuild results", (), ("code", "sources"), 1),
    ("Rebuild results", ("code",), ("sources",), 1),
    ("Rebuild results", ("code", "sources"), (), 0),
])
def test_first_run_uploads_missing_zips_once_and_rerun_reuses_them(
    tmp_path, monkeypatch, mode, existing, selected, expected_uploads
):
    """A fresh runtime offers upload; a rerun must use the completed upload."""
    def write_zip(part):
        with ZipFile(tmp_path / f"cost-aware-market-forecasting-{part}.zip", "w") as archive:
            if part == "code":
                archive.writestr("cost-aware-market-forecasting/run_zip.py",
                                 "def run(action, *, archive_dir, native=False):\n"
                                 "    assert native, 'The notebook must use the native runtime'\n"
                                 "    result = archive_dir / 'selected_zip'\n"
                                 "    result.write_text(action)\n"
                                 "    return result\n")
    for part in existing:
        write_zip(part)
    upload_calls = []

    def upload(*, target_dir):
        assert Path(target_dir) == tmp_path
        upload_calls.append(target_dir)
        assert len(upload_calls) == 1
        for part in selected:
            write_zip(part)
        return {}

    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=upload)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    notebook = json.loads((NOTEBOOKS / "00_run_in_colab.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][2]["source"]).replace('Path("/content")', f"Path({str(tmp_path)!r})")
    for _ in range(2):
        scope = {"mode": mode}
        exec(compile(source, "Colab run cell", "exec"), scope)
        assert scope["project"].read_text() == mode
    assert len(upload_calls) == expected_uploads
    assert (tmp_path / "cost-aware-market-forecasting-code.zip").is_file()
    assert (tmp_path / "cost-aware-market-forecasting-sources.zip").is_file() == (mode == "Rebuild results")


@pytest.mark.parametrize("has_code_zip", [False, True])
def test_launcher_never_runs_a_stray_or_cached_helper(tmp_path, monkeypatch, has_code_zip):
    """A mistaken .py upload must not become the imported project launcher."""
    def stale_run(*args, **kwargs):
        raise AssertionError("Executed the stale helper instead of the ZIP")

    stale = types.ModuleType("run_zip")
    stale.run = stale_run
    monkeypatch.setitem(sys.modules, "run_zip", stale)
    monkeypatch.setattr(sys, "path", list(sys.path))
    colab = types.ModuleType("google.colab")
    upload_calls = []
    colab.files = types.SimpleNamespace(upload=lambda **kwargs: upload_calls.append(kwargs) or {})
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    (tmp_path / "run_zip.py").write_text("raise AssertionError('stray helper')")
    if has_code_zip:
        with ZipFile(tmp_path / "cost-aware-market-forecasting-code.zip", "w") as archive:
            archive.writestr("cost-aware-market-forecasting/run_zip.py",
                             "def run(action, *, archive_dir, native=False):\n"
                             "    assert native, 'The notebook must use the native runtime'\n"
                             "    result = archive_dir / 'selected_zip'\n"
                             "    result.write_text(action)\n"
                             "    return result\n")
    notebook = json.loads((NOTEBOOKS / "00_run_in_colab.ipynb").read_text("utf-8"))
    source = "".join(notebook["cells"][2]["source"]).replace('Path("/content")', f"Path({str(tmp_path)!r})")
    scope = {"mode": "Check installation"}
    if has_code_zip:
        for _ in range(2):
            scope = {"mode": "Check installation"}
            exec(compile(source, "Colab run cell", "exec"), scope)
            assert scope["project"].read_text() == "Check installation"
    else:
        with pytest.raises(ValueError, match="cost-aware-market-forecasting-code.zip"):
            exec(compile(source, "Colab upload cell", "exec"), scope)
        assert not (tmp_path / "selected_zip").exists()
    assert len(upload_calls) == (0 if has_code_zip else 1)


@pytest.mark.parametrize(
    "notebook",
    sorted(p for p in NOTEBOOKS.glob("*.ipynb") if p.name != "00_run_in_colab.ipynb"),
    ids=lambda path: path.name,
)
def test_first_cell_sets_up_the_project_in_colab(notebook, tmp_path, monkeypatch):
    """Catch Windows paths or dependency resolution inside a live Colab kernel."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    source = "".join(nb["cells"][0]["source"])
    assert r"D:\MSC project" not in "".join(
        "".join(cell.get("source", [])) for cell in nb["cells"]
    )

    code_root = tmp_path / "code"
    code_root.mkdir()
    source = source.replace(
        "/content/drive/MyDrive/msc project/code", code_root.as_posix()
    )

    mounted = []
    google = types.ModuleType("google")
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda path, **_: mounted.append(path))
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(subprocess, "check_call", lambda command: installed.append(command))
    installed = []

    old_cwd = Path.cwd()
    old_path = sys.path.copy()
    namespace = {}
    try:
        exec(compile(source, str(notebook), "exec"), namespace)
        assert mounted == ["/content/drive"]
        assert Path(namespace["CODE_ROOT"]) == code_root
        assert Path.cwd() == code_root
        assert sys.path[0] == str(code_root)
        assert len(installed) == 1
        command = installed[0]
        assert command[:8] == [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "--no-deps",
            "-e",
            str(code_root),
        ]
        dependencies = command[8:]
        assert len(dependencies) == len(set(dependencies))
        assert {"catboost==1.2.10", "rapidfuzz==3.14.3"}.issubset(dependencies)
    finally:
        monkeypatch.chdir(old_cwd)
        sys.path[:] = old_path
