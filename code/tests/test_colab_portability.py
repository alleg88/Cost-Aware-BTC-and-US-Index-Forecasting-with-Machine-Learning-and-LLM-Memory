import json
import hashlib
import subprocess
import sys
import types
from pathlib import Path
from zipfile import ZipFile

import pytest


NOTEBOOKS = Path(__file__).parents[1] / "notebooks"


def test_first_notebook_is_the_first_research_experiment():
    from experiments.notebook_hygiene import execution_order

    assert execution_order()[0] == "01_RQ1_A_BTC_data_labels_baseline.ipynb"
    assert (NOTEBOOKS / execution_order()[0]).is_file()


def write_release(path):
    with ZipFile(path, "w") as archive:
        archive.writestr("cost-aware-market-forecasting/run_zip.py",
                         "def prepare_rebuild(notebook_name, source, extra_dependencies=()):\n"
                         "    result = source.parent / 'selected_notebook'\n"
                         "    result.write_text(notebook_name)\n"
                         "    return result\n")


@pytest.mark.parametrize("native,existing", [(False, True), (True, False), (True, True)])
def test_rebuild_notebook_detects_neighbor_or_uploads_once_then_reuses(tmp_path, monkeypatch, native, existing):
    """No selector or state from a previous cell is needed to choose the right ZIP."""
    monkeypatch.chdir(tmp_path)
    archive_name = "Release-Rebuild.zip"
    if existing:
        write_release(tmp_path / archive_name)
    calls = []

    def upload(*, target_dir):
        assert Path(target_dir) == tmp_path
        calls.append(target_dir)
        write_release(tmp_path / archive_name)
        return {}

    if native:
        colab = types.ModuleType("google.colab")
        colab.files = types.SimpleNamespace(upload=upload)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
    else:
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    from experiments.notebook_hygiene import canonical_colab_setup
    source = canonical_colab_setup(notebook_name="02_RQ1_B_BTC_positioning_ablation.ipynb", rebuild=True)
    old_path = sys.path.copy()
    try:
        for _ in range(2):
            scope = {}
            exec(compile(source, "Rebuild notebook", "exec"), scope)
            assert scope["CODE_ROOT"].read_text() == "02_RQ1_B_BTC_positioning_ablation.ipynb"
    finally:
        sys.path[:] = old_path
    assert len(calls) == (1 if native and not existing else 0)


@pytest.mark.parametrize("has_zip", [False, True])
def test_launcher_never_runs_a_stray_or_cached_helper(tmp_path, monkeypatch, has_zip):
    """A mistaken .py upload must not become the imported project launcher."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    stale = types.ModuleType("run_zip")
    stale.prepare_rebuild = lambda *args, **kwargs: pytest.fail("Executed a cached helper")
    monkeypatch.setitem(sys.modules, "run_zip", stale)
    (tmp_path / "run_zip.py").write_text("raise AssertionError('stray helper')")
    if has_zip:
        write_release(tmp_path / "Release-Rebuild.zip")
    from experiments.notebook_hygiene import canonical_colab_setup
    source = canonical_colab_setup(notebook_name="03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb", rebuild=True)
    old_path = sys.path.copy()
    try:
        if has_zip:
            scope = {}
            exec(compile(source, "Start cell", "exec"), scope)
            assert scope["CODE_ROOT"].read_text() == "03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb"
        else:
            with pytest.raises(FileNotFoundError, match="Release-Rebuild.zip"):
                exec(compile(source, "Start cell", "exec"), {})
            assert not (tmp_path / "selected_notebook").exists()
    finally:
        sys.path[:] = old_path


def test_rebuild_notebook_in_subfolder_finds_its_neighbor_distribution(tmp_path, monkeypatch):
    from experiments.notebook_hygiene import canonical_colab_setup

    notebooks = tmp_path / "notebooks"
    notebooks.mkdir()
    write_release(tmp_path / "Release-Rebuild.zip")
    monkeypatch.chdir(notebooks)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    old_path = sys.path.copy()
    try:
        scope = {}
        exec(canonical_colab_setup(notebook_name="02_RQ1_B_BTC_positioning_ablation.ipynb", rebuild=True), scope)
        assert scope["CODE_ROOT"].read_text() == "02_RQ1_B_BTC_positioning_ablation.ipynb"
    finally:
        sys.path[:] = old_path


@pytest.mark.parametrize(
    "notebook",
    sorted(p for p in NOTEBOOKS.glob("*.ipynb")
           if p.name not in {"00_run_in_colab.ipynb", "18_RQ4_A_BTC_LLM_policy_router.ipynb"}),
    ids=lambda path: path.name,
)
@pytest.mark.parametrize("native", [False, True], ids=["local", "colab"])
def test_first_cell_reports_missing_inputs_without_drive_or_installing(
    notebook, tmp_path, monkeypatch, native
):
    """Run the delivered cell against a clean project, not the author's Drive."""
    nb = json.loads(notebook.read_text(encoding="utf-8"))
    source = "".join(nb["cells"][0]["source"])
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "pyproject.toml").write_text('[project]\ndependencies = []\n')
    (tmp_path / "run_zip.py").write_bytes((NOTEBOOKS.parents[1] / "run_zip.py").read_bytes())
    monkeypatch.chdir(code_root)
    if native:
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=lambda *a, **k: pytest.fail("Mounted Drive"))
        colab.files = types.SimpleNamespace(upload=lambda **kw: {})
        monkeypatch.setitem(sys.modules, "google.colab", colab)
        source = source.replace('Path("/content")', f"Path({str(code_root)!r})")
    else:
        monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: pytest.fail("Installed before checking data"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Started a subprocess without data"))
    old_path = sys.path.copy()
    try:
        with pytest.raises(FileNotFoundError) as error:
            exec(compile(source, str(notebook), "exec"), {})
        assert "DATA.md" in str(error.value)
        assert "rerun" in str(error.value).lower()
    finally:
        sys.path[:] = old_path


def test_llm_notebook_uses_local_project_without_downloading(tmp_path, monkeypatch):
    notebook = json.loads(
        (NOTEBOOKS / "18_RQ4_A_BTC_LLM_policy_router.ipynb").read_text(encoding="utf-8")
    )
    setup = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    code_root = tmp_path / "code"
    code_root.mkdir()
    (code_root / "pyproject.toml").write_text('[project]\ndependencies = []\n')
    monkeypatch.chdir(code_root)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: pytest.fail("Downloaded local inputs"))
    old_path = sys.path.copy()
    try:
        scope = {}
        exec(compile("".join(setup["source"]), "LLM notebook setup", "exec"), scope)
        assert scope["CODE_ROOT"] == code_root
    finally:
        sys.path[:] = old_path


@pytest.mark.parametrize("native", [False, True], ids=["local", "colab"])
def test_compact_first_cell_loads_the_real_neighbor_zip_in_an_isolated_python(tmp_path, native):
    """A single notebook must bootstrap without the author's packages or a prior cell."""
    from experiments.notebook_hygiene import canonical_colab_setup

    root = NOTEBOOKS.parents[1]
    payload = {"run_zip.py": (root / "run_zip.py").read_bytes(),
               "code/pyproject.toml": b'[project]\ndependencies = []\n'}
    payload.update({"code/experiments/" + path.name: path.read_bytes()
                    for path in (root / "code/experiments").glob("*.py")})
    manifest = {name: hashlib.sha256(value).hexdigest() for name, value in payload.items()}
    with ZipFile(tmp_path / "Release-Client.zip", "w") as archive:
        for name, value in payload.items():
            archive.writestr("cost-aware-market-forecasting/" + name, value)
        archive.writestr("cost-aware-market-forecasting/release_files.json", json.dumps(manifest))
    source = canonical_colab_setup(notebook_name="01_RQ1_A_BTC_data_labels_baseline.ipynb")
    script = f'''
import pathlib, subprocess, sys, types
if {native!r}:
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda **kw: {{}})
    sys.modules["google.colab"] = colab
subprocess.run = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Installed before data"))
for attempt in range(2):
    try:
        exec({source!r}, {{}})
    except FileNotFoundError as error:
        assert type(error).__name__ == "MissingNotebookInputs", str(error)
        assert "data/btcusdt_m15_2024_2025.parquet" in str(error)
    else:
        raise AssertionError("Did not request missing data")
import experiments.notebook_runtime as runtime
expected = pathlib.Path.cwd() / "Release-Client/cost-aware-market-forecasting/code"
assert pathlib.Path(runtime.__file__).is_relative_to(expected), runtime.__file__
assert not (expected.parent / ".git").exists()
print("ISOLATED_ZIP_INPUT_CHECK_OK")
'''
    result = subprocess.run([sys.executable, "-I", "-S", "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ISOLATED_ZIP_INPUT_CHECK_OK" in result.stdout
