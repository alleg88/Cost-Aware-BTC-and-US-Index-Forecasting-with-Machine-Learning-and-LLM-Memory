"""ZIP launch preserves inputs and rejects malformed packages before execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import sys
import zipfile

import pytest


def launcher():
    path = Path(__file__).parents[2] / "run_zip.py"
    assert path.is_file(), "The ZIP needs its own runnable setup helper"
    spec = importlib.util.spec_from_file_location("run_zip", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def archives(tmp_path, extra=None, changed=False):
    prefix = "cost-aware-market-forecasting/"
    contents = {"prepare_project.py": b"print('ready')", "code/pyproject.toml": b"[project]\n"}
    manifest = {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}
    code = tmp_path / "code.zip"
    with zipfile.ZipFile(code, "w") as archive:
        for name, data in contents.items():
            archive.writestr(prefix + name, b"changed" if changed else data)
        archive.writestr(prefix + "release_files.json", json.dumps(manifest))
        if extra:
            archive.writestr(*extra)
    data = tmp_path / "sources.zip"
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr(prefix + "code/.source_evidence/example.csv", "value\n1\n")
    return code, data


def test_extracts_both_archives_and_preserves_edits_on_repeat(tmp_path):
    code, data = archives(tmp_path)
    target = tmp_path / "run"
    root = launcher().extract_archives(code, data, target)
    assert (root / "code/.source_evidence/example.csv").read_text() == "value\n1\n"
    (root / "code/pyproject.toml").write_text("user edit")
    assert launcher().extract_archives(code, data, target) == root
    assert (root / "code/pyproject.toml").read_text() == "user edit"


def test_extracts_code_without_a_source_archive(tmp_path):
    code, _ = archives(tmp_path)
    target = tmp_path / "run"
    root = launcher().extract_archives(code, None, target)
    assert (root / "prepare_project.py").is_file()
    assert not (root / "code/.source_evidence").exists()
    assert launcher().extract_archives(code, None, target) == root


def test_sources_can_be_added_after_code_only_check_without_losing_edits(tmp_path):
    code, data = archives(tmp_path)
    module = launcher()
    target = tmp_path / "run"
    root = module.extract_archives(code, None, target)
    (root / "code/pyproject.toml").write_text("user edit")
    assert module.extract_archives(code, data, target) == root
    assert (root / "code/pyproject.toml").read_text() == "user edit"
    assert (root / "code/.source_evidence/example.csv").read_text() == "value\n1\n"
    assert module.extract_archives(code, data, target) == root


@pytest.mark.parametrize("add_sources", [False, True])
def test_interrupted_extraction_can_be_retried_without_partial_inputs(tmp_path, monkeypatch, add_sources):
    code, data = archives(tmp_path)
    module = launcher()
    target = tmp_path / "run"
    if add_sources:
        root = module.extract_archives(code, None, target)
        (root / "code/pyproject.toml").write_text("user edit")
        previous_marker = (target / ".archives.json").read_bytes()
    original = zipfile.ZipFile.extractall

    def interrupt(archive, path, *args, **kwargs):
        archive.extract(archive.namelist()[0], path)
        raise OSError("interrupted extraction")

    with monkeypatch.context() as changes:
        changes.setattr(zipfile.ZipFile, "extractall", interrupt)
        with pytest.raises(OSError, match="interrupted extraction"):
            module.extract_archives(code, data, target)
    assert zipfile.ZipFile.extractall is original
    if add_sources:
        assert (root / "code/pyproject.toml").read_text() == "user edit"
        assert (target / ".archives.json").read_bytes() == previous_marker
        assert not (root / "code/.source_evidence").exists()
    else:
        assert not target.exists()
    root = module.extract_archives(code, data, target)
    assert (root / "code/.source_evidence/example.csv").read_text() == "value\n1\n"


def test_source_addition_recovers_if_marker_update_is_interrupted(tmp_path, monkeypatch):
    code, data = archives(tmp_path)
    module = launcher()
    target = tmp_path / "run"
    root = module.extract_archives(code, None, target)
    (root / "code/pyproject.toml").write_text("user edit")
    original = Path.replace

    def interrupt(path, destination):
        if Path(destination) == target / ".archives.json":
            raise OSError("interrupted marker")
        return original(path, destination)

    with monkeypatch.context() as changes:
        changes.setattr(Path, "replace", interrupt)
        with pytest.raises(OSError, match="interrupted marker"):
            module.extract_archives(code, data, target)
    assert (root / "code/.source_evidence/example.csv").is_file()
    assert module.extract_archives(code, data, target) == root
    assert (root / "code/pyproject.toml").read_text() == "user edit"


def test_source_addition_does_not_accept_or_overwrite_conflicting_existing_data(tmp_path):
    code, data = archives(tmp_path)
    module = launcher()
    target = tmp_path / "run"
    root = module.extract_archives(code, None, target)
    source = root / "code/.source_evidence/example.csv"
    source.parent.mkdir()
    source.write_text("user data")
    with pytest.raises(ValueError):
        module.extract_archives(code, data, target)
    assert source.read_text() == "user data"
    assert json.loads((target / ".archives.json").read_text())["data"] is None


def test_native_launch_uses_current_python_for_setup_checks_and_rebuild(tmp_path, monkeypatch):
    module = launcher()
    code, data = archives(tmp_path)
    code.rename(tmp_path / "cost-aware-market-forecasting-code.zip")
    data.rename(tmp_path / "cost-aware-market-forecasting-sources.zip")
    commands = []
    monkeypatch.setattr(module, "_run", lambda command, cwd, title: commands.append(command))
    root = module.run("Rebuild results", native=True, archive_dir=tmp_path, destination=tmp_path / "run")
    assert commands and all(command[0] == sys.executable for command in commands)
    assert not (root / "code/.venv").exists()
    assert any("experiments.colab_runtime" in command and "--install" in command for command in commands)
    assert any("experiments.reproduce_tracked" in command for command in commands)
    assert any("experiments.reproduce_source" in command and "--audit-only" not in command for command in commands)
    assert any("experiments.reproduce_notebooks" in command for command in commands)


def test_code_only_run_checks_code_without_requesting_source_data(tmp_path, monkeypatch, capsys):
    module = launcher()
    code, _ = archives(tmp_path)
    code.rename(tmp_path / "cost-aware-market-forecasting-code.zip")
    commands = []
    # Dependency installation is external; keep archive validation/extraction real.
    monkeypatch.setattr(module, "_run", lambda command, cwd, title: commands.append(command))
    root = module.run(archive_dir=tmp_path, destination=tmp_path / "run")
    assert (root / "prepare_project.py").is_file()
    assert any("experiments.reproduce_tracked" in command for command in commands)
    assert any("experiments.reproduce_source" in command and "--audit-only" in command for command in commands)
    assert not any("experiments.source_evidence" in command for command in commands)
    assert not any("experiments.reproduce_notebooks" in command for command in commands)
    assert "source data" in capsys.readouterr().out.lower()


def test_rebuild_requires_sources_before_creating_or_installing(tmp_path):
    code, _ = archives(tmp_path)
    code.rename(tmp_path / "cost-aware-market-forecasting-code.zip")
    with pytest.raises(ValueError, match="Rebuild results.*sources"):
        launcher().run("Rebuild results", archive_dir=tmp_path, destination=tmp_path / "run")
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("path", ["../outside", "/absolute", "other/code/file", "cost-aware-market-forecasting/../escape", "cost-aware-market-forecasting/.git/config", "cost-aware-market-forecasting/C:drive", "cost-aware-market-forecasting/code\\escape"])
def test_rejects_unsafe_members_without_writing(tmp_path, path):
    code, data = archives(tmp_path, extra=(path, "bad"))
    with pytest.raises(ValueError):
        launcher().extract_archives(code, data, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_rejects_symlink_members(tmp_path):
    item = zipfile.ZipInfo("cost-aware-market-forecasting/code/link")
    item.create_system = 3
    item.external_attr = (stat.S_IFLNK | 0o777) << 16
    code, data = archives(tmp_path, extra=(item, "../../outside"))
    with pytest.raises(ValueError):
        launcher().extract_archives(code, data, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_rejects_changed_code_before_execution(tmp_path):
    code, data = archives(tmp_path, changed=True)
    with pytest.raises(ValueError):
        launcher().extract_archives(code, data, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_different_archive_does_not_overwrite_an_existing_run(tmp_path):
    code, data = archives(tmp_path)
    target = tmp_path / "run"
    root = launcher().extract_archives(code, data, target)
    with zipfile.ZipFile(data, "a") as archive:
        archive.writestr("cost-aware-market-forecasting/code/.source_evidence/new.csv", "new")
    with pytest.raises(ValueError):
        launcher().extract_archives(code, data, target)
    assert not (root / "code/.source_evidence/new.csv").exists()


def test_unknown_action_does_not_create_a_run(tmp_path):
    with pytest.raises(ValueError):
        launcher().run("unrecognised", archive_dir=tmp_path, destination=tmp_path / "run")
    assert not (tmp_path / "run").exists()


def test_failed_step_stops_the_launcher_and_retains_diagnostics(tmp_path, capsys):
    with pytest.raises(RuntimeError):
        launcher()._run([sys.executable, "-c", "print('failure detail'); raise SystemExit(3)"], tmp_path, "Check")
    assert "failure detail" in (tmp_path / "run.log").read_text()
    assert "failure detail" in capsys.readouterr().out


def test_successful_step_keeps_verbose_output_out_of_the_notebook(tmp_path, capsys):
    launcher()._run([sys.executable, "-c", "print('verbose detail')"], tmp_path, "Check")
    assert "verbose detail" not in capsys.readouterr().out
    assert "verbose detail" in (tmp_path / "run.log").read_text()
