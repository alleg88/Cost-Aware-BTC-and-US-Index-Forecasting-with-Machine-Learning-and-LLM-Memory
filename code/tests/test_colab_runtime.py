"""Native setup must preserve the host stack and check actual imports."""
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_native_audit_checks_imports_instead_of_enforcing_the_local_lock(monkeypatch):
    import sys
    from experiments import reproduce_tracked

    monkeypatch.setattr(reproduce_tracked, "is_colab", lambda: True, raising=False)
    assert reproduce_tracked._dependency_check_command() == [
        sys.executable, "-m", "experiments.colab_runtime"
    ]


@pytest.mark.parametrize("catboost_present", [False, True])
def test_install_only_fetches_missing_packages_and_constrains_the_host(tmp_path, monkeypatch, catboost_present):
    from experiments import colab_runtime as runtime

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "example"\ndependencies = ["numpy==2.4.6", "catboost==1.2.10"]\n'
        '[project.optional-dependencies]\nextras = ["ipykernel==7.3.0"]\n'
    )
    present = {"numpy": "2.1.3", "ipykernel": "6.17.1", "google-colab": "1.0.0"}
    if catboost_present:
        present["catboost"] = "1.2.10"
    monkeypatch.setattr(runtime.metadata, "distributions", lambda: [
        SimpleNamespace(metadata={"Name": name}, version=version) for name, version in present.items()
    ])
    calls = []

    def pip(command, **kwargs):
        if "--constraint" in command:
            constraints = Path(command[command.index("--constraint") + 1]).read_text().splitlines()
            assert set(constraints) == {f"{name}=={version}" for name, version in present.items()}
        calls.append(command)

    monkeypatch.setattr(runtime.subprocess, "check_call", pip)
    runtime.install(tmp_path)
    fetch = [command for command in calls if "--constraint" in command]
    assert len(fetch) == (0 if catboost_present else 1)
    if fetch:
        assert fetch[0][-1] == "catboost==1.2.10"
        assert "numpy==2.4.6" not in fetch[0]
        assert "ipykernel==7.3.0" not in fetch[0]
    assert any("--no-deps" in command and "-e" in command and str(tmp_path) in command for command in calls)


def test_runtime_check_fails_on_an_unusable_import(tmp_path, monkeypatch):
    from experiments import colab_runtime as runtime

    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["numpy==2.4.6"]\n')
    monkeypatch.setattr(runtime.metadata, "version", lambda name: "2.1.3")

    def broken_import(name):
        raise ImportError("binary incompatibility")

    monkeypatch.setattr(runtime.importlib, "import_module", broken_import)
    with pytest.raises(ImportError, match="binary incompatibility"):
        runtime.check(tmp_path)
