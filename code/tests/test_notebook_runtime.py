import importlib
import io
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from zipfile import ZipFile

import pytest


def runtime():
    return importlib.import_module("experiments.notebook_runtime")


def project(tmp_path):
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = []\n')
    return tmp_path


def test_missing_inputs_are_named_before_install_or_compute(tmp_path, monkeypatch, capsys):
    module = runtime()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("Unexpected subprocess"))
    with pytest.raises(FileNotFoundError) as error:
        module.prepare_notebook("01_RQ1_A_BTC_data_labels_baseline.ipynb", project(tmp_path))
    message = str(error.value)
    assert "data/btcusdt_m15_2024_2025.parquet" in message
    assert "data/btcusdt_positioning_m15_2024_2026.parquet" in message
    assert "DATA.md" in message
    assert "Traceback" not in "\n".join(error.value._render_traceback_())
    # A plain-text message must survive frontends that summarise exceptions as "Execution failed".
    assert "data/btcusdt_m15_2024_2025.parquet" in capsys.readouterr().out
    assert not (tmp_path / ".git").exists()


def test_supplied_inputs_allow_silent_setup_in_the_selected_folder(tmp_path, monkeypatch, capsys):
    module = runtime()
    root = project(tmp_path)
    (root / "data").mkdir()
    for name in ("btcusdt_m15_2024_2025.parquet", "btcusdt_positioning_m15_2024_2026.parquet"):
        (root / "data" / name).write_bytes(b"fixture")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No package is missing"))
    previous = sys.path.copy()
    try:
        result = module.prepare_notebook("01_RQ1_A_BTC_data_labels_baseline.ipynb", root)
        assert result == root
        assert Path.cwd() == root and str(root) in sys.path
        assert capsys.readouterr().out == ""
    finally:
        sys.path[:] = previous


def test_empty_result_directory_is_not_a_supplied_result(tmp_path):
    root = project(tmp_path)
    (root / "experiments/cache/qualified_union_v1").mkdir(parents=True)
    with pytest.raises(FileNotFoundError, match="qualified_union_v1"):
        runtime().prepare_notebook("08_RQ2_C_BTC_qualified_union_ensemble.ipynb", root)


def test_unknown_notebook_cannot_skip_the_input_check(tmp_path):
    with pytest.raises(ValueError, match="Unknown notebook"):
        runtime().prepare_notebook("unknown.ipynb", project(tmp_path))


def test_saved_q2_reader_does_not_require_a_rebuild_receipt(tmp_path, monkeypatch):
    root = project(tmp_path)
    result = root / "experiments/cache/final_q2_lockbox/8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312"
    result.mkdir(parents=True)
    (result / "COMPLETE.json").write_text("{}")
    (result.parent / "OPENED.json").write_text("{}")
    monkeypatch.chdir(root)
    previous = sys.path.copy()
    try:
        assert runtime().prepare_notebook("22_Lockbox_Q2_2026.ipynb", root) == root
    finally:
        sys.path[:] = previous


def test_index_sentiment_reader_also_requests_its_score_manifests(tmp_path):
    root = project(tmp_path)
    for stream in ("usa500", "usatech"):
        directory = root / "experiments/cache/index_replication" / stream
        directory.mkdir(parents=True)
        (directory / "result.json").write_text("{}")
    with pytest.raises(FileNotFoundError) as error:
        runtime().prepare_notebook("15_RQ3_D_indices_DeBERTa_sentiment.ipynb", root)
    assert "sentiment/raw/scores_usa500.manifest.json" in str(error.value)
    assert "index_all_model_forward/usatech" in str(error.value)


def test_removed_channel_reader_is_not_a_launch_target(tmp_path):
    with pytest.raises(ValueError, match="Unknown notebook"):
        runtime().prepare_notebook("23_RQ5_A_BTC_channel_strategy.ipynb", project(tmp_path))


def colab_upload(monkeypatch, response, before_upload=lambda: None):
    """Replace only the external browser upload; filesystem checks stay real."""
    def upload(*, target_dir):
        before_upload()
        for name, value in response.items():
            (Path(target_dir) / name).write_bytes(value)
        # Colab returns the full saved path when target_dir is supplied.
        return {str(Path(target_dir) / name): value for name, value in response.items()}

    colab = ModuleType("google.colab")
    colab.files = SimpleNamespace(upload=upload)
    monkeypatch.setitem(sys.modules, "google.colab", colab)


def test_colab_names_both_input_files_before_upload_then_continues(tmp_path, monkeypatch, capsys):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    filenames = {
        "btcusdt_m15_2024_2025.parquet": b"market input",
        "btcusdt_positioning_m15_2024_2026.parquet": b"positioning input",
    }

    def prompt_is_visible():
        prompt = capsys.readouterr().out
        assert all(name in prompt for name in filenames)
        assert "Choose files" in prompt
        assert "01_RQ1_A_BTC_data_labels_baseline.ipynb" in prompt

    colab_upload(monkeypatch, filenames, prompt_is_visible)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No package is missing"))
    previous = sys.path.copy()
    try:
        assert runtime().prepare_notebook("01_RQ1_A_BTC_data_labels_baseline.ipynb", root) == root
        assert all((root / "data" / name).read_bytes() == value for name, value in filenames.items())
        assert "continuing" in capsys.readouterr().out.lower()
    finally:
        sys.path[:] = previous


def test_cancelled_upload_reports_missing_file_and_keeps_existing_data(tmp_path, monkeypatch, capsys):
    root = project(tmp_path)
    (root / "data").mkdir()
    existing = root / "data/btcusdt_m15_2024_2025.parquet"
    existing.write_bytes(b"user data")
    colab_upload(monkeypatch, {})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("No installation after cancellation"))
    with pytest.raises(FileNotFoundError, match="btcusdt_positioning_m15_2024_2026.parquet"):
        runtime().prepare_notebook("01_RQ1_A_BTC_data_labels_baseline.ipynb", root)
    assert existing.read_bytes() == b"user data"
    assert "No file was uploaded" in capsys.readouterr().out


def test_wrong_upload_is_rejected_without_changing_user_files(tmp_path, monkeypatch):
    root = project(tmp_path)
    original = (root / "pyproject.toml").read_bytes()
    colab_upload(monkeypatch, {"pyproject.toml": b"not an input"})
    with pytest.raises(ValueError, match="pyproject.toml"):
        runtime().prepare_notebook("01_RQ1_A_BTC_data_labels_baseline.ipynb", root)
    assert (root / "pyproject.toml").read_bytes() == original
    assert not (root / "data").exists()


def input_zip(entries):
    stream = io.BytesIO()
    with ZipFile(stream, "w") as archive:
        for name, value in entries.items():
            archive.writestr(name, value)
    return stream.getvalue()


def test_result_folder_has_one_named_zip_upload_and_uses_code_relative_paths(tmp_path, monkeypatch, capsys):
    root = project(tmp_path)
    monkeypatch.chdir(root)
    relative = "experiments/cache/qualified_union_v1/summary.csv"

    def prompt_is_visible():
        prompt = capsys.readouterr().out
        assert "Notebook-inputs.zip" in prompt
        assert "experiments/cache/qualified_union_v1" in prompt

    colab_upload(monkeypatch, {"Notebook-inputs.zip": input_zip({relative: "value\n1\n"})}, prompt_is_visible)
    previous = sys.path.copy()
    try:
        assert runtime().prepare_notebook("08_RQ2_C_BTC_qualified_union_ensemble.ipynb", root) == root
        assert (root / relative).read_text() == "value\n1\n"
    finally:
        sys.path[:] = previous


@pytest.mark.parametrize("unsafe", ["../outside.txt", "/outside.txt", "code/pyproject.toml"])
def test_input_zip_cannot_write_outside_requested_inputs(tmp_path, monkeypatch, unsafe):
    root = project(tmp_path)
    original = (root / "pyproject.toml").read_bytes()
    colab_upload(monkeypatch, {"Notebook-inputs.zip": input_zip({unsafe: "bad"})})
    with pytest.raises(ValueError) as error:
        runtime().prepare_notebook("08_RQ2_C_BTC_qualified_union_ensemble.ipynb", root)
    assert unsafe in str(error.value)
    assert (root / "pyproject.toml").read_bytes() == original
    assert not (root / "experiments").exists()


def test_input_zip_file_directory_conflict_is_rejected_before_any_write(tmp_path, monkeypatch):
    root = project(tmp_path)
    prefix = "experiments/cache/qualified_union_v1/"
    colab_upload(monkeypatch, {"Notebook-inputs.zip": input_zip({prefix + "x": "one", prefix + "x/y": "two"})})
    with pytest.raises(ValueError, match="conflict"):
        runtime().prepare_notebook("08_RQ2_C_BTC_qualified_union_ensemble.ipynb", root)
    assert not (root / "experiments").exists()
