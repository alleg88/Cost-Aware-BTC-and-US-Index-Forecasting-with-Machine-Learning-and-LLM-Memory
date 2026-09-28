"""A partial check executes only the first reader and retains honest evidence."""
import importlib
import json
import nbformat
from nbclient.exceptions import CellExecutionError
import pytest


def partial_project(tmp_path, source):
    code = tmp_path / "code"
    (code / "notebooks").mkdir(parents=True)
    (code / "data").mkdir()
    for name in ("btcusdt_m15_2024_2025.parquet", "btcusdt_positioning_m15_2024_2026.parquet"):
        (code / "data" / name).write_bytes(b"fixture input")
    notebook = nbformat.v4.new_notebook(cells=[
        nbformat.v4.new_code_cell(source, execution_count=99, outputs=[
            nbformat.v4.new_output("stream", name="stdout", text="OLD SAVED OUTPUT\n"),
        ]),
        nbformat.v4.new_code_cell("print('SECOND CELL')"),
    ])
    path = code / "notebooks/01_RQ1_A_BTC_data_labels_baseline.ipynb"
    nbformat.write(notebook, path)
    (code / "notebooks/02_other.ipynb").write_text("must never be opened")
    return code, path


def test_partial_check_executes_first_reader_without_replacing_saved_reference(tmp_path):
    code, original = partial_project(tmp_path, "print('FRESH RESULT', 2 + 2)")
    before = original.read_bytes()
    module = importlib.import_module("experiments.reproduce_partial")
    report = module.reproduce_partial(code, timeout=60)
    assert report["status"] == "PASSED"
    assert report["full_rebuild_executed"] is False
    assert report["numerical_equivalence_checked"] is False
    assert report["executed_code_cells"] == 2
    assert original.read_bytes() == before
    output = tmp_path / "partial-check/01_RQ1_A_BTC_data_labels_baseline.ipynb"
    executed = nbformat.read(output, as_version=4)
    assert "FRESH RESULT 4" in executed.cells[0].outputs[0].text
    assert "SECOND CELL" in executed.cells[1].outputs[0].text
    assert "OLD SAVED OUTPUT" not in output.read_text(encoding="utf-8")
    assert json.loads((output.parent / "report.json").read_text())["status"] == "PASSED"


def test_partial_failure_preserves_error_and_clears_unexecuted_saved_outputs(tmp_path):
    code, original = partial_project(tmp_path, "raise ValueError('PARTIAL TEST FAILURE')")
    before = original.read_bytes()
    module = importlib.import_module("experiments.reproduce_partial")
    with pytest.raises(CellExecutionError, match="PARTIAL TEST FAILURE"):
        module.reproduce_partial(code, timeout=60)
    output = tmp_path / "partial-check/01_RQ1_A_BTC_data_labels_baseline.ipynb"
    executed = nbformat.read(output, as_version=4)
    assert executed.cells[0].outputs[-1].output_type == "error"
    assert executed.cells[1].execution_count is None
    assert "OLD SAVED OUTPUT" not in output.read_text(encoding="utf-8")
    assert original.read_bytes() == before
    report = json.loads((output.parent / "report.json").read_text())
    assert report["status"] == "FAILED"
    assert "PARTIAL TEST FAILURE" in report["error"]


def test_failed_preflight_cannot_leave_a_previous_pass_report(tmp_path):
    code, _ = partial_project(tmp_path, "print('unused')")
    output = tmp_path / "partial-check"
    output.mkdir()
    (output / "report.json").write_text('{"status": "PASSED"}')
    previous = output / "01_RQ1_A_BTC_data_labels_baseline.ipynb"
    previous.write_bytes(b"previous completed result")
    (code / "data/btcusdt_positioning_m15_2024_2026.parquet").unlink()
    module = importlib.import_module("experiments.reproduce_partial")
    with pytest.raises(FileNotFoundError, match="btcusdt_positioning_m15_2024_2026.parquet"):
        module.reproduce_partial(code)
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "FAILED"
    assert report["executed_code_cells"] == 0
    assert report["output_notebook"] is None
    assert previous.read_bytes() == b"previous completed result"
