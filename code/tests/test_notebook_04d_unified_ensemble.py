from __future__ import annotations

from pathlib import Path

import nbformat
from notebook_assertions import assert_artifact_reader, assert_later_period_guards


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "04d_unified_2021_ensemble.ipynb"


def test_notebook_04d_is_executed_artifact_reader():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert_artifact_reader(notebook)
    source = "\n".join(cell.source for cell in notebook.cells).lower()
    for forbidden in (
        "xgbclassifier(",
        "linearsvc(",
        "torch.optim",
        ".fit(",
        "load_bounded_sources",
        "replay_brackets(",
        "evaluate_oof_policy_grid(",
        "run_unified_2021_ensemble",
    ):
        assert forbidden not in source

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert "google.colab" in code_cells[0].source
    assert "drive.mount" in code_cells[0].source
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )

    assert 'assert summary["decision"] == "development_fail_keep_union_v1"' in source
    assert 'assert sha256(union / filename) == expected' in source
    assert 'assert summary["development"]["qualifying_policies"] == 0' in source


def test_notebook_04d_proves_later_stages_and_lockbox_were_not_opened():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert_artifact_reader(notebook)
    source = "\n".join(cell.source for cell in notebook.cells).lower()
    assert "h1 was not loaded" in source
    assert "forward was not loaded" in source
    assert 'assert summary["max_loaded_timestamp"] < "2025-01-01"' in source
    assert_later_period_guards(notebook)

