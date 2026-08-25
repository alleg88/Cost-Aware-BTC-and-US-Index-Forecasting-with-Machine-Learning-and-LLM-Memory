from __future__ import annotations

from pathlib import Path

import nbformat
from notebook_assertions import assert_artifact_reader


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "04b_xgboost_strong_move_admission.ipynb"


def test_notebook_04b_is_executed_artifact_reader():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert_artifact_reader(notebook)
    source = "\n".join(cell.source for cell in notebook.cells).lower()
    assert "logisticregression(" not in source
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code and all(cell.execution_count is not None for cell in code)
    assert not any(
        output.output_type == "error"
        for cell in code
        for output in cell.get("outputs", [])
    )
