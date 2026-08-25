from __future__ import annotations

from pathlib import Path

import nbformat
import pytest
from notebook_assertions import assert_artifact_reader, assert_later_period_guards, numeric_column, tables


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "04g_lstm_gmadl_shadow.ipynb"


def test_notebook_04g_is_executed_artifact_only_paired_shadow_reader():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert_artifact_reader(notebook)
    source = "\n".join(cell.source for cell in notebook.cells).lower()
    for forbidden in (
        "xgbregressor(",
        "linearsvr(",
        "torch.optim",
        ".fit(",
        "load_bounded_sources",
        "fit_paired_lstm_shadow",
        "run_lstm_gmadl_shadow",
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

    assert 'assert summary["decision"] == "shadow_reject_keep_standard_lstm_and_union_v1"' in source
    assert 'assert summary["maximum_loaded_timestamp"] < "2025-01-01"' in source
    accuracy = next(table for table in tables(notebook) if "value_weighted_accuracy_pct" in table[0])
    assert numeric_column(accuracy, "value_weighted_accuracy_pct") == pytest.approx(
        [51.0937, 50.4149], abs=0.00005
    )


def test_notebook_04g_proves_pairing_admission_and_excluded_stages():
    notebook = nbformat.read(NOTEBOOK, 4)
    assert_later_period_guards(notebook)
    rendered = tables(notebook)
    assert numeric_column(rendered[0], "control_max_abs_delta_from_04f") == [0] * 5
    economics = next(table for table in rendered if "net_pct" in table[0])
    assert numeric_column(economics, "trades") == [147, 163]
    assert numeric_column(economics, "net_pct") == pytest.approx(
        [-12.8283, -32.4585], abs=0.00005
    )

