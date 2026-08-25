from __future__ import annotations

from pathlib import Path

import nbformat
from notebook_assertions import assert_artifact_reader, assert_later_period_guards, numeric_column, tables


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "04h_union_v1_episode_reentry.ipynb"


def _text_outputs(notebook: nbformat.NotebookNode) -> str:
    values: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                values.append(str(output.text))
            elif output.output_type in {"display_data", "execute_result"}:
                values.append(str(output.get("data", {}).get("text/plain", "")))
    return "\n".join(values).lower()


def test_notebook_04h_is_executed_artifact_only_reader() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    assert_artifact_reader(notebook)
    source = "\n".join(cell.source for cell in notebook.cells).lower()
    code = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    ).lower()
    for forbidden in (
        ".fit(",
        "torch.optim",
        "linearsvc(",
        "load_bounded_sources",
        "fit_union_reentry_fold",
        "run_union_v1_episode_reentry",
    ):
        assert forbidden not in code
    assert "artifact_hashes" in code

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


def test_notebook_04h_reports_frequency_economics_and_excluded_stages():
    notebook = nbformat.read(NOTEBOOK, 4)
    assert_later_period_guards(notebook)
    rendered = tables(notebook)
    funnel = next(table for table in rendered if "trades_per_day" in table[0])
    assert numeric_column(funnel, "trades") == [948, 1231]
    assert numeric_column(funnel, "trades_per_day") == [3.251, 4.221]
    economics = next(table for table in rendered if "net_pct" in table[0])
    assert set(numeric_column(economics, "net_pct")) == {-97.69, -20.58, -118.27}
