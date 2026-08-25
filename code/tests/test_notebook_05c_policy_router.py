from __future__ import annotations

from pathlib import Path

import nbformat

from experiments.build_notebook_05c import NOTEBOOK_PATH, build_notebook


def test_notebook_05c_is_artifact_only_registered_policy_router_reader() -> None:
    notebook = build_notebook()
    nbformat.validate(notebook)
    assert notebook.cells[0].cell_type == "code"
    assert notebook.cells[0].source.startswith("# Google Colab / local setup")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook_05c_cell_{index}", "exec")
    source = "\n".join(cell.source for cell in notebook.cells)
    lowered = source.lower()
    for phrase in (
        "causal full-information policy-router",
        "2021-2024 oof development",
        "real memory",
        "no memory",
        "shuffled policy-label memory",
        "deterministic hedge",
        "xgb_strong",
        "boundary",
        "choice_index",
        "deepseek-v4-flash:cloud",
        "5166728b9358990e5f6c34f87cbe48716be2f2cd2d3b98527dff27ea755bf3ba",
        "think=low",
        "registered lexicographic objective",
        "memory_benefit_established",
        "secondary reused forward",
        "2026-q2 lockbox",
        "retain immutable union v1",
    ):
        assert phrase in lowered
    assert "reconcile_final_experiment" in source
    assert "final_report_manifest.json" in source
    assert "table_artifacts" in source
    assert ".fit(" not in source
    assert "import ollama" not in lowered
    assert "from ollama" not in lowered
    assert "run_variant" not in source
    assert "build_development_opportunities" not in source
    assert "build_exact_opportunities" not in source


def test_notebook_05c_has_compact_reader_layout() -> None:
    notebook = build_notebook()
    table_cells = [
        cell
        for cell in notebook.cells
        if "result-table" in cell.get("metadata", {}).get("tags", [])
    ]
    figure_cells = [
        cell
        for cell in notebook.cells
        if "result-figure" in cell.get("metadata", {}).get("tags", [])
    ]
    conclusions = [
        cell
        for cell in notebook.cells
        if "result-conclusion" in cell.get("metadata", {}).get("tags", [])
    ]
    technical_cells = [
        cell
        for cell in notebook.cells
        if "technical-details" in cell.get("metadata", {}).get("tags", [])
    ]

    assert len(table_cells) == 4
    assert len(figure_cells) == 1
    assert len(conclusions) == 5
    assert all("\n" not in cell.source.strip() for cell in conclusions)
    assert len(technical_cells) == 1

    source = "\n".join(cell.source for cell in notebook.cells)
    assert "<details>" in source
    assert "SYSTEM_PROMPT_V4" in source
    assert "ROUTER_TASK_PROMPT" in source
    assert "core_comparison_table" in source
    assert "all_variants_table" in source
    assert "memory_controls_table" in source
    assert "audit_table" in source
    assert source.count("plt.show()") == 1


def test_notebook_05c_executed_copy_is_clean_and_registered() -> None:
    executed = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
    readme = (NOTEBOOK_PATH.parent / "README.md").read_text(encoding="utf-8")
    assert "05c_causal_policy_router_agent.ipynb" in readme


def test_notebook_05c_builder_targets_main_notebook_directory() -> None:
    assert NOTEBOOK_PATH == (
        Path(__file__).parents[1]
        / "notebooks"
        / "05c_causal_policy_router_agent.ipynb"
    )
