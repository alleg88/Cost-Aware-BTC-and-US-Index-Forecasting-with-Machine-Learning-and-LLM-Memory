from pathlib import Path

import nbformat

from experiments.build_notebook_05c import NOTEBOOK_PATH, build_notebook


def test_notebook_18_builds_the_nine_weight_memory_comparison():
    notebook = build_notebook()
    nbformat.validate(notebook)
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"notebook18_cell_{index}", "exec")
    source = "\n".join(cell.source for cell in notebook.cells)
    for name in ("RealMemory", "NoMemory", "ShuffledMemory", "Hedge", "LSTM_fixed"):
        assert name in source
    for phrase in ("nine forecasting models", "four completed seven-day",
                   "permutes model identities", "Qualified Union", "Q2-2026 rows are excluded"):
        assert phrase in source
    assert "verify_reader(CACHE)" in source
    assert "reflection_ensemble_v5" in source
    assert "reflection_policy_router_v4" not in source
    assert "choice_index" not in source
    assert "import ollama" not in source
    assert ".fit(" not in source


def test_notebook_18_uses_individual_drive_inputs_and_saved_experimental_calls():
    notebook = build_notebook()
    setup = next(cell.source for cell in notebook.cells if cell.cell_type == "code")
    assert 'drive.usercontent.google.com/download' in setup
    assert 'manifest_sha256' in setup and 'loader_sha256' in setup
    source = "\n".join(cell.source for cell in notebook.cells)
    assert "llm_call_audit.parquet" in source
    assert "familywise95_low" in source
    assert "weight_decisions.parquet" in source
    assert "summary.parquet" in source


def test_notebook_18_executed_copy_has_real_outputs_and_new_protocol():
    executed = nbformat.read(NOTEBOOK_PATH, as_version=4)
    code_cells = [cell for cell in executed.cells if cell.cell_type == "code"]
    assert all(cell.execution_count is not None for cell in code_cells)
    assert len({cell.execution_count for cell in code_cells}) == len(code_cells)
    assert not any(output.output_type == "error" for cell in code_cells for output in cell.outputs)
    assert "reflection_ensemble_v5" in "\n".join(cell.source for cell in code_cells)
    assert "manifest_id" in code_cells[0].source
    assert NOTEBOOK_PATH.name in (NOTEBOOK_PATH.parent / "README.md").read_text(encoding="utf-8")


def test_notebook_18_preserves_the_existing_artifact_path():
    assert NOTEBOOK_PATH == Path(__file__).parents[1] / "notebooks/18_RQ4_A_BTC_LLM_policy_router.ipynb"
