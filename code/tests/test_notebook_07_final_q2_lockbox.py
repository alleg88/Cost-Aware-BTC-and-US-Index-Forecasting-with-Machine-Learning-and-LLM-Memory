from __future__ import annotations

import ast
from pathlib import Path

import nbformat
import pytest

from experiments.build_notebook_07 import build_notebook


@pytest.mark.parametrize("saved", [False, True])
def test_artifact_paths_resolve_inside_the_result_directory_not_cwd(tmp_path, monkeypatch, saved):
    notebook = (nbformat.read(Path(__file__).parents[1] / "notebooks" / "22_Lockbox_Q2_2026.ipynb", 4)
                if saved else build_notebook())
    function = next(node for node in ast.walk(ast.parse(_code_source(notebook)))
                    if isinstance(node, ast.FunctionDef) and node.name == "resolve_bound_path")
    result_root = tmp_path / "run"
    result_root.mkdir()
    (result_root / "summary.json").write_text("registered", encoding="utf-8")
    (tmp_path / "summary.json").write_text("wrong working directory", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    namespace = {"Path": Path, "RESULT_ROOT": result_root, "CODE_ROOT": tmp_path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "reader", "exec"), namespace)
    resolve = namespace["resolve_bound_path"]
    assert resolve("summary.json") == result_root / "summary.json"
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve("../summary.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        resolve(str(tmp_path / "summary.json"))
    with pytest.raises(FileNotFoundError):
        resolve("missing.json")


def _code_source(notebook) -> str:
    return "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )


def _imports(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_notebook_07_is_an_artifact_only_reader() -> None:
    notebook = build_notebook()
    tree = ast.parse(_code_source(notebook))
    forbidden = {
        "data.load",
        "sentiment.index_scoring",
        "models.zoo",
        "features.index_sentiment",
        "requests",
        "httpx",
        "urllib",
        "socket",
        "ollama",
        "experiments.final_q2_lockbox_inputs",
        "experiments.final_q2_lockbox_runner",
    }

    assert forbidden.isdisjoint(_imports(tree))
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"fit", "predict", "predict_proba"}
        for node in ast.walk(tree)
    )


def test_first_reader_cell_validates_manifest_and_all_artifact_hashes_first() -> None:
    notebook = build_notebook()
    first = next(
        cell.source
        for cell in notebook.cells
        if "artifact-validation" in cell.metadata.get("tags", [])
    )

    assert "from run_zip import prepare_reader" in notebook.cells[0].source
    assert "manifest_sha256" in first
    assert "artifact_hashes" in first
    assert "resolve_bound_path" in first
    assert first.index("validate_manifest") < first.index("validate_all_artifacts")
    assert "read_parquet" not in first


def test_methodology_explains_every_frozen_ensemble_and_cost() -> None:
    notebook = build_notebook()
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert "opposite-signal veto" in markdown
    assert "arithmetic mean of all nine" in markdown
    assert all(
        model in markdown
        for model in (
            "Logistic Regression",
            "Decision Tree",
            "Random Forest",
            "Linear SVM",
            "XGBoost",
            "CatBoost",
            "MLP",
            "LSTM",
            "GRU",
        )
    )
    assert "reconstructed weights were frozen before Q2" in markdown
    assert "10 bps" in markdown and "2 bps" in markdown and "3 bps" in markdown
    assert "batch size 10" in markdown


def test_every_table_and_the_single_figure_have_intro_and_takeaway() -> None:
    notebook = build_notebook()
    tagged = [
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code"
        and set(cell.metadata.get("tags", [])).intersection(
            {"result-table", "result-figure"}
        )
    ]

    assert tagged
    assert sum(
        "result-figure" in notebook.cells[index].metadata.get("tags", [])
        for index in tagged
    ) == 1
    for index in tagged:
        assert index > 0 and notebook.cells[index - 1].cell_type == "markdown"
        assert notebook.cells[index - 1].source.startswith("Method: ")
        assert "display(Markdown(" in notebook.cells[index].source


def test_economic_tables_are_sorted_and_include_net_sharpe_sortino() -> None:
    notebook = build_notebook()
    table_source = "\n".join(
        cell.source
        for cell in notebook.cells
        if "result-table" in cell.metadata.get("tags", [])
    )

    assert "Net %" in table_source
    assert "Sharpe" in table_source
    assert "Sortino" in table_source
    assert "sort_values" in table_source
    assert "files" not in table_source.lower()
