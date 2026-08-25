from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import subprocess


def build_notebook():
    assert importlib.util.find_spec("experiments.build_notebook_07a") is not None
    from experiments.build_notebook_07a import build_notebook as builder

    return builder()


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


def test_notebook_07a_is_an_artifact_only_reader() -> None:
    notebook = build_notebook()
    tree = ast.parse(_code_source(notebook))
    forbidden = {
        "models",
        "features",
        "sentiment",
        "requests",
        "httpx",
        "ollama",
        "experiments.run_q2_sentiment_sensitivity",
    }

    assert forbidden.isdisjoint(_imports(tree))
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr in {"fit", "predict", "predict_proba"}
        for node in ast.walk(tree)
    )


def test_notebook_07a_has_one_explained_table_and_no_figure() -> None:
    notebook = build_notebook()
    table_indices = [
        index
        for index, cell in enumerate(notebook.cells)
        if "result-table" in cell.metadata.get("tags", [])
    ]

    assert len(table_indices) == 1
    assert not any(
        "result-figure" in cell.metadata.get("tags", [])
        for cell in notebook.cells
    )
    index = table_indices[0]
    assert notebook.cells[index - 1].cell_type == "markdown"
    assert notebook.cells[index - 1].source.startswith("Method: ")
    assert "display(Markdown(" in notebook.cells[index].source


def test_notebook_07a_methodology_is_short_complete_and_reader_facing() -> None:
    notebook = build_notebook()
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )

    assert "[2026-04-01, 2026-07-01)" in markdown
    assert "3,947" in markdown and "635" in markdown and "295" in markdown
    assert "DeBERTa-matched SVM" in markdown
    assert "LLM-matched LSTM" in markdown
    assert "2 bps" in markdown and "3 bps" in markdown
    assert "not a price-only ablation" in markdown
    assert "No model fitting" in markdown
    assert "keeps every model visible" not in markdown


def test_notebook_07a_table_has_finite_economics_and_net_sorting() -> None:
    notebook = build_notebook()
    source = next(
        cell.source
        for cell in notebook.cells
        if "result-table" in cell.metadata.get("tags", [])
    )

    assert "Net original %" in source and "Net fresh %" in source
    assert "Sharpe" in source and "Sortino" in source
    assert "sort_values(\"Net fresh %\", ascending=False)" in source
    assert "np.isfinite" in source
    assert "NaN" not in source and "nan" not in source


def test_notebook_07a_validates_hashes_before_reading_results() -> None:
    notebook = build_notebook()
    validation = next(
        cell.source
        for cell in notebook.cells
        if "artifact-validation" in cell.metadata.get("tags", [])
    )

    assert validation.index("artifact_sha256") < validation.index("read_parquet")
    assert "comparison.parquet" in validation
    assert "audit.json" in validation


def test_notebook_07a_compact_reader_artifacts_are_git_tracked() -> None:
    code_root = Path(__file__).resolve().parents[1]
    relative_root = "experiments/cache/q2_sentiment_sensitivity/results"
    expected = {
        f"{relative_root}/{name}"
        for name in (
            "audit.json",
            "comparison.parquet",
            "fresh_summaries.parquet",
            "ledger__usa500_best_single_deberta_svm.parquet",
            "ledger__usa500_deberta_soft_vote.parquet",
            "ledger__usatech_best_single_deepseek_lstm.parquet",
            "ledger__usatech_deepseek_soft_vote.parquet",
            "manifest.json",
            "prediction__usa500_best_single_deberta_svm.parquet",
            "prediction__usa500_deberta_soft_vote.parquet",
            "prediction__usatech_best_single_deepseek_lstm.parquet",
            "prediction__usatech_deepseek_soft_vote.parquet",
        )
    }
    tracked = set(
        subprocess.check_output(
            ["git", "ls-files", relative_root],
            cwd=code_root,
            text=True,
            encoding="utf-8",
        ).splitlines()
    )

    assert tracked == expected
