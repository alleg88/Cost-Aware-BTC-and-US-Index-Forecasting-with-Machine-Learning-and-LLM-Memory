from __future__ import annotations

import importlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient

from experiments.notebook_hygiene import current_python_kernel


def _builder():
    return importlib.import_module("experiments.build_notebook_w")


def _build(tmp_path: Path):
    builder = _builder()
    path = builder.build_notebook(
        output=tmp_path / "W_channel_vs_volatility_ablation.ipynb",
        run_root=tmp_path / "missing",
    )
    return path, nbformat.read(path, as_version=4)


def _source(notebook) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def _outputs(notebook) -> str:
    values: list[str] = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                values.append(output.get("text", ""))
            elif output.get("output_type") in {"display_data", "execute_result"}:
                values.append(output.get("data", {}).get("text/plain", ""))
    return "\n".join(values)


def test_notebook_w_is_short_english_artifact_only_methodology(tmp_path):
    _, notebook = _build(tmp_path)
    source = _source(notebook)
    lower = source.lower()

    assert notebook.cells[0].cell_type == "code"
    assert "google.colab" in notebook.cells[0].source
    assert "Notebook W" in source
    assert "channel windows" in lower
    assert "channel-blind volatility/opportunity windows" in lower
    assert "2.6816 activations/day" in source
    assert "matched top-k" in lower
    assert "label-blind" in lower
    assert "LogReg" in source and "XGBoost" in source
    assert "5 bps entry + 5 bps exit" in source
    assert "RR2" in source and "120-minute" in source
    assert "native 1m" in source and "stop-first" in lower
    assert "forward and Q2 remain sealed" in source
    assert "close the channel branch" in lower
    assert ".fit(" not in source
    assert "run_channel_vs_volatility_ablation(" not in source
    assert "http://" not in lower and "https://" not in lower


def test_notebook_w_executes_gracefully_without_results(tmp_path):
    path, notebook = _build(tmp_path)
    with current_python_kernel() as kernel_name:
        executed = NotebookClient(
            notebook,
            timeout=180,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()

    assert "Completed Notebook W artifacts are not available" in _outputs(executed)
    assert not [
        output
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
