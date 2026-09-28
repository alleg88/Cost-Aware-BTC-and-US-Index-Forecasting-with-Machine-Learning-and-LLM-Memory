from __future__ import annotations

import importlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest

from experiments.notebook_hygiene import current_python_kernel


def _builder():
    try:
        return importlib.import_module("experiments.build_notebook_u")
    except ModuleNotFoundError as error:
        pytest.fail(f"Notebook U builder is missing: {error}")


def _built(tmp_path: Path, run_root: Path):
    builder = _builder()
    path = builder.build_notebook(
        output=tmp_path / "19_RQ5_B_BTC_volatility_feature_consolidation.ipynb",
        run_root=run_root,
    )
    return path, nbformat.read(path, as_version=4)


def _source(notebook) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def _text_outputs(notebook) -> str:
    values = []
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            if output.get("output_type") == "stream":
                values.append(output.get("text", ""))
            elif output.get("output_type") in {"display_data", "execute_result"}:
                values.append(output.get("data", {}).get("text/plain", ""))
    return "\n".join(values)


def test_notebook_u_has_six_concise_continuation_sections(tmp_path):
    _, notebook = _built(tmp_path, tmp_path / "missing")
    sections = [
        cell.source.splitlines()[0]
        for cell in notebook.cells
        if cell.get("metadata", {}).get("reader-section")
    ]
    assert sections == [
        "## 1. Frozen handoff and research question",
        "## 2. Exact 28-feature contract and exclusions",
        "## 3. Causality, symmetry and effective sample",
        "## 4. Paired timing results",
        "## 5. Level re-arm frequency and economics",
        "## 6. Decision",
    ]


def test_notebook_u_is_colab_portable_artifact_only_reader(tmp_path):
    _, notebook = _built(tmp_path, tmp_path / "missing")
    source = _source(notebook)
    lower = source.lower()

    assert "google.colab" in source
    assert "/content/drive/MyDrive/msc project/code" in source
    assert "JSON stores metadata only" in source
    assert "248" in source and "28" in source
    assert "channel_center_distance" in source
    assert "nearest_rail_distance_bps" in source
    assert "rail_approach_15m" in source
    assert "p_hit" in source and "frozen" in lower
    assert "h15" in source and "h30" in source and "h60" in source
    assert "level re-arm" in lower
    assert "calendar excluded" in lower
    assert "impulse excluded" in lower
    assert "direction-70" in lower and "stress test" in lower
    assert "LogReg" in source and "XGBoost" in source
    assert "model.fit(" not in source
    assert "XGBClassifier" not in source
    assert "run_feature_consolidation(" not in source
    assert " disk" not in lower


def test_notebook_u_reader_contract_matches_runner():
    builder = _builder()
    runner = importlib.import_module(
        "experiments.run_event_window_feature_consolidation"
    )

    assert builder.REQUIRED_READER_ARTIFACTS == tuple(runner.READER_ARTIFACTS)
    assert set(builder.EXPECTED_FRAME_COLUMNS).issubset(
        builder.REQUIRED_READER_ARTIFACTS
    )


def test_notebook_u_executes_gracefully_without_full_results(tmp_path):
    path, _ = _built(tmp_path, tmp_path / "missing")
    notebook = nbformat.read(path, as_version=4)
    with current_python_kernel() as kernel_name:
        executed = NotebookClient(
            notebook,
            timeout=120,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(path.parent)}},
        ).execute()

    assert "Completed Notebook U artifacts are not available" in _text_outputs(executed)
    assert not [
        output
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
