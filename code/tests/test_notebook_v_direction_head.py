from __future__ import annotations

import importlib
from pathlib import Path

import nbformat
from nbclient import NotebookClient
import pytest

from experiments.notebook_hygiene import current_python_kernel


def _builder():
    try:
        return importlib.import_module("experiments.build_notebook_v")
    except ModuleNotFoundError as error:
        pytest.fail(f"Notebook V builder is missing: {error}")


def _built(tmp_path: Path, run_root: Path):
    builder = _builder()
    path = builder.build_notebook(
        output=tmp_path / "20_RQ5_C_BTC_economic_direction_head.ipynb",
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


def _execute(path: Path, *, cwd: Path | None = None):
    notebook = nbformat.read(path, as_version=4)
    with current_python_kernel() as kernel_name:
        return NotebookClient(
            notebook,
            timeout=180,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(cwd or path.parent)}},
        ).execute()


def test_notebook_v_has_exact_v0_to_v4_sections(tmp_path):
    _, notebook = _built(tmp_path, tmp_path / "missing")
    sections = [
        cell.source.splitlines()[0]
        for cell in notebook.cells
        if cell.get("metadata", {}).get("reader-section")
    ]
    assert sections == [
        "## V0 — Frozen geometry",
        "## V1 — Economic target and features",
        "## V2 — Expanding training",
        "## V3 — Forced-choice policy",
        "## V4 — Economic decision",
    ]


def test_notebook_v_is_colab_portable_artifact_only_reader(tmp_path):
    _, notebook = _built(tmp_path, tmp_path / "missing")
    source = _source(notebook)
    lower = source.lower()

    assert "google.colab" in source
    assert "/content/drive/MyDrive/msc project/code" in source
    assert "JSON stores metadata only" in source
    assert "frozen U timing selects WHEN and frequency" in source
    assert "V forced direction selects SIDE" in source
    assert "no WAIT" in source
    assert "5 bps entry + 5 bps exit" in source
    assert "RR2" in source and "120-minute" in source
    assert "native one-minute" in lower and "stop-first" in lower
    assert "expanding" in lower and "purged" in lower
    assert "episode-disjoint" in lower
    assert "28-feature" in lower
    assert "forward and q2 remain sealed" in lower
    assert "model.fit(" not in source
    assert ".fit(" not in source
    assert "run_direction_head(" not in source
    assert "replay_brackets(" not in source
    assert "threshold_search(" not in source
    assert "read_csv(\"http" not in lower
    assert "read_parquet(\"http" not in lower
    assert "requests." not in lower
    assert "urllib" not in lower
    assert "http://" not in lower and "https://" not in lower
    assert " disk" not in lower


def test_notebook_v_reader_contract_matches_runner():
    builder = _builder()
    runner = importlib.import_module("experiments.run_event_window_direction_head")

    assert builder.REQUIRED_READER_ARTIFACTS == tuple(runner.READER_ARTIFACTS)
    assert set(builder.EXPECTED_FRAME_COLUMNS).issubset(
        builder.REQUIRED_READER_ARTIFACTS
    )
    assert set(builder.EXPECTED_PARQUET_COLUMNS).issubset(
        builder.REQUIRED_READER_ARTIFACTS
    )


def test_notebook_v_executes_gracefully_without_completed_results(tmp_path):
    path, _ = _built(tmp_path, tmp_path / "missing")
    executed = _execute(path)

    assert "Completed Notebook V artifacts are not available" in _text_outputs(executed)
    assert not [
        output
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


def test_notebook_v_executes_completed_smoke_as_non_evidential(tmp_path):
    runner = importlib.import_module("experiments.run_event_window_direction_head")
    result = runner.run_direction_head(run_root=tmp_path / "runs", smoke=True)
    path, _ = _built(tmp_path, tmp_path / "runs")
    executed = _execute(path)
    outputs = _text_outputs(executed)

    assert result.summary["research_claim"] == "smoke_only_no_claim"
    assert "Loaded completed Notebook V smoke run" in outputs
    assert "SMOKE / NON-EVIDENTIAL" in outputs
    assert "smoke only; no economic claim" in outputs
    assert not [
        output
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]


def test_notebook_v_loads_completed_smoke_when_executed_from_repo_root(
    tmp_path, monkeypatch
):
    builder = _builder()
    runner = importlib.import_module("experiments.run_event_window_direction_head")
    repo_root = tmp_path / "repo"
    code_root = repo_root / "code"
    code_root.mkdir(parents=True)
    (code_root / "pyproject.toml").write_text(
        "[project]\nname = 'notebook-v-portability-fixture'\nversion = '0'\n",
        encoding="utf-8",
    )
    run_root = code_root / "runs"
    result = runner.run_direction_head(run_root=run_root, smoke=True)
    monkeypatch.setattr(builder, "CODE_ROOT", code_root)
    path = builder.build_notebook(
        output=code_root / "notebooks" / "20_RQ5_C_BTC_economic_direction_head.ipynb",
        run_root=run_root,
    )

    executed = _execute(path, cwd=repo_root)
    outputs = _text_outputs(executed)

    assert result.summary["research_claim"] == "smoke_only_no_claim"
    assert "Loaded completed Notebook V smoke run" in outputs
    assert "SMOKE / NON-EVIDENTIAL" in outputs
    assert "Completed Notebook V artifacts are not available" not in outputs
