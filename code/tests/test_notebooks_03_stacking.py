from __future__ import annotations

import json
from pathlib import Path

import nbformat
import pandas as pd
from notebook_assertions import tables

from experiments.all_model_stacking import DEFAULT_ROOT


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_03 = CODE_ROOT / "notebooks" / "03_all_model_stacking.ipynb"
NOTEBOOK_03A = CODE_ROOT / "notebooks" / "03a_stacking_forward.ipynb"


def test_all_model_stacking_artifacts_are_complete_and_sealed():
    manifest = json.loads((DEFAULT_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["base_models"]) == 9
    assert manifest["widths"] == [55, 65, 75]
    assert manifest["lockbox_2026_q2_used"] is False
    assert manifest["artifact_rows"] == {
        "h1_policy_grid": 891,
        "h1_selected_per_dz": 36,
        "h1_selected_candidates": 12,
        "meta_coefficients": 486,
        "forward_monthly": 108,
        "forward_quarterly": 36,
        "forward_summary": 12,
    }


def test_forward_replays_exact_h1_frozen_fields():
    selected = pd.read_parquet(DEFAULT_ROOT / "h1_selected_candidates.parquet")
    forward = pd.read_parquet(DEFAULT_ROOT / "forward_summary.parquet")
    keys = ["sentiment_arm", "variant"]
    for field in ("width_bps", "policy_id", "tau", "tp_bps", "sl_bps", "max_hold"):
        assert selected.set_index(keys)[field].sort_index().equals(
            forward.set_index(keys)[field].sort_index()
        )


def _source(path: Path):
    notebook = nbformat.read(path, as_version=4)
    return notebook, "\n".join(cell.source for cell in notebook.cells)


def test_notebook_03_contains_only_construction_and_h1_selection():
    notebook, source = _source(NOTEBOOK_03)
    for phrase in (
        "all nine fixed model families",
        "18 independent inputs",
        "no optuna is used",
        "five 2024 blockingtimeseriessplit oof blocks",
        "33 fixed confidence/tp/sl policies",
        "forward results are not opened in this notebook",
    ):
        assert phrase in source.lower()
    assert "forward_summary.parquet" not in source
    assert [len(table) - 1 for table in tables(notebook)] == [4, 36, 12]
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code and all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.get("outputs", []))


def test_notebook_03a_contains_only_frozen_forward_and_decision():
    notebook, source = _source(NOTEBOOK_03A)
    for phrase in (
        "continues notebook 03",
        "12 h1-frozen candidates",
        "cannot change here",
        "development-forward evaluation",
        "reject the all-nine logistic regression stack",
    ):
        assert phrase in source.lower()
    assert [len(table) - 1 for table in tables(notebook)] == [12, 3]
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code and all(cell.execution_count is not None for cell in code)
    assert not any(output.output_type == "error" for cell in code for output in cell.get("outputs", []))


def test_notebook_stacking_figures_exist():
    assert (CODE_ROOT / "notebooks" / "artifacts" / "03_stacking_coefficients.png").exists()
    assert (CODE_ROOT / "notebooks" / "artifacts" / "03a_stacking_monthly.png").exists()
