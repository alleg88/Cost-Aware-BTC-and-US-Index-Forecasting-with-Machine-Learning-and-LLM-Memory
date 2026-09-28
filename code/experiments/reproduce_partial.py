"""Execute the complete first notebook, without rebuilding the remaining study."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import sys
from time import perf_counter

import nbformat
from nbclient import NotebookClient

from experiments.notebook_hygiene import current_python_kernel
from experiments.notebook_runtime import notebook_inputs


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = "01_RQ1_A_BTC_data_labels_baseline.ipynb"


def reproduce_partial(code_root: Path = CODE_ROOT, *, timeout: int = 900) -> dict[str, object]:
    """Keep the original reader intact; retain new outputs even if a cell fails."""
    code_root = Path(code_root).resolve()
    output_dir = code_root.parent / "partial-check"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / NOTEBOOK
    notebook = None
    started = perf_counter()
    report = {
        "status": "RUNNING",
        "scope": "Complete first notebook using supplied normalised inputs; no source reconstruction or other notebooks",
        "notebook": NOTEBOOK,
        "full_rebuild_executed": False,
        "numerical_equivalence_checked": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "packages": {},
        "input_sha256": {},
        "output_notebook": None,
        "executed_code_cells": 0,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    try:
        for relative in notebook_inputs(NOTEBOOK):
            path = code_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"Partial check requires {relative}. Use the updated Release-Rebuild.zip.")
            with path.open("rb") as stream:
                report["input_sha256"][relative] = hashlib.file_digest(stream, "sha256").hexdigest()
        report["packages"] = {name: metadata.version(name) for name in
                              ("numpy", "pandas", "scikit-learn", "pyarrow", "nbclient")}
        notebook = nbformat.read(code_root / "notebooks" / NOTEBOOK, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
                cell.metadata.pop("execution", None)
        nbformat.write(notebook, output_path)
        report["output_notebook"] = NOTEBOOK
        print(f"Running {NOTEBOOK}; the supplied normalised data are not rebuilt in this partial check.", flush=True)
        with current_python_kernel() as kernel_name:
            NotebookClient(notebook, timeout=timeout, kernel_name=kernel_name,
                           resources={"metadata": {"path": str(code_root)}}).execute()
        report["status"] = "PASSED"
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = str(error)
        raise
    finally:
        report["elapsed_seconds"] = round(perf_counter() - started, 3)
        if report["output_notebook"] is not None:
            report["executed_code_cells"] = sum(cell.cell_type == "code" and cell.execution_count is not None
                                                 for cell in notebook.cells)
            nbformat.write(notebook, output_path)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Partial check {report['status']}: {report_path}", flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=900, help="Maximum seconds per notebook cell")
    args = parser.parse_args()
    reproduce_partial(timeout=args.timeout)
