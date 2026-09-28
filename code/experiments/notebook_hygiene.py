"""Canonical notebook order and portable first-cell normalization."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from textwrap import dedent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import nbformat


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = CODE_ROOT / "notebooks"

BASE_COLAB_DEPENDENCIES = ("catboost==1.2.10", "rapidfuzz==3.14.3")
NOTEBOOK_EXTRA_DEPENDENCIES: dict[str, tuple[str, ...]] = {}

NOTEBOOK_SEQUENCES = {
    "Bitcoin": (
        "01_RQ1_A_BTC_data_labels_baseline.ipynb",
        "02_RQ1_B_BTC_positioning_ablation.ipynb",
        "03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb",
        "12_RQ3_A_BTC_sentiment_data_methodology.ipynb",
        "13_RQ3_B_BTC_nine_model_sentiment_ablation.ipynb",
        "14_RQ3_C_BTC_sentiment_policy_ablation.ipynb",
        "06_RQ2_A_BTC_all_model_stacking.ipynb",
        "07_RQ2_B_BTC_stacking_forward_validation.ipynb",
        "08_RQ2_C_BTC_qualified_union_ensemble.ipynb",
        "09_RQ2_F_BTC_LSTM_GMADL_shadow.ipynb",
        "18_RQ4_A_BTC_LLM_policy_router.ipynb",
    ),
    "Indices": (
        "04_RQ1_E_indices_nine_model_benchmark.ipynb",
        "05_RQ1_F_indices_VIX_ablation.ipynb",
        "15_RQ3_D_indices_DeBERTa_sentiment.ipynb",
        "16_RQ3_E_indices_LLM_sentiment.ipynb",
        "10_RQ2_H_indices_all_model_ensemble.ipynb",
        "11_RQ2_I_indices_policy_comparison.ipynb",
    ),
    "Channels": (
        "19_RQ5_B_BTC_volatility_feature_consolidation.ipynb",
        "20_RQ5_C_BTC_economic_direction_head.ipynb",
        "21_RQ5_D_BTC_channel_vs_volatility_ablation.ipynb",
    ),
    "Final confirmation": (
        "22_Lockbox_Q2_2026.ipynb",
        "17_RQ3_F_indices_Q2_sentiment_sensitivity.ipynb",
    ),
}


@contextmanager
def current_python_kernel():
    """Expose the active interpreter as an isolated Jupyter kernelspec."""
    with TemporaryDirectory(prefix=".notebook_kernel_", dir=CODE_ROOT) as temp_dir:
        data_root = Path(temp_dir)
        kernel_name = f"msc-reproduce-{os.getpid()}"
        spec_dir = data_root / "kernels" / kernel_name
        spec_dir.mkdir(parents=True)
        (spec_dir / "kernel.json").write_text(
            json.dumps(
                {
                    "argv": [
                        sys.executable,
                        "-m",
                        "ipykernel_launcher",
                        "--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel",
                        "--InteractiveShellApp.extensions=[]",
                        "-f",
                        "{connection_file}",
                    ],
                    "display_name": "MSC reproduction environment",
                    "language": "python",
                }
            ),
            encoding="utf-8",
        )
        previous = os.environ.get("JUPYTER_PATH")
        os.environ["JUPYTER_PATH"] = os.pathsep.join(
            path for path in (str(data_root), previous) if path
        )
        try:
            yield kernel_name
        finally:
            if previous is None:
                os.environ.pop("JUPYTER_PATH", None)
            else:
                os.environ["JUPYTER_PATH"] = previous


def canonical_colab_setup(
    extra_dependencies: tuple[str, ...] = (), *, notebook_name: str | None = None,
    rebuild: bool = False,
) -> str:
    archive = "Release-Rebuild.zip" if rebuild else "Release-Client.zip"
    prepare = "prepare_rebuild" if rebuild else "prepare_reader"
    purpose = ("project code and rebuild inputs). Choose this ZIP to calculate this notebook."
               if rebuild else "project code). Choose this ZIP; data files are requested next.")
    archive_source = ('next((p / "Release-Rebuild.zip" for p in (base, *base.parents) '
                      'if (p / "Release-Rebuild.zip").is_file()), base / "Release-Rebuild.zip")'
                      if rebuild else 'base / "Release-Client.zip"')
    return dedent(
        f'''\
        # Notebook setup
        import sys
        from pathlib import Path
        base = Path.cwd()
        source = next((p for p in (base, *base.parents) if (p / "code/pyproject.toml").is_file()), {archive_source})
        if not source.exists() and "google.colab" in sys.modules:
            from google.colab import files
            print("Required file: {archive} ({purpose}", flush=True)
            files.upload(target_dir=str(base))
        if not source.exists():
            raise FileNotFoundError("Supply {archive}, or open the extracted code/notebooks folder.")
        sys.path.insert(0, str(source / "cost-aware-market-forecasting" if source.is_file() else source))
        sys.modules.pop("run_zip", None)
        from run_zip import {prepare}
        CODE = CODE_ROOT = {prepare}({notebook_name!r}, source, {extra_dependencies!r})
        '''
    ).strip()


def _is_setup_cell(cell: nbformat.NotebookNode) -> bool:
    if cell.cell_type != "code":
        return False
    source = cell.source
    markers = (
        "# Notebook setup",
        "# Google Colab / local setup",
        "# Auto-setup for Google Colab / Local environment",
        "# >>> Set this to your code/ folder path",
    )
    if any(marker in source for marker in markers):
        return True
    return False


def normalize_notebook(
    path: Path,
    *,
    extra_dependencies: tuple[str, ...] = (),
) -> bool:
    """Put one canonical setup cell first while preserving reader content."""
    import nbformat

    notebook = nbformat.read(path, as_version=4)
    if path.name == "18_RQ4_A_BTC_LLM_policy_router.ipynb" and any(
        cell.cell_type == "code"
        and "loader.prepare(" in cell.source
        and "manifest_sha256" in cell.source
        for cell in notebook.cells
    ):
        nbformat.validate(notebook)
        return False
    rebuild = any(cell.cell_type == "code" and "from run_zip import prepare_rebuild" in cell.source
                  for cell in notebook.cells if _is_setup_cell(cell))
    source = canonical_colab_setup(extra_dependencies, notebook_name=path.name, rebuild=rebuild)
    setup_indexes = [
        index for index, cell in enumerate(notebook.cells) if _is_setup_cell(cell)
    ]
    if (
        setup_indexes == [0]
        and notebook.cells[0].source == source
        and notebook.cells[0].cell_type == "code"
    ):
        return False

    reader_cells = [cell for cell in notebook.cells if not _is_setup_cell(cell)]
    notebook.cells = [nbformat.v4.new_code_cell(source), *reader_cells]
    nbformat.write(notebook, path)
    return True


def validate_catalog(notebook_root: Path = NOTEBOOK_ROOT) -> None:
    registered = [name for sequence in NOTEBOOK_SEQUENCES.values() for name in sequence]
    if len(registered) != len(set(registered)):
        raise ValueError("canonical notebook sequences contain a duplicate filename")
    actual = {
        path.name
        for path in notebook_root.glob("*.ipynb")
        if path.name != "00_run_in_colab.ipynb"
    }
    expected = set(registered)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise ValueError(f"notebook catalog mismatch: missing={missing}, unexpected={unexpected}")


def normalize_all(notebook_root: Path = NOTEBOOK_ROOT) -> list[Path]:
    validate_catalog(notebook_root)
    changed: list[Path] = []
    for sequence in NOTEBOOK_SEQUENCES.values():
        for name in sequence:
            path = notebook_root / name
            extras = NOTEBOOK_EXTRA_DEPENDENCIES.get(name, ())
            if normalize_notebook(path, extra_dependencies=extras):
                changed.append(path)
    return changed


def execution_order(sequence_names: tuple[str, ...] | None = None) -> tuple[str, ...]:
    if sequence_names is None:
        return tuple(sorted(name for sequence in NOTEBOOK_SEQUENCES.values() for name in sequence))
    names = sequence_names or tuple(NOTEBOOK_SEQUENCES)
    unknown = [name for name in names if name not in NOTEBOOK_SEQUENCES]
    if unknown:
        raise ValueError(f"unknown notebook sequence(s): {unknown}")
    return tuple(
        notebook
        for sequence_name in names
        for notebook in NOTEBOOK_SEQUENCES[sequence_name]
    )


def execute_notebook(
    path: Path,
    *,
    working_directory: Path = CODE_ROOT,
    timeout: int = 900,
) -> Path:
    """Execute one reader in a clean kernel and write only a completed result."""
    import nbformat
    from nbclient import NotebookClient

    notebook = nbformat.read(path, as_version=4)
    with current_python_kernel() as kernel_name:
        executed = NotebookClient(
            notebook,
            timeout=timeout,
            kernel_name=kernel_name,
            resources={"metadata": {"path": str(working_directory)}},
        ).execute()
    nbformat.write(executed, path)
    return path


def execute_all(
    sequence_names: tuple[str, ...] | None = None,
    *,
    notebook_root: Path = NOTEBOOK_ROOT,
    timeout: int = 900,
    start_at: str | None = None,
) -> list[Path]:
    validate_catalog(notebook_root)
    executed: list[Path] = []
    order = execution_order(sequence_names)
    if start_at is not None:
        if start_at not in order:
            raise ValueError(f"start notebook is outside the selected sequence: {start_at}")
        order = order[order.index(start_at) :]
    for index, name in enumerate(order, start=1):
        path = notebook_root / name
        print(f"[{index:02d}/{len(order):02d}] {name}", flush=True)
        execute_notebook(path, working_directory=CODE_ROOT, timeout=timeout)
        executed.append(path)
    return executed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute the registered readers after normalization",
    )
    parser.add_argument(
        "--sequence",
        action="append",
        choices=tuple(NOTEBOOK_SEQUENCES),
        help="execute only this sequence; may be repeated",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--start-at",
        help="resume the selected sequence at this registered notebook filename",
    )
    args = parser.parse_args()
    changed = normalize_all()
    total = sum(len(sequence) for sequence in NOTEBOOK_SEQUENCES.values())
    print(f"Normalized {len(changed)} of {total} canonical notebooks.")
    if args.execute:
        selected = tuple(args.sequence) if args.sequence else None
        executed = execute_all(selected, timeout=args.timeout, start_at=args.start_at)
        print(f"Executed {len(executed)} canonical notebooks.")


if __name__ == "__main__":
    main()
