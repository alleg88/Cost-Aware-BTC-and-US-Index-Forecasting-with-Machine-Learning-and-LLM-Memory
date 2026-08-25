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

import nbformat
from nbclient import NotebookClient


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_ROOT = CODE_ROOT / "notebooks"

BASE_COLAB_DEPENDENCIES = ("catboost==1.2.10", "rapidfuzz==3.14.3")
NOTEBOOK_EXTRA_DEPENDENCIES: dict[str, tuple[str, ...]] = {}

NOTEBOOK_SEQUENCES = {
    "Bitcoin": (
        "01_data_labels_and_baseline.ipynb",
        "01b_positioning_ablation.ipynb",
        "02b_catboost_economic_optuna.ipynb",
        "02c_sentiment_data_and_methodology.ipynb",
        "02d_all_model_sentiment.ipynb",
        "02e_all_model_sentiment_policy.ipynb",
        "03_all_model_stacking.ipynb",
        "03a_stacking_forward.ipynb",
        "03c_qualified_union_ensemble.ipynb",
        "04a_svm_temperature_calibration.ipynb",
        "04b_xgboost_strong_move_admission.ipynb",
        "04d_unified_2021_ensemble.ipynb",
        "04g_lstm_gmadl_shadow.ipynb",
        "04h_union_v1_episode_reentry.ipynb",
        "05c_causal_policy_router_agent.ipynb",
    ),
    "Indices": (
        "06a_index_nine_models.ipynb",
        "06b_index_vix.ipynb",
        "06c_index_deberta.ipynb",
        "06d_index_llm.ipynb",
        "06g_index_all_model_ensemble.ipynb",
        "06i_index_comparison.ipynb",
    ),
    "Channels": (
        "A_channel_strategy.ipynb",
        "U_volatility_timing_feature_consolidation.ipynb",
        "V_economic_direction_head.ipynb",
        "W_channel_vs_volatility_ablation.ipynb",
    ),
    "Final confirmation": (
        "07_final_q2_lockbox.ipynb",
        "07a_q2_sentiment_sensitivity.ipynb",
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


def canonical_colab_setup(extra_dependencies: tuple[str, ...] = ()) -> str:
    dependencies = tuple(dict.fromkeys((*BASE_COLAB_DEPENDENCIES, *extra_dependencies)))
    quoted = ", ".join(repr(dependency) for dependency in dependencies)
    return dedent(
        f'''\
        # Google Colab / local setup
        import os, sys, subprocess
        from pathlib import Path
        if "google.colab" in sys.modules:
            from google.colab import drive
            drive.mount("/content/drive", force_remount=False)
            CODE_ROOT = Path("/content/drive/MyDrive/msc project/code")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e",
                str(CODE_ROOT), {quoted},
            ])
        else:
            CODE_ROOT = next(
                path for path in (Path.cwd(), *Path.cwd().parents)
                if (path / "pyproject.toml").exists()
            )
        os.chdir(CODE_ROOT)
        sys.path.insert(0, str(CODE_ROOT))
        CODE = CODE_ROOT
        '''
    ).strip()


def _is_setup_cell(cell: nbformat.NotebookNode) -> bool:
    if cell.cell_type != "code":
        return False
    source = cell.source
    markers = (
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
    notebook = nbformat.read(path, as_version=4)
    source = canonical_colab_setup(extra_dependencies)
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
