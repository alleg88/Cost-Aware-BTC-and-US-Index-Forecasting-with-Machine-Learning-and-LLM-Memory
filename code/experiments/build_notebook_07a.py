"""Build the artifact-only reader for the Q2 sentiment availability sensitivity."""
from __future__ import annotations

import argparse
from pathlib import Path
from textwrap import dedent

import nbformat
from nbclient import NotebookClient

from experiments.notebook_hygiene import canonical_colab_setup


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = CODE_ROOT / "notebooks" / "07a_q2_sentiment_sensitivity.ipynb"
RESULT_ROOT = CODE_ROOT / "experiments" / "cache" / "q2_sentiment_sensitivity" / "results"


def _markdown(source: str):
    return nbformat.v4.new_markdown_cell(dedent(source).strip())


def _code(source: str, *, tags: tuple[str, ...] = ()):
    cell = nbformat.v4.new_code_cell(dedent(source).strip())
    if tags:
        cell.metadata["tags"] = list(tags)
    return cell


def build_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.metadata["kernelspec"] = {
        "display_name": "MSC code",
        "language": "python",
        "name": "msc-code",
    }
    notebook.metadata["language_info"] = {"name": "python", "version": "3.12"}
    notebook.metadata["analysis_role"] = "supplementary_q2_sentiment_sensitivity"
    notebook.cells = [
        _code(canonical_colab_setup(), tags=("setup",)),
        _markdown(
            """
            # Q2 sentiment availability sensitivity

            **Finding.** Fresh Q2 sentiment adds one short trade and 0.028 percentage points of
            Net to the USA500 ensemble, leaves two policies unchanged, and reduces USATECH
            LLM-LSTM Net by 0.728 percentage points. The effect is not consistent across policies.
            """
        ),
        _markdown(
            """
            ## Concise methodology

            This supplementary check replays the same four frozen index policies on
            `[2026-04-01, 2026-07-01)`. No model fitting, calibration, threshold change or policy
            selection occurs. Price, admitted VIX, execution and costs remain fixed at **2 bps**
            round trip for USA500 and **3 bps** for USATECH.

            The completed feed contains **3,947** USA500 and **635** USATECH English GDELT rows,
            plus **295** causal direct-event rows per index. DeBERTa and the batch-10 LLM score the
            same source rows before the frozen **DeBERTa-matched SVM**, **LLM-matched LSTM** and
            their all-nine probability ensembles are replayed. The comparison changes only fresh
            Q2 sentiment availability relative to Notebook 07; it is **not a price-only ablation**
            because no H1-eligible frozen price-only index control exists.
            """
        ),
        _code(
            """
            from pathlib import Path
            import hashlib, json
            import numpy as np
            import pandas as pd
            from IPython.display import Markdown, display

            CODE_ROOT = next(
                path for path in (Path.cwd(), *Path.cwd().parents)
                if (path / "pyproject.toml").is_file()
            )
            RESULT_ROOT = (
                CODE_ROOT / "experiments" / "cache" / "q2_sentiment_sensitivity" / "results"
            )

            def sha256(path):
                digest = hashlib.sha256()
                with Path(path).open("rb") as handle:
                    for block in iter(lambda: handle.read(1024 * 1024), b""):
                        digest.update(block)
                return digest.hexdigest()

            manifest = json.loads((RESULT_ROOT / "manifest.json").read_text(encoding="utf-8"))
            artifact_sha256 = manifest["artifact_sha256"]
            for name, expected in artifact_sha256.items():
                path = RESULT_ROOT / name
                assert path.is_file() and sha256(path) == expected, name
            assert manifest["no_fitting"] is True
            assert manifest["no_threshold_selection"] is True
            assert manifest["no_candidate_selection"] is True

            comparison = pd.read_parquet(RESULT_ROOT / "comparison.parquet")
            audit = json.loads((RESULT_ROOT / "audit.json").read_text(encoding="utf-8"))
            numeric = comparison.select_dtypes(include="number").to_numpy(float)
            assert len(comparison) == 4 and np.isfinite(numeric).all()
            baseline_error = max(
                max(details["baseline_max_probability_error"].values())
                for details in audit.values()
            )
            assert baseline_error < 1e-10
            VALIDATED = True
            print(
                f"Integrity verified: {len(artifact_sha256)} artifacts; "
                f"baseline max error {baseline_error:.2e}."
            )
            """,
            tags=("artifact-validation",),
        ),
        _markdown(
            """
            Method: The table compares the original zero-fresh-feed Q2 replay with the completed-feed
            replay for identical frozen policies; rows are sorted by fresh Net after registered costs.
            """
        ),
        _code(
            """
            assert VALIDATED
            labels = {
                "usa500_deberta_soft_vote": "DeBERTa all-nine ensemble",
                "usa500_best_single_deberta_svm": "DeBERTa SVM",
                "usatech_deepseek_soft_vote": "LLM all-nine ensemble",
                "usatech_best_single_deepseek_lstm": "LLM LSTM",
            }
            scorers = {"usa500": "DeBERTa", "usatech": "LLM"}
            table = pd.DataFrame({
                "Market": comparison["stream"].map({"usa500": "USA500", "usatech": "USATECH"}),
                "Frozen policy": comparison["candidate_id"].map(labels),
                "Fresh scorer": comparison["stream"].map(scorers),
                "Trades original": comparison["original_trades"].astype(int),
                "Trades fresh": comparison["fresh_trades"].astype(int),
                "L/S fresh": (
                    comparison["fresh_long_trades"].astype(int).astype(str)
                    + "/"
                    + comparison["fresh_short_trades"].astype(int).astype(str)
                ),
                "Net original %": 100 * comparison["original_net_return"],
                "Net fresh %": 100 * comparison["fresh_net_return"],
                "Delta Net pp": 100 * comparison["delta_net_return"],
                "Sharpe": comparison["fresh_daily_sharpe"],
                "Sortino": comparison["fresh_daily_sortino"],
            })
            table = table.sort_values("Net fresh %", ascending=False).reset_index(drop=True)
            required_numeric = [
                "Trades original", "Trades fresh", "Net original %", "Net fresh %",
                "Delta Net pp", "Sharpe", "Sortino",
            ]
            assert np.isfinite(table[required_numeric].to_numpy(float)).all()
            display(table.style.format({
                "Net original %": "{:.3f}", "Net fresh %": "{:.3f}",
                "Delta Net pp": "{:+.3f}", "Sharpe": "{:.3f}", "Sortino": "{:.3f}",
            }).hide(axis="index"))
            display(Markdown(
                "**Takeaway:** Fresh sentiment slightly improves the USA500 ensemble, materially "
                "reduces USATECH LLM-LSTM Net, and leaves the other policies unchanged, so RQ3 "
                "does not show a consistent net-of-cost improvement."
            ))
            """,
            tags=("result-table",),
        ),
    ]
    return notebook


def write_notebook(path: str | Path = NOTEBOOK_PATH) -> Path:
    destination = Path(path)
    if not (RESULT_ROOT / "manifest.json").is_file():
        raise FileNotFoundError("Run the Q2 sentiment sensitivity before building Notebook 07a")
    destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(build_notebook(), destination)
    return destination


def execute_notebook(path: str | Path = NOTEBOOK_PATH, *, timeout: int = 600) -> Path:
    target = Path(path)
    notebook = nbformat.read(target, as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name="msc-code",
        resources={"metadata": {"path": str(CODE_ROOT)}},
    ).execute()
    nbformat.write(executed, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--path", type=Path, default=NOTEBOOK_PATH)
    args = parser.parse_args()
    path = write_notebook(args.path)
    if args.execute:
        execute_notebook(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
