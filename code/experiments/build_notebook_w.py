"""Build the concise artifact-only Notebook W report."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import nbformat as nbf

from experiments.run_channel_vs_volatility_ablation import (
    READER_ARTIFACTS,
    RUN_ROOT,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
OUT = CODE_ROOT / "notebooks" / "W_channel_vs_volatility_ablation.ipynb"


def _portable(path: Path) -> str:
    return PurePosixPath(*path.parts).as_posix()


def _run_root_locator(output: Path, run_root: Path) -> tuple[str, str]:
    output = output.resolve()
    run_root = run_root.resolve()
    try:
        relative = run_root.relative_to(CODE_ROOT.resolve())
        return "code", _portable(relative)
    except ValueError:
        return "notebook", _portable(Path(os.path.relpath(run_root, output.parent)))


def _section(text: str):
    return nbf.v4.new_markdown_cell(text, metadata={"reader-section": True})


def _code(text: str):
    return nbf.v4.new_code_cell(text, execution_count=None, outputs=[])


def _cells(*, run_root_base: str, run_root_relative: str):
    bootstrap = '''\
# Google Colab / local setup
import os, sys, subprocess
from pathlib import Path

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CODE_ROOT = Path("/content/drive/MyDrive/msc project/code")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(CODE_ROOT),
        "catboost==1.2.10", "rapidfuzz==3.14.3",
    ])
else:
    execution_root = Path.cwd().resolve()
    CODE_ROOT = next(
        (candidate
         for ancestor in (execution_root, *execution_root.parents)
         for candidate in (ancestor, ancestor / "code")
         if (candidate / "pyproject.toml").is_file()),
        execution_root,
    )

os.chdir(CODE_ROOT)
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))
'''
    reader = f'''\
# Validated completed-artifact reader
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

RUN_ROOT_BASE = {run_root_base!r}
RUN_ROOT_RELATIVE = {run_root_relative!r}
RUN_ROOT = (
    (CODE_ROOT / RUN_ROOT_RELATIVE).resolve()
    if RUN_ROOT_BASE == "code"
    else (Path.cwd().resolve() / RUN_ROOT_RELATIVE).resolve()
)
REQUIRED_ARTIFACTS = {tuple(READER_ARTIFACTS)!r}

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _load_completed():
    pointer_path = RUN_ROOT / "latest_dev.json"
    if not pointer_path.is_file():
        return None, "no completed full run pointer"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_hash = str(pointer["run_hash"])
        run_dir = (RUN_ROOT / str(pointer["relative_path"])).resolve()
        if run_dir != (RUN_ROOT / run_hash / "full").resolve():
            raise ValueError("unsafe or non-full run pointer")
        state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "complete" or state.get("run_hash") != run_hash:
            raise ValueError("run state is incomplete or changed")
        records = state.get("artifacts", {{}})
        for name in REQUIRED_ARTIFACTS:
            path = run_dir / name
            record = records.get(name, {{}})
            if not path.is_file() or int(record.get("size", -1)) != path.stat().st_size:
                raise ValueError(f"artifact missing or size changed: {{name}}")
            if str(record.get("sha256", "")) != _sha256(path):
                raise ValueError(f"artifact hash changed: {{name}}")
        protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        frozen = json.loads((run_dir / "frozen_protocol.json").read_text(encoding="utf-8"))
        if state.get("summary") != summary:
            raise ValueError("summary changed from the completed run state")
        required = {{
            "study": "notebook_w_channel_vs_volatility_ablation",
            "stage": "dev",
            "matched_total_activations": 2939,
            "hold_minutes": 120,
            "target_multiple_b": 2.0,
            "round_trip_cost_bps": 10.0,
            "same_minute_ambiguity": "stop_first",
            "forward_or_lockbox_loaded": False,
        }}
        if any(protocol.get(key) != value for key, value in required.items()):
            raise ValueError("Notebook W protocol changed")
        if summary.get("forward_or_lockbox_loaded") is not False:
            raise ValueError("later-period data are not accepted")
        if frozen.get("forward_or_lockbox_loaded") is not False:
            raise ValueError("frozen handoff opened a later period")
        csv_names = (
            "feature_audit.csv", "fold_audit.csv", "predictive_metrics.csv",
            "opportunity_metrics.csv", "economic_metrics.csv",
            "matched_comparisons.csv", "frequency_audit.csv", "leakage_audit.csv",
        )
        frames = {{name: pd.read_csv(run_dir / name) for name in csv_names}}
        if not frames["leakage_audit.csv"]["passed"].astype(bool).all():
            raise ValueError("leakage audit contains a failed check")
        if not frames["frequency_audit.csv"]["fold_counts_exact"].astype(bool).all():
            raise ValueError("matched fold counts changed")
        return (run_dir, protocol, summary, frames), None
    except Exception as error:
        return None, str(error)

LOADED, READER_NOTE = _load_completed()
READER_READY = LOADED is not None
if READER_READY:
    RUN_DIR, PROTOCOL, SUMMARY, FRAMES = LOADED
    print(f"Loaded completed Notebook W run {{RUN_DIR.name}}")
else:
    RUN_DIR = PROTOCOL = SUMMARY = None
    FRAMES = {{}}
    print(f"Completed Notebook W artifacts are not available: {{READER_NOTE}}")
'''
    overview = '''\
if READER_READY:
    display(pd.DataFrame({
        "Item": ["Candidate rows", "Matched activations / arm / model", "Rate", "Decision"],
        "Value": [
            SUMMARY["decision_rows"],
            SUMMARY["selected_activations_per_arm_model"],
            f'{SUMMARY["activations_per_day"]:.4f}/day',
            SUMMARY["decision"],
        ],
    }))
'''
    predictive = '''\
if READER_READY:
    display(FRAMES["predictive_metrics.csv"].round(4))
    display(FRAMES["feature_audit.csv"])
'''
    opportunity_plot = '''\
if READER_READY:
    table = FRAMES["opportunity_metrics.csv"].copy()
    labels = table["model"] + " / " + table["window_source"]
    y = table["opportunity_rate"].to_numpy(float)
    low = table["opportunity_rate_ci_low"].to_numpy(float)
    high = table["opportunity_rate_ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, y, color=["#355070", "#6d597a", "#b56576", "#e56b6f"])
    ax.errorbar(np.arange(len(y)), y, yerr=[y-low, high-y], fmt="none", color="black", capsize=4)
    ax.set_ylabel("Large-move opportunity rate")
    ax.set_title("Matched-frequency opportunity quality")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    display(table.round(4))
'''
    economics_plot = '''\
if READER_READY:
    table = FRAMES["economic_metrics.csv"].copy()
    labels = table["model"] + " / " + table["window_source"]
    y = table["mean_net_r"].to_numpy(float)
    low = table["mean_net_r_ci_low"].to_numpy(float)
    high = table["mean_net_r_ci_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, y, color=["#355070", "#6d597a", "#b56576", "#e56b6f"])
    ax.errorbar(np.arange(len(y)), y, yerr=[y-low, high-y], fmt="none", color="black", capsize=4)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Mean net R per activation")
    ax.set_title("Native 1m RR2 / 120-minute economics after 10 bps")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()
    display(table.round(4))
'''
    decision = '''\
if READER_READY:
    display(FRAMES["matched_comparisons.csv"].round(4))
    display(FRAMES["frequency_audit.csv"].round(4))
    display(FRAMES["leakage_audit.csv"])
    print(SUMMARY["decision"])
'''
    return [
        _code(bootstrap),
        nbf.v4.new_markdown_cell(
            "# Notebook W — Channel vs volatility/opportunity windows\n\n"
            "Final development-only matched-frequency ablation. The objective is to "
            "test whether channel windows add value beyond ordinary channel-blind "
            "volatility/opportunity windows before closing this research branch."
        ),
        _code(reader),
        _section(
            "## W1 — Pre-registered question\n\n"
            "Both arms use the same channel-free LogReg and XGBoost opportunity and "
            "direction heads. The only change is the candidate universe: **channel "
            "windows** versus **channel-blind volatility/opportunity windows** over all "
            "completed 5m bars; forward and Q2 remain sealed."
        ),
        _code(overview),
        _section(
            "## W2 — Exact frequency and execution\n\n"
            "Each model/source receives the frozen fold counts from Notebook V: 2,939 "
            "activations over 1,096 scored days = **2.6816 activations/day**. A "
            "**matched top-k**, label-blind ranking uses only OOF opportunity score and "
            "time, with a global 60-minute refractory period. This is an offline "
            "matched-frequency diagnostic, not a deployable threshold. Execution is "
            "native 1m at the decision-time Open, RR2, 120-minute hold, stop-first on "
            "same-minute ambiguity, and **5 bps entry + 5 bps exit** for every outcome."
        ),
        _section(
            "## W3 — Shared causal models\n\n"
            "Expanding half-year folds train only on labels ending before validation. "
            "The opportunity head sees 20 volatility, activity and positioning-quality "
            "features; the direction head sees 17 signed price, flow and positioning "
            "features. Neither head sees channel geometry, window membership, future "
            "returns or outcomes. Overlapping 120-minute labels receive uniqueness "
            "weights. XGBoost is primary; LogReg is the benchmark."
        ),
        _code(predictive),
        _section("## W4 — Opportunity quality"),
        _code(opportunity_plot),
        _section("## W5 — Economic result"),
        _code(economics_plot),
        _section(
            "## W6 — Stop rule\n\n"
            "Channels are retained only if primary XGBoost has positive net R with its "
            "weekly-bootstrap lower bound above zero, and the channel-minus-blind lower "
            "bounds are above zero for both net R and opportunity rate. Otherwise we "
            "**close the channel branch** and write the dissertation from the evidence "
            "already obtained; LLM assistance remains a separate later question."
        ),
        _code(decision),
    ]


def build_notebook(
    *,
    output: Path = OUT,
    run_root: Path = RUN_ROOT,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base, relative = _run_root_locator(output, Path(run_root))
    notebook = nbf.v4.new_notebook(
        cells=_cells(run_root_base=base, run_root_relative=relative),
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.validate(notebook)
    nbf.write(notebook, output)
    return output


if __name__ == "__main__":
    print(build_notebook())
