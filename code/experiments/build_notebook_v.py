"""Build the artifact-only Notebook V economic direction reader."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import nbformat as nbf

from experiments.run_event_window_direction_head import READER_ARTIFACTS, RUN_ROOT


CODE_ROOT = Path(__file__).resolve().parents[1]
OUT = CODE_ROOT / "notebooks" / "20_RQ5_C_BTC_economic_direction_head.ipynb"
EXPECTED_FRAME_COLUMNS: dict[str, tuple[str, ...]] = {
    "geometry_audit.csv": (
        "geometry",
        "primary",
        "diagnostic_only",
        "target_multiple_b",
        "hold_minutes",
        "entry_cost_bps",
        "exit_cost_bps",
        "round_trip_cost_bps",
        "native_one_minute",
        "stop_first",
        "executed",
        "path_completeness",
        "smoke_deferred",
    ),
    "feature_audit.csv": (
        "feature",
        "position",
        "included",
        "known_at_decision_time",
        "future_column",
        "finite_fraction",
    ),
    "correlation_audit.csv": (
        "feature_left",
        "feature_right",
        "spearman",
        "abs_spearman",
        "diagnostic_only",
    ),
    "fold_audit.csv": (
        "fold_id",
        "train_fit_rows",
        "validation_rows",
        "train_episodes",
        "validation_episodes",
        "episode_overlap",
        "train_label_end_max",
        "validation_start",
    ),
    "predictive_metrics.csv": (
        "model",
        "scored_rows",
        "raw_sign_accuracy",
        "value_weighted_sign_accuracy",
        "mae_delta_r",
        "rmse_delta_r",
    ),
    "economic_metrics.csv": (
        "scenario",
        "activations",
        "path_completeness",
        "mean_net_r",
        "total_net_r",
        "mean_net_bps",
        "mean_net_r_ci_low",
        "mean_net_r_ci_high",
        "mean_oracle_regret_r",
        "oracle_value_capture",
    ),
    "paired_bootstrap.csv": (
        "comparison",
        "candidate",
        "baseline",
        "point_mean_net_r",
        "ci_low",
        "ci_high",
        "draws",
        "bootstrap_unit",
    ),
    "frequency_audit.csv": (
        "scenario",
        "activations",
        "unique_activation_keys",
        "activations_per_day",
        "keys_equal_frozen",
        "timing_owned_by_frozen_u",
    ),
    "concurrency_audit.csv": (
        "source",
        "activations",
        "hold_minutes",
        "mean_active_at_entry",
        "maximum_active_at_entry",
        "overlap_fraction",
        "capacity_suppression",
    ),
    "leakage_audit.csv": ("check", "passed", "detail"),
}
EXPECTED_PARQUET_COLUMNS: dict[str, tuple[str, ...]] = {
    "direction_dataset.parquet": (
        "activation_key",
        "fold_id",
        "best_side",
        "delta_r",
        "economic_value",
    ),
    "economic_paths.parquet": (
        "activation_key",
        "direction",
        "geometry",
        "outcome",
        "net_r",
        "cost_bps",
    ),
    "oof_direction_predictions.parquet": (
        "model",
        "fold_id",
        "activation_key",
        "direction_score",
        "chosen_direction",
    ),
    "policy_paths.parquet": (
        "scenario",
        "activation_key",
        "chosen_direction",
        "net_r",
    ),
    "combined_policy_ledger.parquet": (
        "model",
        "activation_key",
        "decision_time",
        "threshold",
        "activation_score",
        "chosen_direction",
    ),
}
REQUIRED_READER_ARTIFACTS = tuple(READER_ARTIFACTS)


def _portable_parts(path: Path) -> str:
    return PurePosixPath(*path.parts).as_posix()


def _run_root_locator(*, output: Path, run_root: Path) -> tuple[str, str]:
    output = output.resolve()
    run_root = run_root.resolve()
    try:
        relative = run_root.relative_to(CODE_ROOT.resolve())
    except ValueError:
        relative = Path(os.path.relpath(run_root, output.parent))
        return "notebook", _portable_parts(relative)
    return "code", _portable_parts(relative)


def _section(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(source, metadata={"reader-section": True})


def _code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(source, execution_count=None, outputs=[])


def _cells(*, run_root_base: str, run_root_relative: str) -> list[nbf.NotebookNode]:
    bootstrap = '''\
# Google Colab / local setup
import os, sys, subprocess
from pathlib import Path

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CODE_ROOT = Path("/content/drive/MyDrive/msc project/code")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--no-deps",
        "-e",
        str(CODE_ROOT),
        "catboost==1.2.10",
        "rapidfuzz==3.14.3",
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
    setup = f'''\
# Validated artifact reader
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

RUN_ROOT_BASE = {run_root_base!r}
RUN_ROOT_RELATIVE = {run_root_relative!r}
if RUN_ROOT_BASE == "code":
    RUN_ROOT = (CODE_ROOT / RUN_ROOT_RELATIVE).resolve()
else:
    RUN_ROOT = (Path.cwd().resolve() / RUN_ROOT_RELATIVE).resolve()

EXPECTED_FRAME_COLUMNS = {EXPECTED_FRAME_COLUMNS!r}
EXPECTED_PARQUET_COLUMNS = {EXPECTED_PARQUET_COLUMNS!r}
REQUIRED_READER_ARTIFACTS = {REQUIRED_READER_ARTIFACTS!r}

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _candidate_run_dirs():
    candidates = []
    pointer_path = RUN_ROOT / "latest_dev.json"
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_hash = str(pointer.get("run_hash", ""))
        protocol_hash = str(pointer.get("protocol_hash", ""))
        relative_path = Path(str(pointer.get("relative_path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("completed-run pointer path is unsafe")
        run_dir = (RUN_ROOT / relative_path).resolve()
        if run_dir != (RUN_ROOT / run_hash / "full").resolve():
            raise ValueError("completed-run pointer does not identify its full run")
        candidates.append((run_dir, run_hash, protocol_hash, False))
    smoke_states = sorted(
        RUN_ROOT.glob("*/smoke/run_state.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for state_path in smoke_states:
        candidates.append((state_path.parent, state_path.parent.parent.name, None, True))
    return candidates

def _load_candidate(run_dir, run_hash, pointer_protocol_hash, smoke):
    state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
    if state.get("status") != "complete" or state.get("run_hash") != run_hash:
        raise ValueError("run state is not complete or its identity changed")
    if pointer_protocol_hash is not None and state.get("protocol_hash") != pointer_protocol_hash:
        raise ValueError("completed-run pointer protocol does not match")
    records = state.get("artifacts", {{}})
    for name in REQUIRED_READER_ARTIFACTS:
        path = run_dir / name
        record = records.get(name, {{}})
        if not path.is_file() or not record:
            raise ValueError(f"required artifact is missing: {{name}}")
        if int(record.get("size", -1)) != path.stat().st_size:
            raise ValueError(f"artifact size does not match: {{name}}")
        if str(record.get("sha256", "")) != _sha256(path):
            raise ValueError(f"artifact hash does not match: {{name}}")

    protocol = json.loads((run_dir / "protocol.json").read_text(encoding="utf-8"))
    frozen = json.loads((run_dir / "frozen_protocol.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if state.get("summary") != summary:
        raise ValueError("summary does not match the completed run state")
    if any(protocol.get(name) != state.get(name) for name in (
        "run_hash", "protocol_hash", "source_hash", "input_hash"
    )):
        raise ValueError("protocol identity does not match the completed run state")
    if protocol.get("study") != "notebook_v_economic_direction_value_head":
        raise ValueError("artifact study is not Notebook V")
    if protocol.get("stage") != "dev" or bool(protocol.get("smoke")) != smoke:
        raise ValueError("artifact execution mode does not match its run location")
    if protocol.get("frozen_u_run_hash") != frozen.get("frozen_u_run_hash"):
        raise ValueError("frozen Notebook U identity does not reconcile")
    for payload in (protocol, frozen, summary):
        if payload.get("forward_or_lockbox_loaded") is not False:
            raise ValueError("later-period data are not accepted")
    required_protocol = {{
        "direction_feature_count": 28,
        "entry_cost_bps": 5.0,
        "target_exit_cost_bps": 5.0,
        "other_exit_cost_bps": 5.0,
        "round_trip_cost_bps": 10.0,
        "hold_minutes": 120,
        "target_multiple_b": 2.0,
        "forced_direction": True,
        "trade_gate_included": False,
        "threshold_search": False,
        "timing_model_refit": False,
    }}
    if any(protocol.get(key) != value for key, value in required_protocol.items()):
        raise ValueError("Notebook V frozen reader protocol changed")
    required_summary = {{
        "source_activations": 3431,
        "scored_activations": 2939,
        "warmup_activations": 492,
        "direction_feature_count": 28,
        "round_trip_cost_bps": 10.0,
        "timing_model_refit": False,
        "timing_frequency_changed": False,
        "trade_gate_included": False,
        "forced_direction": True,
    }}
    if any(summary.get(key) != value for key, value in required_summary.items()):
        raise ValueError("Notebook V completed summary changed")
    if smoke:
        if summary.get("research_claim") != "smoke_only_no_claim":
            raise ValueError("smoke artifacts were not marked non-evidential")
        if summary.get("selected_model") is not None:
            raise ValueError("smoke artifacts may not select a direction model")
    elif summary.get("research_claim") != "development_result":
        raise ValueError("full development artifacts lack their result provenance")

    frames = {{}}
    for name, required_columns in EXPECTED_FRAME_COLUMNS.items():
        frame = pd.read_csv(run_dir / name)
        missing = [column for column in required_columns if column not in frame]
        if missing:
            raise ValueError(f"artifact schema does not match: {{name}}: {{missing}}")
        frames[name] = frame
    parquets = {{}}
    for name, required_columns in EXPECTED_PARQUET_COLUMNS.items():
        frame = pd.read_parquet(run_dir / name)
        missing = [column for column in required_columns if column not in frame]
        if missing:
            raise ValueError(f"artifact schema does not match: {{name}}: {{missing}}")
        parquets[name] = frame

    feature_audit = frames["feature_audit.csv"]
    if len(feature_audit) != 28 or feature_audit["position"].tolist() != list(range(28)):
        raise ValueError("28-feature audit order changed")
    for column in ("included", "known_at_decision_time"):
        if not feature_audit[column].astype(bool).all():
            raise ValueError(f"feature audit failed: {{column}}")
    if feature_audit["future_column"].astype(bool).any():
        raise ValueError("feature audit contains a future field")
    if not frames["leakage_audit.csv"]["passed"].astype(bool).all():
        raise ValueError("leakage audit contains a failed check")
    if not frames["frequency_audit.csv"]["keys_equal_frozen"].astype(bool).all():
        raise ValueError("direction policy changed frozen activation keys")
    if set(parquets["combined_policy_ledger.parquet"]["chosen_direction"]) - {{"long", "short"}}:
        raise ValueError("combined policy contains an action other than LONG/SHORT")
    # JSON stores metadata only; analytical rows remain in validated Parquet and CSV artifacts.
    return run_dir, protocol, frozen, summary, frames, parquets, smoke

def _load_completed_run():
    errors = []
    try:
        candidates = _candidate_run_dirs()
    except Exception as error:
        return None, None, None, None, {{}}, {{}}, None, str(error)
    for candidate in candidates:
        try:
            loaded = _load_candidate(*candidate)
            return (*loaded, None)
        except Exception as error:
            errors.append(str(error))
    note = errors[0] if errors else "no completed full or smoke run was found"
    return None, None, None, None, {{}}, {{}}, None, note

(RUN_DIR, PROTOCOL, FROZEN, SUMMARY, FRAMES, PARQUETS,
 RUN_IS_SMOKE, READER_NOTE) = _load_completed_run()
READER_READY = RUN_DIR is not None
if READER_READY and RUN_IS_SMOKE:
    EVIDENCE_LABEL = "SMOKE / NON-EVIDENTIAL"
    print(f"Loaded completed Notebook V smoke run: {{RUN_DIR.parent.name}}")
    print("SMOKE / NON-EVIDENTIAL: structural reader execution only; no economic claim.")
elif READER_READY:
    EVIDENCE_LABEL = "FULL DEVELOPMENT EVIDENCE"
    print(f"Loaded completed Notebook V full development run: {{RUN_DIR.parent.name}}")
else:
    EVIDENCE_LABEL = "ARTIFACTS UNAVAILABLE"
    print(f"Completed Notebook V artifacts are not available: {{READER_NOTE}}.")
'''
    return [
        _code(bootstrap),
        nbf.v4.new_markdown_cell(
            "# Notebook V — Economic Direction Value Head\n\n"
            "A separate continuation immediately after Notebook U. The combined "
            "architecture is deliberately narrow: **frozen U timing selects WHEN "
            "and frequency; V forced direction selects SIDE for every scored "
            "activation, with no WAIT**."
        ),
        _code(setup),
        _section(
            "## V0 — Frozen geometry\n\n"
            "Notebook V keeps U's activation keys, times, scores and thresholds. Each "
            "activation is replayed LONG and SHORT with adaptive 1B risk, RR2, a "
            "120-minute native one-minute path and stop-first same-minute handling. "
            "Costs are **5 bps entry + 5 bps exit** for every outcome. Fixed 75 and "
            "120 bps barriers are diagnostic only; oracle value checks feasibility, "
            "not geometry selection."
        ),
        _code('''\
if READER_READY:
    print(EVIDENCE_LABEL)
    geometry = FRAMES["geometry_audit.csv"].copy()
    display(geometry[[
        "geometry", "primary", "diagnostic_only", "target_multiple_b",
        "hold_minutes", "round_trip_cost_bps", "native_one_minute",
        "stop_first", "executed", "path_completeness", "smoke_deferred",
    ]])
    oracle = FRAMES["economic_metrics.csv"].loc[
        FRAMES["economic_metrics.csv"]["scenario"].eq("oracle"),
        ["scenario", "activations", "path_completeness", "mean_net_r",
         "mean_net_r_ci_low", "mean_net_r_ci_high"],
    ]
    display(oracle)
else:
    print("Frozen geometry and oracle feasibility await completed artifacts.")
'''),
        _section(
            "## V1 — Economic target and features\n\n"
            "For each paired path, `delta_r = net_r_long - net_r_short`; its sign "
            "defines the best side and `min(abs(delta_r), 3)` defines economic value. "
            "The fixed 28-feature causal contract includes U's activation margin. "
            "Realised outcomes remain labels only, fold-local imputation handles "
            "missing values, and Spearman correlation is diagnostic rather than a "
            "feature-selection rule."
        ),
        _code('''\
if READER_READY:
    print(EVIDENCE_LABEL)
    feature_audit = FRAMES["feature_audit.csv"].copy()
    display(feature_audit[[
        "position", "feature", "finite_fraction", "known_at_decision_time",
        "future_column",
    ]])
    direction = PARQUETS["direction_dataset.parquet"]
    target_summary = direction[["delta_r", "economic_value"]].describe().T
    display(target_summary)
    side_counts = direction["best_side"].value_counts().reindex(
        ["long", "short", "tie"], fill_value=0
    )
    side_counts.plot(kind="bar", figsize=(6.5, 3.2), color="#4C78A8")
    plt.title("Paired economic target by best side")
    plt.xlabel("Best side")
    plt.ylabel("Frozen activations")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    correlations = FRAMES["correlation_audit.csv"].sort_values(
        "abs_spearman", ascending=False
    ).head(12)
    display(correlations[["feature_left", "feature_right", "spearman", "diagnostic_only"]])
else:
    print("The target, 28-feature audit and correlation diagnostics are pending.")
'''),
        _section(
            "## V2 — Expanding training\n\n"
            "The direction models use 2022H1 as warm-up, then score six expanding "
            "out-of-fold periods. Every fold uses only earlier activations, applies "
            "the complete 120-minute purge, excludes validation episodes from "
            "training, making the folds episode-disjoint, computes interval-uniqueness "
            "weights within training and uses "
            "fold-local preprocessing. Training and validation rows and episodes are "
            "reported separately."
        ),
        _code('''\
if READER_READY:
    print(EVIDENCE_LABEL)
    folds = FRAMES["fold_audit.csv"].copy()
    display(folds[[
        "fold_id", "train_fit_rows", "validation_rows", "train_episodes",
        "validation_episodes", "episode_overlap", "train_label_end_max",
        "validation_start",
    ]])
    fold_rows = folds.set_index("fold_id")[["train_fit_rows", "validation_rows"]]
    fold_rows.columns = ["Training rows", "Validation rows"]
    fold_rows.plot(kind="bar", figsize=(8, 3.5), color=["#4C78A8", "#F58518"])
    plt.title("Expanding and purged direction folds")
    plt.xlabel("Validation fold")
    plt.ylabel("Activations")
    plt.xticks(rotation=25, ha="right")
    plt.legend(title="Sample role")
    plt.tight_layout()
    plt.show()
else:
    print("Expanding fold counts await completed artifacts.")
'''),
        _section(
            "## V3 — Forced-choice policy\n\n"
            "Frozen U timing selects WHEN and preserves frequency. V selects exactly "
            "one LONG or SHORT SIDE for every evaluable activation: there is no WAIT, "
            "confidence gate, threshold search or capacity suppression. LogReg, "
            "XGBoost and all baselines are compared on identical keys and times."
        ),
        _code('''\
if READER_READY:
    print(EVIDENCE_LABEL)
    frequency = FRAMES["frequency_audit.csv"].copy()
    display(frequency[[
        "scenario", "activations", "unique_activation_keys",
        "activations_per_day", "keys_equal_frozen", "timing_owned_by_frozen_u",
    ]])
    combined = PARQUETS["combined_policy_ledger.parquet"]
    side_counts = combined.groupby(["model", "chosen_direction"]).size().unstack(fill_value=0)
    display(side_counts)
    side_counts.plot(kind="bar", stacked=True, figsize=(6.5, 3.2),
                     color=["#4C78A8", "#E45756"])
    plt.title("Forced side decisions at matched frequency")
    plt.xlabel("Direction model")
    plt.ylabel("Scored activations")
    plt.xticks(rotation=0)
    plt.legend(title="Chosen side")
    plt.tight_layout()
    plt.show()
else:
    print("Matched-frequency side decisions await completed artifacts.")
'''),
        _section(
            "## V4 — Economic decision\n\n"
            "Primary evidence is net R under 10 bps round-trip costs, with absolute "
            "and paired episode-cluster intervals versus `channel_side`. Predictive "
            "accuracy is secondary. Viability also requires complete matched paths, "
            "all leakage checks and positive lower bounds. The frozen decision never "
            "adds a trade/no-trade gate, and **forward and Q2 remain sealed** "
            "regardless of the result."
        ),
        _code('''\
if READER_READY:
    print(EVIDENCE_LABEL)
    predictive = FRAMES["predictive_metrics.csv"].copy()
    display(predictive[[
        "model", "scored_rows", "raw_sign_accuracy",
        "value_weighted_sign_accuracy", "mae_delta_r", "rmse_delta_r",
    ]])
    economics = FRAMES["economic_metrics.csv"].copy()
    display(economics[[
        "scenario", "activations", "path_completeness", "mean_net_r",
        "mean_net_r_ci_low", "mean_net_r_ci_high", "mean_net_bps",
        "mean_oracle_regret_r", "oracle_value_capture",
    ]])
    paired = FRAMES["paired_bootstrap.csv"].copy()
    display(paired[[
        "comparison", "point_mean_net_r", "ci_low", "ci_high", "draws",
        "bootstrap_unit",
    ]])
    display(FRAMES["concurrency_audit.csv"])
    leakage = FRAMES["leakage_audit.csv"]
    print(f"Leakage and alignment checks passed: "
          f"{int(leakage['passed'].astype(bool).sum())}/{len(leakage)}")

    chart = economics.set_index("scenario")
    lower_error = np.maximum(0.0, chart["mean_net_r"] - chart["mean_net_r_ci_low"])
    upper_error = np.maximum(0.0, chart["mean_net_r_ci_high"] - chart["mean_net_r"])
    chart["mean_net_r"].plot(
        kind="bar", yerr=np.vstack([lower_error, upper_error]), capsize=3,
        figsize=(8, 3.6), color="#4C78A8",
    )
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Net economics with episode-cluster intervals")
    plt.xlabel("Direction policy")
    plt.ylabel("Mean net R per activation")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.show()

    decision = pd.Series({
        "evidence label": EVIDENCE_LABEL,
        "decision": SUMMARY["decision"],
        "selected direction model": SUMMARY["selected_model"],
        "LogReg viable": SUMMARY["model_viability"]["logreg"],
        "XGBoost viable": SUMMARY["model_viability"]["xgboost"],
        "frozen source activations": SUMMARY["source_activations"],
        "scored activations": SUMMARY["scored_activations"],
        "timing frequency changed": SUMMARY["timing_frequency_changed"],
        "trade gate included": SUMMARY["trade_gate_included"],
        "forward or Q2 loaded": SUMMARY["forward_or_lockbox_loaded"],
    }, name="Notebook V frozen decision")
    display(decision.to_frame())
    print(SUMMARY["decision"])
else:
    print("Economic decision pending; forward and Q2 remain sealed.")
'''),
    ]


def build_notebook(*, output: Path = OUT, run_root: Path = RUN_ROOT) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    base, relative = _run_root_locator(output=output, run_root=Path(run_root))
    notebook = nbf.v4.new_notebook(
        cells=_cells(run_root_base=base, run_root_relative=relative),
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
    )
    nbf.validate(notebook)
    nbf.write(notebook, output)
    return output


def main() -> int:
    print(build_notebook())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_FRAME_COLUMNS",
    "EXPECTED_PARQUET_COLUMNS",
    "OUT",
    "REQUIRED_READER_ARTIFACTS",
    "RUN_ROOT",
    "build_notebook",
]
