"""Build the artifact-only Notebook U volatility/timing consolidation report."""
from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

import nbformat as nbf

from experiments.run_event_window_feature_consolidation import (
    FROZEN_P_RUN_HASH,
    FROZEN_R_RUN_HASH,
    READER_ARTIFACTS,
    RUN_ROOT,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
OUT = CODE_ROOT / "notebooks" / "U_volatility_timing_feature_consolidation.ipynb"
EXPECTED_FRAME_COLUMNS: dict[str, tuple[str, ...]] = {
    "compact_feature_audit.csv": (
        "feature",
        "source_type",
        "included",
        "finite_fraction",
        "missing_fraction",
        "known_at_decision_time",
        "direction_invariant",
    ),
    "feature_redundancy_audit.csv": (
        "scope",
        "pairs_abs_spearman_ge_0_90",
        "features_in_pairs",
        "maximum_abs_spearman",
        "diagnostic_only",
    ),
    "base_reproduction_audit.csv": (
        "source",
        "model",
        "row_identity",
        "max_abs_difference",
        "required",
    ),
    "predictive_metrics.csv": (
        "arm",
        "model",
        "feature_set",
        "fold_id",
        "weighted_brier",
        "weighted_log_loss",
    ),
    "predictive_deltas.csv": (
        "model",
        "fold_id",
        "brier_improvement",
        "log_loss_improvement",
        "nonnegative_brier_folds",
    ),
    "economic_metrics.csv": (
        "arm",
        "scenario",
        "observed_trades",
        "activations_per_calendar_day",
        "path_completeness",
        "mean_net_r",
        "net_mean_r_ci_low",
        "net_mean_r_ci_high",
    ),
    "feature_comparisons.csv": (
        "comparison",
        "candidate_arm",
        "control_arm",
        "scenario",
        "delta_mean_net_r",
        "delta_mean_net_r_ci_low",
        "delta_mean_net_r_ci_high",
    ),
    "frequency_audit.csv": (
        "arm",
        "activations",
        "activations_per_calendar_day",
    ),
    "leakage_audit.csv": ("check", "passed", "detail"),
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
import os, sys
from pathlib import Path

if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CODE_ROOT = Path("/content/drive/MyDrive/msc project/code")
else:
    execution_root = Path.cwd().resolve()
    CODE_ROOT = next(
        (candidate for candidate in (execution_root, *execution_root.parents)
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
import pandas as pd
from IPython.display import display

RUN_ROOT_BASE = {run_root_base!r}
RUN_ROOT_RELATIVE = {run_root_relative!r}
if RUN_ROOT_BASE == "code":
    RUN_ROOT = (CODE_ROOT / RUN_ROOT_RELATIVE).resolve()
else:
    RUN_ROOT = (Path.cwd().resolve() / RUN_ROOT_RELATIVE).resolve()

EXPECTED_FRAME_COLUMNS = {EXPECTED_FRAME_COLUMNS!r}
REQUIRED_READER_ARTIFACTS = {REQUIRED_READER_ARTIFACTS!r}
FROZEN_P_RUN_HASH = {FROZEN_P_RUN_HASH!r}
FROZEN_R_RUN_HASH = {FROZEN_R_RUN_HASH!r}
DERIVED_FEATURES = {{
    "channel_center_distance",
    "nearest_rail_distance_bps",
    "rail_approach_15m",
}}

def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _load_completed_run():
    pointer_path = RUN_ROOT / "latest_dev.json"
    if not pointer_path.is_file():
        return None, None, None, {{}}, None, None, "latest_dev.json is missing"
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        run_hash = str(pointer.get("run_hash", ""))
        protocol_hash = str(pointer.get("protocol_hash", ""))
        relative_path = Path(str(pointer.get("relative_path", "")))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("pointer path is unsafe")
        run_dir = (RUN_ROOT / relative_path).resolve()
        if run_dir != (RUN_ROOT / run_hash / "full").resolve():
            raise ValueError("pointer does not identify the matching full run")

        state = json.loads((run_dir / "run_state.json").read_text(encoding="utf-8"))
        if state.get("status") != "complete":
            raise ValueError("run state is not complete")
        if state.get("run_hash") != run_hash or state.get("protocol_hash") != protocol_hash:
            raise ValueError("run state identity does not match the pointer")
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
            raise ValueError("summary does not match completed run state")
        if frozen.get("frozen_p_run_hash") != FROZEN_P_RUN_HASH:
            raise ValueError("frozen Notebook P identity changed")
        if frozen.get("frozen_r_run_hash") != FROZEN_R_RUN_HASH:
            raise ValueError("frozen Notebook R identity changed")
        for payload in (protocol, frozen, summary):
            if payload.get("forward_or_lockbox_loaded") is not False:
                raise ValueError("later-period data are not accepted")
        if summary.get("p_hit_frozen") is not True:
            raise ValueError("Notebook U changed frozen p_hit")
        if summary.get("timing_model_refit") is not True:
            raise ValueError("Notebook U timing refit is incomplete")
        if summary.get("calendar_features_included") is not False:
            raise ValueError("Notebook U must keep calendar excluded")
        if summary.get("impulse_features_included") is not False:
            raise ValueError("Notebook U must keep impulse excluded")
        if summary.get("economics_evaluated") is not True:
            raise ValueError("Notebook U economics are incomplete")
        if summary.get("base_feature_count") != 248 or summary.get("compact_feature_count") != 28:
            raise ValueError("Notebook U feature counts changed")
        if summary.get("derived_feature_count") != 3:
            raise ValueError("Notebook U derived-feature count changed")

        frames = {{}}
        for name, required_columns in EXPECTED_FRAME_COLUMNS.items():
            frame = pd.read_csv(run_dir / name)
            missing = [column for column in required_columns if column not in frame]
            if missing:
                raise ValueError(f"artifact schema does not match: {{name}}")
            frames[name] = frame
        feature_audit = frames["compact_feature_audit.csv"]
        if len(feature_audit) != 28 or set(
            feature_audit.loc[feature_audit["source_type"].eq("derived"), "feature"]
        ) != DERIVED_FEATURES:
            raise ValueError("compact feature audit does not match the fixed contract")
        for column in ("included", "known_at_decision_time", "direction_invariant"):
            if not feature_audit[column].astype(bool).all():
                raise ValueError(f"compact feature audit failed: {{column}}")
        if not frames["leakage_audit.csv"]["passed"].astype(bool).all():
            raise ValueError("leakage audit contains a failed check")

        predictions = pd.read_parquet(run_dir / "oof_predictions.parquet")
        ledger = pd.read_parquet(run_dir / "activation_ledger.parquet")
        for name, frame, columns in (
            ("OOF predictions", predictions, ("arm", "fold_id", "p_t_le_60")),
            ("activation ledger", ledger, ("activation_key", "arm", "decision_time")),
        ):
            missing = [column for column in columns if column not in frame]
            if missing:
                raise ValueError(f"{{name}} schema does not match")
        # JSON stores metadata only; analytical rows remain tabular.
        return run_dir, protocol, summary, frames, predictions, ledger, None
    except Exception as error:
        return None, None, None, {{}}, None, None, str(error)

RUN_DIR, PROTOCOL, SUMMARY, FRAMES, PREDICTIONS, LEDGER, READER_NOTE = _load_completed_run()
READER_READY = RUN_DIR is not None
if READER_READY:
    print(f"Loaded completed Notebook U run: {{RUN_DIR.name}}")
else:
    print(f"Completed Notebook U artifacts are not available: {{READER_NOTE}}.")
    print("Run: python -m experiments.run_event_window_feature_consolidation --stage dev")
'''
    return [
        _code(bootstrap),
        nbf.v4.new_markdown_cell(
            "# Notebook U — Volatility/Timing Feature Consolidation\n\n"
            "A paired development-only test of whether a fixed 28-feature domain "
            "contract can replace the frozen 248-feature timing inputs without "
            "losing predictive quality, trade frequency, or net economics."
        ),
        _code(setup),
        _section(
            "## 1. Frozen handoff and research question\n\n"
            "Notebook P supplies frozen $p_{hit}$; only the h15, h30 and h60 timing "
            "heads are refit. LogReg and XGBoost each compare the same OOF rows under "
            "248 base features and 28 compact features. The policy remains level re-arm "
            "with a 60-minute cooldown and a target near three activations per day."
        ),
        _code('''\
if READER_READY:
    handoff = pd.Series({
        "development rows": SUMMARY["decision_rows"],
        "OOF folds": SUMMARY["folds"],
        "base features": SUMMARY["base_feature_count"],
        "compact features": SUMMARY["compact_feature_count"],
        "p_hit frozen": SUMMARY["p_hit_frozen"],
        "timing heads refit": ", ".join(PROTOCOL["timing_heads_refit"]),
        "policy": PROTOCOL["alert_policy"],
        "later periods loaded": SUMMARY["forward_or_lockbox_loaded"],
    }, name="Frozen development protocol")
    display(handoff.to_frame())
else:
    print("Awaiting completed development artifacts.")
'''),
        _section(
            "## 2. Exact 28-feature contract and exclusions\n\n"
            "The candidate contains 25 native volatility/activity/timing fields plus "
            "three row-causal transforms: `channel_center_distance`, "
            "`nearest_rail_distance_bps`, and `rail_approach_15m`. Calendar excluded; "
            "impulse excluded. Direction, RSI, OI, funding, positioning and sentiment "
            "are also excluded because this head answers only *how large and how soon*."
        ),
        _code('''\
if READER_READY:
    feature_audit = FRAMES["compact_feature_audit.csv"].copy()
    display(feature_audit[[
        "feature", "source_type", "finite_fraction", "missing_fraction",
        "known_at_decision_time", "direction_invariant",
    ]])
    print(f"Contract: {len(feature_audit)} features; derived: "
          f"{int(feature_audit['source_type'].eq('derived').sum())}.")
else:
    print("The exact feature table will appear after the completed run.")
'''),
        _section(
            "## 3. Causality, symmetry and effective sample\n\n"
            "All inputs are known at decision time, derived transforms are invariant "
            "to long/short reflection, missing history remains explicit, and imputation "
            "is fold-local. Seven expanding episode-purged OOF folds prevent one channel "
            "episode from appearing on both sides. Correlation is diagnostic only: no "
            "feature is selected using outcomes."
        ),
        _code('''\
if READER_READY:
    display(FRAMES["feature_redundancy_audit.csv"])
    reproduction = FRAMES["base_reproduction_audit.csv"]
    display(reproduction[[
        "source", "model", "row_identity", "max_abs_difference", "required"
    ]])
    leakage = FRAMES["leakage_audit.csv"]
    print(f"Leakage checks passed: {int(leakage['passed'].astype(bool).sum())}/{len(leakage)}")
else:
    print("Causality and reproduction audits will appear after the completed run.")
'''),
        _section(
            "## 4. Paired timing results\n\n"
            "Primary predictive metrics are uniqueness-weighted Brier score and log "
            "loss for $P(T\\leq60\\mid hit)$. A compact model must improve both overall "
            "metrics and avoid a negative Brier delta in at least four of seven folds. "
            "Accuracy and win rate do not select the timing head."
        ),
        _code('''\
if READER_READY:
    metrics = FRAMES["predictive_metrics.csv"]
    overall = metrics.loc[metrics["fold_id"].astype(str).eq("overall")]
    display(overall[[
        "arm", "model", "feature_set", "weighted_brier", "weighted_log_loss"
    ]].sort_values(["model", "feature_set"]))
    deltas = FRAMES["predictive_deltas.csv"]
    display(deltas.loc[deltas["fold_id"].astype(str).eq("overall")])
    chart = overall.pivot(index="model", columns="feature_set", values="weighted_brier")
    chart.plot(kind="bar", figsize=(7, 3.5), title="OOF timing Brier score (lower is better)")
    plt.ylabel("Weighted Brier score")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
else:
    print("Paired LogReg and XGBoost results are pending.")
'''),
        _section(
            "## 5. Level re-arm frequency and economics\n\n"
            "Economic replay keeps RR2, a 120-minute hold, native 1-minute paths and "
            "5/2/5 bps entry/target/other costs. The primary causal side is the channel "
            "side. The direction-70 result is a stress test only, never a trained "
            "direction claim. Frequency must remain between 2.5 and 3.5 activations per day."
        ),
        _code('''\
if READER_READY:
    frequency = FRAMES["frequency_audit.csv"].copy()
    display(frequency[["arm", "activations", "activations_per_calendar_day"]])
    economics = FRAMES["economic_metrics.csv"].copy()
    display(economics[[
        "arm", "scenario", "observed_trades", "activations_per_calendar_day",
        "path_completeness", "mean_net_r", "net_mean_r_ci_low", "net_mean_r_ci_high",
    ]].sort_values(["scenario", "arm"]))
    comparisons = FRAMES["feature_comparisons.csv"]
    display(comparisons[[
        "comparison", "scenario", "delta_mean_net_r",
        "delta_mean_net_r_ci_low", "delta_mean_net_r_ci_high",
    ]])
    frequency.set_index("arm")["activations_per_calendar_day"].plot(
        kind="bar", figsize=(7, 3.5), title="Level re-arm activation frequency"
    )
    plt.axhspan(2.5, 3.5, color="green", alpha=0.12, label="admissible range")
    plt.ylabel("Activations per calendar day")
    plt.xticks(rotation=25, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("Frequency and economic replay are pending.")
'''),
        _section(
            "## 6. Decision\n\n"
            "Replacement is deliberately strict: both predictive losses must improve, "
            "at least four folds must have non-negative Brier improvement, frequency "
            "and path completeness must pass, base reproduction and leakage audits must "
            "pass, and the episode-bootstrap lower bound of compact-minus-base net R must "
            "be above zero. Forward and Q2 remain sealed regardless of the result."
        ),
        _code('''\
if READER_READY:
    decision = pd.Series({
        "compact promoted": SUMMARY["compact_promoted"],
        "decision": SUMMARY["decision"],
        "XGBoost Brier improvement": SUMMARY["xgboost_brier_improvement"],
        "XGBoost log-loss improvement": SUMMARY["xgboost_log_loss_improvement"],
        "non-negative Brier folds": SUMMARY["xgboost_nonnegative_brier_folds"],
        "compact minus base mean net R": SUMMARY["xgboost_compact_minus_base_mean_r"],
        "episode CI low": SUMMARY["xgboost_compact_minus_base_ci_low"],
        "episode CI high": SUMMARY["xgboost_compact_minus_base_ci_high"],
        "base reproduced": SUMMARY["base_reproduction_passed"],
        "forward or Q2 loaded": SUMMARY["forward_or_lockbox_loaded"],
    }, name="Notebook U decision")
    display(decision.to_frame())
else:
    print("Decision pending one completed development run; forward and Q2 remain sealed.")
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
    "OUT",
    "REQUIRED_READER_ARTIFACTS",
    "RUN_ROOT",
    "build_notebook",
]
