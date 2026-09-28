"""Build and execute Notebook 04g, the paired GMADL shadow artifact reader."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "09_RQ2_F_BTC_LSTM_GMADL_shadow.ipynb"
KERNEL = {"display_name": "MSC Code", "language": "python", "name": "msc-code"}
COLAB_SETUP = """# Google Colab / local setup
import os, sys, subprocess
from pathlib import Path
if "google.colab" in sys.modules:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    CODE_ROOT = Path("/content/drive/MyDrive/msc project/code")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(CODE_ROOT),
        "catboost==1.2.10", "rapidfuzz==3.14.3"
    ])
else:
    CODE_ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "pyproject.toml").exists())
os.chdir(CODE_ROOT); sys.path.insert(0, str(CODE_ROOT)); CODE = CODE_ROOT
"""


def build_notebook(path: Path = NOTEBOOK) -> Path:
    cells = [
        nbf.v4.new_code_cell(COLAB_SETUP),
        nbf.v4.new_markdown_cell(
            """# 04g — Paired GMADL shadow for the expected-net LSTM

## tl;dr

This **development-only shadow** tests one simple change to Notebook 04f: add
a fixed differentiable directional-agreement term to the two-output LSTM. It
is a **paired GMADL shadow**, with the same rows, features, target scale,
optimizer, epochs, clipping, and **identical initial state and batch order**.

The candidate is a **replacement, not a fourth vote**. It reuses the exact 04f
XGBoost/SVM predictions and never votes beside the control LSTM. The two arms
are evaluated separately under the exact same fixed two-of-three policy.

GMADL increased trades from 147 to 163 but worsened net return from -12.83% to
-32.46% and reduced value-weighted top-quartile side-choice accuracy from
51.09% to 50.41%. It is rejected. H1 was not loaded and forward was not loaded;
Qualified Union v1 remains immutable, and the Q2-2026 lockbox remains sealed.
This notebook only verifies and reads completed artifacts."""
        ),
        nbf.v4.new_code_cell(
            """import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

CACHE = CODE_ROOT / "experiments" / "cache" / "lstm_gmadl_shadow"
CONTROL = CODE_ROOT / "experiments" / "cache" / "unified_expected_net_ensemble"
UNION = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
protocol = json.loads((CACHE / "protocol.json").read_text(encoding="utf-8"))
control_manifest = json.loads((CONTROL / "manifest.json").read_text(encoding="utf-8"))

for filename, expected in manifest["artifact_hashes"].items():
    assert sha256(CACHE / filename) == expected, filename
for filename, expected in manifest["control_dependency_hashes"].items():
    assert sha256(CONTROL / filename) == expected, filename
for filename, expected in control_manifest["union_dependency_hashes"].items():
    assert sha256(UNION / filename) == expected, filename
assert manifest["protocol_sha256"] == summary["protocol_sha256"] == protocol["protocol_sha256"]
assert manifest["control_protocol_sha256"] == summary["control_protocol_sha256"]
assert protocol["development_only"] is True
assert protocol["lstm_role"] == "replacement_not_fourth_vote"
assert protocol["h1_access_allowed"] is False
assert protocol["forward_access_allowed"] is False

print("Validated 04g shadow artifacts and exact 04f dependencies.")
print("Decision:", summary["decision"])
print("Maximum loaded timestamp:", summary["maximum_loaded_timestamp"])"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Registered paired loss

For scaled LONG/SHORT targets $y_L,y_S$ and predictions $p_L,p_S$:

`L_control = MSE(y_L, p_L) + MSE(y_S, p_S)`

`GMADL = mean(-(sigmoid((y_L-y_S)(p_L-p_S)) - 0.5) * abs(y_L-y_S))`

`L_candidate = L_control + 0.25 * GMADL`

Alpha and beta are both 1; lambda is 0.25. There is one seed and no sweep,
magnitude add-on, threshold search, or new feature. Each arm receives its own
later non-negative affine calibration. The candidate **never votes beside the
control LSTM**: each arm independently replaces the single LSTM slot in 04f."""
        ),
        nbf.v4.new_code_cell(
            """training = pd.read_csv(CACHE / "paired_training_audit.csv")
calibration = pd.read_csv(CACHE / "calibration_metrics.csv")
assert training["initial_state_match"].astype(bool).all()
assert training["batch_order_match"].astype(bool).all()
assert training["control_matches_04f"].astype(bool).all()
assert np.allclose(training["alpha"], 1.0)
assert np.allclose(training["beta"], 1.0)
assert np.allclose(training["lambda_gmadl"], 0.25)

pairing = training[[
    "fold_id", "seed", "initial_state_match", "batch_order_match",
    "target_scale_bps", "control_max_abs_delta_from_04f", "control_matches_04f",
]]
display(pairing.round(8))
print(f"Max control prediction delta: {training['control_max_abs_delta_from_04f'].max():.6g}")

calibration_summary = (
    calibration.groupby(["arm", "side"], as_index=False)
    .agg(
        calibrated_mae_bps=("calibrated_mae_bps", "mean"),
        calibrated_rmse_bps=("calibrated_rmse_bps", "mean"),
        slope=("slope", "mean"),
        intercept_bps=("intercept_bps", "mean"),
    )
)
display(calibration_summary.round(4))"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Exact control reproduction and economics

The paired control reproduces every saved 04f LSTM raw and calibrated
prediction exactly (maximum absolute delta 0), then reproduces the same 147
trade ledger and -12.83% net result. Therefore differences below come only
from the registered GMADL loss. XGBoost and SVM were not retrained, and neither
arm has an XGBoost-solo route."""
        ),
        nbf.v4.new_code_cell(
            """control = summary["control"]
candidate = summary["candidate"]
control_ledger = pd.read_parquet(CACHE / "control_trade_ledger.parquet")
candidate_ledger = pd.read_parquet(CACHE / "candidate_trade_ledger.parquet")
frozen_ledger = pd.read_parquet(CONTROL / "development_trade_ledger.parquet")

assert summary["control_reconciles_04f"] is True
assert control_ledger["row_key"].astype(str).tolist() == frozen_ledger["row_key"].astype(str).tolist()
assert np.allclose(control_ledger["net_return"], frozen_ledger["net_return"])
assert not control_ledger["route"].eq("xgboost_solo").any()
assert not candidate_ledger["route"].eq("xgboost_solo").any()
for ledger in (control_ledger, candidate_ledger):
    entry = pd.to_datetime(ledger["entry_time"], utc=True).to_numpy()
    exit_time = pd.to_datetime(ledger["actual_exit_time"], utc=True).to_numpy()
    assert not (entry[1:] <= exit_time[:-1]).any()

comparison = pd.DataFrame([
    {
        "arm": "04f MSE control", "trades": control["trades"],
        "LONG": control["long_trades"], "SHORT": control["short_trades"],
        "gross_pct": 100 * control["gross_return"], "cost_pct": 100 * control["cost_return"],
        "net_pct": 100 * control["net_return"], "trades_per_day": control["trades_per_observed_day"],
    },
    {
        "arm": "04g GMADL replacement", "trades": candidate["trades"],
        "LONG": candidate["long_trades"], "SHORT": candidate["short_trades"],
        "gross_pct": 100 * candidate["gross_return"], "cost_pct": 100 * candidate["cost_return"],
        "net_pct": 100 * candidate["net_return"], "trades_per_day": candidate["trades_per_observed_day"],
    },
])
display(comparison.round(4))
print(f"Control: {control['trades']} trades, {control['long_trades']} LONG / {control['short_trades']} SHORT, net {100 * control['net_return']:.2f}%")
print(f"Candidate: {candidate['trades']} trades, {candidate['long_trades']} LONG / {candidate['short_trades']} SHORT, net {100 * candidate['net_return']:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), layout="constrained")
comparison.plot.bar(x="arm", y=["LONG", "SHORT"], stacked=True, ax=axes[0], color=["#2a9d8f", "#e9c46a"])
comparison.plot.bar(x="arm", y=["gross_pct", "net_pct"], ax=axes[1], color=["#457b9d", "#e76f51"])
axes[0].set(title="Trade count and sides", xlabel="", ylabel="Trades")
axes[1].axhline(0, color="#333333", linewidth=.8)
axes[1].set(title="Gross and net return", xlabel="", ylabel="Percent")
for axis in axes:
    axis.grid(axis="y", alpha=.25)
    axis.tick_params(axis="x", rotation=8)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Side-choice and fold diagnostics

The top-quartile threshold is learned independently from each fold's fit-only
absolute true side spread. Accuracy is then value-weighted on untouched test
rows. This tests whether GMADL improves direction where LONG-versus-SHORT
economic separation is largest, rather than merely changing trade count."""
        ),
        nbf.v4.new_code_cell(
            """side_choice = pd.read_csv(CACHE / "side_choice_audit.csv")
fold_deltas = pd.read_csv(CACHE / "fold_deltas.csv")
admission = summary["admission"]

accuracy = pd.DataFrame([
    {"arm": "control", "value_weighted_accuracy_pct": 100 * admission["control_top_quartile_accuracy"]},
    {"arm": "candidate", "value_weighted_accuracy_pct": 100 * admission["candidate_top_quartile_accuracy"]},
])
display(accuracy.round(4))
display(fold_deltas.assign(net_delta_pct=100 * fold_deltas["net_delta"]).round(4))
print(f"Top-quartile accuracy: control {100 * admission['control_top_quartile_accuracy']:.2f}% vs candidate {100 * admission['candidate_top_quartile_accuracy']:.2f}%")
print(f"Accuracy delta: {100 * admission['top_quartile_accuracy_delta']:.2f} percentage points")
print(f"Nonnegative fold deltas: {admission['nonnegative_delta_folds']} / 5")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Admission verdict and sealed future data

The candidate independently fails the absolute 04f economics. It also fails
total, LONG, and SHORT net non-inferiority and worsens top-quartile side-choice
accuracy. Three folds have a zero delta because their selected ledgers are
unchanged; the other two deltas are negative. More trades are not evidence of
better prediction when gross return collapses before costs.

The result is **shadow reject**. It neither replaces the standard LSTM nor
enters the ensemble. H1 was not loaded; forward was not loaded. Qualified
Union v1 remains immutable for the Reflection Agent. Q2-2026 lockbox remains
sealed."""
        ),
        nbf.v4.new_code_cell(
            """gate_columns = [name for name in admission if name.endswith("_gate")]
gate_table = pd.DataFrame({"gate": gate_columns, "passed": [admission[name] for name in gate_columns]})
display(gate_table)

assert admission["control_absolute_gate"] is False
assert admission["candidate_absolute_gate"] is False
assert admission["total_net_noninferiority_gate"] is False
assert admission["long_net_noninferiority_gate"] is False
assert admission["short_net_noninferiority_gate"] is False
assert admission["top_quartile_side_choice_gate"] is False
assert admission["shadow_admit_for_reflection"] is False
assert summary["shadow_only"] is True
assert summary["h1_loaded"] is False
assert summary["forward_loaded"] is False
assert summary["lockbox_2026_q2_used"] is False
assert manifest["shadow_only"] is True
assert manifest["h1_loaded"] is False
assert manifest["forward_loaded"] is False
assert manifest["lockbox_2026_q2_used"] is False
assert not any(path.name.startswith(("h1_", "forward_")) for path in CACHE.iterdir())

print("Shadow admit for Reflection:", admission["shadow_admit_for_reflection"])
print("H1 was not loaded; forward was not loaded for 04g.")
print("Decision:", summary["decision"])
print("Fallback: standard 04f LSTM remains diagnostic; qualified_union_v1 remains final")"""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = KERNEL
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


def execute_notebook(path: Path = NOTEBOOK) -> Path:
    notebook = nbf.read(path, as_version=4)
    executed = NotebookClient(
        notebook,
        timeout=1800,
        kernel_name="msc-code",
        resources={"metadata": {"path": str(CODE_ROOT)}},
    ).execute()
    nbf.write(executed, path)
    return path


def main() -> int:
    execute_notebook(build_notebook())
    print(NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
