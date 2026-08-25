"""Build and execute Notebook 04h, the Union episode re-entry artifact reader."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "04h_union_v1_episode_reentry.ipynb"
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
            """# 04h — Union-v1-Style Episode Re-entry

## tl;dr

This is an **artifact-only reader** for one preregistered execution experiment.
It reconstructs `union_v1_style_control` on development data from **2021–2024**
using five non-overlapping 80/20 time-series folds and an eight-bar embargo.
This is not the old fitted Union: the **immutable Union v1** remains the external
reference and is neither retrained nor overwritten.

Both members receive the same **64 causal features**. These explicitly include
**funding rate and open interest**, positioning availability/staleness, order
flow, volatility and the transferred channel features. The LSTM uses a 32×64
causal window; Linear SVM uses the current 64-D row. There is **no XGBoost**.

The candidate changes only scheduling: **one earliest extra per same-side episode**,
selected from a qualified bar skipped by the frozen control ledger.
It increased completed trades from 948 to 1,231 (+29.85%), but the extra route
lost 20.58% net after costs and weakened both LONG and SHORT economics. The
candidate therefore fails development. **H1 was not loaded; forward was not
loaded.** Immutable Union v1 remains final and the **Q2-2026 lockbox remains
sealed**."""
        ),
        nbf.v4.new_code_cell(
            """import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

CACHE = CODE_ROOT / "experiments" / "cache" / "union_v1_episode_reentry"
UNION = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

summary = json.loads((CACHE / "summary.json").read_text(encoding="utf-8"))
manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
stage_manifest = json.loads((CACHE / "development_artifacts.json").read_text(encoding="utf-8"))
protocol = json.loads((CACHE / "frozen_protocol.json").read_text(encoding="utf-8"))

for filename, expected in manifest["artifact_hashes"].items():
    assert sha256(CACHE / filename) == expected, filename
for filename, expected in stage_manifest["artifact_hashes"].items():
    assert sha256(CACHE / filename) == expected, filename
for filename, expected in manifest["union_dependency_hashes"].items():
    assert sha256(UNION / filename) == expected, filename
assert manifest["protocol_sha256"] == protocol["protocol_sha256"]
assert protocol["policy_grid"] == []
assert protocol["models"] == ["lstm_dz55", "svm_linear_dz75"]
assert len(protocol["features"]) == 64
assert {"funding_rate", "funding_z", "oi_chg_15m", "oi_chg_1h", "oi_chg_4h", "oi_z"}.issubset(protocol["features"])
assert manifest["lockbox_2026_q2_used"] is False

print("Validated 04h artifacts, frozen protocol, and immutable Union dependencies.")
print("Decision:", summary["decision"])
print("Maximum loaded timestamp:", summary["maximum_loaded_timestamp"])"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Fixed development protocol

The experiment asks one narrow question: can a second entry inside a signal
episode increase frequency by at least 15% without reducing net return overall,
for LONG, or for SHORT?

- Development is `[2021-01-01, 2025-01-01)` with five independent blocked
  80/20 folds and an eight-M15-bar embargo.
- LSTM predicts the next-M15 DZ55 class with sequence 32, hidden size 64, ten
  epochs, seed 42 and directional confidence ≥0.75.
- Linear SVM predicts DZ75 with `C=0.1`; its frozen threshold is zero, so the
  predicted class controls abstention.
- Active members must agree; an opposite pair is vetoed to WAIT.
- Both arms use TP 200 bps, SL 100 bps, one-M15-bar hold and 5 bps per side.
- There is one candidate, no threshold search and no policy grid."""
        ),
        nbf.v4.new_code_cell(
            """fold_manifest = pd.read_parquet(CACHE / "development_fold_manifest.parquet")
training = pd.read_csv(CACHE / "development_training_audit.csv")
fold_rows = []
for fold_id, fold in fold_manifest.groupby("fold_id", sort=True):
    fit = fold.loc[fold["role"].eq("fit")]
    test = fold.loc[fold["role"].eq("test")]
    assert fit["position"].max() + 8 < test["position"].min()
    assert fit["union_target_time"].max() < test["decision_time"].min()
    fold_rows.append({
        "fold": int(fold_id), "fit_rows": len(fit), "test_rows": len(test),
        "fit_end": fit["decision_time"].max(), "test_start": test["decision_time"].min(),
    })
assert fold_manifest["fold_id"].nunique() == 5
assert fold_manifest.loc[fold_manifest["role"].eq("test"), "row_key"].is_unique
assert training["preprocessing_fit_only"].astype(bool).all()
assert training["lstm_probabilities_finite"].astype(bool).all()

feature_groups = pd.DataFrame([
    {"group": "Market / order flow", "examples": "returns, ranges, volume, OFI, taker imbalance"},
    {"group": "Funding / open interest", "examples": "funding_rate/z, OI changes/acceleration/z, price×OI, missing/staleness"},
    {"group": "Risk / transferred channels", "examples": "realised volatility, semivariance, drawdown, channel position/slope, wicks"},
])
display(pd.DataFrame(fold_rows))
display(feature_groups)
print(f"64 causal features include funding rate and open interest; LSTM input is 32×64, SVM input is 64-D.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 2. Paired scheduling and trade funnel

An episode is a maximal consecutive run of one non-zero Union side. The control
ledger is frozen first. The candidate may then accept the earliest qualified
signal bar skipped by control, at most once per episode, provided the next-bar
entry does not overlap any control or prior extra. Selection uses timestamps,
side and occupancy only—never realised PnL or a future confidence value.

The control ledger remains an exact subset of the candidate ledger. Every entry
is replayed on its native one-minute path; a same-minute TP/SL ambiguity is
stop-first, otherwise first touch wins."""
        ),
        nbf.v4.new_code_cell(
            """predictions = pd.read_parquet(CACHE / "development_oof_predictions.parquet")
signals = pd.read_parquet(CACHE / "development_signals.parquet")
episodes = pd.read_parquet(CACHE / "development_episode_table.parquet")
selected = pd.read_parquet(CACHE / "development_selected_reentries.parquet")
control_ledger = pd.read_parquet(CACHE / "development_control_ledger.parquet")
reentry_ledger = pd.read_parquet(CACHE / "development_reentry_ledger.parquet")
candidate_ledger = pd.read_parquet(CACHE / "development_candidate_ledger.parquet")
development = summary["development"]

probability = predictions[["p_short_lstm", "p_flat_lstm", "p_long_lstm"]].to_numpy(float)
assert predictions["row_key"].is_unique and np.isfinite(probability).all()
assert np.allclose(probability.sum(axis=1), 1.0)
assert selected.groupby("episode_id").size().max() <= 1
assert candidate_ledger["entry_time"].is_unique
assert candidate_ledger["trade_key"].is_unique
assert set(control_ledger["trade_key"]).issubset(set(candidate_ledger["trade_key"]))
assert set(reentry_ledger["signal_time"]) == set(selected["signal_time"])

funnel = pd.DataFrame([
    {"arm": "union_v1_style_control", "trades": len(control_ledger), "trades_per_day": development["control_trades_per_day"]},
    {"arm": "episode_reentry_candidate", "trades": len(candidate_ledger), "trades_per_day": development["candidate_trades_per_day"]},
])
display(funnel.round(3))
print(f"OOF exposure: {len(predictions):,} M15 rows = {development['evaluation_days']:.2f} effective days")
print(f"Control: {len(control_ledger):,} trades at {development['control_trades_per_day']:.3f}/day")
print(f"Candidate: {len(candidate_ledger):,} trades at {development['candidate_trades_per_day']:.3f}/day")
print(f"Increase: +{development['trade_count_increase']:,} trades ({100 * development['trade_count_increase_fraction']:.2f}%), {development['control_trades_per_day']:.3f} -> {development['candidate_trades_per_day']:.3f} trades/day")
print(f"Candidate sides: {development['candidate']['long_trades']} LONG / {development['candidate']['short_trades']} SHORT")"""
        ),
        nbf.v4.new_markdown_cell(
            """## 3. Economics: frequency passed, quality failed

The candidate easily passes the count requirement (1,231 versus required
1,091), and both directions have ample trades. That is only the frequency gate.
The reconstructed control has no gross edge before costs, while the extra route
has a small positive gross return that is far below its 10-bps round-trip cost.
Consequently, total, LONG, SHORT and fold-stability gates all fail."""
        ),
        nbf.v4.new_code_cell(
            """control = development["control"]
candidate = development["candidate"]
incremental = development["incremental"]
fold_metrics = pd.read_csv(CACHE / "development_fold_metrics.csv")

economics = pd.DataFrame([
    {"route": "control", "trades": control["trades"], "LONG": control["long_trades"], "SHORT": control["short_trades"], "gross_pct": 100*control["gross_return"], "cost_pct": 100*control["cost_return"], "net_pct": 100*control["net_return"]},
    {"route": "extra only", "trades": incremental["trades"], "LONG": incremental["long_trades"], "SHORT": incremental["short_trades"], "gross_pct": 100*incremental["gross_return"], "cost_pct": 100*incremental["cost_return"], "net_pct": 100*incremental["net_return"]},
    {"route": "candidate", "trades": candidate["trades"], "LONG": candidate["long_trades"], "SHORT": candidate["short_trades"], "gross_pct": 100*candidate["gross_return"], "cost_pct": 100*candidate["cost_return"], "net_pct": 100*candidate["net_return"]},
])
display(economics.round(2))
display(fold_metrics.assign(
    control_net_pct=100*fold_metrics["control_net_return"],
    incremental_net_pct=100*fold_metrics["incremental_net_return"],
    candidate_net_pct=100*fold_metrics["candidate_net_return"],
)[["fold_id", "control_trades", "reentry_trades", "control_net_pct", "incremental_net_pct", "candidate_net_pct"]].round(2))

print(f"Control net: {100*control['net_return']:.2f}% (gross {100*control['gross_return']:.2f}%)")
print(f"Incremental extras: gross {100*incremental['gross_return']:+.2f}%, cost {100*incremental['cost_return']:.2f}%, net {100*incremental['net_return']:.2f}%")
print(f"Candidate net: {100*candidate['net_return']:.2f}%")
print(f"Candidate LONG net: {100*candidate['long_net_return']:.2f}%; SHORT net: {100*candidate['short_net_return']:.2f}%")

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), layout="constrained")
economics.plot.bar(x="route", y=["LONG", "SHORT"], stacked=True, ax=axes[0], color=["#2a9d8f", "#e9c46a"])
economics.plot.bar(x="route", y=["gross_pct", "net_pct"], ax=axes[1], color=["#457b9d", "#e76f51"])
axes[0].set(title="Completed trades by side", xlabel="", ylabel="Trades")
axes[1].axhline(0, color="#333333", linewidth=.8)
axes[1].set(title="Gross versus net return", xlabel="", ylabel="Percent")
for axis in axes:
    axis.grid(axis="y", alpha=.25)
    axis.tick_params(axis="x", rotation=8)
plt.show()"""
        ),
        nbf.v4.new_markdown_cell(
            """## 4. Registered decision and data boundary

This is a clean negative experiment: the scheduling rule found more market
activity, but it did not find more profitable activity. Relaxing thresholds,
changing barriers, filtering sides after seeing outcomes, or adding XGBoost now
would turn the registered one-candidate test into an unreported search.

Therefore the outcome is `development_fail_keep_union_v1`. H1 was not loaded,
forward was not loaded, and no timestamp at or after 2026-04-01 was accessed.
The **immutable Union v1** remains the ensemble handed to the Reflection Agent;
the **Q2-2026 lockbox remains sealed**."""
        ),
        nbf.v4.new_code_cell(
            """gates = development["gates"]
gate_table = pd.DataFrame(
    [(name, value) for name, value in gates.items() if isinstance(value, bool)],
    columns=["gate", "passed"],
)
display(gate_table)

assert gates["frequency_gain"] is True
assert gates["control_total_positive"] is False
assert gates["candidate_total_noninferior"] is False
assert gates["candidate_long_noninferior"] is False
assert gates["candidate_short_noninferior"] is False
assert gates["incremental_total_nonnegative"] is False
assert gates["development_pass"] is False
assert summary["decision"] == "development_fail_keep_union_v1"
assert summary["h1_loaded"] is False
assert summary["forward_loaded"] is False
assert summary["lockbox_2026_q2_used"] is False
assert not any(path.name.startswith(("h1_", "forward_")) for path in CACHE.iterdir())

print("Development pass:", gates["development_pass"])
print("H1 loaded:", summary["h1_loaded"])
print("Forward loaded:", summary["forward_loaded"])
print("Lockbox used:", summary["lockbox_2026_q2_used"])
print("Decision:", summary["decision"])
print("Fallback: immutable qualified_union_v1 remains final for the Reflection Agent")"""
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
