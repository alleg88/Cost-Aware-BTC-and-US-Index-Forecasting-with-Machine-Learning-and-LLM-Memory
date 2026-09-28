"""Rebuild Notebook 02b as the next stage after Notebook 02."""
from __future__ import annotations

from pathlib import Path

import nbformat
import pandas as pd

from experiments.notebook02_handoff import MATCHED_ROOT
from experiments.notebook02b_handoff import (
    load_notebook02b_handoff,
    write_notebook02b_handoff,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb"
ARM_NAMES_FOR_BUILD = {
    "baseline": "Fixed baseline",
    "F1": "F1-tuned",
    "economic": "Economic-tuned",
}


def main() -> int:
    handoff_path = write_notebook02b_handoff()
    downstream_handoff = load_notebook02b_handoff(handoff_path)
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    forward = pd.read_parquet(MATCHED_ROOT / "forward_summary.parquet")
    best = forward.sort_values(["sortino", "sharpe", "net_return"], ascending=False).iloc[0]
    leaders = (
        forward.sort_values(["width_bps", "sortino", "sharpe", "net_return"], ascending=[True, False, False, False])
        .groupby("width_bps", sort=True)
        .first()
    )
    leader_text = "; ".join(
        f"DZ{width}: {ARM_NAMES_FOR_BUILD[str(row.objective)]} "
        f"({row.net_return:+.2%} net, {row.sortino:+.3f} Sortino)"
        for width, row in leaders.iterrows()
    )
    notebook.cells[0].source = """# 02b - Matched CatBoost objective comparison

## Notebook 01 handoff

This notebook continues Notebook 01. It reads the direct `handoff.json` and uses exactly **180 days for DZ55, DZ65 and DZ75** in monthly H1 fits and the frozen-forward fit. Results are shown only after the handoff and methodology are validated; 2026 Q2 remains sealed."""

    notebook.cells[1].source = """from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

CURRENT = Path.cwd()
CODE_ROOT = CURRENT if (CURRENT / "experiments").exists() else CURRENT.parent
sys.path.insert(0, str(CODE_ROOT))

from experiments.notebook02_handoff import (
    MATCHED_ROOT,
    PIPELINE_HANDOFF,
    load_pipeline_handoff,
)
from experiments.notebook02b_handoff import HANDOFF_PATH, load_notebook02b_handoff

ARTIFACT_ROOT = MATCHED_ROOT
upstream = load_pipeline_handoff(PIPELINE_HANDOFF)
downstream = load_notebook02b_handoff(HANDOFF_PATH)
HISTORY = {int(width): int(days) for width, days in upstream["training_histories_days"].items()}
ARM_NAMES = {"baseline": "Fixed baseline", "F1": "F1-tuned", "economic": "Economic-tuned"}

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option("display.max_columns", 40)
pd.set_option("display.width", 180)"""

    notebook.cells[2].source = """## Context & Methods

This matched CatBoost objective comparison uses a monthly walk-forward for H1. All arms share the same 24 price, order-flow and positioning features, labels, five `BlockingTimeSeriesSplit` folds, fees, 1-minute path execution and policy grid.

### Models and selection

The **Fixed baseline** is pre-declared candidate 0: `iterations=300`, `depth=6`, `learning_rate=0.10`, `l2_leaf_reg=3.0`, balanced class weights, seed 42 and multiclass loss. The other 14 configurations came from one seeded Optuna `RandomSampler`; the resulting **15-candidate pool** was frozen before F1 or economic selection, and both tuned arms use this same pool.

### Monthly H1 walk-forward

The 2024 blocking-CV artifacts are **validated and reused**.

1. **2024 candidate selection**: per DZ, F1-tuned ranks the **15 CatBoost candidates** by weakest-regime F1 across five folds, then overall macro-F1. Economic-tuned evaluates the same candidates with **33 temporary policies**: 11 confidence thresholds and TP/SL 150/75, 150/100 or 200/100, held for one bar. It first minimizes failures of **50 total trades**, **15 long and 15 short trades**, and positive net return in **four of five validation folds**; it then ranks robust regime Sortino, pooled Sortino, Sharpe, net return and trades. **2024 candidate selection is not H1 calibration**.
2. **2025 H1 policy calibration**: **one fit per H1 month** is made before that month using the **previous 180 days** for every DZ; each fit predicts only the next month. Confidence is the largest of CatBoost's three class probabilities; below the tested threshold no trade opens. The same H1 predictions replay all **33 policies**: 11 thresholds (`0`, `0.35`-`0.80`) x three TP/SL pairs. Selection first minimizes shortages against 50 trades, 15 per side and four positive months, then ranks worst-regime economics, pooled Sortino, net return and trades. Rows require **complete features**, a **valid label** and a **known past-only regime**; the **final training row** is trimmed. H1 **does not retune CatBoost hyperparameters**.
3. **July 2025-March 2026 frozen forward**: each selected candidate is fitted once on 1 July 2025 with the previous 180 days. Model, threshold and TP/SL remain frozen through March 2026.
4. **2026 Q2 is sealed**: M15 and 1-minute inputs stop before 1 April 2026.

The outputs contain **9 selected candidates**, **9 calibrated policies**, nine forward summaries and **27 quarters**. H1 is calibration evidence; the nine-month period is development-forward evidence; the 2024 search is selection evidence."""

    notebook.cells[3].source = """### 1. Load and validate matched artifacts

This validates the Notebook 01 fingerprint, fixed histories, feature schema, exact artifact counts, and sealed boundary before any result is displayed."""

    notebook.cells[4].source = """artifact_rows = {
    "selected_candidates_2024.parquet": 9,
    "economic_candidate_winners_2024.parquet": 45,
    "selected_policies_2025h1.parquet": 9,
    "forward_summary.parquet": 9,
    "forward_quarterly.parquet": 27,
    "forward_monthly.parquet": 81,
}
artifacts = {}
for name, expected_rows in artifact_rows.items():
    frame = pd.read_parquet(ARTIFACT_ROOT / name)
    assert len(frame) == expected_rows, (name, len(frame), expected_rows)
    artifacts[name] = frame

selected_candidates = artifacts["selected_candidates_2024.parquet"]
candidate_economics = artifacts["economic_candidate_winners_2024.parquet"]
selected_policies = artifacts["selected_policies_2025h1.parquet"]
forward_summary = artifacts["forward_summary.parquet"]
forward_quarterly = artifacts["forward_quarterly.parquet"]
forward_monthly = artifacts["forward_monthly.parquet"]
manifest = json.loads((ARTIFACT_ROOT / "manifest.json").read_text(encoding="utf-8"))

assert manifest["upstream_notebook02_handoff_fingerprint"] == upstream["handoff_fingerprint"]
assert manifest["training_histories_days"] == upstream["training_histories_days"]
assert manifest["feature_columns"] == upstream["features"]["columns"]
assert manifest["sealed_lockbox"] is True
assert manifest["lockbox_start"].startswith("2026-04-01")
assert manifest["calibration"]["method"] == "monthly_walk_forward"
assert manifest["calibration"]["month_count"] == 6
assert len(manifest["calibration"]["physical_fit_ids"]) == 54
assert forward_monthly.groupby(["objective", "width_bps"]).size().eq(9).all()
assert forward_quarterly.groupby(["objective", "width_bps"]).size().eq(3).all()
print("Validated matched artifacts:", artifact_rows)
print("Training histories:", upstream["training_histories_days"])
print("Lockbox:", manifest["lockbox_start"], "sealed =", manifest["sealed_lockbox"])"""

    notebook.cells[5].source = """## Results

### 2. 2024 hyperparameter selection — not H1 calibration

**Selection rule:** Fixed baseline keeps candidate 0; F1-tuned selects highest Robust F1 then Overall macro-F1; Economic-tuned selects the guarded 2024 economic rank. Net, Sortino, Sharpe and Trades are each candidate's best pooled 2024 screening economics; the fixed 180-day operational history is not the CV window."""

    notebook.cells[6].source = """candidate_table = (
    selected_candidates
    .merge(
        candidate_economics[[
            "width_bps", "candidate_id", "trades", "pooled_net", "pooled_sortino", "pooled_sharpe",
        ]],
        on=["width_bps", "candidate_id"],
        how="left",
        validate="one_to_one",
    )
    .assign(
        Objective=lambda frame: frame["objective"].map(ARM_NAMES),
        **{
            "Selection data": "Full 2024 blocking CV",
            "Operational history": lambda frame: frame["width_bps"].map(HISTORY).map(lambda days: f"{days}D"),
        },
    )
    [[
        "Objective", "width_bps", "Operational history", "candidate_id", "overall_f1", "robust_f1",
        "pooled_net", "pooled_sortino", "pooled_sharpe", "trades",
    ]]
    .rename(columns={
        "width_bps": "DZ (bps)",
        "candidate_id": "Candidate",
        "overall_f1": "Overall macro-F1",
        "robust_f1": "Robust F1",
        "pooled_net": "Net return",
        "pooled_sortino": "Sortino",
        "pooled_sharpe": "Sharpe",
        "trades": "Trades",
    })
    .sort_values(["DZ (bps)", "Objective"])
    .reset_index(drop=True)
)
candidate_table[["Overall macro-F1", "Robust F1"]] = candidate_table[["Overall macro-F1", "Robust F1"]].round(4)
candidate_table[["Sortino", "Sharpe"]] = candidate_table[["Sortino", "Sharpe"]].round(3)
candidate_table["Trades"] = candidate_table["Trades"].astype(int)
candidate_table["Net return"] = candidate_table["Net return"].map(lambda value: f"{100 * value:.2f}%")
display(candidate_table)"""

    notebook.cells[7].source = """### 3. 2025 H1 execution-policy calibration

Each selected candidate makes six causal monthly predictions using the previous 180 days. For Fixed baseline, the grid selects DZ55: confidence 0.75, TP/SL 150/100 bps; DZ65: 0.70 and 150/100; DZ75: 0.70 and 200/100. These are external execution rules, not CatBoost hyperparameters.

They are used only for Notebook 02b's calibrated forward comparison. Notebook 02c documents the downstream feature inputs; Notebook 02d receives **baseline candidate 0 without the H1 policy**: it keeps the class probabilities but applies no confidence gate and no TP/SL, using a raw 15-minute hold so the downstream feature layer is the only change."""

    notebook.cells[8].source = """calibration_table = (
    selected_policies
    .assign(
        Objective=lambda frame: frame["objective"].map(ARM_NAMES),
        **{"Training history": lambda frame: frame["width_bps"].map(HISTORY).map(lambda days: f"{days}D")},
    )
    [[
        "Objective", "width_bps", "Training history", "candidate_id", "policy_id", "tau", "tp_bps", "sl_bps",
        "pooled_net", "pooled_sortino", "pooled_sharpe", "trades", "n_long", "n_short", "positive_segments",
    ]]
    .rename(columns={
        "width_bps": "DZ (bps)", "candidate_id": "Candidate",
        "policy_id": "Policy", "tau": "Confidence", "tp_bps": "TP (bps)", "sl_bps": "SL (bps)",
        "pooled_net": "Net return", "pooled_sortino": "Sortino", "pooled_sharpe": "Sharpe",
        "trades": "Trades", "n_long": "Long trades", "n_short": "Short trades",
        "positive_segments": "Positive months",
    })
    .sort_values(["DZ (bps)", "Objective"])
    .reset_index(drop=True)
)
calibration_table["Net return"] = (100 * calibration_table["Net return"]).round(2).astype(str) + "%"
calibration_table[["Sortino", "Sharpe"]] = calibration_table[["Sortino", "Sharpe"]].round(3)
display(calibration_table)"""

    notebook.cells[9].source = """### 4. Frozen-forward economic comparison

This is the July 2025–March 2026 development-forward comparison after the July model fit and H1-selected policy have both been frozen."""

    notebook.cells[10].source = """forward_table = (
    forward_summary
    .assign(
        Objective=lambda frame: frame["objective"].map(ARM_NAMES),
        **{"Training history": lambda frame: frame["width_bps"].map(HISTORY).map(lambda days: f"{days}D")},
    )
    [[
        "Objective", "width_bps", "Training history", "candidate_id", "policy_id", "tau", "tp_bps", "sl_bps",
        "net_return", "sortino", "sharpe", "trades", "n_long", "n_short", "positive_months",
        "long_net", "short_net", "bull_sortino", "sideways_sortino", "bear_sortino", "constraint_violation",
    ]]
    .rename(columns={
        "width_bps": "DZ (bps)", "candidate_id": "Candidate",
        "policy_id": "Policy", "tau": "Confidence", "tp_bps": "TP (bps)", "sl_bps": "SL (bps)",
        "net_return": "Net return", "sortino": "Sortino", "sharpe": "Sharpe", "trades": "Trades",
        "n_long": "Long trades", "n_short": "Short trades", "positive_months": "Positive months",
        "long_net": "Long net", "short_net": "Short net", "bull_sortino": "Bull Sortino",
        "sideways_sortino": "Sideways Sortino", "bear_sortino": "Bear Sortino",
        "constraint_violation": "Diagnostic shortfall",
    })
    .sort_values(["DZ (bps)", "Objective"])
    .reset_index(drop=True)
)
for column in ("Net return", "Long net", "Short net"):
    forward_table[column] = (100 * forward_table[column]).round(2).astype(str) + "%"
forward_table[["Sortino", "Sharpe", "Bull Sortino", "Sideways Sortino", "Bear Sortino"]] = forward_table[["Sortino", "Sharpe", "Bull Sortino", "Sideways Sortino", "Bear Sortino"]].round(3)
display(forward_table)"""

    notebook.cells[11].source = """plot_frame = forward_summary.copy()
plot_frame["Series"] = plot_frame["objective"].map(ARM_NAMES) + " DZ" + plot_frame["width_bps"].astype(str)
colors = {"baseline": "#9d9d9d", "F1": "#4c78a8", "economic": "#f58518"}
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(plot_frame["Series"], 100 * plot_frame["net_return"], color=[colors[value] for value in plot_frame["objective"]])
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].set_title("Frozen net return: Jul 2025–Mar 2026")
axes[0].set_ylabel("Net return (%)")
axes[0].tick_params(axis="x", rotation=55)
axes[1].bar(plot_frame["Series"], plot_frame["sortino"], color=[colors[value] for value in plot_frame["objective"]])
axes[1].axhline(0, color="black", linewidth=0.8)
axes[1].set_title("Frozen Sortino")
axes[1].set_ylabel("Sortino")
axes[1].tick_params(axis="x", rotation=55)
fig.tight_layout()
plt.show()"""

    notebook.cells[12].source = """### 5. Quarterly stability — all 27 quarters

Each of the nine frozen rows is split into three quarters to show whether its aggregate result is stable or concentrated."""

    notebook.cells[13].source = """quarter_table = (
    forward_quarterly
    .assign(
        Objective=lambda frame: frame["objective"].map(ARM_NAMES),
        **{"Training history": lambda frame: frame["width_bps"].map(HISTORY).map(lambda days: f"{days}D")},
    )
    [["Objective", "width_bps", "Training history", "period", "net_return", "sortino", "sharpe", "trades", "n_long", "n_short"]]
    .rename(columns={
        "width_bps": "DZ (bps)", "period": "Quarter",
        "net_return": "Net return", "sortino": "Sortino", "sharpe": "Sharpe",
        "trades": "Trades", "n_long": "Long trades", "n_short": "Short trades",
    })
    .sort_values(["DZ (bps)", "Objective", "Quarter"])
    .reset_index(drop=True)
)
quarter_table["Net return"] = (100 * quarter_table["Net return"]).round(2).astype(str) + "%"
quarter_table[["Sortino", "Sharpe"]] = quarter_table[["Sortino", "Sharpe"]].round(3)
display(quarter_table)"""

    notebook.cells[14].source = """quarter_plot = forward_quarterly.copy()
quarter_plot["Series"] = quarter_plot["objective"].map(ARM_NAMES) + " DZ" + quarter_plot["width_bps"].astype(str)
quarter_pivot = quarter_plot.pivot(index="period", columns="Series", values="net_return") * 100
fig, ax = plt.subplots(figsize=(11, 4))
quarter_pivot.plot(kind="bar", ax=ax, width=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Frozen quarterly net return")
ax.set_xlabel("Quarter")
ax.set_ylabel("Net return (%)")
ax.legend(title="", ncol=3, fontsize=8)
fig.tight_layout()
plt.show()"""

    notebook.cells[15].source = """### 6. How economic selection changed results versus F1 and baseline

For each DZ, this reports F1-minus-baseline, Economic-minus-baseline and Economic-minus-F1 on the full frozen-forward period; these are selector differences, not additional profit."""

    notebook.cells[16].source = """metrics = ["net_return", "sortino", "sharpe", "trades"]
pivot = forward_summary.pivot(index="width_bps", columns="objective", values=metrics)
comparisons = [
    ("F1-tuned minus baseline", "F1", "baseline"),
    ("Economic-tuned minus baseline", "economic", "baseline"),
    ("Economic-tuned minus F1-tuned", "economic", "F1"),
]
rows = []
for label, left, right in comparisons:
    for width in pivot.index:
        rows.append({
            "Comparison": label,
            "DZ (bps)": width,
            "Net-return difference": pivot.loc[width, ("net_return", left)] - pivot.loc[width, ("net_return", right)],
            "Sortino difference": pivot.loc[width, ("sortino", left)] - pivot.loc[width, ("sortino", right)],
            "Sharpe difference": pivot.loc[width, ("sharpe", left)] - pivot.loc[width, ("sharpe", right)],
            "Trade-count difference": pivot.loc[width, ("trades", left)] - pivot.loc[width, ("trades", right)],
        })
difference_table = pd.DataFrame(rows)
difference_table["Net-return difference"] = (100 * difference_table["Net-return difference"]).round(2).astype(str) + "%"
difference_table[["Sortino difference", "Sharpe difference"]] = difference_table[["Sortino difference", "Sharpe difference"]].round(3)
display(difference_table)"""

    notebook.cells[17].source = f"""## Takeaways and handoff to Notebooks 02c–02d

1. The highest observed frozen-forward row is **{ARM_NAMES_FOR_BUILD[str(best['objective'])]} DZ{int(best['width_bps'])}**: {int(best['trades'])} trades, {best['net_return']:+.2%} net, {best['sortino']:+.3f} Sortino and {best['sharpe']:+.3f} Sharpe.
2. The Sortino leader within each DZ is: {leader_text}.
3. All nine rows remain negative after costs, so neither tuned selector demonstrates a robust profitable improvement over the pre-declared model.
4. Notebook 02c next documents and validates the matched downstream data. Notebook 02d then receives **Fixed baseline candidate 0**, DZ55/DZ65/DZ75, 180 days and the unchanged 24-feature schema for the raw feature comparison; no H1 policy is transferred.

### Limitations and decision rule

- BTC/USD is the only instrument; index replication is still required.
- Candidate and temporary-policy search on 2024 creates multiple-comparison risk.
- 2025 H1 selects execution policy and is not final out-of-sample evidence.
- The nine-month span is development-forward evidence because it has already been inspected; **2026 Q2 remains sealed**.
- Every variant is reported; final project inclusion remains a **manual decision**. The downstream baseline handoff is recorded in `{downstream_handoff['handoff_fingerprint'][:12]}...`."""

    nbformat.write(notebook, NOTEBOOK)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
