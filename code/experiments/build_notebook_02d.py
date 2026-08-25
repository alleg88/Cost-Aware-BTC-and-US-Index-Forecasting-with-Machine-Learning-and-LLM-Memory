"""Build Notebook 02d as the all-model raw sentiment feature ablation."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "02d_all_model_sentiment.ipynb"


def build_notebook(path: Path = NOTEBOOK) -> Path:
    cells = [
        nbf.v4.new_markdown_cell(
            """# 02d — All-model raw sentiment feature comparison

This notebook continues Notebook 02c and asks one isolated question: **does adding sentiment improve otherwise unchanged model outputs?** Notebook 02c established the sentiment provenance, scorer values, causal alignment and exact six-column schema. Here, nine fixed model-family baselines use the same raw 15-minute execution, so differences within each Model/DZ come only from the feature layer."""
        ),
        nbf.v4.new_markdown_cell(
            """## Handoff from Notebooks 02b and 02c

Notebook 02b passes **Fixed baseline candidate 0**, DZ55, DZ65 and DZ75, the fixed **180-day** training history and the unchanged 24-feature schema. Notebook 02c passes the matched DeBERTa/LLM feature contract. The cross-model extension uses candidate 0 for Logistic Regression, Decision Tree, Random Forest, Linear SVM, XGBoost, CatBoost, MLP, LSTM and GRU: each is its declared project baseline, not a newly tuned model. No execution policy is transferred."""
        ),
        nbf.v4.new_markdown_cell(
            """## Methodology

For every Model/DZ, four arms are compared. All three sentiment arms see the same sources — the headline block, the shared controls (`sent_tone_decay`, `sent_macro_decay`, `sent_fng_change_7d`) and the Fed/Trump `sent_direct_pulse` — so a difference between them can only come from the scorer and from how each arm weights what it reads, never from one arm having been given more data.

| arm | columns | what it is |
|---|---|---|
| **No sentiment** | 0 | the unchanged Notebook 02b control |
| **DeBERTa** | 6 | encoder scalar, filtered and weighted by DeBERTa's own outputs |
| **LLM-matched** | 6 | LLM scalar with everything unweighted — the control that isolates weighting and structure |
| **LLM-full** | 9 | the LLM's structured output kept as its own channels (`relevance`, `impact`, `asset`) |

Every arm's sentiment columns are `sent_news_decay`, `sent_news_count_24h`, `sent_tone_decay`, `sent_macro_decay`, `sent_fng_change_7d` and `sent_direct_pulse`; LLM-full adds `sent_llm_relevance_decay`, `sent_llm_hi_impact_decay` and `sent_llm_topic_share_24h`. The unchanged 24 base columns are `r1`, `r5`, `r20`, `vol_10`, `vol_20`, `vol_60`, `hl_range`, `co_range`, `rsi_14`, `volume`, `vol_z`, `hour`, `dayofweek`, `ofi`, `ofi_z20`, `ofi_mom5`, `trade_intensity_z`, `funding_rate`, `funding_z`, `oi_chg_1h`, `oi_chg_4h`, `oi_z`, `toptrader_ls_z` and `taker_ls_z`.

**DeBERTa against LLM-full** is the layer-versus-layer comparison the project's novelty claim rests on: each layer improved only by what it can produce alone. **LLM-matched against LLM-full** isolates how much of any difference is the scalar rather than the structure.

The unchanged 24 base columns are `r1`, `r5`, `r20`, `vol_10`, `vol_20`, `vol_60`, `hl_range`, `co_range`, `rsi_14`, `volume`, `vol_z`, `hour`, `dayofweek`, `ofi`, `ofi_z20`, `ofi_mom5`, `trade_intensity_z`, `funding_rate`, `funding_z`, `oi_chg_1h`, `oi_chg_4h`, `oi_z`, `toptrader_ls_z` and `taker_ls_z`. Classification uses the same five 2024 `BlockingTimeSeriesSplit` folds and one-row label-tail trim.

Every Model/Arm/DZ is fitted once on 1 July 2025 using the preceding 180 days. A directional prediction enters at the next M15 open and exits after one M15 bar; overlapping trades are not opened and costs are **5 bps per side**. No probability gate, bracket geometry or one-minute execution data is used."""
        ),
        nbf.v4.new_markdown_cell(
            """### Data timeline and forward status

- **2024:** five `BlockingTimeSeriesSplit` folds provide classification diagnostics.
- **January-June 2025:** the fixed history for the 1 July fit; it is not scored in this notebook.
- **July 2025-March 2026:** one frozen development-forward run with no model refit.
- **2026 Q2:** no rows are loaded or evaluated; it remains the sealed final lockbox.

This separation keeps Notebook 02d a feature-only comparison."""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import sys
import pandas as pd
from IPython.display import display

pd.set_option("display.max_rows", 200)
pd.set_option("display.max_columns", None)

CODE_ROOT = Path.cwd().resolve()
if CODE_ROOT.name == "notebooks":
    CODE_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from experiments.all_model_sentiment_raw import POLICY_COLUMNS
from experiments.all_model_sentiment_scoreboard import ARM_LABELS, build_scoreboards
from experiments.notebook02b_handoff import HANDOFF_PATH, load_notebook02b_handoff
from experiments.raw_hold_control import MODEL_LABELS

handoff = load_notebook02b_handoff(HANDOFF_PATH)
tables = build_scoreboards()
classification = tables["classification"]
economics = tables["economics"]

assert handoff["candidate_id"] == 0
assert handoff["training_history_days"] == 180
assert len(classification) == len(economics) == 108
assert economics["candidate_id"].eq(0).all()
assert set(economics["width_bps"].astype(int)) == {55, 65, 75}
assert economics["lookback_days"].eq(180).all()
assert economics["hold_minutes"].eq(15).all()
assert not POLICY_COLUMNS.intersection(economics.columns)
assert economics["period_end"].max() <= pd.Timestamp("2026-04-01", tz="UTC")
print("Validated 9 models x 4 arms x 3 dead zones and the sealed forward boundary.")"""
        ),
        nbf.v4.new_markdown_cell(
            """## Results

Table 1 diagnoses classification on unchanged 2024 blocking folds and includes the matched raw-forward Sortino only to define a common row order. Table 2 reports the matched raw forward economics. Every row remains visible; all result tables are ordered by Sortino from highest to lowest."""
        ),
        nbf.v4.new_markdown_cell(
            """### 1. Classification diagnostics

Overall macro-F1 summarises all three classes; Robust F1 is the weakest-regime fold score and penalises performance that depends on one market regime."""
        ),
        nbf.v4.new_code_cell(
            """print("Table 1 — 2024 classification diagnostics")
classification_table = (
    classification.merge(
        economics[["sentiment_arm", "model_name", "width_bps", "sortino"]],
        on=["sentiment_arm", "model_name", "width_bps"],
        validate="one_to_one",
    )[["model_name", "Arm", "width_bps", "overall_f1", "robust_f1", "sortino"]]
    .assign(model_name=lambda frame: frame["model_name"].map(MODEL_LABELS))
    .rename(columns={
        "model_name": "Model",
        "width_bps": "DZ",
        "overall_f1": "Overall macro-F1",
        "robust_f1": "Robust F1",
        "sortino": "Sortino",
    })
    .sort_values(["Sortino", "Robust F1"], ascending=False, ignore_index=True)
)
display(classification_table.round({"Overall macro-F1": 3, "Robust F1": 3, "Sortino": 3}))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 2. Raw frozen-forward economics

Every row uses direct class predictions and the same 15-minute hold. Differences within a matched Model/DZ therefore come only from the added sentiment features and their effect on directional predictions."""
        ),
        nbf.v4.new_code_cell(
            """print("Table 2 — Raw frozen-forward economics")
economic_table = (
    economics[[
        "model_name", "Arm", "width_bps", "trades", "gross_return",
        "net_return", "sortino", "sharpe", "positive_months",
    ]]
    .assign(model_name=lambda frame: frame["model_name"].map(MODEL_LABELS))
    .rename(columns={
        "model_name": "Model",
        "width_bps": "DZ",
        "trades": "Trades",
        "gross_return": "Gross return",
        "net_return": "Net return",
        "sortino": "Sortino",
        "sharpe": "Sharpe",
        "positive_months": "Positive months",
    })
    .sort_values(["Sortino", "Sharpe", "Net return"], ascending=False, ignore_index=True)
)
shown = economic_table.copy()
shown[["Sortino", "Sharpe"]] = shown[["Sortino", "Sharpe"]].round(3)
shown["Gross return"] = shown["Gross return"].map(lambda value: f"{100 * value:.2f}%")
shown["Net return"] = shown["Net return"].map(lambda value: f"{100 * value:.2f}%")
display(shown)

leader = economics.sort_values(["sortino", "sharpe", "net_return"], ascending=False).iloc[0]
print(
    f"Relative raw leader: {MODEL_LABELS[leader['model_name']]} / {leader['Arm']} "
    f"DZ{int(leader['width_bps'])}; {int(leader['trades'])} trades, "
    f"{leader['net_return']:.2%} net, {leader['sortino']:.3f} Sortino and "
    f"{leader['sharpe']:.3f} Sharpe."
)

paired = economics.pivot(
    index=["model_name", "width_bps"], columns="sentiment_arm", values="net_return"
)
cells = len(paired)
print(f"Against the no-sentiment control, over {cells} matched Model/DZ cells:")
for arm, label in (("classic", "DeBERTa"), ("llm", "LLM-matched"), ("llm_full", "LLM-full")):
    delta = paired[arm] - paired["none"]
    print(f"  {label:12s} better in {int((delta > 0).sum()):2d}/{cells} cells, "
          f"median delta {delta.median():+.2%}")

print("\\nBetween the sentiment arms — the comparisons the novelty claim rests on:")
for a, b, label in (("llm_full", "classic", "LLM-full vs DeBERTa (layer vs layer)"),
                    ("llm_full", "llm", "LLM-full vs LLM-matched (structure vs scalar)")):
    delta = paired[a] - paired[b]
    print(f"  {label:44s} better in {int((delta > 0).sum()):2d}/{cells} cells, "
          f"median delta {delta.median():+.2%}")

per_trade = economics.assign(
    gross_bps=economics["gross_return"] / economics["trades"].clip(lower=1) * 1e4)
print("\\nMedian gross edge per trade by arm (the number that decides against costs):")
display(per_trade.groupby("sentiment_arm")[["gross_bps", "net_return", "trades"]]
        .median().round(3))

best_per_arm = (per_trade.sort_values("sortino", ascending=False)
                .groupby("sentiment_arm").head(1)
                .sort_values("sortino", ascending=False))
shown = best_per_arm[["sentiment_arm", "model_name", "width_bps", "trades",
                      "gross_return", "net_return", "sortino", "sharpe",
                      "positive_months", "gross_bps"]].copy()
shown["sentiment_arm"] = shown["sentiment_arm"].map(ARM_LABELS)
shown["model_name"] = shown["model_name"].map(MODEL_LABELS)
for column in ("gross_return", "net_return"):
    shown[column] = shown[column].map(lambda value: f"{100 * value:.2f}%")
print("\\nBest row of each arm — the family shortlist above can only show one row "
      "per model, so it hides an arm whose leader shares a family with another arm's:")
display(shown.rename(columns={
    "sentiment_arm": "Arm", "model_name": "Model", "width_bps": "DZ",
    "trades": "Trades", "gross_return": "Gross", "net_return": "Net",
    "sortino": "Sortino", "sharpe": "Sharpe", "positive_months": "Positive months",
    "gross_bps": "Gross bps/trade"}).round(3).reset_index(drop=True))"""
        ),
        nbf.v4.new_markdown_cell(
            """### 3. Five strongest model families

Each model family first keeps its best Sentiment/DZ row. The five distinct models are then ranked by Sortino, with Sharpe and net return as tie-breakers; this prevents several configurations of one model from occupying the whole shortlist."""
        ),
        nbf.v4.new_code_cell(
            """print("Table 3 — Five strongest raw model families")
top_five = (
    economics
    .sort_values(["sortino", "sharpe", "net_return"], ascending=False)
    .drop_duplicates("model_name", keep="first")
    .head(5)
    [[
        "model_name", "Arm", "width_bps", "trades", "gross_return",
        "net_return", "sortino", "sharpe", "positive_months",
    ]]
    .assign(model_name=lambda frame: frame["model_name"].map(MODEL_LABELS))
    .rename(columns={
        "model_name": "Model",
        "width_bps": "DZ",
        "trades": "Trades",
        "gross_return": "Gross return",
        "net_return": "Net return",
        "sortino": "Sortino",
        "sharpe": "Sharpe",
        "positive_months": "Positive months",
    })
)
assert len(top_five) == 5 and top_five["Model"].nunique() == 5
shown_top = top_five.copy()
shown_top[["Sortino", "Sharpe"]] = shown_top[["Sortino", "Sharpe"]].round(3)
shown_top["Gross return"] = shown_top["Gross return"].map(lambda value: f"{100 * value:.2f}%")
shown_top["Net return"] = shown_top["Net return"].map(lambda value: f"{100 * value:.2f}%")
display(shown_top)"""
        ),
        nbf.v4.new_markdown_cell(
            """## Decision and handoff

Two rows in the whole table are positive after costs, and both are Linear SVM at DZ75: the
**no-sentiment control** (40 trades, +7.51% net, 1.838 Sortino) and **LLM-full** (26 trades,
+4.42% net, 1.210 Sortino). Both sit below the 50-event floor Notebook 02e applies, so
neither is promotable on its own.

The aggregate verdict is unchanged from the no-sentiment baseline: **no sentiment arm beats
the control**, each winning only 8–9 of 27 matched cells with a negative median delta.
Sentiment is not promoted.

What *is* new is the ordering **within** the sentiment layer, and it is the part the novelty
claim turns on. LLM-full carries the strongest gross edge per trade of any arm — the
quantity that decides everything against a 10 bps round trip — and beats DeBERTa in 16 of 27
cells. Reducing the same LLM to a single scalar (LLM-matched) gives that back. So the
structured output is doing real work; it is simply not enough work to clear costs.

Notebook 02d makes no policy choice. Notebook 02e calibrates execution policy on exactly
these predictions and keeps the no-sentiment control alongside every sentiment arm.
2026 Q2 remains untouched."""
        ),
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = {
        "display_name": "msc-code",
        "language": "python",
        "name": "msc-code",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, path)
    return path


if __name__ == "__main__":
    print(build_notebook())
