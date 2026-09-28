"""Build and execute concise artifact-only readers for the index experiments."""
from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

from experiments.notebook_hygiene import canonical_colab_setup


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATHS = {
    "nine_models": CODE_ROOT / "notebooks" / "04_RQ1_E_indices_nine_model_benchmark.ipynb",
    "vix": CODE_ROOT / "notebooks" / "05_RQ1_F_indices_VIX_ablation.ipynb",
    "deberta": CODE_ROOT / "notebooks" / "15_RQ3_D_indices_DeBERTa_sentiment.ipynb",
    "llm": CODE_ROOT / "notebooks" / "16_RQ3_E_indices_LLM_sentiment.ipynb",
    "ensemble": CODE_ROOT / "notebooks" / "10_RQ2_H_indices_all_model_ensemble.ipynb",
    "comparison": CODE_ROOT / "notebooks" / "11_RQ2_I_indices_policy_comparison.ipynb",
}
MODEL_LABELS = {
    "logreg": "LogReg",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "svm_linear": "Linear SVM",
    "xgboost_balanced": "XGBoost",
    "catboost_balanced": "CatBoost",
    "mlp": "MLP",
    "lstm": "LSTM",
    "gru": "GRU",
}
KERNEL = {
    "display_name": "MSC Project (Python 3.12)",
    "language": "python",
    "name": "msc-code",
}
COLAB_SETUP = canonical_colab_setup()


def validate_llm_score_manifest(manifest: dict, frozen_identity: dict) -> None:
    """Reject incomplete, post-cutoff or mixed-identity LLM scores."""
    if (
        manifest.get("complete") is not True
        or manifest.get("cutoff_exclusive") != "2026-04-01T00:00:00+00:00"
        or manifest.get("identity") != frozen_identity
    ):
        raise ValueError("score manifest does not match the frozen LLM identity")


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def _notebook(title: str, methodology: str, body: list):
    cells = [
        code(COLAB_SETUP),
        md(
            f"""
# {title}

## Methodology

{methodology}

USA500 and USATECH are separate. Every economic table reports Net, Sharpe,
Sortino and drawdown after a fixed all-in round-trip deduction: 2 bps (0.02%)
for USA500 and 3 bps (0.03%) for USATECH. This covers bid-ask spread plus
commission/slippage; bar-specific spreads remain diagnostic, not additional
costs. Inputs stop before `2026-04-01 00:00 UTC`; Q2-2026 remains reserved and
no model is refitted here.
"""
        ),
        *body,
    ]
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata.kernelspec = KERNEL
    notebook.metadata.language_info = {"name": "python", "version": "3.12"}
    nbf.validate(notebook)
    return notebook


def _vix_notebook():
    body = [
        code(
            """
import json
import numpy as np
import pandas as pd
from IPython.display import display
from experiments.build_notebook_06 import MODEL_LABELS

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_replication"
summary_rows, effect_rows = [], []
for stream in STREAMS:
    decision = json.loads((ROOT / stream / "vix_admission.json").read_text(encoding="utf-8"))
    paired = pd.read_parquet(ROOT / stream / "vix_gate_paired_2024.parquet")
    assert decision["gate_complete"] and decision["frozen_before_sentiment"]
    assert decision["conditions"]["families_positive_majority_5_of_9"]
    summary_rows.append({
        "Index": stream.upper(),
        "Selected input": "Price + VIX" if decision["selected_base"] == "price_vix" else "Price only",
        "Median gate net delta (pp)": 100 * float(decision["metrics"]["median_family_delta"]),
        "Positive models / 9": int(decision["metrics"]["positive_families"]),
        "Positive folds / 5": int(decision["metrics"]["positive_folds"]),
        "Trade retention %": 100 * float(decision["metrics"]["trade_retention"]),
    })
    model_width = paired.groupby(["model_name", "width_bps"], as_index=False).agg(
        price_net=("price_net", "sum"),
        vix_net=("vix_net", "sum"),
        price_trades=("price_trades", "sum"),
        vix_trades=("vix_trades", "sum"),
    )
    model_width["net_delta"] = model_width["vix_net"] - model_width["price_net"]
    representative = model_width.sort_values(
        ["model_name", "net_delta", "width_bps"]
    ).groupby("model_name", as_index=False).nth(1).set_index("model_name")
    for row in decision["family_deltas"]:
        model = row["model_name"]
        pair = representative.loc[model]
        assert abs(float(pair["net_delta"]) - float(row["net_delta"])) < 1e-12
        price_trades = int(pair["price_trades"])
        vix_trades = int(pair["vix_trades"])
        effect_rows.append({
            "Index": stream.upper(),
            "Model": MODEL_LABELS[model],
            "Gate width bps": int(pair["width_bps"]),
            "Price-only gate net %": 100 * float(pair["price_net"]),
            "Price + VIX gate net %": 100 * float(pair["vix_net"]),
            "Gate net delta (pp)": 100 * float(row["net_delta"]),
            "Price trades": price_trades,
            "Price + VIX trades": vix_trades,
            "Trade retention %": 100 * vix_trades / price_trades if price_trades else 0.0,
        })
summary = pd.DataFrame(summary_rows).sort_values(
    ["Median gate net delta (pp)", "Index"], ascending=[False, True]
)
effects = pd.DataFrame(effect_rows).sort_values(
    ["Gate net delta (pp)", "Index", "Model"], ascending=[False, True, True]
)
"""
        ),
        md("## Admission decision"),
        md("Method: The 2024 gate selects VIX when the median net delta is positive, at least 5/9 model families and 3/5 folds are positive, and aggregate trade retention is at least 80%."),
        code(
            """
# non-economic-table
display(summary.round({"Median gate net delta (pp)": 3, "Trade retention %": 1}))
print("Takeaway: VIX improves relative Net for 6/9 USA500 models and 5/9 USATECH models, although the representative gate strategies remain unprofitable.")
"""
        ),
        md("## Numeric model-family effects"),
        md("Method: Five fold results are summed within each width; the middle of the three width-level net deltas is the family vote, and the table shows that exact price/VIX pair and its aggregate gate trades."),
        code(
            """
# vix-numeric-table
# non-economic-table
display(effects.round({"Price-only gate net %": 3, "Price + VIX gate net %": 3, "Gate net delta (pp)": 3, "Trade retention %": 1}))
print("Takeaway: VIX often reduces the loss relative to price-only, but every representative absolute gate return remains negative.")
"""
        ),
    ]
    return _notebook(
        "06b - VIX admission",
        "The VIX experiment pairs price-only and price+VIX M15 inputs across five embargoed 2024 OOF folds, nine model families and three label widths. The simple majority gate tests whether VIX improves net economics consistently before H1 2025 or forward evidence is examined.",
        body,
    )


def _nine_models_notebook():
    body = [
        code(
            """
import json
import numpy as np
import pandas as pd
from IPython.display import display
from experiments.build_notebook_06 import MODEL_LABELS
from experiments.index_replication_protocol import MODEL_NAMES

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_replication"
FORWARD_ROOT = CODE_ROOT / "experiments" / "cache" / "index_all_model_forward"
classification, selected, forward = {}, {}, {}
for stream in STREAMS:
    result = json.loads((ROOT / stream / "result.json").read_text(encoding="utf-8"))
    diagnostic = json.loads((FORWARD_ROOT / stream / "result.json").read_text(encoding="utf-8"))
    assert result["q2_loaded"] is False
    assert diagnostic["q2_loaded"] is False and diagnostic["forward_rows"] == 36
    assert diagnostic["evidence_role"] == "secondary_reused_forward_diagnostic"
    classification[stream] = pd.read_parquet(ROOT / stream / "classification_2024.parquet")
    selected[stream] = pd.read_parquet(ROOT / stream / "h1_selected_policies.parquet")
    forward[stream] = pd.read_parquet(FORWARD_ROOT / stream / "forward_summary.parquet")
    assert classification[stream]["model_name"].nunique() == 9
    assert set(classification[stream]["model_name"]) == set(MODEL_NAMES)
"""
        ),
        md("## 2024 OOF model evidence"),
        md("Method: For each baseline model, the table reports the median classification score across the three registered label widths; economics are not selected on OOF F1."),
        code(
            """
# non-economic-table
rows = []
for stream in STREAMS:
    scope = classification[stream][classification[stream]["arm"].eq("selected_base")]
    view = scope.groupby("model_name", as_index=False).agg(
        macro_f1=("macro_f1", "median"),
        balanced_accuracy=("balanced_accuracy", "median"),
        short_recall=("short_recall", "median"),
        flat_recall=("flat_recall", "median"),
        long_recall=("long_recall", "median"),
    ).assign(Index=stream.upper())
    rows.append(view)
oof_view = pd.concat(rows, ignore_index=True).rename(columns={
    "model_name": "Model", "macro_f1": "Macro-F1", "balanced_accuracy": "Balanced accuracy",
    "short_recall": "SHORT recall", "flat_recall": "FLAT recall", "long_recall": "LONG recall",
})[["Index", "Model", "Macro-F1", "Balanced accuracy", "SHORT recall", "FLAT recall", "LONG recall"]]
oof_view["Model"] = oof_view["Model"].map(MODEL_LABELS)
oof_view = oof_view.sort_values(["Macro-F1", "Balanced accuracy"], ascending=[False, False])
display(oof_view.round(3))
print("Takeaway: All nine model families are evaluated under the same blocked 2024 OOF construction.")
"""
        ),
        md("## H1 policy calibration"),
        md("Method: January-June 2025 calibrates one label width and confidence threshold per model using trade count, side balance, positive months, Net, Sharpe and Sortino."),
        code(
            """
# economic-table
h1_rows = []
for stream in STREAMS:
    scope = selected[stream][selected[stream]["arm"].eq("selected_base")].copy()
    assert scope["model_name"].nunique() == 9 and set(scope["model_name"]) == set(MODEL_NAMES)
    scope["Index"] = stream.upper()
    h1_rows.append(scope)
h1_view = pd.concat(h1_rows, ignore_index=True).rename(columns={
    "model_name": "Model", "width_bps": "Width bps", "tau": "Threshold", "eligible": "H1 status",
    "trades": "Trades", "n_long": "LONG", "n_short": "SHORT", "positive_months": "Positive months",
    "net_return": "Net %", "daily_sharpe": "Sharpe", "daily_sortino": "Sortino",
})
h1_view["Net %"] *= 100
h1_view["Model"] = h1_view["Model"].map(MODEL_LABELS)
h1_view["H1 status"] = h1_view["H1 status"].map({True: "Pass", False: "Below gate"})
h1_view = h1_view.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(h1_view[["Index", "Model", "Width bps", "Threshold", "H1 status", "Trades", "LONG", "SHORT", "Positive months", "Net %", "Sharpe", "Sortino"]].round(3))
passed = int(h1_view["H1 status"].eq("Pass").sum())
print(f"Takeaway: {passed} of {len(h1_view)} baseline configurations meet the H1 quality gate.")
"""
        ),
        md("## Forward results"),
        md("Method: Each model retains its H1-selected width and threshold for a common July 2025-March 2026 descriptive economic comparison."),
        code(
            """
# forward-all-models
# economic-table
active = pd.concat([
    forward[stream][forward[stream]["arm"].eq("selected_base")].assign(Index=stream.upper())
    for stream in STREAMS
], ignore_index=True).rename(columns={
    "model_name": "Model", "trades": "Trades", "n_long": "LONG", "n_short": "SHORT",
    "trades_per_day": "Trades/day", "net_return": "Net %", "daily_sharpe": "Sharpe",
    "daily_sortino": "Sortino", "max_drawdown": "Max drawdown %",
})
active["Model"] = active["Model"].map(MODEL_LABELS)
active["H1 status"] = active["h1_eligible"].map({True: "Pass", False: "Below gate"})
active["Forward status"] = active["status"].map({"traded": "Traded", "no_trades": "No trades"})
active["Net %"] *= 100; active["Max drawdown %"] *= 100
result_columns = ["Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]
assert len(active) == 18 and active.groupby("Index")["Model"].nunique().eq(9).all()
assert np.isfinite(active[result_columns].to_numpy(dtype=float)).all()
active = active.sort_values(["Net %", "Sortino", "Model"], ascending=[False, False, True])
display(active[["Index", "Model", "H1 status", "Forward status", *result_columns]].round(3))
best = active.iloc[0]
print(f"Takeaway: {best['Index']} {best['Model']} has the highest baseline Net at {best['Net %']:.2f}%; this reused-forward result is descriptive because the interval has already been inspected.")
"""
        ),
    ]
    return _notebook(
        "06a - Nine models",
        "Nine models use complete M15 bid/ask-derived index bars and three next-bar direction widths. Blocked 2024 OOF measures classification, H1 2025 calibrates width and confidence threshold, and July 2025-March 2026 provides descriptive reused-forward economics.",
        body,
    )


def _deberta_notebook():
    body = [
        code(
            """
import json
import numpy as np
import pandas as pd
from IPython.display import display
from experiments.build_notebook_06 import MODEL_LABELS
from experiments.index_replication_protocol import MODEL_NAMES

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_replication"
FORWARD_ROOT = CODE_ROOT / "experiments" / "cache" / "index_all_model_forward"
RAW = CODE_ROOT / "sentiment" / "raw"
selected, forward = {}, {}
for stream in STREAMS:
    for prefix in ("", "direct_events_"):
        manifest = json.loads((RAW / f"scores_{prefix}{stream}.manifest.json").read_text(encoding="utf-8"))
        assert manifest["complete"] and manifest["cutoff_exclusive"].startswith("2026-04-01")
    selected[stream] = pd.read_parquet(ROOT / stream / "h1_selected_policies.parquet")
    diagnostic = json.loads((FORWARD_ROOT / stream / "result.json").read_text(encoding="utf-8"))
    assert diagnostic["q2_loaded"] is False and diagnostic["forward_rows"] == 36
    assert diagnostic["evidence_role"] == "secondary_reused_forward_diagnostic"
    forward[stream] = pd.read_parquet(FORWARD_ROOT / stream / "forward_summary.parquet")
    scope = selected[stream][selected[stream]["arm"].eq("deberta_matched")]
    assert scope["model_name"].nunique() == 9 and set(scope["model_name"]) == set(MODEL_NAMES)
"""
        ),
        md("## Feature dictionary"),
        md("Method: The dictionary defines each matched DeBERTa feature and the information available before its M15 trading decision."),
        code(
            """
# non-economic-table
feature_dictionary = pd.DataFrame([
    ("sent_news_decay", "Time-decayed DeBERTa sentiment of first-seen financial headlines", "Only headlines available before the current bar"),
    ("sent_news_count_24h", "Number of first-seen matched headlines in the previous 24 hours", "Backward-looking rolling count"),
    ("sent_direct_decay", "Time-decayed sentiment from first-party Federal Reserve statements/minutes and Trump Truth Social posts", "Only direct_events available before the current bar; this is not linguistic directness"),
    ("sent_tone_decay", "Time-decayed native GDELT news tone", "Only first-seen news available before the current bar"),
    ("sent_macro_decay", "Time-decayed indicator of scheduled FRED macro releases", "Release timestamp must precede the current bar"),
], columns=["Feature", "Meaning", "Causal availability"])
display(feature_dictionary)
print("Takeaway: direct_events is a causal first-party event stream, not a measure of how directly a sentence is written.")
"""
        ),
        md("## H1 results for nine individual models"),
        md("Method: The selected price/VIX base and matched DeBERTa features use the same January-June 2025 calibration rule across all nine model families."),
        code(
            """
# economic-table
rows = []
for stream in STREAMS:
    scope = selected[stream][selected[stream]["arm"].isin(["selected_base", "deberta_matched"])].copy()
    scope["Index"] = stream.upper(); rows.append(scope)
h1 = pd.concat(rows, ignore_index=True).replace({"arm": {"selected_base": "Price/VIX base", "deberta_matched": "DeBERTa matched"}}).rename(columns={
    "arm": "Features", "model_name": "Model", "width_bps": "Width bps", "tau": "Threshold",
    "eligible": "H1 status", "trades": "Trades", "n_long": "LONG", "n_short": "SHORT",
    "net_return": "Net %", "daily_sharpe": "Sharpe", "daily_sortino": "Sortino",
})
h1["Net %"] *= 100
h1["Model"] = h1["Model"].map(MODEL_LABELS)
h1["H1 status"] = h1["H1 status"].map({True: "Pass", False: "Below gate"})
h1 = h1.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(h1[["Index", "Features", "Model", "Width bps", "Threshold", "H1 status", "Trades", "LONG", "SHORT", "Net %", "Sharpe", "Sortino"]].round(3))
best_h1 = h1.iloc[0]
print(f"Takeaway: {best_h1['Index']} {best_h1['Features']} {best_h1['Model']} has the highest H1 Net at {best_h1['Net %']:.2f}%.")
"""
        ),
        md("## Forward results"),
        md("Method: Every DeBERTa model retains its H1-selected width and threshold for the same July 2025-March 2026 descriptive economic comparison."),
        code(
            """
# forward-all-models
# economic-table
active = pd.concat([
    forward[stream][forward[stream]["arm"].eq("deberta_matched")].assign(Index=stream.upper())
    for stream in STREAMS
], ignore_index=True).replace({"arm": {"deberta_matched": "DeBERTa matched"}}).rename(columns={
    "arm": "Features", "model_name": "Model", "trades": "Trades", "trades_per_day": "Trades/day",
    "n_long": "LONG", "n_short": "SHORT", "net_return": "Net %", "daily_sharpe": "Sharpe",
    "daily_sortino": "Sortino", "max_drawdown": "Max drawdown %",
})
active["Model"] = active["Model"].map(MODEL_LABELS)
active["H1 status"] = active["h1_eligible"].map({True: "Pass", False: "Below gate"})
active["Forward status"] = active["status"].map({"traded": "Traded", "no_trades": "No trades"})
active["Net %"] *= 100; active["Max drawdown %"] *= 100
result_columns = ["Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]
assert len(active) == 18 and active.groupby("Index")["Model"].nunique().eq(9).all()
assert np.isfinite(active[result_columns].to_numpy(dtype=float)).all()
active = active.sort_values(["Net %", "Sortino", "Model"], ascending=[False, False, True])
display(active[["Index", "Features", "Model", "H1 status", "Forward status", *result_columns]].round(3))
best = active.iloc[0]
print(f"Takeaway: {best['Index']} {best['Model']} has the highest DeBERTa Net at {best['Net %']:.2f}%; this reused-forward result is descriptive because the interval has already been inspected.")
"""
        ),
    ]
    return _notebook(
        "06c - DeBERTa sentiment",
        "DeBERTa scores first-seen news and direct Federal Reserve or Truth Social events available before each M15 decision and produces five causal matched features. The same nine-model protocol uses 2024 OOF evidence, H1 2025 calibration and descriptive July 2025-March 2026 economics.",
        body,
    )


def _llm_notebook():
    body = [
        code(
            """
import json
import numpy as np
import pandas as pd
from html import escape
from IPython.display import HTML, display
from experiments.build_notebook_06 import MODEL_LABELS, validate_llm_score_manifest
from experiments.index_replication_protocol import MODEL_NAMES
from sentiment.index_scoring import _canonical_hash
from sentiment.score_llm import SYSTEM, _ITEM

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_replication"
FORWARD_ROOT = CODE_ROOT / "experiments" / "cache" / "index_all_model_forward"
RAW = CODE_ROOT / "sentiment" / "raw"
identity_files = list(RAW.glob("index_*_identity.json"))
assert len(identity_files) == 1
identity = json.loads(identity_files[0].read_text(encoding="utf-8"))["identity"]
assert identity["tag"] and identity["digest"]
assert identity["temperature"] == 0.0 and identity["think"] == "low"
assert identity["batch_size"] == 10
selected, forward = {}, {}
for stream in STREAMS:
    for prefix in ("", "direct_events_"):
        manifest = json.loads((RAW / f"scores_llm_{prefix}{stream}.manifest.json").read_text(encoding="utf-8"))
        validate_llm_score_manifest(manifest, identity)
    selected[stream] = pd.read_parquet(ROOT / stream / "h1_selected_policies.parquet")
    diagnostic = json.loads((FORWARD_ROOT / stream / "result.json").read_text(encoding="utf-8"))
    assert diagnostic["q2_loaded"] is False and diagnostic["forward_rows"] == 36
    assert diagnostic["evidence_role"] == "secondary_reused_forward_diagnostic"
    forward[stream] = pd.read_parquet(FORWARD_ROOT / stream / "forward_summary.parquet")

registered_arms = set(selected[STREAMS[0]]["arm"])
LLM_FULL = next(arm for arm in registered_arms if arm.endswith("_full"))
llm_prefix = LLM_FULL.rsplit("_", 1)[0]
LLM_MATCHED = next(arm for arm in registered_arms if arm == f"{llm_prefix}_matched")
LLM_ARMS = (LLM_MATCHED, LLM_FULL)
for stream in STREAMS:
    assert set(selected[stream]["arm"]) == registered_arms
    scope = selected[stream][selected[stream]["arm"].isin(LLM_ARMS)]
    assert scope["model_name"].nunique() == 9
    assert scope.groupby("arm")["model_name"].nunique().eq(9).all()
    assert set(scope["model_name"]) == set(MODEL_NAMES)
"""
        ),
        md("## Frozen LLM scorer contract"),
        md("Method: The block records the frozen LLM digest, deterministic settings, batch size 10, system prompt and output schema used for every score."),
        code(
            """
# non-economic-table
prompt_contract = {"system": SYSTEM, "user_line_template": "n{index}: {title}", "batch_protocol": identity["batch_protocol"]}
schema_contract = {"item": _ITEM, "envelope": "object keyed n0..n{batch_size-1}"}
assert _canonical_hash(prompt_contract) == identity["prompt_hash"]
assert _canonical_hash(schema_contract) == identity["schema_hash"]
technical_html = f'''<details><summary><b>Exact technical contract</b></summary>
<p><b>Frozen scorer digest:</b> {escape(identity["digest"][:16])}...; temperature 0; think=low; batch size 10.</p>
<p><b>System prompt</b></p><pre>{escape(SYSTEM)}</pre>
<p><b>User line:</b> <code>n{{index}}: {{title}}</code></p>
<p><b>Output schema</b></p><pre>{escape(json.dumps(_ITEM, indent=2))}</pre></details>'''
display(HTML(technical_html))
print("Takeaway: Every headline is scored deterministically in indexed batches of exactly ten under one frozen identity.")
"""
        ),
        md("## Matched and full feature sets"),
        md("Method: LLM matched uses the same five causal sentiment fields for a scorer-only comparison; LLM full adds exactly three structured fields."),
        code(
            """
# non-economic-table
features = pd.DataFrame([
    ("LLM matched", "Same five causal fields: news sentiment, 24h news count, direct-event sentiment, news tone and macro releases; only the scorer changes"),
    ("LLM full", "LLM matched plus relevance decay, high-impact sentiment decay and 24h topic share"),
], columns=["Arm", "Feature difference"])
display(features)
print("Takeaway: The full arm changes only three declared structured fields, so the matched arm remains the scorer-only comparison.")
"""
        ),
        md("## H1 results for nine individual models"),
        md("Method: Base, LLM matched and LLM full use the same six-month H1 calibration and are shown for all nine model families."),
        code(
            """
# economic-table
rows = []
labels = {"selected_base": "Price/VIX base", LLM_MATCHED: "LLM matched", LLM_FULL: "LLM full"}
for stream in STREAMS:
    scope = selected[stream][selected[stream]["arm"].isin(labels)].copy()
    scope["Index"] = stream.upper(); rows.append(scope)
h1 = pd.concat(rows, ignore_index=True).replace({"arm": labels}).rename(columns={
    "arm": "Features", "model_name": "Model", "width_bps": "Width bps", "tau": "Threshold",
    "eligible": "H1 status", "trades": "Trades", "n_long": "LONG", "n_short": "SHORT",
    "net_return": "Net %", "daily_sharpe": "Sharpe", "daily_sortino": "Sortino",
})
h1["Net %"] *= 100
h1["Model"] = h1["Model"].map(MODEL_LABELS)
h1["H1 status"] = h1["H1 status"].map({True: "Pass", False: "Below gate"})
h1 = h1.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(h1[["Index", "Features", "Model", "Width bps", "Threshold", "H1 status", "Trades", "LONG", "SHORT", "Net %", "Sharpe", "Sortino"]].round(3))
best_h1 = h1.iloc[0]
print(f"Takeaway: {best_h1['Index']} {best_h1['Features']} {best_h1['Model']} has the highest H1 Net at {best_h1['Net %']:.2f}%.")
"""
        ),
        md("## Forward results"),
        md("Method: Every LLM matched and LLM full model retains its H1-selected width and threshold for the same July 2025-March 2026 descriptive economic comparison."),
        code(
            """
# forward-all-models
# economic-table
active = pd.concat([
    forward[stream][forward[stream]["arm"].isin(LLM_ARMS)].assign(Index=stream.upper())
    for stream in STREAMS
], ignore_index=True).replace({"arm": labels}).rename(columns={
    "arm": "Features", "model_name": "Model", "trades": "Trades", "trades_per_day": "Trades/day",
    "n_long": "LONG", "n_short": "SHORT", "net_return": "Net %", "daily_sharpe": "Sharpe",
    "daily_sortino": "Sortino", "max_drawdown": "Max drawdown %",
})
active["Model"] = active["Model"].map(MODEL_LABELS)
active["H1 status"] = active["h1_eligible"].map({True: "Pass", False: "Below gate"})
active["Forward status"] = active["status"].map({"traded": "Traded", "no_trades": "No trades"})
active["Net %"] *= 100; active["Max drawdown %"] *= 100
result_columns = ["Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]
assert len(active) == 36 and active.groupby(["Index", "Features"])["Model"].nunique().eq(9).all()
assert np.isfinite(active[result_columns].to_numpy(dtype=float)).all()
active = active.sort_values(["Net %", "Sortino", "Model"], ascending=[False, False, True])
display(active[["Index", "Features", "Model", "H1 status", "Forward status", *result_columns]].round(3))
best = active.iloc[0]
print(f"Takeaway: {best['Index']} {best['Features']} {best['Model']} has the highest LLM Net at {best['Net %']:.2f}%; this reused-forward result is descriptive because the interval has already been inspected.")
"""
        ),
    ]
    return _notebook(
        "06d - LLM sentiment",
        "A frozen LLM scorer processes first-seen text in deterministic batches of batch size 10 before each M15 decision. LLM matched uses the same five causal fields, while LLM full adds three structured fields, with H1 2025 calibration and July 2025-March 2026 descriptive economics.",
        body,
    )


def _side_calibration_notebook():
    body = [
        code(
            """
import hashlib
import json
import numpy as np
import pandas as pd
from IPython.display import display
from experiments.build_notebook_06 import MODEL_LABELS
from experiments.index_replication_protocol import MODEL_NAMES

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_side_calibration"
calibrators, selected, forward = {}, {}, {}

def feature_label(value):
    if value == "selected_base":
        return "Price/VIX base"
    if value == "deberta_matched":
        return "DeBERTa matched"
    if value.endswith("_matched"):
        return "LLM matched"
    if value.endswith("_full"):
        return "LLM full"
    return value.replace("_", " ").title()

for stream in STREAMS:
    root = ROOT / stream
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert result["q2_loaded"] is False and protocol["q2_loaded"] is False
    assert result["resumed_forward_policies"] == 36
    assert result["evidence_role"] == "secondary_reused_forward_posthoc_diagnostic"
    assert manifest["protocol_hash"] == protocol["protocol_hash"] and len(manifest["artifacts"]) == 115
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
    calibrators[stream] = pd.read_parquet(root / "calibrators.parquet")
    selected[stream] = pd.read_parquet(root / "h1_model_policies.parquet")
    forward[stream] = pd.read_parquet(root / "forward_summary.parquet")
    assert selected[stream]["model_name"].nunique() == 9
    assert forward[stream]["model_name"].nunique() == 9
    assert set(selected[stream]["model_name"]) == set(MODEL_NAMES)
    assert set(forward[stream]["model_name"]) == set(MODEL_NAMES)
"""
        ),
        md("## 2024 OOF calibration quality"),
        md("Method: Separate unweighted sigmoid mappings are fitted to blocked 2024 OOF SHORT and LONG probabilities at each model's already selected width, and lower Brier score means better probability calibration."),
        code(
            """
# calibration-quality-table
# non-economic-table
rows = []
for stream in STREAMS:
    for policy in selected[stream].itertuples(index=False):
        scope = calibrators[stream][
            calibrators[stream]["arm"].eq(policy.arm)
            & calibrators[stream]["model_name"].eq(policy.model_name)
        ].set_index("side")
        assert set(scope.index) == {"SHORT", "LONG"}
        rows.append({
            "Index": stream.upper(), "Features": feature_label(policy.arm),
            "Model": MODEL_LABELS[policy.model_name], "OOF rows": int(scope.loc["SHORT", "rows"]),
            "SHORT raw Brier": float(scope.loc["SHORT", "raw_brier"]),
            "SHORT calibrated Brier": float(scope.loc["SHORT", "calibrated_brier"]),
            "LONG raw Brier": float(scope.loc["LONG", "raw_brier"]),
            "LONG calibrated Brier": float(scope.loc["LONG", "calibrated_brier"]),
        })
quality = pd.DataFrame(rows)
quality["Mean Brier improvement"] = (
    quality[["SHORT raw Brier", "LONG raw Brier"]].mean(axis=1)
    - quality[["SHORT calibrated Brier", "LONG calibrated Brier"]].mean(axis=1)
)
quality = quality.sort_values(["Mean Brier improvement", "Index", "Model"], ascending=[False, True, True])
display(quality.round(4))
all_calibrators = pd.concat(calibrators.values(), ignore_index=True)
improved = int(all_calibrators["calibrated_brier"].lt(all_calibrators["raw_brier"]).sum())
print(f"Takeaway: Sigmoid calibration lowers OOF Brier score for {improved} of {len(all_calibrators)} SHORT/LONG mappings.")
"""
        ),
        md("## H1 side-policy selection"),
        md("Method: January-June 2025 tests the fixed 0.25-0.80 grid separately for SHORT and LONG, requires 50 trades, 15 per side, 3/6 positive months and positive Net, Sharpe and Sortino, then freezes one feature set per model."),
        code(
            """
# h1-side-policy-table
# economic-table
h1 = pd.concat([
    selected[stream].assign(Index=stream.upper()) for stream in STREAMS
], ignore_index=True).rename(columns={
    "arm": "Features", "model_name": "Model", "width_bps": "Width bps",
    "tau_short": "SHORT threshold", "tau_long": "LONG threshold", "h1_status": "H1 status",
    "trades": "Trades", "n_long": "LONG", "n_short": "SHORT", "positive_months": "Positive months",
    "net_return": "Net %", "daily_sharpe": "Sharpe", "daily_sortino": "Sortino",
})
h1["Features"] = h1["Features"].map(feature_label); h1["Model"] = h1["Model"].map(MODEL_LABELS)
h1["Net %"] *= 100
h1 = h1.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(h1[["Index", "Features", "Model", "Width bps", "SHORT threshold", "LONG threshold", "H1 status", "Trades", "LONG", "SHORT", "Positive months", "Net %", "Sharpe", "Sortino"]].round(3))
passed = int(h1["H1 status"].eq("Pass").sum())
print(f"Takeaway: {passed} of {len(h1)} model-level side policies satisfy the complete H1 gate before forward replay.")
"""
        ),
        md("## Reused-forward comparison"),
        md("Method: Each frozen H1 policy is replayed from July 2025-March 2026 beside the same model, feature set, width and original symmetric threshold; this inspected interval is diagnostic rather than independent confirmation."),
        code(
            """
# forward-side-policy-table
# economic-table
active = pd.concat([
    forward[stream].assign(Index=stream.upper()) for stream in STREAMS
], ignore_index=True).rename(columns={
    "arm": "Features", "model_name": "Model", "width_bps": "Width bps",
    "tau_short": "SHORT threshold", "tau_long": "LONG threshold", "h1_status": "H1 status",
    "trades": "Trades", "trades_per_day": "Trades/day", "n_long": "LONG", "n_short": "SHORT",
    "net_return": "Net %", "daily_sharpe": "Sharpe", "daily_sortino": "Sortino",
    "control_trades": "Control trades", "control_net_return": "Control Net %",
    "delta_trades": "Trade delta", "delta_short": "SHORT delta", "delta_net_return": "Net delta (pp)",
})
active["Features"] = active["Features"].map(feature_label); active["Model"] = active["Model"].map(MODEL_LABELS)
active["Net %"] *= 100; active["Control Net %"] *= 100; active["Net delta (pp)"] *= 100
numeric = ["Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Control trades", "Control Net %", "Trade delta", "SHORT delta", "Net delta (pp)"]
assert len(active) == 18 and np.isfinite(active[numeric].to_numpy(dtype=float)).all()
active = active.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(active[["Index", "Features", "Model", "Width bps", "SHORT threshold", "LONG threshold", "H1 status", *numeric]].round(3))
qualified = int(active["promising"].sum())
print(f"Takeaway: {qualified} of {len(active)} H1-selected policies increase trades, execute both sides and retain non-negative forward Net, Sharpe and Sortino.")
"""
        ),
    ]
    return _notebook(
        "06e - LONG/SHORT probability calibration",
        "All nine M15 models retain their selected label width while separate SHORT and LONG sigmoid mappings use blocked 2024 OOF probabilities. January-June H1 2025 selects two side-specific thresholds; July 2025-March 2026 replays the frozen policy against its symmetric control.",
        body,
    )


def _trade_coverage_notebook():
    body = [
        code(
            """
import hashlib
import json
import pandas as pd
from IPython.display import display
from experiments.build_notebook_06 import MODEL_LABELS

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_trade_coverage"
selected, forward = {}, {}

def feature_label(value):
    if value == "selected_base":
        return "Price/VIX base"
    if value == "deberta_matched":
        return "DeBERTa matched"
    if value.endswith("_matched"):
        return "LLM matched"
    if value.endswith("_full"):
        return "LLM full"
    return value.replace("_", " ").title()

for stream in STREAMS:
    result = json.loads((ROOT / stream / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / stream / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / stream / "manifest.json").read_text(encoding="utf-8"))
    assert result["q2_loaded"] is False and protocol["q2_loaded"] is False
    assert result["evidence_role"] == "secondary_reused_forward"
    assert result["models_screened_per_arm"] == 9 and result["selected_policy_rows"] == 4
    for name, expected in manifest["artifacts"].items():
        assert hashlib.sha256((ROOT / stream / name).read_bytes()).hexdigest() == expected
    selected[stream] = pd.read_parquet(ROOT / stream / "h1_selected_policies.parquet")
    forward[stream] = pd.read_parquet(ROOT / stream / "forward_summary.parquet")
"""
        ),
        md("## H1 coverage calibration"),
        md("Method: All nine models, three widths and eleven thresholds are screened in H1, requiring 50 trades, 15 per side, 3/6 positive months and positive Net, Sharpe and Sortino before maximum trade count breaks ties."),
        code(
            """
# economic-table
rows = []
for stream in STREAMS:
    scope = selected[stream].copy(); scope["Index"] = stream.upper(); rows.append(scope)
h1 = pd.concat(rows, ignore_index=True).rename(columns={
    "arm": "Features", "model_name": "Model", "width_bps": "Width bps", "tau": "Threshold",
    "models_screened": "Models screened", "eligible_models": "Eligible models", "eligible_policies": "Eligible policies",
    "trades": "Trades", "trades_per_day": "Trades/day", "n_long": "LONG", "n_short": "SHORT",
    "positive_months": "Positive months", "net_return": "Net %", "daily_sharpe": "Sharpe", "daily_sortino": "Sortino",
})
h1["Net %"] *= 100
h1["Model"] = h1["Model"].map(MODEL_LABELS)
h1["Features"] = h1["Features"].map(feature_label)
h1 = h1.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(h1[["Index", "Features", "Model", "Width bps", "Threshold", "Models screened", "Eligible models", "Eligible policies", "Trades", "Trades/day", "LONG", "SHORT", "Positive months", "Net %", "Sharpe", "Sortino"]].round(3))
print("Takeaway: The H1 rule selects one high-coverage policy for each of the four feature sets per index.")
"""
        ),
        md("## Reused-forward results"),
        md("Method: The eight H1-selected policies are fitted at July 2025 and evaluated through March 2026, with economic summaries restricted to policies that execute at least one trade."),
        code(
            """
# economic-table
active = pd.concat([
    forward[stream][forward[stream]["trades"].gt(0)].assign(Index=stream.upper())
    for stream in STREAMS
], ignore_index=True).rename(columns={
    "arm": "Features", "model_name": "Model", "trades": "Trades", "trades_per_day": "Trades/day",
    "n_long": "LONG", "n_short": "SHORT", "net_return": "Net %", "daily_sharpe": "Sharpe",
    "daily_sortino": "Sortino", "max_drawdown": "Max drawdown %",
})
active["Net %"] *= 100; active["Max drawdown %"] *= 100
active["Model"] = active["Model"].map(MODEL_LABELS)
active["Features"] = active["Features"].map(feature_label)
active = active.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
assert not active.empty
display(active[["Index", "Features", "Model", "Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]].round(3))
best = active.iloc[0]
print(f"Takeaway: {best['Index']} {best['Features']} {best['Model']} has the highest coverage-first Net at {best['Net %']:.2f}% on {int(best['Trades'])} trades; this reused-forward result is descriptive.")
"""
        ),
    ]
    return _notebook(
        "06f - Coverage-first final experiment",
        "The coverage-first experiment screens all nine M15 models across three widths and eleven confidence thresholds in H1 2025. A policy must reach 50 trades, both sides, 3/6 positive months and positive Net, Sharpe and Sortino before July 2025-March 2026 descriptive evaluation.",
        body,
    )


def _ensemble_notebook():
    body = [
        code(
            """
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from experiments.build_notebook_06 import MODEL_LABELS
from experiments.index_replication_protocol import MODEL_NAMES

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_all_model_ensemble"
VIX_ROOT = CODE_ROOT / "experiments" / "cache" / "index_replication"
assert len(MODEL_NAMES) == 9

def feature_label(value):
    return {
        "selected_base": "Price + VIX",
        "deberta_matched": "DeBERTa matched",
        "deepseek_matched": "LLM matched",
        "deepseek_full": "LLM full",
    }[value]

def variant_label(value):
    return {
        "soft_vote": "Equal probability average (soft vote)",
        "directional_majority": "Directional vote (5/9 majority)",
        "stack": "Causal logistic meta-model (stack)",
        "best_single": "Best single model",
    }[value]

results, selections, h1, controls, forward, vix = {}, {}, {}, {}, {}, {}
for stream in STREAMS:
    root = ROOT / stream
    results[stream] = json.loads((root / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    selections[stream] = json.loads((root / "h1_selection.json").read_text(encoding="utf-8"))
    h1[stream] = pd.read_parquet(root / "h1_selected_candidates.parquet")
    controls[stream] = pd.read_parquet(root / "h1_selected_control.parquet")
    forward[stream] = pd.read_parquet(root / "forward_summary.parquet")
    vix[stream] = json.loads((VIX_ROOT / stream / "vix_admission.json").read_text(encoding="utf-8"))
    assert results[stream]["q2_loaded"] is False and protocol["q2_loaded"] is False
    assert results[stream]["selected_base"] == "price_vix"
    assert results[stream]["models_per_ensemble"] == 9
    assert len(h1[stream]) == 12 and len(controls[stream]) == 1 and len(forward[stream]) == 13
    assert selections[stream]["forward_loaded"] is False
    assert selections[stream]["selection_data_end_exclusive"] == "2025-07-01T00:00:00+00:00"
    assert manifest["protocol_hash"] == protocol["protocol_hash"] and len(manifest["artifacts"]) == 50
    assert pd.Timestamp(results[stream]["max_prediction_timestamp"]) < pd.Timestamp("2026-04-01T00:00:00Z")
    assert not forward[stream].isna().any().any()
"""
        ),
        md(
            """
## How the nine-model ensembles are constructed

**Membership.** All nine describes membership, not a fourth ensemble rule: LogReg, Decision Tree, Random Forest, Linear SVM, XGBoost, CatBoost, MLP, LSTM and GRU contribute one three-class probability forecast each.

**Feature sets.** Four inputs are evaluated independently: Price + VIX; Price + VIX plus matched DeBERTa features; Price + VIX plus matched LLM features; and Price + VIX plus full causal LLM features. Applying three ensemble rules to each input gives `4 x 3 = 12` candidates per index, rather than combining the feature sets with one another.

**Causal timing.** January starts from 2024 OOF only; completed H1 labels are appended month by month through June; the final meta-model and policy are frozen before forward replay. July 2025-March 2026 is reused-forward evidence, while Q2-2026 remains sealed.

**H1 selection.** Eligibility requires at least 50 total trades, at least 15 LONG and 15 SHORT trades, and at least four positive months out of six. Constraint shortfall is minimised first, followed by Sortino, Net, trade count, width and threshold; Sharpe is reported but is not an H1 eligibility condition. Downstream admission additionally requires higher H1 Net and Sortino than the best eligible single model while retaining at least 80% of its trades.
"""
        ),
        md("## Inherited VIX input"),
        md("Method: The ensemble inherits the earlier 2024 admission decision, so Price + VIX is used only because each index passed the fixed model, fold and trade-retention gates."),
        code(
            """
# ensemble-vix-lineage-table
# non-economic-table
rows = []
for stream in STREAMS:
    decision = vix[stream]
    assert decision["selected_base"] == "price_vix" and decision["gate_complete"]
    rows.append({
        "Index": stream.upper(),
        "Input": "Price + VIX",
        "Positive models / 9": int(decision["metrics"]["positive_families"]),
        "Positive folds / 5": int(decision["metrics"]["positive_folds"]),
        "Median gate delta (pp)": 100 * float(decision["metrics"]["median_family_delta"]),
        "Trade retention %": 100 * float(decision["metrics"]["trade_retention"]),
    })
vix_view = pd.DataFrame(rows).sort_values(["Median gate delta (pp)", "Index"], ascending=[False, True])
display(vix_view.round({"Median gate delta (pp)": 3, "Trade retention %": 1}))
print("Takeaway: Both indices retain Price + VIX because the complete 2024 admission rule passed before ensemble construction.")
"""
        ),
        md("## Ensemble membership"),
        md("Method: Each registered model contributes its SHORT and LONG probabilities with equal voting weight, while the stack receives the same two probabilities as separate inputs."),
        code(
            """
# ensemble-membership-table
# non-economic-table
members = pd.DataFrame({
    "Model": [MODEL_LABELS[name] for name in MODEL_NAMES],
    "Voting input": ["P(SHORT), P(LONG)"] * len(MODEL_NAMES),
    "Equal-vote weight %": [100 / len(MODEL_NAMES)] * len(MODEL_NAMES),
    "Stack inputs": [2] * len(MODEL_NAMES),
}).sort_values("Model")
display(members.round({"Equal-vote weight %": 1}))
print("Takeaway: All nine registered model families contribute to each ensemble through the same probability interface.")
"""
        ),
        md("## Ensemble rules"),
        md("Method: The table defines how the same nine probability forecasts become one three-class ensemble decision within each feature set."),
        code(
            """
# ensemble-definition-table
# non-economic-table
definitions = pd.DataFrame([
    {
        "Ensemble rule": "Equal probability average (soft vote)",
        "Base models": 9,
        "How forecasts are combined": "Mean P(SHORT), P(FLAT), P(LONG), equal weight 1/9",
        "Trained combiner": "No",
        "When result is FLAT": "Largest mean class or confidence threshold",
    },
    {
        "Ensemble rule": "Directional vote (5/9 majority)",
        "Base models": 9,
        "How forecasts are combined": "At least five agreeing LONG or SHORT class votes",
        "Trained combiner": "No",
        "When result is FLAT": "Neither direction receives five votes",
    },
    {
        "Ensemble rule": "Causal logistic meta-model (stack)",
        "Base models": 9,
        "How forecasts are combined": "18 inputs: P(SHORT) and P(LONG) from each model; P(FLAT) is implied",
        "Trained combiner": "Yes: StandardScaler + balanced L2 LogisticRegression, C=0.1, seed 42",
        "When result is FLAT": "Meta-model class or confidence threshold",
    },
])
display(definitions)
print("Takeaway: All nine identifies the members, while probability averaging, directional voting and the causal logistic meta-model are the three alternative combination rules.")
"""
        ),
        md("## H1 ensemble policies"),
        md("Method: January-June 2025 first minimises shortfalls from 50 total trades, 15 LONG, 15 SHORT and four positive months, then ranks comparable rows by Sortino, Net, trade count, width and threshold; Sharpe is reported but is not an H1 eligibility condition."),
        code(
            """
# ensemble-h1-table
# economic-table
h1_view = pd.concat([
    h1[stream].assign(Index=stream.upper()) for stream in STREAMS
], ignore_index=True).rename(columns={
    "arm": "Features", "variant": "Ensemble", "width_bps": "Width bps", "tau": "Threshold",
    "trades": "Trades", "trades_per_day": "Trades/day", "n_long": "LONG", "n_short": "SHORT",
    "positive_months": "Positive months", "net_return": "Net %", "daily_sharpe": "Sharpe",
    "daily_sortino": "Sortino", "max_drawdown": "Max drawdown %",
})
h1_view["Features"] = h1_view["Features"].map(feature_label)
h1_view["Ensemble"] = h1_view["Ensemble"].map(variant_label)
h1_view["H1 status"] = h1_view["eligible"].map({True: "Eligible", False: "Below gate"})
h1_view["Net %"] *= 100; h1_view["Max drawdown %"] *= 100
h1_view = h1_view.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(h1_view[["Index", "Features", "Ensemble", "Width bps", "Threshold", "H1 status", "Trades", "Trades/day", "LONG", "SHORT", "Positive months", "Net %", "Sharpe", "Sortino", "Max drawdown %"]].round(3))
eligible_count = int(h1_view["H1 status"].eq("Eligible").sum())
print(f"Takeaway: {eligible_count} of {len(h1_view)} feature-and-rule policies satisfy the complete H1 gate.")
"""
        ),
        md("## Frozen H1 decision against the single-model control"),
        md("Method: Within each index, the strongest constraint-first ensemble is compared with the strongest eligible single model, requiring higher Net, higher Sortino and at least 80% of its trades for downstream admission."),
        code(
            """
# ensemble-decision-table
# economic-table
rows = []
for stream in STREAMS:
    selection = selections[stream]
    winner = h1[stream][h1[stream]["candidate_id"].eq(selection["h1_winner"]["candidate_id"])].iloc[0]
    control = controls[stream].iloc[0]
    gate = selection["promotion"]
    rows.extend([
        {
            "Index": stream.upper(), "Role": "Best-ranked H1 ensemble", "Features": feature_label(winner["arm"]),
            "Policy": variant_label(winner["variant"]), "H1 status": "Eligible" if winner["eligible"] else "Below gate",
            "Downstream gate": "Accepted" if gate["promoted"] else "Not accepted", "Trades": int(winner["trades"]),
            "LONG": int(winner["n_long"]), "SHORT": int(winner["n_short"]), "Net %": 100 * float(winner["net_return"]),
            "Sharpe": float(winner["daily_sharpe"]), "Sortino": float(winner["daily_sortino"]),
            "Net delta vs control (pp)": 100 * (float(winner["net_return"]) - float(control["net_return"])),
            "Sortino delta vs control": float(winner["daily_sortino"]) - float(control["daily_sortino"]),
            "Trade retention %": 100 * float(gate["trade_retention"]),
        },
        {
            "Index": stream.upper(), "Role": "Best single-model control", "Features": feature_label(control["arm"]),
            "Policy": MODEL_LABELS[control["model_name"]], "H1 status": "Eligible" if control["eligible"] else "Below gate",
            "Downstream gate": "Reference", "Trades": int(control["trades"]), "LONG": int(control["n_long"]),
            "SHORT": int(control["n_short"]), "Net %": 100 * float(control["net_return"]),
            "Sharpe": float(control["daily_sharpe"]), "Sortino": float(control["daily_sortino"]),
            "Net delta vs control (pp)": 0.0, "Sortino delta vs control": 0.0, "Trade retention %": 100.0,
        },
    ])
decision_view = pd.DataFrame(rows).sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(decision_view[["Index", "Role", "Features", "Policy", "H1 status", "Downstream gate", "Trades", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Net delta vs control (pp)", "Sortino delta vs control", "Trade retention %"]].round(3))
usa_winner = decision_view[decision_view["Index"].eq("USA500") & decision_view["Role"].eq("Best-ranked H1 ensemble")].iloc[0]
usa_control = decision_view[decision_view["Index"].eq("USA500") & decision_view["Role"].eq("Best single-model control")].iloc[0]
tech_winner = decision_view[decision_view["Index"].eq("USATECH") & decision_view["Role"].eq("Best-ranked H1 ensemble")].iloc[0]
tech_control = decision_view[decision_view["Index"].eq("USATECH") & decision_view["Role"].eq("Best single-model control")].iloc[0]
print(f"Takeaway: USA500 has 0/12 eligible ensembles and its best-ranked DeBERTa probability average ({usa_winner['Net %']:.2f}% Net, {usa_winner['Sortino']:.3f} Sortino) trails SVM ({usa_control['Net %']:.2f}%, {usa_control['Sortino']:.3f}), while USATECH has 1/12 eligible ensemble and its LLM matched probability average ({tech_winner['Net %']:.2f}%, {tech_winner['Sortino']:.3f}) trails LSTM ({tech_control['Net %']:.2f}%, {tech_control['Sortino']:.3f}).")
"""
        ),
        md("## Reused-forward results"),
        md("Method: All twelve frozen ensemble candidates and the single-model control per index are replayed from July 2025 through March 2026 as descriptive evidence, including policies with zero executed trades."),
        code(
            """
# ensemble-forward-table
# economic-table
forward_view = pd.concat([
    forward[stream].assign(Index=stream.upper()) for stream in STREAMS
], ignore_index=True).rename(columns={
    "arm": "Features", "variant": "Ensemble", "trades": "Trades", "trades_per_day": "Trades/day",
    "n_long": "LONG", "n_short": "SHORT", "net_return": "Net %", "daily_sharpe": "Sharpe",
    "daily_sortino": "Sortino", "max_drawdown": "Max drawdown %",
})
forward_view["Features"] = forward_view["Features"].map(feature_label)
forward_view["Ensemble"] = forward_view["Ensemble"].map(variant_label)
forward_view["Role"] = forward_view["role"].map({"ensemble": "Nine-model ensemble", "best_single_control": "Best single-model control"})
forward_view["Model"] = forward_view["model_name"].map(lambda name: "Nine base models" if name == "ALL_NINE" else MODEL_LABELS[name])
forward_view["H1 status"] = forward_view["h1_eligible"].map({True: "Eligible", False: "Below gate"})
forward_view["Forward status"] = forward_view["status"].map({"traded": "Traded", "no_trades": "No trades"})
forward_view["Net %"] *= 100; forward_view["Max drawdown %"] *= 100
numeric = ["Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]
assert len(forward_view) == 26 and np.isfinite(forward_view[numeric].to_numpy(dtype=float)).all()
forward_view = forward_view.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(forward_view[["Index", "Role", "Features", "Ensemble", "Model", "H1 status", "Forward status", *numeric]].round(3))
usa_forward = forward_view[forward_view["Index"].eq("USA500") & forward_view["Features"].eq("Price + VIX") & forward_view["Ensemble"].eq("Directional vote (5/9 majority)")].iloc[0]
usa_h1 = h1_view[h1_view["Index"].eq("USA500") & h1_view["Features"].eq("Price + VIX") & h1_view["Ensemble"].eq("Directional vote (5/9 majority)")].iloc[0]
tech_forward = forward_view[forward_view["Index"].eq("USATECH") & forward_view["Features"].eq("LLM full") & forward_view["Ensemble"].eq("Equal probability average (soft vote)")].iloc[0]
tech_h1 = h1_view[h1_view["Index"].eq("USATECH") & h1_view["Features"].eq("LLM full") & h1_view["Ensemble"].eq("Equal probability average (soft vote)")].iloc[0]
print(f"Takeaway: USA500 Price + VIX directional vote is {usa_forward['Net %']:.2f}% on reused forward but was H1-ineligible at {usa_h1['Net %']:.2f}%, while USATECH LLM full probability average is {tech_forward['Net %']:.2f}% but was H1-ineligible at {tech_h1['Net %']:.2f}%; these descriptive rows do not reopen H1 selection.")
"""
        ),
        md("## Quantity-quality view"),
        md("Method: Each point is one ensemble or control that executes at least one reused-forward trade, with frequency on the horizontal axis and Net on the vertical axis."),
        code(
            """
# economic-figure
plot_view = forward_view[forward_view["Trades"].gt(0)].copy()
colors = {"USA500": "#457b9d", "USATECH": "#e76f51"}
markers = {"Nine-model ensemble": "o", "Best single-model control": "X"}
fig, ax = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
for (index, role), scope in plot_view.groupby(["Index", "Role"], sort=False):
    ax.scatter(scope["Trades/day"], scope["Net %"], color=colors[index], marker=markers[role],
               alpha=.82, label=f"{index} - {role}")
ax.axhline(0, color="black", lw=.8); ax.set_xlabel("Trades per calendar day"); ax.set_ylabel("Net return (%)")
ax.set_title("Nine-model ensembles: quantity versus quality"); ax.grid(alpha=.2); ax.legend(fontsize=7)
plt.show()
print("Takeaway: Higher trade frequency does not consistently produce higher Net across the three nine-model combination rules.")
"""
        ),
    ]
    return _notebook(
        "06g - Ensembles of nine base models",
        "Nine M15 base models - LogReg, Decision Tree, Random Forest, Linear SVM, XGBoost, CatBoost, MLP, LSTM and GRU - are combined separately within four Price + VIX feature sets using probability average, 5/9 directional vote and a causal logistic meta-model. Frozen 2024 OOF probabilities initialise H1 2025; January-June selects one of 4 x 3 = 12 candidates per index before July 2025-March 2026 descriptive replay.",
        body,
    )


def _channels_notebook():
    body = [
        code(
            """
import json
import numpy as np
import pandas as pd
from IPython.display import display

STREAMS = ("usa500", "usatech")
ROOT = CODE_ROOT / "experiments" / "cache" / "index_channel_replication"
summary, funnels, ledgers, windows = {}, {}, {}, {}

def daily_risk(ledger, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    work = ledger.copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True)
    work = work[work["entry_time"].ge(start) & work["entry_time"].lt(end)]
    calendar = pd.date_range(start.normalize(), end.normalize(), freq="1D", inclusive="left", tz="UTC")
    daily = work.set_index("entry_time")["net_return"].resample("1D").sum().reindex(calendar, fill_value=0.0)
    mean = float(daily.mean()) if len(daily) else 0.0
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    downside = daily[daily.lt(0)]
    downside_scale = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    sharpe = mean / std * np.sqrt(365.0) if std > 0 else 0.0
    sortino = mean / downside_scale * np.sqrt(365.0) if downside_scale > 0 else 0.0
    equity = (1.0 + daily).cumprod(); drawdown = equity / equity.cummax() - 1.0
    return {"Net %": 100 * float(work["net_return"].sum()), "Sharpe": sharpe, "Sortino": sortino,
            "Max drawdown %": 100 * float(drawdown.min()) if len(drawdown) else 0.0,
            "LONG": int(work["side"].astype(str).str.lower().eq("long").sum()),
            "SHORT": int(work["side"].astype(str).str.lower().eq("short").sum())}

for stream in STREAMS:
    result = json.loads((ROOT / stream / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / stream / "protocol.json").read_text(encoding="utf-8"))
    assert result["q2_2026_loaded"] is False and protocol["q2_2026_loaded"] is False
    summary[stream] = pd.read_parquet(ROOT / stream / "stage_rr_summary.parquet")
    funnels[stream] = pd.read_parquet(ROOT / stream / "stage_funnel.parquet")
    windows[stream] = protocol["stages"]
    for stage in ("development", "h1_2025", "forward"):
        for rr in (2, 3, 5):
            ledgers[(stream, stage, rr)] = pd.read_parquet(ROOT / stream / "ledgers" / f"{stage}_rr{rr}.parquet")
"""
        ),
        md("## Primary forward result: frozen RR2"),
        md("Method: Window60/RR2 is the only predeclared primary channel policy; forward Net, Sharpe and Sortino use the complete zero-filled UTC-day calendar."),
        code(
            """
# economic-table
rows = []
for stream in STREAMS:
    base = summary[stream][summary[stream]["stage"].eq("forward") & summary[stream]["rr_multiple"].eq(2.0)].iloc[0]
    metrics = daily_risk(ledgers[(stream, "forward", 2)], *windows[stream]["forward"])
    rows.append({"Index": stream.upper(), "Policy": "Window60 / RR2", "Trades": int(base["filled_trades"]),
                 "Trades/day": float(base["trades_per_day"]), **metrics})
primary = pd.DataFrame(rows)
primary["Result"] = np.where(primary["Net %"].gt(0), "Positive", "Negative")
primary = primary.sort_values(["Net %", "Sortino"], ascending=False)
display(primary[["Index", "Policy", "Result", "Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]].round(3))
print("Takeaway: The frozen RR2 policy is negative on both indices: USA500 is near flat at -0.17%, while USATECH loses 3.12%.")
"""
        ),
        md("## Forward target sensitivity"),
        md("Method: RR3 and RR5 reuse the RR2 signals and change only the profit target, providing sensitivity results beside the pre-specified primary target."),
        code(
            """
# economic-table
rows = []
for stream in STREAMS:
    for rr in (2, 3, 5):
        base = summary[stream][summary[stream]["stage"].eq("forward") & summary[stream]["rr_multiple"].eq(float(rr))].iloc[0]
        metrics = daily_risk(ledgers[(stream, "forward", rr)], *windows[stream]["forward"])
        rows.append({"Index": stream.upper(), "Target": f"RR{rr}", "Role": "Primary" if rr == 2 else "Sensitivity only",
                     "Trades": int(base["filled_trades"]), "Trades/day": float(base["trades_per_day"]), **metrics})
sensitivity = pd.DataFrame(rows).sort_values(["Net %", "Sortino"], ascending=False)
display(sensitivity[["Index", "Target", "Role", "Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]].round(3))
print("Takeaway: Only USA500 RR3 is positive (+0.77%), but it is a sensitivity result and does not overturn the negative frozen RR2 conclusion.")
"""
        ),
        md("## What creates the RR2 result"),
        md("Method: RR2 trades are split by setup and LONG/SHORT side on the same complete daily calendar to measure each subgroup's contribution."),
        code(
            """
# economic-table
rows = []
for stream in STREAMS:
    ledger = ledgers[(stream, "forward", 2)]
    for (setup, side), group in ledger.groupby(["setup_type", "side"], sort=True):
        if len(group) == 0:
            continue
        metrics = daily_risk(group, *windows[stream]["forward"])
        rows.append({"Index": stream.upper(), "Setup": setup, "Side": str(side).upper(), "Trades": len(group),
                     "Win rate %": 100 * float(group["r_net"].gt(0).mean()), **metrics})
side_view = pd.DataFrame(rows).sort_values(["Net %", "Sortino"], ascending=False)
display(side_view[["Index", "Setup", "Side", "Trades", "Win rate %", "Net %", "Sharpe", "Sortino", "Max drawdown %"]].round(3))
print("Takeaway: Some subgroups are positive, especially USATECH edge-rejection LONG, but USATECH SHORT losses make the complete frozen RR2 policy negative.")
"""
        ),
        md("## Forward opportunity funnel"),
        md("Method: Counts show raw setups at T1, freshness-qualified setups at T2, rail-qualified setups at T3 and executable entries at ENTRY for each setup and side."),
        code(
            """
# non-economic-table
funnel_view = pd.concat([funnels[stream].assign(Index=stream.upper()) for stream in STREAMS], ignore_index=True)
funnel_view = funnel_view[funnel_view["evaluation_stage"].eq("forward")].pivot_table(
    index=["Index", "setup_type", "side"], columns="funnel_stage", values="count", aggfunc="first", fill_value=0
).reindex(columns=["T1", "T2", "T3", "ENTRY"], fill_value=0).reset_index().rename(columns={
    "setup_type": "Setup", "side": "Side",
})
funnel_view = funnel_view[funnel_view["T1"].gt(0)].sort_values(["ENTRY", "T3", "T2", "T1"], ascending=False)
funnel_view["Side"] = funnel_view["Side"].str.upper()
display(funnel_view[["Index", "Setup", "Side", "T1", "T2", "T3", "ENTRY"]])
print("Takeaway: Most raw T1 opportunities fail the freshness or causal-rail conditions before reaching ENTRY.")
"""
        ),
    ]
    return _notebook(
        "06h - Channel replication",
        "Completed 1H bars define window-60 channels, 5-minute bars generate decisions and native 1-minute bars execute trades. RR2 is the pre-specified primary target, while RR3 and RR5 provide sensitivity analyses over the same July 2025-March 2026 signals.",
        body,
    )


def _comparison_notebook():
    body = [
        code(
            """
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from experiments.build_notebook_06 import MODEL_LABELS

STREAMS = ("usa500", "usatech")
BASE = CODE_ROOT / "experiments" / "cache"

def feature_label(value):
    if value == "selected_base":
        return "Price/VIX base"
    if value == "deberta_matched":
        return "DeBERTa matched"
    if value.endswith("_matched"):
        return "LLM matched"
    if value.endswith("_full"):
        return "LLM full"
    return value.replace("_", " ").title()

def ensemble_label(value):
    return {
        "soft_vote": "Equal probability average (soft vote)",
        "directional_majority": "Directional vote (5/9 majority)",
        "stack": "Causal logistic meta-model (stack)",
        "best_single": "Best single model",
    }[value]

def channel_metrics(ledger, start, end):
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    work = ledger.copy(); work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True)
    calendar = pd.date_range(start.normalize(), end.normalize(), freq="1D", inclusive="left", tz="UTC")
    daily = work.set_index("entry_time")["net_return"].resample("1D").sum().reindex(calendar, fill_value=0.0)
    mean, std = float(daily.mean()), float(daily.std(ddof=1))
    downside = daily[daily.lt(0)]; scale = float(np.sqrt(np.mean(np.square(downside)))) if len(downside) else 0.0
    equity = (1 + daily).cumprod(); drawdown = equity / equity.cummax() - 1
    return {"trades": len(work), "trades_per_day": len(work) / len(calendar),
            "n_long": int(work["side"].astype(str).str.lower().eq("long").sum()),
            "n_short": int(work["side"].astype(str).str.lower().eq("short").sum()),
            "net_return": float(work["net_return"].sum()),
            "daily_sharpe": mean / std * np.sqrt(365) if std > 0 else 0.0,
            "daily_sortino": mean / scale * np.sqrt(365) if scale > 0 else 0.0,
            "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0}

rows = []
for stream in STREAMS:
    original = pd.read_parquet(BASE / "index_replication" / stream / "forward_summary.parquet")
    original = original[original["trades"].gt(0)]
    for row in original.itertuples(index=False):
        rows.append({"Index": stream.upper(), "Experiment": "Original H1 quality-first", "Policy": f"{feature_label(row.arm)} / {MODEL_LABELS[row.model_name]}",
                     **{name: getattr(row, name) for name in ("trades", "trades_per_day", "n_long", "n_short", "net_return", "daily_sharpe", "daily_sortino", "max_drawdown")}})
    coverage_result = json.loads((BASE / "index_trade_coverage" / stream / "result.json").read_text(encoding="utf-8"))
    assert coverage_result["q2_loaded"] is False and coverage_result["evidence_role"] == "secondary_reused_forward"
    coverage = pd.read_parquet(BASE / "index_trade_coverage" / stream / "forward_summary.parquet")
    coverage = coverage[coverage["trades"].gt(0)]
    for row in coverage.itertuples(index=False):
        rows.append({"Index": stream.upper(), "Experiment": "Coverage-first", "Policy": f"{feature_label(row.arm)} / {MODEL_LABELS[row.model_name]}",
                     **{name: getattr(row, name) for name in ("trades", "trades_per_day", "n_long", "n_short", "net_return", "daily_sharpe", "daily_sortino", "max_drawdown")}})
    side_result = json.loads((BASE / "index_side_calibration" / stream / "result.json").read_text(encoding="utf-8"))
    assert side_result["q2_loaded"] is False and side_result["evidence_role"] == "secondary_reused_forward_posthoc_diagnostic"
    side = pd.read_parquet(BASE / "index_side_calibration" / stream / "forward_summary.parquet")
    side = side[side["trades"].gt(0)]
    for row in side.itertuples(index=False):
        rows.append({"Index": stream.upper(), "Experiment": "Side-calibrated", "Policy": f"{feature_label(row.arm)} / {MODEL_LABELS[row.model_name]}",
                     **{name: getattr(row, name) for name in ("trades", "trades_per_day", "n_long", "n_short", "net_return", "daily_sharpe", "daily_sortino", "max_drawdown")}})
    ensemble_result = json.loads((BASE / "index_all_model_ensemble" / stream / "result.json").read_text(encoding="utf-8"))
    assert ensemble_result["q2_loaded"] is False and ensemble_result["evidence_role"] == "secondary_reused_forward_diagnostic"
    ensemble = pd.read_parquet(BASE / "index_all_model_ensemble" / stream / "forward_summary.parquet")
    ensemble = ensemble[ensemble["trades"].gt(0)]
    for row in ensemble.itertuples(index=False):
        experiment = "Nine-model ensemble" if row.role == "ensemble" else "Best single-model control"
        model = "Nine base models" if row.model_name == "ALL_NINE" else MODEL_LABELS[row.model_name]
        rows.append({"Index": stream.upper(), "Experiment": experiment,
                     "Policy": f"{feature_label(row.arm)} / {ensemble_label(row.variant)} / {model}",
                     **{name: getattr(row, name) for name in ("trades", "trades_per_day", "n_long", "n_short", "net_return", "daily_sharpe", "daily_sortino", "max_drawdown")}})
    channel_root = BASE / "index_channel_replication" / stream
    channel_protocol = json.loads((channel_root / "protocol.json").read_text(encoding="utf-8"))
    channel = pd.read_parquet(channel_root / "ledgers" / "forward_rr2.parquet")
    rows.append({"Index": stream.upper(), "Experiment": "Channel RR2", "Policy": "window60 / RR2",
                 **channel_metrics(channel, *channel_protocol["stages"]["forward"])})
active = pd.DataFrame(rows)
"""
        ),
        md("## Cross-index economic results"),
        md("Method: The synthesis compares realised July 2025-March 2026 policies separately by index, with each nine-model row applying one of the three frozen 06g combination rules to one feature set rather than pooling feature sets or correlated policies."),
        code(
            """
# economic-table
view = active.rename(columns={
    "trades": "Trades", "trades_per_day": "Trades/day", "n_long": "LONG", "n_short": "SHORT",
    "net_return": "Net %", "daily_sharpe": "Sharpe", "daily_sortino": "Sortino", "max_drawdown": "Max drawdown %",
})
view["Net %"] *= 100; view["Max drawdown %"] *= 100
view = view.sort_values(["Net %", "Sortino", "Sharpe"], ascending=False)
display(view[["Index", "Experiment", "Policy", "Trades", "Trades/day", "LONG", "SHORT", "Net %", "Sharpe", "Sortino", "Max drawdown %"]].round(3))
best = view.iloc[0]
print(f"Takeaway: {best['Index']} {best['Experiment']} has the highest Net at {best['Net %']:.2f}% with {best['Trades/day']:.3f} trades per day.")
"""
        ),
        md("## One quantity-quality figure"),
        md("Method: Each point is one active policy; horizontal position is trade frequency and vertical position is net return, with no portfolio aggregation."),
        code(
            """
# economic-figure
colors = {"USA500": "#457b9d", "USATECH": "#e76f51"}
markers = {"Original H1 quality-first": "o", "Coverage-first": "s", "Side-calibrated": "D",
           "Nine-model ensemble": "P", "Best single-model control": "X", "Channel RR2": "^"}
fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
for (index, experiment), scope in active.groupby(["Index", "Experiment"], sort=False):
    ax.scatter(scope["trades_per_day"], 100 * scope["net_return"], label=f"{index} - {experiment}",
               color=colors[index], marker=markers[experiment], alpha=.85)
ax.axhline(0, color="black", lw=.8); ax.set_xlabel("Trades per calendar day"); ax.set_ylabel("Net return (%)")
ax.set_title("Active index policies: quantity versus quality"); ax.grid(alpha=.2); ax.legend(fontsize=7)
plt.show()
print("Takeaway: Higher trade count is useful only where the point does not sacrifice the accompanying economic quality.")
"""
        ),
        md("Paired inference: A HAC/Holm sentiment comparison is not estimable because no base/sentiment pair for the same model passed the H1 gate on both sides."),
    ]
    return _notebook(
        "06i - Cross-index comparison",
        "This cross-index synthesis compares original H1, side-calibrated, coverage-first, nine-model ensemble and channel policies with realised trades during July 2025-March 2026. The three ensemble rules are defined in 06g, and each feature set remains separate; correlated policies are evaluated independently rather than summed.",
        body,
    )


def build_notebooks():
    return {
        "nine_models": _nine_models_notebook(),
        "vix": _vix_notebook(),
        "deberta": _deberta_notebook(),
        "llm": _llm_notebook(),
        "ensemble": _ensemble_notebook(),
        "comparison": _comparison_notebook(),
    }


def _require_complete_results() -> None:
    paths: list[Path] = []
    for stream in ("usa500", "usatech"):
        replication = CODE_ROOT / "experiments" / "cache" / "index_replication" / stream
        channel = CODE_ROOT / "experiments" / "cache" / "index_channel_replication" / stream
        coverage = CODE_ROOT / "experiments" / "cache" / "index_trade_coverage" / stream
        all_model = CODE_ROOT / "experiments" / "cache" / "index_all_model_forward" / stream
        side = CODE_ROOT / "experiments" / "cache" / "index_side_calibration" / stream
        ensemble = CODE_ROOT / "experiments" / "cache" / "index_all_model_ensemble" / stream
        paths.extend(
            [
                replication / "result.json",
                replication / "vix_admission.json",
                replication / "vix_gate_paired_2024.parquet",
                replication / "classification_2024.parquet",
                replication / "h1_selected_policies.parquet",
                replication / "forward_summary.parquet",
                channel / "result.json",
                channel / "protocol.json",
                channel / "stage_rr_summary.parquet",
                channel / "stage_funnel.parquet",
                coverage / "result.json",
                coverage / "protocol.json",
                coverage / "manifest.json",
                coverage / "h1_selected_policies.parquet",
                coverage / "forward_summary.parquet",
                all_model / "result.json",
                all_model / "protocol.json",
                all_model / "manifest.json",
                all_model / "h1_selected_policies.parquet",
                all_model / "forward_summary.parquet",
                side / "result.json",
                side / "protocol.json",
                side / "manifest.json",
                side / "calibrators.parquet",
                side / "h1_arm_policies.parquet",
                side / "h1_model_policies.parquet",
                side / "forward_arm_summary.parquet",
                side / "forward_summary.parquet",
                ensemble / "result.json",
                ensemble / "protocol.json",
                ensemble / "manifest.json",
                ensemble / "h1_selected_candidates.parquet",
                ensemble / "h1_selected_control.parquet",
                ensemble / "h1_selection.json",
                ensemble / "forward_summary.parquet",
            ]
        )
        for stage in ("development", "h1_2025", "forward"):
            for rr in (2, 3, 5):
                paths.append(channel / "ledgers" / f"{stage}_rr{rr}.parquet")
        for prefix in ("", "direct_events_"):
            paths.extend(
                [
                    CODE_ROOT / "sentiment" / "raw" / f"scores_{prefix}{stream}.manifest.json",
                    CODE_ROOT / "sentiment" / "raw" / f"scores_llm_{prefix}{stream}.manifest.json",
                ]
            )
    identity_paths = list((CODE_ROOT / "sentiment" / "raw").glob("index_*_identity.json"))
    if len(identity_paths) != 1:
        raise FileNotFoundError("Notebook 06 requires exactly one frozen LLM identity")
    paths.extend(identity_paths)
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Notebook 06 requires completed sealed artifacts:\n" + "\n".join(missing))


def execute_notebooks() -> dict[str, Path]:
    _require_complete_results()
    outputs = {}
    for name, notebook in build_notebooks().items():
        executed = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name="msc-code",
            resources={"metadata": {"path": str(CODE_ROOT)}},
        ).execute()
        nbf.validate(executed)
        path = NOTEBOOK_PATHS[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(executed, path)
        outputs[name] = path
    return outputs


def main() -> int:
    print(json.dumps({key: str(value) for key, value in execute_notebooks().items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MODEL_LABELS",
    "NOTEBOOK_PATHS",
    "build_notebooks",
    "execute_notebooks",
    "validate_llm_score_manifest",
]
