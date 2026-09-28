from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re

import nbformat
import numpy as np
import pandas as pd
import pytest

import experiments.build_notebook_06 as notebook_06_builder
from experiments.build_notebook_06 import (
    NOTEBOOK_PATHS,
    build_notebooks,
    validate_llm_score_manifest,
)


CODE_ROOT = Path(__file__).parents[1]
EXPECTED = {
    "nine_models": CODE_ROOT / "notebooks" / "04_RQ1_E_indices_nine_model_benchmark.ipynb",
    "vix": CODE_ROOT / "notebooks" / "05_RQ1_F_indices_VIX_ablation.ipynb",
    "deberta": CODE_ROOT / "notebooks" / "15_RQ3_D_indices_DeBERTa_sentiment.ipynb",
    "llm": CODE_ROOT / "notebooks" / "16_RQ3_E_indices_LLM_sentiment.ipynb",
    "ensemble": CODE_ROOT / "notebooks" / "10_RQ2_H_indices_all_model_ensemble.ipynb",
    "comparison": CODE_ROOT / "notebooks" / "11_RQ2_I_indices_policy_comparison.ipynb",
}
DISPLAY_MODELS = {
    "LogReg",
    "Decision Tree",
    "Random Forest",
    "Linear SVM",
    "XGBoost",
    "CatBoost",
    "MLP",
    "LSTM",
    "GRU",
}
READER_WORKFLOW_PATTERNS = (
    r"keeps? every model visible",
    r"no model is hidden",
    r"every model is listed",
    r"remains? visible",
    r"forward routing",
    r"chosen randomly",
    r"cannot suppress",
    r"cannot revise",
    r"cannot change selection",
    r"promot",
    r"empty controls",
    r"not displayed",
    r"no empty inferential table",
)


def test_llm_manifest_validator_api_exists():
    assert callable(getattr(notebook_06_builder, "validate_llm_score_manifest", None))


def _source(notebook) -> str:
    return "\n".join(cell.source for cell in notebook.cells)


def _reader_copy(notebook) -> str:
    visible = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
    visible.extend(
        line.strip()
        for cell in notebook.cells
        if cell.cell_type == "code"
        for line in cell.source.splitlines()
        if "Takeaway:" in line
    )
    return "\n".join(visible)


def _is_one_sentence(text: str, prefix: str) -> bool:
    current = text.strip()
    return (
        current.startswith(prefix)
        and "\n" not in current
        and len(re.findall(r"[.!?](?=\s|$)", current)) == 1
    )


class _TableParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"} and self._row is not None:
            self._cell = []

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self._row is not None and self._cell is not None:
            self._row.append("".join(self._cell).strip())
            self._cell = None
        elif tag == "tr" and self._row is not None:
            if self._row:
                self.rows.append(self._row)
            self._row = None


def _html_frame(source: str) -> pd.DataFrame:
    parser = _TableParser()
    parser.feed(source)
    assert len(parser.rows) >= 2
    width = len(parser.rows[0])
    rows = [row for row in parser.rows if len(row) == width]
    return pd.DataFrame(rows[1:], columns=rows[0])


def _cell_tables(cell) -> list[pd.DataFrame]:
    return [
        _html_frame(output.get("data", {}).get("text/html", ""))
        for output in cell.get("outputs", [])
        if output.get("output_type") in {"display_data", "execute_result"}
        and "<table" in output.get("data", {}).get("text/html", "").lower()
    ]


def test_builder_returns_six_experiment_isolated_readers():
    notebooks = build_notebooks()

    assert NOTEBOOK_PATHS == EXPECTED
    assert list(NOTEBOOK_PATHS) == list(EXPECTED)
    assert list(notebooks) == list(EXPECTED)
    assert set(notebooks) == set(EXPECTED)
    for notebook in notebooks.values():
        source = _source(notebook).lower()
        assert "methodology" in source
        assert "usa500" in source and "usatech" in source
        assert "2026-04-01" in source
        assert "fit(" not in source
        assert "ollama.chat" not in source


def test_each_reader_contains_only_its_declared_experiment():
    source = {name: _source(notebook).lower() for name, notebook in build_notebooks().items()}

    assert "price-only" in source["vix"] and "price+vix" in source["vix"]
    assert "simple majority" in source["vix"] and "5/9" in source["vix"]
    assert "deberta" not in source["vix"] and "llm" not in source["vix"]

    assert "nine models" in source["nine_models"] and "2024 oof" in source["nine_models"]
    assert "deberta" not in source["nine_models"] and "llm" not in source["nine_models"]

    assert "deberta" in source["deberta"] and "all nine" in source["deberta"]
    assert "llm" not in source["deberta"]

    assert "llm scorer" in source["llm"]
    assert "all nine" in source["llm"]
    assert "batch size 10" in source["llm"] or '"batch_size"] == 10' in source["llm"]
    assert "deberta" not in source["llm"]
    assert "deepseek" not in source["llm"]

    assert "ensembles of nine base models" in source["ensemble"]
    assert "equal probability average" in source["ensemble"]
    assert "directional vote" in source["ensemble"] and "5/9" in source["ensemble"]
    assert "causal logistic meta-model" in source["ensemble"]
    assert "price + vix" in source["ensemble"]
    assert "index_all_model_ensemble" in source["ensemble"]



def test_vix_reader_reports_numeric_family_effects_not_boolean_votes():
    source = _source(build_notebooks()["vix"])

    assert "# vix-numeric-table" in source
    assert "Price-only gate net %" in source
    assert "Price + VIX gate net %" in source
    assert "Gate net delta (pp)" in source
    assert "Price trades" in source and "Price + VIX trades" in source
    assert "VIX improves net result" not in source


def test_sentiment_forward_results_use_all_model_artifacts():
    sources = {name: _source(build_notebooks()[name]) for name in ("deberta", "llm")}

    for source in sources.values():
        assert "# forward-all-models" in source
        assert "H1 status" in source
        assert "index_all_model_forward" in source
        assert "diagnostic" in source.lower()
        assert "H1 decision" not in source
        assert "pd.NA" not in source


def test_reader_copy_uses_no_editorial_workflow_language():
    for name, notebook in build_notebooks().items():
        copy = _reader_copy(notebook)
        for pattern in READER_WORKFLOW_PATTERNS:
            assert re.search(pattern, copy, flags=re.IGNORECASE) is None, (
                name,
                pattern,
            )


def test_baseline_and_sentiment_readers_use_reader_first_status_terms():
    for name in ("nine_models", "deberta", "llm"):
        source = _source(build_notebooks()[name])
        assert "## Forward results" in source
        assert "Forward routing" not in source
        assert "H1 status" in source
        assert "H1 decision" not in source
        assert 'True: "Pass"' in source
        assert 'False: "Below gate"' in source


def test_opening_methodology_is_short_complete_and_notebook_specific():
    required = {
        "nine_models": ("m15", "nine models", "2024", "h1 2025", "july 2025"),
        "vix": ("m15", "vix", "2024", "five", "nine model"),
        "deberta": ("m15", "deberta", "first-seen", "h1 2025", "july 2025"),
        "llm": ("m15", "llm matched", "llm full", "batch size 10", "h1 2025"),
        "ensemble": ("m15", "nine base models", "2024 oof", "probability average", "directional vote", "logistic meta-model", "h1 2025", "july 2025"),
        "comparison": ("original", "coverage", "nine-model ensemble", "channel", "july 2025", "correlated"),
    }
    for name, notebook in build_notebooks().items():
        methodology = notebook.cells[1].source.lower()
        assert len(methodology.split()) <= 145, name
        for common in ("usa500", "usatech", "2 bps", "3 bps", "2026-04-01"):
            assert common in methodology, (name, common)
        for term in required[name]:
            assert term in methodology, (name, term)


def test_deberta_reader_defines_direct_events_in_plain_language():
    source = _source(build_notebooks()["deberta"]).lower()

    assert "direct_events" in source
    assert "federal reserve" in source
    assert "truth social" in source
    assert "available before" in source
    assert "linguistic directness" in source


def test_deberta_and_llm_readers_verify_all_nine_model_families():
    source = {name: _source(build_notebooks()[name]) for name in ("deberta", "llm")}

    for value in source.values():
        assert 'nunique() == 9' in value
        assert 'MODEL_NAMES' in value


def test_llm_reader_verifies_compact_actual_frozen_prompt_contract():
    source = _source(build_notebooks()["llm"])

    assert 'identity["batch_size"] == 10' in source
    assert '_canonical_hash(prompt_contract) == identity["prompt_hash"]' in source
    assert '_canonical_hash(schema_contract) == identity["schema_hash"]' in source
    assert "System prompt" in source
    assert "Output schema" in source
    assert "validate_llm_score_manifest" in source


def test_llm_reader_defines_matched_and_full_in_plain_language():
    source = _source(build_notebooks()["llm"]).lower()

    assert "llm matched" in source
    assert "same five causal" in source
    assert "llm full" in source
    assert "relevance decay" in source
    assert "high-impact sentiment decay" in source
    assert "24h topic share" in source


def test_llm_manifest_identity_mismatch_fails_closed():
    identity = {
        "tag": "provider-model-version",
        "digest": "frozen",
        "batch_size": 10,
        "batch_protocol": "indexed-json-object-v1",
        "prompt_hash": "prompt",
        "schema_hash": "schema",
    }
    manifest = {
        "complete": True,
        "cutoff_exclusive": "2026-04-01T00:00:00+00:00",
        "identity": identity.copy(),
    }

    validate_llm_score_manifest(manifest, identity)
    manifest["identity"]["digest"] = "mixed"
    with pytest.raises(ValueError, match="frozen LLM identity"):
        validate_llm_score_manifest(manifest, identity)


def test_no_reader_displays_file_or_hash_inventory_tables():
    for notebook in build_notebooks().values():
        source = _source(notebook).lower()
        assert "source_sha256" not in source
        assert "file inventory" not in source
        assert "path inventory" not in source


def test_final_comparison_includes_side_calibrated_policies():
    source = _source(build_notebooks()["comparison"])

    assert "index_side_calibration" in source
    assert "Side-calibrated" in source


def test_ensemble_reader_reports_vix_membership_h1_gate_and_forward_results():
    source = _source(build_notebooks()["ensemble"])

    for marker in (
        "# ensemble-vix-lineage-table",
        "# ensemble-membership-table",
        "# ensemble-h1-table",
        "# ensemble-decision-table",
        "# ensemble-forward-table",
    ):
        assert marker in source
    assert "MODEL_NAMES" in source and "len(MODEL_NAMES) == 9" in source
    assert "h1_selected_candidates.parquet" in source
    assert "h1_selected_control.parquet" in source
    assert "forward_summary.parquet" in source
    assert "Net delta vs control (pp)" in source
    assert "Sortino delta vs control" in source
    assert "Trade retention %" in source
    assert "Q2-2026" in source


def test_ensemble_reader_defines_membership_feature_sets_and_three_rules():
    source = _source(build_notebooks()["ensemble"])

    for model in DISPLAY_MODELS:
        assert model in source
    for phrase in (
        "all nine describes membership, not a fourth ensemble rule",
        "Price + VIX plus matched DeBERTa features",
        "Price + VIX plus matched LLM features",
        "Price + VIX plus full causal LLM features",
        "4 x 3 = 12",
        "Equal probability average (soft vote)",
        "Directional vote (5/9 majority)",
        "Causal logistic meta-model (stack)",
        "18 inputs",
        "C=0.1",
        "2024 OOF only",
        "completed H1 labels",
        "frozen before forward",
    ):
        assert phrase.lower() in source.lower()
    for phrase in (
        "at least 50 total trades",
        "at least 15 LONG and 15 SHORT trades",
        "at least four positive months out of six",
        "Sharpe is reported but is not an H1 eligibility condition",
    ):
        assert phrase.lower() in source.lower()
    assert "All-nine H1 winner" not in source
    assert "Best-ranked H1 ensemble" in source


def test_individual_model_and_comparison_readers_use_unambiguous_ensemble_labels():
    source = {name: _source(build_notebooks()[name]) for name in ("deberta", "llm", "comparison")}

    assert "H1 results for nine individual models" in source["deberta"]
    assert "H1 results for nine individual models" in source["llm"]
    assert "All-nine H1 comparison" not in source["deberta"] + source["llm"]
    assert "Nine-model ensemble" in source["comparison"]
    assert "Best single-model control" in source["comparison"]
    assert "36-model" not in source["comparison"]


def test_ensemble_reader_explains_h1_and_reused_forward_outcomes():
    source = _source(build_notebooks()["ensemble"])

    assert "USA500 has 0/12 eligible ensembles" in source
    assert "USATECH has 1/12 eligible ensemble" in source
    assert "USA500 Price + VIX directional vote" in source
    assert "USATECH LLM full probability average" in source
    assert "descriptive rows do not reopen H1 selection" in source


def test_final_comparison_includes_active_nine_model_ensemble_results():
    source = _source(build_notebooks()["comparison"])

    assert "index_all_model_ensemble" in source
    assert "Nine-model ensemble" in source
    assert "Best single-model control" in source


def test_every_display_is_framed_and_tagged_by_table_type():
    for name, notebook in build_notebooks().items():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code" or "display(" not in cell.source:
                continue
            assert index > 0 and notebook.cells[index - 1].cell_type == "markdown", name
            method = notebook.cells[index - 1].source.strip()
            assert _is_one_sentence(method, "Method:"), (name, method)
            assert cell.source.count("display(") == 1, name
            assert cell.source.count("Takeaway:") == 1, name
            assert cell.source.index("display(") < cell.source.index("Takeaway:"), name
            tags = ("# economic-table" in cell.source, "# non-economic-table" in cell.source)
            assert sum(tags) == 1, name


def test_every_economic_table_contains_both_sharpe_and_sortino():
    for name, notebook in build_notebooks().items():
        for cell in notebook.cells:
            if cell.cell_type != "code" or "# economic-table" not in cell.source:
                continue
            assert "Sharpe" in cell.source, name
            assert "Sortino" in cell.source, name
            assert "display(" in cell.source, name


def test_comparison_filters_zero_trade_rows_instead_of_displaying_flat_controls():
    source = _source(build_notebooks()["comparison"]).lower()
    assert "ineligible_flat" not in source
    assert ".gt(0)" in source or "trades > 0" in source


def test_every_generated_reader_code_cell_compiles():
    for name, notebook in build_notebooks().items():
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == "code":
                compile(cell.source, f"{name}:cell-{index}", "exec")


def test_executed_readers_are_compact_and_error_free():
    for path in EXPECTED.values():
        assert path.exists(), path
        notebook = nbformat.read(path, as_version=4)
        assert not any(
            output.get("output_type") == "error"
            for cell in notebook.cells
            for output in cell.get("outputs", [])
        )
        plotting_cells = [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and ("plt." in cell.source or ".plot(" in cell.source)
        ]
        assert len(plotting_cells) <= 1


@pytest.mark.parametrize(
    ("name", "groups"),
    [("nine_models", 2), ("deberta", 2), ("llm", 4)],
)
def test_executed_forward_result_tables_show_every_model(name: str, groups: int):
    notebook = nbformat.read(EXPECTED[name], as_version=4)
    required_columns = {"Index", "Model", "H1 status", "Forward status", "Trades", "Net %", "Sharpe", "Sortino"}
    tables = [
        table
        for cell in notebook.cells
        if cell.cell_type == "code"
        for table in _cell_tables(cell)
        if required_columns.issubset(table.columns)
    ]
    assert len(tables) == 1
    table = tables[0]
    assert "H1 status" in table.columns
    assert set(table["H1 status"]).issubset({"Pass", "Below gate"})
    group_columns = ["Index"] + (["Features"] if "Features" in table.columns else [])
    grouped = list(table.groupby(group_columns, dropna=False))
    assert len(grouped) == groups
    for _, group in grouped:
        assert set(group["Model"]) == DISPLAY_MODELS
    assert not table["Forward status"].isin(["Not run", "Admitted; 0 trades"]).any()
    assert set(table["Forward status"]).issubset({"Traded", "No trades"})
    for column in ("Trades", "Net %", "Sharpe", "Sortino"):
        values = pd.to_numeric(table[column], errors="raise")
        assert np.isfinite(values).all()


def test_executed_readers_render_no_missing_or_vendor_tokens():
    missing = re.compile(r"(?i)(?<![A-Za-z])nan(?![A-Za-z])|<NA>|\bN/A\b")
    for name, path in EXPECTED.items():
        notebook = nbformat.read(path, as_version=4)
        visible = [
            cell.source for cell in notebook.cells if cell.cell_type == "markdown"
        ]
        for cell in notebook.cells:
            for output in cell.get("outputs", []):
                if output.get("output_type") == "stream":
                    visible.append(str(output.get("text", "")))
                visible.extend(
                    str(value)
                    for mime, value in output.get("data", {}).items()
                    if mime in {"text/html", "text/plain", "text/markdown"}
                )
        rendered = "\n".join(visible)
        assert missing.search(rendered) is None, name
        assert re.search(r"(?i)deepseek", rendered) is None, name
        for pattern in READER_WORKFLOW_PATTERNS:
            assert re.search(pattern, rendered, flags=re.IGNORECASE) is None, (
                name,
                pattern,
            )


def test_every_executed_table_has_brief_caption_and_one_final_notebook_takeaway():
    for name, path in EXPECTED.items():
        notebook = nbformat.read(path, as_version=4)
        visible = [cell.source for cell in notebook.cells if cell.cell_type == "markdown"]
        for index, cell in enumerate(notebook.cells):
            caption = (
                notebook.cells[index - 1].source.strip()
                if index > 0 and notebook.cells[index - 1].cell_type == "markdown"
                else ""
            )
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                if "text/markdown" in data:
                    caption = data["text/markdown"].strip()
                    visible.append(caption)
                if output.get("output_type") == "stream":
                    visible.append(str(output.get("text", "")))
                if "<table" not in data.get("text/html", "").lower():
                    continue
                assert 5 <= len(caption.split()) <= 100, (name, index, caption)
                assert not caption.startswith("#"), (name, index, caption)
                assert re.search(r"[.!?]$", caption), (name, index, caption)
                caption = ""
        assert notebook.cells[-1].cell_type == "markdown", name
        closing = notebook.cells[-1].source.strip()
        assert closing.startswith("## Results\n"), name
        takeaway = closing.split("\n\n")[-1]
        assert _is_one_sentence(takeaway, "Takeaway:"), (name, takeaway)
        assert len(re.findall(r"Takeaway:", "\n".join(visible), flags=re.IGNORECASE)) == 1, name


def test_executed_vix_family_effects_are_numeric_and_ranked_best_first():
    notebook = nbformat.read(EXPECTED["vix"], as_version=4)
    required_columns = {"Index", "Model", "Price-only gate net %", "Price + VIX gate net %", "Gate net delta (pp)"}
    tables = [
        table
        for cell in notebook.cells
        if cell.cell_type == "code"
        for table in _cell_tables(cell)
        if required_columns.issubset(table.columns)
    ]
    assert len(tables) == 1
    table = tables[0]
    values = pd.to_numeric(table["Gate net delta (pp)"], errors="raise")
    price = pd.to_numeric(table["Price-only gate net %"], errors="raise")
    vix = pd.to_numeric(table["Price + VIX gate net %"], errors="raise")
    assert len(table) == 18
    assert values.is_monotonic_decreasing
    assert (vix.sub(price).sub(values).abs() <= 0.002).all()
    assert not table.astype(str).isin(["True", "False"]).any().any()


def test_executed_ensemble_tables_are_complete_finite_and_net_ranked():
    notebook = nbformat.read(EXPECTED["ensemble"], as_version=4)
    marked = {
        marker: [
            cell
            for cell in notebook.cells
            if cell.cell_type == "code" and marker in cell.source
        ]
        for marker in (
            "# ensemble-vix-lineage-table",
            "# ensemble-membership-table",
            "# ensemble-definition-table",
            "# ensemble-h1-table",
            "# ensemble-decision-table",
            "# ensemble-forward-table",
        )
    }
    assert all(len(cells) == 1 for cells in marked.values())
    vix = _cell_tables(marked["# ensemble-vix-lineage-table"][0])[0]
    assert len(vix) == 2
    for column in (
        "Positive models / 9",
        "Positive folds / 5",
        "Median gate delta (pp)",
        "Trade retention %",
    ):
        assert np.isfinite(pd.to_numeric(vix[column], errors="raise")).all()
    members = _cell_tables(marked["# ensemble-membership-table"][0])[0]
    assert len(members) == 9 and set(members["Model"]) == DISPLAY_MODELS
    definitions = _cell_tables(marked["# ensemble-definition-table"][0])[0]
    assert len(definitions) == 3
    assert set(definitions["Ensemble rule"]) == {
        "Equal probability average (soft vote)",
        "Directional vote (5/9 majority)",
        "Causal logistic meta-model (stack)",
    }
    assert pd.to_numeric(definitions["Base models"], errors="raise").eq(9).all()
    h1 = _cell_tables(marked["# ensemble-h1-table"][0])[0]
    assert len(h1) == 24
    assert h1.groupby("Index")["Features"].nunique().eq(4).all()
    assert h1.groupby(["Index", "Features"])["Ensemble"].nunique().eq(3).all()
    decisions = _cell_tables(marked["# ensemble-decision-table"][0])[0]
    assert len(decisions) == 4
    assert decisions.groupby("Index")["Role"].nunique().eq(2).all()
    forward = _cell_tables(marked["# ensemble-forward-table"][0])[0]
    assert len(forward) == 26
    for table in (h1, decisions, forward):
        for column in ("Trades", "Net %", "Sharpe", "Sortino"):
            assert np.isfinite(pd.to_numeric(table[column], errors="raise")).all()
        assert pd.to_numeric(table["Net %"], errors="raise").is_monotonic_decreasing


def test_readers_never_sum_correlated_models_as_a_portfolio():
    for notebook in build_notebooks().values():
        source = _source(notebook).lower()
        assert 'net_return=("net_return", "sum")' not in source
        assert "summed net return" not in source


def test_executed_display_tables_are_nonempty_not_all_zero_and_pair_risk_metrics():
    for name, path in EXPECTED.items():
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "code" or "-table" not in cell.source:
                continue
            html_outputs = [
                output.get("data", {}).get("text/html", "")
                for output in cell.get("outputs", [])
                if output.get("output_type") in {"display_data", "execute_result"}
            ]
            tables = [
                _html_frame(html)
                for html in html_outputs
                if "<table" in html.lower()
            ]
            for table in tables:
                assert not table.empty, name
                numeric = table.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
                if not numeric.empty:
                    assert not numeric.fillna(0).eq(0).all().all(), name
                if "# economic-table" in cell.source:
                    assert "Sharpe" in table.columns, name
                    assert "Sortino" in table.columns, name


def test_executed_economic_tables_rank_net_return_best_first():
    for name, path in EXPECTED.items():
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type != "code" or "# economic-table" not in cell.source:
                continue
            for table in _cell_tables(cell):
                if "Net %" not in table.columns:
                    continue
                values = pd.to_numeric(table["Net %"], errors="coerce").dropna()
                assert values.is_monotonic_decreasing, name
