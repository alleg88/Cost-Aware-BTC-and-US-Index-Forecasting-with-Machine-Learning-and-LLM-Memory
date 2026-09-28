from pathlib import Path

import nbformat
from notebook_assertions import assert_descending, tables


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "14_RQ3_C_BTC_sentiment_policy_ablation.ipynb"
)


def _load():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells)
    return notebook, source


def test_notebook_02e_documents_the_frozen_protocol():
    _, source = _load()
    for phrase in (
        "predictions from Notebook 13",
        "nine model families, four sentiment arms",
        "No sentiment is the market-only control",
        "DeBERTa and LLM-matched each add six sentiment columns",
        "LLM-full adds nine",
        "180-day training history",
        "one-M15-bar hold",
        "5 bps per side",
        "For each H1-2025 month, models fit only preceding data and predict that month",
        "six causal monthly blocks",
        "33 fixed policies",
        "July 2025-March 2026",
        "one-minute candles resolve barrier execution",
    ):
        assert phrase.lower() in source.lower()


def test_notebook_02e_has_the_requested_result_tables():
    notebook, source = _load()
    assert [len(table) - 1 for table in tables(notebook)] == [108, 108, 5, 5, 4]
    assert "expected = len(ARMS) * 9 * len(WIDTHS)" in source
    assert "assert len(policies) == expected" in source
    assert "assert len(economics) == expected" in source


def test_notebook_02e_reports_full_economics_and_five_distinct_leaders():
    _, source = _load()
    for field in (
        "Model", "Sentiment", "DZ", "Threshold", "TP", "SL",
        "Hold (min)", "Trades", "Net return", "Sortino", "Sharpe",
        "Positive months",
    ):
        assert field in source
    assert ".drop_duplicates('model_name').head(5)" in source


def test_notebook_02e_has_policy_free_and_calibrated_leader_tables():
    notebook, source = _load()
    raw_cell = next(
        cell for cell in notebook.cells
        if cell.cell_type == "code"
        and "raw_leaders =" in cell.source
    )
    calibrated_cell = next(
        cell for cell in notebook.cells
        if cell.cell_type == "code"
        and "leaders = economics" in cell.source
    )
    assert "raw_economics" in raw_cell.source
    for policy_field in ("Threshold", "TP", "SL"):
        assert policy_field not in raw_cell.source
        assert policy_field in calibrated_cell.source
    assert "sortino_improvements" in calibrated_cell.source
    assert "net_improvements" in calibrated_cell.source
    assert "assert len(improvement) == 9" in calibrated_cell.source


def test_notebook_02e_displays_every_row_and_orders_tables_by_sortino():
    notebook, source = _load()
    assert "pd.set_option('display.max_rows', 200)" in source
    for caption in (
        "h1 = policies",
        "forward = economics",
    ):
        cell = next(
            cell for cell in notebook.cells
            if cell.cell_type == "code" and caption in cell.source
        )
        html = "".join(
            output.get("data", {}).get("text/html", "")
            for output in cell.get("outputs", [])
            if output.output_type in {"display_data", "execute_result"}
        )
        assert html.count("<tr") >= 109
        assert "Sortino" in cell.source
        assert "ascending=False" in cell.source

    leader_cell = next(
        cell for cell in notebook.cells
        if cell.cell_type == "code" and "leaders = economics" in cell.source
    )
    leader_html = "".join(
        output.get("data", {}).get("text/html", "")
        for output in leader_cell.get("outputs", [])
        if output.output_type in {"display_data", "execute_result"}
    )
    # the cell renders the five family leaders and then one row per arm
    assert leader_html.count("<tr") == 6 + (1 + 4)
    for table in tables(notebook)[:4]:
        assert_descending(table, "Sortino")


def test_notebook_02e_does_not_repeat_notebook_02d_results():
    _, source = _load()
    for phrase in (
        "Overall macro-F1",
        "Robust F1",
        "2024 fixed 15-minute control",
        "Raw frozen-forward comparison",
        "classification diagnostics",
    ):
        assert phrase.lower() not in source.lower()


def test_notebook_02e_is_executed_without_errors():
    notebook, _ = _load()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells
    assert "from run_zip import prepare_" in code_cells[0].source
    assert len(code_cells) > 1 and all(cell.execution_count is not None for cell in code_cells[1:])
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
