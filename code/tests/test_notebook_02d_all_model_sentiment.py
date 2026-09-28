from pathlib import Path

import nbformat
import pandas as pd

from notebook_assertions import assert_descending, tables


CODE_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = CODE_ROOT / "notebooks" / "13_RQ3_B_BTC_nine_model_sentiment_ablation.ipynb"


def _source():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    return notebook, "\n".join(cell.source for cell in notebook.cells)


def test_raw_reader_displays_all_model_arm_width_pairs_without_truncation():
    classification, economic, *_ = tables(_source()[0])
    for table in (classification, economic):
        assert len(table) - 1 == 108
        columns = [table[0].index(name) for name in ("Model", "Arm", "DZ")]
        keys = [tuple(row[index] for index in columns) for row in table[1:]]
        assert len(set(keys)) == 108
        assert len({key[0] for key in keys}) == 9
        assert len({key[1] for key in keys}) == 4
        assert {float(key[2]) for key in keys} == {55, 65, 75}
    assert {"Overall macro-F1", "Robust F1"}.issubset(classification[0])
    assert {"Trades", "Net return", "Sortino", "Sharpe"}.issubset(economic[0])
    assert {"Threshold", "TP", "SL"}.isdisjoint(economic[0])


def test_raw_reader_ranks_results_and_retains_five_distinct_families():
    rendered = tables(_source()[0])
    for table in (rendered[0], rendered[1], rendered[-1]):
        assert_descending(table, "Sortino")
    leaders = rendered[-1]
    assert len(leaders) - 1 == 5
    model_index = leaders[0].index("Model")
    assert len({row[model_index] for row in leaders[1:]}) == 5


def test_raw_sentiment_artifacts_use_all_models_and_fixed_candidate_zero():
    from experiments.all_model_sentiment_raw import POLICY_COLUMNS
    from experiments.all_model_sentiment_scoreboard import build_scoreboards

    tables = build_scoreboards()
    assert set(tables) == {"classification", "economics"}
    assert len(tables["classification"]) == len(tables["economics"]) == 108
    economics = tables["economics"]
    assert economics["candidate_id"].eq(0).all()
    assert economics["lookback_days"].eq(180).all()
    assert economics["hold_minutes"].eq(15).all()
    assert economics["trades"].eq(economics["n_long"] + economics["n_short"]).all()
    assert not POLICY_COLUMNS.intersection(economics.columns)
    assert pd.to_datetime(economics["period_end"], utc=True).le(
        pd.Timestamp("2026-04-01", tz="UTC")
    ).all()


def test_notebook_02d_is_executed_without_errors():
    notebook, _ = _source()
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert "from run_zip import prepare_" in code_cells[0].source
    assert len(code_cells) > 1 and all(cell.execution_count is not None for cell in code_cells[1:])
    assert not any(
        output.output_type == "error"
        for cell in code_cells
        for output in cell.get("outputs", [])
    )
