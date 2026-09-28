from pathlib import Path

import nbformat

from notebook_assertions import assert_artifact_reader, numeric_column, tables


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "03_RQ1_C_BTC_CatBoost_economic_objectives.ipynb"


def test_matched_comparison_retains_all_nine_rows_at_each_stage():
    notebook = nbformat.read(NOTEBOOK, 4)
    assert_artifact_reader(notebook)
    rendered = tables(notebook)
    assert [len(table) - 1 for table in rendered] == [9, 9, 9, 9]
    for table in rendered[:3]:
        assert set(numeric_column(table, "DZ (bps)")) == {55, 65, 75}
        assert {"Sortino", "Sharpe", "Net return", "Trades"}.issubset(table[0])
        column = table[0].index("Objective")
        assert len({row[column] for row in table[1:]}) == 3


def test_h1_and_forward_keep_identical_selected_candidate_and_policy():
    _, h1, forward, _ = tables(nbformat.read(NOTEBOOK, 4))
    def keyed(table):
        columns = ["Objective", "DZ (bps)", "Candidate", "Policy", "Confidence", "TP (bps)", "SL (bps)"]
        indices = [table[0].index(name) for name in columns]
        return sorted(tuple(row[i] for i in indices) for row in table[1:])
    assert keyed(h1) == keyed(forward)
    for table in (h1, forward):
        assert numeric_column(table, "Trades") == [
            long + short for long, short in zip(numeric_column(table, "Long trades"), numeric_column(table, "Short trades"))
        ]


def test_forward_conclusion_matches_the_negative_economic_evidence():
    _, _, forward, differences = tables(nbformat.read(NOTEBOOK, 4))
    net = numeric_column(forward, "Net return")
    assert all(value < 0 for value in net)
    leader = forward[1 + net.index(max(net))]
    assert leader[forward[0].index("Objective")] == "F1-tuned"
    assert float(leader[forward[0].index("Trades")]) == 67
    sortino = numeric_column(forward, "Sortino")
    risk_adjusted_leader = forward[1 + sortino.index(max(sortino))]
    assert risk_adjusted_leader[forward[0].index("Objective")] == "Economic-tuned"
    assert float(risk_adjusted_leader[forward[0].index("Trades")]) == 86
    assert {"Net-return difference", "Sortino difference", "Sharpe difference", "Trade-count difference"}.issubset(differences[0])


def test_notebook_02b_adds_code_root_before_importing_experiments():
    source = "\n".join(cell.source for cell in nbformat.read(NOTEBOOK, 4).cells)
    assert source.index("sys.path.insert(0, str(CODE_ROOT))") < source.index("from experiments.notebook02_handoff")


def test_notebook_02b_uses_no_more_than_two_plots():
    notebook = nbformat.read(NOTEBOOK, 4)
    figures = [output for cell in notebook.cells for output in cell.get("outputs", [])
               if "image/png" in output.get("data", {})]
    assert 1 <= len(figures) <= 2
