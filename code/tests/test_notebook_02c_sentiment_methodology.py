from pathlib import Path

import nbformat

from notebook_assertions import assert_artifact_reader, numeric_column, tables


NOTEBOOK = Path(__file__).parents[1] / "notebooks" / "12_RQ3_A_BTC_sentiment_data_methodology.ipynb"


def test_sentiment_reader_reports_data_without_fitting_forecasts():
    notebook = nbformat.read(NOTEBOOK, 4)
    assert_artifact_reader(notebook)
    assert len(tables(notebook)) == 9
    figures = [o for c in notebook.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
    assert len(figures) == 2


def test_all_shared_sentiment_features_are_present_in_rendered_diagnostics():
    notebook = nbformat.read(NOTEBOOK, 4)
    rendered = "\n".join(str(table) for table in tables(notebook))
    for name in ("sent_news_decay", "sent_news_count_24h", "sent_tone_decay",
                 "sent_macro_decay", "sent_fng_change_7d"):
        assert name in rendered
    assert "direct_pulse" in {row[0] for row in tables(notebook)[-1][1:]}
    for arm in ("DeBERTa", "LLM-matched", "LLM-full"):
        assert arm in rendered


def test_deployed_feature_budgets_keep_matched_and_full_arms_distinct():
    table = next(t for t in tables(nbformat.read(NOTEBOOK, 4)) if "sentiment columns" in t[0])
    index = table[0].index("sentiment columns")
    budgets = {row[0]: int(row[index]) for row in table[1:]}
    assert budgets == {"DeBERTa": 6, "LLM-matched": 6, "LLM-full": 9}
    assert numeric_column(table, "direct-event channels") == [1, 1, 1]
