"""Saved readers expose complete, explained evidence without development logs."""
from pathlib import Path
import re

import nbformat
import pytest

from experiments.notebook_hygiene import execution_order
from notebook_assertions import tables


@pytest.mark.parametrize("name", execution_order())
def test_every_rendered_table_has_context_and_no_missing_numeric_placeholder(name):
    notebook = nbformat.read(Path(__file__).parents[1] / "notebooks" / name, 4)
    events = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            events.append(("text", cell.source))
        else:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                html = data.get("text/html", "")
                if "<table" in html:
                    events.append(("table", html))
                    assert not re.search(r">\s*(?:NaN|nan)\s*<", html)
                elif data.get("text/markdown"):
                    events.append(("text", data["text/markdown"]))
                elif output.get("text", "").strip():
                    events.append(("text", output["text"]))
    assert tables(notebook)
    for index, (kind, _) in enumerate(events):
        if kind == "table":
            assert index > 0 and events[index - 1][0] == "text"
            assert index + 1 < len(events) and events[index + 1][0] == "text"
