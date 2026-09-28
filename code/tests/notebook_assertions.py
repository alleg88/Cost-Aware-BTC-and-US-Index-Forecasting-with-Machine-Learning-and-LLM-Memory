"""Assertions on rendered notebook evidence, independent of prose wording."""
import ast
from html.parser import HTMLParser
import pytest


class _Tables(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables = []
        self.table = None
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            self.table = []
        elif tag == "tr" and self.table is not None:
            self.row = []
        elif tag in {"th", "td"} and self.row is not None:
            self.cell = []

    def handle_data(self, data):
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):
        if tag in {"th", "td"} and self.cell is not None:
            self.row.append("".join(self.cell).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.table.append(self.row)
            self.row = None
        elif tag == "table" and self.table is not None:
            self.tables.append(self.table)
            self.table = None


def tables(notebook):
    parser = _Tables()
    for cell in notebook.cells:
        for output in cell.get("outputs", []):
            parser.feed(output.get("data", {}).get("text/html", ""))
    return parser.tables


def numeric_column(table, name):
    index = table[0].index(name)
    return [float(row[index].replace(",", "").replace("%", "")) for row in table[1:]]


def assert_artifact_reader(notebook):
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    # The environment-specific launcher has no saved run; result cells do.
    assert "from run_zip import prepare_" in code[0].source
    assert len(code) > 1 and all(cell.execution_count is not None for cell in code[1:])
    assert not any(o.output_type == "error" for cell in code for o in cell.get("outputs", []))
    tree = ast.parse("\n".join(cell.source for cell in code))
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in {"fit", "predict", "predict_proba"}
                   for node in ast.walk(tree))


def assert_descending(table, name):
    values = numeric_column(table, name)
    assert values == sorted(values, reverse=True)


def assert_later_period_guards(notebook):
    tree = ast.parse("\n".join(c.source for c in notebook.cells if c.cell_type == "code"))
    for key in ("h1_loaded", "forward_loaded", "lockbox_2026_q2_used"):
        checks = [node for node in ast.walk(tree) if isinstance(node, ast.Assert)
                  and isinstance(node.test, ast.Compare)
                  and isinstance(node.test.left, ast.Subscript)
                  and isinstance(node.test.left.value, ast.Name)
                  and node.test.left.value.id == "summary"
                  and isinstance(node.test.left.slice, ast.Constant)
                  and node.test.left.slice.value == key]
        assert len(checks) == 1
        code = compile(ast.Module(body=checks, type_ignores=[]), "period guard", "exec")
        exec(code, {"summary": {key: False}})
        with pytest.raises(AssertionError):
            exec(code, {"summary": {key: True}})
