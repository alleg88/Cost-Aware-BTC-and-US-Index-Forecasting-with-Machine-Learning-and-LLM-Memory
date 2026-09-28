from importlib import import_module
from pathlib import Path
import sys

from nbclient.exceptions import CellExecutionError


def test_master_reproduction_executes_selected_readers_then_audits(monkeypatch):
    module = import_module("experiments.reproduce_notebooks")
    calls = []

    def fake_execute_all(sequence_names, *, timeout, start_at):
        calls.append(("execute", sequence_names, timeout, start_at))
        return [Path("first.ipynb"), Path("second.ipynb")]

    def fake_audit_repository(repo_root):
        calls.append(("audit", repo_root))
        return {"status": "READY", "canonical_notebooks": 27}

    monkeypatch.setattr(module, "execute_all", fake_execute_all)
    monkeypatch.setattr(module, "audit_repository", fake_audit_repository)

    result = module.reproduce_notebooks(
        sequence_names=("Bitcoin",),
        timeout=123,
        start_at="01_RQ1_A_BTC_data_labels_baseline.ipynb",
    )

    assert calls == [
        (
            "execute",
            ("Bitcoin",),
            123,
            "01_RQ1_A_BTC_data_labels_baseline.ipynb",
        ),
        ("audit", module.REPOSITORY_ROOT),
    ]
    assert result == {
        "executed_notebooks": ("first.ipynb", "second.ipynb"),
        "release_audit": {"status": "READY", "canonical_notebooks": 27},
    }


def test_master_reproduction_fails_closed_when_release_audit_is_not_ready(
    monkeypatch,
):
    module = import_module("experiments.reproduce_notebooks")
    monkeypatch.setattr(module, "execute_all", lambda *args, **kwargs: [])
    monkeypatch.setattr(
        module,
        "audit_repository",
        lambda repo_root: {"status": "NOT_READY", "problems": ["broken"]},
    )

    try:
        module.reproduce_notebooks()
    except module.NotebookReproductionError as error:
        assert "broken" in str(error)
    else:
        raise AssertionError("a failed post-execution audit must stop reproduction")


def test_cli_reports_a_notebook_cell_failure_without_an_uncaught_traceback(
    monkeypatch,
    capsys,
):
    module = import_module("experiments.reproduce_notebooks")
    monkeypatch.setattr(sys, "argv", ["reproduce_notebooks"])
    monkeypatch.setattr(
        module,
        "reproduce_notebooks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            CellExecutionError("trace", "ValueError", "missing input")
        ),
    )

    assert module.main() == 1
    output = capsys.readouterr().out
    assert '"status": "NOT_READY"' in output
    assert "missing input" in output
