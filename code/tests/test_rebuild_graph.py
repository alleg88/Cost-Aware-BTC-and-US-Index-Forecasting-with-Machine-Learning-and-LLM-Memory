from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from experiments.rebuild_graph import (
    GraphError,
    RebuildContext,
    RebuildGraph,
    TaskExecutionError,
    execute_graph,
)


def _module(root: Path, name: str, body: str) -> None:
    package = root / "fixture"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / f"{name}.py").write_text(body, encoding="utf-8")


def _context(tmp_path: Path) -> RebuildContext:
    work = tmp_path / "work"
    work.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tmp_path)
    return RebuildContext(
        code_root=work,
        repository_root=tmp_path,
        state_root=work / ".rebuild_state",
        profile="canonical",
        env=env,
    )


def _graph() -> RebuildGraph:
    return RebuildGraph.from_dict(
        {
            "schema_version": 1,
            "sources": ["raw.txt"],
            "tasks": [
                {
                    "id": "source",
                    "depends_on": [],
                    "module": "fixture.source",
                    "args": [],
                    "inputs": ["raw.txt"],
                    "outputs": ["built.txt"],
                    "profile": "canonical",
                    "comparison": "exact",
                },
                {
                    "id": "reader",
                    "depends_on": ["source"],
                    "module": "fixture.reader",
                    "args": [],
                    "inputs": ["built.txt"],
                    "outputs": ["result.json"],
                    "profile": "canonical",
                    "comparison": "exact",
                },
            ],
        }
    )


def test_graph_orders_dependencies_and_resumes_only_hash_identical_tasks(tmp_path):
    _module(
        tmp_path,
        "source",
        "from pathlib import Path\nPath('built.txt').write_text(Path('raw.txt').read_text().upper())\n",
    )
    _module(
        tmp_path,
        "reader",
        "from pathlib import Path\nimport json\nPath('result.json').write_text(json.dumps({'value': Path('built.txt').read_text()}))\n",
    )
    context = _context(tmp_path)
    (context.code_root / "raw.txt").write_text("one", encoding="utf-8")
    graph = _graph()

    assert graph.topological_order() == ("source", "reader")
    first = execute_graph(graph, context)
    second = execute_graph(graph, context)

    assert first.executed == ("source", "reader")
    assert second.skipped == ("source", "reader")
    (context.code_root / "raw.txt").write_text("two", encoding="utf-8")
    third = execute_graph(graph, context)
    assert third.executed == ("source", "reader")


def test_output_tamper_reruns_only_affected_task(tmp_path):
    _module(tmp_path, "source", "from pathlib import Path\nPath('built.txt').write_text('built')\n")
    _module(tmp_path, "reader", "from pathlib import Path\nPath('result.json').write_text('{}')\n")
    context = _context(tmp_path)
    (context.code_root / "raw.txt").write_text("raw", encoding="utf-8")
    graph = _graph()
    execute_graph(graph, context)
    (context.code_root / "result.json").write_text("tampered", encoding="utf-8")

    report = execute_graph(graph, context)

    assert report.skipped == ("source",)
    assert report.executed == ("reader",)


def test_graph_rejects_cycles_and_unregistered_inputs():
    cyclic = {
        "schema_version": 1,
        "sources": [],
        "tasks": [
            {"id": "a", "depends_on": ["b"], "module": "a", "args": [], "inputs": [], "outputs": ["a"], "profile": "canonical", "comparison": "exact"},
            {"id": "b", "depends_on": ["a"], "module": "b", "args": [], "inputs": [], "outputs": ["b"], "profile": "canonical", "comparison": "exact"},
        ],
    }
    missing = {
        "schema_version": 1,
        "sources": [],
        "tasks": [
            {"id": "a", "depends_on": [], "module": "a", "args": [], "inputs": ["orphan.txt"], "outputs": ["a.txt"], "profile": "canonical", "comparison": "exact"}
        ],
    }

    with pytest.raises(GraphError, match="cycle"):
        RebuildGraph.from_dict(cyclic).topological_order()
    with pytest.raises(GraphError, match="producer"):
        RebuildGraph.from_dict(missing).validate_artifacts()


def test_graph_requires_artifact_producer_in_dependency_chain():
    payload = {
        "schema_version": 1,
        "sources": [],
        "tasks": [
            {"id": "producer", "depends_on": [], "module": "p", "args": [], "inputs": [], "outputs": ["built.txt"], "profile": "canonical", "comparison": "exact"},
            {"id": "consumer", "depends_on": [], "module": "c", "args": [], "inputs": ["built.txt"], "outputs": ["result.txt"], "profile": "canonical", "comparison": "exact"},
        ],
    }

    with pytest.raises(GraphError, match="dependency"):
        RebuildGraph.from_dict(payload).validate_artifacts()


def test_failed_task_records_failure_and_can_recover(tmp_path):
    _module(tmp_path, "flaky", "raise SystemExit(7)\n")
    context = _context(tmp_path)
    graph = RebuildGraph.from_dict(
        {
            "schema_version": 1,
            "sources": [],
            "tasks": [
                {"id": "flaky", "depends_on": [], "module": "fixture.flaky", "args": [], "inputs": [], "outputs": ["done.txt"], "profile": "canonical", "comparison": "exact"}
            ],
        }
    )

    with pytest.raises(TaskExecutionError, match="exit code 7"):
        execute_graph(graph, context)
    state = json.loads((context.state_root / "flaky.json").read_text("utf-8"))
    assert state["status"] == "failed"

    _module(tmp_path, "flaky", "from pathlib import Path\nPath('done.txt').write_text('ok')\n")
    report = execute_graph(graph, context)
    assert report.executed == ("flaky",)


def test_missing_input_fails_before_subprocess(tmp_path, monkeypatch):
    context = _context(tmp_path)
    graph = _graph()
    monkeypatch.setattr(
        "experiments.rebuild_graph.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("subprocess launched")),
    )

    with pytest.raises(TaskExecutionError, match="missing input"):
        execute_graph(graph, context)


def test_task_environment_is_part_of_identity_and_reaches_subprocess(tmp_path):
    _module(
        tmp_path,
        "environment",
        "from pathlib import Path\nimport os\n"
        "Path('environment.txt').write_text(os.environ['MSC_CANONICAL_OFFLINE'])\n",
    )
    context = _context(tmp_path)
    graph = RebuildGraph.from_dict(
        {
            "schema_version": 1,
            "sources": [],
            "tasks": [
                {
                    "id": "environment",
                    "depends_on": [],
                    "module": "fixture.environment",
                    "args": [],
                    "inputs": [],
                    "outputs": ["environment.txt"],
                    "profile": "canonical",
                    "comparison": "exact",
                    "environment": {"MSC_CANONICAL_OFFLINE": "1"},
                }
            ],
        }
    )

    report = execute_graph(graph, context)

    assert report.executed == ("environment",)
    assert graph.tasks[0].to_dict()["environment"] == {"MSC_CANONICAL_OFFLINE": "1"}
    assert (context.code_root / "environment.txt").read_text(encoding="utf-8") == "1"


def test_visible_run_reports_child_output_and_reuses_only_upstream(tmp_path, capsys):
    _module(tmp_path, "source", "from pathlib import Path\nPath('built.txt').write_text('built')\n")
    _module(tmp_path, "reader", "from pathlib import Path\nprint('fitting selected model', flush=True)\nPath('result.json').write_text('{}')\n")
    context = _context(tmp_path)
    (context.code_root / "raw.txt").write_text("raw")
    execute_graph(_graph(), context)

    report = execute_graph(_graph(), context, selected_tasks=["reader"],
                           force_tasks=["reader"], stream=True)

    assert report.executed == ("reader",)
    assert report.skipped == ("source",)
    output = capsys.readouterr().out
    assert "REUSE source" in output and "RUN reader" in output
    assert "fitting selected model" in output and "DONE reader" in output
    assert "fitting selected model" in (context.state_root / "reader.log").read_text()


def test_visible_failure_keeps_diagnostic_in_output_and_log(tmp_path, capsys):
    _module(tmp_path, "source", "print('input has invalid timestamps', flush=True)\nraise SystemExit(7)\n")
    context = _context(tmp_path)
    (context.code_root / "raw.txt").write_text("raw")
    with pytest.raises(TaskExecutionError, match="exit code 7"):
        execute_graph(_graph(), context, selected_tasks=["source"], stream=True)
    assert "input has invalid timestamps" in capsys.readouterr().out
    assert "input has invalid timestamps" in (context.state_root / "source.log").read_text()


def test_selected_branch_does_not_execute_an_unrelated_reader(tmp_path):
    _module(tmp_path, "source", "from pathlib import Path\nPath('built.txt').write_text('built')\n")
    _module(tmp_path, "reader", "raise AssertionError('unrelated notebook ran')\n")
    context = _context(tmp_path)
    (context.code_root / "raw.txt").write_text("raw")
    report = execute_graph(_graph(), context, selected_tasks=["source"])
    assert report.executed == ("source",)
    assert not (context.code_root / "result.json").exists()


def test_changed_kernel_packages_invalidate_upstream_receipts(tmp_path):
    from dataclasses import replace

    _module(tmp_path, "source", "from pathlib import Path\nPath('built.txt').write_text('built')\n")
    context = _context(tmp_path)
    (context.code_root / "raw.txt").write_text("raw")
    first = replace(context, runtime_identity="python-and-packages-A")
    second = replace(context, runtime_identity="python-and-packages-B")
    execute_graph(_graph(), first, selected_tasks=["source"])
    assert execute_graph(_graph(), first, selected_tasks=["source"]).skipped == ("source",)
    assert execute_graph(_graph(), second, selected_tasks=["source"]).executed == ("source",)
