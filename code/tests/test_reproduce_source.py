from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path(__file__).parents[1]


def test_master_audit_reports_every_notebook_sequence_ready(tmp_path):
    graph_payload = json.loads(
        (CODE_ROOT / "configs" / "rebuild_tasks.json").read_text(encoding="utf-8")
    )
    report = tmp_path / "audit.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.reproduce_source",
            "--audit-only",
            "--report",
            str(report),
        ],
        cwd=CODE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "READY"
    if graph_payload.get("result_comparison") == "not_required":
        # The compact Rebuild intentionally replaces one handoff task with
        # six recomputed channel stages and therefore has a different graph
        # shape from the canonical source checkout.
        assert payload["task_count"] == 65
        assert payload["registered_outputs"] == 190
    else:
        assert payload["task_count"] == 60
        assert payload["registered_outputs"] == 185
    assert payload["notebook_sequences"] == {
        "Bitcoin": "READY",
        "Channels": "READY",
        "Final confirmation": "READY",
        "Indices": "READY",
    }


def test_compact_source_run_does_not_compare_predictions_to_archived_outputs(monkeypatch, tmp_path):
    from experiments import reproduce_source as runner
    from experiments.rebuild_graph import RebuildReport
    graph = json.loads((CODE_ROOT / "configs/rebuild_tasks.json").read_text())
    graph["result_comparison"] = "not_required"
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(graph))
    report = tmp_path / "report.json"
    monkeypatch.setattr(runner, "execute_graph", lambda *args, **kwargs: RebuildReport(executed=("channel.j",), skipped=()))
    def no_reference(*args, **kwargs):
        raise AssertionError("compact Rebuild compared historical predictions")
    monkeypatch.setattr(runner, "compare_task_to_git", no_reference)
    assert runner.main(["--graph", str(path), "--report", str(report)]) == 0
    actual = json.loads(report.read_text())
    assert actual["reference"] is None and actual["comparisons"] == []
    assert actual["numerical_equivalence_checked"] is False
