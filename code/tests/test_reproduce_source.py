from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path(__file__).parents[1]


def test_master_audit_reports_every_notebook_sequence_ready(tmp_path):
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
    assert payload["task_count"] == 59
    assert payload["registered_outputs"] == 184
    assert payload["notebook_sequences"] == {
        "Bitcoin": "READY",
        "Channels": "READY",
        "Final confirmation": "READY",
        "Indices": "READY",
    }
