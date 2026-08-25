from __future__ import annotations

from pathlib import Path
import subprocess
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]


def test_audit_only_entrypoint_runs_the_real_tracked_release_contract() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "experiments.reproduce_tracked", "--audit-only"],
        cwd=CODE_ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert '"status": "READY"' in completed.stdout
    diagnostics = completed.stdout + completed.stderr
    assert any(
        marker in diagnostics
        for marker in (
            "No broken requirements found",
            "All installed packages are compatible",
            "Native Colab package imports passed.",
        )
    )
