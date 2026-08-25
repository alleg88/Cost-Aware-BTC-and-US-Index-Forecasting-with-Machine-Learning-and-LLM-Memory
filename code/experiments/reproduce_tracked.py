"""Run the minimal tracked-code reproducibility checks from one entrypoint."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from experiments.colab_runtime import is_colab


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent


def _run(arguments: list[str]) -> None:
    print("+", subprocess.list2cmdline(arguments), flush=True)
    subprocess.run(arguments, cwd=CODE_ROOT, check=True)


def _dependency_check_command() -> list[str]:
    if is_colab():
        return [sys.executable, "-m", "experiments.colab_runtime"]
    if importlib.util.find_spec("pip") is not None:
        return [sys.executable, "-m", "pip", "check"]
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("dependency check requires pip or uv")
    return [uv, "pip", "check", "--python", sys.executable]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="run dependency and repository audits without the tracked pytest suite",
    )
    args = parser.parse_args()

    _run(_dependency_check_command())
    _run(
        [
            sys.executable,
            "-m",
            "experiments.repository_release",
            "--repo-root",
            str(REPOSITORY_ROOT),
        ]
    )
    if not args.audit_only:
        with tempfile.TemporaryDirectory(
            prefix=".reproduce_tmp_", dir=CODE_ROOT
        ) as temp_dir:
            _run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    "--clean-clone",
                    f"--basetemp={temp_dir}",
                ]
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
