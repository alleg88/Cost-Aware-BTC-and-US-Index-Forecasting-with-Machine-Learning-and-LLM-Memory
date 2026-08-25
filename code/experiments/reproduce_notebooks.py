"""Execute the canonical dissertation notebooks and verify their saved outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nbclient.exceptions import CellExecutionError

from experiments.notebook_hygiene import NOTEBOOK_SEQUENCES, execute_all
from experiments.repository_release import audit_repository


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent


class NotebookReproductionError(RuntimeError):
    """Raised when execution does not leave a release-ready notebook set."""


def reproduce_notebooks(
    sequence_names: tuple[str, ...] | None = None,
    *,
    timeout: int = 900,
    start_at: str | None = None,
) -> dict[str, object]:
    """Execute registered readers in order, then fail closed on the release audit."""
    executed = execute_all(
        sequence_names,
        timeout=timeout,
        start_at=start_at,
    )
    report = audit_repository(REPOSITORY_ROOT)
    if report.get("status") != "READY":
        problems = report.get("problems") or ["unknown release-audit failure"]
        raise NotebookReproductionError("; ".join(str(item) for item in problems))
    return {
        "executed_notebooks": tuple(path.name for path in executed),
        "release_audit": report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence",
        action="append",
        choices=tuple(NOTEBOOK_SEQUENCES),
        help="run only this sequence; may be repeated",
    )
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument(
        "--start-at",
        help="resume at this notebook within the selected sequence",
    )
    args = parser.parse_args()
    selected = tuple(args.sequence) if args.sequence else None
    try:
        result = reproduce_notebooks(
            selected,
            timeout=args.timeout,
            start_at=args.start_at,
        )
    except (OSError, ValueError, CellExecutionError, NotebookReproductionError) as error:
        detail = (
            f"{error.ename}: {error.evalue}"
            if isinstance(error, CellExecutionError)
            else str(error)
        )
        print(json.dumps({"status": "NOT_READY", "error": detail}, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": "READY",
                "executed_notebooks": list(result["executed_notebooks"]),
                "release_audit": result["release_audit"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
