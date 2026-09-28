"""Audit or execute the canonical source-to-results rebuild from one command."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from experiments.notebook_hygiene import NOTEBOOK_SEQUENCES
from experiments.rebuild_compare import compare_task_to_git
from experiments.rebuild_graph import RebuildContext, execute_graph, load_graph
from experiments.rebuild_notebook_dependencies import audit_notebook_dependencies


CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent
DEFAULT_GRAPH = CODE_ROOT / "configs" / "rebuild_tasks.json"


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def audit_payload(graph_path: Path) -> dict[str, object]:
    graph = load_graph(graph_path)
    sequences: dict[str, str] = {}
    for sequence in sorted(NOTEBOOK_SEQUENCES):
        audit = audit_notebook_dependencies(sequence, graph)
        sequences[sequence] = (
            "READY"
            if not audit.unregistered and not audit.duplicate_producers
            else "NOT_READY"
        )
    status = "READY" if set(sequences.values()) == {"READY"} else "NOT_READY"
    return {
        "status": status,
        "task_count": len(graph.tasks),
        "registered_outputs": len(graph.registered_outputs()),
        "notebook_sequences": sequences,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--profile", choices=("canonical", "live"), default="canonical")
    parser.add_argument("--task", action="append", default=None)
    parser.add_argument("--state-root", type=Path, default=Path(".rebuild/state"))
    parser.add_argument("--reference", default="HEAD")
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--report", type=Path, default=Path(".rebuild/report.json"))
    args = parser.parse_args(argv)
    no_compare = args.no_compare or json.loads(args.graph.read_text(encoding="utf-8")).get("result_comparison") == "not_required"

    audit = audit_payload(args.graph)
    if args.audit_only:
        _atomic_json(args.report, audit)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 0 if audit["status"] == "READY" else 1

    graph = load_graph(args.graph)
    context = RebuildContext(
        code_root=CODE_ROOT,
        repository_root=REPOSITORY_ROOT,
        state_root=args.state_root,
        profile=args.profile,
        env=os.environ,
    )
    execution = execute_graph(graph, context, selected_tasks=args.task)
    task_ids = (*execution.executed, *execution.skipped)
    comparisons = []
    if not no_compare:
        tasks = graph.task_map()
        comparisons = [
            compare_task_to_git(
                tasks[task_id],
                code_root=CODE_ROOT,
                repository_root=REPOSITORY_ROOT,
                revision=args.reference,
            )
            for task_id in task_ids
        ]
    differences = sum(item.status == "different" for item in comparisons)
    payload = {
        **audit,
        "status": "READY" if audit["status"] == "READY" and differences == 0 else "NOT_READY",
        "profile": args.profile,
        "executed": list(execution.executed),
        "skipped": list(execution.skipped),
        "reference": None if no_compare else args.reference,
        "numerical_equivalence_checked": bool(comparisons),
        "comparisons": [item.to_dict() for item in comparisons],
        "different_tasks": differences,
    }
    _atomic_json(args.report, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["audit_payload", "main"]
