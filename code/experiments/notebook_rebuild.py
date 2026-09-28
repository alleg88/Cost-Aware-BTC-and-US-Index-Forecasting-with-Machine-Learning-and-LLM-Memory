"""Prepare one Rebuild notebook's inputs; its ordinary cells run in its kernel."""
from __future__ import annotations

from importlib import metadata
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys

from experiments.colab_runtime import _name, _requirements
from experiments.channel_rebuild_contract import MODE_ENV
from experiments.notebook_runtime import _missing_inputs, notebook_inputs
from experiments.rebuild_graph import GraphError, RebuildContext, RebuildGraph, RebuildReport, execute_graph, load_graph
from experiments.rebuild_notebook_dependencies import NOTEBOOK_INPUTS


def notebook_tasks(graph: RebuildGraph, notebook_name: str) -> tuple[str, ...]:
    """Select exact producers, including metadata read by the notebook cells."""
    contracts = {name: paths for group in NOTEBOOK_INPUTS.values() for name, paths in group.items()}
    required = tuple(dict.fromkeys((*contracts[notebook_name], *notebook_inputs(notebook_name))))
    selected = set()
    for path in required:
        exact = [task.id for task in graph.tasks if path in task.outputs]
        owners = exact or [task.id for task in graph.tasks for output in task.outputs
                          if path.startswith(output + "/") or PurePosixPath(output).match(path)]
        if not owners and not any(path == source or path.startswith(source + "/")
                                  or PurePosixPath(source).match(path) for source in graph.sources):
            raise GraphError(f"{notebook_name}: no producer or supplied source for {path}")
        selected.update(owners)
    return tuple(task_id for task_id in graph.topological_order() if task_id in selected)


def rebuild_notebook_inputs(notebook_name: str, code_root: Path) -> RebuildReport:
    code_root = Path(code_root).resolve(strict=True)
    print(f"Rebuild: {notebook_name}", flush=True)
    graph_path = code_root / "configs/rebuild_tasks.json"
    compact = graph_path.is_file() and json.loads(graph_path.read_text(encoding="utf-8")).get("result_comparison") == "not_required"
    if compact:
        os.environ[MODE_ENV] = "recomputed"
        print("Compact Rebuild: source identities are preserved; historical prediction equality is not required.", flush=True)
    else:
        os.environ.pop(MODE_ENV, None)
    versions = {"python": sys.version.split()[0]}
    if (code_root / "pyproject.toml").is_file():
        for requirement in _requirements(code_root):
            name = _name(requirement)
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = "not installed"
    print(f"Notebook kernel: Python {versions['python']}. Package versions recorded; numerical equivalence is not asserted.", flush=True)
    state = code_root / ".rebuild" / "notebooks" / Path(notebook_name).stem
    state.mkdir(parents=True, exist_ok=True)
    diagnostic = {"notebook": notebook_name, "packages": versions, "status": "PREPARING_INPUTS",
                  "channel_handoffs": "recomputed" if compact else "historical",
                  "numerical_equivalence_checked": False, "full_project_executed": False}
    report_path = state / "preparation.json"
    report_path.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    try:
        if notebook_name == "01_RQ1_A_BTC_data_labels_baseline.ipynb" and not _missing_inputs(code_root, notebook_name):
            print("Using supplied normalized inputs; model fitting runs in the following notebook cells. Raw history is not rebuilt.", flush=True)
            report = RebuildReport(executed=(), skipped=())
        else:
            graph = load_graph(graph_path)
            selected = notebook_tasks(graph, notebook_name)
            print("Selected stages: " + ", ".join(selected), flush=True)
            print("Required upstream stages run automatically; verified stages may be reused. This can take hours.", flush=True)
            identity = hashlib.sha256(json.dumps(versions, sort_keys=True).encode()).hexdigest()
            report = execute_graph(graph, RebuildContext(code_root, code_root.parent, code_root / ".rebuild" / "state",
                                                         runtime_identity=identity),
                                   selected_tasks=selected, force_tasks=selected, stream=True)
        missing = _missing_inputs(code_root, notebook_name)
        if missing:
            raise FileNotFoundError("Rebuild did not produce required notebook inputs: " + ", ".join(missing))
        diagnostic.update(status="INPUTS_READY", executed=list(report.executed), reused=list(report.skipped))
    except BaseException as error:
        diagnostic.update(status="FAILED", error=str(error))
        raise
    finally:
        report_path.write_text(json.dumps(diagnostic, indent=2) + "\n", encoding="utf-8")
    print("Inputs ready. Run All continues through this notebook; new tables and plots appear below.", flush=True)
    return report
