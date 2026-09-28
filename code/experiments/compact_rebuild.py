"""Pure transformations for the bounded compact-Rebuild candidate."""

from __future__ import annotations

from copy import deepcopy
import hashlib

from experiments.source_evidence import manifest_identity


_CHANNEL_ENV = {"MSC_REBUILD_CHANNEL_HANDOFFS": "recomputed"}
_CHANNEL_TASKS = frozenset(
    {"channel.j", "channel.m", "channel.n", "channel.o", "channel.p", "channel.q", "channel.r", "channel.u", "channel.v", "channel.w"}
)
_RAW_GRIDS = [
    "data/btcusdt_1m_2021_2026.parquet",
    "data/btcusdt_5min_2021_2026.parquet",
    "data/btcusdt_1h_2021_2026.parquet",
    "data/btcusdt_positioning_15min_2021_2026.parquet",
]
_J_CACHE = "experiments/cache/event_window_tcn"
_M_CACHE = "experiments/cache/event_window_adaptive_large_move"
_N_CACHE = "experiments/cache/event_window_opportunity_head"
_O_CACHE = "experiments/cache/event_window_magnitude_timing"
_P_CACHE = "experiments/cache/event_window_conditional_opportunity"
_Q_CACHE = "experiments/cache/event_window_economic_feasibility"
_R_CACHE = "experiments/cache/event_window_timing_policy_repair"
_MINUTE_GRID = _RAW_GRIDS[0]
_COMPACT_U_DEPENDENCIES = [
    "channel.r",
    "channel.p",
    "channel.n",
    "channel.o",
    "channel.j",
    "market.binance.build_m1",
]
_COMPACT_U_INPUTS = [
    _R_CACHE,
    _P_CACHE,
    _N_CACHE,
    _O_CACHE,
    _J_CACHE,
    *_RAW_GRIDS,
]


def _task(
    task_id: str,
    depends_on: list[str],
    module: str,
    inputs: list[str],
    output: str,
) -> dict[str, object]:
    return {
        "id": task_id,
        "depends_on": depends_on,
        "module": module,
        "args": ["--stage", "dev"],
        "inputs": inputs,
        "outputs": [output],
        "profile": "canonical",
        "comparison": {"mode": "numeric", "atol": 1e-7, "rtol": 1e-6},
        "environment": dict(_CHANNEL_ENV),
    }


def _compact_tasks() -> list[dict[str, object]]:
    return [
        _task(
            "channel.m",
            ["channel.j"],
            "experiments.run_event_window_adaptive_large_move",
            [_J_CACHE, *_RAW_GRIDS],
            _M_CACHE,
        ),
        _task(
            "channel.n",
            ["channel.m", "channel.j"],
            "experiments.run_event_window_opportunity_head",
            [_M_CACHE, _J_CACHE, *_RAW_GRIDS],
            _N_CACHE,
        ),
        _task(
            "channel.o",
            ["channel.n", "channel.j"],
            "experiments.run_event_window_magnitude_timing",
            [_N_CACHE, _J_CACHE, *_RAW_GRIDS],
            _O_CACHE,
        ),
        _task(
            "channel.p",
            ["channel.o", "channel.n", "channel.j"],
            "experiments.run_event_window_conditional_opportunity",
            [_O_CACHE, _N_CACHE, _J_CACHE, *_RAW_GRIDS],
            _P_CACHE,
        ),
        _task(
            "channel.q",
            ["channel.p", "channel.o", "market.binance.build_m1"],
            "experiments.run_event_window_economic_feasibility",
            [_P_CACHE, _O_CACHE, _MINUTE_GRID],
            _Q_CACHE,
        ),
        _task(
            "channel.r",
            [
                "channel.q",
                "channel.p",
                "channel.o",
                "market.binance.build_m1",
            ],
            "experiments.run_event_window_timing_policy_repair",
            [_Q_CACHE, _P_CACHE, _O_CACHE, _MINUTE_GRID],
            _R_CACHE,
        ),
    ]


def compact_graph(payload: dict[str, object]) -> dict[str, object]:
    """Return a compact graph without mutating the canonical graph payload."""
    if not isinstance(payload, dict):
        raise TypeError("rebuild graph payload must be a dictionary")
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list) or not all(isinstance(task, dict) for task in raw_tasks):
        raise ValueError("rebuild graph tasks must be a list of dictionaries")

    result = deepcopy(payload)
    tasks = result["tasks"]
    assert isinstance(tasks, list)
    task_ids = [task.get("id") for task in tasks if isinstance(task, dict)]
    required = {"channel.stage_handoffs", "channel.j", "channel.u"}
    missing = sorted(required.difference(task_ids))
    if missing:
        raise ValueError(f"canonical graph is missing compact anchors: {missing}")
    new_tasks = _compact_tasks()
    new_ids = {task["id"] for task in new_tasks}
    if new_ids.intersection(task_ids):
        raise ValueError("canonical graph already contains compact channel tasks")

    transformed: list[dict[str, object]] = []
    inserted = False
    for task in tasks:
        assert isinstance(task, dict)
        if task.get("id") == "channel.stage_handoffs":
            continue
        transformed.append(task)
        if task.get("id") == "channel.j":
            transformed.extend(deepcopy(new_tasks))
            inserted = True
    if not inserted:
        raise ValueError("canonical graph is missing channel.j")

    for task in transformed:
        if task.get("id") == "channel.u":
            task["depends_on"] = list(_COMPACT_U_DEPENDENCIES)
            task["inputs"] = list(_COMPACT_U_INPUTS)
        if task.get("id") not in _CHANNEL_TASKS:
            continue
        environment = task.get("environment", {})
        if not isinstance(environment, dict):
            raise ValueError(f"task {task.get('id')} environment must be a dictionary")
        task["environment"] = {**environment, **_CHANNEL_ENV}
    result["tasks"] = transformed
    result["result_comparison"] = "not_required"
    return result


def compact_manifest(
    payload: dict[str, object], compact_policy_bytes: bytes | None = None
) -> dict[str, object]:
    """Drop only handoff rows and recompute the source-manifest identity fields."""
    if not isinstance(payload, dict):
        raise TypeError("source manifest payload must be a dictionary")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list) or not all(isinstance(row, dict) for row in raw_files):
        raise ValueError("source manifest files must be a list of dictionaries")

    retained = [
        deepcopy(row)
        for row in raw_files
        if row.get("class") != "frozen_handoff"
    ]
    for row in retained:
        value = row.get("bytes")
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"source manifest row has invalid byte count: {row.get('path')!r}")

    class_summary: dict[str, dict[str, int]] = {}
    for row in retained:
        evidence_class = str(row.get("class", ""))
        summary = class_summary.setdefault(evidence_class, {"bytes": 0, "files": 0})
        summary["bytes"] += int(row["bytes"])
        summary["files"] += 1

    result = deepcopy(payload)
    result["files"] = retained
    result["file_count"] = len(retained)
    result["total_bytes"] = sum(int(row["bytes"]) for row in retained)
    result["class_summary"] = {
        name: class_summary[name] for name in sorted(class_summary)
    }
    result["manifest_identity"] = manifest_identity(retained)
    if compact_policy_bytes is not None:
        if not isinstance(compact_policy_bytes, bytes):
            raise TypeError("compact policy bytes must be bytes")
        result["policy_sha256"] = hashlib.sha256(compact_policy_bytes).hexdigest()
    return result


__all__ = ["compact_graph", "compact_manifest"]
