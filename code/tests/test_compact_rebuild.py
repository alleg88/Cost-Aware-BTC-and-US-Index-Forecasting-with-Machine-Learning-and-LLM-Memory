from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from experiments.compact_rebuild import compact_graph, compact_manifest
from experiments.rebuild_graph import RebuildGraph
from experiments.source_evidence import load_manifest, manifest_identity


def _graph_payload() -> dict[str, object]:
    """Return a small canonical graph fixture for pure transformation tests."""
    raw_grids = [
        "data/btcusdt_1m_2021_2026.parquet",
        "data/btcusdt_5min_2021_2026.parquet",
        "data/btcusdt_1h_2021_2026.parquet",
        "data/btcusdt_positioning_15min_2021_2026.parquet",
    ]

    def task(
        task_id: str,
        depends_on: list[str],
        module: str,
        inputs: list[str],
        outputs: list[str],
    ) -> dict[str, object]:
        return {
            "id": task_id,
            "depends_on": depends_on,
            "module": module,
            "args": [],
            "inputs": inputs,
            "outputs": outputs,
            "profile": "canonical",
            "comparison": "exact",
            "environment": {},
        }

    return {
        "schema_version": 1,
        "sources": [
            ".source_evidence",
            "source_evidence_manifest.json",
            "configs/final_q2_lockbox_protocol.json",
            "configs/final_q2_lockbox_manifest.json",
            "experiments/cache/final_q2_lockbox/OPENED.json",
            "experiments/cache/final_q2_lockbox/8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312",
            "experiments/cache/q2_sentiment_sensitivity",
        ],
        "tasks": [
            task(
                "source.verify",
                [],
                "experiments.rebuild_market_sentiment",
                [".source_evidence", "source_evidence_manifest.json"],
                [".rebuild/source_verified.json"],
            ),
            task(
                "source.stage_snapshots",
                ["source.verify"],
                "experiments.rebuild_market_sentiment",
                [
                    ".source_evidence",
                    "source_evidence_manifest.json",
                    ".rebuild/source_verified.json",
                ],
                [".rebuild/snapshots_staged.json"],
            ),
            task(
                "news.llm.stage_frozen",
                ["source.verify"],
                "experiments.rebuild_market_sentiment",
                [
                    ".source_evidence",
                    "source_evidence_manifest.json",
                    ".rebuild/source_verified.json",
                ],
                ["experiments/cache/news_llm_frozen"],
            ),
            task(
                "market.binance.build_m1",
                [],
                "experiments.rebuild_market_sentiment",
                [],
                raw_grids,
            ),
            task(
                "channel.stage_handoffs",
                ["source.verify"],
                "experiments.rebuild_remaining",
                [
                    ".source_evidence",
                    "source_evidence_manifest.json",
                    ".rebuild/source_verified.json",
                ],
                [".rebuild/channel_handoffs_staged.json"],
            ),
            task(
                "channel.j",
                ["market.binance.build_m1"],
                "experiments.run_event_window_tcn",
                raw_grids,
                ["experiments/cache/event_window_tcn"],
            ),
            task(
                "channel.u",
                ["channel.stage_handoffs", "channel.j", "market.binance.build_m1"],
                "experiments.run_event_window_feature_consolidation",
                [
                    ".rebuild/channel_handoffs_staged.json",
                    "experiments/cache/event_window_tcn",
                    raw_grids[0],
                    raw_grids[1],
                ],
                ["experiments/cache/event_window_feature_consolidation"],
            ),
            task(
                "channel.v",
                ["channel.u", "channel.j"],
                "experiments.run_event_window_direction_head",
                [
                    "experiments/cache/event_window_feature_consolidation",
                    "experiments/cache/event_window_tcn",
                    raw_grids[0],
                ],
                ["experiments/cache/event_window_direction_head"],
            ),
            task(
                "channel.w",
                ["channel.u", "channel.j"],
                "experiments.run_channel_vs_volatility_ablation",
                [
                    "experiments/cache/event_window_feature_consolidation",
                    "experiments/cache/event_window_tcn",
                    raw_grids[0],
                    raw_grids[1],
                    raw_grids[3],
                ],
                ["experiments/cache/channel_vs_volatility_ablation"],
            ),
            task(
                "final.q2.verify",
                [],
                "experiments.rebuild_remaining",
                [
                    "configs/final_q2_lockbox_protocol.json",
                    "configs/final_q2_lockbox_manifest.json",
                    "experiments/cache/final_q2_lockbox/OPENED.json",
                    "experiments/cache/final_q2_lockbox/8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312",
                ],
                [".rebuild/final_q2_verified.json"],
            ),
            task(
                "final.q2.sentiment_sensitivity",
                ["final.q2.verify"],
                "experiments.rebuild_remaining",
                [
                    ".rebuild/final_q2_verified.json",
                    "experiments/cache/q2_sentiment_sensitivity",
                ],
                [".rebuild/q2_sentiment_sensitivity_verified.json"],
            ),
        ],
    }


def _tasks(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    return {task["id"]: task for task in payload["tasks"]}  # type: ignore[index]


def test_compact_graph_is_pure_and_replaces_handoff_with_rebuild_chain() -> None:
    payload = _graph_payload()
    before = copy.deepcopy(payload)

    compact = compact_graph(payload)

    assert payload == before
    assert compact["result_comparison"] == "not_required"
    tasks = compact["tasks"]
    assert isinstance(tasks, list)
    assert len(tasks) == len(payload["tasks"]) - 1 + 6  # type: ignore[arg-type]
    ids = [task["id"] for task in tasks]
    assert "channel.stage_handoffs" not in ids
    assert ids[ids.index("channel.j") + 1 : ids.index("channel.u")] == [
        "channel.m",
        "channel.n",
        "channel.o",
        "channel.p",
        "channel.q",
        "channel.r",
    ]


def test_compact_graph_validates_and_declares_all_channel_contracts() -> None:
    compact = compact_graph(_graph_payload())
    RebuildGraph.from_dict(compact).validate_artifacts()
    tasks = _tasks(compact)
    expected_dependencies = {
        "channel.m": ["channel.j"],
        "channel.n": ["channel.m", "channel.j"],
        "channel.o": ["channel.n", "channel.j"],
        "channel.p": ["channel.o", "channel.n", "channel.j"],
        "channel.q": ["channel.p", "channel.o", "market.binance.build_m1"],
        "channel.r": [
            "channel.q",
            "channel.p",
            "channel.o",
            "market.binance.build_m1",
        ],
        "channel.u": [
            "channel.r",
            "channel.p",
            "channel.n",
            "channel.o",
            "channel.j",
            "market.binance.build_m1",
        ],
    }
    for task_id, dependencies in expected_dependencies.items():
        assert tasks[task_id]["depends_on"] == dependencies

    raw_grids = {
        "data/btcusdt_1m_2021_2026.parquet",
        "data/btcusdt_5min_2021_2026.parquet",
        "data/btcusdt_1h_2021_2026.parquet",
        "data/btcusdt_positioning_15min_2021_2026.parquet",
    }
    for task_id in ("channel.m", "channel.n", "channel.o", "channel.p"):
        assert raw_grids.issubset(tasks[task_id]["inputs"])

    assert tasks["channel.q"]["inputs"] == [
        "experiments/cache/event_window_conditional_opportunity",
        "experiments/cache/event_window_magnitude_timing",
        "data/btcusdt_1m_2021_2026.parquet",
    ]
    assert tasks["channel.r"]["inputs"] == [
        "experiments/cache/event_window_economic_feasibility",
        "experiments/cache/event_window_conditional_opportunity",
        "experiments/cache/event_window_magnitude_timing",
        "data/btcusdt_1m_2021_2026.parquet",
    ]
    assert ".rebuild/channel_handoffs_staged.json" not in tasks["channel.u"]["inputs"]
    assert {
        "experiments/cache/event_window_tcn",
        "experiments/cache/event_window_opportunity_head",
        "experiments/cache/event_window_magnitude_timing",
        "experiments/cache/event_window_conditional_opportunity",
        "experiments/cache/event_window_timing_policy_repair",
        "data/btcusdt_1m_2021_2026.parquet",
        "data/btcusdt_5min_2021_2026.parquet",
        "data/btcusdt_1h_2021_2026.parquet",
        "data/btcusdt_positioning_15min_2021_2026.parquet",
    }.issubset(tasks["channel.u"]["inputs"])

    for task_id in (
        "channel.j",
        "channel.m",
        "channel.n",
        "channel.o",
        "channel.p",
        "channel.q",
        "channel.r",
        "channel.u",
        "channel.v",
        "channel.w",
    ):
        assert tasks[task_id]["environment"] == {
            "MSC_REBUILD_CHANNEL_HANDOFFS": "recomputed"
        }


def test_compact_graph_preserves_legacy_and_q2_tasks_byte_semantically() -> None:
    payload = _graph_payload()
    compact = compact_graph(payload)
    original_tasks = _tasks(payload)
    compact_tasks = _tasks(compact)

    for task_id in (
        "source.verify",
        "source.stage_snapshots",
        "news.llm.stage_frozen",
        "final.q2.verify",
        "final.q2.sentiment_sensitivity",
    ):
        assert compact_tasks[task_id] == original_tasks[task_id]
    assert compact["sources"] == payload["sources"]
    assert compact_tasks["channel.v"]["depends_on"] == original_tasks["channel.v"]["depends_on"]
    assert compact_tasks["channel.w"]["depends_on"] == original_tasks["channel.w"]["depends_on"]


def test_compact_manifest_drops_only_frozen_handoffs_and_recomputes_identity(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "id": "public",
            "class": "public_raw",
            "provider": "fixture",
            "path": "data/public.txt",
            "stage_path": "data/public.txt",
            "bytes": 11,
            "sha256": "b" * 64,
        },
        {
            "id": "snapshot",
            "class": "snapshot_raw",
            "provider": "fixture",
            "path": "data/snapshot.txt",
            "stage_path": "data/snapshot.txt",
            "bytes": 3,
            "sha256": "c" * 64,
        },
        {
            "id": "handoff",
            "class": "frozen_handoff",
            "provider": "fixture",
            "path": "cache/handoff.txt",
            "stage_path": "cache/handoff.txt",
            "bytes": 7,
            "sha256": "d" * 64,
        },
    ]
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "policy_sha256": "a" * 64,
        "manifest_identity": manifest_identity(rows),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "class_summary": {
            "frozen_handoff": {"bytes": 7, "files": 1},
            "public_raw": {"bytes": 11, "files": 1},
            "snapshot_raw": {"bytes": 3, "files": 1},
        },
        "files": rows,
    }
    before_rows = copy.deepcopy(payload["files"])
    compact_policy = b'{"schema_version":1,"rules":[]}'

    compact = compact_manifest(payload, compact_policy)

    assert payload["files"] == before_rows
    assert compact["file_count"] == 2
    assert compact["total_bytes"] == 14
    assert compact["class_summary"] == {
        "public_raw": {"bytes": 11, "files": 1},
        "snapshot_raw": {"bytes": 3, "files": 1},
    }
    assert compact["policy_sha256"] == hashlib.sha256(compact_policy).hexdigest()
    assert compact["manifest_identity"] == manifest_identity(compact["files"])
    assert not any(row["class"] == "frozen_handoff" for row in compact["files"])
    assert [row for row in compact["files"] if row["class"] != "frozen_handoff"] == [
        row for row in before_rows if row["class"] != "frozen_handoff"
    ]

    derived_path = tmp_path / "source_evidence_manifest.json"
    derived_path.write_text(
        json.dumps(compact, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    assert load_manifest(derived_path) == compact


def test_compact_manifest_without_policy_bytes_preserves_policy_identity() -> None:
    payload = {
        "schema_version": 1,
        "hash_algorithm": "sha256",
        "policy_sha256": "a" * 64,
        "manifest_identity": "unused",
        "file_count": 2,
        "total_bytes": 3,
        "class_summary": {},
        "files": [
            {
                "id": "raw",
                "class": "snapshot_raw",
                "provider": "fixture",
                "path": "data/raw.txt",
                "stage_path": "data/raw.txt",
                "bytes": 1,
                "sha256": "b" * 64,
            },
            {
                "id": "handoff",
                "class": "frozen_handoff",
                "provider": "fixture",
                "path": "cache/handoff.txt",
                "stage_path": "cache/handoff.txt",
                "bytes": 2,
                "sha256": "c" * 64,
            },
        ],
    }

    compact = compact_manifest(payload)

    assert compact["policy_sha256"] == "a" * 64
    assert compact["file_count"] == 1
    assert compact["total_bytes"] == 1
