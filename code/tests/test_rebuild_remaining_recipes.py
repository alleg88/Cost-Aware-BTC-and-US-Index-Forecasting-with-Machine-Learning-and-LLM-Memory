from __future__ import annotations

from pathlib import Path, PurePosixPath, PureWindowsPath
import hashlib
import json
import subprocess
import sys

import pytest

from experiments.rebuild_graph import load_graph
from experiments.rebuild_notebook_dependencies import audit_notebook_dependencies
from experiments.rebuild_remaining import verify_completed_q2


CODE_ROOT = Path(__file__).parents[1]
REBUILD_TASKS = CODE_ROOT / "configs" / "rebuild_tasks.json"
CHANNEL_TAG = "BTCUSDT_dev_1hw60_5min_macro-hard_fa48d9681c"


@pytest.mark.parametrize("sequence", ("Indices", "Channels", "Final confirmation"))
def test_every_remaining_notebook_reader_has_a_registered_input(sequence):
    report = audit_notebook_dependencies(sequence, load_graph(REBUILD_TASKS))

    assert report.unregistered == ()
    assert report.duplicate_producers == ()


def test_index_graph_registers_the_same_isolated_pipeline_for_both_streams():
    tasks = load_graph(REBUILD_TASKS).task_map()
    modules = {
        "gate": "experiments.index_replication",
        "models": "experiments.index_replication",
        "forward": "experiments.index_all_model_forward",
        "coverage": "experiments.index_trade_coverage",
        "side": "experiments.index_side_calibration",
        "ensemble": "experiments.index_all_model_ensemble",
        "channel": "experiments.index_channel_replication",
    }

    for stream in ("usa500", "usatech"):
        for step, module in modules.items():
            task = tasks[f"index.{stream}.{step}"]
            assert task.module == module
            assert task.args[:2] == ("--stream", stream)
            assert all(
                f"/{stream}" in path or f"{stream}_" in path
                for path in task.outputs
            )


def test_channel_graph_uses_the_frozen_handoffs_before_rebuilding_u_v_w():
    graph = load_graph(REBUILD_TASKS)
    tasks = graph.task_map()
    compact = json.loads(REBUILD_TASKS.read_text(encoding="utf-8")).get(
        "result_comparison"
    ) == "not_required"

    assert tasks["channel.study"].module == "experiments.channel_study"
    assert tasks["channel.ranking"].args == (
        "--events",
        f"experiments/cache/channel_study/{CHANNEL_TAG}/events.parquet",
    )
    assert tasks["channel.u"].module == "experiments.run_event_window_feature_consolidation"
    assert tasks["channel.v"].module == "experiments.run_event_window_direction_head"
    assert tasks["channel.w"].module == "experiments.run_channel_vs_volatility_ablation"
    order = graph.topological_order()
    if compact:
        assert "channel.stage_handoffs" not in tasks
        assert [
            task_id
            for task_id in order
            if task_id in {"channel.j", "channel.m", "channel.n", "channel.o", "channel.p", "channel.q", "channel.r"}
        ] == [
            "channel.j",
            "channel.m",
            "channel.n",
            "channel.o",
            "channel.p",
            "channel.q",
            "channel.r",
        ]
        assert order.index("channel.r") < order.index("channel.u")
    else:
        assert tasks["channel.stage_handoffs"].module == "experiments.rebuild_remaining"
        assert order.index("channel.stage_handoffs") < order.index("channel.u")
    assert order.index("channel.u") < order.index("channel.v")
    assert order.index("channel.u") < order.index("channel.w")


def test_final_graph_only_verifies_the_once_opened_identity():
    graph = load_graph(REBUILD_TASKS)
    tasks = graph.task_map()

    assert tasks["final.q2.verify"].module == "experiments.rebuild_remaining"
    assert tasks["final.q2.verify"].args == ("verify_q2",)
    assert tasks["final.q2.sentiment_sensitivity"].module == "experiments.rebuild_remaining"
    assert tasks["final.q2.sentiment_sensitivity"].args == (
        "verify_q2_sensitivity",
    )
    assert all(
        task.module != "experiments.final_q2_lockbox_reconstruction"
        and "open-q2" not in task.args
        for task in graph.tasks
    )


def test_channel_handoff_cli_stages_exact_bytes_and_writes_a_receipt(tmp_path):
    evidence = tmp_path / "evidence"
    source = evidence / "files" / "experiments" / "cache" / "handoff" / "state.json"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"channel handoff")
    import hashlib

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifest_identity": "a" * 64,
                "files": [
                    {
                        "id": "channel_handoff_n",
                        "class": "frozen_handoff",
                        "path": "experiments/cache/handoff/state.json",
                        "stage_path": "experiments/cache/handoff/state.json",
                        "bytes": source.stat().st_size,
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    work = tmp_path / "work"
    work.mkdir()
    receipt = tmp_path / "receipt.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.rebuild_remaining",
            "stage_channel",
            "--evidence-root",
            str(evidence),
            "--manifest",
            str(manifest),
            "--code-root",
            str(work),
            "--receipt",
            str(receipt),
        ],
        cwd=CODE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (work / "experiments/cache/handoff/state.json").read_bytes() == b"channel handoff"
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload == {
        "bytes": len(b"channel handoff"),
        "files": 1,
        "kind": "channel_handoffs",
        "manifest_identity": "a" * 64,
    }


def test_completed_q2_verifier_checks_frozen_results_not_later_source_files(tmp_path):
    receipt = tmp_path / "q2_verified.json"

    payload = verify_completed_q2(receipt)

    assert payload["state"] == "COMPLETE"
    assert payload["protocol_hash"] == (
        "8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312"
    )
    assert payload["artifacts"] > 0
    assert json.loads(receipt.read_text(encoding="utf-8")) == payload


def test_completed_q2_manifest_is_portable_and_lf_stable():
    protocol_hash = (
        "8185dee25cc468ed9be8b79aad59eb33eeda874f4cb245c31883ef127f205312"
    )
    result_root = CODE_ROOT / "experiments" / "cache" / "final_q2_lockbox" / protocol_hash
    complete = json.loads((result_root / "COMPLETE.json").read_text(encoding="utf-8"))
    manifest = result_root / "manifest.json"
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    for entry in manifest_payload["artifact_hashes"].values():
        registered = str(entry["path"])
        assert not PurePosixPath(registered).is_absolute()
        assert not PureWindowsPath(registered).is_absolute()
    input_audit = json.loads((result_root / "input_audit.json").read_text("utf-8"))
    for registered in input_audit["sentiment_outputs"].values():
        assert not PurePosixPath(registered).is_absolute()
        assert not PureWindowsPath(registered).is_absolute()
    manifest_bytes = manifest.read_bytes()
    assert b"\r\n" not in manifest_bytes
    assert "*.json text eol=lf" in (CODE_ROOT.parent / ".gitattributes").read_text(
        encoding="utf-8"
    )
    assert hashlib.sha256(manifest_bytes).hexdigest() == complete["result_hashes"][
        "manifest"
    ]


def test_final_q2_json_writer_is_lf_stable(tmp_path):
    from experiments.final_q2_lockbox_runner import _atomic_json

    output = _atomic_json(tmp_path / "artifact.json", {"alpha": 1, "beta": 2})

    assert b"\r\n" not in output.read_bytes()


def test_q2_sensitivity_audit_hash_is_lf_stable():
    root = CODE_ROOT / "experiments" / "cache" / "q2_sentiment_sensitivity" / "results"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    audit = (root / "audit.json").read_bytes()

    assert b"\r\n" not in audit
    assert hashlib.sha256(audit).hexdigest() == manifest["artifact_sha256"]["audit.json"]
