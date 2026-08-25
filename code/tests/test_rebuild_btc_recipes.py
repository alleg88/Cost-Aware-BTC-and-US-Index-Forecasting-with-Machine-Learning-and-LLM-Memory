from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.frozen_evidence import KIND_IDS
from experiments import run_reflection_policy_router as reflection_runner
from experiments.rebuild_btc import project_legacy_layout, write_baseline_handoff
from experiments.rebuild_graph import load_graph
from experiments.rebuild_notebook_dependencies import audit_notebook_dependencies
from experiments.notebook02_handoff import load_notebook01_handoff, load_pipeline_handoff
from experiments.replay_reflection_policy_router import (
    OfflineProviderError,
    offline_provider_guard,
)


CODE_ROOT = Path(__file__).parents[1]
REBUILD_TASKS = CODE_ROOT / "configs" / "rebuild_tasks.json"


def test_every_bitcoin_reader_input_has_one_producer():
    report = audit_notebook_dependencies(
        sequence="Bitcoin",
        graph=load_graph(REBUILD_TASKS),
    )

    assert report.unregistered == ()
    assert report.duplicate_producers == ()


def test_bitcoin_graph_uses_actual_final_notebook_entry_points():
    graph = load_graph(REBUILD_TASKS)
    tasks = graph.task_map()
    expected_modules = {
        "btc.baseline_handoff": "experiments.rebuild_btc",
        "btc.positioning_ablation": "experiments.run_positioning_ablation",
        "btc.catboost_matched": "experiments.run_catboost_matched_ablation",
        "btc.notebook02b_handoff": "experiments.notebook02b_handoff",
        "btc.all_model_raw": "experiments.all_model_sentiment_raw",
        "btc.all_model_policy": "experiments.all_model_sentiment_policy",
        "btc.stacking": "experiments.all_model_stacking",
        "btc.legacy_layout": "experiments.rebuild_btc",
        "btc.qualified_union": "experiments.run_qualified_union",
        "btc.svm_temperature": "experiments.svm_temperature_calibration",
        "btc.xgb_admission": "experiments.run_xgb_strong_move_admission",
        "btc.unified_ensemble": "experiments.run_unified_2021_ensemble",
        "btc.expected_net_control": "experiments.run_unified_expected_net_ensemble",
        "btc.lstm_gmadl": "experiments.run_lstm_gmadl_shadow",
        "btc.union_reentry": "experiments.run_union_v1_episode_reentry",
        "btc.agent.stage_frozen": "experiments.rebuild_btc",
        "btc.agent_prepare": "experiments.replay_reflection_policy_router",
        "btc.agent_real_memory": "experiments.replay_reflection_policy_router",
        "btc.agent_no_memory": "experiments.replay_reflection_policy_router",
        "btc.agent_shuffled": "experiments.replay_reflection_policy_router",
        "btc.agent_controls": "experiments.replay_reflection_policy_router",
        "btc.agent_reconcile": "experiments.reconcile_reflection_policy_router",
    }

    assert {task_id: tasks[task_id].module for task_id in expected_modules} == expected_modules
    assert tasks["btc.catboost_matched"].args[:1] == ("--full",)
    assert "--handoff" in tasks["btc.catboost_matched"].args
    assert "--preflight" not in {
        argument
        for task_id, task in tasks.items()
        if task_id.startswith("btc.agent")
        for argument in task.args
    }


def test_all_canonical_btc_agent_tasks_are_fail_closed_offline():
    tasks = load_graph(REBUILD_TASKS).task_map()
    agent_tasks = [task for task_id, task in tasks.items() if task_id.startswith("btc.agent")]

    assert agent_tasks
    assert all(dict(task.environment).get("MSC_CANONICAL_OFFLINE") == "1" for task in agent_tasks)


def test_offline_adapter_blocks_provider_without_changing_registered_agent(tmp_path, monkeypatch):
    monkeypatch.setenv("MSC_CANONICAL_OFFLINE", "1")
    inner = reflection_runner.DeepSeekSchemaCaller.__new__(
        reflection_runner.DeepSeekSchemaCaller
    )
    inner.config = object()
    caller = reflection_runner.CachedRouterCaller(
        inner,
        tmp_path,
        protocol_hash="a" * 64,
    )

    with offline_provider_guard(), pytest.raises(
        OfflineProviderError, match="provider call blocked"
    ):
        caller.call(
            role="router",
            messages=[{"role": "user", "content": "test"}],
            response_model=reflection_runner.RouterChoice,
            allowed_ids={"choice_indices": [0, 1]},
        )


def test_btc_agent_source_family_includes_calls_and_preflight():
    policy = json.loads((CODE_ROOT / "configs" / "source_evidence_policy.json").read_text("utf-8"))
    ids = {rule["id"] for rule in policy["rules"]}

    assert {"btc_agent_calls", "btc_agent_preflight"}.issubset(ids)
    assert KIND_IDS["agent_calls"] == frozenset(
        {"btc_agent_calls", "btc_agent_preflight"}
    )


def test_write_baseline_handoff_materialises_the_frozen_width_contract(tmp_path):
    metrics = pd.DataFrame(
        {
            "width_bps": [75, 65, 55],
            "sortino": [1.5, 1.0, 0.5],
            "sharpe": [0.8, 0.6, 0.4],
            "net_return": [0.03, 0.02, 0.01],
            "trades": [30, 40, 50],
        }
    )
    widths = tmp_path / "notebook01" / "selected_widths.parquet"
    pipeline = tmp_path / "notebook02" / "handoff.json"

    write_baseline_handoff(metrics, widths_path=widths, pipeline_path=pipeline)

    assert load_notebook01_handoff(widths)["width_bps"].tolist() == [75, 65, 55]
    assert load_pipeline_handoff(pipeline)["widths"] == [55, 65, 75]


def test_legacy_projection_is_an_exact_layout_bridge(tmp_path):
    raw_source = tmp_path / "v3_raw" / "none"
    policy_source = tmp_path / "v3_policy" / "none"
    raw_source.mkdir(parents=True)
    policy_source.mkdir(parents=True)
    (raw_source / "raw.txt").write_text("raw", encoding="utf-8")
    (policy_source / "policy.txt").write_text("policy", encoding="utf-8")
    raw_target = tmp_path / "legacy_raw" / "none"
    policy_target = tmp_path / "legacy_policy" / "none"

    project_legacy_layout(raw_source, policy_source, raw_target, policy_target)

    assert (raw_target / "raw.txt").read_text(encoding="utf-8") == "raw"
    assert (policy_target / "policy.txt").read_text(encoding="utf-8") == "policy"
