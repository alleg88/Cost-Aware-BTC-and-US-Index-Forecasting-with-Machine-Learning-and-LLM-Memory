from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.run_reflection_agent_v2 import (
    _apply_policy_transition,
    _record_closed_evaluation,
    run_preflight,
    run_frozen_forward,
    run_variant,
)
from reflection_agent.v2.contracts import AllowRule, Predicate, ProposalOutput
from reflection_agent.v2.policy import ActiveAllowRule
from reflection_agent.v2.transport import SchemaCallResult


CONFIG = Path(__file__).parents[1] / "configs" / "reflection_agent_v2.yaml"
T0 = datetime(2024, 1, 1, tzinfo=UTC)


def source_hash() -> str:
    return hashlib.sha256(b"runner-source").hexdigest()


def opportunity_frame(
    *,
    stage: str = "development",
    start: datetime = T0,
    count: int = 120,
) -> pd.DataFrame:
    rows = []
    for index in range(count):
        decision = start + timedelta(hours=6 * index)
        route = "UNION_BASE" if index % 2 == 0 else "REENTRY"
        side = "LONG" if index % 4 < 2 else "SHORT"
        net = 0.001 if route == "UNION_BASE" else 0.002
        rows.append(
            {
                "opportunity_id": f"opportunity-{index}",
                "stage": stage,
                "source_role": "OOF_TEST" if stage == "development" else "FROZEN_EXACT",
                "fold_id": 0,
                "row_key": f"row-{index}",
                "source_artifact_hash": source_hash(),
                "route": route,
                "side": side,
                "member_pattern": "LSTM_ONLY",
                "episode_bar_bucket": "SECOND" if route == "REENTRY" else None,
                "previous_exit_reason": "TIMEOUT" if route == "REENTRY" else None,
                "previous_exit_reason_available_time": (
                    decision + timedelta(minutes=14) if route == "REENTRY" else pd.NaT
                ),
                "vol_regime": "HIGH" if side == "SHORT" else "NORMAL",
                "trend_regime": "DOWN" if side == "SHORT" else "UP",
                "funding_regime": "NEUTRAL",
                "oi_regime": "FLAT",
                "decision_time": decision,
                "feature_available_time": decision,
                "entry_time": decision + timedelta(minutes=15),
                "exit_time": decision + timedelta(minutes=15),
                "outcome_available_time": decision + timedelta(minutes=29),
                "gross_return": net + 0.001,
                "net_return": net,
                "round_trip_cost": 0.001,
                "exit_reason": "TIMEOUT",
                "signal_episode_id": index,
                "signal_episode_bar": 2 if route == "REENTRY" else 1,
                "path_complete": True,
            }
        )
    return pd.DataFrame(rows)


class NoChangeCaller:
    def __init__(self) -> None:
        self.calls = 0

    def call(self, *, role, messages, response_model, allowed_ids):
        self.calls += 1
        assert role == "proposal"
        value = ProposalOutput(
            source_episode_id=allowed_ids["source_episode_id"],
            decision="NO_CHANGE",
            diagnosis_code="INSUFFICIENT_EVIDENCE",
            evidence_ids=[allowed_ids["evidence_ids"][0]],
            memory_ids_used=[],
            proposed_rule=None,
            target_rule_id=None,
            hypothesis=None,
            falsifiers=[],
            confidence="LOW",
        )
        return SchemaCallResult(
            status="success",
            value=value,
            raw_content=value.model_dump_json(),
            request_hash="a" * 64,
            response_hash="b" * 64,
            schema_hash="c" * 64,
            attempts=1,
            latency_seconds=0.01,
            metadata={"model": "fake"},
            errors=(),
        )


def test_variant_run_is_complete_isolated_and_resume_idempotent(tmp_path) -> None:
    caller = NoChangeCaller()
    first = run_variant(
        "reflection_real_memory",
        output_root=tmp_path,
        opportunities=opportunity_frame(),
        caller=caller,
        config_path=CONFIG,
    )
    assert first["status"] == "complete"
    assert first["opportunities"] == 120
    assert first["union_base_trades"] == 60
    assert first["eligible_reentries"] == 60
    assert first["selected_trades"] == 60
    assert first["proposal_calls"] == 2
    assert caller.calls == 2
    variant_root = tmp_path / "development" / "reflection_real_memory"
    for name in (
        "agent_state.sqlite",
        "opportunity_ledger.parquet",
        "episode_ledger.parquet",
        "memory_ledger.parquet",
        "policy_history.jsonl",
        "evaluation_summary.json",
        "manifest.json",
    ):
        assert (variant_root / name).is_file()
    manifest = json.loads((variant_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "development"
    assert manifest["variant_id"] == "reflection_real_memory"
    assert manifest["lockbox_2026_q2_used"] is False
    assert manifest["input_hash"] == first["input_hash"]
    assert manifest["implementation_hash"] == first["implementation_hash"]
    assert len(manifest["implementation_hash"]) == 64
    assert all(len(value) == 64 for value in manifest["artifact_hashes"].values())

    resumed = run_variant(
        "reflection_real_memory",
        output_root=tmp_path,
        opportunities=opportunity_frame(),
        caller=caller,
        config_path=CONFIG,
    )
    assert resumed["resumed"] is True
    assert caller.calls == 2


def test_controls_use_same_input_but_static_and_union_never_call_llm(tmp_path) -> None:
    caller = NoChangeCaller()
    frame = opportunity_frame()
    no_memory = run_variant(
        "reflection_no_memory",
        output_root=tmp_path,
        opportunities=frame,
        caller=caller,
        config_path=CONFIG,
    )
    calls_after_memory = caller.calls
    static = run_variant(
        "static_add_all",
        output_root=tmp_path,
        opportunities=frame,
        caller=caller,
        config_path=CONFIG,
    )
    union = run_variant(
        "union_baseline",
        output_root=tmp_path,
        opportunities=frame,
        caller=caller,
        config_path=CONFIG,
    )
    assert no_memory["input_hash"] == static["input_hash"] == union["input_hash"]
    assert static["selected_trades"] == 120
    assert union["selected_trades"] == 60
    assert static["proposal_calls"] == union["proposal_calls"] == 0
    assert caller.calls == calls_after_memory
    assert (tmp_path / "development" / "reflection_no_memory" / "agent_state.sqlite").is_file()
    assert (tmp_path / "development" / "static_add_all" / "agent_state.sqlite").is_file()


def test_q2_timestamp_aborts_before_caller_or_artifact_write(tmp_path) -> None:
    caller = NoChangeCaller()
    frame = opportunity_frame()
    frame.loc[0, "outcome_available_time"] = datetime(2026, 4, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="Q2"):
        run_variant(
            "reflection_real_memory",
            output_root=tmp_path,
            opportunities=frame,
            caller=caller,
            config_path=CONFIG,
        )
    assert caller.calls == 0
    assert not (tmp_path / "development" / "reflection_real_memory").exists()


def test_preflight_records_exact_model_schema_and_probe_hashes(tmp_path) -> None:
    caller = NoChangeCaller()
    result = run_preflight(
        output_root=tmp_path,
        config_path=CONFIG,
        caller=caller,
        model_record={
            "model": "deepseek-v4-flash:cloud",
            "digest": "d" * 64,
            "capabilities": ["completion", "thinking", "tools"],
            "ollama_version": "0.32.5",
        },
    )
    assert result["passed"] is True
    assert result["model"] == "deepseek-v4-flash:cloud"
    assert result["model_digest"] == "d" * 64
    assert result["probe_status"] == "success"
    assert caller.calls == 1
    stored = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert stored == result
    assert set(result["prompt_hashes"]) == {"system", "proposal", "reflection"}
    assert len(result["implementation_hash"]) == 64
    assert result["lockbox_2026_q2_used"] is False


def test_forced_fold_close_deactivates_removed_rule_in_policy_history() -> None:
    active = ActiveAllowRule(
        rule_id="rule-to-remove",
        source_candidate_id="candidate-source",
        source_fold_id=0,
        activates_at_utc=T0,
        rule=AllowRule(
            action="ALLOW_REENTRY",
            predicates=[
                Predicate(field="side", operator="EQ", value="SHORT")
            ],
        ),
    )
    shadow_cutoff = T0 + timedelta(days=2)
    closure = SimpleNamespace(
        new_active_rule=None,
        removed_rule_id="rule-to-remove",
        evaluation=SimpleNamespace(shadow_cutoff_utc=shadow_cutoff),
    )

    active_rules, policy_history = _apply_policy_transition(
        closure,
        active_rules=[active],
        policy_history=[active],
    )

    assert active_rules == []
    assert policy_history[0].deactivates_at_utc == shadow_cutoff + timedelta(
        microseconds=1
    )


def test_only_terminal_candidate_evaluation_enters_reported_results() -> None:
    evaluations = []
    still_open = SimpleNamespace(
        status="still_open",
        evaluation=SimpleNamespace(model_dump=lambda **_: {"decision": "INCONCLUSIVE"}),
    )
    closed = SimpleNamespace(
        status="closed",
        evaluation=SimpleNamespace(model_dump=lambda **_: {"decision": "REJECT"}),
    )

    assert not _record_closed_evaluation(still_open, evaluations)
    assert _record_closed_evaluation(closed, evaluations)
    assert evaluations == [{"decision": "REJECT"}]


def test_forward_agent_cannot_learn_directly_before_artifact_write(tmp_path) -> None:
    caller = NoChangeCaller()
    forward = opportunity_frame(
        stage="forward",
        start=datetime(2025, 7, 1, tzinfo=UTC),
        count=80,
    )

    with pytest.raises(ValueError, match="frozen H1 snapshot"):
        run_variant(
            "reflection_real_memory",
            output_root=tmp_path,
            opportunities=forward,
            caller=caller,
            config_path=CONFIG,
        )

    assert caller.calls == 0
    assert not (tmp_path / "forward" / "reflection_real_memory").exists()


def test_frozen_forward_reuses_h1_snapshot_without_llm_calls(tmp_path) -> None:
    caller = NoChangeCaller()
    h1 = opportunity_frame(
        stage="h1",
        start=datetime(2025, 1, 1, tzinfo=UTC),
    )
    run_variant(
        "reflection_real_memory",
        output_root=tmp_path,
        opportunities=h1,
        caller=caller,
        config_path=CONFIG,
    )
    calls_after_h1 = caller.calls
    forward = opportunity_frame(
        stage="forward",
        start=datetime(2025, 7, 1, tzinfo=UTC),
        count=80,
    )

    result = run_frozen_forward(
        "reflection_real_memory",
        output_root=tmp_path,
        opportunities=forward,
        config_path=CONFIG,
    )

    assert caller.calls == calls_after_h1
    assert result["stage"] == "forward"
    assert result["proposal_calls"] == result["reflection_calls"] == 0
    assert result["selected_trades"] == result["union_base_trades"] == 40
    root = tmp_path / "forward" / "reflection_real_memory"
    snapshot = json.loads((root / "snapshot_manifest.json").read_text(encoding="utf-8"))
    assert snapshot["freeze_at_utc"] == "2025-07-01T00:00:00+00:00"
    assert snapshot["source_stage"] == "h1"
    assert len(snapshot["source_manifest_hash"]) == 64
    assert (root / "agent_state_snapshot.sqlite").is_file()
    assert (root / "call_log.jsonl").read_text(encoding="utf-8") == ""
