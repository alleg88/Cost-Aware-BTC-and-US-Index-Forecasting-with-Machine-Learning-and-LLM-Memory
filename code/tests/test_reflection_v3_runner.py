from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.run_reflection_agent_v3 import (
    _update_policy_history,
    run_preflight,
    run_variant,
)
from reflection_agent.v2.transport import SchemaCallResult
from reflection_agent.v3.config import load_v3_config
from reflection_agent.v3.contracts import ProposalChoiceOutput
from reflection_agent.v3.orchestrator import ReflectionOrchestrator
from reflection_agent.v3.policy import ActiveAllowRule


UTC = timezone.utc
CONFIG = Path(__file__).parents[1] / "configs" / "reflection_agent_v3.yaml"
DIGEST = "5166728b9358990e5f6c34f87cbe48716be2f2cd2d3b98527dff27ea755bf3ba"


def _stage_frame(stage: str) -> pd.DataFrame:
    starts = {
        "development": datetime(2024, 1, 1, tzinfo=UTC),
        "h1": datetime(2025, 1, 1, tzinfo=UTC),
        "forward": datetime(2025, 7, 1, tzinfo=UTC),
    }
    source_role = "OOF_TEST" if stage == "development" else "FROZEN_EXACT"
    start = starts[stage]
    rows: list[dict[str, object]] = []
    for index in range(20):
        decision = start + timedelta(minutes=30 * (index + 1))
        side = "LONG" if index % 2 == 0 else "SHORT"
        rows.append(
            {
                "opportunity_id": f"{stage}_candidate_{index:02d}",
                "stage": stage,
                "source_role": source_role,
                "fold_id": 0,
                "row_key": f"{stage}_row_{index:02d}",
                "source_artifact_hash": "a" * 64,
                "decision_time": decision,
                "feature_available_time": decision,
                "outcome_available_time": decision + timedelta(minutes=25),
                "entry_time": decision + timedelta(minutes=15),
                "exit_time": decision + timedelta(minutes=15),
                "route": "COVERAGE_CANDIDATE",
                "side": side,
                "confidence_tier": "HIGH_EXTRA",
                "signal_run_bucket": "FIRST",
                "gross_return": 0.003,
                "net_return": 0.002,
                "round_trip_cost": 0.001,
                "exit_reason": "TAKE_PROFIT",
                "vol_regime": "NORMAL",
                "trend_regime": "UP" if side == "LONG" else "DOWN",
                "funding_regime": "NEUTRAL",
                "oi_regime": "RISING",
                "path_complete": True,
            }
        )
    for index in range(2):
        decision = start + timedelta(minutes=5 + 300 * index)
        rows.append(
            {
                "opportunity_id": f"{stage}_union_{index}",
                "stage": stage,
                "source_role": source_role,
                "fold_id": 0,
                "row_key": f"{stage}_union_row_{index}",
                "source_artifact_hash": "a" * 64,
                "decision_time": decision,
                "feature_available_time": decision,
                "outcome_available_time": decision + timedelta(minutes=20),
                "entry_time": decision + timedelta(minutes=15),
                "exit_time": decision + timedelta(minutes=15),
                "route": "UNION_BASE",
                "side": "LONG" if index == 0 else "SHORT",
                "confidence_tier": pd.NA,
                "signal_run_bucket": pd.NA,
                "gross_return": 0.002,
                "net_return": 0.001,
                "round_trip_cost": 0.001,
                "exit_reason": "TIMEOUT",
                "vol_regime": "NORMAL",
                "trend_regime": "FLAT",
                "funding_regime": "NEUTRAL",
                "oi_regime": "FLAT",
                "path_complete": True,
            }
        )
    return pd.DataFrame(rows)


def _frames() -> dict[str, pd.DataFrame]:
    return {stage: _stage_frame(stage) for stage in ("development", "h1", "forward")}


class NoChangeCaller:
    def __init__(self, *, fail: bool = False) -> None:
        self.config = load_v3_config(CONFIG)
        self.fail = fail
        self.calls = 0

    def call(self, *, role, messages, response_model, allowed_ids):
        self.calls += 1
        schema_hash, request_hash = ReflectionOrchestrator.transport_hashes(
            self.config,
            role=role,
            messages=messages,
            response_model=response_model,
            allowed_ids=allowed_ids,
        )
        if self.fail:
            return SchemaCallResult(
                status="transport_error",
                value=None,
                raw_content="",
                request_hash=request_hash,
                response_hash="0" * 64,
                schema_hash=schema_hash,
                attempts=1,
                latency_seconds=0.0,
                metadata={},
                errors=("TimeoutError: test",),
            )
        value = ProposalChoiceOutput(
            choice_index=0,
            evidence_indices=[allowed_ids["evidence_indices"][0]],
            memory_indices=[],
        )
        return SchemaCallResult(
            status="success",
            value=value,
            raw_content=value.model_dump_json(),
            request_hash=request_hash,
            response_hash="1" * 64,
            schema_hash=schema_hash,
            attempts=1,
            latency_seconds=0.0,
            metadata={},
            errors=(),
        )


class ExplodingCaller:
    calls = 0

    def call(self, **kwargs):
        self.calls += 1
        raise AssertionError("static control called the model")


def _table_count(database: Path, table: str) -> int:
    with sqlite3.connect(database) as connection:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def test_removed_policy_rule_closes_strictly_after_shadow_cutoff() -> None:
    activation = datetime(2025, 1, 1, tzinfo=UTC)
    cutoff = activation + timedelta(days=1)
    rule = ActiveAllowRule(
        rule_id="rule-1",
        source_candidate_id="candidate-1",
        source_stage="h1",
        source_fold_id=0,
        activates_at_utc=activation,
        rule={
            "action": "ALLOW_CANDIDATE",
            "predicates": [{"field": "side", "operator": "EQ", "value": "LONG"}],
        },
    )
    closure = SimpleNamespace(
        status="closed",
        new_active_rule=None,
        removed_rule_id=rule.rule_id,
        evaluation=SimpleNamespace(shadow_cutoff_utc=cutoff),
    )

    updated = _update_policy_history([rule], [closure])

    assert updated[0].deactivates_at_utc == cutoff + timedelta(microseconds=1)


def test_one_database_spans_stages_and_completed_resume_duplicates_nothing(tmp_path) -> None:
    caller = NoChangeCaller()
    first = run_variant(
        "reflection_real_memory",
        output_root=tmp_path,
        stage_frames=_frames(),
        caller=caller,
        stages=("development", "h1", "forward"),
    )
    database = tmp_path / "reflection_real_memory" / "agent_state.sqlite"
    assert first["status"] == "complete"
    assert first["completed_stages"] == ["development", "h1", "forward"]
    assert _table_count(database, "stage_checkpoints") == 3
    before = {
        table: _table_count(database, table)
        for table in ("runner_call_cache", "call_audit", "retrieval_audits")
    }
    calls_before = caller.calls
    resumed = run_variant(
        "reflection_real_memory",
        output_root=tmp_path,
        stage_frames=_frames(),
        caller=caller,
        stages=("development", "h1", "forward"),
    )
    after = {
        table: _table_count(database, table)
        for table in ("runner_call_cache", "call_audit", "retrieval_audits")
    }
    assert resumed["resumed"] is True
    assert before == after
    assert caller.calls == calls_before


def test_variants_share_opportunity_hash_but_keep_isolated_static_stores(tmp_path) -> None:
    frames = _frames()
    exploding = ExplodingCaller()
    union = run_variant(
        "union_baseline",
        output_root=tmp_path,
        stage_frames=frames,
        caller=exploding,
        stages=("development", "h1", "forward"),
    )
    high = run_variant(
        "static_high_extra",
        output_root=tmp_path,
        stage_frames=frames,
        caller=exploding,
        stages=("development", "h1", "forward"),
    )
    assert exploding.calls == 0
    assert union["opportunity_hash"] == high["opportunity_hash"]
    assert union["total_calls"] == high["total_calls"] == 0
    assert (
        tmp_path / "union_baseline" / "agent_state.sqlite"
    ) != tmp_path / "static_high_extra" / "agent_state.sqlite"
    for stage in ("development", "h1", "forward"):
        union_ledger = pd.read_parquet(
            tmp_path / "union_baseline" / "stages" / stage / "selected_ledger.parquet"
        )
        high_ledger = pd.read_parquet(
            tmp_path / "static_high_extra" / "stages" / stage / "selected_ledger.parquet"
        )
        columns = ["opportunity_id", "gross_return", "net_return", "round_trip_cost"]
        left = union_ledger.loc[union_ledger["route"].eq("UNION_BASE"), columns]
        right = high_ledger.loc[high_ledger["route"].eq("UNION_BASE"), columns]
        pd.testing.assert_frame_equal(
            left.sort_values("opportunity_id").reset_index(drop=True),
            right.sort_values("opportunity_id").reset_index(drop=True),
        )


def test_transport_failure_coverage_invalidates_stage_and_q2_is_unreachable(tmp_path) -> None:
    failed = run_variant(
        "reflection_no_memory",
        output_root=tmp_path,
        stage_frames={"development": _stage_frame("development")},
        caller=NoChangeCaller(fail=True),
        stages=("development",),
    )
    assert failed["stage_summaries"]["development"]["status"] == (
        "INVALID_TRANSPORT_COVERAGE"
    )
    assert failed["stage_summaries"]["development"]["call_failure_fraction"] == 1.0

    contaminated = _stage_frame("forward")
    contaminated.loc[0, "outcome_available_time"] = pd.Timestamp(
        "2026-04-01", tz="UTC"
    )
    with pytest.raises(ValueError, match="Q2"):
        run_variant(
            "union_baseline",
            output_root=tmp_path / "q2",
            stage_frames={"forward": contaminated},
            stages=("forward",),
        )


def test_preflight_pins_digest_schema_sources_and_q2_guard(tmp_path) -> None:
    caller = NoChangeCaller()
    result = run_preflight(
        output_root=tmp_path,
        caller=caller,
        model_record={
            "model": "deepseek-v4-flash:cloud",
            "digest": DIGEST,
            "capabilities": ["thinking"],
            "ollama_version": "0.32.5",
        },
        stage_frames=_frames(),
    )
    assert result["passed"] is True
    assert result["model_digest"] == DIGEST
    assert result["stage_candidate_counts"] == {
        "development": 20,
        "h1": 20,
        "forward": 20,
    }
    assert result["lockbox_2026_q2_used"] is False
    persisted = json.loads((tmp_path / "preflight.json").read_text(encoding="utf-8"))
    assert persisted["opportunity_hash"] == result["opportunity_hash"]
