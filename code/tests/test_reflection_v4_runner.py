from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments import run_reflection_policy_router as runner
from reflection_agent.v2.transport import SchemaCallResult
from reflection_agent.v4.config import load_v4_config
from reflection_agent.v4.contracts import RouterChoice


def test_registered_implementation_identity_is_cross_platform():
    repository_root = Path(__file__).parents[2]
    attributes = (repository_root / ".gitattributes").read_text(encoding="utf-8")
    assert "*.py text eol=lf" in attributes
    assert "*.yaml text eol=lf" in attributes

    manifest = json.loads(
        (
            runner.CACHE / "common" / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert runner._implementation_hash() == manifest["implementation_hash"]


class FakeCaller:
    def __init__(self, choice_index: int) -> None:
        self.choice_index = choice_index
        self.calls: list[dict[str, object]] = []

    def call(self, **kwargs):
        self.calls.append(kwargs)
        allowed = kwargs["allowed_ids"]["choice_indices"]
        choice = self.choice_index if self.choice_index in allowed else 0
        value = RouterChoice(
            choice_index=choice,
            evidence_indices=[],
            memory_indices=[],
        )
        return SchemaCallResult(
            status="success",
            value=value,
            raw_content=value.model_dump_json(),
            request_hash=f"{len(self.calls):064x}",
            response_hash=f"{len(self.calls) + 100:064x}",
            schema_hash="c" * 64,
            attempts=1,
            latency_seconds=0.01,
            metadata={},
            errors=(),
        )


def _stage_frame(stage: str, start: str) -> pd.DataFrame:
    base = pd.Timestamp(start, tz="UTC")
    common = {
        "confidence_tier": "HIGH_EXTRA",
        "signal_run_bucket": "FIRST",
        "vol_regime": "HIGH",
        "trend_regime": "FLAT",
        "funding_regime": "POSITIVE",
        "oi_regime": "FLAT",
        "round_trip_cost": 0.001,
        "xgb_available": True,
        "xgb_p_move_raw": 0.90,
        "xgb_direction_confidence": 0.90,
        "xgb_side": "LONG",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "opportunity_id": f"{stage}-union",
                "route": "UNION_BASE",
                "side": "LONG",
                "decision_time": base,
                "entry_time": base + pd.Timedelta(minutes=15),
                "outcome_available_time": base + pd.Timedelta(minutes=29),
                "gross_return": 0.011,
                "net_return": 0.010,
            },
            {
                **common,
                "opportunity_id": f"{stage}-candidate",
                "route": "COVERAGE_CANDIDATE",
                "side": "LONG",
                "decision_time": base + pd.Timedelta(hours=1),
                "entry_time": base + pd.Timedelta(hours=1, minutes=15),
                "outcome_available_time": base + pd.Timedelta(hours=1, minutes=29),
                "gross_return": 0.003,
                "net_return": 0.002,
            },
        ]
    )


def test_preflight_then_control_and_agent_share_one_hashed_input(
    tmp_path: Path, monkeypatch
) -> None:
    frames = {
        "development": _stage_frame("development", "2024-01-01"),
        "h1": _stage_frame("h1", "2025-01-01"),
        "forward": _stage_frame("forward", "2025-07-01"),
    }
    monkeypatch.setattr(
        runner,
        "EXPECTED_COUNTS",
        {
            stage: {"rows": 2, "union": 1, "candidates": 1, "blocks": 1}
            for stage in frames
        },
    )
    config = load_v4_config(runner.DEFAULT_CONFIG)
    preflight_caller = FakeCaller(0)
    preflight = runner.run_preflight(
        output_root=tmp_path,
        caller=preflight_caller,
        model_record={
            "model": config.model,
            "digest": config.required_model_digest,
            "capabilities": ["thinking"],
            "ollama_version": "test",
        },
        stage_frames=frames,
    )
    assert preflight["passed"] is True
    assert preflight["lockbox_2026_q2_used"] is False
    assert len(preflight_caller.calls) == 1

    control = runner.run_variant("static_lstm_all", output_root=tmp_path)
    assert control["transport_calls"] == 0
    assert control["controls_called_llm"] is False
    assert control["stage_summaries"]["development"]["selected_trades"] == 2

    agent_caller = FakeCaller(1)
    agent = runner.run_variant(
        "reflection_real_memory", output_root=tmp_path, caller=agent_caller
    )
    assert agent["protocol_hash"] == preflight["protocol_hash"]
    assert agent["transport_calls"] == 3
    assert agent["transport_failures"] == 0
    assert len(agent_caller.calls) == 3
    h1_payload = agent_caller.calls[1]["messages"][-1]["content"]
    assert '"coverage_status":{"additional_trades":0' in h1_payload
    assert '"block_id":"development-' in h1_payload
