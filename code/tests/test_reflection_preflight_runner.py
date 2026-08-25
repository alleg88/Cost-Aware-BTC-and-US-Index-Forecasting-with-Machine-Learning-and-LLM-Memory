import json

from reflection_agent.preflight import PreflightProbe, _probe_output_mode
from reflection_agent.transport import BackendResponse
from reflection_agent.config import load_config
from experiments.build_reflection_cache import CONFIG_PATH


class ProbeBackend:
    def __init__(self):
        self.formats = []

    def chat(self, **kwargs):
        self.formats.append(kwargs["response_format"])
        return BackendResponse(content=PreflightProbe(status="ok", value=12).model_dump_json())


class FencedProbeBackend(ProbeBackend):
    def chat(self, **kwargs):
        response = super().chat(**kwargs)
        return BackendResponse(content=f"```json\n{response.content}\n```")


def test_preflight_prefers_schema_mode():
    backend = ProbeBackend()
    assert _probe_output_mode(backend, load_config(CONFIG_PATH)) == "schema"
    assert isinstance(backend.formats[0], dict)


def test_preflight_accepts_glm_cloud_single_json_fence():
    backend = FencedProbeBackend()
    assert _probe_output_mode(backend, load_config(CONFIG_PATH)) == "schema"


def test_preflight_checks_the_compiled_pydantic_extension():
    from importlib.machinery import EXTENSION_SUFFIXES
    from pydantic_core import _pydantic_core

    extension_path = str(_pydantic_core.__file__).lower()
    assert any(extension_path.endswith(suffix.lower()) for suffix in EXTENSION_SUFFIXES)


def test_store_window_resume_is_idempotent(tmp_path):
    from datetime import UTC, datetime
    from reflection_agent.store import AgentStore

    store = AgentStore(tmp_path / "state.sqlite")
    cutoff = datetime(2025, 7, 13, 23, 59, tzinfo=UTC)
    store.save_window(
        window_id="2025-W28", protocol_hash="p", cutoff_utc=cutoff,
        status="completed", payload={"candidate_ids": ["c1"]},
    )
    store.save_window(
        window_id="2025-W28", protocol_hash="p", cutoff_utc=cutoff,
        status="completed", payload={"candidate_ids": ["c1"]},
    )
    assert store.load_window("2025-W28", protocol_hash="p")["payload"] == {"candidate_ids": ["c1"]}
    assert store.llm_call_count() == 0


def test_window_state_is_scoped_to_protocol_hash(tmp_path):
    from datetime import UTC, datetime
    from reflection_agent.store import AgentStore

    store = AgentStore(tmp_path / "state.sqlite")
    cutoff = datetime(2025, 7, 13, 23, 59, tzinfo=UTC)
    store.save_window(
        window_id="2025-W28", protocol_hash="p1", cutoff_utc=cutoff,
        status="completed", payload={"candidate_ids": []},
    )
    assert store.load_window("2025-W28", protocol_hash="p1") is not None
    assert store.load_window("2025-W28", protocol_hash="p2") is None


def test_first_weekly_observation_is_completed_and_before_lockbox():
    from experiments.run_reflection_agent import build_weekly_observation
    from experiments.build_reflection_cache import DEFAULT_OUTPUT

    report = build_weekly_observation(config=load_config(CONFIG_PATH), cache_root=DEFAULT_OUTPUT)
    assert report.window_id == "2025-W28"
    assert report.cutoff_utc.isoformat().startswith("2025-07-13")
    assert max(item.available_at_utc for item in report.news.top_items) <= report.cutoff_utc
    assert report.active_policy.active_edit_ids == ["consensus-agreement"]
    assert sum(model.ensemble_weight for model in report.models) == 1.0
    assert all(model.ensemble_enabled for model in report.models)


def test_pipeline_smoke_window_has_strictly_prior_historical_screen():
    from experiments.run_reflection_agent import build_weekly_observation
    from experiments.build_reflection_cache import DEFAULT_OUTPUT
    import pandas as pd

    report = build_weekly_observation(
        config=load_config(CONFIG_PATH), cache_root=DEFAULT_OUTPUT, window_offset=12
    )
    source_start = report.cutoff_utc + pd.Timedelta(microseconds=1) - pd.Timedelta(weeks=1)
    assert source_start < report.cutoff_utc
    assert source_start >= pd.Timestamp("2025-09-01", tz="UTC")
    assert report.cutoff_utc < load_config(CONFIG_PATH).sealed_start_utc


def test_news_ablation_modes_keep_aggregate_causality_without_raw_text():
    from experiments.run_reflection_agent import build_weekly_observation
    from experiments.build_reflection_cache import DEFAULT_OUTPUT

    config = load_config(CONFIG_PATH)
    bounded = build_weekly_observation(config=config, cache_root=DEFAULT_OUTPUT, news_mode="bounded_text")
    aggregate = build_weekly_observation(config=config, cache_root=DEFAULT_OUTPUT, news_mode="aggregate")
    none = build_weekly_observation(config=config, cache_root=DEFAULT_OUTPUT, news_mode="none")
    assert bounded.news.top_items
    assert aggregate.news.top_items == []
    assert aggregate.news.aggregate_features == bounded.news.aggregate_features
    assert none.news.aggregate_features["event_count"] == 0.0


def test_weekly_observation_reports_conditional_model_active_fraction():
    from experiments.run_reflection_agent import build_weekly_observation
    from experiments.build_reflection_cache import DEFAULT_OUTPUT
    from reflection_agent.contracts import ConditionPredicate, ConditionTree, PolicyEdit, PolicyRule

    rule = PolicyRule(
        rule_id="monday-lstm",
        conditions=ConditionTree(all=[
            ConditionPredicate(field="day_of_week", operator="eq", value="Mon")
        ]),
        edits=[PolicyEdit(edit_id="select-monday-lstm", action="select_frozen_expert", target="lstm")],
    )
    report = build_weekly_observation(
        config=load_config(CONFIG_PATH), cache_root=DEFAULT_OUTPUT,
        active_policy_id="monday-lstm", active_edit_ids=["select-monday-lstm"],
        active_rules=[rule],
    )
    by_id = {model.model_id: model for model in report.models}
    assert by_id["lstm"].ensemble_active_fraction == 1.0
    assert 0.0 < by_id["gru"].ensemble_active_fraction < 1.0
    assert by_id["gru"].active_weight_edit_ids == ["select-monday-lstm"]
