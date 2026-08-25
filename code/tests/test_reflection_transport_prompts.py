from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace

import pandas as pd
import pytest

from reflection_agent.contracts import CandidateBatch, MarketContext, ProbabilityVector
from reflection_agent.index_v1.transport import OllamaCloudBackend as IndexOllamaCloudBackend
from reflection_agent.news import select_balanced_events
from reflection_agent.observation import build_observation
from reflection_agent.prompts import SYSTEM_PROMPT, actor_messages
from reflection_agent.store import AgentStore
from reflection_agent.transport import (
    BackendResponse,
    OllamaClientTransport,
    OllamaHttpTransport,
    StructuredCaller,
    normalize_json_object,
)


@dataclass
class FakeBackend:
    outputs: list[object]
    calls: int = 0

    def chat(self, **kwargs):
        item = self.outputs[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return BackendResponse(content=item, metadata={"fake": True})


VALID = """{
  "schema_version":"1.0",
  "diagnosis":"No supported bounded change is justified.",
  "candidates":[]
}"""


def _observation():
    models = (
        "logreg", "decision_tree", "random_forest", "svm_linear", "xgboost_balanced",
        "catboost_balanced", "mlp", "lstm", "gru",
    )
    cutoff = datetime(2025, 7, 13, 23, 59, tzinfo=UTC)
    empty = pd.DataFrame(columns=[
        "event_id", "available_at_utc", "source_family", "publisher_category", "summary", "impact", "sentiment"
    ])
    return build_observation(
        window_id="2025-W28",
        cutoff_utc=cutoff,
        active_policy_id="p0",
        market=MarketContext(vol_regime="normal", trend_regime="flat", realized_volatility=0.1, recent_return=0.0),
        probabilities={model: ProbabilityVector(short=0.2, flat=0.6, long=0.2) for model in models},
        news=select_balanced_events(empty, cutoff_utc=cutoff),
    )


def test_prompts_define_untrusted_data_and_forbid_trades_tools_and_price_prediction():
    prompt = SYSTEM_PROMPT.lower()
    assert "untrusted" in prompt
    assert "do not predict prices" in prompt
    assert "choose trades" in prompt
    assert "call tools" in prompt
    assert "exactly one json object" in prompt
    messages = actor_messages(_observation())
    assert "zero to six" in messages[1]["content"].lower()
    assert "exact_candidate_shape" in messages[1]["content"].lower()
    assert '"expected_effect":{"net_return"' in messages[1]["content"]
    assert "no dotted field names" in messages[1]["content"]
    assert '"ensemble_weight"' in messages[1]["content"]
    assert '"ensemble_enabled"' in messages[1]["content"]
    assert '"ensemble_active_fraction"' in messages[1]["content"]


def test_ollama_client_transports_honour_cloud_environment(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.com")
    monkeypatch.setenv("OLLAMA_API_KEY", "test-only-key")

    clients = [OllamaClientTransport().client, IndexOllamaCloudBackend().client]

    for client in clients:
        assert str(client._client.base_url).rstrip("/") == "https://ollama.com"
        assert client._client.headers["authorization"] == "Bearer test-only-key"


def test_ollama_transports_send_no_tool_definitions(monkeypatch):
    client_calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def chat(self, **kwargs):
            client_calls.append(kwargs)
            return SimpleNamespace(
                message=SimpleNamespace(content=VALID), model="glm-5.2:cloud",
                created_at=None, done_reason="stop", total_duration=1,
                prompt_eval_count=1, eval_count=1,
            )

    http_calls = []

    class FakeHttpResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": VALID}}

    def fake_post(url, *, json, timeout):
        http_calls.append({"url": url, "json": json, "timeout": timeout})
        return FakeHttpResponse()

    monkeypatch.setattr("reflection_agent.transport.ollama.Client", FakeClient)
    monkeypatch.setattr("reflection_agent.transport.requests.post", fake_post)
    kwargs = {
        "model": "glm-5.2:cloud",
        "messages": [{"role": "user", "content": "probe"}],
        "response_format": "json",
        "think": "high",
        "temperature": 0.0,
    }
    OllamaClientTransport().chat(**kwargs)
    OllamaHttpTransport().chat(**kwargs)
    assert "tools" not in client_calls[0]
    assert "tools" not in http_calls[0]["json"]


def test_structured_call_validates_and_reuses_success_cache(tmp_path):
    backend = FakeBackend([VALID])
    store = AgentStore(tmp_path / "state.sqlite")
    caller = StructuredCaller(
        model="glm-5.2:cloud", output_mode="schema", primary=backend, store=store,
        protocol_hash="p", retry_delays_seconds=(), sleep=lambda _: None,
    )
    first = caller.call(role="actor", messages=actor_messages(_observation()), response_model=CandidateBatch, temperature=0.2)
    second = caller.call(role="actor", messages=actor_messages(_observation()), response_model=CandidateBatch, temperature=0.2)
    assert first.status == "success"
    assert second.status == "cached"
    assert backend.calls == 1


def test_structured_call_cache_is_protocol_scoped(tmp_path):
    backend = FakeBackend([VALID, VALID])
    store = AgentStore(tmp_path / "state.sqlite")
    for protocol_hash in ("p1", "p2"):
        caller = StructuredCaller(
            model="glm-5.2:cloud", output_mode="schema", primary=backend, store=store,
            protocol_hash=protocol_hash, retry_delays_seconds=(), sleep=lambda _: None,
        )
        assert caller.call(
            role="actor", messages=actor_messages(_observation()),
            response_model=CandidateBatch, temperature=0.2,
        ).status == "success"
    assert backend.calls == 2


def test_structured_call_cache_is_ablation_run_scoped(tmp_path):
    backend = FakeBackend([VALID, VALID])
    store = AgentStore(tmp_path / "state.sqlite")
    for run_scope in ("news-none", "news-bounded"):
        caller = StructuredCaller(
            model="glm-5.2:cloud", output_mode="schema", primary=backend, store=store,
            protocol_hash="same-manifest", run_scope=run_scope,
            retry_delays_seconds=(), sleep=lambda _: None,
        )
        caller.call(
            role="actor", messages=actor_messages(_observation()),
            response_model=CandidateBatch, temperature=0.2,
        )
    assert backend.calls == 2


def test_json_normalizer_accepts_only_object_or_single_fence():
    assert normalize_json_object('{"status":"ok"}') == '{"status":"ok"}'
    assert normalize_json_object('```json\n{"status":"ok"}\n```') == '{"status":"ok"}'
    with pytest.raises(ValueError):
        normalize_json_object('Result: {"status":"ok"}')
    with pytest.raises(ValueError):
        normalize_json_object('{"a":1}{"b":2}')


def test_invalid_json_gets_two_repairs_then_noop():
    backend = FakeBackend(["not json", "{}", "still invalid"])
    caller = StructuredCaller(
        model="glm-5.2:cloud", output_mode="json", primary=backend,
        retry_delays_seconds=(), sleep=lambda _: None,
    )
    result = caller.call(role="actor", messages=actor_messages(_observation()), response_model=CandidateBatch, temperature=0.2)
    assert result.status == "noop"
    assert result.value is None
    assert backend.calls == 3
    assert len(result.errors) == 3


def test_primary_failure_uses_http_compatible_fallback():
    primary = FakeBackend([RuntimeError("client schema error")])
    fallback = FakeBackend([VALID])
    caller = StructuredCaller(
        model="glm-5.2:cloud", output_mode="schema", primary=primary, fallback=fallback,
        retry_delays_seconds=(), sleep=lambda _: None,
    )
    result = caller.call(role="actor", messages=actor_messages(_observation()), response_model=CandidateBatch, temperature=0.2)
    assert result.status == "success"
    assert result.backend == "FakeBackend"
    assert primary.calls == fallback.calls == 1
