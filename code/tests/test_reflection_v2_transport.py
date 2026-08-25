from __future__ import annotations

import json
from pathlib import Path

from reflection_agent.v2.config import load_v2_config
from reflection_agent.v2.contracts import ProposalOutput
from reflection_agent.v2.transport import DeepSeekSchemaCaller, OllamaCloudBackend


CONFIG = Path(__file__).parents[1] / "configs" / "reflection_agent_v2.yaml"


def valid_proposal() -> dict[str, object]:
    return {
        "schema_version": "2.0",
        "source_episode_id": "episode-1",
        "decision": "ADD_ALLOW_RULE",
        "diagnosis_code": "REGIME_SPECIFIC_EDGE",
        "evidence_ids": ["evidence-1"],
        "memory_ids_used": [],
        "proposed_rule": {
            "action": "ALLOW_REENTRY",
            "predicates": [
                {"field": "side", "operator": "EQ", "value": "SHORT"},
                {"field": "vol_regime", "operator": "EQ", "value": "HIGH"},
            ],
        },
        "target_rule_id": None,
        "hypothesis": "High-volatility short re-entries retain a repeated net edge.",
        "falsifiers": ["Future incremental SHORT net return is non-positive."],
        "confidence": "MEDIUM",
    }


class FakeBackend:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_cloud_backend_accepts_explicit_host_and_environment_key(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_API_KEY", "test-only-key")

    backend = OllamaCloudBackend(host="https://ollama.com")

    assert str(backend.client._client.base_url).rstrip("/") == "https://ollama.com"
    assert backend.client._client.headers["authorization"] == "Bearer test-only-key"


def response(payload: dict[str, object], *, thinking: str = "secret") -> dict[str, object]:
    return {
        "model": "deepseek-v4-flash:cloud",
        "done_reason": "stop",
        "prompt_eval_count": 100,
        "eval_count": 50,
        "message": {"content": json.dumps(payload), "thinking": thinking},
    }


def test_exact_deepseek_schema_request_has_no_fallback(tmp_path) -> None:
    backend = FakeBackend([response(valid_proposal())])
    caller = DeepSeekSchemaCaller(
        load_v2_config(CONFIG), backend=backend, call_log_path=tmp_path / "calls.jsonl"
    )
    result = caller.call(
        role="proposal",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        response_model=ProposalOutput,
        allowed_ids={"evidence_ids": ["evidence-1"], "memory_ids": [], "rule_ids": []},
    )
    assert result.status == "success"
    assert result.value is not None
    assert len(backend.calls) == 1
    request = backend.calls[0]
    assert request["model"] == "deepseek-v4-flash:cloud"
    assert request["stream"] is False
    assert request["think"] == "high"
    assert request["format"] == ProposalOutput.model_json_schema()
    assert request["options"] == {"temperature": 0.0, "num_predict": 4096}
    logged = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert "thinking" not in json.dumps(logged).lower()
    assert logged["response_hash"] == result.response_hash


def test_one_schema_repair_uses_only_errors_and_registered_ids() -> None:
    backend = FakeBackend(
        [
            {"message": {"content": '{"bad":true}'}},
            response(valid_proposal()),
        ]
    )
    caller = DeepSeekSchemaCaller(load_v2_config(CONFIG), backend=backend)
    result = caller.call(
        role="proposal",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "original"}],
        response_model=ProposalOutput,
        allowed_ids={"evidence_ids": ["evidence-1"], "memory_ids": [], "rule_ids": []},
    )
    assert result.status == "repaired"
    assert result.attempts == 2
    repair = backend.calls[1]["messages"][-1]["content"]
    assert "VALIDATION_ERRORS=" in repair
    assert "ALLOWED_IDS=" in repair
    assert "PRIOR_JSON=" not in repair


def test_second_schema_failure_fails_closed_without_third_call() -> None:
    backend = FakeBackend(
        [{"message": {"content": "{}"}}, {"message": {"content": "{}"}}]
    )
    result = DeepSeekSchemaCaller(load_v2_config(CONFIG), backend=backend).call(
        role="proposal",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        response_model=ProposalOutput,
        allowed_ids={"evidence_ids": [], "memory_ids": [], "rule_ids": []},
    )
    assert result.status == "schema_failure"
    assert result.value is None
    assert result.attempts == 2
    assert len(backend.calls) == 2


def test_transport_error_fails_closed_without_retry_or_fallback() -> None:
    backend = FakeBackend([TimeoutError("cloud timeout")])
    result = DeepSeekSchemaCaller(load_v2_config(CONFIG), backend=backend).call(
        role="proposal",
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
        response_model=ProposalOutput,
        allowed_ids={"evidence_ids": [], "memory_ids": [], "rule_ids": []},
    )
    assert result.status == "transport_error"
    assert result.value is None
    assert len(backend.calls) == 1
    assert "TimeoutError" in result.errors[0]
