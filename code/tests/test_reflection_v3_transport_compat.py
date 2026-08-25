from __future__ import annotations

import json
from pathlib import Path

from reflection_agent.v2.transport import DeepSeekSchemaCaller
from reflection_agent.v3.config import load_v3_config
from reflection_agent.v3.contracts import ProposalChoiceOutput, ReflectionChoiceOutput


CONFIG = Path(__file__).parents[1] / "configs" / "reflection_agent_v3.yaml"


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


def _response(payload: dict[str, object]) -> dict[str, object]:
    return {
        "model": "deepseek-v4-flash:cloud",
        "done_reason": "stop",
        "message": {
            "content": json.dumps(payload),
            "thinking": "discard this hidden chain",
        },
    }


def _proposal() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "choice_index": 1,
        "evidence_indices": [0],
        "memory_indices": [],
    }


def _reflection() -> dict[str, object]:
    return {
        "schema_version": "3.0",
        "evidence_indices": [0],
        "memory_action_index": 1,
    }


def test_v2_generic_transport_accepts_exact_v3_contract(tmp_path) -> None:
    backend = FakeBackend([_response(_proposal()), _response(_reflection())])
    caller = DeepSeekSchemaCaller(
        load_v3_config(CONFIG),
        backend=backend,
        call_log_path=tmp_path / "calls.jsonl",
    )
    proposal = caller.call(
        role="proposal",
        messages=[{"role": "system", "content": "system"}],
        response_model=ProposalChoiceOutput,
        allowed_ids={
            "choice_indices": [0, 1],
            "evidence_indices": [0],
            "memory_indices": [],
        },
    )
    reflection = caller.call(
        role="reflection",
        messages=[{"role": "system", "content": "system"}],
        response_model=ReflectionChoiceOutput,
        allowed_ids={
            "evidence_indices": [0],
            "memory_action_indices": [0, 1, 2],
        },
    )
    assert proposal.status == reflection.status == "success"
    assert isinstance(proposal.value, ProposalChoiceOutput)
    assert isinstance(reflection.value, ReflectionChoiceOutput)
    for request, model in zip(backend.calls, (ProposalChoiceOutput, ReflectionChoiceOutput)):
        assert request["model"] == "deepseek-v4-flash:cloud"
        assert request["stream"] is False
        assert request["think"] == "low"
        assert request["format"] == model.model_json_schema()
        assert request["options"] == {"temperature": 0.0, "num_predict": 4096}
        assert "tools" not in request
    logged = (tmp_path / "calls.jsonl").read_text(encoding="utf-8")
    assert "thinking" not in logged.lower()
    assert proposal.request_hash in logged
    assert reflection.response_hash in logged


def test_v3_schema_gets_exactly_one_repair() -> None:
    backend = FakeBackend(
        [
            {"message": {"content": '{"bad":true}'}},
            _response(_proposal()),
        ]
    )
    result = DeepSeekSchemaCaller(load_v3_config(CONFIG), backend=backend).call(
        role="proposal",
        messages=[{"role": "system", "content": "system"}],
        response_model=ProposalChoiceOutput,
        allowed_ids={
            "choice_indices": [0, 1],
            "evidence_indices": [0],
            "memory_indices": [],
        },
    )
    assert result.status == "repaired"
    assert result.attempts == 2
    assert len(backend.calls) == 2
    repair = backend.calls[-1]["messages"][-1]["content"]
    assert "VALIDATION_ERRORS=" in repair
    assert "ALLOWED_IDS=" in repair
    assert "PRIOR_JSON=" not in repair
