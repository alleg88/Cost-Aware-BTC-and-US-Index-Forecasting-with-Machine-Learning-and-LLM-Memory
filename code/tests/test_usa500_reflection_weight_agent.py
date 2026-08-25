from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from reflection_agent.index_v1.config import load_index_agent_config
from reflection_agent.index_v1.contracts import (
    DirectBatchDecision,
    WeeklyWeightDecision,
    validate_direct_batch,
    validate_weekly_weights,
)
from reflection_agent.index_v1.prompts import (
    direct_messages,
    prompt_hashes,
    weekly_weight_messages,
)
from reflection_agent.index_v1.transport import CachedIndexSchemaCaller, IndexSchemaCaller


CODE_ROOT = Path(__file__).parents[1]
CONFIG = CODE_ROOT / "configs" / "usa500_reflection_weight_agent_v1.yaml"
EXPECTED_DIGEST = "d3f1c87447216481a8001f48c517a51e13bfb141853a8df5e52f81bf765dabc3"
MODEL_NAMES = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)


def _direct_payload(decision_count: int) -> dict:
    return {
        "schema_version": "1.0",
        "decisions": [
            {
                "opportunity_index": index,
                "side": "LONG" if index % 2 == 0 else "SHORT",
                "evidence_indices": [0],
                "memory_indices": [],
            }
            for index in range(decision_count)
        ],
    }


def _weight_payload() -> dict:
    return {
        "schema_version": "1.0",
        "weights": [0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
        "evidence_indices": [0],
        "memory_indices": [],
    }


def test_config_enforces_frozen_usa500_runtime_and_boundaries():
    config = load_index_agent_config(CONFIG)

    assert config.stream_name == "usa500"
    assert config.source_candidate_id == "selected_base__directional_majority"
    assert config.model == "deepseek-v4-flash:0731-cloud"
    assert config.required_model_digest == EXPECTED_DIGEST
    assert config.model_names == MODEL_NAMES
    assert config.direct_batch_size == 10
    assert config.forward_end_utc == config.q2_start_utc
    assert config.ollama_stream is False
    assert config.seed == 42


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_name", "usatech"),
        ("model", "deepseek-v4-flash:cloud"),
        ("required_model_digest", "0" * 64),
        ("forward_end_utc", "2026-04-02T00:00:00Z"),
        ("q2_start_utc", "2026-04-02T00:00:00Z"),
        ("direct_batch_size", 11),
    ],
)
def test_config_rejects_identity_or_boundary_drift(tmp_path: Path, field: str, value):
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises((ValidationError, ValueError)):
        load_index_agent_config(path)


def test_direct_batch_accepts_one_to_ten_unique_binary_decisions():
    value = DirectBatchDecision.model_validate(_direct_payload(10))

    assert len(value.decisions) == 10
    assert {item.side for item in value.decisions} == {"LONG", "SHORT"}
    assert validate_direct_batch(
        value,
        allowed_opportunity_indices=list(range(10)),
        allowed_evidence_indices=[0],
        allowed_memory_indices=[],
    ) == tuple(1 if index % 2 == 0 else -1 for index in range(10))


def test_direct_batch_rejects_oversize_duplicate_extra_or_invalid_references():
    with pytest.raises(ValidationError):
        DirectBatchDecision.model_validate(_direct_payload(11))

    duplicate = _direct_payload(2)
    duplicate["decisions"][1]["opportunity_index"] = 0
    with pytest.raises(ValidationError):
        DirectBatchDecision.model_validate(duplicate)

    extra = _direct_payload(1)
    extra["unexpected"] = True
    with pytest.raises(ValidationError):
        DirectBatchDecision.model_validate(extra)

    invalid_reference = DirectBatchDecision.model_validate(_direct_payload(1))
    with pytest.raises(ValueError, match="reference"):
        validate_direct_batch(
            invalid_reference,
            allowed_opportunity_indices=[0],
            allowed_evidence_indices=[],
            allowed_memory_indices=[],
        )


def test_weekly_weight_contract_accepts_exact_grid_vector():
    value = WeeklyWeightDecision.model_validate(_weight_payload())

    assert validate_weekly_weights(
        value,
        allowed_evidence_indices=[0],
        allowed_memory_indices=[],
    ) == pytest.approx(tuple(_weight_payload()["weights"]), abs=1e-12)


@pytest.mark.parametrize(
    "mutation",
    [
        {"weights": [1 / 9] * 9},
        {"weights": [0.81, 0.05, 0.05, 0.04, 0.01, 0.01, 0.01, 0.01, 0.01]},
        {"weights": [0.80, 0.20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        {"weights": [0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11]},
        {"weights": [float("nan"), 0.20, 0.20, 0.20, 0.20, 0.10, 0.05, 0.03, 0.02]},
        {"memory_indices": [1]},
    ],
)
def test_weekly_weight_contract_rejects_invalid_vector_or_reference(mutation: dict):
    payload = deepcopy(_weight_payload())
    payload.update(mutation)
    try:
        value = WeeklyWeightDecision.model_validate(payload)
    except ValidationError:
        return

    with pytest.raises(ValueError):
        validate_weekly_weights(
            value,
            allowed_evidence_indices=[0],
            allowed_memory_indices=[],
        )


def _safe_memory_card() -> dict:
    return {
        "memory_index": 0,
        "opportunities": 12,
        "model_statistics": [
            {
                "model_index": index,
                "sample_count": 12,
                "directional_accuracy": 0.50,
                "net_return": 0.001 * index,
                "brier_score": 0.40,
                "mean_confidence": 0.60,
            }
            for index in range(9)
        ],
        "controls": {"original": {"net_return": 0.001}},
        "market_state": {
            "vix_regime": 0.2,
            "trailing_volatility": 0.01,
            "trailing_trend": -0.005,
        },
    }


def _direct_prompt_payload() -> dict:
    return {
        "schema_version": "1.0",
        "market_state": {
            "vix_regime": 0.1,
            "trailing_volatility": 0.02,
            "trailing_trend": 0.01,
        },
        "opportunities": [
            {
                "opportunity_index": 0,
                "original_side": "LONG",
                "model_probabilities": [
                    {"model_index": index, "short": 0.2, "flat": 0.1, "long": 0.7}
                    for index in range(9)
                ],
                "agreement": 1.0,
                "probability_dispersion": 0.0,
            }
        ],
        "model_evidence": [],
        "memory_cards": [_safe_memory_card()],
    }


def _weekly_prompt_payload() -> dict:
    return {
        "schema_version": "1.0",
        "market_state": {
            "vix_regime": 0.1,
            "trailing_volatility": 0.02,
            "trailing_trend": 0.01,
        },
        "model_evidence": [
            {
                "evidence_index": index,
                "model_index": index,
                "rolling_1": {"net_return": 0.0},
                "rolling_4": {"net_return": 0.0},
                "rolling_12": {"net_return": 0.0},
            }
            for index in range(9)
        ],
        "memory_cards": [_safe_memory_card()],
        "previous_weights": [0.12, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11, 0.11],
    }


def test_prompts_are_anonymous_causal_and_define_only_bounded_authority():
    direct = direct_messages(_direct_prompt_payload())
    weekly = weekly_weight_messages(_weekly_prompt_payload())
    rendered = json.dumps([direct, weekly]).lower()

    assert "authority boundary" in rendered
    assert "causality" in rendered
    assert "exactly one json object" in rendered
    assert "long" in rendered and "short" in rendered
    assert "0.01" in rendered and "0.80" in rendered
    assert "same-week outcomes" in rendered
    for forbidden in (
        "usa500",
        "usatech",
        "logreg",
        "xgboost",
        "2025-",
        "2026-",
        "entry_price",
        "exit_price",
        "file_path",
    ):
        assert forbidden not in rendered
    hashes = prompt_hashes()
    assert set(hashes) == {"direct_system", "direct_task", "weekly_system", "weekly_task"}
    assert all(len(value) == 64 for value in hashes.values())


def test_prompt_builder_rejects_identity_time_price_or_outcome_fields():
    for forbidden_key in ("asset", "timestamp", "entry_price", "future_outcome"):
        payload = _direct_prompt_payload()
        payload[forbidden_key] = "forbidden"
        with pytest.raises(ValueError, match="anonymous"):
            direct_messages(payload)


class _FakeBackend:
    def __init__(self, contents: list[str]):
        self.contents = list(contents)
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "model": kwargs["model"],
            "done_reason": "stop",
            "message": {"content": self.contents.pop(0)},
        }


def test_index_transport_binds_schema_seed_runtime_and_request_hash():
    backend = _FakeBackend([json.dumps(_direct_payload(1))])
    config = load_index_agent_config(CONFIG)
    caller = IndexSchemaCaller(config, backend=backend)

    result = caller.call(
        role="direct",
        messages=direct_messages(_direct_prompt_payload()),
        response_model=DirectBatchDecision,
        allowed_ids={"opportunity_indices": [0], "evidence_indices": [], "memory_indices": [0]},
    )

    assert result.status == "success"
    assert result.value is not None
    assert len(result.request_hash) == len(result.response_hash) == len(result.schema_hash) == 64
    request = backend.calls[0]
    assert request["model"] == "deepseek-v4-flash:0731-cloud"
    assert request["stream"] is False
    assert request["think"] == "low"
    assert request["options"] == {"temperature": 0.0, "num_predict": 4096, "seed": 42}
    assert request["format"] == DirectBatchDecision.model_json_schema()


def test_index_transport_repairs_once_and_exact_cache_prevents_second_call(tmp_path: Path):
    backend = _FakeBackend(["not-json", json.dumps(_direct_payload(1))])
    config = load_index_agent_config(CONFIG)
    inner = IndexSchemaCaller(config, backend=backend)
    caller = CachedIndexSchemaCaller(inner, tmp_path / "calls", protocol_hash="a" * 64)
    kwargs = {
        "role": "direct",
        "messages": direct_messages(_direct_prompt_payload()),
        "response_model": DirectBatchDecision,
        "allowed_ids": {
            "opportunity_indices": [0],
            "evidence_indices": [],
            "memory_indices": [0],
        },
    }

    first = caller.call(**kwargs)
    second = caller.call(**kwargs)

    assert first.status == second.status == "repaired"
    assert first.request_hash == second.request_hash
    assert len(backend.calls) == 2
