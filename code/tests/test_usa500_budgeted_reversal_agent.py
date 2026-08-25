from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from reflection_agent.index_v2.config import load_budgeted_reversal_config
from reflection_agent.index_v2.contracts import (
    ReversalScoreBatch,
    validate_score_batch,
)
from reflection_agent.index_v2.prompts import prompt_hashes, score_messages


CODE_ROOT = Path(__file__).parents[1]
CONFIG = CODE_ROOT / "configs" / "usa500_budgeted_reversal_agent_v1.yaml"
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
VARIANTS = (
    "frozen_parent",
    "budgeted_real_memory",
    "budgeted_no_memory",
    "budgeted_shuffled_memory",
    "uncertainty_control",
    "seeded_hash_control",
)


def _score_payload(decision_count: int) -> dict:
    return {
        "schema_version": "1.0",
        "decisions": [
            {
                "opportunity_index": index,
                "reversal_score": index,
                "evidence_indices": [0],
                "memory_indices": [],
            }
            for index in range(decision_count)
        ],
    }


def _memory_card() -> dict:
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


def _prompt_payload() -> dict:
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
                    {
                        "model_index": index,
                        "short": 0.2,
                        "flat": 0.1,
                        "long": 0.7,
                    }
                    for index in range(9)
                ],
                "agreement": 1.0,
                "probability_dispersion": 0.0,
                "uncertainty_prior": 420,
            }
        ],
        "model_evidence": [],
        "memory_cards": [_memory_card()],
    }


def test_config_freezes_parent_runtime_rates_and_q2_boundary():
    config = load_budgeted_reversal_config(CONFIG)

    assert config.protocol_version == "usa500-budgeted-reversal-agent-v1.2"
    assert config.stream_name == "usa500"
    assert config.source_candidate_id == "selected_base__directional_majority"
    assert config.model_names == MODEL_NAMES
    assert config.registered_variants == VARIANTS
    assert config.target_change_rates == (0.15, 0.20, 0.25)
    assert config.minimum_forward_change_rate == 0.15
    assert config.maximum_forward_change_rate == 0.30
    assert config.model == "deepseek-v4-flash:0731-cloud"
    assert config.required_model_digest == EXPECTED_DIGEST
    assert config.score_batch_size == 10
    assert config.forward_end_utc == config.q2_start_utc
    assert config.ollama_stream is False
    assert config.think is False
    assert config.seed == 42


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_name", "usatech"),
        ("protocol_version", "usa500-budgeted-reversal-agent-v1.1"),
        ("source_candidate_id", "another_parent"),
        ("model", "deepseek-v4-flash:cloud"),
        ("required_model_digest", "0" * 64),
        ("target_change_rates", [0.10, 0.20, 0.30]),
        ("score_batch_size", 11),
        ("think", "low"),
        ("forward_end_utc", "2026-04-02T00:00:00Z"),
        ("q2_start_utc", "2026-04-02T00:00:00Z"),
    ],
)
def test_config_rejects_identity_policy_or_boundary_drift(
    tmp_path: Path, field: str, value
):
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "drift.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises((ValidationError, ValueError)):
        load_budgeted_reversal_config(path)


def test_score_batch_accepts_one_to_ten_unique_integer_scores():
    value = ReversalScoreBatch.model_validate(_score_payload(10))

    assert validate_score_batch(
        value,
        allowed_opportunity_indices=list(range(10)),
        allowed_evidence_indices=[0],
        allowed_memory_indices=[],
        uncertainty_priors=list(range(10)),
    ) == tuple(range(10))


@pytest.mark.parametrize("invalid_score", [-1, 1001, 0.5, 1.0, "10"])
def test_score_batch_rejects_out_of_range_or_non_integer_scores(invalid_score):
    payload = _score_payload(1)
    payload["decisions"][0]["reversal_score"] = invalid_score

    with pytest.raises(ValidationError):
        ReversalScoreBatch.model_validate(payload)


def test_score_batch_rejects_oversize_duplicate_extra_missing_or_bad_reference():
    with pytest.raises(ValidationError):
        ReversalScoreBatch.model_validate(_score_payload(11))

    duplicate = _score_payload(2)
    duplicate["decisions"][1]["opportunity_index"] = 0
    with pytest.raises(ValidationError):
        ReversalScoreBatch.model_validate(duplicate)

    extra = _score_payload(1)
    extra["unexpected"] = True
    with pytest.raises(ValidationError):
        ReversalScoreBatch.model_validate(extra)

    missing = ReversalScoreBatch.model_validate(_score_payload(2))
    with pytest.raises(ValueError, match="exact opportunity"):
        validate_score_batch(
            missing,
            allowed_opportunity_indices=[0],
            allowed_evidence_indices=[0],
            allowed_memory_indices=[],
            uncertainty_priors=[0],
        )

    invalid_reference = ReversalScoreBatch.model_validate(_score_payload(1))
    with pytest.raises(ValueError, match="reference"):
        validate_score_batch(
            invalid_reference,
            allowed_opportunity_indices=[0],
            allowed_evidence_indices=[],
            allowed_memory_indices=[],
            uncertainty_priors=[0],
        )


def test_score_batch_enforces_the_250_point_residual_boundary():
    boundary = ReversalScoreBatch.model_validate(
        {
            "schema_version": "1.0",
            "decisions": [
                {
                    "opportunity_index": 0,
                    "reversal_score": 750,
                    "evidence_indices": [],
                    "memory_indices": [],
                }
            ],
        }
    )
    assert validate_score_batch(
        boundary,
        allowed_opportunity_indices=[0],
        allowed_evidence_indices=[],
        allowed_memory_indices=[],
        uncertainty_priors=[500],
    ) == (750,)

    outside = boundary.model_copy(
        update={
            "decisions": [
                boundary.decisions[0].model_copy(update={"reversal_score": 751})
            ]
        }
    )
    with pytest.raises(ValueError, match="residual"):
        validate_score_batch(
            outside,
            allowed_opportunity_indices=[0],
            allowed_evidence_indices=[],
            allowed_memory_indices=[],
            uncertainty_priors=[500],
        )


def test_prompt_is_anonymous_causal_pointwise_and_bounded():
    messages = score_messages(_prompt_payload())
    rendered = json.dumps(messages).lower()

    assert "counterfactual_reversal_scorer" in rendered
    assert "ranking score" in rendered
    assert "not a probability" in rendered
    assert "host" in rendered and "keep" in rendered and "reverse" in rendered
    assert "independently" in rendered
    assert "same-week outcomes" in rendered
    assert "uncertainty_prior" in rendered
    assert "causal integer baseline" in rendered
    assert "at most 250" in rendered
    assert "return json immediately" in rendered
    assert "return no citation or reference fields" in rendered
    assert "cite only supplied" not in rendered
    assert "exactly one json object" in rendered
    assert "abstention" in rendered
    assert "select_long_or_short" not in rendered
    assert "usa500" not in rendered
    assert "usatech" not in rendered
    assert "2025" not in rendered
    assert "price" not in rendered
    hashes = prompt_hashes()
    assert set(hashes) == {"score_system", "score_task"}
    assert all(len(value) == 64 for value in hashes.values())


@pytest.mark.parametrize(
    "forbidden_key",
    ["asset", "timestamp", "entry_time", "price", "path", "outcome", "future"],
)
def test_prompt_rejects_identity_time_price_path_or_outcome_fields(forbidden_key):
    payload = deepcopy(_prompt_payload())
    payload[forbidden_key] = "hidden"

    with pytest.raises(ValueError, match="forbidden key"):
        score_messages(payload)


def test_prompt_requires_one_to_ten_rows_and_nine_model_vectors():
    empty = _prompt_payload()
    empty["opportunities"] = []
    with pytest.raises(ValueError):
        score_messages(empty)

    too_many = _prompt_payload()
    too_many["opportunities"] = too_many["opportunities"] * 11
    with pytest.raises(ValueError):
        score_messages(too_many)

    missing_model = _prompt_payload()
    missing_model["opportunities"][0]["model_probabilities"].pop()
    with pytest.raises(ValueError, match="nine"):
        score_messages(missing_model)

    missing_prior = _prompt_payload()
    del missing_prior["opportunities"][0]["uncertainty_prior"]
    with pytest.raises(ValueError, match="uncertainty_prior"):
        score_messages(missing_prior)

    invalid_prior = _prompt_payload()
    invalid_prior["opportunities"][0]["uncertainty_prior"] = 1001
    with pytest.raises(ValueError, match="uncertainty_prior"):
        score_messages(invalid_prior)
