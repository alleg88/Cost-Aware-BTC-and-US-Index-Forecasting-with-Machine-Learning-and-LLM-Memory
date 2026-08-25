from __future__ import annotations

import pytest
from pydantic import ValidationError

from reflection_agent.v4.contracts import RouterChoice
from reflection_agent.v4.prompts import SYSTEM_PROMPT_V4, prompt_hashes, router_messages


def _payload() -> dict[str, object]:
    return {
        "schema_version": "4.0",
        "coverage_status": {
            "trade_ratio": 1.10,
            "long_growth": 0.05,
            "short_growth": 0.02,
        },
        "policy_menu": [
            {"choice_index": 0, "policy_id": "UNION_ONLY"},
            {"choice_index": 1, "policy_id": "LSTM_HIGH"},
        ],
        "policy_statistics": [],
        "memory_cards": [],
    }


def test_router_choice_is_numeric_and_closed() -> None:
    choice = RouterChoice.model_validate(
        {
            "schema_version": "4.0",
            "choice_index": 1,
            "evidence_indices": [],
            "memory_indices": [],
        }
    )
    assert choice.choice_index == 1
    with pytest.raises(ValidationError):
        RouterChoice.model_validate(
            {
                "schema_version": "4.0",
                "choice_index": 9,
                "evidence_indices": [],
                "memory_indices": [],
            }
        )


def test_prompt_exposes_only_host_policy_indices() -> None:
    messages = router_messages(_payload())
    rendered = "\n".join(item["content"] for item in messages)
    assert "choice_index" in rendered
    assert "never place trades" in SYSTEM_PROMPT_V4
    assert "price" not in str(_payload()).lower()
    assert set(prompt_hashes()) == {"system", "router"}
