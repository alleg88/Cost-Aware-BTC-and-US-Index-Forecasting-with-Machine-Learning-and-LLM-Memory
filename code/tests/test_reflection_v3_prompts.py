from __future__ import annotations

import json

from reflection_agent.v3.prompts import (
    SYSTEM_PROMPT_V3,
    prompt_hashes,
    proposal_messages,
    reflection_messages,
)


def observation() -> dict[str, object]:
    return {
        "protocol_summary": {
            "scope": "continuous",
            "allowed_fields": ["side", "confidence_tier", "vol_regime"],
        },
        "source_episode_id": "episode-opaque-1",
        "episode_metrics": {"candidate_count": 20, "long_count": 10, "short_count": 10},
        "evidence_cards": [
            {
                "evidence_id": "evidence-opaque-1",
                "route": "COVERAGE_CANDIDATE",
                "side": "SHORT",
                "confidence_tier": "HIGH_EXTRA",
                "count": 8,
                "net_return": 0.001,
            }
        ],
        "active_rules": [],
        "eligible_memories": [],
        "rejected_edit_buffer": [],
    }


def test_common_prompt_is_anonymous_and_closes_authority() -> None:
    assert "anonymous-market M15 experiment" in SYSTEM_PROMPT_V3
    assert "Candidate direction is supplied by the host" in SYSTEM_PROMPT_V3
    assert "Return exactly one JSON object" in SYSTEM_PROMPT_V3
    for forbidden in ("BTC", "Bitcoin", "Binance", "2025-"):
        assert forbidden not in SYSTEM_PROMPT_V3


def test_proposal_prompt_inserts_only_canonical_compact_json() -> None:
    payload = observation()
    messages = proposal_messages(payload)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "TASK: SELECT_HOST_POLICY_CHOICE" in messages[1]["content"]
    assert "choice_index" in messages[1]["content"]
    assert "evidence_indices" in messages[1]["content"]
    assert "__CANONICAL_COMPACT_INPUT_JSON__" not in messages[1]["content"]
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    assert messages[1]["content"].endswith("INPUT_JSON=" + encoded)


def test_reflection_prompt_cannot_rewrite_host_verdict() -> None:
    candidate = {"candidate_id": "candidate-opaque-1", "hypothesis": "Future-test me."}
    evaluation = {
        "decision": "REJECT",
        "evidence_cards": [{"evidence_index": 0, "side": "SHORT"}],
        "gate_results": {"net_noninferior": False},
    }
    messages = reflection_messages(candidate, evaluation)
    assert "memory_action_index" in messages[1]["content"]
    assert "cannot change EVALUATION_JSON.decision" in messages[1]["content"]
    assert "CANDIDATE_JSON=" in messages[1]["content"]
    assert "EVALUATION_JSON=" in messages[1]["content"]


def test_prompt_hashes_are_stable_sha256_values() -> None:
    first = prompt_hashes()
    assert first == prompt_hashes()
    assert set(first) == {"system", "proposal", "reflection"}
    assert all(len(value) == 64 for value in first.values())
    assert len(set(first.values())) == 3
