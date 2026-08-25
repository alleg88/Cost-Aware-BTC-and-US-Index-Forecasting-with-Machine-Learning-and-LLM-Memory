from __future__ import annotations

import json
from types import SimpleNamespace

from memory.proposer import MODEL, OllamaLessonProposer, lesson_schema, propose_lessons


def _payload():
    return {
        "lessons": [{
            "condition": "minutes_to_high_impact_event < 30",
            "action": "downweight",
            "target": "lstm",
            "factor": 0.5,
            "evidence": "LSTM underperformed near events in the weekly digest.",
            "confidence": 0.7,
        }]
    }


def test_propose_lessons_uses_notebook3_ollama_call_shape():
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps(_payload())}}

    lessons = propose_lessons(
        report={"overall": {}},
        active_lessons=[],
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm", "catboost"},
        chat_fn=fake_chat,
    )

    assert MODEL == "glm-5.2:cloud"
    assert calls[0]["model"] == "glm-5.2:cloud"
    # Cloud models get NO format= (they return empty strings when a schema is attached);
    # the output contract lives in the system prompt instead.
    assert "format" not in calls[0]
    assert calls[0]["options"]["temperature"] == 0
    assert calls[0]["think"] == "high"
    assert calls[0]["messages"][0]["role"] == "system"
    assert "Output contract" in calls[0]["messages"][0]["content"]
    assert lessons[0].metadata["reflection_model"] == "glm-5.2:cloud"
    assert lessons[0].metadata["reflection_prompt_version"] == "4"


def test_propose_lessons_can_restrict_action_space():
    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps(_payload())}}

    propose_lessons(
        report={"overall": {}},
        active_lessons=[],
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm", "catboost"},
        allowed_actions={"downweight", "upweight", "force_flat"},
        chat_fn=fake_chat,
    )

    user_payload = json.loads(calls[0]["messages"][1]["content"])
    assert user_payload["allowed_actions"] == ["downweight", "force_flat", "upweight"]
    assert "widen_deadzone" not in user_payload["allowed_actions"]



def test_default_chat_can_send_max_think_via_raw_http(monkeypatch):
    import memory.proposer as proposer_module

    calls = []

    def fake_raw_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps(_payload())}}

    monkeypatch.setattr(proposer_module, "_raw_ollama_chat", fake_raw_chat)
    lessons = propose_lessons(
        report={"overall": {}},
        active_lessons=[],
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm"},
        think="max",
    )

    assert calls[0]["think"] == "max"
    assert lessons[0].target == "lstm"


def test_propose_lessons_sends_schema_to_local_models_and_strips_fences():
    calls = []
    fenced = "```json\n" + json.dumps(_payload()) + "\n```"

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": fenced}}

    lessons = propose_lessons(
        report={"overall": {}},
        active_lessons=[],
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm"},
        model="gemma4:12b",
        chat_fn=fake_chat,
    )

    assert calls[0]["format"] == lesson_schema()   # local models keep the hard constraint
    assert lessons[0].target == "lstm"             # fenced JSON still parses


def test_proposer_adapter_supports_object_style_response_and_caps_lessons():
    payload = {"lessons": _payload()["lessons"] * 2}

    def fake_chat(**_kwargs):
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))

    proposer = OllamaLessonProposer(
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm"},
        max_lessons=1,
        chat_fn=fake_chat,
    )
    lessons = proposer.propose({"overall": {}}, [])

    assert len(lessons) == 1
    assert lessons[0].target == "lstm"


def test_active_lessons_sent_compact_with_track_record_digest():
    from memory.schema import Lesson

    calls = []

    def fake_chat(**kwargs):
        calls.append(kwargs)
        return {"message": {"content": json.dumps({"lessons": []})}}

    active = Lesson(
        condition="hour >= 20", action="force_flat", target="lstm",
        factor=1.0, evidence="weekend noise", confidence=0.8, status="accepted",
        metadata={
            "raw_response": "x" * 5000,   # must NOT reach the prompt
            "track_record": {"windows": [f"wf_{i:03d}" for i in range(8)],
                             "marginal_scores": [0.01] * 7 + [-0.02]},
        },
    )
    propose_lessons(
        report={"overall": {}},
        active_lessons=[active],
        feature_columns={"hour"},
        model_names={"lstm"},
        chat_fn=fake_chat,
    )

    sent = json.loads(calls[0]["messages"][1]["content"])["active_lessons"][0]
    assert "raw_response" not in json.dumps(sent)
    assert sent["track_record"]["windows_active"] == 8
    assert len(sent["track_record"]["recent_marginal_scores"]) == 6
    assert sent["track_record"]["recent_marginal_scores"][-1] == -0.02
    assert "track_record" in calls[0]["messages"][0]["content"]  # prompt explains it


def test_transient_chat_errors_are_retried_then_give_up_cleanly(monkeypatch):
    import memory.proposer as proposer_module

    monkeypatch.setattr(proposer_module.time, "sleep", lambda _s: None)
    attempts = {"n": 0}

    def flaky_chat(**_kwargs):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise ConnectionError("502 wsarecv")
        return {"message": {"content": json.dumps(_payload())}}

    lessons = propose_lessons(
        report={"overall": {}},
        active_lessons=[],
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm"},
        chat_fn=flaky_chat,
    )
    assert attempts["n"] == 3 and lessons[0].target == "lstm"

    def dead_chat(**_kwargs):
        raise ConnectionError("502 wsarecv")

    lessons = propose_lessons(
        report={"overall": {}},
        active_lessons=[],
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm"},
        chat_fn=dead_chat,
    )
    assert lessons == []   # exhausted outage yields zero lessons, never raises


def test_propose_lessons_returns_empty_on_bad_json_after_retries():
    def fake_chat(**_kwargs):
        return {"message": {"content": "not json"}}

    lessons = propose_lessons(
        report={"overall": {}},
        active_lessons=[],
        feature_columns={"minutes_to_high_impact_event"},
        model_names={"lstm"},
        chat_fn=fake_chat,
    )

    assert lessons == []
