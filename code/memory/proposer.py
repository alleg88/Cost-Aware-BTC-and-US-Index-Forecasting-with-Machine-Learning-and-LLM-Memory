"""LLM lesson proposer for the RQ3 reflection loop.

Uses Ollama chat with a fixed JSON schema and deterministic settings.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any
from urllib import request

from memory.schema import ALLOWED_ACTIONS, Lesson


MODEL = "glm-5.2:cloud"
VERSION = "4"
MAX_LESSONS = 3
NUM_CTX = 16384
# Thinking models (glm) spend most of the budget on hidden reasoning before the
# visible JSON, so the ceiling must be far above the answer's own length.
NUM_PREDICT = 6000

_LESSON = {
    "type": "object",
    "properties": {
        "condition": {"type": "string"},
        "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
        "target": {"type": "string"},
        "factor": {"type": "number"},
        "evidence": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["condition", "action", "target", "factor", "evidence", "confidence"],
    "additionalProperties": False,
}

SYSTEM = (
    "You are a financial-market reflection agent. Read one weekly error report and propose "
    "0 to 3 cautious, testable memory lessons for the next walk-forward window. "
    "Do not predict prices. Do not decide whether a lesson is good; validation code will "
    "test every candidate. Conditions must use only the supplied feature columns. Targets "
    "must use only the supplied model names. Prefer zero lessons over vague lessons. "
    "Each active lesson carries a track_record: its marginal contribution to the "
    "acceptance score in each window since it was accepted (positive = helping). Use it: "
    "do not duplicate lessons that already work; propose a narrower condition or an "
    "opposing action when an active lesson's recent marginals are negative. "
    # Cloud models ignore Ollama's format= schema, so the contract lives in the prompt.
    "Output contract: reply with ONE JSON object only — no prose, no markdown fences: "
    '{"lessons": [{"condition": "<boolean expression over allowed_feature_columns, '
    'e.g. vol_regime == 2 and hour >= 14>", "action": "<one of allowed_actions>", '
    '"target": "<one of allowed_model_names>", "factor": <number>, '
    '"evidence": "<short reason quoting the report>", "confidence": <0..1>}]}. '
    "An empty lessons list is allowed."
)


def lesson_schema(
    max_lessons: int = MAX_LESSONS,
    allowed_actions: set[str] | list[str] | None = None,
) -> dict:
    """JSON schema for a single report -> candidate lessons response."""
    return {
        "type": "object",
        "properties": {
            "lessons": {
                "type": "array",
                "items": {
                    **_LESSON,
                    "properties": {
                        **_LESSON["properties"],
                        "action": {
                            "type": "string",
                            "enum": sorted(allowed_actions or ALLOWED_ACTIONS),
                        },
                    },
                },
                "minItems": 0,
                "maxItems": max_lessons,
            },
        },
        "required": ["lessons"],
        "additionalProperties": False,
    }


def _raw_ollama_chat(**kwargs) -> dict[str, Any]:
    """Call Ollama /api/chat directly for options newer than the Python client."""
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    url = host if host.endswith("/api/chat") else f"{host}/api/chat"
    payload = {**kwargs, "stream": False}
    data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = os.environ.get("OLLAMA_API_KEY")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = request.Request(url, data=data, headers=headers, method="POST")
    with request.urlopen(req, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


# Ollama Cloud intermittently drops connections (502 / wsarecv resets). One bad
# request must never abort a 52-window walk-forward, so every chat call retries
# with backoff and, if the outage outlasts the retries, the window just yields
# zero lessons and the run continues.
RETRY_DELAYS = (5, 15, 45, 90)


def _chat_with_retry(chat_fn, **kwargs) -> Any:
    last: Exception | None = None
    for delay in RETRY_DELAYS:
        try:
            return chat_fn(**kwargs)
        except Exception as exc:  # network resets arrive as several exception types
            last = exc
            print(f"[proposer] transient chat error: {exc!r}; retrying in {delay}s")
            time.sleep(delay)
    try:
        return chat_fn(**kwargs)
    except Exception as exc:
        raise RuntimeError(f"chat failed after {len(RETRY_DELAYS) + 1} attempts") from (last or exc)


def _default_chat_fn(think: str | bool | None):
    if think == "max":
        return _raw_ollama_chat
    import ollama

    return ollama.chat


def _response_content(response: Any) -> str:
    """Support both dict-style and object-style Ollama responses."""
    if isinstance(response, dict):
        return response["message"]["content"]
    return response.message.content


def _extract_json(content: str) -> str:
    """Strip markdown fences / surrounding prose down to the outermost JSON object."""
    content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content.strip())
    start, end = content.find("{"), content.rfind("}")
    if start >= 0 and end > start:
        return content[start:end + 1]
    return content


def _parse(
    content: str,
    model: str,
    version: str,
    max_lessons: int,
    allowed_actions: set[str],
) -> list[Lesson]:
    payload = json.loads(_extract_json(content))
    rows = payload.get("lessons", [])
    if not isinstance(rows, list):
        raise ValueError("reflection response must contain a lessons list")

    lessons: list[Lesson] = []
    for row in rows[:max_lessons]:
        lesson = Lesson.from_dict(row)
        if lesson.action not in allowed_actions:
            continue
        lesson = Lesson(
            condition=lesson.condition,
            action=lesson.action,
            target=lesson.target,
            factor=lesson.factor,
            evidence=lesson.evidence,
            confidence=lesson.confidence,
            metadata={
                **lesson.metadata,
                "reflection_model": model,
                "reflection_prompt_version": version,
                "raw_response": content,
            },
        )
        lessons.append(lesson)
    return lessons


def _lesson_prompt_view(lesson: Lesson) -> dict[str, Any]:
    """Compact serialization of an active lesson for the prompt.

    Drops bulky metadata (raw_response would replay every past LLM response into
    every prompt) and summarizes the track record to a bounded digest.
    """
    view = {
        "condition": lesson.condition,
        "action": lesson.action,
        "target": lesson.target,
        "factor": lesson.factor,
        "evidence": lesson.evidence,
        "confidence": lesson.confidence,
        "source_window": lesson.source_window,
    }
    record = lesson.metadata.get("track_record") or {}
    scores = list(record.get("marginal_scores", []))
    if scores:
        view["track_record"] = {
            "windows_active": len(scores),
            "mean_marginal_score": sum(scores) / len(scores),
            "recent_marginal_scores": [round(s, 6) for s in scores[-6:]],
        }
    return view


def propose_lessons(
    report: dict[str, Any],
    active_lessons: list[Lesson],
    feature_columns: list[str] | set[str],
    model_names: list[str] | set[str],
    *,
    model: str = MODEL,
    max_lessons: int = MAX_LESSONS,
    version: str = VERSION,
    chat_fn=None,
    allowed_actions: set[str] | list[str] | None = None,
    think: str | bool | None = "high",
) -> list[Lesson]:
    """Return candidate lessons from an Ollama chat response."""
    if chat_fn is None:
        chat_fn = _default_chat_fn(think)

    action_set = set(allowed_actions or ALLOWED_ACTIONS)
    user_payload = {
        "allowed_feature_columns": sorted(feature_columns),
        "allowed_model_names": sorted(model_names),
        "allowed_actions": sorted(action_set),
        "active_lessons": [_lesson_prompt_view(lesson) for lesson in active_lessons],
        "weekly_error_report": report,
    }
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, default=str)},
    ]

    # Cloud models return an empty string when format= carries a JSON schema, so they
    # rely on the in-prompt contract; local models get the schema as a hard constraint.
    kwargs = {} if model.endswith(":cloud") else {"format": lesson_schema(max_lessons, action_set)}
    if think is not None:
        kwargs["think"] = think

    for _ in range(2):
        try:
            response = _chat_with_retry(
                chat_fn,
                model=model,
                options={"temperature": 0, "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT},
                messages=messages,
                **kwargs,
            )
        except RuntimeError as exc:
            print(f"[proposer] giving up on this window (0 lessons): {exc}")
            return []
        content = _response_content(response)
        try:
            lessons = _parse(
                content,
                model=model,
                version=version,
                max_lessons=max_lessons,
                allowed_actions=action_set,
            )
            if lessons or json.loads(_extract_json(content)).get("lessons") == []:
                return lessons
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return []


class OllamaLessonProposer:
    """Adapter implementing the `LessonProposer` protocol used by the weekly loop."""

    def __init__(
        self,
        feature_columns: list[str] | set[str],
        model_names: list[str] | set[str],
        *,
        model: str = MODEL,
        max_lessons: int = MAX_LESSONS,
        version: str = VERSION,
        chat_fn=None,
        allowed_actions: set[str] | list[str] | None = None,
        think: str | bool | None = "high",
    ):
        self.feature_columns = set(feature_columns)
        self.model_names = set(model_names)
        self.model = model
        self.max_lessons = max_lessons
        self.version = version
        self.chat_fn = chat_fn
        self.allowed_actions = set(allowed_actions or ALLOWED_ACTIONS)
        self.think = think

    def propose(self, report: dict[str, Any], active_lessons: list[Lesson]) -> list[Lesson]:
        return propose_lessons(
            report=report,
            active_lessons=active_lessons,
            feature_columns=self.feature_columns,
            model_names=self.model_names,
            model=self.model,
            max_lessons=self.max_lessons,
            version=self.version,
            chat_fn=self.chat_fn,
            allowed_actions=self.allowed_actions,
            think=self.think,
        )
