"""Strict DeepSeek-V4-Flash Ollama Cloud structured-output transport."""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, Protocol, Sequence, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from reflection_agent.v2.config import ProtocolConfigV2

T = TypeVar("T", bound=BaseModel)


def _canonical_json(payload: object) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class ChatBackend(Protocol):
    def chat(self, **kwargs: Any) -> Any: ...


class OllamaCloudBackend:
    def __init__(
        self,
        *,
        host: str = "http://localhost:11434",
        timeout_seconds: float = 300.0,
    ) -> None:
        self.client = ollama.Client(host=host, timeout=timeout_seconds)

    def chat(self, **kwargs: Any) -> Any:
        return self.client.chat(**kwargs)


@dataclass(frozen=True)
class SchemaCallResult(Generic[T]):
    status: str
    value: T | None
    raw_content: str
    request_hash: str
    response_hash: str
    schema_hash: str
    attempts: int
    latency_seconds: float
    metadata: dict[str, Any]
    errors: tuple[str, ...]


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        if isinstance(message, dict):
            return str(message.get("content") or "")
        return str(getattr(message, "content", "") or "")
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _safe_metadata(response: Any) -> dict[str, Any]:
    keys = (
        "model",
        "created_at",
        "done_reason",
        "total_duration",
        "load_duration",
        "prompt_eval_count",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    )
    if isinstance(response, dict):
        return {key: response.get(key) for key in keys if response.get(key) is not None}
    return {
        key: getattr(response, key)
        for key in keys
        if getattr(response, key, None) is not None
    }


def _strict_json_object(content: str) -> dict[str, Any]:
    value = json.loads(content)
    if not isinstance(value, dict):
        raise ValueError("response must be exactly one JSON object")
    return value


def _validation_summary(error: Exception) -> str:
    if isinstance(error, ValidationError):
        compact = [
            {
                "loc": list(item["loc"]),
                "type": item["type"],
                "msg": item["msg"],
            }
            for item in error.errors(include_url=False, include_context=False, include_input=False)
        ]
        return _canonical_json(compact)
    return f"{type(error).__name__}: {str(error)[:500]}"


class DeepSeekSchemaCaller:
    """One exact model, one optional schema repair, and no retry or fallback."""

    def __init__(
        self,
        config: ProtocolConfigV2,
        *,
        backend: ChatBackend | None = None,
        call_log_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.backend = backend or OllamaCloudBackend(
            timeout_seconds=float(config.timeout_seconds)
        )
        self.call_log_path = Path(call_log_path) if call_log_path is not None else None

    def call(
        self,
        *,
        role: str,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
        allowed_ids: dict[str, Any],
    ) -> SchemaCallResult[T]:
        schema = response_model.model_json_schema()
        schema_hash = _sha256_text(_canonical_json(schema))
        request_payload = {
            "model": self.config.model,
            "role": role,
            "messages": list(messages),
            "schema_hash": schema_hash,
            "think": self.config.think,
            "stream": self.config.stream,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.num_predict,
            },
            "allowed_ids": allowed_ids,
        }
        request_hash = _sha256_text(_canonical_json(request_payload))
        started = time.monotonic()
        errors: list[str] = []
        raw_content = ""
        metadata: dict[str, Any] = {}
        value: T | None = None
        status = "schema_failure"
        attempts = 0
        current_messages = list(messages)

        for repair_index in range(self.config.repair_attempts + 1):
            attempts += 1
            try:
                response = self.backend.chat(
                    model=self.config.model,
                    messages=current_messages,
                    stream=self.config.stream,
                    think=self.config.think,
                    format=schema,
                    options={
                        "temperature": self.config.temperature,
                        "num_predict": self.config.num_predict,
                    },
                )
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                status = "transport_error"
                break
            raw_content = _response_content(response)
            metadata = _safe_metadata(response)
            try:
                value = response_model.model_validate(_strict_json_object(raw_content))
                status = "success" if repair_index == 0 else "repaired"
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}")
                if repair_index >= self.config.repair_attempts:
                    status = "schema_failure"
                    break
                repair_message = (
                    "TASK: REPAIR_JSON_SCHEMA_OUTPUT\n"
                    "Return exactly one corrected JSON object. Do not add prose.\n"
                    "VALIDATION_ERRORS="
                    + _validation_summary(error)
                    + "\nALLOWED_IDS="
                    + _canonical_json(allowed_ids)
                )
                current_messages = list(messages) + [
                    {"role": "user", "content": repair_message}
                ]

        result = SchemaCallResult(
            status=status,
            value=value,
            raw_content=raw_content,
            request_hash=request_hash,
            response_hash=_sha256_text(raw_content),
            schema_hash=schema_hash,
            attempts=attempts,
            latency_seconds=time.monotonic() - started,
            metadata=metadata,
            errors=tuple(errors),
        )
        self._log(role, result)
        return result

    def _log(self, role: str, result: SchemaCallResult[Any]) -> None:
        if self.call_log_path is None:
            return
        self.call_log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "role": role,
            "status": result.status,
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "schema_hash": result.schema_hash,
            "attempts": result.attempts,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
            "errors": list(result.errors),
            "validated_content": (
                result.value.model_dump(mode="json") if result.value is not None else None
            ),
        }
        with self.call_log_path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload) + "\n")


__all__ = [
    "DeepSeekSchemaCaller",
    "OllamaCloudBackend",
    "SchemaCallResult",
]
