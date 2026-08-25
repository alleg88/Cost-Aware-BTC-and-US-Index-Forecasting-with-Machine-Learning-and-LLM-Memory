"""Strict seeded Ollama Cloud structured-output transport for index agents."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol, Sequence, TypeVar

import ollama
from pydantic import BaseModel, ValidationError

from reflection_agent.index_v1.config import IndexAgentConfig
from reflection_agent.v2.transport import SchemaCallResult


T = TypeVar("T", bound=BaseModel)


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )


def _hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


class ChatBackend(Protocol):
    def chat(self, **kwargs: Any) -> Any: ...


class OllamaCloudBackend:
    def __init__(self, *, host: str | None = None, timeout_seconds: float = 300.0):
        if host is None:
            self.client = ollama.Client(timeout=timeout_seconds)
        else:
            self.client = ollama.Client(host=host, timeout=timeout_seconds)

    def chat(self, **kwargs: Any) -> Any:
        return self.client.chat(**kwargs)


def _response_content(response: Any) -> str:
    if isinstance(response, dict):
        message = response.get("message", {})
        return str(message.get("content", "")) if isinstance(message, dict) else ""
    message = getattr(response, "message", None)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", "") or "")


def _metadata(response: Any) -> dict[str, Any]:
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
        return {key: response[key] for key in keys if response.get(key) is not None}
    return {
        key: getattr(response, key)
        for key in keys
        if getattr(response, key, None) is not None
    }


def _validation_summary(error: Exception) -> str:
    if isinstance(error, ValidationError):
        return _canonical_json(
            [
                {"loc": list(item["loc"]), "type": item["type"], "msg": item["msg"]}
                for item in error.errors(
                    include_url=False, include_context=False, include_input=False
                )
            ]
        )
    return f"{type(error).__name__}: {str(error)[:500]}"


class IndexSchemaCaller:
    """One exact seeded model, one optional schema repair and no fallback model."""

    def __init__(
        self,
        config: IndexAgentConfig,
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
        schema_hash = _hash(schema)
        options = {
            "temperature": self.config.temperature,
            "num_predict": self.config.num_predict,
            "seed": self.config.seed,
        }
        request = {
            "model": self.config.model,
            "model_digest": self.config.required_model_digest,
            "role": role,
            "messages": list(messages),
            "schema_hash": schema_hash,
            "think": self.config.think,
            "stream": self.config.ollama_stream,
            "options": options,
            "allowed_ids": allowed_ids,
        }
        request_hash = _hash(request)
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
                    stream=self.config.ollama_stream,
                    think=self.config.think,
                    format=schema,
                    options=options,
                )
            except Exception as error:
                errors.append(f"{type(error).__name__}: {error}")
                status = "transport_error"
                break
            raw_content = _response_content(response)
            metadata = _metadata(response)
            try:
                decoded = json.loads(raw_content)
                if not isinstance(decoded, dict):
                    raise ValueError("response must be exactly one JSON object")
                value = response_model.model_validate(decoded)
                status = "success" if repair_index == 0 else "repaired"
                break
            except (json.JSONDecodeError, ValidationError, ValueError) as error:
                errors.append(f"{type(error).__name__}: {error}")
                if repair_index >= self.config.repair_attempts:
                    break
                current_messages = list(messages) + [
                    {
                        "role": "user",
                        "content": (
                            "TASK: REPAIR_JSON_SCHEMA_OUTPUT\n"
                            "Return exactly one corrected JSON object and no prose.\n"
                            "VALIDATION_ERRORS="
                            + _validation_summary(error)
                            + "\nALLOWED_IDS="
                            + _canonical_json(allowed_ids)
                        ),
                    }
                ]
        result = SchemaCallResult(
            status=status,
            value=value,
            raw_content=raw_content,
            request_hash=request_hash,
            response_hash=hashlib.sha256(raw_content.encode("utf-8")).hexdigest(),
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


class CachedIndexSchemaCaller:
    """Replay terminal schema calls only under the exact protocol request."""

    def __init__(self, inner: IndexSchemaCaller, cache_dir: str | Path, *, protocol_hash: str):
        if len(protocol_hash) != 64:
            raise ValueError("cached caller requires a 64-character protocol hash")
        self.inner = inner
        self.config = inner.config
        self.cache_dir = Path(cache_dir)
        self.protocol_hash = protocol_hash

    def call(self, *, role, messages, response_model, allowed_ids):
        key = _hash(
            {
                "protocol_hash": self.protocol_hash,
                "role": role,
                "messages": messages,
                "schema": response_model.model_json_schema(),
                "allowed_ids": allowed_ids,
            }
        )
        path = self.cache_dir / f"{key}.json"
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("protocol_hash") != self.protocol_hash:
                raise ValueError("cached index call protocol identity changed")
            value = (
                response_model.model_validate(payload["validated_content"])
                if payload.get("validated_content") is not None
                else None
            )
            return SchemaCallResult(
                status=payload["status"],
                value=value,
                raw_content=payload["raw_content"],
                request_hash=payload["request_hash"],
                response_hash=payload["response_hash"],
                schema_hash=payload["schema_hash"],
                attempts=int(payload["attempts"]),
                latency_seconds=float(payload["latency_seconds"]),
                metadata=payload["metadata"],
                errors=tuple(payload["errors"]),
            )
        result = self.inner.call(
            role=role,
            messages=messages,
            response_model=response_model,
            allowed_ids=allowed_ids,
        )
        payload = {
            "protocol_hash": self.protocol_hash,
            "status": result.status,
            "validated_content": (
                result.value.model_dump(mode="json") if result.value is not None else None
            ),
            "raw_content": result.raw_content,
            "request_hash": result.request_hash,
            "response_hash": result.response_hash,
            "schema_hash": result.schema_hash,
            "attempts": result.attempts,
            "latency_seconds": result.latency_seconds,
            "metadata": result.metadata,
            "errors": list(result.errors),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        return result


__all__ = [
    "CachedIndexSchemaCaller",
    "IndexSchemaCaller",
    "OllamaCloudBackend",
]
