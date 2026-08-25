"""Schema-constrained Ollama transports with cache, retry, repair, and no-op."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, Protocol, Sequence, TypeVar

import ollama
import requests
from pydantic import BaseModel, ValidationError

from reflection_agent.manifest import sha256_payload
from reflection_agent.store import AgentStore

T = TypeVar("T", bound=BaseModel)


def normalize_json_object(content: str) -> str:
    """Accept one JSON object, with only an optional Markdown JSON fence."""
    normalized = content.strip()
    if normalized.startswith("```json") and normalized.endswith("```"):
        normalized = normalized[len("```json"):-len("```")].strip()
    elif normalized.startswith("```") and normalized.endswith("```"):
        normalized = normalized[len("```"):-len("```")].strip()
    decoder = json.JSONDecoder()
    value, end = decoder.raw_decode(normalized)
    if not isinstance(value, dict) or normalized[end:].strip():
        raise ValueError("response must contain exactly one JSON object")
    return normalized


@dataclass(frozen=True)
class BackendResponse:
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ChatBackend(Protocol):
    def chat(
        self,
        *,
        model: str,
        messages: Sequence[dict[str, str]],
        response_format: str | dict[str, Any],
        think: str,
        temperature: float,
    ) -> BackendResponse: ...


class OllamaClientTransport:
    def __init__(self, *, host: str | None = None, timeout_seconds: float = 300.0):
        if host is None:
            self.client = ollama.Client(timeout=timeout_seconds)
        else:
            self.client = ollama.Client(host=host, timeout=timeout_seconds)

    def chat(self, *, model, messages, response_format, think, temperature) -> BackendResponse:
        response = self.client.chat(
            model=model,
            messages=list(messages),
            stream=False,
            think=think,
            format=response_format,
            options={"temperature": temperature},
        )
        metadata = {
            key: getattr(response, key, None)
            for key in ("model", "created_at", "done_reason", "total_duration", "prompt_eval_count", "eval_count")
        }
        return BackendResponse(content=response.message.content or "", metadata=metadata)


class OllamaHttpTransport:
    def __init__(self, *, host: str = "http://localhost:11434", timeout_seconds: float = 300.0):
        self.url = host.rstrip("/") + "/api/chat"
        self.timeout_seconds = timeout_seconds

    def chat(self, *, model, messages, response_format, think, temperature) -> BackendResponse:
        response = requests.post(
            self.url,
            json={
                "model": model,
                "messages": list(messages),
                "stream": False,
                "think": think,
                "format": response_format,
                "options": {"temperature": temperature},
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
        return BackendResponse(
            content=str(payload.get("message", {}).get("content", "")),
            metadata={key: payload.get(key) for key in (
                "model", "created_at", "done_reason", "total_duration", "prompt_eval_count", "eval_count"
            )},
        )


@dataclass(frozen=True)
class StructuredCallResult(Generic[T]):
    status: str
    value: T | None
    raw_content: str
    request_hash: str
    attempts: int
    backend: str | None
    errors: tuple[str, ...]


class StructuredCaller:
    def __init__(
        self,
        *,
        model: str,
        output_mode: str,
        primary: ChatBackend,
        fallback: ChatBackend | None = None,
        store: AgentStore | None = None,
        protocol_hash: str = "preflight",
        run_scope: str | None = None,
        think: str = "high",
        retry_delays_seconds: Sequence[float] = (5.0, 15.0, 45.0),
        repair_attempts: int = 2,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if output_mode not in {"schema", "json"}:
            raise ValueError("output_mode must be schema or json")
        self.model = model
        self.output_mode = output_mode
        self.primary = primary
        self.fallback = fallback
        self.store = store
        self.protocol_hash = protocol_hash
        self.run_scope = run_scope or protocol_hash
        self.think = think
        self.retry_delays = tuple(retry_delays_seconds)
        self.repair_attempts = repair_attempts
        self.sleep = sleep

    def _request(
        self,
        *,
        messages: Sequence[dict[str, str]],
        response_format: str | dict[str, Any],
        temperature: float,
    ) -> tuple[BackendResponse, int, str]:
        errors = []
        attempts = 0
        for retry_index in range(len(self.retry_delays) + 1):
            attempts += 1
            try:
                return self.primary.chat(
                    model=self.model,
                    messages=messages,
                    response_format=response_format,
                    think=self.think,
                    temperature=temperature,
                ), attempts, type(self.primary).__name__
            except Exception as primary_error:
                errors.append(f"{type(primary_error).__name__}: {primary_error}")
                if self.fallback is not None:
                    try:
                        return self.fallback.chat(
                            model=self.model,
                            messages=messages,
                            response_format=response_format,
                            think=self.think,
                            temperature=temperature,
                        ), attempts, type(self.fallback).__name__
                    except Exception as fallback_error:
                        errors.append(f"{type(fallback_error).__name__}: {fallback_error}")
                if retry_index < len(self.retry_delays):
                    self.sleep(self.retry_delays[retry_index])
        raise RuntimeError("; ".join(errors))

    def call(
        self,
        *,
        role: str,
        messages: Sequence[dict[str, str]],
        response_model: type[T],
        temperature: float,
    ) -> StructuredCallResult[T]:
        schema = response_model.model_json_schema()
        request = {
            "protocol_hash": self.protocol_hash,
            "run_scope": self.run_scope,
            "model": self.model,
            "role": role,
            "messages": list(messages),
            "schema": schema,
            "output_mode": self.output_mode,
            "think": self.think,
            "temperature": temperature,
        }
        request_hash = sha256_payload(request)
        if self.store:
            cached = self.store.cached_llm_call(request_hash)
            if cached:
                value = response_model.model_validate(cached["parsed"])
                return StructuredCallResult(
                    status="cached", value=value, raw_content=cached["raw"], request_hash=request_hash,
                    attempts=int(cached["attempts"]), backend=cached.get("backend"), errors=tuple(cached.get("errors", [])),
                )

        current_messages = list(messages)
        all_errors: list[str] = []
        total_attempts = 0
        last_raw = ""
        last_backend = None
        started = time.monotonic()
        status = "noop"
        parsed: T | None = None
        for repair_index in range(self.repair_attempts + 1):
            response_format: str | dict[str, Any] = schema if self.output_mode == "schema" else "json"
            try:
                response, attempts, backend = self._request(
                    messages=current_messages,
                    response_format=response_format,
                    temperature=temperature,
                )
                total_attempts += attempts
                last_raw = response.content
                last_backend = backend
                parsed = response_model.model_validate_json(normalize_json_object(last_raw))
                status = "success"
                break
            except (ValidationError, ValueError) as validation_error:
                all_errors.append(f"validation: {validation_error}")
                if repair_index < self.repair_attempts:
                    current_messages = list(messages) + [{
                        "role": "user",
                        "content": (
                            "Your prior JSON was invalid. Correct only the listed validation errors and return one "
                            "schema-matching JSON object. Use exact JSON Schema property names, enum values and types; "
                            "do not use synonyms or explanatory keys. ERRORS=" + str(validation_error)[:3000] +
                            " PRIOR_JSON=" + last_raw[:4000]
                        ),
                    }]
            except Exception as request_error:
                all_errors.append(f"request: {type(request_error).__name__}: {request_error}")
                break

        payload = {
            "request": request,
            "raw": last_raw,
            "parsed": parsed.model_dump(mode="json") if parsed else None,
            "attempts": total_attempts,
            "backend": last_backend,
            "errors": all_errors,
            "latency_seconds": time.monotonic() - started,
        }
        if self.store:
            self.store.save_llm_call(
                call_id=f"{role}-{request_hash[:20]}",
                request_hash=request_hash,
                protocol_hash=self.protocol_hash,
                role=role,
                status=status,
                payload=payload,
            )
        return StructuredCallResult(
            status=status,
            value=parsed,
            raw_content=last_raw,
            request_hash=request_hash,
            attempts=total_attempts,
            backend=last_backend,
            errors=tuple(all_errors),
        )
