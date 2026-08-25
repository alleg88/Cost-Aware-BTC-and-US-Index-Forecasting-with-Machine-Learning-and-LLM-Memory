"""Fail-closed environment and artifact checks for the reflection agent."""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable, Literal

import ollama
import pandas as pd
from pydantic_core import _pydantic_core
from pydantic import Field

from reflection_agent.config import ProtocolConfig, load_config
from reflection_agent.contracts import StrictModel
from reflection_agent.manifest import sha256_payload
from reflection_agent.store import AgentStore
from reflection_agent.transport import ChatBackend, OllamaClientTransport, normalize_json_object

REQUIRED_ARTIFACTS = (
    "frozen_probability_panel.parquet",
    "market_context.parquet",
    "news_events.parquet",
    "protocol_manifest.json",
)


class PreflightProbe(StrictModel):
    status: Literal["ok"]
    value: int = Field(ge=12, le=12)


class PreflightCheck(StrictModel):
    name: str
    status: Literal["pass"] = "pass"
    detail: str


class PreflightReport(StrictModel):
    checked_at_utc: datetime
    model: str
    think: str
    output_mode: Literal["schema", "json"]
    protocol_hash: str
    checks: list[PreflightCheck]


def _check(condition: bool, name: str, detail: str) -> PreflightCheck:
    if not condition:
        raise RuntimeError(f"preflight failed [{name}]: {detail}")
    return PreflightCheck(name=name, detail=detail)


def _default_model_check(model: str, *, host: str) -> None:
    ollama.Client(host=host, timeout=30.0).show(model)


def _probe_output_mode(backend: ChatBackend, config: ProtocolConfig) -> Literal["schema", "json"]:
    messages = [{
        "role": "user",
        "content": "Return one JSON object with status equal to ok and value equal to 12.",
    }]
    errors: list[str] = []
    for output_mode, response_format in (("schema", PreflightProbe.model_json_schema()), ("json", "json")):
        try:
            response = backend.chat(
                model=config.model,
                messages=messages,
                response_format=response_format,
                think=config.think,
                temperature=0.0,
            )
            PreflightProbe.model_validate_json(normalize_json_object(response.content))
            return output_mode
        except Exception as error:
            errors.append(f"{output_mode}: {type(error).__name__}: {error}")
    raise RuntimeError("structured-output probe failed: " + "; ".join(errors))


def run_preflight(
    *,
    config_path: Path,
    cache_root: Path,
    state_path: Path,
    report_path: Path,
    host: str = "http://localhost:11434",
    backend: ChatBackend | None = None,
    model_check: Callable[[str], None] | None = None,
) -> PreflightReport:
    config = load_config(config_path)
    checks = [
        _check(sys.version_info[:2] == (3, 12), "python", f"{sys.version_info.major}.{sys.version_info.minor}"),
        _check(
            "cp312" in str(_pydantic_core.__file__).lower(),
            "pydantic_core_abi",
            str(_pydantic_core.__file__),
        ),
    ]
    missing = [name for name in REQUIRED_ARTIFACTS if not (cache_root / name).exists()]
    checks.append(_check(not missing, "artifacts", f"present={len(REQUIRED_ARTIFACTS)} missing={missing}"))

    manifest = json.loads((cache_root / "protocol_manifest.json").read_text(encoding="utf-8"))
    declared_hash = manifest.get("protocol_hash")
    manifest_body = {key: value for key, value in manifest.items() if key != "protocol_hash"}
    checks.append(_check(
        declared_hash == sha256_payload(manifest_body),
        "manifest_hash",
        str(declared_hash),
    ))
    panel_path = cache_root / "frozen_probability_panel.parquet"
    panel = pd.read_parquet(panel_path, columns=["timestamp"])
    panel_columns = set(pd.read_parquet(panel_path).columns)
    forbidden_targets = panel_columns.intersection({"y_true", "label", "target", "future_return"})
    checks.append(_check(
        not forbidden_targets,
        "target_isolation",
        f"forbidden_columns={sorted(forbidden_targets)}",
    ))
    context = pd.read_parquet(cache_root / "market_context.parquet", columns=["timestamp"])
    news = pd.read_parquet(cache_root / "news_events.parquet", columns=["available_at_utc"])
    sealed = pd.Timestamp(config.sealed_start_utc)
    maximums = {
        "panel": pd.to_datetime(panel["timestamp"], utc=True).max(),
        "context": pd.to_datetime(context["timestamp"], utc=True).max(),
        "news": pd.to_datetime(news["available_at_utc"], utc=True).max(),
    }
    checks.append(_check(
        all(pd.notna(value) and value < sealed for value in maximums.values()),
        "sealed_cutoff",
        ", ".join(f"{key}={value.isoformat()}" for key, value in maximums.items()),
    ))

    store = AgentStore(state_path)
    with store.connect() as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
    checks.append(_check(integrity == "ok", "sqlite", f"{state_path}: {integrity}"))
    sqlite3.connect(state_path).close()

    checker = model_check or (lambda model: _default_model_check(model, host=host))
    checker(config.model)
    checks.append(PreflightCheck(name="ollama_model", detail=config.model))
    selected_backend = backend or OllamaClientTransport(host=host, timeout_seconds=config.timeout_seconds)
    output_mode = _probe_output_mode(selected_backend, config)
    checks.append(PreflightCheck(name="structured_output", detail=output_mode))

    report = PreflightReport(
        checked_at_utc=datetime.now(UTC),
        model=config.model,
        think=config.think,
        output_mode=output_mode,
        protocol_hash=str(declared_hash),
        checks=checks,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return report
