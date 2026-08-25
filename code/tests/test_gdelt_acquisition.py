from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest

from experiments.external_evidence import sha256_file
from experiments.gdelt_acquisition import (
    run_query_snapshot,
    sha256_payload,
    sha256_text,
)


class _Rows:
    def __init__(self, frame: pd.DataFrame):
        self._frame = frame

    def to_dataframe(self) -> pd.DataFrame:
        return self._frame.copy()


def _client(rows: list[dict[str, object]], *, secret: str = "") -> tuple[Mock, Mock]:
    job = Mock()
    job.job_id = "job-123"
    job.location = "EU"
    job.created = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)
    job.total_bytes_processed = 42
    job.referenced_tables = ("gdelt-bq.gdeltv2.gkg_partitioned",)
    job.result.return_value = _Rows(pd.DataFrame(rows))
    client = Mock()
    client.secret = secret
    client.query.return_value = job
    return client, job


def test_query_snapshot_waits_sorts_and_binds_job_metadata(tmp_path, monkeypatch):
    parameters = {"start": "2024-01-01", "limit": 2}
    config = object()
    monkeypatch.setattr(
        "experiments.gdelt_acquisition._build_job_config",
        lambda supplied: config if supplied == parameters else None,
    )
    client, job = _client(
        [
            {"DATE": 20240102000000, "url": "z"},
            {"DATE": 20240101000000, "url": "a"},
        ]
    )

    manifest = run_query_snapshot(client, "SELECT @start", parameters, tmp_path)

    client.query.assert_called_once_with("SELECT @start", job_config=config)
    job.result.assert_called_once_with()
    assert manifest["job_id"] == "job-123"
    assert manifest["total_bytes_processed"] == 42
    assert manifest["sql_sha256"] == sha256_text("SELECT @start")
    assert manifest["parameters_sha256"] == sha256_payload(parameters)
    assert manifest["row_count"] == 2
    artifact = tmp_path / "part-00000.parquet"
    assert manifest["artifact_sha256"] == sha256_file(artifact)
    assert pd.read_parquet(artifact)["url"].tolist() == ["a", "z"]
    assert json.loads((tmp_path / "acquisition_metadata.json").read_text("utf-8")) == manifest


def test_query_snapshot_never_serializes_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.gdelt_acquisition._build_job_config", lambda _: object())
    client, _ = _client([{"DATE": 20240101000000, "url": "u"}], secret="token-value")

    manifest = run_query_snapshot(client, "SELECT 1", {}, tmp_path)

    assert "token-value" not in json.dumps(manifest)
    assert "token-value" not in (tmp_path / "acquisition_metadata.json").read_text("utf-8")


def test_query_snapshot_refuses_to_overwrite_frozen_output(tmp_path, monkeypatch):
    monkeypatch.setattr("experiments.gdelt_acquisition._build_job_config", lambda _: object())
    client, _ = _client([{"DATE": 20240101000000, "url": "u"}])
    run_query_snapshot(client, "SELECT 1", {}, tmp_path)

    with pytest.raises(FileExistsError, match="frozen acquisition"):
        run_query_snapshot(client, "SELECT 1", {}, tmp_path)


def test_parameter_hash_is_canonical():
    assert sha256_payload({"b": 2, "a": 1}) == sha256_payload({"a": 1, "b": 2})
