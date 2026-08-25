"""Create an immutable, auditable GDELT BigQuery result snapshot.

The optional Google client is imported only for a live refresh. Tests and all
offline rebuilds use the frozen Parquet plus ``acquisition_metadata.json``.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from experiments.external_evidence import sha256_file


ARTIFACT_NAME = "part-00000.parquet"
METADATA_NAME = "acquisition_metadata.json"


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_payload(payload: Mapping[str, object]) -> str:
    return sha256_text(_canonical_json(dict(payload)))


def _parameter_type(value: object) -> tuple[str, object]:
    if isinstance(value, bool):
        return "BOOL", value
    if isinstance(value, int):
        return "INT64", value
    if isinstance(value, float):
        return "FLOAT64", value
    if isinstance(value, datetime):
        return "TIMESTAMP", value
    if isinstance(value, date):
        return "DATE", value
    if isinstance(value, str):
        return "STRING", value
    raise TypeError(f"unsupported BigQuery parameter type: {type(value).__name__}")


def _build_job_config(parameters: Mapping[str, object]) -> object:
    try:
        from google.cloud import bigquery
    except ImportError as exc:  # pragma: no cover - exercised only in live acquisition
        raise RuntimeError(
            "Live GDELT acquisition requires: pip install -e .[extras,acquisition]"
        ) from exc
    query_parameters = []
    for name in sorted(parameters):
        type_name, value = _parameter_type(parameters[name])
        query_parameters.append(bigquery.ScalarQueryParameter(name, type_name, value))
    return bigquery.QueryJobConfig(query_parameters=query_parameters)


def _created_text(value: object) -> str | None:
    if value is None:
        return None
    if hasattr(value, "isoformat"):
        return str(value.isoformat())
    return str(value)


def run_query_snapshot(
    client: object,
    sql: str,
    parameters: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    """Run one parameterised query and freeze its sorted Parquet plus metadata."""
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"frozen acquisition already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    job_config = _build_job_config(parameters)
    query_job = client.query(sql, job_config=job_config)
    rows = query_job.result()
    frame = rows.to_dataframe()
    sort_columns = [column for column in ("DATE", "url") if column in frame.columns]
    if sort_columns:
        frame = frame.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    artifact_path = output_dir / ARTIFACT_NAME
    frame.to_parquet(artifact_path, index=False)
    referenced = sorted(str(table) for table in (query_job.referenced_tables or ()))
    manifest: dict[str, object] = {
        "schema_version": 1,
        "provider": "Google BigQuery",
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": sha256_file(artifact_path),
        "created": _created_text(query_job.created),
        "job_id": str(query_job.job_id),
        "location": None if query_job.location is None else str(query_job.location),
        "parameters_sha256": sha256_payload(parameters),
        "referenced_tables": referenced,
        "row_count": int(len(frame)),
        "sql_sha256": sha256_text(sql),
        "total_bytes_processed": int(query_job.total_bytes_processed or 0),
    }
    (output_dir / METADATA_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_parameters(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("BigQuery parameters JSON must contain an object")
    return {str(key): value for key, value in payload.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sql", type=Path, required=True, help="tracked SQL file")
    parser.add_argument("--parameters", type=Path, required=True, help="canonical JSON object")
    parser.add_argument("--output", type=Path, required=True, help="new empty snapshot directory")
    parser.add_argument("--project")
    parser.add_argument("--location")
    args = parser.parse_args()

    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise RuntimeError(
            "Live GDELT acquisition requires: pip install -e .[extras,acquisition]"
        ) from exc
    client = bigquery.Client(project=args.project, location=args.location)
    report = run_query_snapshot(
        client,
        args.sql.read_text(encoding="utf-8"),
        _load_parameters(args.parameters),
        args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
