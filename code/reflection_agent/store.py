"""Transactional SQLite state store with idempotent JSON records."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator

from reflection_agent.manifest import canonical_json

RECORD_TABLES = {"candidates", "policies", "shadows", "evaluations", "reflections", "memories"}


class AgentStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS windows (
                    window_id TEXT PRIMARY KEY,
                    protocol_hash TEXT NOT NULL,
                    cutoff_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS llm_calls (
                    call_id TEXT PRIMARY KEY,
                    request_hash TEXT NOT NULL UNIQUE,
                    protocol_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL
                );
                """
            )
            for table in sorted(RECORD_TABLES):
                connection.execute(
                    f"""CREATE TABLE IF NOT EXISTS {table} (
                        record_id TEXT PRIMARY KEY,
                        protocol_hash TEXT NOT NULL,
                        window_id TEXT,
                        payload_json TEXT NOT NULL,
                        created_at_utc TEXT NOT NULL
                    )"""
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _window_storage_id(window_id: str, protocol_hash: str) -> str:
        return f"{protocol_hash}:{window_id}"

    @staticmethod
    def _idempotent_insert(connection: sqlite3.Connection, table: str, key_column: str, values: dict[str, Any]) -> None:
        existing = connection.execute(
            f"SELECT * FROM {table} WHERE {key_column} = ?", (values[key_column],)
        ).fetchone()
        if existing:
            generated_timestamps = {"created_at_utc", "updated_at_utc"}
            comparable = {key: existing[key] for key in values if key not in generated_timestamps}
            requested = {key: value for key, value in values.items() if key not in generated_timestamps}
            if comparable != requested:
                raise ValueError(f"conflicting idempotent write to {table}:{values[key_column]}")
            return
        columns = ", ".join(values)
        placeholders = ", ".join("?" for _ in values)
        connection.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(values.values()))

    def save_run(self, run_id: str, protocol_hash: str, payload: dict[str, Any]) -> None:
        with self.transaction() as connection:
            self._idempotent_insert(connection, "runs", "run_id", {
                "run_id": run_id,
                "protocol_hash": protocol_hash,
                "payload_json": canonical_json(payload),
                "created_at_utc": self._now(),
            })

    def save_window(
        self,
        *,
        window_id: str,
        protocol_hash: str,
        cutoff_utc: datetime,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        storage_id = self._window_storage_id(window_id, protocol_hash)
        with self.transaction() as connection:
            self._idempotent_insert(connection, "windows", "window_id", {
                "window_id": storage_id,
                "protocol_hash": protocol_hash,
                "cutoff_utc": cutoff_utc.isoformat(),
                "status": status,
                "payload_json": canonical_json(payload),
                "updated_at_utc": self._now(),
            })

    def load_window(self, window_id: str, *, protocol_hash: str) -> dict[str, Any] | None:
        storage_id = self._window_storage_id(window_id, protocol_hash)
        with self.connect() as connection:
            row = connection.execute(
                "SELECT protocol_hash, cutoff_utc, status, payload_json FROM windows WHERE window_id = ?",
                (storage_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "protocol_hash": row["protocol_hash"],
            "cutoff_utc": row["cutoff_utc"],
            "status": row["status"],
            "payload": json.loads(row["payload_json"]),
        }

    def llm_call_count(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0])

    def save_record(
        self,
        table: str,
        record_id: str,
        protocol_hash: str,
        payload: dict[str, Any],
        *,
        window_id: str | None = None,
    ) -> None:
        if table not in RECORD_TABLES:
            raise ValueError(f"unsupported record table: {table}")
        with self.transaction() as connection:
            self._idempotent_insert(connection, table, "record_id", {
                "record_id": record_id,
                "protocol_hash": protocol_hash,
                "window_id": window_id,
                "payload_json": canonical_json(payload),
                "created_at_utc": self._now(),
            })

    def load_record(
        self, table: str, record_id: str, *, protocol_hash: str | None = None
    ) -> dict[str, Any] | None:
        if table not in RECORD_TABLES:
            raise ValueError(f"unsupported record table: {table}")
        where = "record_id = ? AND protocol_hash = ?" if protocol_hash is not None else "record_id = ?"
        parameters = (record_id, protocol_hash) if protocol_hash is not None else (record_id,)
        with self.connect() as connection:
            row = connection.execute(
                f"SELECT payload_json FROM {table} WHERE {where}", parameters
            ).fetchone()
        return json.loads(row["payload_json"]) if row else None

    def list_records(self, table: str, *, protocol_hash: str | None = None) -> list[dict[str, Any]]:
        if table not in RECORD_TABLES:
            raise ValueError(f"unsupported record table: {table}")
        where = " WHERE protocol_hash = ?" if protocol_hash is not None else ""
        parameters = (protocol_hash,) if protocol_hash is not None else ()
        with self.connect() as connection:
            rows = connection.execute(
                f"SELECT record_id, protocol_hash, window_id, payload_json, created_at_utc FROM {table}"
                f"{where} ORDER BY record_id",
                parameters,
            ).fetchall()
        return [
            {
                "record_id": row["record_id"],
                "protocol_hash": row["protocol_hash"],
                "window_id": row["window_id"],
                "payload": json.loads(row["payload_json"]),
                "created_at_utc": row["created_at_utc"],
            }
            for row in rows
        ]

    def replace_record(
        self,
        table: str,
        record_id: str,
        protocol_hash: str,
        payload: dict[str, Any],
        *,
        window_id: str | None = None,
    ) -> None:
        if table not in RECORD_TABLES:
            raise ValueError(f"unsupported record table: {table}")
        with self.transaction() as connection:
            row = connection.execute(
                f"SELECT protocol_hash FROM {table} WHERE record_id = ?", (record_id,)
            ).fetchone()
            if not row:
                raise KeyError(f"missing record for transition: {table}:{record_id}")
            if row["protocol_hash"] != protocol_hash:
                raise ValueError("protocol hash changed during state transition")
            connection.execute(
                f"UPDATE {table} SET window_id = ?, payload_json = ? WHERE record_id = ?",
                (window_id, canonical_json(payload), record_id),
            )

    def save_llm_call(
        self,
        *,
        call_id: str,
        request_hash: str,
        protocol_hash: str,
        role: str,
        status: str,
        payload: dict[str, Any],
    ) -> None:
        with self.transaction() as connection:
            self._idempotent_insert(connection, "llm_calls", "call_id", {
                "call_id": call_id,
                "request_hash": request_hash,
                "protocol_hash": protocol_hash,
                "role": role,
                "status": status,
                "payload_json": canonical_json(payload),
                "created_at_utc": self._now(),
            })

    def cached_llm_call(self, request_hash: str) -> dict[str, Any] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT status, payload_json FROM llm_calls WHERE request_hash = ?", (request_hash,)
            ).fetchone()
        if not row or row["status"] != "success":
            return None
        return json.loads(row["payload_json"])
