import sqlite3

import pytest

from reflection_agent.store import AgentStore


def test_store_is_idempotent_and_rejects_conflicting_payload(tmp_path):
    store = AgentStore(tmp_path / "agent.sqlite")
    store.save_record("candidates", "c1", "protocol", {"value": 1}, window_id="w1")
    store.save_record("candidates", "c1", "protocol", {"value": 1}, window_id="w1")
    assert store.load_record("candidates", "c1") == {"value": 1}
    with pytest.raises(ValueError, match="conflicting"):
        store.save_record("candidates", "c1", "protocol", {"value": 2}, window_id="w1")


def test_store_rolls_back_transaction_and_caches_only_success(tmp_path):
    store = AgentStore(tmp_path / "agent.sqlite")
    with pytest.raises(RuntimeError):
        with store.transaction() as connection:
            connection.execute(
                "INSERT INTO candidates VALUES (?, ?, ?, ?, ?)",
                ("c1", "p", "w", "{}", "now"),
            )
            raise RuntimeError("stop")
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0
    store.save_llm_call(
        call_id="call1", request_hash="hash1", protocol_hash="p", role="actor", status="success", payload={"ok": True}
    )
    assert store.cached_llm_call("hash1") == {"ok": True}
    assert store.cached_llm_call("missing") is None


def test_store_rejects_unknown_dynamic_table(tmp_path):
    store = AgentStore(tmp_path / "agent.sqlite")
    with pytest.raises(ValueError, match="unsupported"):
        store.save_record("runs; DROP TABLE runs", "x", "p", {})


def test_record_listing_can_be_protocol_scoped(tmp_path):
    store = AgentStore(tmp_path / "agent.sqlite")
    store.save_record("memories", "m1", "p1", {"value": 1})
    store.save_record("memories", "m2", "p2", {"value": 2})
    assert [row["record_id"] for row in store.list_records("memories", protocol_hash="p1")] == ["m1"]
    assert store.load_record("memories", "m1", protocol_hash="p1") == {"value": 1}
    assert store.load_record("memories", "m1", protocol_hash="p2") is None
