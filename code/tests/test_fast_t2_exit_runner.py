"""Freeze, protocol, and training-ledger guards for Notebook F."""

import json

import pandas as pd
import pytest

from experiments.run_fast_t2_exit_policy import (
    build_exit_protocol,
    load_entry_freeze,
    select_training_proxy_entries,
    validate_exit_resume,
    write_exit_protocol,
)


def _entry_protocol() -> dict[str, object]:
    return {
        "protocol_hash": "entry-protocol-123",
        "period_start": "2021-01-01T00:00:00+00:00",
        "period_end_exclusive": "2025-07-01T00:00:00+00:00",
        "forward_or_lockbox_loaded": False,
    }


def _entry_freeze() -> dict[str, object]:
    return {
        "protocol_hash": "entry-protocol-123",
        "selected_model": "logreg",
        "selected_policy": "nested_first_crossing",
        "decision_ledger_hash": "entry-ledger-456",
        "filled_ledger_hash": "filled-ledger-789",
        "filled_trades": 1417,
        "forward_or_lockbox_loaded": False,
    }


def test_exit_protocol_copies_frozen_entry_selection(tmp_path):
    protocol = build_exit_protocol(
        _entry_protocol(), _entry_freeze(), output_dir=tmp_path
    )

    assert protocol["entry_protocol_hash"] == "entry-protocol-123"
    assert protocol["entry_model"] == "logreg"
    assert protocol["entry_policy_changed"] is False
    assert protocol["period_end_exclusive"] == "2025-07-01T00:00:00+00:00"
    assert protocol["forward_or_lockbox_loaded"] is False
    assert len(protocol["protocol_hash"]) == 64


def test_exit_runner_rejects_noncomplete_entry_run(tmp_path):
    entry_dir = tmp_path / "entry"
    entry_dir.mkdir()
    (entry_dir / "run_state.json").write_text(
        json.dumps({"status": "running", "consolidated": False}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="complete Notebook E run"):
        load_entry_freeze(entry_dir)


def test_exit_resume_rejects_changed_entry_ledger(tmp_path):
    write_exit_protocol(tmp_path, _entry_protocol(), _entry_freeze())
    changed = {**_entry_freeze(), "decision_ledger_hash": "changed"}

    with pytest.raises(ValueError, match="protocol hash"):
        validate_exit_resume(tmp_path, _entry_protocol(), changed)


def test_training_proxy_is_one_label_blind_feasible_entry_per_window():
    decisions = pd.DataFrame(
        {
            "window_id": ["a", "a", "a", "b", "b"],
            "decision_id": ["a0", "a1", "a2", "b0", "b1"],
            "minutes_since_t2": [0, 1, 2, 0, 1],
            "filled": [True, True, False, True, True],
            "r_net": [-9.0, 9.0, 100.0, 5.0, -5.0],
            "label_net_positive": [0, 1, 1, 1, 0],
        }
    )

    selected = select_training_proxy_entries(decisions)
    changed_labels = decisions.assign(
        r_net=-decisions["r_net"],
        label_net_positive=1 - decisions["label_net_positive"],
    )
    selected_after = select_training_proxy_entries(changed_labels)

    assert selected["window_id"].is_unique
    assert selected["filled"].all()
    assert selected["decision_id"].tolist() == selected_after["decision_id"].tolist()
    assert set(selected["decision_id"]) <= {"a0", "a1", "b0", "b1"}
