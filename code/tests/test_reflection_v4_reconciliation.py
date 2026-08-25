from __future__ import annotations

import pytest

from experiments.reconcile_reflection_policy_router import (
    audit_prompt_payload,
    coverage_gates,
    lexicographic_objective,
)


def _metrics(**updates):
    values = {
        "selected_trades": 125,
        "selected_long_trades": 66,
        "selected_short_trades": 59,
        "additional_trades": 25,
        "net_return": 0.095,
        "long_net_return": 0.048,
        "short_net_return": 0.0475,
        "sortino": 0.90,
        "max_drawdown": 0.06,
    }
    values.update(updates)
    return values


def test_registered_coverage_gates_are_exact_at_the_boundary() -> None:
    union = _metrics(
        selected_trades=100,
        selected_long_trades=60,
        selected_short_trades=40,
        additional_trades=0,
        net_return=0.10,
        long_net_return=0.05,
        short_net_return=0.05,
        sortino=1.0,
        max_drawdown=0.05,
    )
    gates = coverage_gates(
        _metrics(),
        union,
        additional_by_block={"b0": 15, "b1": 10},
        transport_failure_fraction=0.05,
        audits_passed=True,
    )
    assert all(gates.values())
    failed = coverage_gates(
        _metrics(selected_short_trades=43),
        union,
        additional_by_block={"b0": 25},
        transport_failure_fraction=0.051,
        audits_passed=True,
    )
    assert failed["additional_short_growth_10pct"] is False
    assert failed["distributed_additional_trades"] is False
    assert failed["transport_and_audits"] is False


def test_memory_objective_uses_development_and_h1_only() -> None:
    gates = {
        "development": {"a": True, "b": True},
        "h1": {"a": True, "b": False},
    }
    result = lexicographic_objective(
        gates,
        {
            "development": _metrics(additional_trades=20, net_return=0.10),
            "h1": _metrics(additional_trades=30, net_return=0.20),
            "forward": _metrics(additional_trades=999, net_return=9.0),
        },
    )
    assert result == (0, 3, 50, pytest.approx(0.30), pytest.approx(-0.06))


def test_prompt_payload_audit_rejects_dates_and_paths() -> None:
    clean = {
        "schema_version": "4.0",
        "coverage_status": {},
        "policy_menu": [],
        "policy_statistics": [],
        "memory_cards": [],
    }
    assert audit_prompt_payload(clean)["passed"] is True
    dirty = {**clean, "memory_cards": [{"available_at": "2025-01-01T00:00:00Z"}]}
    assert audit_prompt_payload(dirty)["passed"] is False
