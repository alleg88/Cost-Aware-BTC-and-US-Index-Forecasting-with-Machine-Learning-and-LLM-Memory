"""Registered dev-only protocol for Notebook H."""

from importlib import import_module

import pytest


def _module():
    try:
        return import_module("experiments.run_fast_t2_causal_ev")
    except ModuleNotFoundError:
        pytest.fail("causal EV runner is not implemented")


def _entry_protocol():
    return {
        "protocol_hash": "entry-protocol",
        "period_start": "2021-01-01T00:00:00+00:00",
        "period_end_exclusive": "2025-07-01T00:00:00+00:00",
        "forward_or_lockbox_loaded": False,
    }


def _manifest():
    return {
        "dataset_hash": "entry-dataset",
        "decision_ledger_hash": "entry-ledger",
        "max_loaded_timestamp": "2025-06-30T23:59:00+00:00",
    }


def test_h_protocol_fixes_models_geometry_ev_rule_and_frequency_gate():
    module = _module()
    protocol = module.build_protocol(_entry_protocol(), _manifest())

    assert protocol["models"] == ["catboost", "xgboost"]
    assert protocol["primary_min_risk_bps"] == 25.0
    assert protocol["sensitivity_min_risk_bps"] == 40.0
    assert protocol["window_cancellation"] == "permanent after completed 1m stop touch"
    assert protocol["entry_rule"] == "strict predicted EV > 0; no threshold tuning"
    assert protocol["minimum_trades_per_day"] == 1.0
    assert protocol["frequency_is_admission_only"] is True
    assert protocol["capacity"] == "unlimited; no capacity selection"
    assert protocol["frozen_baseline"] == "Notebook E LogReg"
    assert protocol["forward_or_lockbox_loaded"] is False
    assert len(protocol["protocol_hash"]) == 64


def test_h_protocol_rejects_open_forward_or_lockbox():
    module = _module()
    opened = {**_entry_protocol(), "forward_or_lockbox_loaded": True}

    with pytest.raises(ValueError, match="sealed"):
        module.build_protocol(opened, _manifest())


def test_h_registry_has_no_split_recurrent_or_logreg_training_arms():
    module = _module()

    assert module.MODELS == ("catboost", "xgboost")
    assert all("split" not in name for name in module.MODELS)
    assert "logreg" not in module.MODELS
