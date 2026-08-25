"""Registered dev-only protocol for Notebook I."""

from importlib import import_module

import pytest


def _module():
    try:
        return import_module("experiments.run_pre_t2_entry_models")
    except ModuleNotFoundError:
        pytest.fail("pre-T2 runner is not implemented")


def _source():
    return {
        "period_start": "2021-01-01T00:00:00+00:00",
        "period_end_exclusive": "2025-07-01T00:00:00+00:00",
        "max_loaded_timestamp": "2025-06-30T23:59:00+00:00",
        "channel_source_hash": "channel",
        "minute_source_hash": "minute",
        "forward_or_lockbox_loaded": False,
    }


def test_protocol_freezes_lifecycle_features_models_and_geometry():
    module = _module()
    protocol = module.build_protocol(_source())

    assert protocol["models"] == ["xgboost", "gru"]
    assert protocol["ensemble"] == "xgboost_gru_50_50"
    assert protocol["tabular_feature_count"] == 34
    assert protocol["sequence_shape"] == [30, 5]
    assert protocol["include_expired_t1"] is True
    assert protocol["pre_t2_decisions"] == "every completed 1m boundary from T1 until confirmation or expiry"
    assert protocol["post_t2_decision_minutes"] == 15
    assert protocol["primary_min_risk_bps"] == 25.0
    assert "sensitivity_min_risk_bps" not in protocol
    assert protocol["geometry"] == "frozen structural stop and opposite channel rail"
    assert protocol["entry_rule"] == "first strict predicted EV > 0; no threshold tuning"
    assert protocol["frozen_baseline"] == "Notebook H causal LogReg + 25 bps"
    assert protocol["forward_or_lockbox_loaded"] is False
    assert len(protocol["protocol_hash"]) == 64


def test_protocol_rejects_open_forward_or_lockbox():
    module = _module()
    opened = {**_source(), "forward_or_lockbox_loaded": True}

    with pytest.raises(ValueError, match="sealed"):
        module.build_protocol(opened)


def test_active_registry_excludes_catboost_lstm_and_40bps():
    module = _module()

    assert module.MODELS == ("xgboost", "gru")
    assert module.SCORERS == ("xgboost", "gru", "xgboost_gru_50_50")

