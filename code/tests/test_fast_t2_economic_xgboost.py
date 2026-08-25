"""Guards for the XGBoost-only Notebook G extension."""

from importlib import import_module

import pytest


def _module():
    try:
        return import_module("experiments.run_fast_t2_economic_xgboost")
    except ModuleNotFoundError:
        pytest.fail("XGBoost economic extension runner is not implemented")


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


def test_xgboost_extension_is_matched_dev_only_and_does_not_refit_baselines():
    module = _module()
    protocol = module.build_xgboost_extension_protocol(
        _entry_protocol(), _manifest(), base_protocol_hash="base-economic-protocol"
    )

    assert protocol["arms"] == ["xgboost_pooled", "xgboost_split_side"]
    assert protocol["target"] == "winsorised_net_r_enter_minus_skip_0R"
    assert protocol["validation"] == "seven expanding six-month episode-purged folds"
    assert protocol["minimum_trades_per_day"] == 1.0
    assert protocol["base_protocol_hash"] == "base-economic-protocol"
    assert protocol["catboost_retrained"] is False
    assert protocol["logreg_retrained"] is False
    assert protocol["post_hoc_model_extension"] is True
    assert protocol["forward_or_lockbox_loaded"] is False
    assert len(protocol["protocol_hash"]) == 64


def test_xgboost_extension_rejects_open_forward_or_lockbox():
    module = _module()
    opened = {**_entry_protocol(), "forward_or_lockbox_loaded": True}

    with pytest.raises(ValueError, match="sealed"):
        module.build_xgboost_extension_protocol(
            opened, _manifest(), base_protocol_hash="base-economic-protocol"
        )


def test_xgboost_extension_cannot_overwrite_base_g_artifacts():
    module = _module()

    with pytest.raises(ValueError, match="must not overwrite"):
        module.run(output_dir=module.BASE_OUT)
