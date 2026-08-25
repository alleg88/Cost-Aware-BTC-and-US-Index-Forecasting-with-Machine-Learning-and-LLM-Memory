import json

from experiments.model_zoo_protocol import (
    BASE_MODELS,
    TAUS,
    WIDTHS,
    candidate_pool,
    protocol_fingerprint,
    protocol_payload,
)


def test_every_base_model_gets_same_seeded_budget():
    assert BASE_MODELS == (
        "logreg",
        "decision_tree",
        "random_forest",
        "svm_linear",
        "xgboost_balanced",
        "catboost_balanced",
        "mlp",
        "lstm",
        "gru",
    )
    for model in BASE_MODELS:
        first = candidate_pool(model)
        assert len(first) == 15
        assert first[0] == {}
        assert first == candidate_pool(model)
        assert len({json.dumps(params, sort_keys=True) for params in first}) == 15


def test_candidate_spaces_are_model_specific():
    assert set(candidate_pool("logreg")[1]) == {"C"}
    assert set(candidate_pool("svm_linear")[1]) == {"C"}
    assert "max_depth" in candidate_pool("decision_tree")[1]
    assert "learning_rate" in candidate_pool("catboost_balanced")[1]
    assert "max_delta_step" in candidate_pool("xgboost_balanced")[1]
    assert "hidden" in candidate_pool("mlp")[1]
    assert "seq_len" in candidate_pool("lstm")[1]
    assert "seq_len" in candidate_pool("gru")[1]


def test_protocol_identity_freezes_dates_execution_and_guards():
    payload = protocol_payload()
    assert payload["widths"] == list(WIDTHS) == [55, 65, 75]
    assert payload["taus"] == list(TAUS)
    assert payload["candidate_count"] == 15
    assert payload["fold_count"] == 15
    assert payload["lookback_days"] == 90
    assert payload["train_tail_trim_bars"] == 1
    assert payload["development_end_exclusive"] == "2025-07-01"
    assert payload["execution"] == {
        "tp_bps": 150.0,
        "sl_bps": 75.0,
        "max_hold_m15_bars": 1,
        "fee_bps_per_side": 5.0,
    }
    assert payload["guards"]["minimum_trades"] == 50
    assert payload["guards"]["minimum_trades_per_side"] == 15
    assert protocol_fingerprint() == protocol_fingerprint()
    assert len(protocol_fingerprint()) == 16


def test_candidate_pool_rejects_unknown_model_and_tiny_budget():
    import pytest

    with pytest.raises(ValueError, match="unsupported base model"):
        candidate_pool("unknown")
    with pytest.raises(ValueError, match="at least two candidates"):
        candidate_pool("logreg", n_trials=1)
