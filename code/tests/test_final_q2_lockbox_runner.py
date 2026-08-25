from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.final_q2_lockbox_contract import load_lockbox_protocol
from experiments.final_q2_lockbox_runner import (
    OPEN_AUTHORIZATION,
    open_q2,
    replay_btc_candidates,
    replay_index_candidates,
    resume_q2,
    score_btc_candidates,
    score_index_candidates,
    _load_candidate_checkpoint,
    _persist_candidate_checkpoint,
    _write_result_artifacts,
)
from experiments.final_q2_lockbox_runner import ReplayAudit, ReplayResult
import experiments.final_q2_lockbox_runner as runner_module
from experiments.final_q2_lockbox_state import OpeningIdentity


CODE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_lockbox_protocol(CODE_ROOT / "configs/final_q2_lockbox_protocol.json")
START = pd.Timestamp("2026-04-01T00:00:00Z")
END = pd.Timestamp("2026-07-01T00:00:00Z")


class FixedEstimator:
    classes_ = np.array([0, 1, 2])

    def __init__(self, probabilities):
        self.probabilities = np.asarray(probabilities, dtype=float)

    def predict_proba(self, X):
        if len(self.probabilities) == 1:
            return np.repeat(self.probabilities, len(X), axis=0)
        if len(self.probabilities) != len(X):
            raise AssertionError("fixture probability rows differ from X")
        return self.probabilities.copy()


def _features() -> pd.DataFrame:
    index = pd.date_range(START - pd.Timedelta(minutes=30), periods=6, freq="15min")
    return pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)


def _identity(seed: str = "a") -> OpeningIdentity:
    return OpeningIdentity(
        implementation_commit=seed * 40,
        manifest_commit="b" * 40,
        protocol_hash="c" * 64,
        manifest_sha256="d" * 64,
        q2_source_hashes={"fixture": "e" * 64},
    )


def test_btc_scoring_uses_frozen_member_taus_and_opposite_signal_veto() -> None:
    features = _features()
    probabilities = {
        "btcusdt:none:lstm:w55": FixedEstimator(
            [
                [0.10, 0.10, 0.80],
                [0.10, 0.10, 0.80],
                [0.80, 0.10, 0.10],
                [0.10, 0.20, 0.70],
                [0.10, 0.10, 0.80],
                [0.10, 0.10, 0.80],
            ]
        ),
        "btcusdt:none:svm_linear:w75": FixedEstimator(
            [
                [0.10, 0.10, 0.80],
                [0.10, 0.10, 0.80],
                [0.10, 0.10, 0.80],
                [0.80, 0.10, 0.10],
                [0.10, 0.10, 0.80],
                [0.10, 0.10, 0.80],
            ]
        ),
    }

    scored = score_btc_candidates(
        features,
        probabilities,
        PROTOCOL,
        start=START,
        end=START + pd.Timedelta(hours=1),
    )

    union = scored["btc_qualified_union_v1"]
    control = scored["btc_lstm_dz55"]
    assert union.index.min() == START
    assert union["signal"].tolist() == [0.0, -1.0, 1.0, 1.0]
    assert control["signal"].tolist() == [-1.0, 0.0, 1.0, 1.0]


def test_index_scoring_keeps_registered_single_and_arithmetic_all_nine_vote() -> None:
    features = _features()
    estimators = {}
    for model in (
        "logreg",
        "decision_tree",
        "random_forest",
        "svm_linear",
        "xgboost_balanced",
        "catboost_balanced",
        "mlp",
        "lstm",
        "gru",
    ):
        probability = [0.1, 0.2, 0.7]
        if model == "svm_linear":
            probability = [0.7, 0.2, 0.1]
        estimators[f"usa500:deberta_matched:{model}:w15"] = FixedEstimator(
            [probability]
        )

    scored = score_index_candidates(
        "usa500",
        features,
        estimators,
        PROTOCOL,
        start=START,
        end=START + pd.Timedelta(hours=1),
    )

    single = scored["usa500_best_single_deberta_svm"]
    vote = scored["usa500_deberta_soft_vote"]
    assert single["signal"].eq(-1.0).all()
    assert vote["signal"].eq(1.0).all()
    assert vote["p_long"].iloc[0] == pytest.approx((8 * 0.7 + 0.1) / 9)


def _index_bars() -> pd.DataFrame:
    index = pd.date_range(END - pd.Timedelta(hours=1), periods=4, freq="15min")
    return pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0, 103.0],
            "close": [100.5, 101.5, 102.5, 103.5],
            "complete_bar": True,
            "available_at": index + pd.Timedelta(minutes=15),
        },
        index=index,
    )


def test_index_terminal_decision_is_censored_and_cost_is_round_trip() -> None:
    bars = _index_bars()
    predictions = {
        "usa500_best_single_deberta_svm": pd.DataFrame(
            {"signal": [1.0, 1.0, 1.0, 1.0]}, index=bars.index
        ),
        "usa500_deberta_soft_vote": pd.DataFrame(
            {"signal": [0.0, 0.0, 0.0, 0.0]}, index=bars.index
        ),
    }

    results = replay_index_candidates(
        "usa500", bars, predictions, PROTOCOL, start=bars.index[0], end=END
    )
    primary = results["usa500_best_single_deberta_svm"]

    assert primary.audit.candidate_decisions == 4
    assert primary.audit.executed_decisions == 2
    assert primary.audit.terminal_censored_count == 2
    assert primary.ledger["exit_time"].lt(END).all()
    assert primary.ledger["cost_return"].eq(0.0002).all()
    assert np.allclose(
        primary.ledger["gross_return"] - primary.ledger["cost_return"],
        primary.ledger["net_return"],
    )


def _btc_market() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range(END - pd.Timedelta(hours=1), periods=4, freq="15min")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
        },
        index=index,
    )
    minute_index = pd.date_range(index[0], END, inclusive="left", freq="min")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.1, "low": 99.9, "close": 100.0},
        index=minute_index,
    )
    return bars, minute


def test_btc_replay_starts_flat_and_censors_the_last_two_incomplete_paths() -> None:
    bars, minute = _btc_market()
    predictions = {
        "btc_qualified_union_v1": pd.DataFrame(
            {"signal": [1.0, 1.0, 1.0, 1.0]}, index=bars.index
        ),
        "btc_lstm_dz55": pd.DataFrame(
            {"signal": [0.0, 0.0, 0.0, 0.0]}, index=bars.index
        ),
    }

    results = replay_btc_candidates(
        bars, minute, predictions, PROTOCOL, start=bars.index[0], end=END
    )
    union = results["btc_qualified_union_v1"]

    assert union.audit.started_flat is True
    assert union.audit.candidate_decisions == 4
    assert union.audit.executed_decisions == 1
    assert union.audit.terminal_censored_count == 2
    assert union.audit.nonoverlap_censored_count == 1
    assert union.ledger["signal_time"].ge(bars.index[0]).all()
    assert union.ledger["cost_return"].eq(0.001).all()


def test_replay_rejects_carry_in_and_unregistered_candidates() -> None:
    bars = _index_bars()
    carry = pd.DataFrame(
        {"signal": [1.0]}, index=[bars.index[0] - pd.Timedelta(minutes=15)]
    )
    with pytest.raises(ValueError, match="interval"):
        replay_index_candidates(
            "usa500",
            bars,
            {"usa500_best_single_deberta_svm": carry},
            PROTOCOL,
            start=bars.index[0],
            end=END,
        )
    with pytest.raises(ValueError, match="registered"):
        replay_index_candidates(
            "usa500",
            bars,
            {"agent_overlay": pd.DataFrame({"signal": [0.0] * 4}, index=bars.index)},
            PROTOCOL,
            start=bars.index[0],
            end=END,
        )


def test_open_requires_literal_authorization_and_exact_candidate_registry(
    tmp_path: Path,
) -> None:
    identity = _identity()
    with pytest.raises(PermissionError, match="authorization"):
        open_q2(
            identity,
            PROTOCOL,
            authorization="yes",
            run=lambda _: {},
            root=tmp_path,
        )
    assert not (tmp_path / "OPENED.json").exists()

    with pytest.raises(ValueError, match="registered"):
        open_q2(
            identity,
            PROTOCOL,
            authorization=OPEN_AUTHORIZATION,
            run=lambda _: {},
            root=tmp_path,
            candidate_ids=("agent_overlay",),
        )
    assert not (tmp_path / "OPENED.json").exists()


def test_open_failure_is_resume_only_with_the_exact_identity(tmp_path: Path) -> None:
    identity = _identity()

    def fail(_):
        raise RuntimeError("synthetic failure")

    with pytest.raises(RuntimeError, match="synthetic"):
        open_q2(
            identity,
            PROTOCOL,
            authorization=OPEN_AUTHORIZATION,
            run=fail,
            root=tmp_path,
        )
    assert (tmp_path / "OPENED.json").is_file()
    assert (tmp_path / identity.protocol_hash / "FAILED_AFTER_OPEN.json").is_file()

    with pytest.raises(PermissionError, match="identity"):
        resume_q2(_identity("f"), PROTOCOL, run=lambda _: {}, root=tmp_path)

    payload = b"complete"
    result_hash = hashlib.sha256(payload).hexdigest()
    result = resume_q2(
        identity,
        PROTOCOL,
        run=lambda _: {"fixture": result_hash},
        root=tmp_path,
    )
    assert result == {"fixture": result_hash}
    assert (tmp_path / identity.protocol_hash / "COMPLETE.json").is_file()
    with pytest.raises(PermissionError, match="already COMPLETE"):
        resume_q2(identity, PROTOCOL, run=lambda _: {}, root=tmp_path)


def test_candidate_checkpoint_is_write_once_hash_bound_and_resumable(
    tmp_path: Path,
) -> None:
    index = pd.date_range(START, periods=2, freq="15min")
    prediction = pd.DataFrame({"signal": [1.0, 0.0]}, index=index)
    ledger = pd.DataFrame(
        {
            "signal_time": [index[0]],
            "entry_time": [index[1]],
            "exit_time": [index[1] + pd.Timedelta(minutes=15)],
            "side": [1],
            "gross_return": [0.01],
            "cost_return": [0.001],
            "net_return": [0.009],
        }
    )
    result = ReplayResult(
        "fixture",
        ledger,
        pd.Series(0.0, index=index),
        ReplayAudit(1, 1, 0, 0),
    )

    _persist_candidate_checkpoint(tmp_path, "fixture", prediction, result)
    loaded_prediction, loaded_result = _load_candidate_checkpoint(
        tmp_path, "fixture"
    )

    pd.testing.assert_frame_equal(loaded_prediction, prediction, check_freq=False)
    pd.testing.assert_frame_equal(loaded_result.ledger, ledger, check_dtype=False)
    changed = prediction.assign(signal=[-1.0, 0.0])
    with pytest.raises(ValueError, match="immutable"):
        _persist_candidate_checkpoint(tmp_path, "fixture", changed, result)
    ledger_path = tmp_path / "ledgers" / "fixture.parquet"
    ledger_path.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hash"):
        _load_candidate_checkpoint(tmp_path, "fixture")


def test_opening_identity_rejects_a_lightweight_tag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner_module, "_git_text", lambda *args: "commit")
    with pytest.raises(ValueError, match="annotated"):
        runner_module._assert_annotated_tag("pre-lockbox-q2-2026")
    monkeypatch.setattr(runner_module, "_git_text", lambda *args: "tag")
    runner_module._assert_annotated_tag("pre-lockbox-q2-2026")


def test_final_writer_hashes_sentiment_inputs_and_never_overwrites_checkpoints(
    tmp_path: Path,
) -> None:
    predictions = {}
    results = {}
    timestamp = START
    for number, candidate in enumerate(PROTOCOL.candidates, start=1):
        prediction = pd.DataFrame({"signal": [1.0]}, index=[timestamp])
        ledger = pd.DataFrame(
            {
                "signal_time": [timestamp],
                "entry_time": [timestamp + pd.Timedelta(minutes=15)],
                "exit_time": [timestamp + pd.Timedelta(minutes=30)],
                "side": [1 if number % 2 else -1],
                "gross_return": [0.002 + number * 0.0001],
                "cost_return": [PROTOCOL.costs[candidate.stream].round_trip_bps / 10_000],
            }
        )
        ledger["net_return"] = ledger["gross_return"] - ledger["cost_return"]
        result = ReplayResult(
            candidate.candidate_id,
            ledger,
            pd.Series(dtype=float),
            ReplayAudit(1, 1, 0, 0),
        )
        predictions[candidate.candidate_id] = prediction
        results[candidate.candidate_id] = result
    prelockbox = tmp_path / "prelockbox.json"
    prelockbox.write_text("{}", encoding="utf-8")
    result_root = tmp_path / "result"
    sentiment = result_root / "sentiment_scores" / "scores_llm.parquet"
    sentiment.parent.mkdir(parents=True)
    sentiment.write_bytes(b"bound sentiment state")
    audit = {
        "interval": [START.isoformat(), END.isoformat()],
        "streams": {
            stream: {"warmup_max": (START - pd.Timedelta(minutes=15)).isoformat()}
            for stream in ("btcusdt", "usa500", "usatech")
        },
        "sentiment_outputs": {
            "fixture": sentiment.relative_to(result_root).as_posix()
        },
        "source_hashes_reverified": True,
    }

    first = _write_result_artifacts(
        result_root,
        PROTOCOL,
        predictions,
        results,
        audit,
        _identity(),
        prelockbox,
        intermediate_artifacts={"sentiment__fixture": sentiment},
    )
    second = _write_result_artifacts(
        result_root,
        PROTOCOL,
        predictions,
        results,
        audit,
        _identity(),
        prelockbox,
        intermediate_artifacts={"sentiment__fixture": sentiment},
    )
    final = json.loads((result_root / "manifest.json").read_text())

    assert first == second
    assert "sentiment__fixture" in final["artifact_hashes"]
    assert len(list((result_root / "checkpoints").glob("*.json"))) == 6
