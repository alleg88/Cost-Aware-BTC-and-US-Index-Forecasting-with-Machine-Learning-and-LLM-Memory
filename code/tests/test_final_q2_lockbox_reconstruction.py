from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.final_q2_lockbox_contract import load_lockbox_protocol
from experiments.final_q2_lockbox_reconstruction import (
    PROBABILITY_COLUMNS,
    ReconstructionSpec,
    assert_panel_equivalence,
    assert_threshold_decision_equivalence,
    assert_union_decision_equivalence,
    assert_soft_vote_decision_equivalence,
    build_index_reconstruction_dataset,
    fit_panel_estimator,
    load_serialized_estimator,
    reconstruction_specs,
    reconstruct_from_frames,
    reconstruct_btc_spec,
    reconstruct_all,
    reconstruct_index_spec,
    serialize_estimator,
)
from experiments.run_catboost_matched_ablation import _read_before
from experiments.run_walkforward import _read_parquet_before


CODE_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = CODE_ROOT / "configs" / "final_q2_lockbox_protocol.json"


def _panel(*, probability_shift: float = 0.0) -> pd.DataFrame:
    timestamp = pd.date_range("2025-07-01", periods=3, freq="15min", tz="UTC")
    short = np.array([0.7, 0.1, 0.2])
    flat = np.array([0.2, 0.8, 0.2])
    long = np.array([0.1, 0.1, 0.6])
    long = long + probability_shift
    flat = flat - probability_shift
    probability = np.column_stack([short, flat, long])
    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "y_true": [0, 1, 2],
            "pred": probability.argmax(axis=1),
            "confidence": probability.max(axis=1),
            "p_short": probability[:, 0],
            "p_flat": probability[:, 1],
            "p_long": probability[:, 2],
        }
    )


def test_panel_equivalence_requires_exact_keys_classes_and_bounded_probabilities() -> None:
    reference = _panel()
    rebuilt = _panel(probability_shift=5e-7)

    audit = assert_panel_equivalence(reference, rebuilt)
    assert audit.rows == 3
    assert audit.max_probability_error == pytest.approx(5e-7)

    with pytest.raises(ValueError, match="class"):
        assert_panel_equivalence(reference, rebuilt.assign(pred=1))
    with pytest.raises(ValueError, match="probability"):
        assert_panel_equivalence(reference, _panel(probability_shift=2e-6))
    with pytest.raises(ValueError, match="timestamp"):
        assert_panel_equivalence(reference, rebuilt.iloc[::-1].reset_index(drop=True))


def test_panel_equivalence_rejects_non_normalized_or_nonfinite_probabilities() -> None:
    reference = _panel()
    bad_sum = _panel()
    bad_sum.loc[0, "p_long"] += 1e-4
    with pytest.raises(ValueError, match="normalized"):
        assert_panel_equivalence(reference, bad_sum)
    bad_finite = _panel()
    bad_finite.loc[0, "p_short"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        assert_panel_equivalence(reference, bad_finite)


def test_btc_exception_still_enforces_mean_error_and_exact_decisions() -> None:
    reference = _panel()
    rebuilt = _panel(probability_shift=9e-7)

    with pytest.raises(ValueError, match="mean probability"):
        assert_panel_equivalence(
            reference,
            rebuilt,
            probability_atol=2e-5,
            mean_probability_atol=5e-7,
    )
    assert_threshold_decision_equivalence(reference, rebuilt, tau=0.75)
    changed = rebuilt.copy()
    changed.loc[0, "confidence"] = 0.9
    with pytest.raises(ValueError, match="decision"):
        assert_threshold_decision_equivalence(reference, changed, tau=0.75)


def test_qualified_union_signal_vector_must_match_exactly() -> None:
    lstm_reference = _panel()
    svm_reference = _panel()
    lstm_rebuilt = _panel(probability_shift=5e-7)
    svm_rebuilt = _panel()

    assert_union_decision_equivalence(
        lstm_reference,
        svm_reference,
        lstm_rebuilt,
        svm_rebuilt,
        lstm_tau=0.75,
        svm_tau=0.0,
    )
    changed = lstm_rebuilt.copy()
    changed.loc[0, ["pred", "confidence"]] = [2, 0.9]
    with pytest.raises(ValueError, match="Union"):
        assert_union_decision_equivalence(
            lstm_reference,
            svm_reference,
            changed,
            svm_rebuilt,
            lstm_tau=0.75,
            svm_tau=0.0,
        )


def test_soft_vote_decision_vector_must_match_exactly() -> None:
    reference = {"a": _panel(), "b": _panel()}
    rebuilt = {"a": _panel(probability_shift=5e-7), "b": _panel()}

    assert_soft_vote_decision_equivalence(reference, rebuilt, tau=0.55)
    changed = {key: value.copy() for key, value in rebuilt.items()}
    changed["a"].loc[0, ["p_short", "p_flat", "p_long", "pred", "confidence"]] = [
        0.1,
        0.8,
        0.1,
        1,
        0.8,
    ]
    with pytest.raises(ValueError, match="soft-vote"):
        assert_soft_vote_decision_equivalence(reference, changed, tau=0.55)


def test_reconstruction_registry_has_exactly_21_unique_fits() -> None:
    protocol = load_lockbox_protocol(PROTOCOL_PATH)
    specs = reconstruction_specs(protocol, code_root=CODE_ROOT)

    assert len(specs) == 21
    assert len({spec.fit_key for spec in specs}) == 21
    assert sum(spec.stream == "btcusdt" for spec in specs) == 2
    assert sum(spec.stream == "usa500" for spec in specs) == 9
    assert sum(spec.stream == "usatech" for spec in specs) == 10
    assert any(
        spec.stream == "usatech"
        and spec.model_name == "lstm"
        and spec.width_bps == 15
        for spec in specs
    )
    special = next(spec for spec in specs if spec.fit_key == "btcusdt:none:lstm:w55")
    assert special.probability_atol == 2e-5
    assert special.mean_probability_atol == 5e-7
    assert special.torch_threads == 16
    assert all(
        spec.probability_atol == 1e-6
        for spec in specs
        if spec.fit_key != special.fit_key
    )


def test_reconstruction_specs_bind_existing_reference_panel_hashes() -> None:
    specs = reconstruction_specs(
        load_lockbox_protocol(PROTOCOL_PATH), code_root=CODE_ROOT
    )

    for spec in specs:
        assert spec.reference_panel.is_file()
        assert len(spec.reference_sha256) == 64
        assert spec.fit_cutoff == pd.Timestamp("2025-07-01T00:00:00Z")
        assert spec.test_start == pd.Timestamp("2025-07-01T00:00:00Z")
        assert spec.test_end == pd.Timestamp("2026-04-01T00:00:00Z")


class _RecordingModel:
    classes_ = np.array([0, 1, 2])

    def __init__(self):
        self.fit_index = None

    def fit(self, X, y, sample_weight=None):
        self.fit_index = X.index.copy()
        return self

    def predict_proba(self, X):
        output = np.tile(np.array([[0.7, 0.2, 0.1]]), (len(X), 1))
        return output


def test_fit_panel_estimator_never_fits_at_or_after_cutoff() -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    X = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], index=index)
    fit_cutoff = index[5]
    test_end = index[-1] + pd.Timedelta(minutes=15)

    model, prediction = fit_panel_estimator(
        X=X,
        y=y,
        model_factory=lambda: _RecordingModel(),
        fit_start=index[0],
        fit_cutoff=fit_cutoff,
        test_start=fit_cutoff,
        test_end=test_end,
        label_tail_trim=1,
    )

    assert model.fit_index.max() < fit_cutoff
    assert prediction["timestamp"].min() == fit_cutoff
    assert tuple(column for column in PROBABILITY_COLUMNS if column in prediction) == PROBABILITY_COLUMNS


def test_serialized_estimator_round_trips_with_hash_manifest(tmp_path: Path) -> None:
    model = _RecordingModel()
    path, digest = serialize_estimator(model, tmp_path / "estimator.joblib")
    loaded = load_serialized_estimator(path, expected_sha256=digest)

    assert isinstance(loaded, _RecordingModel)
    with pytest.raises(ValueError, match="hash"):
        load_serialized_estimator(path, expected_sha256="0" * 64)


def test_reconstruct_from_frames_writes_hash_bound_pre_q2_artifacts(
    tmp_path: Path,
) -> None:
    index = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    X = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], index=index)
    fit_cutoff = index[5]
    test_end = index[-1] + pd.Timedelta(minutes=15)
    reference = pd.DataFrame(
        {
            "timestamp": index[5:],
            "y_true": y.loc[index[5:]].to_numpy(),
            "pred": [0, 0, 0],
            "confidence": [0.7, 0.7, 0.7],
            "p_short": [0.7, 0.7, 0.7],
            "p_flat": [0.2, 0.2, 0.2],
            "p_long": [0.1, 0.1, 0.1],
        }
    )
    reference_path = tmp_path / "reference.parquet"
    reference.to_parquet(reference_path, index=False)
    reference_hash = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    spec = ReconstructionSpec(
        stream="fixture",
        arm="none",
        model_name="fixture_model",
        width_bps=5,
        fit_start=index[0],
        fit_cutoff=fit_cutoff,
        test_start=fit_cutoff,
        test_end=test_end,
        reference_panel=reference_path,
        reference_sha256=reference_hash,
    )

    result = reconstruct_from_frames(
        spec,
        X=X,
        y=y,
        model_factory=lambda: _RecordingModel(),
        output_root=tmp_path / "models",
        label_tail_trim=1,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.estimator_path.is_file()
    assert result.manifest_path.is_file()
    assert result.rebuilt_panel_path.is_file()
    assert result.rebuilt_panel_sha256 == hashlib.sha256(
        result.rebuilt_panel_path.read_bytes()
    ).hexdigest()
    assert result.audit.max_probability_error <= 1e-15
    assert manifest["q2_decoded"] is False
    assert manifest["fit_key"] == spec.fit_key
    assert manifest["serialized_estimator_sha256"] == result.estimator_sha256
    assert manifest["rebuilt_panel_sha256"] == result.rebuilt_panel_sha256
    assert manifest["reference_panel_sha256"] == reference_hash


def _fixture_spec_and_data(tmp_path: Path):
    index = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    X = pd.DataFrame({"x": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series([0, 1, 2, 0, 1, 2, 0, 1], index=index)
    fit_cutoff = index[5]
    test_end = index[-1] + pd.Timedelta(minutes=15)
    reference = pd.DataFrame(
        {
            "timestamp": index[5:],
            "y_true": y.loc[index[5:]].to_numpy(),
            "pred": [0, 0, 0],
            "confidence": [0.7, 0.7, 0.7],
            "p_short": [0.7, 0.7, 0.7],
            "p_flat": [0.2, 0.2, 0.2],
            "p_long": [0.1, 0.1, 0.1],
        }
    )
    reference_path = tmp_path / "reference_wrapper.parquet"
    reference.to_parquet(reference_path, index=False)
    spec = ReconstructionSpec(
        stream="fixture",
        arm="none",
        model_name="fixture_model",
        width_bps=5,
        fit_start=index[0],
        fit_cutoff=fit_cutoff,
        test_start=fit_cutoff,
        test_end=test_end,
        reference_panel=reference_path,
        reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
    )
    return spec, X, y


def test_btc_wrapper_uses_regime_weighted_pre_q2_fit(tmp_path: Path) -> None:
    spec, X, y = _fixture_spec_and_data(tmp_path)
    prepared = SimpleNamespace(
        features={5: (X, y)},
        regimes=pd.Series("bull", index=X.index),
    )

    result = reconstruct_btc_spec(
        spec,
        prepared=prepared,
        model_factory=lambda: _RecordingModel(),
        output_root=tmp_path / "btc_models",
    )

    assert result.audit.rows == 3
    assert load_serialized_estimator(
        result.estimator_path, expected_sha256=result.estimator_sha256
    ).fit_index.max() < spec.fit_cutoff


def test_index_wrapper_uses_runner_dataset_and_one_label_tail(tmp_path: Path) -> None:
    spec, X, y = _fixture_spec_and_data(tmp_path)

    class Runner:
        def dataset(self, arm, width_bps):
            assert (arm, width_bps) == (spec.arm, spec.width_bps)
            return X, y

    result = reconstruct_index_spec(
        spec,
        runner=Runner(),
        model_factory=lambda: _RecordingModel(),
        output_root=tmp_path / "index_models",
    )

    assert result.audit.rows == 3


def test_index_reconstruction_dataset_uses_frozen_price_vix_without_reopening_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "close": [100.0, 101.0, 100.0, 102.0],
            "available_at": index + pd.Timedelta(minutes=15),
        },
        index=index,
    )
    price = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0]}, index=index)
    price_vix = price.assign(vix=0.5)

    class Runner:
        def __init__(self):
            self.config = SimpleNamespace(stream="usa500")
            self.bars = bars

        def _price_and_vix_frames(self):
            return price, price_vix

        def dataset(self, arm, width_bps):
            raise AssertionError("stale VIX gate was reopened")

    monkeypatch.setattr(
        "experiments.final_q2_lockbox_reconstruction.build_matched_index_features",
        lambda stream, bar_index, scorer: pd.DataFrame(
            {"sent": np.arange(len(bar_index), dtype=float)}, index=bar_index
        ),
    )
    spec, _, _ = _fixture_spec_and_data(tmp_path)
    spec = replace(spec, stream="usa500", arm="deberta_matched", width_bps=5)

    X, y = build_index_reconstruction_dataset(Runner(), spec)

    assert list(X.columns) == ["price", "vix", "sent"]
    assert y.index.equals(X.index)
    assert set(y.unique()).issubset({0, 1, 2})


def test_reconstruct_all_initializes_index_runners_only_in_lockbox_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_output_bases: list[Path] = []

    class FakeConfig:
        @classmethod
        def for_stream(cls, stream, *, output_base):
            observed_output_bases.append(Path(output_base))
            return SimpleNamespace(stream=stream)

    class FakeRunner:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(
        "experiments.index_replication.IndexReplicationConfig", FakeConfig
    )
    monkeypatch.setattr(
        "experiments.index_replication.IndexReplicationRunner", FakeRunner
    )
    monkeypatch.setattr(
        "experiments.all_model_sentiment_raw.prepare_arm", lambda arm: object()
    )
    monkeypatch.setattr(
        "experiments.final_q2_lockbox_reconstruction.reconstruction_specs",
        lambda protocol, code_root: (),
    )
    monkeypatch.setattr(
        "experiments.final_q2_lockbox_reconstruction.validate_registered_candidate_decisions",
        lambda protocol, results: {},
    )
    monkeypatch.setattr(
        "experiments.final_q2_lockbox_state.GLOBAL_SENTINEL_PATH",
        tmp_path / "absent_OPENED.json",
    )

    output_root = tmp_path / "reconstructed_models"
    reconstruct_all(protocol_path=PROTOCOL_PATH, output_root=output_root)

    assert observed_output_bases == [
        tmp_path / "runner_scratch",
        tmp_path / "runner_scratch",
    ]


@pytest.mark.parametrize("reader", [_read_before, _read_parquet_before])
def test_pre_q2_readers_push_cutoff_into_parquet_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader
) -> None:
    path = tmp_path / "broad.parquet"
    index = pd.DatetimeIndex(
        ["2026-03-31T23:45:00Z", "2026-04-01T00:00:00Z"], name="timestamp"
    )
    pd.DataFrame({"value": [1.0, 2.0]}, index=index).to_parquet(path)
    original = pd.read_parquet
    observed = {}

    def capture(*args, **kwargs):
        observed["filters"] = kwargs.get("filters")
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", capture)
    loaded = reader(path, end_exclusive=pd.Timestamp("2026-04-01T00:00:00Z"))

    assert observed["filters"] == [
        ("timestamp", "<", pd.Timestamp("2026-04-01T00:00:00Z").to_pydatetime())
    ]
    assert loaded.index.tolist() == [pd.Timestamp("2026-03-31T23:45:00Z")]
