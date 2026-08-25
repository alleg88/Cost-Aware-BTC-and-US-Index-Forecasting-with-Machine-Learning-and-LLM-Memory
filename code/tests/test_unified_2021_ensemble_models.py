from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.unified_2021_ensemble_data import (
    UnifiedDataset,
    build_causal_sequences,
)
from experiments.unified_2021_ensemble_models import (
    BinaryLSTMHead,
    SigmoidCalibrator,
    UnifiedModelConfig,
    fit_cross_calibrated_fold,
    fit_historical_snapshot,
    run_cross_calibrated_oof,
    sha256_keys,
)


def _tiny_dataset(rows: int = 96, features: int = 6) -> UnifiedDataset:
    rng = np.random.default_rng(42)
    tabular = rng.normal(size=(rows, features)).astype(np.float32)
    tabular[:, 0] += np.linspace(-1.0, 1.0, rows, dtype=np.float32)
    decision_time = pd.date_range(
        "2023-01-01 00:15", periods=rows, freq="15min", tz="UTC"
    )
    opportunity = (np.arange(rows) % 3 != 0).astype(np.int8)
    side = np.where((np.arange(rows) // 3) % 2 == 0, "long", "short")
    side = np.where(opportunity == 1, side, "tie")
    decisions = pd.DataFrame(
        {
            "row_key": [f"tiny-{position:03d}" for position in range(rows)],
            "decision_time": decision_time,
            "entry_time": decision_time + pd.Timedelta(minutes=1),
            "label_end": decision_time + pd.Timedelta(minutes=30),
            "path_complete": True,
            "opportunity": opportunity,
            "side": side,
            "side_eligible": opportunity.astype(bool),
            "adaptive_barrier_bps": 100.0,
        }
    )
    return UnifiedDataset(
        decisions=decisions,
        tabular=tabular,
        sequences=build_causal_sequences(tabular, sequence_length=4),
        feature_names=tuple(f"feature_{position}" for position in range(features)),
        economic_paths=pd.DataFrame(),
    )


def _tiny_manifest(dataset: UnifiedDataset) -> pd.DataFrame:
    roles = np.full(len(dataset.decisions), "outer_embargo", dtype=object)
    roles[:48] = "fit"
    roles[48:56] = "inner_embargo"
    roles[56:72] = "calibration"
    roles[80:] = "test"
    return pd.DataFrame(
        {
            "row_key": dataset.decisions["row_key"],
            "decision_time": dataset.decisions["decision_time"],
            "label_end": dataset.decisions["label_end"],
            "path_complete": True,
            "fold_id": 0,
            "position": np.arange(len(dataset.decisions)),
            "role": roles,
            "calibration_start_position": 56,
            "inner_embargo_start_position": 48,
            "test_start_position": 80,
        }
    )


def _tiny_config() -> UnifiedModelConfig:
    return UnifiedModelConfig(
        sequence_length=4,
        lstm_hidden_size=4,
        lstm_epochs=1,
        lstm_batch_size=32,
        xgb_estimators=5,
        xgb_depth=2,
        xgb_min_child_weight=1.0,
        n_jobs=1,
    )


def test_sigmoid_calibrator_uses_natural_prevalence_and_intercept():
    raw = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    target = np.array([0, 0, 0, 1, 1])

    calibrator = SigmoidCalibrator().fit(raw, target)
    probability = calibrator.predict_proba(raw)

    assert calibrator.intercept_.shape == (1,)
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert probability.tolist() == sorted(probability.tolist())


def test_side_head_fit_uses_only_opportunity_non_tie_labels():
    dataset = _tiny_dataset()
    manifest = _tiny_manifest(dataset)

    result = fit_cross_calibrated_fold(
        dataset, manifest, fold_id=0, config=_tiny_config()
    )

    fit_keys = set(manifest.loc[manifest["role"].eq("fit"), "row_key"])
    expected_side = dataset.decisions.loc[
        dataset.decisions["row_key"].isin(fit_keys)
        & dataset.decisions["opportunity"].eq(1)
        & dataset.decisions["side_eligible"].astype(bool),
        "row_key",
    ]
    side_audit = result.fit_audit.loc[result.fit_audit["head"].eq("side")]
    assert set(side_audit["fit_keys_sha256"]) == {sha256_keys(expected_side)}
    assert set(side_audit["fit_rows"]) == {len(expected_side)}
    assert not side_audit["calibration_overlap"].astype(bool).any()
    assert len(result.test_predictions) == 16


def test_all_models_emit_identical_oof_decision_keys():
    dataset = _tiny_dataset()

    result = run_cross_calibrated_oof(
        dataset, _tiny_manifest(dataset), _tiny_config()
    )

    expected_hash = sha256_keys(result.predictions["row_key"])
    for model in ("xgboost", "lstm", "svm_linear"):
        assert result.predictions[f"p_opportunity_{model}"].notna().all()
        assert result.predictions[f"p_long_{model}"].notna().all()
        actual_hash = result.model_key_audit.loc[
            result.model_key_audit["model"].eq(model), "keys_sha256"
        ].item()
        assert actual_hash == expected_hash
    assert set(result.calibration_metrics["head"]) == {"opportunity", "side"}
    assert set(result.calibration_metrics["model"]) == {
        "xgboost",
        "lstm",
        "svm_linear",
    }


def test_lstm_scoring_context_never_extends_past_scored_row():
    dataset = _tiny_dataset(rows=24)
    target = dataset.decisions["opportunity"].to_numpy(np.int8)
    model = BinaryLSTMHead(
        sequence_length=4,
        hidden_size=4,
        epochs=1,
        batch_size=16,
        seed=42,
    )
    model.fit(dataset.tabular[:12], target[:12])

    scored = model.predict_raw(
        dataset.tabular[12:16], context=dataset.tabular[9:12]
    )
    changed_future = dataset.tabular.copy()
    changed_future[16:] = 999999.0
    rescored = model.predict_raw(
        changed_future[12:16], context=changed_future[9:12]
    )

    np.testing.assert_allclose(scored, rescored, rtol=0.0, atol=0.0)


def test_historical_snapshot_is_purged_before_cutoff():
    dataset = _tiny_dataset(rows=160)
    cutoff = dataset.decisions.iloc[144]["decision_time"]

    snapshot = fit_historical_snapshot(dataset, cutoff, _tiny_config())

    assert snapshot.fit_max_label_end < snapshot.calibration_start
    assert snapshot.calibration_max_label_end < cutoff
    assert set(snapshot.models) == {
        (model, head)
        for model in ("xgboost", "lstm", "svm_linear")
        for head in ("opportunity", "side")
    }
    assert set(snapshot.calibrators) == set(snapshot.models)

