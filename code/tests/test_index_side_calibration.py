from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.index_side_calibration import (
    SIDE_THRESHOLDS,
    IndexPredictionSource,
    IndexSideCalibrationConfig,
    IndexSideCalibrationRunner,
    apply_side_calibrators,
    fit_side_calibrator,
    gate_side_predictions,
    select_arm_policy,
    select_model_policy,
    validate_selected_policies,
)
from experiments.index_replication import ARMS
from experiments.index_replication_protocol import (
    CALIBRATION_START,
    CUTOFF,
    FORWARD_END,
    FORWARD_START,
    MODEL_NAMES,
    SELECTION_END,
    SELECTION_START,
)


UTC = "UTC"
CODE_ROOT = Path(__file__).resolve().parents[1]
LIVE_CACHE = CODE_ROOT / "experiments" / "cache" / "index_side_calibration"


def _oof_predictions() -> pd.DataFrame:
    timestamps = pd.date_range("2024-02-01", periods=10, freq="15min", tz=UTC)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "y_true": [0, 1, 2, 0, 2, 1, 0, 2, 1, 2],
            "pred": [0, 1, 2, 0, 2, 1, 0, 2, 1, 2],
            "p_short": [0.72, 0.20, 0.08, 0.65, 0.12, 0.22, 0.61, 0.14, 0.18, 0.10],
            "p_flat": [0.20, 0.60, 0.22, 0.25, 0.18, 0.58, 0.28, 0.16, 0.64, 0.20],
            "p_long": [0.08, 0.20, 0.70, 0.10, 0.70, 0.20, 0.11, 0.70, 0.18, 0.70],
        }
    )


def _candidate(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "stream": "usa500",
        "arm": "selected_base",
        "model_name": "logreg",
        "width_bps": 10,
        "tau_short": 0.50,
        "tau_long": 0.50,
        "trades": 60,
        "n_long": 30,
        "n_short": 30,
        "positive_months": 3,
        "net_return": 0.02,
        "daily_sharpe": 1.0,
        "daily_sortino": 1.2,
    }
    row.update(overrides)
    return row


def test_threshold_grid_is_the_fixed_inclusive_five_point_grid():
    assert SIDE_THRESHOLDS == (
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
    )


def test_oof_sigmoid_calibrator_is_deterministic_and_serialisable():
    frame = _oof_predictions()

    first = fit_side_calibrator(frame, side_class=0)
    second = fit_side_calibrator(frame, side_class=0)

    assert first == second
    assert first["side"] == "SHORT"
    assert first["rows"] == 10
    assert first["positive_rows"] == 3
    for key in ("coefficient", "intercept", "raw_brier", "calibrated_brier"):
        assert np.isfinite(float(first[key]))
    assert len(str(first["source_sha256"])) == 64


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True), "duplicate"),
        (
            lambda frame: frame.assign(
                timestamp=pd.date_range("2025-01-01", periods=len(frame), freq="15min", tz=UTC)
            ),
            "2024 OOF",
        ),
        (lambda frame: frame.assign(y_true=1), "both binary classes"),
        (lambda frame: frame.assign(p_short=1.2), "probabilities"),
    ],
)
def test_oof_sigmoid_calibrator_fails_closed_on_invalid_sources(mutate, message):
    with pytest.raises(ValueError, match=message):
        fit_side_calibrator(mutate(_oof_predictions()), side_class=0)


def test_side_gating_preserves_argmax_direction_and_uses_separate_thresholds():
    frame = _oof_predictions().iloc[:5].copy()
    short = fit_side_calibrator(_oof_predictions(), side_class=0)
    long = fit_side_calibrator(_oof_predictions(), side_class=2)
    calibrated = apply_side_calibrators(frame, short, long)

    gated = gate_side_predictions(calibrated, tau_short=0.75, tau_long=0.25)

    assert gated.loc[frame["pred"].eq(1), "pred"].eq(1).all()
    assert set(gated["pred"]).issubset({0, 1, 2})
    assert not ((frame["pred"].eq(0)) & gated["pred"].eq(2)).any()
    assert not ((frame["pred"].eq(2)) & gated["pred"].eq(0)).any()
    assert gated.loc[frame["pred"].eq(2), "pred"].eq(2).any()
    assert gated["confidence"].between(0.0, 1.0).all()


def test_arm_policy_prefers_trade_count_only_after_every_h1_gate_passes():
    candidates = pd.DataFrame(
        [
            _candidate(tau_short=0.35, tau_long=0.35, trades=80, daily_sortino=0.8),
            _candidate(tau_short=0.40, tau_long=0.45, trades=75, daily_sortino=1.8),
            _candidate(tau_short=0.30, tau_long=0.30, trades=200, n_short=10),
        ]
    )

    winner = select_arm_policy(candidates)

    assert winner["h1_status"] == "Pass"
    assert winner["tau_short"] == pytest.approx(0.35)
    assert winner["tau_long"] == pytest.approx(0.35)


def test_arm_policy_fallback_uses_structural_shortfall_then_economics():
    candidates = pd.DataFrame(
        [
            _candidate(tau_short=0.30, n_short=10, daily_sortino=5.0, net_return=0.2),
            _candidate(tau_short=0.35, n_short=14, daily_sortino=0.5, net_return=0.01),
            _candidate(tau_short=0.40, n_short=14, daily_sortino=0.7, net_return=0.005),
        ]
    )

    winner = select_arm_policy(candidates)

    assert winner["h1_status"] == "Below gate"
    assert winner["structural_shortfall"] == 1
    assert winner["tau_short"] == pytest.approx(0.40)


def test_model_policy_uses_h1_status_then_frequency_and_fixed_arm_order():
    policies = pd.DataFrame(
        [
            _candidate(arm="deberta_matched", h1_status="Pass", trades=70),
            _candidate(arm="selected_base", h1_status="Pass", trades=70),
            _candidate(arm="deepseek_full", h1_status="Below gate", trades=500),
        ]
    )

    winner = select_model_policy(policies)

    assert winner["arm"] == "selected_base"


def test_config_refuses_any_boundary_other_than_the_sealed_q2_cutoff(tmp_path: Path):
    with pytest.raises(PermissionError, match="2026-04-01"):
        IndexSideCalibrationConfig.for_stream(
            "usa500",
            source_base=tmp_path / "source",
            output_base=tmp_path / "output",
            data_dir=tmp_path,
            end_exclusive=pd.Timestamp("2026-04-02", tz=UTC),
        )


def _selected_policies() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "arm": arm,
                "model_name": model_name,
                "width_bps": 5,
                "tau": 0.50,
                "eligible": False,
                "h1_execution_status": "diagnostic_only_no_eligible_policy",
            }
            for arm, model_name in product(ARMS, MODEL_NAMES)
        ]
    )


def test_selected_policy_source_requires_the_exact_all_nine_four_arm_grid():
    actual = validate_selected_policies(
        _selected_policies().sample(frac=1.0, random_state=9)
    )

    assert list(zip(actual["arm"], actual["model_name"])) == list(
        product(ARMS, MODEL_NAMES)
    )
    with pytest.raises(ValueError, match="exact 36-policy grid"):
        validate_selected_policies(_selected_policies().iloc[:-1])


def test_duplicate_cache_resolution_ignores_provenance_only_differences(tmp_path: Path):
    first = _oof_predictions().assign(
        train_start=pd.Timestamp("2023-01-01", tz=UTC),
        fingerprint="old-protocol",
    )
    second = _oof_predictions().assign(
        train_start=pd.Timestamp("2023-02-01", tz=UTC),
        fingerprint="current-protocol",
    )
    first_path = tmp_path / "w5_fold0_a.parquet"
    second_path = tmp_path / "w5_fold0_b.parquet"
    first.to_parquet(first_path, index=False)
    second.to_parquet(second_path, index=False)

    resolved = IndexPredictionSource._resolve_equivalent_group(
        [second_path, first_path], "fold0"
    )

    assert resolved == first_path


def test_duplicate_cache_resolution_rejects_different_probabilities(tmp_path: Path):
    first_path = tmp_path / "w5_fold0_a.parquet"
    second_path = tmp_path / "w5_fold0_b.parquet"
    _oof_predictions().to_parquet(first_path, index=False)
    _oof_predictions().assign(p_short=lambda frame: frame["p_short"] + 0.01).to_parquet(
        second_path, index=False
    )

    with pytest.raises(ValueError, match="ambiguous non-equivalent"):
        IndexPredictionSource._resolve_equivalent_group(
            [first_path, second_path], "fold0"
        )


def test_duplicate_cache_resolution_uses_exact_frozen_fingerprint(tmp_path: Path):
    first_path = tmp_path / "w5_fold0_a.parquet"
    second_path = tmp_path / "w5_fold0_b.parquet"
    _oof_predictions().assign(fingerprint="old" * 16).to_parquet(
        first_path, index=False
    )
    _oof_predictions().assign(
        p_short=lambda frame: frame["p_short"] + 0.01,
        fingerprint="current" * 8,
    ).to_parquet(second_path, index=False)

    resolved = IndexPredictionSource._resolve_equivalent_group(
        [first_path, second_path],
        "fold0",
        expected_fingerprint="current" * 8,
    )

    assert resolved == second_path


def _prediction_segment(
    starts: pd.DatetimeIndex,
    *,
    signals_per_month: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    bar_frames: list[pd.DataFrame] = []
    for month_start in starts:
        index = pd.date_range(
            month_start + pd.Timedelta(hours=10),
            periods=signals_per_month + 1,
            freq="15min",
        )
        sides = np.resize(np.array([0, 2], dtype=int), signals_per_month)
        prediction_rows.extend(
            {
                "timestamp": timestamp,
                "y_true": int(side),
                "pred": int(side),
                "confidence": 0.80,
                "p_short": 0.80 if side == 0 else 0.05,
                "p_flat": 0.15,
                "p_long": 0.80 if side == 2 else 0.05,
            }
            for timestamp, side in zip(index[:-1], sides)
        )
        closes = np.full(len(index), 100.0)
        for offset, side in enumerate(sides, start=1):
            closes[offset] = 99.9 if side == 0 else 100.1
        bar_frames.append(
            pd.DataFrame(
                {
                    "open": 100.0,
                    "high": np.maximum(100.0, closes),
                    "low": np.minimum(100.0, closes),
                    "close": closes,
                    "volume": 1.0,
                    "available_at": index + pd.Timedelta(minutes=15),
                    "complete_bar": True,
                },
                index=index,
            )
        )
    return pd.DataFrame(prediction_rows), pd.concat(bar_frames).sort_index()


class _FakePredictionSource:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str, int]] = []
        self.selection = _oof_predictions()
        self.h1, h1_bars = _prediction_segment(
            pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS", inclusive="left"),
            signals_per_month=11,
        )
        self.forward, forward_bars = _prediction_segment(
            pd.date_range(FORWARD_START, FORWARD_END, freq="MS", inclusive="left"),
            signals_per_month=5,
        )
        self.bars = pd.concat([h1_bars, forward_bars]).sort_index()

    def identity(self) -> dict[str, object]:
        return {
            "source_protocol_hash": "1" * 64,
            "selected_policy_sha256": "2" * 64,
            "prediction_catalog_sha256": "3" * 64,
            "selection": [SELECTION_START.isoformat(), SELECTION_END.isoformat()],
            "calibration": [CALIBRATION_START.isoformat(), FORWARD_START.isoformat()],
            "forward": [FORWARD_START.isoformat(), FORWARD_END.isoformat()],
            "q2_loaded": False,
        }

    def selected_policies(self) -> pd.DataFrame:
        return _selected_policies()

    def load_predictions(
        self, stage: str, arm: str, model_name: str, width_bps: int
    ) -> pd.DataFrame:
        self.calls.append((stage, arm, model_name, int(width_bps)))
        frame = {
            "selection": self.selection,
            "calibration": self.h1,
            "forward": self.forward,
        }[stage]
        return frame.assign(
            arm=arm,
            model_name=model_name,
            width_bps=int(width_bps),
            fit_id=f"test::{stage}::{arm}::{model_name}::{width_bps}",
        ).copy()


def _runner_fixture(tmp_path: Path):
    source = _FakePredictionSource()
    config = IndexSideCalibrationConfig.for_stream(
        "usa500",
        source_base=tmp_path / "source",
        output_base=tmp_path / "output",
        data_dir=tmp_path / "data",
    )
    runner = IndexSideCalibrationRunner(config, source=source, bars=source.bars)
    return runner, config, source


def test_runner_materialises_complete_causal_artifacts_and_resumes(tmp_path: Path):
    runner, config, source = _runner_fixture(tmp_path)

    first = runner.run()

    assert first["calibrator_rows"] == 72
    assert first["h1_candidate_rows"] == 5_184
    assert first["h1_arm_policy_rows"] == 36
    assert first["h1_model_policy_rows"] == 9
    assert first["forward_arm_rows"] == 36
    assert first["forward_rows"] == 9
    assert first["resumed_forward_policies"] == 0
    assert first["q2_loaded"] is False
    assert pd.Timestamp(first["max_prediction_timestamp"]) < CUTOFF
    assert len(source.calls) == 108

    calibrators = pd.read_parquet(config.output_root / "calibrators.parquet")
    candidates = pd.read_parquet(config.output_root / "h1_candidates.parquet")
    arm_policies = pd.read_parquet(config.output_root / "h1_arm_policies.parquet")
    model_policies = pd.read_parquet(config.output_root / "h1_model_policies.parquet")
    forward_arm = pd.read_parquet(config.output_root / "forward_arm_summary.parquet")
    forward = pd.read_parquet(config.output_root / "forward_summary.parquet")
    assert calibrators.groupby(["arm", "model_name"])["side"].nunique().eq(2).all()
    assert len(candidates) == 36 * len(SIDE_THRESHOLDS) ** 2
    assert len(arm_policies) == 36
    assert len(model_policies) == 9
    assert len(forward_arm) == 36
    assert len(forward) == 9
    assert not forward.isna().any().any()
    assert np.isfinite(forward.select_dtypes(include="number").to_numpy()).all()
    assert forward["n_long"].gt(0).all() and forward["n_short"].gt(0).all()
    assert len(list((config.output_root / "forward").glob("*.json"))) == 36
    assert len(list((config.output_root / "forward_ledgers").glob("*.parquet"))) == 72

    source.calls.clear()
    resumed = runner.run()
    assert resumed["resumed_forward_policies"] == 36
    assert source.calls == []


def test_runner_resume_fails_closed_when_a_ledger_hash_changes(tmp_path: Path):
    runner, config, _source = _runner_fixture(tmp_path)
    runner.run()
    ledger_path = next((config.output_root / "forward_ledgers").glob("*.parquet"))
    ledger = pd.read_parquet(ledger_path)
    ledger.assign(net_return=ledger["net_return"] + 0.01).to_parquet(
        ledger_path, index=False
    )

    with pytest.raises(ValueError, match="artifact hash changed"):
        runner.run()


def test_canonical_parity_accepts_identical_empty_ledgers(tmp_path: Path):
    runner, _config, source = _runner_fixture(tmp_path)
    calibrated = source.h1.assign(
        calibrated_short=0.10,
        calibrated_long=0.10,
    )
    events = runner._dense_events(
        calibrated, start=CALIBRATION_START, end=FORWARD_START
    )
    filtered = runner._filter_events(events, tau_short=0.90, tau_long=0.90)

    per_bar = runner._assert_canonical_parity(
        calibrated,
        filtered,
        start=CALIBRATION_START,
        end=FORWARD_START,
        tau_short=0.90,
        tau_long=0.90,
    )

    assert filtered.empty
    assert per_bar.eq(0.0).all()


@pytest.mark.parametrize(
    ("stream", "improved_brier"),
    (("usa500", 72), ("usatech", 70)),
)
def test_completed_side_calibration_artifacts_reconcile(
    stream: str, improved_brier: int
):
    root = LIVE_CACHE / stream
    result = json.loads((root / "result.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    calibrators = pd.read_parquet(root / "calibrators.parquet")
    candidates = pd.read_parquet(root / "h1_candidates.parquet")
    arm_policies = pd.read_parquet(root / "h1_arm_policies.parquet")
    model_policies = pd.read_parquet(root / "h1_model_policies.parquet")
    forward_arm = pd.read_parquet(root / "forward_arm_summary.parquet")
    forward = pd.read_parquet(root / "forward_summary.parquet")

    assert result["q2_loaded"] is False
    assert result["resumed_forward_policies"] == 36
    assert result["calibrator_rows"] == 72
    assert result["h1_candidate_rows"] == 5_184
    assert result["h1_arm_policy_rows"] == 36
    assert result["h1_model_policy_rows"] == 9
    assert result["forward_arm_rows"] == 36
    assert result["forward_rows"] == 9
    assert pd.Timestamp(result["max_prediction_timestamp"]) < CUTOFF
    assert protocol["q2_loaded"] is False
    assert protocol["source_identity"]["q2_loaded"] is False
    assert protocol["source_identity"]["prediction_file_count"] == 432
    assert manifest["protocol_hash"] == protocol["protocol_hash"]
    assert manifest["q2_loaded"] is False
    assert len(manifest["artifacts"]) == 115
    for relative, expected_hash in manifest["artifacts"].items():
        path = root / relative
        assert path.exists()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_hash

    assert len(calibrators) == 72
    assert calibrators.groupby(["arm", "model_name"])["side"].nunique().eq(2).all()
    assert set(calibrators["arm"]) == set(ARMS)
    assert set(calibrators["model_name"]) == set(MODEL_NAMES)
    assert int(calibrators["calibrated_brier"].lt(calibrators["raw_brier"]).sum()) == improved_brier
    assert len(candidates) == 36 * len(SIDE_THRESHOLDS) ** 2
    assert set(candidates["tau_short"]) == set(SIDE_THRESHOLDS)
    assert set(candidates["tau_long"]) == set(SIDE_THRESHOLDS)
    assert len(arm_policies) == 36 and len(model_policies) == 9
    assert model_policies["model_name"].nunique() == 9
    assert len(forward_arm) == 36 and len(forward) == 9
    assert not forward.isna().any().any()
    assert np.isfinite(forward.select_dtypes(include="number").to_numpy()).all()
    assert int(forward["promising"].sum()) == 0
    assert len(list((root / "forward").glob("*.json"))) == 36
    ledgers = list((root / "forward_ledgers").glob("*.parquet"))
    assert len(ledgers) == 72
    for path in ledgers:
        frame = pd.read_parquet(path)
        if "timestamp" in frame:
            assert pd.to_datetime(frame["timestamp"], utc=True).lt(CUTOFF).all()
        if "entry_time" in frame and len(frame):
            assert pd.to_datetime(frame["entry_time"], utc=True).lt(CUTOFF).all()
            assert pd.to_datetime(frame["exit_time"], utc=True).le(CUTOFF).all()
