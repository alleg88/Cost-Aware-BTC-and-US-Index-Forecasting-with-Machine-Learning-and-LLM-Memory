import json
from pathlib import Path

import pandas as pd
import pytest

from experiments.run_event_window_adaptive_large_move import (
    AdaptiveMoveStudyConfig,
    READER_ARTIFACTS,
    _choose_model,
    _sha256,
    _validated_completed_summary,
    protocol_dict,
    run_adaptive_large_move_study,
)


def test_protocol_freezes_adaptive_target_and_seals_later_periods():
    protocol = protocol_dict(AdaptiveMoveStudyConfig())
    assert protocol["target_classes"] == ["NO_BIG_MOVE", "UP_BIG", "DOWN_BIG"]
    assert "75 + 0.5" in protocol["adaptive_barrier"]
    assert protocol["desired_trades_per_day"] == [1.0, 2.0]
    assert protocol["old_oof_selected_trades_reused"] is False
    assert protocol["forward_or_lockbox_loaded"] is False
    assert "one-minute Open" in protocol["barrier_reference"]
    assert "ddof=1" in protocol["volatility_estimator"]
    assert "lift > 1" in protocol["predictive_gate"]
    assert protocol["exact_direction_target"] == pytest.approx(0.70)


def test_runner_rejects_forward_before_loading_any_artifact(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("input loader must not run")

    monkeypatch.setattr(
        "experiments.run_event_window_adaptive_large_move.load_frozen_j_artifacts",
        forbidden,
    )
    with pytest.raises(ValueError, match="sealed"):
        run_adaptive_large_move_study(
            stage="forward",
            data_root=Path(tmp_path),
            frozen_j_root=Path(tmp_path),
            run_root=Path(tmp_path),
        )


def test_completed_cache_requires_every_recorded_artifact_hash(tmp_path):
    identity = {
        "run_hash": "run",
        "protocol_hash": "protocol",
        "source_hash": "source",
        "input_hash": "input",
    }
    summary = {**identity, "decision": "test"}
    protocol = {**identity, "stage": "dev", "smoke": False}
    frozen_protocol = {
        "frozen_j_run_hash": "j",
        "manifest_hash": "manifest",
        "frozen_input_hash": "frozen-input",
        "old_oof_selected_trades_reused": False,
    }
    for name in READER_ARTIFACTS:
        path = tmp_path / name
        if name == "summary.json":
            path.write_text(json.dumps(summary), encoding="utf-8")
        elif name == "protocol.json":
            path.write_text(json.dumps(protocol), encoding="utf-8")
        elif name == "frozen_protocol.json":
            path.write_text(json.dumps(frozen_protocol), encoding="utf-8")
        else:
            path.write_text(name, encoding="utf-8")
    frozen_protocol["labels_hash"] = _sha256(tmp_path / "adaptive_labels.parquet")
    (tmp_path / "frozen_protocol.json").write_text(
        json.dumps(frozen_protocol), encoding="utf-8"
    )
    artifacts = {
        name: {
            "size": (tmp_path / name).stat().st_size,
            "sha256": _sha256(tmp_path / name),
        }
        for name in READER_ARTIFACTS
    }
    state = {**identity, "status": "complete", "summary": summary, "artifacts": artifacts}
    (tmp_path / "run_state.json").write_text(json.dumps(state), encoding="utf-8")

    validated = _validated_completed_summary(
        tmp_path,
        identity,
        frozen_j_run_hash="j",
        frozen_manifest_hash="manifest",
        frozen_input_hash="frozen-input",
    )
    assert validated == summary

    (tmp_path / "policy_results.csv").write_text("tampered", encoding="utf-8")
    assert _validated_completed_summary(
        tmp_path,
        identity,
        frozen_j_run_hash="j",
        frozen_manifest_hash="manifest",
        frozen_input_hash="frozen-input",
    ) is None


def test_xgboost_requires_paired_economic_increment_over_logreg():
    registered = pd.DataFrame(
        {
            "frequency_pass": [False, True],
            "economic_pass": [False, True],
            "predictive_pass": [False, True],
        },
        index=["logreg", "xgboost"],
    )
    assert _choose_model(registered, {"delta_ci_low": -0.1}) is None
    assert _choose_model(registered, {"delta_ci_low": 0.1}) == "xgboost"
