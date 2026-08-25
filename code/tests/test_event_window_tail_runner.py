from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from experiments.run_event_window_tail_models import (
    FROZEN_J_ARTIFACTS,
    ProtocolMismatchError,
    TailStudyConfig,
    _ArtifactStore,
    _load_fold_checkpoint,
    _write_fold_checkpoint,
    _validate_loaded_input_fingerprint,
    load_frozen_j_artifacts,
    protocol_dict,
    run_event_window_tail_study,
)
from experiments.event_window_tail_oof import TailOOFResult


RUN_HASH = "2c6e19d5eae7ba9ffaaa"


def test_legacy_j_path_metadata_identity_can_only_be_deferred_explicitly():
    loaded = SimpleNamespace(input_fingerprint="same-content-different-path")
    frozen = SimpleNamespace(input_hash="legacy-path-size-mtime")

    with pytest.raises(ProtocolMismatchError, match="bounded inputs"):
        _validate_loaded_input_fingerprint(loaded, frozen, required=True)
    _validate_loaded_input_fingerprint(loaded, frozen, required=False)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    if path.suffix == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)


def _write_frozen_j_fixture(root: Path) -> Path:
    run_dir = root / RUN_HASH / "full"
    run_dir.mkdir(parents=True)
    manifest = pd.DataFrame(
        {
            "window_id": [f"w{row:05d}" for row in range(14_510)],
            "channel_episode_id": [f"e{row // 3:05d}" for row in range(14_510)],
            "side": ["long" if row % 2 == 0 else "short" for row in range(14_510)],
            "source_bar_time": pd.date_range("2021-01-01", periods=14_510, freq="5min", tz="UTC"),
            "window_start": pd.date_range("2021-01-01 00:05", periods=14_510, freq="5min", tz="UTC"),
            "window_end": pd.date_range("2021-01-01 01:05", periods=14_510, freq="5min", tz="UTC"),
        }
    )
    labels = pd.DataFrame(
        {
            "window_id": ["w00000", "w00000"],
            "channel_episode_id": ["e00000", "e00000"],
            "side": ["long", "long"],
            "step": [0, 1],
            "decision_time": pd.date_range("2022-01-01", periods=2, freq="5min", tz="UTC"),
            "label_start": pd.date_range("2022-01-01", periods=2, freq="5min", tz="UTC"),
            "label_end": pd.date_range("2022-01-01 00:10", periods=2, freq="5min", tz="UTC"),
            "risk_bps": [50.0, 50.0],
            "outcome": ["sl", "timeout"],
            "r_net": [-1.2, 0.1],
            "model_target_valid": [True, True],
        }
    )
    frames = {
        "window_manifest.parquet": manifest,
        "labels_rr2.parquet": labels,
        "labels_rr3.parquet": labels.assign(r_net=[-1.2, 0.2]),
        "oof_scores.parquet": labels[["window_id", "channel_episode_id", "side", "step", "decision_time"]].assign(fold_id="2022H1", score=0.1),
        "selected_trades.parquet": labels.iloc[[0]],
        "example_windows.parquet": manifest.iloc[:2],
    }
    for name, frame in frames.items():
        _write_frame(run_dir / name, frame)
    for name in FROZEN_J_ARTIFACTS:
        path = run_dir / name
        if path.exists() or name in {"protocol.json", "summary.json"}:
            continue
        _write_frame(path, pd.DataFrame({"value": [1]}))
    protocol = {
        "run_hash": RUN_HASH,
        "input_hash": "input-hash",
        "forward_or_lockbox_loaded": False,
        "max_loaded_timestamp": "2025-06-30 23:59:00+00:00",
        "read_end_exclusive": "2025-07-01 00:00:00+00:00",
    }
    summary = {
        "run_hash": RUN_HASH,
        "manifest_windows": 14_510,
        "attempted_trades": 1_448,
        "window_calendar_days": 1_642,
        "trades_per_calendar_day": 1_448 / 1_277,
        "total_net_r": -344.3721264672618,
        "forward_or_lockbox_loaded": False,
    }
    (run_dir / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    artifacts = {
        name: {"sha256": _sha256(run_dir / name), "size": (run_dir / name).stat().st_size}
        for name in FROZEN_J_ARTIFACTS
    }
    state = {
        "status": "complete",
        "run_hash": RUN_HASH,
        "protocol_hash": "protocol-hash",
        "input_hash": "input-hash",
        "source_hash": "source-hash",
        "artifacts": artifacts,
    }
    (run_dir / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    (root / "latest_dev.json").write_text(
        json.dumps(
            {
                "run_hash": RUN_HASH,
                "relative_path": f"{RUN_HASH}/full",
                "protocol_hash": "protocol-hash",
            }
        ),
        encoding="utf-8",
    )
    return root


def test_runner_rejects_any_stage_except_dev(tmp_path):
    with pytest.raises((ValueError, PermissionError), match="development-only"):
        run_event_window_tail_study(
            stage="forward", run_root=tmp_path, data_root=tmp_path
        )


def test_frozen_j_hash_and_run_id_are_exact(tmp_path):
    frozen = load_frozen_j_artifacts(_write_frozen_j_fixture(tmp_path))
    assert frozen.run_hash == RUN_HASH
    assert frozen.manifest_sha256 == _sha256(
        frozen.run_dir / "window_manifest.parquet"
    )
    assert len(frozen.manifest) == 14_510


def test_runner_rejects_manifest_or_rr_label_key_drift(tmp_path):
    root = _write_frozen_j_fixture(tmp_path)
    run_dir = root / RUN_HASH / "full"
    rr3 = pd.read_parquet(run_dir / "labels_rr3.parquet").iloc[:1]
    rr3.to_parquet(run_dir / "labels_rr3.parquet", index=False)
    state_path = run_dir / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["artifacts"]["labels_rr3.parquet"] = {
        "sha256": _sha256(run_dir / "labels_rr3.parquet"),
        "size": (run_dir / "labels_rr3.parquet").stat().st_size,
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ProtocolMismatchError, match="RR2.*RR3|RR3.*RR2"):
        load_frozen_j_artifacts(root)


def test_frozen_reader_rejects_corrupt_artifact(tmp_path):
    root = _write_frozen_j_fixture(tmp_path)
    path = root / RUN_HASH / "full" / "summary.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ProtocolMismatchError, match="hash"):
        load_frozen_j_artifacts(root)


def test_protocol_freezes_gru_four_model_contest_and_integer_frequency():
    protocol = protocol_dict(TailStudyConfig())
    assert protocol["models"] == ["logreg", "xgboost", "tcn", "gru"]
    assert protocol["reference_attempts"] == 1_448
    assert protocol["reference_calendar_days"] == 1_277
    assert protocol["primary_policy"] == "first calibrated EV >= 0"
    assert protocol["forward_or_lockbox_loaded"] is False


def test_corrupt_fold_checkpoint_is_rejected_without_touching_other_folds(tmp_path):
    frozen_root = _write_frozen_j_fixture(tmp_path / "frozen")
    frozen = load_frozen_j_artifacts(frozen_root)
    store = _ArtifactStore(
        tmp_path / "run",
        run_hash="run",
        protocol_hash="protocol",
        input_hash="input",
        source_hash="source",
    )
    scores = pd.DataFrame(
        {
            "model": ["gru"],
            "fold_id": ["2022H1"],
            "window_id": ["w"],
            "step": [0],
            "p_sl": [0.2],
            "p_tp": [0.3],
            "p_timeout": [0.5],
            "ev_score": [0.1],
            "matched_rate_threshold": [0.0],
        }
    )
    result = TailOOFResult(
        "gru",
        scores,
        pd.DataFrame([{"model": "gru", "fold_id": "2022H1"}]),
        pd.DataFrame([{"model": "gru", "fold_id": "2022H1"}]),
    )
    _write_fold_checkpoint(store, frozen, result)
    assert _load_fold_checkpoint(
        store, frozen, model="gru", fold_id="2022H1"
    ) is not None
    score_path = store.run_dir / "checkpoints" / "gru" / "2022H1.parquet"
    score_path.write_bytes(b"corrupt")
    assert _load_fold_checkpoint(
        store, frozen, model="gru", fold_id="2022H1"
    ) is None
