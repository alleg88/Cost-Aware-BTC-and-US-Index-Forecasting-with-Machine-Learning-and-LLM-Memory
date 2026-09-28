"""Compact Rebuild accepts new fits without weakening input integrity."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import pandas as pd

from experiments.channel_rebuild_contract import handoff_run_hash, recomputed_handoffs
from experiments.run_event_window_tail_models import ProtocolMismatchError, load_frozen_j_artifacts
from test_event_window_tail_runner import RUN_HASH, _write_frozen_j_fixture


MODE = "MSC_REBUILD_CHANNEL_HANDOFFS"


def _json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _new_j(root: Path) -> tuple[Path, Path]:
    _write_frozen_j_fixture(root)
    new_hash = "a" * 20
    old = root / RUN_HASH
    new = root / new_hash
    assert old.resolve().is_relative_to(root.resolve()) and new.resolve().is_relative_to(root.resolve())
    old.rename(new)
    run = new / "full"
    state = json.loads((run / "run_state.json").read_text())
    state["run_hash"] = new_hash
    for name in ("protocol.json", "summary.json"):
        payload = json.loads((run / name).read_text())
        payload["run_hash"] = new_hash
        if name == "summary.json":
            payload["attempted_trades"] = 1
        _json(run / name, payload)
        state["artifacts"][name] = {
            "sha256": hashlib.sha256((run / name).read_bytes()).hexdigest(),
            "size": (run / name).stat().st_size,
        }
    _json(run / "run_state.json", state)
    _json(root / "latest_dev.json", {"run_hash": new_hash, "protocol_hash": "protocol-hash", "relative_path": f"{new_hash}/full"})
    return root, run


def test_recomputed_handoffs_require_explicit_recipe_opt_in(monkeypatch, tmp_path):
    monkeypatch.delenv(MODE, raising=False)
    assert not recomputed_handoffs()
    assert handoff_run_hash(tmp_path, RUN_HASH) == RUN_HASH
    monkeypatch.setenv(MODE, "1")
    assert not recomputed_handoffs()
    monkeypatch.setenv(MODE, "recomputed")
    assert recomputed_handoffs()
    with pytest.raises(FileNotFoundError):
        handoff_run_hash(tmp_path, RUN_HASH)


@pytest.mark.parametrize("value", ("../../escape", "", "z" * 20))
def test_recomputed_pointer_rejects_invalid_identity(monkeypatch, tmp_path, value):
    monkeypatch.setenv(MODE, "recomputed")
    _json(tmp_path / "latest_dev.json", {"run_hash": value, "relative_path": f"{value}/full"})
    with pytest.raises(ValueError, match="identity"):
        handoff_run_hash(tmp_path, RUN_HASH)


def test_recomputed_j_accepts_new_run_and_prediction_count(monkeypatch, tmp_path):
    root, _ = _new_j(tmp_path)
    monkeypatch.delenv(MODE, raising=False)
    with pytest.raises(ProtocolMismatchError, match="run hash changed"):
        load_frozen_j_artifacts(root)
    monkeypatch.setenv(MODE, "recomputed")
    actual = load_frozen_j_artifacts(root)
    assert actual.run_hash == "a" * 20
    assert actual.summary["attempted_trades"] == len(actual.selected_trades) == 1


def test_recomputed_j_still_rejects_changed_artifact_bytes(monkeypatch, tmp_path):
    root, run = _new_j(tmp_path)
    monkeypatch.setenv(MODE, "recomputed")
    with (run / "labels_rr2.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ProtocolMismatchError, match="artifact hash changed"):
        load_frozen_j_artifacts(root)


def test_recomputed_j_still_rejects_later_period(monkeypatch, tmp_path):
    root, run = _new_j(tmp_path)
    monkeypatch.setenv(MODE, "recomputed")
    protocol = json.loads((run / "protocol.json").read_text())
    protocol["max_loaded_timestamp"] = "2025-07-01T00:00:00Z"
    _json(run / "protocol.json", protocol)
    state = json.loads((run / "run_state.json").read_text())
    state["artifacts"]["protocol.json"] = {
        "sha256": hashlib.sha256((run / "protocol.json").read_bytes()).hexdigest(),
        "size": (run / "protocol.json").stat().st_size,
    }
    _json(run / "run_state.json", state)
    with pytest.raises(ProtocolMismatchError, match="development boundary"):
        load_frozen_j_artifacts(root)


def test_recomputed_u_identity_remains_self_consistent(monkeypatch):
    from experiments import run_event_window_direction_head as direction
    monkeypatch.setenv(MODE, "recomputed")
    payload = {"stage": "dev", "development_end_exclusive": "2025-07-01"}
    protocol_hash = direction._sha_payload(payload)
    identity = {"protocol_hash": protocol_hash, "source_hash": "a" * 64, "input_hash": "b" * 64}
    run_hash = direction._sha_payload(identity)[:20]
    state = {**identity, "run_hash": run_hash, "status": "complete"}
    pointer = {"run_hash": run_hash, "protocol_hash": protocol_hash, "relative_path": f"{run_hash}/full"}
    direction._validate_frozen_u_identity(pointer=pointer, state=state,
        protocol={**payload, **identity, "run_hash": run_hash}, manifest_sha256="c" * 64)
    with pytest.raises(ValueError, match="protocol hash changed"):
        direction._validate_frozen_u_identity(pointer=pointer, state=state,
            protocol={**payload, **identity, "run_hash": run_hash, "stage": "forward"}, manifest_sha256="c" * 64)


def _completed_fixture(root, names, *, protocol=None, summary=None, frozen=None, frames=None):
    from experiments.run_event_window_cost_aware_entry import _sha_payload
    protocol = {"stage": "dev", "smoke": False, "development_end_exclusive": "2025-07-01",
                "forward_or_lockbox_loaded": False, **(protocol or {})}
    identity = {"protocol_hash": _sha_payload(protocol), "source_hash": "a" * 64, "input_hash": "b" * 64}
    identity["run_hash"] = _sha_payload(identity)[:20]
    summary = {**identity, "forward_or_lockbox_loaded": False, **(summary or {})}
    run = root / identity["run_hash"] / "full"
    run.mkdir(parents=True)
    payloads = {"protocol.json": {**protocol, **identity}, "summary.json": summary,
                "frozen_protocol.json": frozen or {}}
    for name in names:
        path = run / name
        if name.endswith(".json"):
            _json(path, payloads.get(name, {}))
        else:
            frame = (frames or {}).get(name, pd.DataFrame({"value": [1]}))
            if name.endswith(".parquet"):
                frame.to_parquet(path, index=False)
            else:
                frame.to_csv(path, index=False)
    records = {name: {"size": (run / name).stat().st_size,
        "sha256": hashlib.sha256((run / name).read_bytes()).hexdigest()} for name in names}
    _json(run / "run_state.json", {**identity, "status": "complete", "summary": summary, "artifacts": records})
    _json(root / "latest_dev.json", {"run_hash": identity["run_hash"],
        "protocol_hash": identity["protocol_hash"], "relative_path": f"{identity['run_hash']}/full"})
    return run


def test_recomputed_p_uses_new_complete_pointer(monkeypatch, tmp_path):
    from experiments import run_event_window_economic_feasibility as q
    from experiments.run_event_window_conditional_opportunity import READER_ARTIFACTS
    score = pd.DataFrame({"model": ["xgboost"], "fold_id": ["2024H1"], "window_id": ["w"],
        "channel_episode_id": ["e"], "step": [0], "decision_time": [pd.Timestamp("2024-01-01", tz="UTC")],
        "p_t_le_60": [0.5], "p0_t_le_60": [0.4]})
    run = _completed_fixture(tmp_path, READER_ARTIFACTS,
        summary={"direction_head_trained": False, "economics_evaluated": False},
        frames={"oof_predictions.parquet": score})
    monkeypatch.setenv(MODE, "recomputed")
    loaded = q.load_frozen_p_artifacts(tmp_path)
    assert loaded.run_dir == run and loaded.run_hash != q.FROZEN_P_RUN_HASH
    with (run / "policy_metrics.csv").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        q.load_frozen_p_artifacts(tmp_path)


@pytest.mark.parametrize("stage", ("q", "r"))
@pytest.mark.parametrize("later", (False, True))
def test_recomputed_q_r_require_completed_development_handoff(monkeypatch, tmp_path, stage, later):
    from experiments import run_event_window_economic_feasibility as q
    from experiments import run_event_window_timing_policy_repair as r
    from experiments import run_event_window_calendar_ablation as s
    upstream = "1" * 20
    protocol = {"frozen_p_run_hash": upstream, "stage": "forward" if later else "dev"}
    summary = {"timing_model_refit": False, "frozen_p_run_hash": upstream,
               "level_rearm_selected_for_direction_head": True}
    run = _completed_fixture(tmp_path, q.READER_ARTIFACTS if stage == "q" else r.READER_ARTIFACTS,
        protocol=protocol, summary=summary, frozen={"frozen_p_run_hash": upstream})
    loader = r.load_frozen_q_handoff if stage == "q" else s.load_frozen_r_handoff
    monkeypatch.setenv(MODE, "recomputed")
    if later:
        with pytest.raises(ValueError, match="development"):
            loader(tmp_path)
    else:
        assert loader(tmp_path).run_dir == run


def test_recomputed_u_accepts_new_activation_counts_but_checks_artifacts(monkeypatch, tmp_path):
    from experiments import run_event_window_direction_head as v
    folds = ["2022H1", *v.SCORED_FOLDS]
    ledger = pd.DataFrame({"arm": [v.FROZEN_TIMING_ARM] * len(folds), "fold_id": folds,
        "activation_key": [f"a{i}" for i in range(len(folds))],
        "window_id": [f"w{i}" for i in range(len(folds))],
        "channel_episode_id": [f"e{i}" for i in range(len(folds))], "step": [0] * len(folds),
        "decision_time": [pd.Timestamp(f"{f[:4]}-{'01' if f[-1] == '1' else '07'}-02", tz="UTC") for f in folds],
        "threshold": [0.5] * len(folds), "activation_score": [0.8] * len(folds),
        "reference_price": [100.0] * len(folds), "adaptive_barrier_bps": [20.0] * len(folds)})
    for field in v._TIMING_COLUMNS:
        if field not in ledger:
            ledger[field] = "test"
    run = _completed_fixture(tmp_path, v.FROZEN_U_ARTIFACTS,
        summary={"activation_counts": {v.FROZEN_TIMING_ARM: len(folds)}},
        frames={"activation_ledger.parquet": ledger,
                "oof_predictions.parquet": ledger.assign(p_t_le_60=0.8),
                "threshold_audit.csv": ledger[["arm", "fold_id", "threshold"]]})
    monkeypatch.setenv(MODE, "recomputed")
    loaded = v.load_frozen_u_artifacts(tmp_path)
    assert len(loaded.ledger) == len(folds) != v.EXPECTED_SOURCE_ACTIVATIONS
    assert loaded.run_dir == run
    with (run / "activation_ledger.parquet").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="artifact changed"):
        v.load_frozen_u_artifacts(tmp_path)


class _SelectionCompleted(Exception):
    pass


def _stop_after_selection(*args, **kwargs):
    raise _SelectionCompleted


def test_recomputed_q_does_not_require_historical_trade_distribution(monkeypatch):
    from experiments import run_event_window_economic_feasibility as q
    source = SimpleNamespace(run_hash="a" * 20, frozen={"frozen_o_run_hash": "b" * 20})
    ledger = pd.DataFrame({"arm": ["conditional"] * 3, "target_activations_per_day": [2.0] * 3,
        "channel_episode_id": ["e"] * 3, "decision_time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")})
    monkeypatch.setattr(q, "load_frozen_p_artifacts", lambda *args: source)
    monkeypatch.setattr(q, "reconstruct_activation_ledger", lambda *args: ledger)
    monkeypatch.setattr(q, "load_frozen_o_artifacts", _stop_after_selection)
    monkeypatch.delenv(MODE, raising=False)
    with pytest.raises(AssertionError, match="activation supply changed"):
        q.run_economic_feasibility()
    monkeypatch.setenv(MODE, "recomputed")
    with pytest.raises(_SelectionCompleted):
        q.run_economic_feasibility()
    ledger["decision_time"] = pd.date_range("2024-01-01", periods=3, freq="30min", tz="UTC")
    with pytest.raises(AssertionError, match="cooldown"):
        q.run_economic_feasibility()


def test_recomputed_r_does_not_require_historical_supply_but_keeps_frequency_band(monkeypatch):
    from experiments import run_event_window_timing_policy_repair as r
    p = SimpleNamespace(run_hash="a" * 20, oof_sha256="c" * 64, calibration_sha256="d" * 64)
    q = SimpleNamespace(run_hash="b" * 20, frozen={"frozen_p_run_hash": p.run_hash,
        "frozen_p_oof_sha256": p.oof_sha256, "frozen_p_calibration_sha256": p.calibration_sha256,
        "frozen_o_run_hash": "e" * 20})
    ledger = pd.DataFrame({"arm": ["xgboost_conditional_level_rearm"] * 3,
        "decision_time": pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")})
    supply = pd.DataFrame({"arm": ["xgboost_conditional"], "level_rearm_same_threshold_activations": [3]})
    monkeypatch.setattr(r, "load_frozen_p_artifacts", lambda *args: p)
    monkeypatch.setattr(r, "load_frozen_q_handoff", lambda *args: q)
    monkeypatch.setattr(r, "reconstruct_policy_ledger", lambda *args: (ledger, pd.DataFrame(), supply))
    monkeypatch.setattr(r, "load_frozen_o_artifacts", _stop_after_selection)
    monkeypatch.delenv(MODE, raising=False)
    with pytest.raises(AssertionError, match="supply changed"):
        r.run_timing_policy_repair()
    monkeypatch.setenv(MODE, "recomputed")
    with pytest.raises(_SelectionCompleted):
        r.run_timing_policy_repair()
    ledger.drop(index=[1, 2], inplace=True)
    with pytest.raises(AssertionError, match="frequency band"):
        r.run_timing_policy_repair()
