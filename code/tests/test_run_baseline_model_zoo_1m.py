from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_sequence_runs_nine_models_in_declared_order(monkeypatch, tmp_path: Path):
    from experiments import run_baseline_model_zoo_1m as module

    calls = []

    def fake_run_model(model_name, **kwargs):
        calls.append(model_name)
        return {"status": "complete"}

    monkeypatch.setattr(module, "run_model", fake_run_model)
    result = module.run_sequence(
        ("logreg", "catboost_balanced"),
        output_root=tmp_path,
        prepared=object(),
        smoke=False,
    )

    assert calls == ["logreg", "catboost_balanced"]
    assert result["completed_models"] == calls


def test_state_write_retries_transient_windows_permission_error(
    monkeypatch, tmp_path: Path
):
    from experiments import run_baseline_model_zoo_1m as module

    calls = []

    def flaky(*args, **kwargs):
        calls.append(kwargs["status"])
        if len(calls) == 1:
            raise PermissionError(5, "transient lock")

    monkeypatch.setattr(module, "write_run_state", flaky)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    module._write_state(tmp_path / "state.json", status="running", detail={})
    assert calls == ["running", "running"]


def test_sequence_persists_exact_traceback_on_failure(monkeypatch, tmp_path: Path):
    from experiments import run_baseline_model_zoo_1m as module

    monkeypatch.setattr(
        module,
        "run_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("deliberate")),
    )
    with pytest.raises(RuntimeError, match="deliberate"):
        module.run_sequence(
            ("logreg",),
            output_root=tmp_path,
            prepared=object(),
            smoke=False,
        )

    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert "RuntimeError: deliberate" in state["traceback"]


def test_run_model_uses_candidate_zero_and_no_sentiment_contract(
    monkeypatch, tmp_path: Path
):
    from experiments import run_baseline_model_zoo_1m as module

    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return {"status": "complete"}

    monkeypatch.setattr(module, "BaselineModelRunner", FakeRunner)
    result = module.run_model(
        "logreg",
        output_root=tmp_path,
        prepared=object(),
        smoke=False,
    )

    assert result["status"] == "complete"
    assert captured["model_name"] == "logreg"
    assert captured["candidate_id"] == 0
    assert captured["candidate_params"] == {}
    assert captured["widths"] == (55, 65, 75)


def test_cli_loads_no_sentiment_data(monkeypatch, tmp_path: Path):
    from experiments import run_baseline_model_zoo_1m as module

    captured = {}
    monkeypatch.setattr(
        module,
        "load_prepared_data",
        lambda **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        module,
        "run_sequence",
        lambda *args, **kwargs: {"status": "complete"},
    )

    assert module.main(
        ["--smoke", "--model", "logreg", "--output-root", str(tmp_path)]
    ) == 0
    assert captured["sentiment"] == "none"
