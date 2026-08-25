from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def test_matched_model_zoo_launcher_module_exists():
    assert importlib.util.find_spec("experiments.run_matched_model_zoo_1m") is not None


def test_sequence_runs_models_in_declared_order(monkeypatch, tmp_path):
    from experiments import run_matched_model_zoo_1m as module

    calls = []

    def fake_run_model(model_name, **kwargs):
        calls.append(model_name)
        return {"status": "complete", "model_name": model_name}

    monkeypatch.setattr(module, "run_model", fake_run_model)
    result = module.run_sequence(
        ("logreg", "decision_tree"),
        output_root=tmp_path,
        prepared=object(),
        store=object(),
        smoke=False,
    )

    assert calls == ["logreg", "decision_tree"]
    assert result["status"] == "complete"
    assert result["completed_models"] == ["logreg", "decision_tree"]


def test_sequence_records_traceback_and_stops_after_failure(monkeypatch, tmp_path):
    from experiments import run_matched_model_zoo_1m as module

    calls = []

    def failing_run_model(model_name, **kwargs):
        calls.append(model_name)
        raise RuntimeError("deliberate launcher failure")

    monkeypatch.setattr(module, "run_model", failing_run_model)
    with pytest.raises(RuntimeError, match="deliberate launcher failure"):
        module.run_sequence(
            ("logreg", "decision_tree"),
            output_root=tmp_path,
            prepared=object(),
            store=object(),
            smoke=False,
        )

    state = json.loads((tmp_path / "run_state.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert "RuntimeError: deliberate launcher failure" in state["traceback"]
    assert calls == ["logreg"]


@pytest.mark.parametrize(
    ("smoke", "expected_widths", "expected_candidates", "expected_folds"),
    [
        (True, (55,), 1, 1),
        (False, (55, 65, 75), 15, 5),
    ],
)
def test_run_model_configures_the_frozen_runner(
    monkeypatch,
    tmp_path: Path,
    smoke: bool,
    expected_widths: tuple[int, ...],
    expected_candidates: int,
    expected_folds: int,
):
    from experiments import run_matched_model_zoo_1m as module

    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return {"status": "complete"}

    monkeypatch.setattr(module, "ExecutionResolutionRunner", FakeRunner)
    result = module.run_model(
        "logreg",
        output_root=tmp_path,
        prepared=object(),
        store=object(),
        smoke=smoke,
    )

    model_root = tmp_path / "logreg"
    assert result["status"] == "complete"
    assert captured["output_root"] == model_root
    assert captured["prediction_root"] == model_root / "prediction_cache"
    assert captured["widths"] == expected_widths
    assert len(captured["candidates"]) == expected_candidates
    assert captured["fold_limit"] == expected_folds
    assert captured["stage1_only"] is smoke
    assert captured["model_name"] == "logreg"
    manifest = json.loads(
        (model_root / "candidate_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["model_name"] == "logreg"
    assert len(manifest["candidates"]) == 15


def test_cli_prepares_one_minute_data_and_selected_models(monkeypatch, tmp_path):
    from experiments import run_matched_model_zoo_1m as module

    prepared = object()
    store = object()
    captured = {}
    monkeypatch.setattr(module, "load_prepared_data", lambda **kwargs: prepared)
    monkeypatch.setattr(
        module.PartitionedIntrabarStore,
        "one_minute",
        lambda path: store,
    )

    def fake_sequence(model_names, **kwargs):
        captured["model_names"] = tuple(model_names)
        captured.update(kwargs)
        return {"status": "complete"}

    monkeypatch.setattr(module, "run_sequence", fake_sequence)
    assert module.main(
        [
            "--smoke",
            "--model",
            "logreg",
            "--output-root",
            str(tmp_path),
        ]
    ) == 0
    assert captured["model_names"] == ("logreg",)
    assert captured["prepared"] is prepared
    assert captured["store"] is store
    assert captured["smoke"] is True


def test_full_run_skips_only_matching_valid_completed_model(monkeypatch, tmp_path):
    from experiments import run_matched_model_zoo_1m as module

    model_root = tmp_path / "logreg"
    model_root.mkdir(parents=True)
    manifest = module.candidate_manifest("logreg")
    (model_root / "candidate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (model_root / "run_state.json").write_text(
        json.dumps({"status": "complete"}), encoding="utf-8"
    )
    (model_root / "result.json").write_text(
        json.dumps({"status": "complete", "model_name": "logreg"}),
        encoding="utf-8",
    )
    validated = []
    monkeypatch.setattr(
        module,
        "validate_model_artifacts",
        lambda root, model_name: validated.append((root, model_name)) or {},
    )

    class MustNotRun:
        def __init__(self, **kwargs):
            raise AssertionError("valid completed work must be skipped")

    monkeypatch.setattr(module, "ExecutionResolutionRunner", MustNotRun)
    result = module.run_model(
        "logreg",
        output_root=tmp_path,
        prepared=object(),
        store=object(),
        smoke=False,
    )

    assert result["status"] == "complete"
    assert result["resumed"] is True
    assert validated == [(model_root, "logreg")]
