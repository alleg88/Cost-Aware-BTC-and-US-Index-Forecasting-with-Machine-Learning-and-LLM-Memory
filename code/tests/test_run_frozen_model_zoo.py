import json

import pytest

from experiments.model_zoo_protocol import BASE_MODELS, protocol_fingerprint
from experiments.run_frozen_model_zoo import (
    completed_state,
    main,
    parse_models,
    should_resume,
)


def test_parse_models_preserves_frozen_order_and_rejects_unknown():
    assert parse_models("all") == list(BASE_MODELS)
    assert parse_models("gru,logreg") == ["gru", "logreg"]
    with pytest.raises(ValueError, match="unknown model"):
        parse_models("gru,unknown")


def test_completed_model_is_skipped_only_when_fingerprint_matches():
    state = completed_state("gru", protocol_fingerprint())
    assert should_resume(state, "gru", protocol_fingerprint())
    assert not should_resume(state, "gru", "different")
    assert not should_resume(state, "logreg", protocol_fingerprint())


def test_launcher_runs_models_sequentially_and_records_completion(
    tmp_path, monkeypatch
):
    import experiments.run_frozen_model_zoo as runner

    calls = []

    def fake_run(model, **kwargs):
        calls.append((model, kwargs))
        return {"model": model, "decision": "no_trade"}

    state_path = tmp_path / "run_state.json"
    monkeypatch.setattr(runner, "RUN_STATE_PATH", state_path)
    monkeypatch.setattr(runner, "run_model_study", fake_run)

    assert main(
        [
            "--models",
            "gru,logreg",
            "--trials",
            "2",
            "--fold-limit",
            "1",
            "--candidate-limit",
            "1",
        ]
    ) == 0

    assert [model for model, _ in calls] == ["gru", "logreg"]
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["models"]["gru"]["status"] == "complete"
    assert state["models"]["logreg"]["status"] == "complete"
    assert state["protocol_fingerprint"] == protocol_fingerprint()


def test_resume_skips_matching_complete_model(tmp_path, monkeypatch):
    import experiments.run_frozen_model_zoo as runner

    state_path = tmp_path / "run_state.json"
    state_path.write_text(
        json.dumps(completed_state("gru", protocol_fingerprint())),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner, "RUN_STATE_PATH", state_path)
    monkeypatch.setattr(
        runner,
        "run_model_study",
        lambda *args, **kwargs: pytest.fail("completed model reran"),
    )

    assert main(["--models", "gru", "--resume"]) == 0
