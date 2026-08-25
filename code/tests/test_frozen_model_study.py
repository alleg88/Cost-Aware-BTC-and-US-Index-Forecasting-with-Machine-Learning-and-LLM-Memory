from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.frozen_model_study import (
    StudyPaths,
    prediction_cache_path,
    study_command,
    validate_prediction_frame,
)
from experiments.model_zoo_protocol import protocol_fingerprint


def _prediction_frame(model: str = "gru") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "y_true": [1, 2],
            f"{model}_pred": [1, 2],
            f"{model}_conf": [0.6, 0.7],
            f"{model}_p0": [0.2, 0.1],
            f"{model}_p1": [0.6, 0.2],
            f"{model}_p2": [0.2, 0.7],
        },
        index=pd.to_datetime(["2025-06-01 00:00", "2025-06-01 00:15"], utc=True),
    )


def test_model_outputs_are_isolated_and_smoke_cannot_resume_full():
    cat = StudyPaths.for_model("catboost_balanced")
    gru = StudyPaths.for_model("gru")
    smoke = StudyPaths.for_model("gru", smoke=True)

    assert cat.root != gru.root
    assert "antibull_model_zoo" in cat.root.as_posix()
    assert smoke.root != gru.root
    assert smoke.root.parts[-2:] == ("smoke", "gru")


def test_prediction_cache_identity_contains_full_protocol():
    paths = StudyPaths.for_model("gru")
    cache = prediction_cache_path(
        paths,
        "gru",
        width=75,
        candidate_id=6,
        params={"epochs": 10, "seq_len": 32},
        fold_id=12,
        month="2025-04",
    )

    name = cache.name
    assert name.startswith(f"{protocol_fingerprint()}_gru_w75_candidate_06_")
    assert name.endswith("_fold_12_2025-04.parquet")


def test_prediction_validation_accepts_aligned_probabilities():
    frame = _prediction_frame()
    actual = validate_prediction_frame(
        frame,
        model="gru",
        development_end=pd.Timestamp("2025-07-01", tz="UTC"),
    )
    pd.testing.assert_frame_equal(actual, frame)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda frame: frame.set_axis(
                pd.to_datetime(["2025-06-01 00:00", "2025-07-01 00:00"], utc=True)
            ),
            "development boundary",
        ),
        (
            lambda frame: frame.assign(gru_p2=[np.nan, 0.7]),
            "finite",
        ),
        (
            lambda frame: frame.assign(gru_p2=[0.4, 0.7]),
            "sum to one",
        ),
    ],
)
def test_prediction_validation_rejects_invalid_caches(mutate, message):
    with pytest.raises(ValueError, match=message):
        validate_prediction_frame(
            mutate(_prediction_frame()),
            model="gru",
            development_end=pd.Timestamp("2025-07-01", tz="UTC"),
        )


def test_study_paths_expose_every_primary_artifact():
    paths = StudyPaths.for_model("logreg")
    expected = {
        "predictions",
        "candidates",
        "classification",
        "economics",
        "audit",
        "ungated",
        "objective",
        "manifest",
        "result",
    }
    assert expected <= set(paths.__dataclass_fields__)
    assert all(isinstance(getattr(paths, name), Path) for name in expected)

def test_study_command_uses_model_zoo_namespace_and_limits():
    command = study_command(
        "gru", n_trials=15, fold_limit=1, candidate_limit=2
    )
    assert command[:3] == [
        command[0],
        "-m",
        "experiments.run_tune_antibull_widths",
    ]
    assert command[command.index("--model") + 1] == "gru"
    assert command[command.index("--trials") + 1] == "15"
    assert command[command.index("--fold-limit") + 1] == "1"
    assert command[command.index("--candidate-limit") + 1] == "2"
    assert "--model-zoo" in command
