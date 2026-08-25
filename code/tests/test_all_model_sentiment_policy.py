from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def test_policy_protocol_covers_every_model_arm_dead_zone_combinations():
    from experiments.all_model_sentiment_policy import (
        ARMS,
        MODEL_NAMES,
        WIDTHS,
        expected_combination_count,
        protocol_manifest,
    )

    assert ARMS == ("none", "classic", "llm", "llm_full")
    assert WIDTHS == (55, 65, 75)
    assert len(MODEL_NAMES) == 9
    assert expected_combination_count() == 108
    manifest = protocol_manifest("logreg", "classic")
    assert manifest["policy_count_per_model_arm_width"] == 33
    assert manifest["policy_calibration_used"] is True
    assert manifest["one_minute_execution_used"] is True
    assert manifest["reuse_notebook02c_raw_forward"] is True
    assert manifest["rerun_2024_folds"] is False
    assert manifest["lookback_days"] == [180]
    assert manifest["hold_bars"] == [1]


def test_policy_sequence_runs_every_arm_model_pair_once(monkeypatch, tmp_path: Path):
    from experiments import all_model_sentiment_policy as module

    calls = []

    def fake_run_one(arm, model_name, **kwargs):
        calls.append((arm, model_name))
        return {"status": "complete"}

    monkeypatch.setattr(module, "run_one", fake_run_one)
    result = module.run_sequence(
        arms=("none", "classic"),
        model_names=("logreg", "svm_linear"),
        output_root=tmp_path,
        prepared_by_arm={"none": object(), "classic": object()},
        smoke=False,
    )
    assert calls == [
        ("none", "logreg"),
        ("none", "svm_linear"),
        ("classic", "logreg"),
        ("classic", "svm_linear"),
    ]
    assert result["completed"] == [f"{arm}/{model}" for arm, model in calls]


def test_run_one_passes_raw_source_and_sentiment_identity_to_policy_runner(
    monkeypatch, tmp_path: Path
):
    from experiments import all_model_sentiment_policy as module

    captured = {}

    class FakeRunner:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run(self):
            return {"status": "complete"}

    monkeypatch.setattr(module, "PolicyOnlyModelRunner", FakeRunner)
    result = module.run_one(
        "llm",
        "logreg",
        output_root=tmp_path,
        raw_root=tmp_path / "raw",
        prepared=object(),
        smoke=False,
    )
    assert result["status"] == "complete"
    assert captured["sentiment_mode"] == "llm"
    assert captured["widths"] == (55, 65, 75)
    assert captured["raw_model_root"] == tmp_path / "raw" / "llm" / "logreg"
    protocol = json.loads(
        (tmp_path / "llm" / "logreg" / "protocol.json").read_text(encoding="utf-8")
    )
    assert protocol["sentiment_arm"] == "llm"


def test_policy_scoreboard_has_one_row_per_arm_model_dead_zone(
    monkeypatch, tmp_path: Path
):
    from experiments import all_model_sentiment_policy_scoreboard as module

    monkeypatch.setattr(module, "MODEL_NAMES", ("logreg",))
    for arm in module.ARMS:
        root = tmp_path / arm / "logreg"
        root.mkdir(parents=True)
        common = {
            "model_name": ["logreg"] * 3,
            "width_bps": [55, 65, 75],
            "policy_id": [1, 2, 3],
            "tau": [0.55, 0.60, 0.65],
            "tp_bps": [150, 150, 200],
            "sl_bps": [75, 100, 100],
            "max_hold": [1, 1, 1],
        }
        pd.DataFrame(
            {
                **common,
                "monthly_fit_count": [6, 6, 6],
                "trades": [60, 61, 62],
                "positive_segments": [4, 5, 6],
                "pooled_sortino": [0.1, 0.2, 0.3],
                "pooled_sharpe": [0.05, 0.10, 0.15],
                "pooled_net": [0.01, 0.02, 0.03],
            }
        ).to_parquet(root / "selected_policies_2025h1.parquet", index=False)
        pd.DataFrame(
            {
                **common,
                "period_start": [pd.Timestamp("2025-07-01", tz="UTC")] * 3,
                "period_end": [pd.Timestamp("2026-04-01", tz="UTC")] * 3,
                "trades": [50, 51, 52],
                "sortino": [0.3, 0.2, 0.1],
                "sharpe": [0.15, 0.10, 0.05],
                "net_return": [0.03, 0.02, 0.01],
                "positive_months": [6, 5, 4],
            }
        ).to_parquet(root / "forward_summary.parquet", index=False)
    monkeypatch.setattr(
        module,
        "validate_policy_model_artifacts",
        lambda *args, **kwargs: {},
    )
    tables = module.build_scoreboards(tmp_path)
    assert len(tables["policies"]) == 12
    assert len(tables["economics"]) == 12
    assert tables["policies"].duplicated(
        ["sentiment_arm", "model_name", "width_bps"]
    ).sum() == 0
