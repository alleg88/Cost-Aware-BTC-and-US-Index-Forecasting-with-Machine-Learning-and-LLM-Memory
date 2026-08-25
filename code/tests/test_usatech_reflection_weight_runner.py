from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.index_all_model_ensemble import AlignedPanel
import experiments.run_usa500_reflection_weight_agent as usa500_runner
from reflection_agent.index_v1.config import MODEL_NAMES
from reflection_agent.usatech_v1.engine import build_soft_vote_opportunities


CODE_ROOT = Path(__file__).parents[1]


def test_usatech_reflection_runner_is_registered():
    assert importlib.util.find_spec(
        "experiments.run_usatech_reflection_weight_agent"
    ) is not None


def _runner():
    return importlib.import_module("experiments.run_usatech_reflection_weight_agent")


def _panel(start: str) -> AlignedPanel:
    timestamp = pd.date_range(start, periods=4, freq="15min", tz="UTC")
    probabilities = {}
    for model_index, model in enumerate(MODEL_NAMES):
        probabilities[model] = np.asarray(
            [
                [0.05, 0.05, 0.90] if model_index < 4 else [0.51, 0.00, 0.49],
                [0.90, 0.05, 0.05] if model_index < 4 else [0.49, 0.00, 0.51],
                [0.02, 0.03, 0.95],
                [0.95, 0.03, 0.02],
            ],
            dtype=float,
        )
    return AlignedPanel(
        timestamp=timestamp,
        y_true=np.asarray([2, 0, 2, 0], dtype=int),
        probabilities=probabilities,
        fit_ids={model: np.asarray([f"{model}-fit"] * 4) for model in MODEL_NAMES},
    )


def _bars(start: str) -> pd.DataFrame:
    index = pd.date_range(start, periods=6, freq="15min", tz="UTC")
    opens = np.asarray([100.0, 101.0, 100.0, 102.0, 101.0, 103.0])
    closes = np.asarray([100.5, 100.0, 102.0, 101.0, 103.0, 102.0])
    return pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 1.0,
            "low": np.minimum(opens, closes) - 1.0,
            "close": closes,
            "volume": 10.0,
            "available_at": index + pd.Timedelta(minutes=15),
            "complete_bar": True,
        },
        index=index,
    )


def _state(start: str) -> pd.DataFrame:
    index = pd.date_range(start, periods=6, freq="15min", tz="UTC")
    return pd.DataFrame(
        {
            "available_at": index + pd.Timedelta(minutes=15),
            "state_vix_regime": np.linspace(-0.5, 0.5, 6),
            "state_trailing_vol": np.linspace(0.01, 0.02, 6),
            "state_trailing_trend": np.linspace(-0.01, 0.01, 6),
        },
        index=index,
    )


def _synthetic_inputs() -> dict:
    h1_start = "2025-01-06T00:00:00Z"
    forward_start = "2025-07-01T00:00:00Z"
    h1_panel = _panel(h1_start)
    forward_panel = _panel(forward_start)
    bars = pd.concat([_bars(h1_start), _bars(forward_start)]).sort_index()
    state = pd.concat([_state(h1_start), _state(forward_start)]).sort_index()
    _, expected = build_soft_vote_opportunities(
        forward_panel,
        bars,
        start="2025-07-01T00:00:00Z",
        end="2026-04-01T00:00:00Z",
        tau=0.55,
        cost_bps=3.0,
        state_frame=state,
    )
    return {
        "bars": bars,
        "h1_panel": h1_panel,
        "forward_panel": forward_panel,
        "expected_forward_ledger": expected,
        "source_identity": {"synthetic_fixture_sha256": "a" * 64},
        "state_frame": state,
    }


def test_prepare_reconciles_soft_vote_parent_and_restores_usa500_module(tmp_path: Path):
    runner = _runner()
    usa500_hash_before = usa500_runner._implementation_hash()

    manifest = runner.prepare_common_artifacts(
        output_root=tmp_path / "usatech_reflection_weight_agent",
        **_synthetic_inputs(),
    )

    assert manifest["stage_counts"] == {"h1": 4, "forward": 4}
    assert manifest["reconciliation"] == {
        "exact_rows": True,
        "exact_keys": True,
        "exact_sides": True,
        "exact_economics": True,
    }
    assert manifest["q2_loaded"] is False
    assert manifest["implementation_hash"] == runner._implementation_hash()
    assert manifest["implementation_hash"] != usa500_hash_before
    assert usa500_runner._implementation_hash() == usa500_hash_before
    protocol = json.loads(
        (tmp_path / "usatech_reflection_weight_agent/common/protocol.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["config"]["stream_name"] == "usatech"
    assert protocol["config"]["source_candidate_id"] == "deepseek_full__soft_vote"
    assert protocol["config"]["round_trip_cost_bps"] == 3.0


def test_real_parent_preparation_reconciles_366_h1_and_23_forward(tmp_path: Path):
    runner = _runner()

    manifest = runner.prepare_common_artifacts(
        output_root=tmp_path / "real_usatech_reflection_weight_agent"
    )

    assert manifest["stage_counts"] == {"h1": 366, "forward": 23}
    assert manifest["reconciliation"]["exact_economics"] is True
    assert manifest["q2_loaded"] is False
    assert pd.Timestamp(manifest["maximum_signal_time"]) < pd.Timestamp(
        "2026-04-01T00:00:00Z"
    )


def test_completed_real_audit_reconciles_the_frozen_three_bps_cost():
    runner = _runner()
    root = CODE_ROOT / "experiments" / "cache" / "usatech_reflection_weight_agent"

    summary = runner.finalize_experiment(output_root=root)
    audit = pd.read_parquet(root / "leakage_audit.parquet").set_index("check_id")

    assert summary["all_integrity_checks_pass"] is True
    assert bool(audit.loc["execution_reconciled", "passed"]) is True
    assert "3 bps" in str(audit.loc["execution_reconciled", "detail"])
