from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import nbformat
import pandas as pd

from experiments import qualified_union


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"


def test_protocol_freezes_union_v1():
    protocol = qualified_union.protocol()

    assert protocol["protocol_version"] == "qualified-union-v1"
    assert protocol["members"] == [
        {"model": "lstm", "width_bps": 55, "tau": 0.75},
        {"model": "svm_linear", "width_bps": 75, "tau": 0.0},
    ]
    assert protocol["execution"] == {
        "tp_bps": 200,
        "sl_bps": 100,
        "max_hold": 1,
        "fee_bps_per_side": 5.0,
    }
    assert protocol["lockbox_start"] == "2026-04-01T00:00:00+00:00"
    assert protocol["lockbox_2026_q2_used"] is False


def test_combine_union_uses_union_with_conflict_veto():
    index = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    members = pd.DataFrame(
        {
            "lstm_signal": [1.0, 0.0, -1.0, 1.0, 0.0],
            "svm_signal": [0.0, -1.0, -1.0, -1.0, 0.0],
        },
        index=index,
    )

    actual = qualified_union.combine_union(members)

    expected = pd.Series(
        [1.0, -1.0, -1.0, 0.0, 0.0],
        index=index,
        name="union_signal",
    )
    pd.testing.assert_series_equal(actual, expected)


def test_temperature_or_confidence_cannot_modify_svm_tau_zero_signal():
    frame = pd.DataFrame(
        {
            "pred": [0, 1, 2],
            "confidence": [0.01, 0.50, 0.99],
        },
        index=pd.date_range("2025-01-01", periods=3, freq="15min", tz="UTC"),
    )

    signal = qualified_union.member_signal(frame, tau=0.0)

    np.testing.assert_array_equal(signal.to_numpy(), [-1.0, 0.0, 1.0])


def test_frozen_artifacts_reproduce_published_forward_result():
    protocol = json.loads((CACHE / "protocol.json").read_text(encoding="utf-8"))
    manifest = json.loads((CACHE / "manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(CACHE / "summary.csv").set_index("phase")
    forward = summary.loc["forward"]

    assert protocol["lockbox_2026_q2_used"] is False
    assert manifest["artifact_hashes"]
    assert int(forward["trades"]) == 74
    assert int(forward["long_trades"]) == 32
    assert int(forward["short_trades"]) == 42
    assert abs(float(forward["net_return"]) - 0.0631) < 5e-4
    assert abs(float(forward["sortino"]) - 2.106) < 5e-3
    assert abs(float(forward["sharpe"]) - 1.165) < 5e-3


def test_frozen_signal_and_returns_stop_before_lockbox():
    lockbox = pd.Timestamp("2026-04-01", tz="UTC")
    for filename, timestamp_column in (
        ("forward_signals.parquet", "timestamp"),
        ("forward_per_bar.parquet", "timestamp"),
        ("forward_ledger.parquet", "signal_time"),
    ):
        frame = pd.read_parquet(CACHE / filename)
        timestamp = pd.to_datetime(frame[timestamp_column], utc=True)
        assert timestamp.max() < lockbox


def test_notebook_03c_is_executed_artifact_reader():
    from experiments.build_notebook_03c import NOTEBOOK

    notebook = nbformat.read(NOTEBOOK, as_version=4)
    source = "\n".join(cell.source for cell in notebook.cells).lower()
    for phrase in (
        "qualified union v1",
        "immutable baseline",
        "74",
        "xgboost remains a challenger",
    ):
        assert phrase in source
    assert "glob.glob" not in source
    assert "simulate_bracket_trades_intrabar" not in source
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code and all(cell.execution_count is not None for cell in code)
    assert not any(
        output.output_type == "error"
        for cell in code
        for output in cell.get("outputs", [])
    )
