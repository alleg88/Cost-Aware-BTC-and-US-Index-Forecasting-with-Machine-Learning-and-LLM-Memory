import math

import pandas as pd

from experiments.build_reflection_cache import DEFAULT_OUTPUT, build_cache
from reflection_agent.execution import build_frozen_baselines


def test_frozen_execution_reproduces_exact_current_controls():
    if not (DEFAULT_OUTPUT / "frozen_probability_panel.parquet").exists():
        build_cache()
    summary = build_frozen_baselines().set_index("control_id")
    assert int(summary.loc["lstm", "trades"]) == 54
    assert math.isclose(summary.loc["lstm", "net_return"], 0.0358181428405421, abs_tol=1e-12)
    assert int(summary.loc["unanimity_consensus", "trades"]) == 22
    assert math.isclose(summary.loc["unanimity_consensus", "net_return"], 0.012003442975841907, abs_tol=1e-12)
    assert int(summary.loc["deterministic_router", "trades"]) == 68
    assert math.isclose(summary.loc["deterministic_router", "net_return"], -0.04482144562218393, abs_tol=1e-12)


def test_control_artifacts_are_aligned_and_stop_before_q2():
    returns = pd.read_parquet(DEFAULT_OUTPUT / "frozen_baseline_returns.parquet")
    timestamp = pd.to_datetime(returns["timestamp"], utc=True)
    assert timestamp.max() < pd.Timestamp("2026-04-01", tz="UTC")
    assert returns[["lstm", "unanimity_consensus", "deterministic_router"]].notna().all().all()
