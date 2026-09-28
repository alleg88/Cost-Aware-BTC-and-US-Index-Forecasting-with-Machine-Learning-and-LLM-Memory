from __future__ import annotations

import importlib
import json

import pandas as pd
import pytest


MODE = "MSC_REBUILD_CHANNEL_HANDOFFS"
CANONICAL_COUNTS = {
    "2022H2": 236,
    "2023H1": 541,
    "2023H2": 305,
    "2024H1": 910,
    "2024H2": 532,
    "2025H1": 415,
}


def _runner():
    return importlib.import_module(
        "experiments.run_channel_vs_volatility_ablation"
    )


def _u_ledger(counts: dict[str, int]) -> pd.DataFrame:
    rows = [{"fold_id": "2022H1"}]
    for fold_id, count in counts.items():
        rows.extend({"fold_id": fold_id} for _ in range(count))
    return pd.DataFrame(rows)


def _predictions(runner, counts: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in runner.MODELS:
        for fold_id, count in counts.items():
            start = pd.Timestamp(runner.FOLD_BOUNDS[fold_id][0], tz="UTC")
            for position in range(count):
                rows.append(
                    {
                        "model": model,
                        "fold_id": fold_id,
                        "decision_time": start + pd.Timedelta(hours=2 * position),
                        "reference_price": 100.0,
                        "adaptive_barrier_bps": 100.0,
                        "in_channel_window": True,
                        "path_complete": True,
                        "opportunity_score": 0.9 - position * 0.01,
                    }
                )
            rows.append(
                {
                    "model": model,
                    "fold_id": fold_id,
                    "decision_time": start + pd.Timedelta(hours=2 * count),
                    "reference_price": 100.0,
                    "adaptive_barrier_bps": 100.0,
                    "in_channel_window": False,
                    "path_complete": True,
                    "opportunity_score": 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_recomputed_mode_uses_u_fold_counts_without_mutating_canonical_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    counts = {
        "2022H2": 2,
        "2023H1": 1,
        "2023H2": 3,
        "2024H1": 1,
        "2024H2": 2,
        "2025H1": 1,
    }
    ledger = _u_ledger(counts)

    monkeypatch.delenv(MODE, raising=False)
    assert runner.effective_matched_fold_counts(ledger) == CANONICAL_COUNTS
    canonical_protocol = runner.protocol_dict()
    assert canonical_protocol["matched_fold_counts"] == CANONICAL_COUNTS
    assert canonical_protocol["matched_total_activations"] == 2_939
    assert canonical_protocol["matching_target_source"] == "canonical_frozen_v_fold_counts"
    with pytest.raises(ValueError, match="supports only"):
        runner.select_matched_activations(_predictions(runner, counts))
    monkeypatch.setenv(MODE, "recomputed")
    assert runner.effective_matched_fold_counts(ledger) == counts
    assert runner.MATCHED_FOLD_COUNTS == CANONICAL_COUNTS
    assert runner.EXPECTED_MATCHED_ACTIVATIONS == sum(CANONICAL_COUNTS.values())

    protocol = runner.protocol_dict(matched_fold_counts=counts)
    assert protocol["matched_fold_counts"] == counts
    assert protocol["matched_total_activations"] == sum(counts.values())
    assert protocol["target_activations_per_day"] == sum(counts.values()) / 1_096
    assert protocol["matching_target_source"] == "recomputed_u_scored_timing_ledger"


def test_recomputed_selection_matches_each_fold_across_both_window_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner()
    counts = {
        "2022H2": 2,
        "2023H1": 1,
        "2023H2": 3,
        "2024H1": 1,
        "2024H2": 2,
        "2025H1": 1,
    }
    monkeypatch.setenv(MODE, "recomputed")
    targets = runner.effective_matched_fold_counts(_u_ledger(counts))
    selected = runner.select_matched_activations(
        _predictions(runner, counts),
        matched_fold_counts=targets,
    )

    observed = (
        selected.groupby(["model", "window_source", "fold_id"])
        .size()
        .unstack(["model", "window_source"], fill_value=0)
    )
    for fold_id, target in counts.items():
        assert observed.loc[fold_id].eq(target).all()
    assert len(selected) == sum(counts.values()) * len(runner.MODELS) * 2

    audit = runner._frequency_audit(selected, matched_fold_counts=targets)
    assert audit["activations"].eq(sum(counts.values())).all()
    assert audit["fold_counts_exact"].all()
    assert audit["global_60m_refractory_passed"].all()
    assert audit["matching_target_source"].eq(
        "recomputed_u_scored_timing_ledger"
    ).all()
    assert audit["matched_fold_counts"].map(json.loads).map(lambda value: value == counts).all()
    assert audit["observed_fold_counts"].map(json.loads).map(lambda value: value == counts).all()
