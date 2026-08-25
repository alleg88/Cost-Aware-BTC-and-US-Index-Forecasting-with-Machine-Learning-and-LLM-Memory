import json
from pathlib import Path

import pandas as pd
import pytest


def _economic_row(width: int, candidate: int, policy: int, **overrides) -> dict:
    row = {
        "width_bps": width,
        "candidate_id": candidate,
        "policy_id": policy,
        "trades": 60,
        "n_long": 30,
        "n_short": 30,
        "positive_segments": 4,
        "robust_score": 0.2,
        "pooled_sortino": 0.4,
        "pooled_net": 0.01,
    }
    row.update(overrides)
    return row


def test_execution_policy_grid_has_exactly_66_deterministic_choices():
    from experiments.catboost_execution_resolution import (
        HOLDS,
        TP_SL_PAIRS,
        policy_choices,
    )

    choices = policy_choices()
    assert HOLDS == (1, 2)
    assert TP_SL_PAIRS == ((150, 75), (150, 100), (200, 100))
    assert len(choices) == 66
    assert len(set(choices)) == 66
    assert {geometry[2] for _, geometry in choices} == {1, 2}


def test_policy_fingerprint_is_resolution_and_source_specific():
    from experiments.catboost_execution_resolution import execution_policy_fingerprint

    common = {
        "stage": "selection",
        "width_bps": 55,
        "candidate_id": 3,
        "prediction_fingerprints": ["prediction-a"],
        "m15_fingerprint": "m15",
        "fee_bps": 5.0,
    }
    one_minute = execution_policy_fingerprint(
        **common, resolution="1m", execution_data_fingerprint="minute-a"
    )
    one_second = execution_policy_fingerprint(
        **common, resolution="1s", execution_data_fingerprint="second-a"
    )
    changed_source = execution_policy_fingerprint(
        **common, resolution="1s", execution_data_fingerprint="second-b"
    )

    assert one_minute != one_second
    assert one_second != changed_source
    assert one_second == execution_policy_fingerprint(
        **common, resolution="1s", execution_data_fingerprint="second-a"
    )


def test_economic_selection_returns_one_candidate_per_width_without_no_trade_gate():
    from experiments.catboost_execution_resolution import select_economic_candidates

    rows = []
    for width in (55, 65, 75):
        rows.extend(
            [
                _economic_row(width, 0, 0, pooled_sortino=0.9, trades=10),
                _economic_row(width, 1, 1, pooled_sortino=0.5, pooled_net=0.02),
                _economic_row(width, 2, 2, pooled_sortino=0.3, pooled_net=-0.01),
            ]
        )

    selected = select_economic_candidates(pd.DataFrame(rows))

    assert selected["width_bps"].tolist() == [55, 65, 75]
    assert selected["candidate_id"].tolist() == [1, 1, 1]
    assert selected["objective"].eq("economic").all()
    assert "decision" not in selected.columns


def test_full_arm_expected_artifact_counts_are_fixed():
    from experiments.catboost_execution_resolution import (
        EXPECTED_ARM_ROWS,
        assert_arm_artifact_counts,
    )

    assert EXPECTED_ARM_ROWS == {
        "economic_policy_grid_2024": 2970,
        "economic_candidate_winners_2024": 45,
        "selected_candidates_2024": 3,
        "calibration_policy_grid_2025h1": 198,
        "selected_policies_2025h1": 3,
        "forward_monthly": 27,
        "forward_quarterly": 9,
        "forward_summary": 3,
    }
    assert_arm_artifact_counts(EXPECTED_ARM_ROWS)
    wrong = dict(EXPECTED_ARM_ROWS)
    wrong["forward_summary"] = 2
    with pytest.raises(AssertionError, match="artifact row-count mismatch"):
        assert_arm_artifact_counts(wrong)


def test_partitioned_store_loads_only_requested_1s_months_and_seals_q2(tmp_path: Path):
    from data.build_1s import build_all
    from experiments.catboost_execution_resolution import PartitionedIntrabarStore

    raw = tmp_path / "raw"
    root = tmp_path / "one-second"
    raw.mkdir()
    columns = [
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "count", "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    for month, timestamp in (
        ("2025-01", 1_735_689_600_000),
        ("2025-02", 1_738_368_000_000),
    ):
        values = [timestamp, 100, 101, 99, 100.5, 1, timestamp + 999, 1, 1, 1, 1, 0]
        (raw / f"BTCUSDT-1s-{month}.csv").write_text(
            ",".join(columns) + "\n" + ",".join(map(str, values)) + "\n",
            encoding="utf-8",
        )
    build_all(raw, root, start_month="2025-01", end_month="2025-02")
    store = PartitionedIntrabarStore.one_second(root)

    january = store.load_span(
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-02-01", tz="UTC"),
    )

    assert len(january) == 1
    assert january.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert store.loaded_months == ("2025-01",)
    with pytest.raises(ValueError, match="sealed boundary"):
        store.load_span(
            pd.Timestamp("2026-03-01", tz="UTC"),
            pd.Timestamp("2026-05-01", tz="UTC"),
        )


def test_run_state_is_atomic_and_preserves_failure_traceback(tmp_path: Path):
    from experiments.catboost_execution_resolution import write_run_state

    path = tmp_path / "run_state.json"
    write_run_state(path, status="running", detail={"arm": "1s"})
    assert json.loads(path.read_text(encoding="utf-8"))["status"] == "running"

    write_run_state(path, status="failed", detail={"traceback": "boom"})
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["traceback"] == "boom"
    assert not path.with_suffix(".json.part").exists()


def test_continuous_policy_grid_scores_all_66_policies_with_real_execution():
    from experiments.run_catboost_execution_resolution import (
        score_continuous_policy_grid,
    )

    index = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0},
        index=index,
    )
    minute_index = pd.date_range(index[0], periods=8 * 15, freq="1min", tz="UTC")
    execution = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0},
        index=minute_index,
    )
    execution.loc[index[1] + pd.Timedelta(minutes=1), "high"] = 102.0
    predictions = pd.DataFrame(
        {
            "timestamp": index,
            "pred": [2, 1, 1, 1, 1, 1, 1, 1],
            "confidence": [0.9] * len(index),
        }
    )
    regimes = pd.Series("sideways", index=index)

    grid = score_continuous_policy_grid(
        stage="test",
        width_bps=55,
        candidate_id=0,
        prediction_frame=predictions,
        bars=bars,
        execution=execution,
        regimes=regimes,
        start=index[0],
        end=index[-1] + pd.Timedelta(minutes=15),
        resolution="1m",
        fee_bps=5.0,
    )

    assert len(grid) == 66
    assert set(grid["max_hold"]) == {1, 2}
    assert grid["policy_id"].tolist() == list(range(66))
    assert {"ambiguous_exits", "ambiguous_share"}.issubset(grid.columns)

def test_runner_smoke_writes_one_width_economic_artifacts(tmp_path: Path):
    import numpy as np

    from experiments.catboost_execution_resolution import PartitionedIntrabarStore
    from experiments.run_catboost_execution_resolution import ExecutionResolutionRunner
    from experiments.run_catboost_matched_ablation import PreparedData, RecordingToyModel

    index = pd.date_range("2024-01-01", periods=300, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        index=index,
    )
    minute_index = pd.date_range(index[0], periods=300 * 15, freq="1min", tz="UTC")
    minute = pd.DataFrame(
        {"open": 100.0, "high": 100.2, "low": 99.8, "close": 100.0},
        index=minute_index,
    )
    minute_path = tmp_path / "minute.parquet"
    minute.to_parquet(minute_path)
    X = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series(np.arange(len(index)) % 3, index=index)
    regimes = pd.Series(
        np.asarray(("bull", "sideways", "bear"))[np.arange(len(index)) % 3],
        index=index,
    )
    prepared = PreparedData(
        bars=bars,
        minute=minute,
        features={55: (X, y)},
        regimes=regimes,
        m15_fingerprint="m15-toy",
        minute_fingerprint="minute-toy",
    )
    runner = ExecutionResolutionRunner(
        output_root=tmp_path / "output",
        prediction_root=tmp_path / "predictions",
        store=PartitionedIntrabarStore.one_minute(minute_path),
        prepared=prepared,
        widths=(55,),
        candidates=({"iterations": 1, "depth": 1},),
        candidate_ids=(0,),
        model_factory=RecordingToyModel,
        smoke=True,
        fold_limit=1,
        stage1_only=True,
    )

    result = runner.run()

    assert result["status"] == "stage_1_complete"
    assert result["policy_rows"] == 66
    assert result["winner_rows"] == 1
    assert result["selected_rows"] == 1
    selected = pd.read_parquet(tmp_path / "output" / "selected_candidates_2024.parquet")
    assert selected["objective"].tolist() == ["economic"]
    assert json.loads((tmp_path / "output" / "run_state.json").read_text())["status"] == "complete"

def test_post_selection_model_is_fit_once_on_2024_and_reused_through_2026_q1(tmp_path: Path):
    import numpy as np

    from experiments.catboost_execution_resolution import PartitionedIntrabarStore
    from experiments.run_catboost_execution_resolution import ExecutionResolutionRunner
    from experiments.run_catboost_matched_ablation import PreparedData, RecordingToyModel

    train_index = pd.date_range("2024-01-01", periods=30, freq="15min", tz="UTC")
    test_index = pd.DatetimeIndex(
        ["2025-01-01T00:00Z", "2025-06-30T23:45Z", "2025-07-01T00:00Z", "2026-03-31T23:45Z"]
    )
    index = train_index.append(test_index)
    X = pd.DataFrame({"feature": np.arange(len(index), dtype=float)}, index=index)
    y = pd.Series(np.arange(len(index)) % 3, index=index)
    regimes = pd.Series("sideways", index=index)
    bars = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index
    )
    minute_path = tmp_path / "minute.parquet"
    bars.iloc[:1].to_parquet(minute_path)
    prepared = PreparedData(
        bars=bars,
        minute=bars.iloc[:1],
        features={55: (X, y)},
        regimes=regimes,
        m15_fingerprint="m15-frozen",
        minute_fingerprint="minute-frozen",
    )
    RecordingToyModel.fit_calls = 0
    runner = ExecutionResolutionRunner(
        output_root=tmp_path / "output",
        prediction_root=tmp_path / "predictions",
        store=PartitionedIntrabarStore.one_minute(minute_path),
        prepared=prepared,
        widths=(55,),
        candidates=({"iterations": 1, "depth": 1},),
        candidate_ids=(0,),
        model_factory=RecordingToyModel,
        smoke=True,
        stage1_only=False,
    )

    first, fingerprint = runner._fit_or_load_frozen_predictions(55, 0, X, y, regimes)
    for column in ("timestamp", "train_start", "train_end", "test_start", "test_end"):
        assert first[column].dt.unit == "ns", "Parquet timestamps need a stable precision"
    second, second_fingerprint = runner._fit_or_load_frozen_predictions(55, 0, X, y, regimes)

    assert fingerprint == second_fingerprint
    assert RecordingToyModel.fit_calls == 1
    assert first["train_end"].max() < pd.Timestamp("2025-01-01", tz="UTC")
    assert first["test_start"].nunique() == 1
    assert first["test_start"].iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert first["test_end"].iloc[0] == pd.Timestamp("2026-04-01", tz="UTC")
    pd.testing.assert_frame_equal(first, second)

def test_paired_ledger_comparison_counts_changed_and_unmatched_trades():
    from experiments.run_catboost_execution_resolution import compare_paired_ledgers

    entries = pd.DatetimeIndex(["2025-07-01T00:15Z", "2025-07-01T01:00Z"])
    one_minute = pd.DataFrame(
        {
            "entry_time": entries,
            "side": [1, -1],
            "exit_reason": ["stop_loss", "timeout"],
            "net_return": [-0.01, 0.002],
            "ambiguous_touch": [True, False],
        }
    )
    one_second = pd.DataFrame(
        {
            "entry_time": [entries[0], pd.Timestamp("2025-07-01T01:15Z")],
            "side": [1, -1],
            "exit_reason": ["take_profit", "timeout"],
            "net_return": [0.02, 0.001],
            "ambiguous_touch": [False, False],
        }
    )

    summary, outcomes, transitions = compare_paired_ledgers(one_minute, one_second)

    assert summary == {
        "shared_trades": 1,
        "one_minute_only": 1,
        "one_second_only": 1,
        "changed_exit_reason": 1,
        "changed_net_result": 1,
    }
    assert set(outcomes["match_status"]) == {"shared", "one_minute_only", "one_second_only"}
    assert transitions.to_dict(orient="records") == [
        {"exit_reason_1m": "stop_loss", "exit_reason_1s": "take_profit", "trades": 1}
    ]

def test_one_minute_store_filters_source_rows_at_the_sealed_boundary(tmp_path: Path):
    from experiments.catboost_execution_resolution import PartitionedIntrabarStore

    index = pd.DatetimeIndex(["2026-03-31T23:59Z", "2026-04-01T00:00Z"])
    frame = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=index
    )
    path = tmp_path / "minute.parquet"
    frame.to_parquet(path)
    store = PartitionedIntrabarStore.one_minute(path)

    loaded = store.load_span(
        pd.Timestamp("2026-03-31T23:00Z"), pd.Timestamp("2026-04-01T00:00Z")
    )

    assert loaded.index.tolist() == [pd.Timestamp("2026-03-31T23:59Z")]
