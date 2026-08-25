from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest


def _runner():
    try:
        return importlib.import_module(
            "experiments.run_event_window_economic_feasibility"
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"Notebook Q runner is not implemented: {error}")


def _attempt(decision_time: str = "2024-01-01 00:00:00+00:00") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "activation_key": ["a1"],
            "window_id": ["w1"],
            "step": [0],
            "channel_episode_id": ["e1"],
            "decision_time": [pd.Timestamp(decision_time)],
            "channel_side": ["long"],
            "adaptive_barrier_bps": [100.0],
            "reference_price": [100.0],
        }
    )


def test_protocol_freezes_q_economic_ceiling_and_sealed_periods():
    runner = _runner()
    protocol = runner.protocol_dict()
    assert protocol["frozen_p_run_hash"] == "0474798f6d0eb56e64d3"
    assert protocol["primary_activation_target_per_day"] == 2.0
    assert protocol["activation_rate_sensitivities_per_day"] == [3.0]
    assert protocol["primary_hold_minutes"] == 60
    assert protocol["hold_sensitivities_minutes"] == [120]
    assert protocol["primary_target_multiple_b"] == 2.0
    assert protocol["target_sensitivities_b"] == [3.0, 5.0]
    assert protocol["entry_cost_bps"] == 5.0
    assert protocol["target_exit_cost_bps"] == 2.0
    assert protocol["other_exit_cost_bps"] == 5.0
    assert protocol["direction_head_trained"] is False
    assert protocol["forward_or_lockbox_loaded"] is False


def test_non_dev_stage_is_rejected_before_frozen_or_market_loading(monkeypatch):
    runner = _runner()
    called = False

    def fail_if_called(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("frozen handoff should not be loaded")

    monkeypatch.setattr(runner, "load_frozen_p_artifacts", fail_if_called)
    with pytest.raises(ValueError, match="development only"):
        runner.run_economic_feasibility(stage="forward")
    assert called is False


def test_same_minute_target_and_stop_is_resolved_as_stop():
    runner = _runner()
    minute = pd.DataFrame(
        {
            "open": [100.0],
            "high": [103.0],
            "low": [98.0],
            "close": [101.0],
        },
        index=pd.DatetimeIndex([pd.Timestamp("2024-01-01 00:00:00+00:00")]),
    )
    result = runner.replay_brackets(
        _attempt(), minute, target_multiples=(2.0,), hold_minutes=(1,)
    )
    assert set(result["outcome"]) == {"sl"}
    assert np.allclose(result["gross_r"], -1.0)
    assert np.allclose(result["cost_bps"], 10.0)


def test_timeout_uses_last_close_and_outcome_specific_cost():
    runner = _runner()
    minute = pd.DataFrame(
        {
            "open": [100.0, 100.1, 100.4],
            "high": [100.5, 100.6, 101.2],
            "low": [99.5, 99.7, 100.0],
            "close": [100.1, 100.4, 101.0],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="1min", tz="UTC"),
    )
    result = runner.replay_brackets(
        _attempt(), minute, target_multiples=(2.0,), hold_minutes=(3,)
    )
    long = result.loc[result.direction.eq("long")].iloc[0]
    expected_gross_bps = np.log(101.0 / 100.0) * 1e4
    assert long.outcome == "timeout"
    assert long.gross_bps == pytest.approx(expected_gross_bps)
    assert long.net_bps == pytest.approx(expected_gross_bps - 10.0)
    assert long.net_r == pytest.approx((expected_gross_bps - 10.0) / 100.0)


def test_missing_minute_censors_attempt_instead_of_imputing_it():
    runner = _runner()
    minute = pd.DataFrame(
        {
            "open": [100.0, 100.2],
            "high": [100.5, 100.7],
            "low": [99.5, 99.8],
            "close": [100.1, 100.3],
        },
        index=pd.DatetimeIndex(
            [
                pd.Timestamp("2024-01-01 00:00:00+00:00"),
                pd.Timestamp("2024-01-01 00:02:00+00:00"),
            ]
        ),
    )
    result = runner.replay_brackets(
        _attempt(), minute, target_multiples=(2.0,), hold_minutes=(3,)
    )
    assert result["censored"].all()
    assert result["outcome"].eq("censored").all()
    assert result["net_r"].isna().all()


def test_direction_scenarios_keep_causal_and_stress_results_distinct():
    runner = _runner()
    paths = pd.DataFrame(
        {
            "activation_key": ["a1", "a1"],
            "window_id": ["w1", "w1"],
            "step": [0, 0],
            "channel_episode_id": ["e1", "e1"],
            "decision_time": [
                pd.Timestamp("2024-01-01", tz="UTC"),
                pd.Timestamp("2024-01-01", tz="UTC"),
            ],
            "channel_side": ["long", "long"],
            "direction": ["long", "short"],
            "target_multiple_b": [2.0, 2.0],
            "hold_minutes": [60, 60],
            "censored": [False, False],
            "outcome": ["tp", "sl"],
            "gross_r": [2.07, -1.0],
            "net_r": [2.0, -1.1],
            "cost_bps": [7.0, 10.0],
            "cost_r": [0.07, 0.1],
        }
    )
    scenarios = runner.build_direction_scenarios(paths)
    values = scenarios.set_index("scenario")["net_r"]
    assert values["channel_side"] == pytest.approx(2.0)
    assert values["random_50"] == pytest.approx(0.45)
    assert values["direction_70"] == pytest.approx(1.07)
    assert values["oracle"] == pytest.approx(2.0)
    assert scenarios.groupby("scenario")[
        ["tp_weight", "sl_weight", "timeout_weight"]
    ].sum().sum(axis=1).eq(1.0).all()
    assert scenarios["best_net_r"].eq(2.0).all()
    assert scenarios["worst_net_r"].eq(-1.1).all()


def test_cluster_bootstrap_resamples_whole_episodes():
    runner = _runner()
    values = pd.DataFrame(
        {
            "channel_episode_id": ["e1", "e1", "e2"],
            "net_r": [1.0, 1.0, -1.0],
        }
    )
    draws = runner.cluster_bootstrap_draws(
        values,
        value_column="net_r",
        episode_column="channel_episode_id",
        draws=200,
        seed=42,
    )
    assert set(np.round(draws, 12)).issubset({-1.0, round(1.0 / 3.0, 12), 1.0})
    assert {-1.0, round(1.0 / 3.0, 12), 1.0}.issubset(
        set(np.round(draws, 12))
    )


def test_frozen_p_crossing_reconstruction_matches_registered_counts():
    runner = _runner()
    frozen = runner.load_frozen_p_artifacts()
    ledger = runner.reconstruct_activation_ledger(frozen)
    counts = ledger.groupby(["arm", "target_activations_per_day"]).size()
    assert counts[("conditional", 2.0)] == 2_148
    assert counts[("conditional", 3.0)] == 3_780
    assert counts[("anchored_empirical", 2.0)] == 2_379
    assert counts[("anchored_empirical", 3.0)] == 3_505
    assert not ledger.duplicated(
        ["arm", "target_activations_per_day", "channel_episode_id", "decision_time"]
    ).any()


def test_frozen_p_loader_is_independent_of_a_newer_latest_pointer(monkeypatch):
    runner = _runner()
    read_json = runner._read_json

    def read_with_advanced_latest(path):
        if path.name == "latest_dev.json":
            return {
                "run_hash": "newer-causal-replication",
                "protocol_hash": "newer-protocol",
                "relative_path": "newer-causal-replication/full",
            }
        return read_json(path)

    monkeypatch.setattr(runner, "_read_json", read_with_advanced_latest)
    frozen = runner.load_frozen_p_artifacts()
    assert frozen.run_hash == runner.FROZEN_P_RUN_HASH


def test_completed_smoke_publishes_hashed_economics_and_leakage_artifacts(tmp_path):
    runner = _runner()
    result = runner.run_economic_feasibility(smoke=True, run_root=tmp_path)
    assert result.summary["frozen_p_run_hash"] == "0474798f6d0eb56e64d3"
    assert result.summary["frozen_activation_counts"] == {
        "anchored_empirical_2": 2_379,
        "anchored_empirical_3": 3_505,
        "conditional_2": 2_148,
        "conditional_3": 3_780,
    }
    assert result.summary["direction_head_trained"] is False
    assert result.summary["timing_model_refit"] is False
    assert result.summary["economics_evaluated"] is True
    assert result.summary["forward_or_lockbox_loaded"] is False
    leakage = pd.read_csv(result.run_dir / "leakage_audit.csv")
    assert leakage["passed"].astype(bool).all()
    assert set(runner.READER_ARTIFACTS).issubset(
        {path.name for path in result.run_dir.iterdir()}
    )
    state = runner._read_json(result.run_dir / "run_state.json")
    assert state["status"] == "complete"
    assert state["summary"] == result.summary
