import json

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from experiments.catboost_execution_scoring import simulate_policy
from experiments.raw_hold_control import MODEL_NAMES
from reflection_agent.ensemble import (
    BAR, WeightChoice, candidate_outcomes, hedge_weights, memory_cards,
    messages_for, replay_signals, shuffle_cards, weight_signal,
)
from experiments.rq4_ensemble_reader import _replay


def test_weights_reject_missing_nonfinite_and_nonunit_allocations():
    weights = dict.fromkeys(MODEL_NAMES, 0.0)
    weights[MODEL_NAMES[0]] = 1.0
    WeightChoice.model_validate({"weights": weights, "reason": "valid"})
    for changed in ({**weights, MODEL_NAMES[0]: float("nan")},
                    {**weights, MODEL_NAMES[0]: 0.9},
                    {**weights, MODEL_NAMES[1]: -0.1},
                    {k: v for k, v in weights.items() if k != MODEL_NAMES[-1]}):
        with pytest.raises(ValidationError):
            WeightChoice.model_validate({"weights": changed, "reason": "invalid"})


def test_decimal_sum_tolerance_accepts_boundary_and_rejects_outside():
    weights = dict.fromkeys(MODEL_NAMES, 0.111111)
    assert abs(sum(weights.values()) - 1.0) > 1e-6
    WeightChoice.model_validate({"weights": weights, "reason": "decimal boundary"})
    weights[MODEL_NAMES[0]] -= 0.00000001
    with pytest.raises(ValidationError):
        WeightChoice.model_validate({"weights": weights, "reason": "outside tolerance"})


def test_memory_uses_actual_closed_availability_and_excludes_partial_windows():
    decision = pd.Timestamp("2024-02-01", tz="UTC")
    ledger = pd.DataFrame({
        "available_time": [decision - pd.Timedelta(minutes=1), decision,
                           decision + pd.Timedelta(minutes=1)],
        "net_return": [-0.01, 99.0, 99.0],
    })
    ledgers = dict.fromkeys(MODEL_NAMES, ledger)
    cards = memory_cards(ledgers, decision, decision - pd.Timedelta(days=7))
    assert len(cards) == 1
    assert cards[0]["values"] == [[-100.0, 1, 100.0]] * 9
    assert memory_cards(ledgers, decision, decision - pd.Timedelta(days=6)) == []
    message = json.loads(messages_for({"probabilities": []}, [])[1]["content"])
    assert message["memory"] == []
    assert "previous_weights" not in message["current_state"]


def test_shuffle_breaks_identity_without_altering_numbers_and_hedge_responds():
    card = {"age_weeks": 1, "values": [[i * 10.0, i, i * 2.0] for i in range(9)]}
    shuffled = shuffle_cards([card], 42)[0]
    assert sorted(shuffled["values"]) == sorted(card["values"])
    assert all(a != b for a, b in zip(shuffled["values"], card["values"]))
    assert shuffled == shuffle_cards([card], 42)[0]
    weights = hedge_weights([card], eta=1)
    assert np.isclose(weights.sum(), 1)
    assert weights[-1] > weights[0]
    assert np.allclose(hedge_weights([], eta=1), np.ones(9) / 9)


def test_candidate_replay_matches_engine_and_preserves_week_boundary_occupancy():
    start = pd.Timestamp("2024-01-01", tz="UTC")
    grid = pd.date_range(start, periods=12, freq="15min")
    minute_grid = pd.date_range(start, periods=180, freq="min")
    minute = pd.DataFrame({"open": 100.0, "high": 100.4, "low": 99.7,
                           "close": 100 + np.sin(np.arange(180)) * 0.2}, index=minute_grid)
    bars = minute.resample("15min").agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    end = start + len(grid) * BAR
    paths = candidate_outcomes(bars, minute, start, end)
    sides = pd.Series([1, -1, 0, 1, -1, 1, 1, -1, 0, -1, 1, 0], index=grid)
    direct, direct_returns = simulate_policy(
        bars=bars, execution=minute,
        prediction_frame=pd.DataFrame({"timestamp": grid, "pred": sides.to_numpy() + 1, "confidence": 1.0}),
        start=start, end=end, resolution="1m", tau=0, tp_bps=150,
        sl_bps=100, max_hold=1, fee_bps=5,
    )
    replay, returns = replay_signals(grid, sides, paths)
    pd.testing.assert_frame_equal(replay[direct.columns], direct)
    np.testing.assert_allclose(returns, direct_returns, atol=1e-15)
    reader_ledger, reader_returns = _replay(grid, sides, pd.Series(150, index=grid), paths)
    pd.testing.assert_frame_equal(reader_ledger[direct.columns], direct)
    np.testing.assert_allclose(reader_returns, direct_returns, atol=1e-15)
    # A failure-policy switch must not re-admit the preceding trade's entry bar.
    tp = pd.Series(np.where(np.arange(12) < 4, 150, 200), index=grid)
    switched, _ = replay_signals(grid, sides, paths, tp)
    assert grid[4] not in set(switched.signal_time)
    assert switched.loc[switched.signal_time >= grid[4], "policy_tp_bps"].eq(200).all()


def test_flat_is_an_ordinary_abstention():
    probabilities = np.zeros((2, 9, 3))
    probabilities[:, :, 1] = 0.9
    probabilities[:, :, 2] = 0.1
    assert np.array_equal(weight_signal(probabilities, np.ones((2, 9)) / 9), [0, 0])


def test_current_state_cannot_see_future_bars_or_forecasts():
    from experiments.run_reflection_ensemble import current_state
    grid = pd.date_range("2024-01-01", periods=800, freq="15min", tz="UTC")
    bars = pd.DataFrame({"close": np.arange(800) + 1000.0, "volume": 10.0}, index=grid)
    times = grid + BAR
    probabilities = np.tile([.1, .7, .2], (800, 9, 1))
    decision = grid[700]
    before = current_state(bars, times, probabilities, decision)
    bars.loc[grid >= decision, "close"] = 999999
    bars.loc[grid >= decision, "volume"] = 999999
    probabilities[times > decision] = [.9, .05, .05]
    assert current_state(bars, times, probabilities, decision) == before
