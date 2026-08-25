from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.event_window_tail_policy import (
    REFERENCE_ATTEMPTS,
    REFERENCE_CALENDAR_DAYS,
    TailPolicyResult,
    compare_tail_models,
    episode_pair_bootstrap,
    frequency_noninferior,
    matched_count_diagnostic,
    paired_window_ledger,
    replay_positive_ev,
    replay_same_entries,
)


START = pd.Timestamp("2022-01-01", tz="UTC")
END = pd.Timestamp("2025-07-01", tz="UTC")


def _labels(
    count: int = 3,
    *,
    rr: float = 2.0,
    first_path_observed: bool = True,
) -> pd.DataFrame:
    times = START + pd.to_timedelta(np.arange(count) % 1000, unit="min")
    observed = np.ones(count, dtype=bool)
    observed[0] = first_path_observed
    return pd.DataFrame(
        {
            "window_id": ["w1"] * count,
            "channel_episode_id": ["episode-1"] * count,
            "side": ["long"] * count,
            "step": np.arange(count),
            "decision_time": times,
            "entry_time": times,
            "geometry_valid": True,
            "path_observed": observed,
            "model_target_valid": observed,
            "outcome": np.where(observed, "tp", "censored"),
            "r_net": np.where(observed, rr - 0.1, np.nan),
        }
    )


def _unique_window_labels(count: int) -> pd.DataFrame:
    frame = _labels(count)
    frame["window_id"] = [f"w{number:04d}" for number in range(count)]
    frame["channel_episode_id"] = [f"e{number // 2:04d}" for number in range(count)]
    frame["step"] = 0
    return frame


def _scores(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "window_id": ["w1"] * len(values),
            "step": np.arange(len(values)),
            "ev_score": values,
        }
    )


def test_primary_policy_takes_first_nonnegative_ev_only():
    replay = replay_positive_ev(_scores([-0.2, 0.1, 0.4]), _labels(), START, END)
    assert replay.trades[["window_id", "step"]].to_records(index=False).tolist() == [
        ("w1", 1)
    ]


def test_negative_ev_is_never_added_for_frequency():
    replay = replay_positive_ev(_scores([-0.4, -0.1]), _labels(2), START, END)
    assert replay.attempted_trades == 0


def test_one_trade_per_window_is_hard():
    replay = replay_positive_ev(_scores([0.1, 0.2, 0.3]), _labels(), START, END)
    assert replay.trades.groupby("window_id").size().max() == 1


def test_rr3_uses_exact_rr2_keys():
    rr2 = replay_positive_ev(_scores([0.1, 0.2]), _labels(2, rr=2.0), START, END)
    rr3 = replay_same_entries(rr2.trades, _labels(2, rr=3.0))
    assert set(
        rr2.trades[["window_id", "step"]].itertuples(index=False, name=None)
    ) == set(rr3[["window_id", "step"]].itertuples(index=False, name=None))


def test_censored_first_crossing_consumes_window():
    replay = replay_positive_ev(
        _scores([0.1, 0.4]),
        _labels(2, first_path_observed=False),
        START,
        END,
    )
    assert replay.attempted_trades == 1
    assert replay.observed_trades == 0
    assert replay.trades.step.tolist() == [0]


def test_frozen_frequency_floor_uses_integer_count_not_rounded_rate():
    assert frequency_noninferior(attempts=1448, calendar_days=1277)
    assert not frequency_noninferior(attempts=1447, calendar_days=1277)
    assert not frequency_noninferior(attempts=1448, calendar_days=1276)
    assert 1448 / 1277 < 1.1339076


def test_reference_calendar_is_zero_filled_before_frequency_is_calculated():
    replay = replay_positive_ev(_scores([0.1]), _labels(1), START, END)
    assert len(replay.daily) == REFERENCE_CALENDAR_DAYS
    assert replay.daily.at[START, "attempted_trades"] == 1
    assert replay.daily["attempted_trades"].sum() == 1
    assert replay.trades_per_day == 1 / REFERENCE_CALENDAR_DAYS


def test_matched_threshold_tie_break_is_nonnegative_and_conservative():
    labels = _unique_window_labels(4)
    scores = pd.DataFrame(
        {
            "window_id": labels.window_id,
            "step": 0,
            "ev_score": [0.1, 0.1, 0.2, 0.2],
        }
    )
    result = matched_count_diagnostic(scores, labels, target_attempts=3)
    assert result.threshold >= 0.0
    assert result.threshold == 0.2


def test_posthoc_matched_count_never_selects_winner():
    labels = _unique_window_labels(2000)
    scores = pd.DataFrame(
        {
            "window_id": labels.window_id,
            "step": 0,
            "ev_score": np.arange(2000, dtype=float),
        }
    )
    result = matched_count_diagnostic(
        scores, labels, target_attempts=REFERENCE_ATTEMPTS
    )
    assert result.attempted_trades == REFERENCE_ATTEMPTS
    assert result.exploratory and not result.can_select


def _manual_replay(
    name: str,
    trades: pd.DataFrame,
    universe: pd.DataFrame,
    *,
    attempts: int,
    mean: float,
    total: float,
) -> TailPolicyResult:
    trades = trades.copy()
    trades.attrs["window_universe"] = universe.copy()
    return TailPolicyResult(
        model_name=name,
        trades=trades,
        daily=pd.DataFrame(),
        attempted_trades=attempts,
        observed_trades=int(trades.get("path_observed", pd.Series(dtype=bool)).sum()),
        trades_per_day=attempts / REFERENCE_CALENDAR_DAYS,
        mean_net_r=mean,
        total_net_r=total,
        frequency_noninferior=frequency_noninferior(attempts, REFERENCE_CALENDAR_DAYS),
    )


def test_paired_ledger_uses_full_window_universe_and_zero_for_wait_or_censor():
    universe = pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w3"],
            "channel_episode_id": ["e1", "e1", "e2"],
        }
    )
    trades = pd.DataFrame(
        {
            "window_id": ["w1", "w2"],
            "channel_episode_id": ["e1", "e1"],
            "step": [0, 0],
            "outcome": ["tp", "censored"],
            "path_observed": [True, False],
            "r_net": [1.0, np.nan],
        }
    )
    replay = _manual_replay("logreg", trades, universe, attempts=2, mean=1.0, total=1.0)
    ledger = paired_window_ledger({"logreg": replay})
    assert len(ledger) == len(universe)
    assert ledger.loc[ledger.outcome.isin(["wait", "censored"]), "net_r"].eq(0.0).all()
    assert ledger.set_index("window_id").at["w3", "outcome"] == "wait"


def test_episode_bootstrap_resamples_whole_episode_deltas_deterministically():
    ledger = pd.DataFrame(
        {
            "window_id": ["w1", "w2", "w3"],
            "channel_episode_id": ["e1", "e1", "e2"],
            "candidate_net_r": [1.0, 2.0, -1.0],
            "comparator_net_r": [0.0, 0.0, 0.0],
        }
    )
    first = episode_pair_bootstrap(ledger, draws=50, seed=7)
    second = episode_pair_bootstrap(ledger, draws=50, seed=7)
    assert first.point_delta_total_net_r == 2.0
    assert first.draws.equals(second.draws)


def test_comparator_excludes_frequency_failures_and_breaks_tie_by_simplicity():
    candidates = {
        "logreg": {
            "attempted_trades": 1500,
            "calendar_days": 1277,
            "mean_net_r": 0.1,
            "total_net_r": 10.0,
            "paired_delta_total_net_r": 5.0,
            "paired_delta_ci_low": 1.0,
        },
        "xgboost": {
            "attempted_trades": 1500,
            "calendar_days": 1277,
            "mean_net_r": 0.1,
            "total_net_r": 10.0,
            "paired_delta_total_net_r": 5.0,
            "paired_delta_ci_low": 1.0,
        },
        "tcn": {
            "attempted_trades": 1447,
            "calendar_days": 1277,
            "mean_net_r": 2.0,
            "total_net_r": 100.0,
            "paired_delta_total_net_r": 95.0,
            "paired_delta_ci_low": 10.0,
        },
    }
    references = {
        "first_entry": {
            "attempted_trades": 1448,
            "calendar_days": 1277,
            "mean_net_r": 0.01,
            "total_net_r": 5.0,
        }
    }
    comparison = compare_tail_models(candidates, references)
    assert comparison.strongest_comparator == "logreg"
    assert comparison.chosen_model == "logreg"
    assert "frequency" in comparison.rejection_reasons["tcn"]


def test_comparison_computes_paired_episode_delta_from_replays():
    universe = pd.DataFrame(
        {
            "window_id": ["w1", "w2"],
            "channel_episode_id": ["e1", "e2"],
        }
    )
    reference_trades = pd.DataFrame(
        {
            "window_id": ["w1", "w2"],
            "channel_episode_id": ["e1", "e2"],
            "step": [0, 0],
            "outcome": ["timeout", "timeout"],
            "path_observed": [True, True],
            "r_net": [0.0, 0.0],
        }
    )
    candidate_trades = reference_trades.copy()
    candidate_trades["r_net"] = [1.0, 2.0]
    reference = _manual_replay(
        "first_entry", reference_trades, universe, attempts=1448, mean=0.01, total=0.1
    )
    candidate = _manual_replay(
        "logreg", candidate_trades, universe, attempts=1448, mean=1.5, total=3.0
    )
    comparison = compare_tail_models({"logreg": candidate}, {"first_entry": reference})
    assert comparison.passes["logreg"]
    row = comparison.table.iloc[0]
    assert row.paired_delta_total_net_r == 3.0
    assert row.paired_delta_ci_low > 0.0


def test_frozen_constants_are_exact():
    assert (REFERENCE_ATTEMPTS, REFERENCE_CALENDAR_DAYS) == (1448, 1277)
