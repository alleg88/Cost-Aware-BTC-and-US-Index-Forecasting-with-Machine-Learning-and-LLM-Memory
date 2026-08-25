from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from evaluation.channel_window_validation import PurgedFold
from experiments.event_window_dataset import EventWindowSequences
from experiments.event_window_tail_dataset import TailDecisionDataset
from experiments.event_window_tail_neural import NeuralTailPrediction, TailNeuralConfig
from experiments.event_window_tail_oof import (
    TailOOFConfig,
    _outer_folds,
    chronological_inner_partitions,
    run_tail_fold,
    run_tail_model_oof,
)
from experiments.event_window_tail_tabular import TabularTailPrediction


def _decision_dataset(
    episode_starts: list[pd.Timestamp],
    *,
    timeout_counts: list[int] | None = None,
) -> TailDecisionDataset:
    windows = len(episode_starts)
    active_bars = 12
    rng = np.random.default_rng(42)
    metadata = pd.DataFrame(
        {
            "window_id": [f"w{i:04d}" for i in range(windows)],
            "channel_episode_id": [f"ep{i:04d}" for i in range(windows)],
            "side": np.where(np.arange(windows) % 2 == 0, "long", "short"),
            "window_start": episode_starts,
            "window_end": [value + pd.Timedelta("1h") for value in episode_starts],
        }
    )
    sequence = rng.normal(size=(windows, 36, 2)).astype(np.float32)
    context = rng.normal(size=(windows, active_bars, 1)).astype(np.float32)
    source_times = np.empty((windows, active_bars), dtype="datetime64[ns]")
    decision_times = np.empty((windows, active_bars), dtype="datetime64[ns]")
    decisions: list[dict[str, object]] = []
    tabular: list[list[float]] = []
    for window, row in metadata.iterrows():
        start = pd.Timestamp(row.window_start)
        requested_timeouts = None if timeout_counts is None else timeout_counts[window]
        for step in range(active_bars):
            decision = start + pd.Timedelta(minutes=5 * step)
            source_times[window, step] = (decision - pd.Timedelta("5min")).tz_localize(None)
            decision_times[window, step] = decision.tz_localize(None)
            if requested_timeouts is None:
                outcome = step % 3
            elif step < requested_timeouts:
                outcome = 2
            else:
                outcome = (step - requested_timeouts) % 2
            timeout = 0.05 + 0.01 * step if outcome == 2 else np.nan
            decisions.append(
                {
                    "window_id": row.window_id,
                    "channel_episode_id": row.channel_episode_id,
                    "side": row.side,
                    "step": step,
                    "source_bar_time": decision - pd.Timedelta("5min"),
                    "decision_time": decision,
                    "label_start": decision,
                    "label_end": decision + pd.Timedelta("20min"),
                    "model_target_valid": True,
                    "outcome_code": outcome,
                    "timeout_net_r": timeout,
                    "tp_net_r": 1.8,
                    "sl_net_r": -1.2,
                }
            )
            tabular.append([window + step / 20.0, np.sin(step)])
    sequences = EventWindowSequences(
        metadata=metadata,
        sequence=sequence,
        context=context,
        source_bar_times=source_times,
        decision_times=decision_times,
        sequence_valid=np.ones((windows, 36), dtype=bool),
        decision_valid=np.ones((windows, active_bars), dtype=bool),
        sequence_features=("s0", "s1"),
        context_features=("c0",),
    )
    return TailDecisionDataset(
        decisions=pd.DataFrame(decisions),
        tabular=np.asarray(tabular, dtype=np.float32),
        tabular_features=("x0", "x1"),
        sequences=sequences,
    )


def _small_dataset() -> TailDecisionDataset:
    starts = list(pd.date_range("2021-01-05", periods=30, freq="10D", tz="UTC"))
    for block in pd.date_range("2022-01-01", "2025-01-01", freq="6MS", tz="UTC"):
        starts.extend([block + pd.Timedelta("10D"), block + pd.Timedelta("40D")])
    return _decision_dataset(starts)


def _fake_tabular(model_name: str, **kwargs) -> TabularTailPrediction:
    score = np.asarray(kwargs["score_x"], dtype=float)
    value = score[:, 0] / max(float(np.nanmax(np.abs(score[:, 0]))), 1.0)
    logits = np.column_stack([-value, 0.5 * value, 0.25 - 0.25 * value])
    return TabularTailPrediction(
        logits=logits,
        timeout_net_r=np.full(len(score), 0.1),
        model_metadata={"model_name": model_name},
    )


def _fake_neural(
    model_name: str,
    fit_tensors,
    early_tensors,
    score_tensors,
    config,
) -> NeuralTailPrediction:
    value = np.asarray(score_tensors.sequence[:, 24:, 0], dtype=float)
    scale = max(float(np.max(np.abs(value))), 1.0)
    value = value / scale
    logits = np.stack([-value, 0.5 * value, 0.25 - 0.25 * value], axis=2)
    return NeuralTailPrediction(
        logits=logits.astype(np.float32),
        timeout_net_r=np.full(value.shape, 0.1, dtype=np.float32),
        scaler_median=np.zeros(3, dtype=np.float32),
        scaler_scale=np.ones(3, dtype=np.float32),
        audit={"model_name": model_name},
    )


@pytest.fixture(autouse=True)
def _fast_models(monkeypatch):
    import experiments.event_window_tail_oof as module

    monkeypatch.setattr(module, "fit_predict_tail_model", _fake_tabular)
    monkeypatch.setattr(module, "fit_predict_neural_tail_model", _fake_neural)


def _small_oof(
    model_name: str,
    *,
    mutate_outer_outcomes: bool = False,
    mutate_calibration_outcomes: bool = False,
):
    data = _small_dataset()
    decisions = data.decisions.copy()
    if mutate_outer_outcomes:
        latest = decisions["decision_time"].ge(pd.Timestamp("2025-01-01", tz="UTC"))
        decisions.loc[latest, "outcome_code"] = (
            decisions.loc[latest, "outcome_code"].to_numpy() + 1
        ) % 3
    if mutate_calibration_outcomes:
        first_train = np.flatnonzero(
            decisions["decision_time"].lt(pd.Timestamp("2022-01-01", tz="UTC"))
        )
        parts = chronological_inner_partitions(
            decisions, first_train, TailOOFConfig().fold
        )
        decisions.loc[parts.calibration, "outcome_code"] = 0
    return run_tail_model_oof(
        model_name,
        replace(data, decisions=decisions),
        TailOOFConfig(neural=TailNeuralConfig(epochs=1, patience=1)),
    )


def test_all_models_receive_identical_outer_score_keys():
    results = [_small_oof(name) for name in ("logreg", "xgboost", "tcn", "gru")]
    expected = set(
        results[0].scores[["window_id", "step"]].itertuples(index=False, name=None)
    )
    assert expected
    assert all(
        set(result.scores[["window_id", "step"]].itertuples(index=False, name=None))
        == expected
        for result in results
    )


def test_real_logreg_fold_emits_finite_probabilities(monkeypatch):
    import experiments.event_window_tail_oof as module
    from experiments.event_window_tail_tabular import (
        fit_predict_tail_model as real_tabular_predict,
    )

    monkeypatch.setattr(module, "fit_predict_tail_model", real_tabular_predict)
    data = _small_dataset()
    decisions = data.decisions
    valid_start = pd.Timestamp("2022-01-01", tz="UTC")
    valid_end = pd.Timestamp("2022-07-01", tz="UTC")
    fold = PurgedFold(
        fold_id="2022H1",
        train=np.flatnonzero(decisions["decision_time"].lt(valid_start)),
        valid=np.flatnonzero(
            decisions["decision_time"].ge(valid_start)
            & decisions["decision_time"].lt(valid_end)
        ),
        train_end=valid_start,
        valid_start=valid_start,
        valid_end=valid_end,
    )
    result = run_tail_fold("logreg", fold, data, TailOOFConfig())
    probabilities = result.scores[["p_sl", "p_tp", "p_timeout"]].to_numpy()
    assert np.isfinite(probabilities).all()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)


def test_fit_early_calibration_and_outer_have_zero_episode_or_live_label_overlap():
    result = _small_oof("logreg")
    overlap_columns = [name for name in result.fold_audit if "overlap" in name]
    assert overlap_columns
    assert result.fold_audit[overlap_columns].eq(0).all().all()


def test_half_open_label_ending_at_next_partition_start_is_not_purged():
    data = _small_dataset()
    decisions = data.decisions.copy()
    train = np.flatnonzero(
        decisions["decision_time"].lt(pd.Timestamp("2022-01-01", tz="UTC"))
    )
    initial = chronological_inner_partitions(decisions, train, TailOOFConfig().fold)
    boundary = decisions.iloc[initial.early]["decision_time"].min()
    candidate = initial.fit[-1]
    decisions.loc[candidate, "label_end"] = boundary
    updated = chronological_inner_partitions(decisions, train, TailOOFConfig().fold)
    assert candidate in updated.fit


def test_censored_outer_row_without_label_interval_stays_on_score_calendar():
    data = _small_dataset()
    decisions = data.decisions.copy()
    censored = decisions["decision_time"].ge(pd.Timestamp("2025-01-01", tz="UTC"))
    position = decisions.index[censored][0]
    expected_key = tuple(decisions.loc[position, ["window_id", "step"]])
    decisions.loc[position, "model_target_valid"] = False
    decisions.loc[position, "outcome_code"] = -1
    decisions.loc[position, ["label_start", "label_end"]] = pd.NaT
    result = run_tail_model_oof(
        "logreg", replace(data, decisions=decisions), TailOOFConfig()
    )
    score_keys = set(
        result.scores[["window_id", "step"]].itertuples(index=False, name=None)
    )
    assert expected_key in score_keys


def test_outer_fold_dates_match_notebook_j():
    result = _small_oof("logreg")
    assert result.fold_audit["fold_id"].tolist() == [
        "2022H1",
        "2022H2",
        "2023H1",
        "2023H2",
        "2024H1",
        "2024H2",
        "2025H1",
    ]


def test_outer_fold_excludes_episode_whose_live_label_crosses_block_end():
    data = _decision_dataset([pd.Timestamp("2022-06-30 22:00", tz="UTC")])
    decisions = data.decisions.copy()
    decisions["label_end"] = pd.Timestamp("2022-07-01 00:05", tz="UTC")
    fold = next(value for value in _outer_folds(decisions) if value.fold_id == "2022H1")
    assert len(fold.valid) == 0


def test_outer_outcome_mutation_cannot_change_outer_score():
    first = _small_oof("logreg")
    second = _small_oof("logreg", mutate_outer_outcomes=True)
    np.testing.assert_allclose(first.scores.ev_score, second.scores.ev_score)
    pd.testing.assert_frame_equal(first.calibration_audit, second.calibration_audit)


def test_calibration_outcome_mutation_changes_calibration_but_not_fit_model():
    first = _small_oof("logreg")
    second = _small_oof("logreg", mutate_calibration_outcomes=True)
    assert first.fold_audit.loc[0, "fit_model_hash"] == second.fold_audit.loc[0, "fit_model_hash"]
    assert first.calibration_audit.loc[0, "temperature"] != second.calibration_audit.loc[0, "temperature"]


def test_matched_rate_threshold_is_fold_local_nonnegative_and_audited():
    result = _small_oof("logreg")
    assert result.scores["matched_rate_threshold"].ge(0.0).all()
    assert result.scores.groupby("fold_id")["matched_rate_threshold"].nunique().eq(1).all()
    assert result.calibration_audit["matched_rate_threshold"].ge(0.0).all()
    assert result.calibration_audit["matched_rate_calendar_days"].gt(0).all()


def _boundary_result(model_name: str, timeout_rows: int, timeout_episodes: int):
    fit_counts = [0] * 20
    remaining = timeout_rows
    for index in range(timeout_episodes):
        take = min(12, remaining - max(0, timeout_episodes - index - 1))
        fit_counts[index] = take
        remaining -= take
    assert remaining == 0
    starts = list(pd.date_range("2021-01-01", periods=29, freq="10D", tz="UTC"))
    starts.append(pd.Timestamp("2022-02-01", tz="UTC"))
    data = _decision_dataset(starts, timeout_counts=fit_counts + [4] * 10)
    split = 29 * 12
    fold = PurgedFold(
        fold_id="2022H1",
        train=np.arange(split),
        valid=np.arange(split, split + 12),
        train_end=pd.Timestamp("2022-01-01", tz="UTC"),
        valid_start=pd.Timestamp("2022-01-01", tz="UTC"),
        valid_end=pd.Timestamp("2022-07-01", tz="UTC"),
    )
    return run_tail_fold(model_name, fold, data, TailOOFConfig())


def test_timeout_fallback_flag_is_written_per_fold():
    result = _boundary_result("xgboost", timeout_rows=10, timeout_episodes=10)
    assert result.calibration_audit["timeout_fallback"].all()


@pytest.mark.parametrize("model_name", ["xgboost", "tcn", "gru"])
def test_timeout_head_support_boundary_is_exact(model_name):
    assert _boundary_result(model_name, 199, 20).calibration_audit.timeout_fallback.all()
    assert _boundary_result(model_name, 200, 19).calibration_audit.timeout_fallback.all()
    assert not _boundary_result(model_name, 200, 20).calibration_audit.timeout_fallback.any()


def test_model_name_rejects_unknown_value():
    with pytest.raises(ValueError, match="model_name"):
        run_tail_model_oof("catboost", _small_dataset(), TailOOFConfig())
