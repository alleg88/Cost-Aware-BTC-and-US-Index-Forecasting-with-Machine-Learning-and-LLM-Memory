from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.event_window_dataset import EventWindowSequences
from experiments.event_window_tcn import (
    EventWindowTCN,
    TCNConfig,
    _half_open_interval_uniqueness,
    _inner_episode_split,
    run_event_window_oof,
    within_window_pairwise_loss,
)


def test_later_active_bars_cannot_change_earlier_scores():
    model = EventWindowTCN(5, 4, TCNConfig()).eval()
    sequence = torch.randn(2, 36, 5)
    context = torch.randn(2, 12, 4)
    first = model(sequence, context).detach()
    sequence[:, 31:, :] += 1000
    context[:, 7:, :] -= 1000
    second = model(sequence, context).detach()
    torch.testing.assert_close(first[:, :7], second[:, :7])


def test_first_score_has_gradient_to_earliest_pre_window_bar():
    model = EventWindowTCN(5, 4, TCNConfig()).eval()
    sequence = torch.randn(1, 36, 5, requires_grad=True)
    model(sequence, torch.zeros(1, 12, 4))[0, 0].backward()
    assert sequence.grad is not None
    assert sequence.grad[0, 0].abs().sum() > 0


def test_model_is_small_and_emits_twelve_scores():
    model = EventWindowTCN(17, 25, TCNConfig())
    assert sum(parameter.numel() for parameter in model.parameters()) < 100_000
    output = model(torch.zeros(3, 36, 17), torch.zeros(3, 12, 25))
    assert output.shape == (3, 12)


def test_pairwise_loss_penalises_reversed_within_window_order():
    target = torch.tensor([[1.0, -1.0]])
    mask = torch.ones_like(target, dtype=torch.bool)
    good = within_window_pairwise_loss(torch.tensor([[0.8, -0.8]]), target, mask)
    bad = within_window_pairwise_loss(torch.tensor([[-0.8, 0.8]]), target, mask)
    assert good < bad


def _oof_fixture() -> tuple[EventWindowSequences, pd.DataFrame]:
    windows = []
    for year, half, count in ((2021, 1, 4), (2021, 2, 4), (2022, 1, 4), (2022, 2, 4)):
        month = 2 if half == 1 else 8
        for number in range(count):
            windows.append(
                {
                    "window_id": f"w-{year}-{half}-{number}",
                    "channel_episode_id": f"ep-{year}-{half}-{number // 2}",
                    "side": "long" if number % 2 == 0 else "short",
                    "window_start": pd.Timestamp(year, month, 1 + number, tz="UTC"),
                    "window_end": pd.Timestamp(year, month, 1 + number, tz="UTC")
                    + pd.Timedelta("60min"),
                    "source_bar_time": pd.Timestamp(year, month, 1 + number, tz="UTC")
                    - pd.Timedelta("5min"),
                }
            )
    metadata = pd.DataFrame(windows)
    rng = np.random.default_rng(42)
    n = len(metadata)
    sequence = rng.normal(size=(n, 36, 5)).astype(np.float32)
    context = rng.normal(size=(n, 12, 4)).astype(np.float32)
    sequence_valid = np.ones((n, 36), dtype=bool)
    decision_valid = np.ones((n, 12), dtype=bool)
    source_times = np.empty((n, 12), dtype="datetime64[ns]")
    decision_times = np.empty((n, 12), dtype="datetime64[ns]")
    labels = []
    for row, window in metadata.iterrows():
        start = pd.Timestamp(window["window_start"])
        for step in range(12):
            decision = start + pd.Timedelta(minutes=5 * step)
            source = decision - pd.Timedelta("5min")
            source_times[row, step] = source.tz_localize(None).to_datetime64()
            decision_times[row, step] = decision.tz_localize(None).to_datetime64()
            labels.append(
                {
                    "window_id": window["window_id"],
                    "channel_episode_id": window["channel_episode_id"],
                    "step": step,
                    "decision_time": decision,
                    "label_start": decision,
                    "label_end": decision + pd.Timedelta("10min"),
                    "geometry_valid": True,
                    "model_target_valid": True,
                    "r_net": float(np.sin(step / 3.0) + (0.1 if window["side"] == "long" else -0.1)),
                }
            )
    data = EventWindowSequences(
        metadata=metadata,
        sequence=sequence,
        context=context,
        source_bar_times=source_times,
        decision_times=decision_times,
        sequence_valid=sequence_valid,
        decision_valid=decision_valid,
        sequence_features=tuple(f"s{i}" for i in range(5)),
        context_features=tuple(f"c{i}" for i in range(4)),
    )
    return data, pd.DataFrame(labels)


def test_oof_never_shares_channel_episode_or_live_label():
    sequences, labels = _oof_fixture()
    result = run_event_window_oof(
        sequences,
        labels,
        TCNConfig(epochs=2, patience=1, batch_size=4),
    )
    assert not result.scores.empty
    assert result.fold_audit["train_valid_episode_overlap"].eq(0).all()
    assert result.fold_audit["live_label_overlap"].eq(0).all()
    assert result.fold_audit["inner_early_live_label_overlap"].eq(0).all()
    assert result.scores[["window_id", "step", "score"]].notna().all().all()


def test_validation_extreme_does_not_enter_train_scaler():
    sequences, labels = _oof_fixture()
    validation = sequences.metadata["window_start"].dt.year.eq(2022).to_numpy()
    sequences.sequence[validation] += 10_000.0
    result = run_event_window_oof(
        sequences,
        labels,
        TCNConfig(epochs=1, patience=1, batch_size=4),
    )
    assert result.sequence_scalers
    assert max(np.abs(item["median"]).max() for item in result.sequence_scalers) < 100.0


def test_oof_rejects_a_label_on_the_wrong_decision_clock():
    sequences, labels = _oof_fixture()
    labels.loc[0, "decision_time"] += pd.Timedelta("5min")
    with pytest.raises(ValueError, match="decision_time"):
        run_event_window_oof(sequences, labels, TCNConfig(epochs=1, patience=1))


def test_inner_early_stop_split_purges_still_live_training_labels():
    metadata = pd.DataFrame(
        {
            "window_id": ["train", "early"],
            "channel_episode_id": ["ep-train", "ep-early"],
            "window_start": pd.to_datetime(
                ["2021-01-01 00:00", "2021-01-01 01:00"], utc=True
            ),
        }
    )
    labels = pd.DataFrame(
        {
            "window_id": ["train", "early"],
            "channel_episode_id": ["ep-train", "ep-early"],
            "geometry_valid": [True, True],
            "label_end": pd.to_datetime(
                ["2021-01-01 02:00", "2021-01-01 03:00"], utc=True
            ),
        }
    )
    inner, early = _inner_episode_split(np.array([0, 1]), metadata, labels)
    assert inner.size == 0
    np.testing.assert_array_equal(early, [1])


def test_uniqueness_treats_adjacent_half_open_labels_as_nonoverlapping():
    labels = pd.DataFrame(
        {
            "label_start": pd.to_datetime(
                ["2021-01-01 00:00", "2021-01-01 00:01"], utc=True
            ),
            "label_end": pd.to_datetime(
                ["2021-01-01 00:01", "2021-01-01 00:03"], utc=True
            ),
        }
    )
    weights = _half_open_interval_uniqueness(labels, np.array([0, 1]))
    np.testing.assert_allclose(weights, [1.0, 1.0])
