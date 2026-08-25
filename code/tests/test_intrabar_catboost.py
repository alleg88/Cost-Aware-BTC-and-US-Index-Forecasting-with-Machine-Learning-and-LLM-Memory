import numpy as np
import pandas as pd

from experiments.run_intrabar_catboost import (
    apply_meta_filter,
    calibrate_meta_threshold,
    calibration_probability_thresholds,
    experiment_masks,
    outcome_safe_signal_mask,
    weekly_meta_probabilities,
)


class RecordingModel:
    def __init__(self, fits: list[list[pd.Timestamp]]):
        self.fits = fits
        self.classes_ = np.array([0, 1, 2])

    def fit(self, X, y):
        self.fits.append(list(X.index))
        return self

    def predict_proba(self, X):
        return np.tile([0.2, 0.2, 0.6], (len(X), 1))


def test_weekly_meta_fit_uses_only_outcomes_closed_before_prediction_week():
    index = pd.DatetimeIndex(
        ["2025-01-01", "2025-01-02", "2025-01-08", "2025-01-15"], tz="UTC"
    )
    candidates = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0, 4.0],
            "outcome": [0, 2, 1, 2],
            "outcome_close_time": pd.to_datetime(
                ["2025-01-07", "2025-01-07", "2025-01-09", "2025-01-16"], utc=True
            ),
            "prediction_week_start": pd.to_datetime(
                ["2025-01-01", "2025-01-01", "2025-01-08", "2025-01-15"], utc=True
            ),
        },
        index=index,
    )
    fits: list[list[pd.Timestamp]] = []

    probabilities = weekly_meta_probabilities(
        candidates,
        ("feature",),
        min_train=2,
        model_factory=lambda: RecordingModel(fits),
    )

    assert probabilities.loc[index[0], "meta_p_tp"] == 1 / 3
    assert probabilities.loc[index[2], "meta_p_tp"] == 0.6
    assert fits[0] == [index[0], index[1]]
    assert fits[1] == [index[0], index[1], index[2]]


def test_meta_threshold_requires_fifty_evaluation_events():
    def evaluate(theta: float) -> dict[str, float]:
        return {
            "sortino": theta,
            "trade_count": 49 if theta == 0.8 else 50,
        }

    best = calibrate_meta_threshold((0.7, 0.8), evaluate, floor=50)

    assert best == (0.7, 0.7)


def test_meta_filter_flattens_low_or_missing_tp_probabilities():
    index = pd.date_range("2025-07-01", periods=4, freq="15min", tz="UTC")
    signals = pd.Series([2, 0, 1, 2], index=index)
    p_tp = pd.Series([0.8, 0.4], index=index[:2])

    filtered = apply_meta_filter(signals, p_tp, threshold=0.6)

    assert filtered.tolist() == [2, 1, 1, 1]


def test_experiment_masks_keep_q2_lockbox_sealed():
    index = pd.DatetimeIndex(
        ["2025-06-30 23:45", "2025-07-01", "2026-03-31 23:45", "2026-04-01"],
        tz="UTC",
    )

    calibration, evaluation = experiment_masks(index)

    assert calibration.tolist() == [True, False, False, False]
    assert evaluation.tolist() == [False, True, True, False]


def test_outcome_safe_signal_mask_embargoes_paths_crossing_lockbox():
    index = pd.DatetimeIndex(
        ["2026-03-31 23:30", "2026-03-31 23:45"], tz="UTC"
    )

    safe = outcome_safe_signal_mask(index, max_hold=1)

    assert safe.tolist() == [True, False]


def test_probability_thresholds_use_trained_calibration_scores_only():
    index = pd.DatetimeIndex(
        ["2025-06-01", "2025-06-02", "2025-06-03", "2025-07-02"], tz="UTC"
    )
    probabilities = pd.Series([0.1, 0.2, 0.3, 0.99], index=index)
    trained = pd.Series([False, True, True, True], index=index)

    thresholds = calibration_probability_thresholds(
        probabilities, trained, quantiles=(0.0, 0.5, 1.0)
    )

    assert thresholds == (0.0, 0.2, 0.25, 0.3)
    assert 0.99 not in thresholds
