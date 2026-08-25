"""Full-path ordinal magnitude labels for Notebook O.

The frozen Notebook N decision calendar and causal N3 feature matrix are reused.
Only the future-dependent target is rebuilt, over the complete half-open path
``[decision_time, decision_time + 120 minutes)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from experiments.event_window_large_move_dataset import (
    AdaptiveMoveConfig,
    LargeMoveDecisionDataset,
    VOLATILITY_FEATURE_COLUMNS,
    _continuous,
    _history_features,
    _utc_minute,
)


MAGNITUDE_THRESHOLDS = (0.75, 1.0, 1.5, 2.0)
MAGNITUDE_BIN_LABELS = (
    "<0.75B",
    "0.75-1.0B",
    "1.0-1.5B",
    "1.5-2.0B",
    ">=2.0B",
)
TTH_HORIZONS_MINUTES = (5, 15, 30, 60, 120)
TTH_COLUMNS = {
    0.75: "tth_075_min",
    1.0: "tth_100_min",
    1.5: "tth_150_min",
    2.0: "tth_200_min",
}

_IDENTITY_COLUMNS = (
    "window_id",
    "channel_episode_id",
    "side",
    "step",
    "decision_time",
)
_TARGET_DENY_TOKENS = (
    "future_",
    "magnitude_",
    "tth_",
    "label_start",
    "label_end",
    "target_valid",
    "path_complete",
    "reference_price",
    "hit_by_",
)


@dataclass(frozen=True)
class MagnitudeTargetConfig:
    adaptive: AdaptiveMoveConfig = field(default_factory=AdaptiveMoveConfig)
    thresholds: tuple[float, ...] = MAGNITUDE_THRESHOLDS
    time_to_hit_horizons: tuple[int, ...] = TTH_HORIZONS_MINUTES

    def __post_init__(self) -> None:
        if self.adaptive.horizon_minutes != 120:
            raise ValueError("Notebook O requires one frozen 120-minute label interval")
        values = np.asarray(self.thresholds, dtype=float)
        if (
            len(values) != 4
            or not np.isfinite(values).all()
            or not np.array_equal(values, np.sort(values))
            or not np.array_equal(values, np.asarray(MAGNITUDE_THRESHOLDS))
        ):
            raise ValueError("Notebook O magnitude thresholds are frozen")
        horizons = tuple(int(value) for value in self.time_to_hit_horizons)
        if horizons != TTH_HORIZONS_MINUTES:
            raise ValueError("Notebook O time-to-hit horizons are frozen")


def magnitude_class(ratio: float) -> int:
    """Return the registered ordinal class; exact boundaries enter the upper bin."""
    if not np.isfinite(ratio) or ratio < 0.0:
        return -1
    return int(np.searchsorted(MAGNITUDE_THRESHOLDS, ratio, side="right"))


def _aligned_causal_features(
    decisions: pd.DataFrame,
    causal_features: pd.DataFrame | None,
) -> pd.DataFrame | None:
    if causal_features is None:
        return None
    keys = ["window_id", "step"]
    required = {*keys, "decision_time", *VOLATILITY_FEATURE_COLUMNS}
    missing = sorted(required.difference(causal_features.columns))
    if missing:
        raise ValueError(f"causal feature handoff missing columns: {missing}")
    if causal_features.duplicated(keys).any():
        raise ValueError("causal feature handoff has duplicate decision keys")
    expected = decisions[keys].reset_index(drop=True)
    indexed = causal_features.set_index(keys)
    wanted = pd.MultiIndex.from_frame(expected)
    absent = wanted.difference(indexed.index)
    if len(absent):
        raise ValueError("causal feature handoff does not cover every decision")
    aligned = indexed.reindex(wanted).reset_index()
    if not expected.equals(aligned[keys]):
        raise AssertionError("causal feature handoff changed decision order")
    left = pd.to_datetime(decisions["decision_time"], utc=True, errors="raise")
    right = pd.to_datetime(aligned["decision_time"], utc=True, errors="raise")
    if not left.equals(right):
        raise ValueError("causal feature handoff changed decision timestamps")
    return aligned


def _causal_features_from_history(
    one: pd.DataFrame,
    decision_time: pd.Timestamp,
    position: int,
    config: AdaptiveMoveConfig,
) -> dict[str, float] | None:
    history_bars = int(config.feature_history_minutes) + 1
    start = position - history_bars
    history = one.iloc[max(0, start):position]
    expected_start = decision_time - pd.Timedelta(minutes=history_bars)
    valid = (
        start >= 0
        and len(history) == history_bars
        and _continuous(history.index, expected_start)
        and np.isfinite(history[["open", "high", "low", "close"]].to_numpy(float)).all()
    )
    if not valid:
        return None
    features = _history_features(
        history,
        volatility_lookback_minutes=config.volatility_lookback_minutes,
    )
    features["adaptive_barrier_bps"] = float(
        np.clip(
            max(
                config.minimum_barrier_bps,
                config.minimum_barrier_bps
                + config.volatility_addon_weight
                * features["past_sigma_5m_bps"]
                * np.sqrt(config.horizon_minutes / 5.0),
            ),
            config.minimum_barrier_bps,
            config.maximum_barrier_bps,
        )
    )
    return features


def label_full_path_magnitude(
    base_decisions: pd.DataFrame,
    minute: pd.DataFrame,
    config: MagnitudeTargetConfig = MagnitudeTargetConfig(),
    *,
    causal_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build strict full-path magnitude and time-to-hit labels.

    A decision at ``t`` uses the 1-minute Open at ``t`` as its reference and
    observes bars ``t`` through ``t+119``. The first observed bar is therefore
    time-to-hit minute 1. A missing/non-finite bar censors the row; it is never
    converted into a negative target. ``label_end`` remains ``t+120`` for every
    row so purging cannot shrink after an early touch.
    """
    missing = sorted(set(_IDENTITY_COLUMNS).difference(base_decisions.columns))
    if missing:
        raise ValueError(f"base decisions missing columns: {missing}")
    one = _utc_minute(minute)
    decisions = base_decisions.reset_index(drop=True).copy()
    decisions["decision_time"] = pd.to_datetime(
        decisions["decision_time"], utc=True, errors="raise"
    )
    aligned = _aligned_causal_features(decisions, causal_features)
    horizon = int(config.adaptive.horizon_minutes)
    minute_ns = int(pd.Timedelta(minutes=1).value)
    offsets_ns = np.arange(horizon, dtype=np.int64) * minute_ns
    # pandas 3 can preserve a microsecond DatetimeIndex resolution; normalise
    # explicitly before comparing it with nanosecond Timestamp values.
    index_ns = one.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    open_values = pd.to_numeric(one["open"], errors="coerce").to_numpy(float)
    high_values = pd.to_numeric(one["high"], errors="coerce").to_numpy(float)
    low_values = pd.to_numeric(one["low"], errors="coerce").to_numpy(float)
    close_values = pd.to_numeric(one["close"], errors="coerce").to_numpy(float)

    rows: list[dict[str, object]] = []
    for row_number, row in enumerate(decisions.itertuples(index=False)):
        decision_time = pd.Timestamp(row.decision_time)
        label_end = decision_time + pd.Timedelta(minutes=horizon)
        position = int(np.searchsorted(index_ns, decision_time.value))
        causal = None
        if aligned is not None:
            causal = {
                name: float(pd.to_numeric(aligned.iloc[row_number][name], errors="coerce"))
                for name in VOLATILITY_FEATURE_COLUMNS
            }
        elif position < len(one) and index_ns[position] == decision_time.value:
            causal = _causal_features_from_history(
                one, decision_time, position, config.adaptive
            )

        reference = np.nan
        up_excursion = np.nan
        down_excursion = np.nan
        ratio = np.nan
        target_class = -1
        path_complete = False
        censor_reason = "missing causal history"
        tth = {column: np.nan for column in TTH_COLUMNS.values()}
        if causal is not None and np.isfinite(causal["adaptive_barrier_bps"]):
            end = position + horizon
            path_complete = (
                position < len(one)
                and end <= len(one)
                and index_ns[position] == decision_time.value
                and np.array_equal(index_ns[position:end], decision_time.value + offsets_ns)
                and np.isfinite(
                    np.column_stack(
                        (
                            open_values[position:end],
                            high_values[position:end],
                            low_values[position:end],
                            close_values[position:end],
                        )
                    )
                ).all()
            )
            censor_reason = "incomplete or non-finite 120-minute path"
            if path_complete:
                reference = float(open_values[position])
                highs = high_values[position:end]
                lows = low_values[position:end]
                barrier = float(causal["adaptive_barrier_bps"])
                up_path = np.log(highs / reference) * 1e4
                down_path = np.log(reference / lows) * 1e4
                up_excursion = float(np.max(up_path))
                down_excursion = float(np.max(down_path))
                magnitude_bps = max(0.0, up_excursion, down_excursion)
                ratio = float(magnitude_bps / barrier)
                for boundary in MAGNITUDE_THRESHOLDS:
                    if np.isclose(ratio, boundary, rtol=0.0, atol=1e-12):
                        ratio = float(boundary)
                target_class = magnitude_class(ratio)
                for threshold, column in TTH_COLUMNS.items():
                    upper = reference * np.exp(threshold * barrier / 1e4)
                    lower = reference * np.exp(-threshold * barrier / 1e4)
                    touched = np.flatnonzero(
                        (highs >= upper)
                        | (lows <= lower)
                    )
                    if touched.size:
                        tth[column] = float(touched[0] + 1)
                censor_reason = ""
        else:
            causal = {name: np.nan for name in VOLATILITY_FEATURE_COLUMNS}

        output = {
            "window_id": row.window_id,
            "channel_episode_id": row.channel_episode_id,
            "side": str(row.side),
            "step": int(row.step),
            "decision_time": decision_time,
            "reference_price": reference,
            "label_start": decision_time,
            "label_end": label_end,
            "path_complete_120m": bool(path_complete),
            "magnitude_target_valid": bool(path_complete),
            "model_target_valid": bool(path_complete),
            "censor_reason": censor_reason,
            "future_up_excursion_bps": up_excursion,
            "future_down_excursion_bps": down_excursion,
            "magnitude_ratio": ratio,
            "magnitude_class": int(target_class),
            "magnitude_bin": (
                MAGNITUDE_BIN_LABELS[target_class] if target_class >= 0 else "censored"
            ),
            **tth,
            **causal,
        }
        rows.append(output)
    result = pd.DataFrame(rows)
    for horizon_minutes in TTH_HORIZONS_MINUTES:
        result[f"hit_by_{horizon_minutes:03d}m"] = (
            result["tth_100_min"].notna()
            & result["tth_100_min"].le(horizon_minutes)
            & result["magnitude_target_valid"]
        )
    return result


def assert_magnitude_feature_isolation(feature_names: tuple[str, ...]) -> None:
    lowered = tuple(name.lower() for name in feature_names)
    offending = sorted(
        name
        for name, low in zip(feature_names, lowered, strict=True)
        if any(token in low for token in _TARGET_DENY_TOKENS)
    )
    if offending:
        raise AssertionError(f"future/target fields entered model features: {offending}")


def align_magnitude_dataset(
    frozen_n3: LargeMoveDecisionDataset,
    labels: pd.DataFrame,
) -> LargeMoveDecisionDataset:
    """Attach new labels while preserving the exact frozen N3 feature matrix."""
    keys = ["window_id", "step"]
    expected = frozen_n3.decisions[keys].reset_index(drop=True)
    work = labels.reset_index(drop=True).copy()
    if work.duplicated(keys).any() or not expected.equals(work[keys]):
        raise ValueError("magnitude labels must align row-for-row with frozen N3")
    old_time = pd.to_datetime(
        frozen_n3.decisions["decision_time"], utc=True, errors="raise"
    )
    new_time = pd.to_datetime(work["decision_time"], utc=True, errors="raise")
    if not old_time.equals(new_time):
        raise ValueError("magnitude labels changed the N3 decision calendar")
    old_barrier = pd.to_numeric(
        frozen_n3.decisions["adaptive_barrier_bps"], errors="coerce"
    ).to_numpy(float)
    new_barrier = pd.to_numeric(work["adaptive_barrier_bps"], errors="coerce").to_numpy(float)
    if not np.allclose(old_barrier, new_barrier, equal_nan=True, rtol=0.0, atol=1e-6):
        raise ValueError("magnitude target changed the frozen causal barrier")
    assert_magnitude_feature_isolation(frozen_n3.tabular_features)
    return LargeMoveDecisionDataset(
        decisions=work,
        tabular=np.asarray(frozen_n3.tabular, dtype=np.float32),
        tabular_features=frozen_n3.tabular_features,
        dropped_features=frozen_n3.dropped_features,
        feature_set="frozen_N3_side_neutral_volatility",
    )


def time_to_hit_bucket(values: pd.Series) -> pd.Categorical:
    """Registered 1x barrier timing buckets, including an explicit no-hit cell."""
    numeric = pd.to_numeric(values, errors="coerce")
    labels = ("<=5", "6-15", "16-30", "31-60", "61-120", "no hit")
    output = np.select(
        [
            numeric.le(5),
            numeric.between(6, 15),
            numeric.between(16, 30),
            numeric.between(31, 60),
            numeric.between(61, 120),
        ],
        labels[:-1],
        default=labels[-1],
    )
    return pd.Categorical(output, categories=labels, ordered=True)


__all__ = [
    "MAGNITUDE_BIN_LABELS",
    "MAGNITUDE_THRESHOLDS",
    "MagnitudeTargetConfig",
    "TTH_COLUMNS",
    "TTH_HORIZONS_MINUTES",
    "align_magnitude_dataset",
    "assert_magnitude_feature_isolation",
    "label_full_path_magnitude",
    "magnitude_class",
    "time_to_hit_bucket",
]
