"""Aligned causal features and economic labels for Notebook V direction models."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.event_window_large_move_dataset import LargeMoveDecisionDataset


DIRECTION_FEATURES = (
    "raw_log_return",
    "raw_log_return_mean_3",
    "raw_log_return_mean_12",
    "raw_log_return_mean_24",
    "raw_body_bps",
    "lower_wick_fraction",
    "upper_wick_fraction",
    "raw_taker_imbalance",
    "raw_taker_imbalance_mean_3",
    "raw_taker_imbalance_mean_12",
    "raw_taker_imbalance_delta",
    "raw_channel_slope",
    "channel_r2",
    "channel_width_pct",
    "raw_channel_position",
    "raw_distance_mid_bps",
    "raw_rsi_channel",
    "raw_oi_chg_15m",
    "raw_oi_chg_1h",
    "raw_oi_chg_4h",
    "raw_oi_accel_1h",
    "raw_price_oi_interaction",
    "raw_funding_z",
    "positioning_stale",
    "positioning_age_log",
    "adaptive_barrier_bps",
    "window_age_fraction",
    "activation_margin",
)

_CAUSAL_FEATURES = DIRECTION_FEATURES[:-1]
_JOIN_KEYS = ("window_id", "channel_episode_id", "step", "decision_time")
_LEDGER_REQUIRED = (
    "activation_key",
    *_JOIN_KEYS,
    "threshold",
    "activation_score",
)
_PAIRED_REQUIRED = (
    "activation_key",
    "net_r_long",
    "net_r_short",
    "delta_r",
    "best_side",
    "economic_value",
)
_V_PATH_CONTRACT = {
    "target_multiple_b": 2.0,
    "hold_minutes": 120.0,
    "cost_bps": 10.0,
}


@dataclass(frozen=True)
class DirectionDataset:
    """One causal feature row and one paired economic label per activation."""

    decisions: pd.DataFrame
    tabular: np.ndarray
    tabular_features: tuple[str, ...]


def pair_direction_paths(
    paths: pd.DataFrame, *, tie_tolerance: float = 1e-12
) -> pd.DataFrame:
    """Pair exactly one LONG and SHORT economic path for each activation."""
    required = {"activation_key", "direction", "net_r"}
    missing = sorted(required.difference(paths.columns))
    if missing:
        raise ValueError(f"direction paths missing columns: {missing}")
    if not np.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise ValueError("tie_tolerance must be finite and non-negative")
    missing_contract = sorted(set(_V_PATH_CONTRACT).difference(paths.columns))
    if missing_contract:
        raise ValueError(
            f"direction paths missing V primary geometry columns: {missing_contract}"
        )
    for column, expected in _V_PATH_CONTRACT.items():
        values = pd.to_numeric(paths[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isclose(
            values, expected, rtol=0.0, atol=1e-12
        ).all():
            label = {
                "target_multiple_b": "target multiple",
                "hold_minutes": "hold",
                "cost_bps": "cost",
            }[column]
            raise ValueError(
                f"direction paths must use V primary {label} {expected:g}"
            )
    work = paths.loc[:, ["activation_key", "direction", "net_r"]].copy()
    if work["activation_key"].isna().any():
        raise ValueError("direction paths contain missing activation keys")
    if not work["direction"].isin(("long", "short")).all():
        raise ValueError("direction paths must contain only long and short rows")
    if work.duplicated(["activation_key", "direction"]).any():
        raise ValueError("direction paths contain duplicate activation-direction rows")

    wide = work.pivot(index="activation_key", columns="direction", values="net_r")
    missing_directions = sorted({"long", "short"}.difference(wide.columns))
    values = wide.reindex(columns=["long", "short"]).to_numpy(dtype=float)
    if missing_directions or not np.isfinite(values).all():
        raise ValueError("each activation requires finite long and short net_r paths")
    wide = wide[["long", "short"]]
    delta = wide["long"] - wide["short"]
    return pd.DataFrame(
        {
            "activation_key": wide.index,
            "net_r_long": wide["long"],
            "net_r_short": wide["short"],
            "delta_r": delta,
            "best_side": np.where(
                delta > tie_tolerance,
                "long",
                np.where(delta < -tie_tolerance, "short", "tie"),
            ),
            "economic_value": np.minimum(np.abs(delta), 3.0),
        }
    ).reset_index(drop=True)


def interval_uniqueness(
    decision_times: pd.Series | pd.DatetimeIndex | list[object] | np.ndarray,
    *,
    horizon_minutes: int = 120,
    normalize: bool = False,
) -> np.ndarray:
    """Return half-open payoff-interval uniqueness for the supplied rows only.

    The default preserves raw mean reciprocal concurrency, which makes overlap
    visible in audits.  A caller that has already excluded ties can request
    fold-local positive-weight normalisation with ``normalize=True``.
    """
    if isinstance(horizon_minutes, bool) or int(horizon_minutes) != horizon_minutes:
        raise ValueError("horizon_minutes must be a positive integer")
    horizon = int(horizon_minutes)
    if horizon <= 0:
        raise ValueError("horizon_minutes must be a positive integer")
    times = pd.DatetimeIndex(pd.to_datetime(decision_times, utc=True, errors="raise"))
    if times.hasnans:
        raise ValueError("decision times must be present")
    if not len(times):
        return np.empty(0, dtype=float)
    if not times.equals(times.floor("min")):
        raise ValueError("decision times must align to whole minutes")

    minute_ns = int(pd.Timedelta(minutes=1).value)
    starts = times.asi8 // minute_ns
    concurrency: dict[int, int] = {}
    for start in starts:
        for minute in range(int(start), int(start) + horizon):
            concurrency[minute] = concurrency.get(minute, 0) + 1
    weights = np.asarray(
        [
            np.mean(
                [1.0 / concurrency[minute] for minute in range(int(start), int(start) + horizon)]
            )
            for start in starts
        ],
        dtype=float,
    )
    if normalize:
        positive = np.isfinite(weights) & (weights > 0.0)
        if positive.any():
            weights[positive] /= float(weights[positive].mean())
    return weights


def _validate_directional_matrix(
    directional: LargeMoveDecisionDataset,
) -> tuple[pd.DataFrame, np.ndarray]:
    if directional.tabular.ndim != 2:
        raise ValueError("causal directional matrix must be two-dimensional")
    if directional.tabular.shape[0] != len(directional.decisions):
        raise ValueError("causal directional matrix rows do not match decisions")
    if directional.tabular.shape[1] != len(directional.tabular_features):
        raise ValueError("causal directional matrix columns do not match feature names")
    if len(directional.tabular_features) != len(set(directional.tabular_features)):
        raise ValueError("causal directional feature names must be unique")
    missing_features = sorted(set(_CAUSAL_FEATURES).difference(directional.tabular_features))
    if missing_features:
        raise ValueError(
            f"causal directional matrix missing direction features: {missing_features}"
        )
    decisions = directional.decisions.copy().reset_index(drop=True)
    missing_keys = sorted(set(_JOIN_KEYS).difference(decisions.columns))
    if missing_keys:
        raise ValueError(f"causal directional decisions missing join keys: {missing_keys}")
    decisions["decision_time"] = pd.to_datetime(
        decisions["decision_time"], utc=True, errors="raise"
    )
    if decisions[list(_JOIN_KEYS)].isna().any().any():
        raise ValueError("causal directional decisions contain missing join keys")
    if decisions.duplicated(list(_JOIN_KEYS)).any():
        raise ValueError("causal directional decisions contain duplicate join keys")
    lookup = {name: index for index, name in enumerate(directional.tabular_features)}
    matrix = np.asarray(
        directional.tabular[:, [lookup[name] for name in _CAUSAL_FEATURES]],
        dtype=np.float32,
    )
    return decisions, matrix


def build_direction_dataset(
    activation_ledger: pd.DataFrame,
    causal_directional: LargeMoveDecisionDataset,
    paired_paths: pd.DataFrame,
) -> DirectionDataset:
    """Key-join frozen activations, causal features and paired economic labels."""
    missing_ledger = sorted(set(_LEDGER_REQUIRED).difference(activation_ledger.columns))
    if missing_ledger:
        raise ValueError(f"activation ledger missing columns: {missing_ledger}")
    ledger = activation_ledger.copy().reset_index(drop=True)
    ledger["decision_time"] = pd.to_datetime(
        ledger["decision_time"], utc=True, errors="raise"
    )
    if ledger[list(_LEDGER_REQUIRED)].isna().any().any():
        raise ValueError("activation ledger contains missing keys or timing values")
    timing_values = ledger[["threshold", "activation_score"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=float)
    if not np.isfinite(timing_values).all():
        raise ValueError("activation ledger timing values must be finite")
    if ledger["activation_key"].duplicated().any():
        raise ValueError("activation ledger contains duplicate activation keys")
    if ledger.duplicated(list(_JOIN_KEYS)).any():
        raise ValueError("activation ledger contains duplicate directional join keys")

    causal_decisions, causal_matrix = _validate_directional_matrix(causal_directional)
    causal = causal_decisions.loc[:, list(_JOIN_KEYS)].copy()
    causal.loc[:, _CAUSAL_FEATURES] = causal_matrix
    joined = ledger.merge(causal, on=list(_JOIN_KEYS), how="left", sort=False, validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"].ne("both"), "activation_key"].tolist()
        raise ValueError(f"causal directional matrix does not cover activations: {missing}")
    joined = joined.drop(columns="_merge")

    missing_paired = sorted(set(_PAIRED_REQUIRED).difference(paired_paths.columns))
    if missing_paired:
        raise ValueError(f"paired paths missing columns: {missing_paired}")
    paired = paired_paths.loc[:, list(_PAIRED_REQUIRED)].copy()
    if paired["activation_key"].isna().any() or paired["activation_key"].duplicated().any():
        raise ValueError("paired paths must have one non-missing row per activation")
    joined = joined.merge(paired, on="activation_key", how="left", sort=False, validate="one_to_one", indicator=True)
    if not joined["_merge"].eq("both").all():
        missing = joined.loc[joined["_merge"].ne("both"), "activation_key"].tolist()
        raise ValueError(f"paired paths do not cover activations: {missing}")
    joined = joined.drop(columns="_merge")
    if len(joined) != len(ledger):
        raise AssertionError("direction dataset changed frozen activation count")

    joined["activation_margin"] = (
        pd.to_numeric(joined["activation_score"], errors="raise")
        - pd.to_numeric(joined["threshold"], errors="raise")
    )
    tabular = joined.loc[:, DIRECTION_FEATURES].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(dtype=np.float32)
    if tabular.shape != (len(joined), len(DIRECTION_FEATURES)):
        raise AssertionError("direction dataset feature contract changed")
    return DirectionDataset(
        decisions=joined.reset_index(drop=True),
        tabular=tabular,
        tabular_features=DIRECTION_FEATURES,
    )


__all__ = [
    "DIRECTION_FEATURES",
    "DirectionDataset",
    "build_direction_dataset",
    "interval_uniqueness",
    "pair_direction_paths",
]
