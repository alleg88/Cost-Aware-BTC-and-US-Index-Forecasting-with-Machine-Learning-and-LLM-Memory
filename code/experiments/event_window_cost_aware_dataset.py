"""Maker-entry labels and causal cost-aware features for Notebook L.

The module deliberately rebuilds execution labels from the native one-minute
path.  It does not reprice or reuse Notebook K's selected OOF trades.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.channel_backtest import _resolve_1m
from experiments.event_window_tail_dataset import TailDecisionDataset
from features.event_window_inputs import known_structural_stop


OUTCOME_CODES = {"unfilled": 0, "sl": 1, "tp": 2, "timeout": 3}
_DROP_PREFIXES = (
    "pre_window_mask",
    "active_window_mask",
    "bar_complete",
    "cadence_gap",
)
_DROP_EXACT = frozenset({"window_reason_code", "time_remaining_fraction"})


@dataclass(frozen=True)
class MakerExecutionConfig:
    rr_multiple: float = 2.0
    stop_buffer_bps: float = 5.0
    trailing_stop_bars: int = 12
    limit_offset_bps: float = 5.0
    fill_confirmation_bps: float = 1.0
    fill_window_minutes: int = 20
    max_hold_minutes: int = 120
    maker_entry_bps: float = 2.0
    maker_tp_exit_bps: float = 2.0
    taker_sl_exit_bps: float = 5.0
    timeout_exit_bps: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "rr_multiple",
            "trailing_stop_bars",
            "fill_window_minutes",
            "max_hold_minutes",
        )
        non_negative = (
            "stop_buffer_bps",
            "limit_offset_bps",
            "fill_confirmation_bps",
            "maker_entry_bps",
            "maker_tp_exit_bps",
            "taker_sl_exit_bps",
            "timeout_exit_bps",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in non_negative:
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")

    def round_trip_bps(self, outcome: str) -> float:
        exits = {
            "tp": self.maker_tp_exit_bps,
            "sl": self.taker_sl_exit_bps,
            "timeout": self.timeout_exit_bps,
        }
        if outcome == "unfilled":
            return 0.0
        if outcome not in exits:
            raise ValueError(f"unsupported outcome: {outcome!r}")
        return float(self.maker_entry_bps + exits[outcome])


@dataclass(frozen=True)
class CostAwareDecisionDataset:
    decisions: pd.DataFrame
    tabular: np.ndarray
    tabular_features: tuple[str, ...]
    dropped_features: tuple[str, ...]


def _utc_frame(frame: pd.DataFrame, required: set[str], name: str) -> pd.DataFrame:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name} must use a DatetimeIndex")
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="raise")
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError(f"{name} timestamps must be unique and increasing")
    return out


def _fill_time(
    minute: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    side: str,
    limit_price: float,
    confirmation_bps: float,
) -> tuple[pd.Timestamp | pd.NaT, str, pd.Timestamp]:
    position = int(minute.index.searchsorted(start))
    final_observed = start
    for timestamp in pd.date_range(start, end, freq="1min", inclusive="left"):
        final_observed = timestamp + pd.Timedelta(minutes=1)
        if position >= len(minute.index) or minute.index[position] != timestamp:
            return pd.NaT, "censored", timestamp
        high = float(minute.iloc[position]["high"])
        low = float(minute.iloc[position]["low"])
        if not np.isfinite(high) or not np.isfinite(low):
            return pd.NaT, "censored", timestamp
        confirmation = confirmation_bps / 1e4
        touched = (
            low <= limit_price * (1.0 - confirmation)
            if side == "long"
            else high >= limit_price * (1.0 + confirmation)
        )
        if touched:
            return timestamp, "filled", final_observed
        position += 1
    return pd.NaT, "unfilled", final_observed


def label_maker_window_steps(
    sequences,
    five_minute: pd.DataFrame,
    minute: pd.DataFrame,
    config: MakerExecutionConfig = MakerExecutionConfig(),
) -> pd.DataFrame:
    """Label every causal decision as unfilled, SL, TP, timeout, or censored."""
    five = _utc_frame(five_minute, {"open", "high", "low", "close"}, "five_minute")
    one = _utc_frame(minute, {"open", "high", "low", "close"}, "minute")
    metadata = sequences.metadata.reset_index(drop=True)
    rows: list[dict[str, object]] = []
    for sample_no, meta in metadata.iterrows():
        side = str(meta["side"]).lower()
        if side not in {"long", "short"}:
            raise ValueError(f"invalid side: {side!r}")
        window_end = pd.Timestamp(meta["window_end"])
        window_end = (
            window_end.tz_localize("UTC")
            if window_end.tzinfo is None
            else window_end.tz_convert("UTC")
        )
        for step in np.flatnonzero(sequences.decision_valid[sample_no]):
            source_time = pd.Timestamp(sequences.source_bar_times[sample_no, step], tz="UTC")
            decision_time = pd.Timestamp(sequences.decision_times[sample_no, step], tz="UTC")
            order_reference = float(five.at[source_time, "close"]) if source_time in five.index else np.nan
            direction = -1.0 if side == "long" else 1.0
            entry = order_reference * (1.0 + direction * config.limit_offset_bps / 1e4)
            stop_history = five.loc[:source_time].tail(config.trailing_stop_bars)
            stop = known_structural_stop(
                stop_history,
                side=side,
                source_bar_time=source_time,
                lookback=config.trailing_stop_bars,
                buffer_bps=config.stop_buffer_bps,
            )
            risk = entry - stop if side == "long" else stop - entry
            risk_bps = (
                float(risk / entry * 1e4)
                if np.isfinite(entry) and entry > 0 and np.isfinite(risk) and risk > 0
                else np.nan
            )
            target = entry + config.rr_multiple * risk if side == "long" else entry - config.rr_multiple * risk
            geometry_valid = bool(
                np.isfinite(entry)
                and entry > 0
                and np.isfinite(stop)
                and stop > 0
                and np.isfinite(target)
                and target > 0
                and np.isfinite(risk_bps)
                and risk_bps > 0
            )
            outcome = "invalid_geometry"
            entry_time = pd.NaT
            exit_time = pd.NaT
            label_end = decision_time
            filled = False
            path_observed = False
            r_gross = np.nan
            r_net = np.nan
            if geometry_valid:
                fill_end = min(
                    decision_time + pd.Timedelta(minutes=config.fill_window_minutes),
                    window_end,
                )
                if fill_end <= decision_time:
                    outcome = "unfilled"
                    label_end = decision_time
                    path_observed = True
                    r_gross = r_net = 0.0
                else:
                    entry_time, fill_status, label_end = _fill_time(
                        one,
                        start=decision_time,
                        end=fill_end,
                        side=side,
                        limit_price=float(entry),
                        confirmation_bps=config.fill_confirmation_bps,
                    )
                    if fill_status == "censored":
                        outcome = "censored"
                    elif fill_status == "unfilled":
                        outcome = "unfilled"
                        path_observed = True
                        r_gross = r_net = 0.0
                    else:
                        filled = True
                        exit_time, exit_price, outcome = _resolve_1m(
                            side,
                            float(entry),
                            float(stop),
                            float(target),
                            one,
                            entry_time,
                            config.max_hold_minutes,
                            None,
                            None,
                            ignore_target_at_start=True,
                        )
                        exit_time = pd.Timestamp(exit_time)
                        path_observed = outcome in {"sl", "tp", "timeout"}
                        label_end = exit_time + pd.Timedelta(minutes=1)
                        if path_observed:
                            move = exit_price - entry if side == "long" else entry - exit_price
                            r_gross = float(move / risk)
                            r_net = float(
                                r_gross - config.round_trip_bps(outcome) / risk_bps
                            )
            model_target_valid = bool(geometry_valid and path_observed)
            rows.append(
                {
                    "window_id": meta["window_id"],
                    "channel_episode_id": meta["channel_episode_id"],
                    "side": side,
                    "step": int(step),
                    "source_bar_time": source_time,
                    "decision_time": decision_time,
                    "order_time": decision_time,
                    "entry_time": entry_time,
                    "exit_time": exit_time,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "risk_bps": risk_bps,
                    "filled": filled,
                    "outcome": outcome,
                    "r_gross": r_gross,
                    "r_net": r_net,
                    "label_start": decision_time,
                    "label_end": label_end,
                    "geometry_valid": geometry_valid,
                    "path_observed": path_observed,
                    "model_target_valid": model_target_valid,
                }
            )
    return pd.DataFrame(rows)


def _drop_feature(name: str) -> bool:
    return name in _DROP_EXACT or any(
        name == prefix or name.startswith(prefix + "_") for prefix in _DROP_PREFIXES
    )


def build_cost_aware_dataset(
    base: TailDecisionDataset,
    labels: pd.DataFrame,
    config: MakerExecutionConfig = MakerExecutionConfig(),
) -> CostAwareDecisionDataset:
    """Align new maker labels, prune deterministic duplicates, and add fee burden."""
    keys = ["window_id", "step"]
    expected = base.decisions[keys].copy().reset_index(drop=True)
    work = labels.copy().reset_index(drop=True)
    if work.duplicated(keys).any():
        raise ValueError("maker labels contain duplicate keys")
    actual = work[keys]
    if not expected.equals(actual):
        raise ValueError("maker labels must align row-for-row with base decisions")
    decisions = work.copy()
    decisions["outcome_code"] = decisions["outcome"].map(OUTCOME_CODES).fillna(-1).astype("int8")
    valid_risk = decisions["risk_bps"].gt(0) & np.isfinite(decisions["risk_bps"])
    decisions["tp_net_r"] = np.where(
        valid_risk,
        config.rr_multiple - config.round_trip_bps("tp") / decisions["risk_bps"],
        np.nan,
    )
    decisions["sl_net_r"] = np.where(
        valid_risk,
        -1.0 - config.round_trip_bps("sl") / decisions["risk_bps"],
        np.nan,
    )
    decisions["timeout_gross_r"] = np.where(
        decisions["outcome"].eq("timeout"), decisions["r_gross"], np.nan
    )

    decisions = decisions.sort_values(keys, kind="stable").reset_index(drop=True)
    decisions["enter_advantage_target"] = np.nan
    decisions["advantage_valid"] = False
    decisions["label_end"] = pd.to_datetime(decisions["label_end"], utc=True, errors="coerce")
    for _, positions in decisions.groupby("window_id", sort=False).groups.items():
        index = np.asarray(list(positions), dtype=int)
        valid = decisions.loc[index, "model_target_valid"].to_numpy(bool)
        net = decisions.loc[index, "r_net"].to_numpy(float)
        ends = decisions.loc[index, "label_end"].to_numpy(dtype="datetime64[ns]")
        suffix_valid = np.logical_and.accumulate(valid[::-1])[::-1]
        suffix_end = np.maximum.accumulate(ends[::-1])[::-1]
        future_best = np.zeros(len(index), dtype=float)
        running_best = 0.0
        for local in range(len(index) - 1, -1, -1):
            future_best[local] = running_best
            if valid[local] and np.isfinite(net[local]):
                running_best = max(running_best, float(net[local]), 0.0)
        advantage_valid = valid & suffix_valid
        decisions.loc[index, "advantage_valid"] = advantage_valid
        decisions.loc[index[advantage_valid], "enter_advantage_target"] = (
            net[advantage_valid] - future_best[advantage_valid]
        )
        decisions.loc[index[advantage_valid], "label_end"] = pd.to_datetime(
            suffix_end[advantage_valid], utc=True
        )

    keep = [not _drop_feature(name) for name in base.tabular_features]
    kept_names = [name for name, retained in zip(base.tabular_features, keep, strict=True) if retained]
    dropped = tuple(
        name for name, retained in zip(base.tabular_features, keep, strict=True) if not retained
    )
    tabular = np.asarray(base.tabular[:, keep], dtype=np.float32)
    costs = np.column_stack(
        [
            decisions["risk_bps"].to_numpy(dtype=float),
            config.round_trip_bps("tp") / decisions["risk_bps"].to_numpy(dtype=float),
            config.round_trip_bps("sl") / decisions["risk_bps"].to_numpy(dtype=float),
            config.round_trip_bps("timeout") / decisions["risk_bps"].to_numpy(dtype=float),
        ]
    ).astype(np.float32)
    tabular = np.column_stack([tabular, costs]).astype(np.float32, copy=False)
    features = tuple(
        [*kept_names, "maker_risk_bps", "tp_cost_r", "sl_cost_r", "timeout_cost_r"]
    )
    return CostAwareDecisionDataset(decisions, tabular, features, dropped)


__all__ = [
    "CostAwareDecisionDataset",
    "MakerExecutionConfig",
    "OUTCOME_CODES",
    "build_cost_aware_dataset",
    "label_maker_window_steps",
]
