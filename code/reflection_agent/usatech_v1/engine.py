"""Exact soft-vote opportunity reconstruction for the USATECH agents."""
from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.index_all_model_ensemble import (
    AlignedPanel,
    _prediction_frame,
    combine_probabilities,
)
from experiments.index_replication import simulate_one_bar
from experiments.index_replication_protocol import CUTOFF, FORWARD_START
from reflection_agent.index_v1.config import MODEL_NAMES


PROBABILITY_LABELS = ("short", "flat", "long")


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _assert_before_q2(*values: pd.Timestamp) -> None:
    if any(_utc(value) >= CUTOFF for value in values):
        raise PermissionError("Q2 timestamps are sealed for the USATECH agent")


def build_soft_vote_opportunities(
    panel: AlignedPanel,
    bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tau: float,
    cost_bps: float,
    state_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise the frozen equal-probability soft-vote parent and evidence."""
    start_utc, end_utc = _utc(start), _utc(end)
    if end_utc <= start_utc:
        raise ValueError("opportunity interval must be positive")
    _assert_before_q2(start_utc, end_utc - pd.Timedelta(nanoseconds=1))
    if tuple(panel.probabilities) != MODEL_NAMES:
        raise ValueError("opportunity panel changed the frozen model order")
    if len(panel.timestamp):
        _assert_before_q2(pd.Timestamp(panel.timestamp.max()))

    prediction = _prediction_frame(
        panel,
        combine_probabilities("soft_vote", panel),
        "index-agent:usatech:frozen-soft-vote",
    )
    ledger, _ = simulate_one_bar(
        bars,
        prediction,
        start=start_utc,
        end=end_utc,
        tau=float(tau),
        cost_bps=float(cost_bps),
    )
    if ledger.empty:
        raise ValueError("frozen ensemble produced no registered opportunities")
    ledger = ledger.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    signal_times = pd.DatetimeIndex(pd.to_datetime(ledger["signal_bar_open"], utc=True))
    positions = panel.timestamp.get_indexer(signal_times)
    if (positions < 0).any():
        raise ValueError("registered signal is absent from the probability panel")

    prefix = "FWD" if start_utc >= FORWARD_START else "H1"
    opportunities = ledger.copy()
    opportunities.insert(
        0, "opportunity_id", [f"{prefix}-{index:06d}" for index in range(len(ledger))]
    )
    opportunities["original_side"] = opportunities["side"].astype(int)
    opportunities["outcome_available_at"] = pd.to_datetime(
        opportunities["exit_time"], utc=True
    )
    signal_series = pd.Series(signal_times, index=opportunities.index)
    opportunities["week_start"] = signal_series.dt.normalize() - pd.to_timedelta(
        signal_series.dt.weekday, unit="D"
    )
    opportunities["y_true"] = np.asarray(panel.y_true, dtype=int)[positions]
    for model_index, model in enumerate(MODEL_NAMES):
        values = np.asarray(panel.probabilities[model], dtype=float)[positions]
        if values.shape != (len(opportunities), 3):
            raise ValueError("model probability shape changed at opportunities")
        for class_index, label in enumerate(PROBABILITY_LABELS):
            opportunities[f"m{model_index:02d}_p_{label}"] = values[:, class_index]

    if state_frame is not None:
        required_state = {
            "available_at",
            "state_vix_regime",
            "state_trailing_vol",
            "state_trailing_trend",
        }
        missing = required_state.difference(state_frame.columns)
        if missing:
            raise ValueError(f"causal state misses columns: {sorted(missing)}")
        state = state_frame.copy()
        state.index = pd.to_datetime(state.index, utc=True)
        state["available_at"] = pd.to_datetime(state["available_at"], utc=True)
        if state.index.duplicated().any() or not state.index.is_monotonic_increasing:
            raise ValueError("causal state index must be unique and sorted")
        selected = state.reindex(signal_times)
        if selected[list(required_state)].isna().any().any():
            raise ValueError("causal state is incomplete at a registered opportunity")
        decision = pd.to_datetime(opportunities["decision_time"], utc=True).to_numpy()
        available = pd.to_datetime(selected["available_at"], utc=True).to_numpy()
        if (available > decision).any():
            raise ValueError("causal state was not available by the opportunity decision")
        for column in (
            "state_vix_regime",
            "state_trailing_vol",
            "state_trailing_trend",
        ):
            values = selected[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("causal state values must be finite")
            opportunities[column] = values

    if opportunities["opportunity_id"].duplicated().any():
        raise AssertionError("registered opportunity identifiers are not unique")
    ordered = opportunities.sort_values("entry_time", kind="mergesort")
    previous_exit = pd.to_datetime(ordered["exit_time"], utc=True).shift(1)
    current_entry = pd.to_datetime(ordered["entry_time"], utc=True)
    if (current_entry.iloc[1:] < previous_exit.iloc[1:]).any():
        raise AssertionError("registered opportunity schedule overlaps")
    probability_columns = opportunities.filter(
        regex=r"^m\d{2}_p_(short|flat|long)$"
    )
    if not np.isfinite(probability_columns.to_numpy(float)).all():
        raise ValueError("registered probabilities must be finite")

    ledger.insert(0, "opportunity_id", opportunities["opportunity_id"])
    return opportunities, ledger


__all__ = ["build_soft_vote_opportunities"]
