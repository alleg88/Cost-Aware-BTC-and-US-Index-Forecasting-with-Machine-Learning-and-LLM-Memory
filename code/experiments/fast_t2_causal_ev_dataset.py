"""Causal geometry state and outcome-EV labels for Fast-T2 decisions."""
from __future__ import annotations

import numpy as np
import pandas as pd


OUTCOME_TO_CLASS = {"sl": 0, "tp": 1, "timeout": 2}
PRIMARY_MIN_RISK_BPS = 25.0
SENSITIVITY_MIN_RISK_BPS = 40.0
ROUND_TRIP_COST_BPS = 10.0


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def _as_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True, errors="raise")


def annotate_causal_ev_rows(
    decisions: pd.DataFrame,
    minute_bars: pd.DataFrame,
) -> pd.DataFrame:
    """Add past-only stop/target state and fixed economic labels.

    A bar stamped at the decision boundary is not completed yet.  Therefore a
    decision at ``t`` sees minute bars strictly earlier than ``t``.  Once a
    completed bar touches the frozen stop, all later decisions in that window
    remain invalid.
    """
    required = {
        "window_id",
        "decision_id",
        "side",
        "t2_time",
        "decision_time",
        "entry_price",
        "stop_price",
        "target_price",
        "distance_to_stop_bps",
        "distance_to_target_bps",
        "rr_proxy",
        "outcome",
        "r_net",
        "filled",
    }
    _require_columns(decisions, required, "decisions")
    _require_columns(minute_bars, {"high", "low"}, "minute bars")
    if not isinstance(minute_bars.index, pd.DatetimeIndex):
        raise ValueError("minute bars need a DatetimeIndex")
    if minute_bars.index.tz is None:
        raise ValueError("minute bars need a timezone-aware DatetimeIndex")

    minute = minute_bars.sort_index(kind="stable")
    if minute.index.has_duplicates:
        raise ValueError("minute bars contain duplicate timestamps")
    minute_index = minute.index.tz_convert("UTC")
    high = pd.to_numeric(minute["high"], errors="raise").to_numpy(float)
    low = pd.to_numeric(minute["low"], errors="raise").to_numpy(float)

    work = decisions.copy()
    work["t2_time"] = _as_utc(work["t2_time"])
    work["decision_time"] = _as_utc(work["decision_time"])
    work["_original_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(
        ["window_id", "decision_time", "decision_id"], kind="stable"
    )

    stop_flags = pd.Series(False, index=work.index, dtype=bool)
    target_flags = pd.Series(False, index=work.index, dtype=bool)
    for _, group in work.groupby("window_id", sort=False):
        first = group.iloc[0]
        side = str(first["side"])
        if side not in {"long", "short"}:
            raise ValueError(f"unsupported side: {side}")
        stop = float(first["stop_price"])
        target = float(first["target_price"])
        if not np.isfinite([stop, target]).all():
            raise ValueError("stop and target must be finite")
        start = int(minute_index.searchsorted(first["t2_time"], side="left"))
        stop_hit = False
        target_hit = False
        for row_index, row in group.iterrows():
            cutoff = int(
                minute_index.searchsorted(row["decision_time"], side="left")
            )
            if cutoff > start:
                observed_high = high[start:cutoff]
                observed_low = low[start:cutoff]
                if side == "long":
                    stop_hit = stop_hit or bool(np.any(observed_low <= stop))
                    target_hit = target_hit or bool(np.any(observed_high >= target))
                else:
                    stop_hit = stop_hit or bool(np.any(observed_high >= stop))
                    target_hit = target_hit or bool(np.any(observed_low <= target))
                start = cutoff
            stop_flags.loc[row_index] = stop_hit
            target_flags.loc[row_index] = target_hit

    work["stop_invalidated_before_decision"] = stop_flags
    work["target_reached_before_decision"] = target_flags
    risk_bps = pd.to_numeric(work["distance_to_stop_bps"], errors="coerce")
    target_bps = pd.to_numeric(work["distance_to_target_bps"], errors="coerce")
    rr = pd.to_numeric(work["rr_proxy"], errors="coerce")
    r_net = pd.to_numeric(work["r_net"], errors="coerce")
    recognised = work["outcome"].isin(OUTCOME_TO_CLASS)
    filled = work["filled"].astype(bool)

    work["planned_cost_r_10bps"] = np.divide(
        ROUND_TRIP_COST_BPS,
        risk_bps,
        out=np.full(len(work), np.inf, dtype=float),
        where=risk_bps.to_numpy(float) > 0.0,
    )
    work["planned_tp_gross_r"] = rr
    work["outcome_class"] = (
        work["outcome"].map(OUTCOME_TO_CLASS).fillna(-1).astype("int8")
    )
    work["observed_gross_r"] = r_net + work["planned_cost_r_10bps"]
    work["catastrophic_loss"] = r_net.lt(-2.0)

    common = (
        ~work["stop_invalidated_before_decision"]
        & filled
        & recognised
        & risk_bps.notna()
        & target_bps.gt(0.0)
        & rr.gt(0.0)
    )
    work["primary_eligible"] = common & risk_bps.ge(PRIMARY_MIN_RISK_BPS)
    work["sensitivity_40_eligible"] = common & risk_bps.ge(
        SENSITIVITY_MIN_RISK_BPS
    )
    work["target_cancel_eligible"] = (
        work["primary_eligible"] & ~work["target_reached_before_decision"]
    )
    work = work.sort_values("_original_order", kind="stable").drop(
        columns="_original_order"
    )
    return work.reset_index(drop=True)
