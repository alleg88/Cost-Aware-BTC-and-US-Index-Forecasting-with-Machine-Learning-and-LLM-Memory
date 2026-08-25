"""Build the leak-safe E7 event table for the channel study.

One row is an order candidate.  Model inputs are frozen at the signal-bar close;
the next Open or later maker fill may determine the label, but can never become an
input.  Unfilled and channel-cancelled orders are economic zeroes and remain in the
table.  Censored paths have no knowable label and are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


CHANNEL_EVENT_FEATURES = (
    "channel_slope",
    "channel_r2",
    "channel_width_pct",
    "channel_pos",
    "channel_confluence",
    "risk_bps_decision",
    "rr_planned_decision",
    "rsi_regime_pct",
    "pullback_depth",
    "taker_imbalance",
    "funding_z",
    "oi_chg_4h",
    "episode_trade_number",
)

FORBIDDEN_EVENT_FEATURES = frozenset({
    "bars_held", "outcome", "exit_price", "exit_time", "entry_time",
    "r_gross", "r_net", "net_return", "filled", "order_status",
    "label_net_positive",
})


def causal_rolling_percentile(
    values: pd.Series, *, window: int, min_periods: int | None = None,
) -> pd.Series:
    """Percentile rank of the current value inside its trailing-only window."""
    minimum = window if min_periods is None else min_periods
    return values.rolling(window, min_periods=minimum).rank(pct=True)


def prepare_positioning_features(
    raw: pd.DataFrame,
    *,
    bar_size: str = "15min",
    funding_window: int = 672,
    oi_change_periods: int = 16,
) -> pd.DataFrame:
    """Create causal funding/OI inputs and timestamp them at source-bar close."""
    required = {"funding_rate", "sum_open_interest"}
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise ValueError(f"positioning data missing columns: {missing}")
    out = pd.DataFrame(index=raw.index)
    funding = raw["funding_rate"].astype(float)
    out["funding_z"] = (
        (funding - funding.rolling(funding_window).mean())
        / funding.rolling(funding_window).std().replace(0.0, np.nan)
    )
    oi = np.log(raw["sum_open_interest"].astype(float).where(lambda x: x > 0))
    out["oi_chg_4h"] = oi.diff(oi_change_periods)
    stale = (raw["positioning_stale"].fillna(True).astype(bool)
             if "positioning_stale" in raw else pd.Series(False, index=raw.index))
    out["positioning_stale"] = stale
    if "positioning_age_min" in raw:
        out["positioning_age_min"] = raw["positioning_age_min"].astype(float)
    out.loc[stale, ["funding_z", "oi_chg_4h"]] = np.nan
    out.index = out.index + pd.Timedelta(bar_size)
    return out


def build_channel_event_dataset(
    signals: pd.DataFrame,
    orders: pd.DataFrame,
    *,
    swing_lookback: int = 12,
    stop_buffer_bps: float = 5.0,
    target_mode: str = "rail",
    rr_multiple: float = 1.5,
    tp_pct: float | None = None,
    min_risk_bps: float = 25.0,
    max_risk_bps: float = 250.0,
    min_rr: float = 0.0,
) -> pd.DataFrame:
    """Return labelled E7 rows using only information known at decision time."""
    required_signal = {
        "close", "high", "low", "channel_slope", "channel_r2",
        "channel_mid", "channel_upper", "channel_lower", "channel_width",
        "channel_pos", "channel_confluence", "rsi_regime_pct",
        "taker_imbalance", "funding_z",
        "oi_chg_4h", "channel_episode_id",
    }
    missing = sorted(required_signal.difference(signals.columns))
    if missing:
        raise ValueError(f"signals missing E7 columns: {missing}")
    required_order = {
        "signal_time", "decision_time", "side", "status", "filled", "r_net",
        "channel_episode_id",
    }
    missing = sorted(required_order.difference(orders.columns))
    if missing:
        raise ValueError(f"orders missing E7 columns: {missing}")

    ordered = orders.sort_values(["decision_time", "signal_time"]).copy()
    ordered["episode_trade_number"] = (
        ordered.groupby(["channel_episode_id", "side"], sort=False).cumcount() + 1
    )
    labelled = ordered[(ordered["status"] != "censored") & ordered["r_net"].notna()]

    swing_low = signals["low"].rolling(swing_lookback).min()
    swing_high = signals["high"].rolling(swing_lookback).max()
    rows: list[dict] = []
    for order in labelled.itertuples(index=False):
        if order.signal_time not in signals.index:
            raise ValueError(f"order signal_time not found in signals: {order.signal_time}")
        bar = signals.loc[order.signal_time]
        side_sign = 1 if order.side == "long" else -1
        decision_price = float(bar["close"])
        ref = float(swing_low.loc[order.signal_time] if side_sign > 0
                    else swing_high.loc[order.signal_time])
        buffer = 1.0 - stop_buffer_bps / 1e4 if side_sign > 0 else 1.0 + stop_buffer_bps / 1e4
        stop = ref * buffer
        risk = decision_price - stop if side_sign > 0 else stop - decision_price
        risk_bps = risk / decision_price * 1e4
        if target_mode == "measured":
            reward = float(swing_high.loc[order.signal_time]
                           - swing_low.loc[order.signal_time])
        elif target_mode == "rr":
            reward = rr_multiple * risk
        elif target_mode == "pct":
            if tp_pct is None:
                raise ValueError("target_mode='pct' needs tp_pct")
            reward = decision_price * tp_pct
        elif target_mode == "rail":
            target = float(bar["channel_upper"] if side_sign > 0
                           else bar["channel_lower"])
            reward = (target - decision_price if side_sign > 0
                      else decision_price - target)
        else:
            raise ValueError(f"unknown target_mode: {target_mode!r}")
        if (not np.isfinite(risk_bps) or not min_risk_bps <= risk_bps <= max_risk_bps
                or not np.isfinite(reward) or reward <= 0 or reward / risk < min_rr):
            continue

        mid = float(bar["channel_mid"])
        rail = float(bar["channel_lower"] if side_sign > 0 else bar["channel_upper"])
        pullback_scale = abs(mid - rail)
        pullback = ((mid - decision_price) if side_sign > 0 else (decision_price - mid))
        pullback = pullback / pullback_scale if pullback_scale > 0 else np.nan
        row = {
            "signal_time": order.signal_time,
            "decision_time": order.decision_time,
            "side": order.side,
            "channel_episode_id": order.channel_episode_id,
            "order_status": order.status,
            "filled": bool(order.filled),
            "r_net": float(order.r_net),
            "label_net_positive": int(order.r_net > 0),
            "channel_slope": float(bar["channel_slope"]),
            "channel_r2": float(bar["channel_r2"]),
            "channel_width_pct": float(bar["channel_width"]) / decision_price,
            "channel_pos": float(bar["channel_pos"]),
            "channel_confluence": int(bar["channel_confluence"]),
            "risk_bps_decision": risk_bps,
            "rr_planned_decision": reward / risk,
            "rsi_regime_pct": float(bar["rsi_regime_pct"]),
            "pullback_depth": pullback,
            "taker_imbalance": float(bar["taker_imbalance"]),
            "funding_z": float(bar["funding_z"]),
            "oi_chg_4h": float(bar["oi_chg_4h"]),
            "episode_trade_number": int(order.episode_trade_number),
        }
        rows.append(row)

    columns = [
        "signal_time", "decision_time", "side", "channel_episode_id",
        "order_status", "filled", "r_net", "label_net_positive",
        *CHANNEL_EVENT_FEATURES,
    ]
    events = pd.DataFrame(rows, columns=columns)
    events = events.dropna(subset=list(CHANNEL_EVENT_FEATURES))
    return events.sort_values("decision_time").reset_index(drop=True)
