"""Deterministic backtest for channel faucet signals.

Execution rules, all chosen to avoid flattering the strategy:

  entry      the next bar's OPEN, never the signal bar's close. A close-based fill
             assumes an order placed on information that arrives with that close.
  stop first a bar touching both levels is scored as the stop. Bar data cannot say
             which came first, and the optimistic reading inflates results exactly
             where volatility is highest.
  targets    frozen at entry. A target recomputed from later fits is not the target
             the trade was sized against, and it cannot be resting in the book.
  timeout    closed at market and kept in the statistics. Dropping timeouts removes
             the trades that went nowhere, which is most of the losing tail.

Public API:
    backtest_channel_strategy(df_sig, ...) -> dict with summary + trades_df
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADE_COLUMNS = (
    "entry_time", "exit_time", "side", "entry", "stop", "target", "exit_price",
    "outcome", "risk_bps", "rr_planned", "r_gross", "r_net", "net_return",
    "bars_held", "channel_episode_id",
)

ORDER_COLUMNS = (
    "signal_time", "decision_time", "side", "status", "filled",
    "entry_time", "exit_time", "active_end_time", "entry", "stop", "target",
    "outcome", "r_net", "channel_episode_id",
)


def _geometry_arrays(
    df_sig: pd.DataFrame,
    *,
    swing_lookback: int,
    swing_low_col: str | None,
    swing_high_col: str | None,
    measured_move_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return signal-time geometry, optionally supplied by a coarser grid."""
    swing_lo = (
        df_sig[swing_low_col].to_numpy(dtype=float)
        if swing_low_col
        else df_sig["low"].rolling(swing_lookback).min().to_numpy()
    )
    swing_hi = (
        df_sig[swing_high_col].to_numpy(dtype=float)
        if swing_high_col
        else df_sig["high"].rolling(swing_lookback).max().to_numpy()
    )
    measured = (
        df_sig[measured_move_col].to_numpy(dtype=float)
        if measured_move_col
        else None
    )
    return swing_lo, swing_hi, measured


def _resolve(
    side: str, entry: float, stop: float, target: float,
    high: np.ndarray, low: np.ndarray, close: np.ndarray,
    start: int, max_bars: int, regime: np.ndarray | None, want: str | None,
) -> tuple[int, float, str]:
    """Walk forward to the first terminal event. Returns (bar, price, outcome)."""
    if max_bars < 1:
        raise ValueError("max_bars must be >= 1")
    requested_last = start + max_bars - 1
    last = min(requested_last, len(close) - 1)
    for j in range(start, last + 1):
        if side == "long":
            if low[j] <= stop:
                return j, stop, "sl"
            if high[j] >= target:
                return j, target, "tp"
        else:
            if high[j] >= stop:
                return j, stop, "sl"
            if low[j] <= target:
                return j, target, "tp"
        if regime is not None and want is not None and regime[j] != want:
            return j, close[j], "channel_gone"
    outcome = "censored" if requested_last >= len(close) else "timeout"
    return last, close[last], outcome


def _resolve_1m(
    side: str, entry: float, stop: float, target: float,
    minute: pd.DataFrame, start_time: pd.Timestamp, hold_minutes: int,
    regime: pd.Series | None, want: str | None, *,
    ignore_target_at_start: bool = False,
) -> tuple[pd.Timestamp, float, str]:
    """Resolve an otherwise ambiguous execution-grid bar on its observed 1m path."""
    minute_index = minute.index
    high = minute["high"].to_numpy(dtype=float, copy=False)
    low = minute["low"].to_numpy(dtype=float, copy=False)
    close = minute["close"].to_numpy(dtype=float, copy=False)
    regime_values = regime.to_numpy(copy=False) if regime is not None else None
    pos = int(minute_index.searchsorted(start_time))
    for offset in range(hold_minutes):
        t = start_time + pd.Timedelta(minutes=offset)
        if pos >= len(minute_index) or minute_index[pos] != t:
            return t, entry, "censored"
        if not (np.isfinite(high[pos]) and np.isfinite(low[pos]) and np.isfinite(close[pos])):
            return t, entry, "censored"
        if regime_values is not None and want is not None and regime_values[pos] != want:
            return t, float(close[pos]), "channel_gone"
        if side == "long":
            if low[pos] <= stop:
                return t, stop, "sl"
            if high[pos] >= target and not (ignore_target_at_start and offset == 0):
                return t, target, "tp"
        else:
            if high[pos] >= stop:
                return t, stop, "sl"
            if low[pos] <= target and not (ignore_target_at_start and offset == 0):
                return t, target, "tp"
        pos += 1
    return t, float(close[pos - 1]), "timeout"


def backtest_channel_strategy(
    df_sig: pd.DataFrame,
    tp_pct: float | None = None,
    sl_pct: float | None = None,
    max_trades_per_day: int | None = None,
    *,
    target_mode: str = "pct",
    stop_mode: str = "pct",
    swing_lookback: int = 12,
    stop_buffer_bps: float = 5.0,
    rr_multiple: float = 1.5,
    min_risk_bps: float = 0.0,
    max_risk_bps: float = np.inf,
    min_rr: float = 0.0,
    max_hold_bars: int = 48,
    cost_bps: float = 10.0,
    entry_mode: str = "market",
    limit_offset_bps: float = 0.0,
    fill_window_bars: int = 4,
    maker_fee_bps: float | None = None,
    taker_fee_bps: float | None = None,
    regime_col: str | None = None,
    episode_col: str | None = None,
    max_concurrent: int | None = 1,
    execution_1m: pd.DataFrame | None = None,
    swing_low_col: str | None = None,
    swing_high_col: str | None = None,
    measured_move_col: str | None = None,
    max_hold_minutes: int | None = None,
    fill_window_minutes: int | None = None,
) -> dict:
    """Run every non-zero `signal` through the bracket and summarise the result.

    target_mode 'pct' uses tp_pct; 'rr' uses rr_multiple x risk; 'measured' uses
    the signal-time swing range; 'rail' uses the opposite channel boundary. Every
    target is frozen when the order enters. stop_mode 'pct' uses sl_pct, 'swing'
    places the stop beyond the recent swing extreme. max_concurrent=None removes
    the portfolio-capacity gate.

    min_risk_bps / max_risk_bps reject candidates whose stop is too tight for the
    round trip to be worth paying, or so wide that the position becomes a different
    trade. Both bounds matter: at a 20 bps stop a 10 bps round trip is half the risk.
    """
    if target_mode == "pct" and tp_pct is None:
        raise ValueError("target_mode='pct' needs tp_pct")
    if stop_mode == "pct" and sl_pct is None:
        raise ValueError("stop_mode='pct' needs sl_pct")
    if max_hold_minutes is not None and max_hold_minutes < 1:
        raise ValueError("max_hold_minutes must be >= 1")
    if fill_window_minutes is not None and fill_window_minutes < 1:
        raise ValueError("fill_window_minutes must be >= 1")

    df = df_sig
    idx = df.index
    o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
    sig = df["signal"].to_numpy()
    upper = df["channel_upper"].to_numpy(dtype=float) if "channel_upper" in df else None
    lower = df["channel_lower"].to_numpy(dtype=float) if "channel_lower" in df else None
    regime = df[regime_col].to_numpy() if regime_col else None
    episode = df[episode_col].to_numpy() if episode_col else np.zeros(len(df), dtype=int)
    regime_series = df[regime_col] if regime_col else None
    swing_lo, swing_hi, measured_move_values = _geometry_arrays(
        df,
        swing_lookback=swing_lookback,
        swing_low_col=swing_low_col,
        swing_high_col=swing_high_col,
        measured_move_col=measured_move_col,
    )

    # How many signals never became trades, and why. Capacity is a portfolio choice
    # and gating is a signal choice; conflating the two hides the fact that raising
    # capacity adds trades without touching signal quality, whereas loosening gates
    # buys frequency by admitting worse candidates.
    # Cost depends on how each side of the trade actually executes. A resting entry
    # earns the maker fee, but a stop is a market order by construction and always
    # pays taker; only a target sitting at the far rail can rest. Charging a single
    # blended rate would quietly credit maker pricing to the losing trades, which is
    # precisely where it is never available.
    if maker_fee_bps is None and taker_fee_bps is None:
        fee_in = fee_out_tp = fee_out_stop = cost_bps / 2.0
    else:
        maker = cost_bps / 2.0 if maker_fee_bps is None else maker_fee_bps
        taker = cost_bps / 2.0 if taker_fee_bps is None else taker_fee_bps
        fee_in = maker if entry_mode == "maker_limit" else taker
        fee_out_tp, fee_out_stop = maker, taker

    trades, orders, per_day, open_until = [], [], {}, []
    skipped_capacity = skipped_daily_cap = skipped_geometry = unfilled = 0
    censored = channel_cancelled = 0
    if execution_1m is not None:
        if len(idx) < 2:
            raise ValueError("execution_1m needs at least two execution-grid bars")
        grid_delta = pd.Series(idx[1:] - idx[:-1]).mode().iloc[0]
        minutes_per_bar = int(grid_delta / pd.Timedelta(minutes=1))
        hold_minutes = (
            max_hold_minutes
            if max_hold_minutes is not None
            else minutes_per_bar * max_hold_bars
        )
        maker_fill_minutes = (
            fill_window_minutes
            if fill_window_minutes is not None
            else minutes_per_bar * fill_window_bars
        )
        minute_index = execution_1m.index
        minute_high = execution_1m["high"].to_numpy(dtype=float, copy=False)
        minute_low = execution_1m["low"].to_numpy(dtype=float, copy=False)
        execution_regime = (regime_series.reindex(minute_index, method="ffill")
                            if regime_series is not None else None)
        execution_regime_values = (execution_regime.to_numpy(copy=False)
                                   if execution_regime is not None else None)

    for i in np.flatnonzero(sig):
        if i + 1 >= len(df):
            continue
        open_until = [x for x in open_until if x > i]
        if max_concurrent is not None and len(open_until) >= max_concurrent:
            skipped_capacity += 1
            continue
        side = "long" if sig[i] > 0 else "short"
        decision_time = idx[i + 1]
        day = idx[i].normalize() if hasattr(idx[i], "normalize") else idx[i]
        if max_trades_per_day is not None and per_day.get(day, 0) >= max_trades_per_day:
            skipped_daily_cap += 1
            continue

        if entry_mode == "market":
            entry = o[i + 1]                                # next open
            fill_bar = i + 1
            fill_time = idx[fill_bar]
            if regime is not None and regime[fill_bar] != regime[i]:
                channel_cancelled += 1
                orders.append((idx[i], decision_time, side, "channel_cancelled", False,
                               pd.NaT, pd.NaT, decision_time, entry, np.nan, np.nan,
                               "channel_cancelled", 0.0, episode[i]))
                continue
        else:
            # A resting limit, placed away from the signal close in the direction
            # that favours us. It becomes a trade only if price comes back to it.
            side_sign = -1.0 if side == "long" else 1.0
            entry = c[i] * (1.0 + side_sign * limit_offset_bps / 1e4)
            fill_bar = -1
            fill_time = None
            active_end_time = pd.NaT
            cancelled = fill_censored = False
            if execution_1m is None:
                for j in range(i + 1, min(i + 1 + fill_window_bars, len(df))):
                    if regime is not None and regime[j] != regime[i]:
                        channel_cancelled += 1
                        cancelled = True
                        active_end_time = idx[j]
                        break
                    if ((side == "long" and l[j] <= entry)
                            or (side == "short" and h[j] >= entry)):
                        fill_bar = j
                        fill_time = idx[j]
                        break
            else:
                fill_start = idx[i + 1]
                pos = int(minute_index.searchsorted(fill_start))
                for offset in range(maker_fill_minutes):
                    t = fill_start + pd.Timedelta(minutes=offset)
                    if pos >= len(minute_index) or minute_index[pos] != t:
                        censored += 1
                        fill_censored = True
                        active_end_time = t
                        break
                    if not (np.isfinite(minute_high[pos]) and np.isfinite(minute_low[pos])):
                        censored += 1
                        fill_censored = True
                        active_end_time = t
                        break
                    if (execution_regime_values is not None
                            and execution_regime_values[pos] != regime[i]):
                        channel_cancelled += 1
                        cancelled = True
                        active_end_time = t
                        break
                    if ((side == "long" and minute_low[pos] <= entry)
                            or (side == "short" and minute_high[pos] >= entry)):
                        fill_time = t
                        fill_bar = max(i + 1, int(idx.searchsorted(t, side="right") - 1))
                        break
                    pos += 1
            if fill_bar < 0:
                if pd.isna(active_end_time):
                    if execution_1m is not None:
                        active_end_time = fill_start + pd.Timedelta(minutes=maker_fill_minutes)
                    else:
                        expiry = min(i + 1 + fill_window_bars, len(idx) - 1)
                        active_end_time = idx[expiry]
                if cancelled:
                    orders.append((idx[i], decision_time, side, "channel_cancelled", False,
                                   pd.NaT, pd.NaT, active_end_time, entry, np.nan, np.nan,
                                   "channel_cancelled", 0.0, episode[i]))
                elif fill_censored:
                    orders.append((idx[i], decision_time, side, "censored", False,
                                   pd.NaT, pd.NaT, active_end_time, entry, np.nan, np.nan,
                                   "censored", np.nan, episode[i]))
                else:
                    unfilled += 1
                    orders.append((idx[i], decision_time, side, "unfilled", False,
                                   pd.NaT, pd.NaT, active_end_time, entry, np.nan, np.nan,
                                   "unfilled", 0.0, episode[i]))
                continue
        if stop_mode == "pct":
            stop = entry * (1 - sl_pct) if side == "long" else entry * (1 + sl_pct)
        else:
            ref = swing_lo[i] if side == "long" else swing_hi[i]
            if not np.isfinite(ref):
                skipped_geometry += 1
                continue
            buf = 1 - stop_buffer_bps / 1e4 if side == "long" else 1 + stop_buffer_bps / 1e4
            stop = ref * buf
        risk = (entry - stop) if side == "long" else (stop - entry)
        if not np.isfinite(risk) or risk <= 0:
            skipped_geometry += 1
            continue
        risk_bps = risk / entry * 1e4
        if not (min_risk_bps <= risk_bps <= max_risk_bps):
            skipped_geometry += 1
            continue

        if target_mode == "pct":
            target = entry * (1 + tp_pct) if side == "long" else entry * (1 - tp_pct)
        elif target_mode == "rr":
            target = entry + rr_multiple * risk if side == "long" else entry - rr_multiple * risk
        elif target_mode == "measured":
            measured_move = (
                measured_move_values[i]
                if measured_move_values is not None
                else swing_hi[i] - swing_lo[i]
            )
            if not np.isfinite(measured_move) or measured_move <= 0:
                skipped_geometry += 1
                continue
            target = entry + measured_move if side == "long" else entry - measured_move
        elif target_mode == "rail":
            if upper is None or lower is None:
                raise ValueError("target_mode='rail' needs channel bounds")
            target = upper[i] if side == "long" else lower[i]   # frozen at entry
        else:
            raise ValueError(f"unknown target_mode: {target_mode!r}")
        reward = (target - entry) if side == "long" else (entry - target)
        if not np.isfinite(reward) or reward <= 0 or reward / risk < min_rr:
            skipped_geometry += 1
            continue

        want = regime[i] if regime is not None else None
        if execution_1m is None:
            j, px, outcome = _resolve(side, entry, stop, target, h, l, c,
                                      fill_bar, max_hold_bars, regime, want)
            exit_time = idx[j]
        else:
            exit_time, px, outcome = _resolve_1m(
                side, entry, stop, target, execution_1m, fill_time,
                hold_minutes, execution_regime, want,
                ignore_target_at_start=(entry_mode == "maker_limit"),
            )
            j = max(fill_bar, min(int(idx.searchsorted(exit_time, side="right") - 1),
                                  len(idx) - 1))
        if outcome == "censored":
            censored += 1
            orders.append((idx[i], decision_time, side, "censored", True,
                           fill_time, exit_time, exit_time, entry, stop, target, outcome,
                           np.nan, episode[i]))
            continue
        move = (px - entry) if side == "long" else (entry - px)
        r_gross = move / risk
        round_trip = fee_in + (fee_out_tp if outcome == "tp" else fee_out_stop)
        r_net = r_gross - (round_trip / 1e4) * entry / risk
        open_until.append(j)
        per_day[day] = per_day.get(day, 0) + 1
        orders.append((idx[i], decision_time, side, "filled", True,
                       fill_time, exit_time, exit_time, entry, stop, target, outcome,
                       r_net, episode[i]))
        trades.append((fill_time, exit_time, side, entry, stop, target, px, outcome,
                       risk_bps, reward / risk, r_gross, r_net,
                       move / entry - round_trip / 1e4, j - fill_bar + 1, episode[i]))

    skipped = {"capacity": skipped_capacity, "daily_cap": skipped_daily_cap,
               "geometry": skipped_geometry, "unfilled": unfilled,
               "censored": censored, "channel_cancelled": channel_cancelled}
    orders_df = pd.DataFrame(orders, columns=list(ORDER_COLUMNS))
    trades_df = pd.DataFrame(trades, columns=list(TRADE_COLUMNS))
    if trades_df.empty:
        return {"num_trades": 0, "win_rate": float("nan"), "total_net_return": 0.0,
                "mean_r_net": float("nan"), "mean_r_gross": float("nan"),
                "tp_first_rate": float("nan"), "timeout_rate": float("nan"),
                "profit_factor": float("nan"), "num_episodes": 0,
                "trades_df": trades_df, "orders_df": orders_df, "skipped": skipped}

    wins = trades_df["r_net"] > 0
    gain = trades_df.loc[wins, "r_net"].sum()
    loss = -trades_df.loc[~wins, "r_net"].sum()
    return {
        "num_trades": int(len(trades_df)),
        "win_rate": float(wins.mean()),
        "total_net_return": float(trades_df["net_return"].sum()),
        "mean_r_net": float(trades_df["r_net"].mean()),
        "mean_r_gross": float(trades_df["r_gross"].mean()),
        "se_r_gross": float(trades_df["r_gross"].std(ddof=1) / np.sqrt(len(trades_df))),
        "tp_first_rate": float((trades_df["outcome"] == "tp").mean()),
        "timeout_rate": float((trades_df["outcome"] == "timeout").mean()),
        "profit_factor": float(gain / loss) if loss > 0 else float("inf"),
        "num_episodes": int(trades_df["channel_episode_id"].nunique()),
        "skipped": skipped,
        "trades_df": trades_df,
        "orders_df": orders_df,
    }
