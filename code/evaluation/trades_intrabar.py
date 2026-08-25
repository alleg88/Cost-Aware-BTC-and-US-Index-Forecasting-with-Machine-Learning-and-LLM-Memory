"""Sub-M15 exit engine for bracket trades at one-minute or one-second resolution.

Signals still live on the M15 grid — a gated class prediction at the close of M15
bar *t* opens a trade at the OPEN of M15 bar *t+1*, exactly as in
`evaluation.trades.simulate_bracket_trades`. The only thing that changes here is
**how the exit is resolved**: instead of scanning M15 OHLC (which cannot order two
barrier touches inside one 15-minute bar, forcing the conservative stop-first
assumption), the trade's life is replayed on the selected underlying execution bars.

Why this matters:
  * Across different 1m bars the true touch order is known, so a take-profit that
    is reached before the stop is now booked as a win (the M15 engine would have
    mis-charged it as a stop whenever both sat inside the same M15 bar).
  * Tight targets (25-50 bps) become simulatable honestly — they no longer alias
    inside a 15-minute range.
  * It enables path-dependent exits the M15 engine cannot express: a **breakeven**
    stop (move the stop to entry once the trade is far enough in profit) and a
    **trailing** stop (ratchet the stop behind the best price reached).

The residual stop-first assumption survives only when TP and SL fall inside the
*same source* bar — a far smaller and rarer ambiguity than at M15.

Output contract matches the M15 engine: a trade ledger plus a per-bar return
series on the **M15 grid** (marks = M15 closes, bracketed by the entry open and
the intrabar exit price) whose sum reconciles exactly with the ledger, so
`economics_summary` / `diebold_mariano` stay directly comparable with notebooks
01-06.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.trades import TRADE_COLUMNS, _empty_ledger, trade_stats  # noqa: F401

_MIN15_NS = 15 * 60 * 1_000_000_000  # 15 minutes in nanoseconds


def simulate_bracket_trades_intrabar(
    bars: pd.DataFrame,
    minute_bars: pd.DataFrame,
    pred: pd.Series,
    conf: pd.Series | None = None,
    *,
    tau: float = 0.0,
    tp_bps: float,
    sl_bps: float,
    max_hold: int,
    fee_bps: float,
    slippage_bps: float = 0.0,
    tp_bps_short: float | None = None,
    sl_bps_short: float | None = None,
    vol_scale_col: str | None = None,
    trail_bps: float | pd.Series | None = None,
    be_trigger_bps: float | None = None,
    cooldown_bars: int = 0,
    expected_interval: pd.Timedelta | None = None,
    include_audit: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Bracket trades with exits resolved on one-minute or one-second bars.

    bars: M15 OHLC frame (the signal/return grid), UTC index.
    minute_bars: execution OHLC frame (1m or 1s) covering the same span, UTC index.
    pred/conf, tau, tp_bps/sl_bps, max_hold, fee_bps, slippage_bps, tp_bps_short,
        sl_bps_short, vol_scale_col: as in evaluation.trades.simulate_bracket_trades
        (max_hold is counted in M15 bars; barriers are bps of entry price, or
        multiples of bars[vol_scale_col] at the signal bar when vol_scale_col is set).
    trail_bps: trailing-stop distance behind the best price reached since entry
        (same unit as tp/sl). A Series supplies a causal distance per signal bar;
        None disables trailing.
    be_trigger_bps: once favourable excursion reaches this distance, the stop is
        moved to the entry price (breakeven). None disables. Combines with
        trailing (whichever stop is tighter wins).

    Returns (ledger, per_bar_returns) with per_bar_returns on the M15 grid.
    """
    if max_hold < 1:
        raise ValueError("max_hold must be >= 1")
    if cooldown_bars < 0:
        raise ValueError("cooldown_bars must be >= 0")
    if vol_scale_col is not None and vol_scale_col not in bars.columns:
        raise ValueError(f"vol_scale_col {vol_scale_col!r} not in bars")
    interval_ns = None
    interval_seconds = np.nan
    interval_label = "intrabar"
    if expected_interval is not None:
        expected_interval = pd.Timedelta(expected_interval)
        if expected_interval <= pd.Timedelta(0):
            raise ValueError("expected_interval must be positive")
        interval_ns = int(expected_interval.as_unit("ns").value)
        interval_seconds = expected_interval.total_seconds()
        interval_label = f"{interval_seconds:g}s"

    m_index = minute_bars.index
    if m_index.tz is None:
        raise ValueError("minute_bars must be tz-aware (UTC)")
    m_ns = m_index.as_unit("ns").view("int64")
    m_open = minute_bars["open"].to_numpy(dtype=float)
    m_high = minute_bars["high"].to_numpy(dtype=float)
    m_low = minute_bars["low"].to_numpy(dtype=float)
    m_close = minute_bars["close"].to_numpy(dtype=float)

    b_ns = bars.index.as_unit("ns").view("int64")
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    n = len(bars)

    signal = pd.Series(np.nan, index=bars.index)
    signal.loc[pred.index.intersection(bars.index)] = pred.astype(float)
    if conf is not None and tau > 0.0:
        conf_full = pd.Series(np.nan, index=bars.index)
        conf_full.loc[conf.index.intersection(bars.index)] = conf.astype(float)
        signal = signal.where(conf_full >= tau)
    sig = signal.to_numpy()

    sigma = bars[vol_scale_col].to_numpy(dtype=float) if vol_scale_col else None
    trail_values = None
    if isinstance(trail_bps, pd.Series):
        trail_values = trail_bps.reindex(bars.index).to_numpy(dtype=float)
    cost_per_side = (float(fee_bps) + float(slippage_bps)) / 10_000.0
    tp_s = tp_bps if tp_bps_short is None else tp_bps_short
    sl_s = sl_bps if sl_bps_short is None else sl_bps_short

    per_bar = np.zeros(n, dtype=float)
    trades: list[dict] = []

    i = 0
    while i < n - 1:  # a signal on the last M15 bar has no next open to enter at
        s = sig[i]
        if not (s == 0.0 or s == 2.0):
            i += 1
            continue
        side = 1 if s == 2.0 else -1

        # barrier / trail / breakeven distances as fractions of the entry price
        if vol_scale_col is not None:
            scale = sigma[i]
            if not np.isfinite(scale) or scale <= 0.0:
                i += 1
                continue
            unit = scale
        else:
            unit = 1.0 / 10_000.0
        tp_dist = (tp_bps if side == 1 else tp_s) * unit
        sl_dist = (sl_bps if side == 1 else sl_s) * unit
        trail_value = trail_values[i] if trail_values is not None else trail_bps
        trail_dist = (
            float(trail_value) * unit
            if trail_value is not None and np.isfinite(trail_value)
            else None
        )
        be_dist = be_trigger_bps * unit if be_trigger_bps is not None else None

        entry_idx = i + 1
        entry_px = open_[entry_idx]

        # 1-minute slice for the hold window: [entry bar start, +max_hold*15min)
        start_ns = b_ns[entry_idx]
        end_ns = start_ns + max_hold * _MIN15_NS
        lo = int(np.searchsorted(m_ns, start_ns, side="left"))
        hi = int(np.searchsorted(m_ns, end_ns, side="left"))
        if interval_ns is not None:
            expected_count = (end_ns - start_ns) // interval_ns
            observed = m_ns[lo:hi]
            complete = (
                len(observed) == expected_count
                and len(observed) > 0
                and observed[0] == start_ns
                and observed[-1] == end_ns - interval_ns
                and np.all(np.diff(observed) == interval_ns)
            )
            if not complete:
                raise ValueError(
                    f"missing {interval_label} execution data in "
                    f"[{pd.Timestamp(start_ns, tz='UTC')}, {pd.Timestamp(end_ns, tz='UTC')})"
                )
        if hi <= lo:                       # no intrabar data for this window
            i += 1
            continue

        if side == 1:
            tp_px = entry_px * (1.0 + tp_dist)
            sl_px = entry_px * (1.0 - sl_dist)
        else:
            tp_px = entry_px * (1.0 - tp_dist)
            sl_px = entry_px * (1.0 + sl_dist)

        best = entry_px                    # best (most favourable) price reached
        ambiguous_touch = False
        exit_ns, exit_px, exit_reason = m_ns[hi - 1], m_close[hi - 1], "timeout"
        for j in range(lo, hi):
            hit_sl = m_low[j] <= sl_px if side == 1 else m_high[j] >= sl_px
            hit_tp = m_high[j] >= tp_px if side == 1 else m_low[j] <= tp_px
            if hit_sl:                     # conservative stop-first within one source bar
                ambiguous_touch = bool(hit_tp)
                exit_ns, exit_px, exit_reason = m_ns[j], sl_px, "stop_loss"
                break
            if hit_tp:
                exit_ns, exit_px, exit_reason = m_ns[j], tp_px, "take_profit"
                break
            # no exit this bar: ratchet the dynamic stop for the next bars
            if trail_dist is not None or be_dist is not None:
                best = max(best, m_high[j]) if side == 1 else min(best, m_low[j])
                fav = (best - entry_px) / entry_px if side == 1 else (entry_px - best) / entry_px
                if trail_dist is not None:
                    trail_px = best * (1.0 - trail_dist) if side == 1 else best * (1.0 + trail_dist)
                    sl_px = max(sl_px, trail_px) if side == 1 else min(sl_px, trail_px)
                if be_dist is not None and fav >= be_dist:
                    sl_px = max(sl_px, entry_px) if side == 1 else min(sl_px, entry_px)

        # map the intrabar exit back onto the M15 grid
        exit_idx = int(np.searchsorted(b_ns, exit_ns, side="right")) - 1
        exit_idx = max(entry_idx, min(exit_idx, entry_idx + max_hold - 1, n - 1))

        gross = side * (exit_px / entry_px - 1.0)
        net = gross - 2.0 * cost_per_side

        marks = np.concatenate(([entry_px], close[entry_idx:exit_idx], [exit_px]))
        increments = side * np.diff(marks) / entry_px
        per_bar[entry_idx:exit_idx + 1] += increments
        per_bar[entry_idx] -= cost_per_side
        per_bar[exit_idx] -= cost_per_side

        trade = {
            "entry_time": bars.index[entry_idx],
            "exit_time": bars.index[exit_idx],
            "side": side,
            "entry_price": float(entry_px),
            "exit_price": float(exit_px),
            "bars_held": int(exit_idx - entry_idx + 1),
            "exit_reason": exit_reason,
            "gross_return": float(gross),
            "net_return": float(net),
        }
        if include_audit:
            trade["ambiguous_touch"] = ambiguous_touch
            trade["execution_interval_seconds"] = interval_seconds
            trade["intrabar_exit_time"] = pd.Timestamp(exit_ns, unit="ns", tz="UTC")
        trades.append(trade)
        i = exit_idx + max(1, cooldown_bars)

    audit_columns = [
        "ambiguous_touch", "execution_interval_seconds", "intrabar_exit_time"
    ] if include_audit else []
    if trades:
        ledger = pd.DataFrame(trades, columns=TRADE_COLUMNS + audit_columns)
    else:
        ledger = _empty_ledger()
        if include_audit:
            ledger["ambiguous_touch"] = pd.Series(dtype=bool)
            ledger["execution_interval_seconds"] = pd.Series(dtype=float)
            ledger["intrabar_exit_time"] = pd.Series(dtype="datetime64[ns, UTC]")
    return ledger, pd.Series(per_bar, index=bars.index, name="bracket_return")
