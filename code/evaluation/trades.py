"""Bracket-order trade simulation: take-profit / stop-loss / time-out exits.

Turns per-bar class predictions into discrete trades instead of per-bar position
flipping. A gated signal at the close of bar t opens a trade at the OPEN of bar
t+1 (no lookahead); the trade then exits at the first of:

  - take-profit touched (long: high >= entry * (1 + tp); short mirrored),
  - stop-loss touched  (long: low  <= entry * (1 - sl); short mirrored),
  - time-out after `max_hold` bars (exit at that bar's close).

Barrier checks include the entry bar itself (entry is at the open, so the same
bar's high/low can already touch a barrier). When both barriers fall inside one
bar the fill order is unknowable from OHLC data, so the STOP is assumed to fill
first — the conservative convention. One trade is open at a time; signals that
arrive while a trade is open are ignored.

Costs (fee + slippage, in bps) are charged per side, on entry and on exit.

Two outputs per simulation:
  - a trade ledger (one row per trade, with exit_reason);
  - a per-bar return series where each holding bar contributes
    side * (mark_i - mark_{i-1}) / entry_price (marks = closes, bracketed by the
    entry and exit prices). Increments are normalized by the entry price so the
    series sums EXACTLY to the ledger's net return — consistent with the
    additive (non-compounded) cumulative curves used across the evaluation
    stack, and directly usable by economics_summary / diebold_mariano.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADE_COLUMNS = [
    "entry_time", "exit_time", "side", "entry_price", "exit_price",
    "bars_held", "exit_reason", "gross_return", "net_return",
]


def _empty_ledger() -> pd.DataFrame:
    return pd.DataFrame({
        "entry_time": pd.Series(dtype="datetime64[ns, UTC]"),
        "exit_time": pd.Series(dtype="datetime64[ns, UTC]"),
        "side": pd.Series(dtype="int64"),
        "entry_price": pd.Series(dtype="float64"),
        "exit_price": pd.Series(dtype="float64"),
        "bars_held": pd.Series(dtype="int64"),
        "exit_reason": pd.Series(dtype="object"),
        "gross_return": pd.Series(dtype="float64"),
        "net_return": pd.Series(dtype="float64"),
    })


def simulate_bracket_trades(
    bars: pd.DataFrame,
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
) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate bracket trades from gated class predictions.

    bars: OHLC frame (open/high/low/close) on the full bar grid, UTC index.
    pred/conf: class predictions (down=0/flat=1/up=2) and top-class probability,
        indexed by the bar whose CLOSE the prediction was made at. Bars missing
        from `pred` simply produce no signal.
    tp_bps/sl_bps: barrier distances in bps of the entry price (long side; the
        short side uses tp_bps_short/sl_bps_short when given, else the same).
    vol_scale_col: when set, barriers become distance = multiplier * sigma where
        sigma is `bars[vol_scale_col]` at the SIGNAL bar (a fractional
        volatility) and tp_bps/sl_bps are reinterpreted as multipliers.
    max_hold: maximum bars held, counting the entry bar.

    Returns (ledger, per_bar_returns); per_bar_returns is 0.0 on flat bars and
    covers the full `bars` index.
    """
    if max_hold < 1:
        raise ValueError("max_hold must be >= 1")
    if vol_scale_col is not None and vol_scale_col not in bars.columns:
        raise ValueError(f"vol_scale_col {vol_scale_col!r} not in bars")

    open_ = bars["open"].to_numpy(dtype=float)
    high = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
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
    cost_per_side = (float(fee_bps) + float(slippage_bps)) / 10_000.0
    tp_s = tp_bps if tp_bps_short is None else tp_bps_short
    sl_s = sl_bps if sl_bps_short is None else sl_bps_short

    per_bar = np.zeros(n, dtype=float)
    trades: list[dict] = []

    i = 0
    while i < n - 1:  # a signal on the last bar has no next open to enter at
        s = sig[i]
        if not (s == 0.0 or s == 2.0):
            i += 1
            continue
        side = 1 if s == 2.0 else -1
        if vol_scale_col is not None:
            scale = sigma[i]
            if not np.isfinite(scale) or scale <= 0.0:
                i += 1
                continue
            tp_dist = (tp_bps if side == 1 else tp_s) * scale
            sl_dist = (sl_bps if side == 1 else sl_s) * scale
        else:
            tp_dist = (tp_bps if side == 1 else tp_s) / 10_000.0
            sl_dist = (sl_bps if side == 1 else sl_s) / 10_000.0

        entry_idx = i + 1
        entry_px = open_[entry_idx]
        if side == 1:
            tp_px = entry_px * (1.0 + tp_dist)
            sl_px = entry_px * (1.0 - sl_dist)
        else:
            tp_px = entry_px * (1.0 - tp_dist)
            sl_px = entry_px * (1.0 + sl_dist)

        last_idx = min(entry_idx + max_hold - 1, n - 1)
        exit_idx, exit_px, exit_reason = last_idx, close[last_idx], "timeout"
        for j in range(entry_idx, last_idx + 1):
            hit_sl = low[j] <= sl_px if side == 1 else high[j] >= sl_px
            hit_tp = high[j] >= tp_px if side == 1 else low[j] <= tp_px
            if hit_sl:  # stop assumed first when both touch in one bar
                exit_idx, exit_px, exit_reason = j, sl_px, "stop_loss"
                break
            if hit_tp:
                exit_idx, exit_px, exit_reason = j, tp_px, "take_profit"
                break

        gross = side * (exit_px / entry_px - 1.0)
        net = gross - 2.0 * cost_per_side

        # Per-bar increments over entry price: entry px -> closes -> exit px.
        marks = np.concatenate(([entry_px], close[entry_idx:exit_idx], [exit_px]))
        increments = side * np.diff(marks) / entry_px
        per_bar[entry_idx:exit_idx + 1] += increments
        per_bar[entry_idx] -= cost_per_side
        per_bar[exit_idx] -= cost_per_side

        trades.append({
            "entry_time": bars.index[entry_idx],
            "exit_time": bars.index[exit_idx],
            "side": side,
            "entry_price": float(entry_px),
            "exit_price": float(exit_px),
            "bars_held": int(exit_idx - entry_idx + 1),
            "exit_reason": exit_reason,
            "gross_return": float(gross),
            "net_return": float(net),
        })
        i = exit_idx + 1  # signals during the open trade are ignored

    ledger = pd.DataFrame(trades, columns=TRADE_COLUMNS) if trades else _empty_ledger()
    return ledger, pd.Series(per_bar, index=bars.index, name="bracket_return")


def trade_stats(ledger: pd.DataFrame) -> dict[str, float]:
    """Per-trade economics: the numbers a trading desk would ask for first."""
    out = {
        "n_trades": int(len(ledger)),
        "n_long": int((ledger["side"] == 1).sum()) if len(ledger) else 0,
        "n_short": int((ledger["side"] == -1).sum()) if len(ledger) else 0,
    }
    if not len(ledger):
        out.update({"win_rate": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
                    "profit_factor": 0.0, "expectancy_bps": 0.0,
                    "median_bars_held": 0.0,
                    "tp_rate": 0.0, "sl_rate": 0.0, "timeout_rate": 0.0})
        return out
    net = ledger["net_return"].astype(float)
    wins, losses = net[net > 0.0], net[net <= 0.0]
    gross_win, gross_loss = wins.sum(), -losses.sum()
    reasons = ledger["exit_reason"].value_counts(normalize=True)
    out.update({
        "win_rate": float(len(wins) / len(net)),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "expectancy_bps": float(net.mean() * 10_000.0),
        "median_bars_held": float(ledger["bars_held"].median()),
        "tp_rate": float(reasons.get("take_profit", 0.0)),
        "sl_rate": float(reasons.get("stop_loss", 0.0)),
        "timeout_rate": float(reasons.get("timeout", 0.0)),
    })
    return out
