"""The three-stage entry trigger ("faucet") evaluated inside a channel.

The three stages deliberately sit on different bars and draw on different data,
so that agreement between them carries information:

  stage 1  a state: price is in the near band of the channel and RSI is stretched
  stage 2  an event: a reversal candle prints while the state is still armed
  stage 3  a confirmation: the next bars follow through, backed by taker flow

Requiring all three on one bar would collapse them into a single observation and
lose the sequencing the design is built on; each stage therefore opens a bounded
window for the next, and the sequence resets if that window expires.

The flow input is `taker_imbalance` = 2 * taker_buy_base / volume - 1, derived from
Binance kline aggressor volume. It is a coarse proxy for order-flow imbalance and
is not the limit-order-book OFI of the microstructure literature, which needs L2
depth this dataset does not carry.

Public API:
    generate_channel_faucet_signals(df_ch, ...) -> DataFrame + signal columns
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SIGNAL_COLUMNS = (
    "signal_1_long", "signal_1_short", "signal_2", "signal_3",
    "signal", "taker_imbalance",
)


def _taker_imbalance(df: pd.DataFrame) -> pd.Series:
    if "taker_buy_base" not in df.columns or "volume" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="taker_imbalance")
    vol = df["volume"].replace(0.0, np.nan)
    return (2.0 * df["taker_buy_base"] / vol - 1.0).rename("taker_imbalance")


def _reversal_candles(df: pd.DataFrame, wick_frac: float, close_frac: float
                      ) -> tuple[pd.Series, pd.Series]:
    """Hammer/rejection or engulfing, in each direction."""
    o, h, l, c = (df[k] for k in ("open", "high", "low", "close"))
    rng = (h - l).replace(0.0, np.nan)

    lower_wick = (np.minimum(o, c) - l) / rng
    upper_wick = (h - np.maximum(o, c)) / rng
    close_high = (c - l) / rng                 # 1.0 = closed on the high
    close_low = (h - c) / rng

    body_lo, body_hi = np.minimum(o, c), np.maximum(o, c)
    prev_lo, prev_hi = body_lo.shift(1), body_hi.shift(1)

    bull = (((lower_wick >= wick_frac) & (close_high >= 1.0 - close_frac))
            | ((c > o) & (body_lo <= prev_lo) & (body_hi >= prev_hi)))
    bear = (((upper_wick >= wick_frac) & (close_low >= 1.0 - close_frac))
            | ((c < o) & (body_lo <= prev_lo) & (body_hi >= prev_hi)))
    return bull.fillna(False), bear.fillna(False)


def generate_channel_faucet_signals(
    df_ch: pd.DataFrame,
    long_pos_threshold: float = 0.30,
    short_pos_threshold: float = 0.70,
    rsi_oversold: float = 35.0,
    rsi_overbought: float = 65.0,
    arm_max_bars: int = 3,
    *,
    confirm_max_bars: int = 2,
    require_confirmation: bool = True,
    require_flow: bool = False,
    wick_frac: float = 0.40,
    close_frac: float = 0.35,
    regime_col: str | None = None,
    arm_rsi_col: str = "rsi",
) -> pd.DataFrame:
    """Run the state machine over `df_ch` and return it with the signal columns.

    `signal` is +1 / -1 on the bar where the sequence completes and 0 elsewhere.
    The entry itself belongs on the NEXT bar's open; this function marks the
    decision bar, not the fill.

    regime_col, when given, restricts each side to bars whose channel regime allows
    it, so a long sequence cannot survive the channel flipping underneath it.
    """
    out = df_ch.copy()
    out["taker_imbalance"] = _taker_imbalance(out)
    bull, bear = _reversal_candles(out, wick_frac, close_frac)

    # Stage 1 reads RSI on the channel's own timeframe: "stretched" is a property
    # of the trend being faded, not of the bar being entered on. Stage 3 reads the
    # execution-grid RSI, because follow-through is a property of that bar.
    pos = out["channel_pos"]
    rsi = out[arm_rsi_col]
    exec_rsi = out["rsi"] if "rsi" in out.columns else rsi
    slope = out["channel_slope"]
    regime = out[regime_col] if regime_col else None

    allow_long = (slope > 0) if regime is None else (regime == "up")
    allow_short = (slope < 0) if regime is None else (regime == "down")

    arm_long = (allow_long & (pos <= long_pos_threshold) & (rsi <= rsi_oversold)).fillna(False)
    arm_short = (allow_short & (pos >= short_pos_threshold) & (rsi >= rsi_overbought)).fillna(False)

    out["signal_1_long"] = arm_long.astype(int)
    out["signal_1_short"] = arm_short.astype(int)

    h, c = out["high"].to_numpy(), out["close"].to_numpy()
    flow = out["taker_imbalance"].to_numpy()
    rsi_v = exec_rsi.to_numpy()
    al, ash = arm_long.to_numpy(), arm_short.to_numpy()
    bl, br = bull.to_numpy(), bear.to_numpy()
    lo = out["low"].to_numpy()
    n = len(out)

    stage2 = np.zeros(n, dtype=int)
    stage3 = np.zeros(n, dtype=int)
    signal = np.zeros(n, dtype=int)
    allow_arrays = {"long": allow_long.to_numpy(), "short": allow_short.to_numpy()}

    armed_at = {"long": -(10 ** 9), "short": -(10 ** 9)}
    rev_at = {"long": None, "short": None}

    for i in range(n):
        for side, armed, rev_ok in (("long", al, bl), ("short", ash, br)):
            allow = allow_arrays[side]
            if not allow[i]:
                rev_at[side] = None
                armed_at[side] = -(10 ** 9)
                continue
            if (rev_at[side] is None and armed_at[side] >= 0
                    and i - armed_at[side] > arm_max_bars):
                armed_at[side] = -(10 ** 9)
            if armed[i] and rev_at[side] is None and armed_at[side] < 0:
                armed_at[side] = i
            pending = rev_at[side]
            if pending is not None:
                if i - pending > confirm_max_bars:
                    rev_at[side] = None
                else:
                    broke = c[i] > h[pending] if side == "long" else c[i] < lo[pending]
                    flow_ok = (not require_flow) or (
                        np.isfinite(flow[i])
                        and (flow[i] > 0 if side == "long" else flow[i] < 0)
                    )
                    if broke and flow_ok:
                        stage3[i] = 1
                        signal[i] = 1 if side == "long" else -1
                        rev_at[side] = None
                        armed_at[side] = -(10 ** 9)
                        continue
            if 0 < i - armed_at[side] <= arm_max_bars and rev_ok[i]:
                stage2[i] = 1
                if require_confirmation:
                    rev_at[side] = i
                else:
                    signal[i] = 1 if side == "long" else -1
                    armed_at[side] = -(10 ** 9)

    out["signal_2"] = stage2
    out["signal_3"] = stage3
    out["signal"] = signal
    return out


def faucet_funnel(df_sig: pd.DataFrame) -> pd.DataFrame:
    """Count survivors at each stage. A trigger that discards almost everything is
    not obviously wrong, but it must be visible rather than inferred from the trade
    count at the end."""
    total = len(df_sig)
    rows = [
        ("all bars", total),
        ("stage 1 armed", int((df_sig["signal_1_long"] + df_sig["signal_1_short"]).gt(0).sum())),
        ("stage 2 reversal", int(df_sig["signal_2"].sum())),
        ("stage 3 confirmed", int(df_sig["signal_3"].sum())),
        ("signal", int((df_sig["signal"] != 0).sum())),
    ]
    frame = pd.DataFrame(rows, columns=["stage", "count"])
    frame["share_of_bars"] = frame["count"] / max(total, 1)
    return frame
