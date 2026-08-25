"""Rolling linear-regression channels and the regime slicing built on them.

A channel is an ordinary least-squares fit of price against time over a trailing
window of fully closed bars, with a band drawn around it. Everything here is
causal: the window ending at bar t contains bar t and the bars before it, and no
value is ever derived from a bar that had not closed at that point.

The band can be drawn two ways. `std` places it at a multiple of the residual
standard deviation, which is the conventional form. `quantile` places it at the
10th and 90th residual percentiles, which does not assume symmetric or normal
residuals and is not moved by a single spike; on BTC that difference is material,
because one outlying hour can widen a standard-deviation band enough to swallow
the very touch the channel is supposed to identify.

Public API:
    compute_rsi(close, period)                       -> Series
    compute_linear_regression_channels(df, ...)      -> DataFrame + channel columns
    slice_dataset_by_channel_regime(df_ch, ...)      -> (long rows, short rows)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CHANNEL_COLUMNS = (
    "channel_slope", "channel_mid", "channel_upper", "channel_lower",
    "channel_pos", "channel_r2", "channel_width", "rsi",
)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI. Uses only past closes, so the value at t is known at t."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # A window with no losses is maximally overbought; with no gains, oversold.
    rsi = rsi.where(loss != 0, 100.0).where(gain != 0, 0.0)
    return rsi.rename("rsi")


def _rolling_fit(y: np.ndarray, window: int) -> tuple[np.ndarray, ...]:
    """OLS of y on 0..window-1 for every trailing window. Returns arrays aligned to
    the LAST bar of each window: slope per bar, fitted value there, residual sd,
    R^2, and the residuals themselves."""
    n = len(y)
    slope = np.full(n, np.nan)
    mid = np.full(n, np.nan)
    sd = np.full(n, np.nan)
    r2 = np.full(n, np.nan)
    resid = np.full((n, window), np.nan)
    if n < window:
        return slope, mid, sd, r2, resid

    panes = np.lib.stride_tricks.sliding_window_view(y, window)   # (n-window+1, window)
    x = np.arange(window, dtype=float)
    xc = x - x.mean()
    sxx = (xc * xc).sum()

    ym = panes.mean(axis=1, keepdims=True)
    b = (panes - ym) @ xc / sxx
    fitted = ym + b[:, None] * xc
    res = panes - fitted
    sse = (res ** 2).sum(axis=1)
    sst = ((panes - ym) ** 2).sum(axis=1)

    at = slice(window - 1, None)
    slope[at] = b
    mid[at] = fitted[:, -1]
    sd[at] = res.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2[at] = np.where(sst > 0, 1.0 - sse / sst, np.nan)
    resid[at] = res
    return slope, mid, sd, r2, resid


def compute_linear_regression_channels(
    df: pd.DataFrame,
    window: int = 30,
    num_std: float = 2.0,
    *,
    price: str = "close",
    log_price: bool = False,
    method: str = "std",
    quantile: float = 0.10,
    require_complete_bars: bool = True,
) -> pd.DataFrame:
    """Attach channel geometry and RSI to `df`.

    log_price fits log(price), which makes the slope a proportional rate and lets
    channels from different price levels be compared; the returned mid/upper/lower
    are converted back to price either way.

    require_complete_bars drops any window that contains a bar built from fewer
    minutes than it should have (see data/build_grids.py). Such a bar has a real
    OHLC but a truncated range, and a channel fitted across one is quietly wrong
    rather than obviously missing.
    """
    if method not in ("std", "quantile"):
        raise ValueError(f"unknown band method: {method!r}")
    out = df.copy()
    raw = out[price].to_numpy(dtype=float)
    y = np.log(raw) if log_price else raw

    slope, mid, sd, r2, resid = _rolling_fit(y, window)

    if method == "std":
        up_off, lo_off = num_std * sd, -num_std * sd
    else:
        valid_residuals = np.isfinite(resid).any(axis=1)
        up_off = np.full(len(resid), np.nan)
        lo_off = np.full(len(resid), np.nan)
        up_off[valid_residuals] = np.nanquantile(
            resid[valid_residuals], 1.0 - quantile, axis=1
        )
        lo_off[valid_residuals] = np.nanquantile(
            resid[valid_residuals], quantile, axis=1
        )

    upper, lower = mid + up_off, mid + lo_off
    if log_price:
        mid, upper, lower = np.exp(mid), np.exp(upper), np.exp(lower)

    if require_complete_bars and "minute_count" in out.columns:
        expected = out["minute_count"].max()
        incomplete = (out["minute_count"] < expected).to_numpy()
        if isinstance(out.index, pd.DatetimeIndex) and len(out.index) > 1:
            expected_delta = pd.to_timedelta(float(expected), unit="min")
            broken_cadence = (out.index.to_series().diff() != expected_delta).to_numpy(copy=True)
            broken_cadence[0] = False
            incomplete = incomplete | broken_cadence
        # a window is contaminated if any bar inside it is incomplete
        touched = (pd.Series(incomplete, index=out.index)
                   .rolling(window, min_periods=1).max().to_numpy() > 0)
        slope = np.where(touched, np.nan, slope)

    span = upper - lower
    with np.errstate(invalid="ignore", divide="ignore"):
        pos = np.where(span > 0, (raw - lower) / span, np.nan)

    out["channel_slope"] = slope
    out["channel_mid"] = mid
    out["channel_upper"] = upper
    out["channel_lower"] = lower
    out["channel_pos"] = pos
    out["channel_r2"] = np.where(np.isnan(slope), np.nan, r2)
    out["channel_width"] = span
    out["rsi"] = compute_rsi(out[price]).to_numpy()
    return out


def label_channel_regime(
    df_ch: pd.DataFrame,
    *,
    min_slope: float = 0.0,
    min_r2: float = 0.0,
    persist_bars: int = 1,
) -> pd.Series:
    """Label each bar 'up', 'down' or 'none'.

    persist_bars requires the raw condition to hold that many bars in a row before
    the regime is considered live. A channel that flickers on for one bar and off
    the next is a fitting artefact, not a market state, and trading it produces
    entries whose context has already vanished by the time the order fills.
    """
    slope = df_ch["channel_slope"]
    r2 = df_ch["channel_r2"]
    ok = r2 >= min_r2
    raw = pd.Series(
        np.where(ok & (slope >= min_slope), "up",
                 np.where(ok & (slope <= -min_slope), "down", "none")),
        index=df_ch.index, dtype=object,
    )
    if persist_bars <= 1:
        return raw.rename("channel_regime")
    same = raw.groupby((raw != raw.shift()).cumsum()).cumcount() + 1
    return raw.where(same >= persist_bars, "none").rename("channel_regime")


def channel_episode_id(regime: pd.Series) -> pd.Series:
    """Consecutive bars sharing a regime form one episode.

    Trades inside a single episode share a market state, so this is the unit that
    cross-validation folds and bootstrap resamples must be drawn over.
    """
    return (regime != regime.shift()).cumsum().rename("channel_episode_id")


def channel_confluence(
    regimes: pd.DataFrame,
    *,
    primary: str,
    min_agree: int = 2,
) -> pd.DataFrame:
    """Count channel windows that agree with the primary window's direction.

    A majority that opposes the primary geometry is not an eligible setup: the
    primary window defines the swing, rails, and episode used by execution.
    """
    if primary not in regimes.columns:
        raise ValueError(f"primary channel {primary!r} is missing")
    if min_agree < 1:
        raise ValueError("min_agree must be >= 1")

    primary_regime = regimes[primary]
    count = regimes.eq(primary_regime, axis="index").sum(axis=1)
    count = count.where(primary_regime.isin(["up", "down"]), 0).astype("int16")
    return pd.DataFrame(
        {
            "channel_confluence_count": count,
            "channel_confluence": (count >= min_agree).astype("int8"),
        },
        index=regimes.index,
    )


def gate_channel_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Return the policy stream with non-confluent candidates set to flat."""
    if "channel_confluence" not in signals:
        raise ValueError("signals missing channel_confluence")
    out = signals.copy()
    out.loc[~out["channel_confluence"].astype(bool), "signal"] = 0
    return out


def slice_dataset_by_channel_regime(
    df_ch: pd.DataFrame,
    min_slope: float = 0.0,
    channel_pos_entry_bound: float = 0.40,
    *,
    min_r2: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into the rows each side would consider: long near the lower rail of a
    rising channel, short near the upper rail of a falling one.

    The two frames are kept separate rather than carrying a side flag because the
    long and short models are trained independently, and a shared frame invites a
    feature computed across both.
    """
    pos = df_ch["channel_pos"]
    slope = df_ch["channel_slope"]
    r2 = df_ch["channel_r2"].fillna(-np.inf)

    long_rows = df_ch[(slope > min_slope) & (r2 >= min_r2)
                      & (pos <= channel_pos_entry_bound)]
    short_rows = df_ch[(slope < -min_slope) & (r2 >= min_r2)
                       & (pos >= 1.0 - channel_pos_entry_bound)]
    return long_rows.copy(), short_rows.copy()
