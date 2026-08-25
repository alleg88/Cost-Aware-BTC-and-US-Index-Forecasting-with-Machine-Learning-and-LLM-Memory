"""Turn clean M15 OHLCV into a model feature table + a 3-class direction label.

Leak-free by construction: every feature at bar t uses only information known at the
close of bar t (the bar's own OHLCV and earlier bars). The label looks one bar AHEAD
(that is the prediction target), and the final row is dropped because it has no future.

Indicators are computed by hand (no pandas-ta — it has no Python 3.11 / numpy 2 build),
which also keeps the pipeline reproducible.

Class encoding: down=0, flat=1, up=2.
Design notes: papers/wiki/design/evaluation-and-splits.md, news-scoring-howto.md
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "r1", "r5", "r20",
    "vol_10", "vol_20", "vol_60",
    "hl_range", "co_range",
    "rsi_14",
    "volume", "vol_z",
    "hour", "dayofweek",
]

# Microstructure block, part of the DEFAULT feature set wherever the source
# carries aggressor-side volume + trade count (Binance klines do; Dukascopy
# index CFDs do not — there add_features simply skips it). Kept as a separate
# list so price-only / no-order-flow ablations can still exclude it explicitly.
ORDERFLOW_FEATURE_COLS = ["ofi", "ofi_z20", "ofi_mom5", "trade_intensity_z"]

# Positioning block (futures funding rate + open interest + long/short ratios,
# see data/build_positioning.py). Computed only when the raw columns have been
# joined onto the bars — the block is an explicit ablation, not a default,
# because it exists for BTC perps only. Design note:
# papers/wiki/design/positioning-features-funding-oi.md
POSITIONING_SOURCE_COLS = ["funding_rate", "sum_open_interest",
                           "toptrader_ls", "taker_ls"]
POSITIONING_FEATURE_COLS = ["funding_rate", "funding_z", "oi_chg_1h",
                            "oi_chg_4h", "oi_z", "toptrader_ls_z", "taker_ls_z"]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append the price-only feature columns (all past/current-bar information)."""
    out = df.copy()
    logret = np.log(out["close"]).diff()

    out["r1"] = logret
    out["r5"] = np.log(out["close"]).diff(5)
    out["r20"] = np.log(out["close"]).diff(20)

    out["vol_10"] = logret.rolling(10).std()
    out["vol_20"] = logret.rolling(20).std()
    out["vol_60"] = logret.rolling(60).std()

    out["hl_range"] = (out["high"] - out["low"]) / out["close"]
    out["co_range"] = (out["close"] - out["open"]) / out["open"]

    out["rsi_14"] = _rsi(out["close"], 14)

    vol_mean = out["volume"].rolling(60).mean()
    vol_std = out["volume"].rolling(60).std()
    out["vol_z"] = (out["volume"] - vol_mean) / vol_std.replace(0.0, np.nan)

    # Order-flow features (Binance kline aggressor side). Each uses only the
    # bar's own completed values, so it is known at the bar close — leak-free
    # for predicting the next bar, exactly like volume. Skipped when the source
    # lacks taker volume / trade count (e.g. index CFDs).
    if {"taker_buy_base", "volume", "count"}.issubset(out.columns):
        vol = out["volume"].replace(0.0, np.nan)
        # signed order-flow imbalance: -1 = all aggressive sells, +1 = all buys
        ofi = 2.0 * (out["taker_buy_base"] / vol) - 1.0
        out["ofi"] = ofi
        out["ofi_z20"] = (ofi - ofi.rolling(20).mean()) / ofi.rolling(20).std().replace(0.0, np.nan)
        out["ofi_mom5"] = ofi.rolling(5).mean()
        cnt = out["count"].astype(float)
        cnt_std = cnt.rolling(60).std().replace(0.0, np.nan)
        out["trade_intensity_z"] = (cnt - cnt.rolling(60).mean()) / cnt_std

    # Positioning features (perp funding + open interest + long/short ratios),
    # computed only when data/build_positioning.py columns were joined onto the
    # bars. Raw columns are last-known-at-bar-close by construction (leak-free).
    if set(POSITIONING_SOURCE_COLS).issubset(out.columns):
        fr = out["funding_rate"].astype(float)
        # z over ~1 week of bars (672 = 7d x 96); funding steps every 8h
        out["funding_z"] = (fr - fr.rolling(672).mean()) / \
            fr.rolling(672).std().replace(0.0, np.nan)
        oi = np.log(out["sum_open_interest"].astype(float))
        out["oi_chg_1h"] = oi.diff(4)
        out["oi_chg_4h"] = oi.diff(16)
        out["oi_z"] = (oi - oi.rolling(96).mean()) / \
            oi.rolling(96).std().replace(0.0, np.nan)     # z over ~1 day
        for src, dst in [("toptrader_ls", "toptrader_ls_z"),
                         ("taker_ls", "taker_ls_z")]:
            v = out[src].astype(float)
            out[dst] = (v - v.rolling(96).mean()) / \
                v.rolling(96).std().replace(0.0, np.nan)

    out["hour"] = out.index.hour
    out["dayofweek"] = out.index.dayofweek
    return out


def make_label(df: pd.DataFrame, threshold_bps: float = 5.0, horizon: int = 1) -> pd.Series:
    """3-class dead-zone label on the next-bar (horizon) forward return.

    up (2)   if fwd_ret >= +threshold
    down (0) if fwd_ret <= -threshold
    flat (1) otherwise
    """
    thr = threshold_bps / 1e4  # bps -> fraction
    fwd_ret = df["close"].shift(-horizon) / df["close"] - 1.0
    label = pd.Series(1, index=df.index, dtype="int64")  # default flat
    label[fwd_ret >= thr] = 2
    label[fwd_ret <= -thr] = 0
    label[fwd_ret.isna()] = -1  # mark rows with no future (dropped downstream)
    return label.rename("label")


def make_barrier_label(
    df: pd.DataFrame, *, up_bps: float, down_bps: float, max_hold: int
) -> pd.Series:
    """3-class triple-barrier label: which barrier does price touch FIRST?

    Barriers sit at close[t] * (1 + up_bps) and close[t] * (1 - down_bps); the
    next `max_hold` bars' highs/lows are scanned in order:

    up (2)   if the up barrier is touched first (via a later bar's high)
    down (0) if the down barrier is touched first (via a later bar's low)
    flat (1) if neither barrier is touched within max_hold bars (time-out)

    Asymmetric by construction (up_bps and down_bps are independent), matching
    the asymmetric behaviour of up and down moves. When both barriers fall
    inside the same bar the DOWN barrier wins — the same conservative
    stop-first convention as the bracket trade simulator.

    Rows whose window is truncated by the end of the data can only claim a
    barrier touch, never a time-out; undecided truncated rows get -1 (dropped
    downstream). NOTE: the label looks up to max_hold bars ahead, so CV embargo
    and walk-forward train-tail trimming must cover max_hold bars.
    """
    if max_hold < 1:
        raise ValueError("max_hold must be >= 1")
    n = len(df)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    up_px = close * (1.0 + up_bps / 1e4)
    down_px = close * (1.0 - down_bps / 1e4)

    label = np.full(n, -2, dtype=np.int64)  # -2 = not yet decided
    pad = np.full(1, np.nan)
    for h in range(1, max_hold + 1):
        hi = np.concatenate([high[h:], np.repeat(pad, h)])
        lo = np.concatenate([low[h:], np.repeat(pad, h)])
        undecided = label == -2
        label[undecided & (lo <= down_px)] = 0   # down checked first: tie -> down
        undecided = label == -2
        label[undecided & (hi >= up_px)] = 2
    label[label == -2] = 1                       # neither barrier: time-out flat

    # A time-out claim needs the full window; truncated rows can't make it.
    if max_hold < n:
        truncated = np.arange(n) >= n - max_hold
        label[truncated & (label == 1)] = -1
    else:
        label[label == 1] = -1
    return pd.Series(label, index=df.index, name="label")


def default_feature_cols(feat: pd.DataFrame) -> list[str]:
    """The default feature list: price block + order-flow block where available."""
    return FEATURE_COLS + [c for c in ORDERFLOW_FEATURE_COLS if c in feat.columns]


def build_dataset(
    df: pd.DataFrame, threshold_bps: float = 5.0, horizon: int = 1,
    orderflow: bool = True, positioning: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) aligned and free of warm-up NaNs and the label-less tail.

    Order-flow features are part of the default set (where the source provides
    the aggressor-side columns); pass orderflow=False for a price-only ablation.
    positioning=True additionally includes the funding/OI block — requires the
    build_positioning.py columns to be joined onto df first (BTC perps only).
    """
    feat = add_features(df)
    y = make_label(feat, threshold_bps=threshold_bps, horizon=horizon)
    cols = default_feature_cols(feat) if orderflow else list(FEATURE_COLS)
    if positioning:
        missing = [c for c in POSITIONING_FEATURE_COLS if c not in feat.columns]
        if missing:
            raise ValueError(f"positioning=True but columns missing: {missing} — "
                             "join data/build_positioning.py output onto df first")
        cols = cols + POSITIONING_FEATURE_COLS
    X = feat[cols]

    valid = X.notna().all(axis=1) & (y != -1)
    return X[valid], y[valid]


def build_dataset_barrier(
    df: pd.DataFrame, *, up_bps: float, down_bps: float, max_hold: int,
    orderflow: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """build_dataset twin for the triple-barrier label (same features)."""
    feat = add_features(df)
    y = make_barrier_label(feat, up_bps=up_bps, down_bps=down_bps, max_hold=max_hold)
    X = feat[default_feature_cols(feat) if orderflow else FEATURE_COLS]

    valid = X.notna().all(axis=1) & (y != -1)
    return X[valid], y[valid]
