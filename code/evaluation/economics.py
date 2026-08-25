"""Net-of-cost strategy economics for walk-forward prediction caches.

Positions follow the classification rule up -> long, down -> short, flat -> out,
optionally gated by a confidence threshold tau (top-class probability below tau
means stay out). Costs are charged per side on position changes. Returns are
simple per-bar net returns; cumulative curves are additive (not compounded),
matching the rest of the evaluation stack.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_YEAR_M15 = 35_040       # 4 bars/hour x 24h x 365d
POSITION = {0: -1.0, 1: 0.0, 2: 1.0}


def positions_from_predictions(pred: pd.Series, conf: pd.Series | None = None,
                               tau: float = 0.0) -> pd.Series:
    """Map class predictions to positions; below-threshold confidence stays out."""
    pos = pred.astype(int).map(POSITION).fillna(0.0)
    if conf is not None and tau > 0.0:
        pos = pos.where(conf.astype(float) >= tau, 0.0)
    return pos


def strategy_returns(pred: pd.Series, forward_return: pd.Series, fee_bps: float,
                     conf: pd.Series | None = None, tau: float = 0.0) -> pd.Series:
    """Per-bar net returns of the long/short/out rule under a confidence gate."""
    pos = positions_from_predictions(pred, conf, tau)
    turnover = (pos - pos.shift(1, fill_value=0.0)).abs()
    return pos * forward_return.astype(float).fillna(0.0) - turnover * (float(fee_bps) / 10_000.0)


def confidence_weights(conf: pd.Series, tau: float, cap: float = 1.0,
                       floor: float = 0.0) -> pd.Series:
    """Scale a unit position by how far confidence clears the gate.

    Because the ensemble probabilities are isotonic-calibrated, `conf` is an
    honest frequency, so a linear ramp from the gate `tau` (weight `floor`) to
    full confidence 1.0 (weight `cap`) puts more capital on the trades most
    likely to be right. Below `tau` the weight is 0 (the gate still applies).
    Weights are clipped to [floor, cap]; tau=1 degenerates to a flat `cap`.
    """
    c = conf.astype(float)
    span = max(1.0 - float(tau), 1e-9)
    w = floor + (cap - floor) * (c - float(tau)) / span
    return w.clip(lower=floor, upper=cap)


def strategy_returns_sized(pred: pd.Series, forward_return: pd.Series, fee_bps: float,
                           conf: pd.Series, tau: float = 0.0, cap: float = 1.0,
                           floor: float = 0.0) -> pd.Series:
    """Per-bar net returns with confidence-weighted position sizing.

    Same long/short/out signal as `strategy_returns`, but each position is
    scaled by `confidence_weights` instead of taking unit size. Costs are
    charged on the change in the *sized* position, so scaling up or down a held
    position also pays turnover — the honest accounting of a resized book.
    """
    signal = positions_from_predictions(pred, conf, tau)
    pos = signal * confidence_weights(conf, tau, cap=cap, floor=floor)
    turnover = (pos - pos.shift(1, fill_value=0.0)).abs()
    return pos * forward_return.astype(float).fillna(0.0) - turnover * (float(fee_bps) / 10_000.0)


def max_drawdown(returns: pd.Series) -> float:
    """Largest peak-to-trough fall of the additive cumulative net-return curve."""
    curve = returns.fillna(0.0).cumsum()
    return float((curve.cummax() - curve).max())


def economics_summary(returns: pd.Series, pred: pd.Series | None = None,
                      conf: pd.Series | None = None, tau: float = 0.0,
                      bars_per_year: int = BARS_PER_YEAR_M15) -> dict[str, float]:
    """Headline economic metrics for one net-return series."""
    r = returns.dropna().astype(float)
    std = r.std(ddof=1)
    # Sortino downside deviation: RMS of min(r, 0) over ALL observations (the
    # standard definition). The negative-subset std variant explodes for sparse
    # trade series where most bars are 0.
    downside = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) if len(r) else 0.0
    out = {
        "net_return_sum": float(r.sum()),
        "net_return_mean": float(r.mean()) if len(r) else 0.0,
        "sharpe": float(r.mean() / std * np.sqrt(bars_per_year)) if std and np.isfinite(std) else 0.0,
        "sortino": float(r.mean() / downside * np.sqrt(bars_per_year))
        if downside and np.isfinite(downside) else 0.0,
        "max_drawdown": max_drawdown(r),
        "tau": float(tau),
    }
    if pred is not None:
        pos = positions_from_predictions(pred, conf, tau)
        turnover = (pos - pos.shift(1, fill_value=0.0)).abs()
        out["trade_count"] = int((turnover > 0).sum())
        out["exposure"] = float((pos != 0).mean())
    return out


def sweep_tau(pred: pd.Series, conf: pd.Series, forward_return: pd.Series,
              fee_bps: float, taus: tuple[float, ...],
              bars_per_year: int = BARS_PER_YEAR_M15) -> pd.DataFrame:
    """Economics at each confidence threshold (tau=0 disables the gate)."""
    rows = []
    for tau in taus:
        returns = strategy_returns(pred, forward_return, fee_bps, conf, tau)
        rows.append(economics_summary(returns, pred, conf, tau, bars_per_year))
    return pd.DataFrame(rows)


def diebold_mariano(returns_a: pd.Series, returns_b: pd.Series,
                    lag: int = 1) -> dict[str, float]:
    """Diebold-Mariano test on the net-return differential (b minus a).

    Uses a Newey-West variance with `lag` autocovariance terms and the standard
    normal reference. Positive statistic = b outperforms a.
    """
    joined = pd.concat([returns_a, returns_b], axis=1, join="inner").dropna()
    d = (joined.iloc[:, 1] - joined.iloc[:, 0]).to_numpy(dtype=float)
    n = len(d)
    if n < 10:
        raise ValueError("need at least 10 overlapping observations for a DM test")
    d_bar = d.mean()
    centered = d - d_bar
    variance = centered @ centered / n
    for k in range(1, lag + 1):
        gamma = centered[k:] @ centered[:-k] / n
        variance += 2.0 * (1.0 - k / (lag + 1)) * gamma
    if variance <= 0:
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_diff": float(d_bar), "n": n}
    from scipy.stats import norm

    stat = d_bar / np.sqrt(variance / n)
    return {
        "dm_stat": float(stat),
        "p_value": float(2.0 * (1.0 - norm.cdf(abs(stat)))),
        "mean_diff": float(d_bar),
        "n": n,
    }
