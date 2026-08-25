"""Harmonised, zero-filled daily economics for the final Q2 lockbox."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


Q2_START = pd.Timestamp("2026-04-01T00:00:00Z")
Q2_END = pd.Timestamp("2026-07-01T00:00:00Z")
DAILY_ANNUALISATION = 365
PRIMARY_ESTIMAND = "btc_union_minus_lstm_total_net"


@dataclass(frozen=True)
class BootstrapResult:
    estimand: str
    point_estimate: float
    lower_95: float
    upper_95: float
    reps: int
    seed: int
    block_count: int
    primary_confirmatory_support: bool | None


def _utc(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _daily_grid(
    start: str | pd.Timestamp, end: str | pd.Timestamp
) -> pd.DatetimeIndex:
    start_utc = _utc(start, label="start")
    end_utc = _utc(end, label="end")
    if start_utc != start_utc.normalize() or end_utc != end_utc.normalize():
        raise ValueError("daily metric boundaries must be UTC midnights")
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    return pd.date_range(start_utc, end_utc, inclusive="left", freq="D", name="date")


def _ledger_time(ledger: pd.DataFrame) -> pd.Series:
    for column in ("exit_time", "timestamp"):
        if column in ledger:
            values = pd.to_datetime(ledger[column], utc=True, errors="raise")
            return pd.Series(values, index=ledger.index, name=column)
    if isinstance(ledger.index, pd.DatetimeIndex):
        values = pd.to_datetime(ledger.index, utc=True, errors="raise")
        return pd.Series(values, index=ledger.index, name="timestamp")
    raise ValueError("ledger requires exit_time, timestamp or a DatetimeIndex")


def _return_columns(ledger: pd.DataFrame) -> pd.DataFrame:
    required = {"gross_return", "net_return"}
    if not required.issubset(ledger.columns):
        raise ValueError(f"ledger misses return columns: {sorted(required - set(ledger))}")
    current = ledger.copy()
    current["gross_return"] = pd.to_numeric(
        current["gross_return"], errors="raise"
    ).astype(float)
    current["net_return"] = pd.to_numeric(
        current["net_return"], errors="raise"
    ).astype(float)
    if "cost_return" in current:
        current["cost_return"] = pd.to_numeric(
            current["cost_return"], errors="raise"
        ).astype(float)
    else:
        current["cost_return"] = current["gross_return"] - current["net_return"]
    values = current[["gross_return", "cost_return", "net_return"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("ledger returns must be finite")
    if (current["cost_return"] < -1e-12).any():
        raise ValueError("cost_return must be a non-negative deduction")
    if not np.allclose(
        current["gross_return"] - current["cost_return"],
        current["net_return"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("gross - cost does not reconcile to net")
    return current


def daily_net_series(
    ledger: pd.DataFrame,
    *,
    start: str | pd.Timestamp = Q2_START,
    end: str | pd.Timestamp = Q2_END,
) -> pd.Series:
    """Aggregate trade net returns onto every UTC calendar day in the interval."""
    grid = _daily_grid(start, end)
    if ledger.empty:
        return pd.Series(0.0, index=grid, name="net_return")
    current = _return_columns(ledger)
    timestamps = _ledger_time(current)
    start_utc, end_utc = grid[0], grid[-1] + pd.Timedelta(days=1)
    if timestamps.lt(start_utc).any() or timestamps.ge(end_utc).any():
        raise ValueError("ledger contains a row outside the half-open interval")
    day = timestamps.dt.normalize()
    daily = current["net_return"].groupby(day).sum().reindex(grid, fill_value=0.0)
    daily.index.name = "date"
    daily.name = "net_return"
    return daily.astype(float)


def double_cost_ledger(ledger: pd.DataFrame) -> pd.DataFrame:
    """Return the same trades with exactly twice the registered cost deduction."""
    stressed = _return_columns(ledger)
    stressed["cost_return"] = 2.0 * stressed["cost_return"]
    stressed["net_return"] = stressed["gross_return"] - stressed["cost_return"]
    return stressed


def _daily_ratios(daily: pd.Series) -> tuple[float, float]:
    values = daily.astype(float)
    standard_deviation = float(values.std(ddof=1))
    mean = float(values.mean())
    sharpe = (
        mean / standard_deviation * np.sqrt(DAILY_ANNUALISATION)
        if standard_deviation > 0.0 and np.isfinite(standard_deviation)
        else 0.0
    )
    downside = float(np.sqrt(np.mean(np.minimum(values.to_numpy(), 0.0) ** 2)))
    sortino = (
        mean / downside * np.sqrt(DAILY_ANNUALISATION)
        if downside > 0.0 and np.isfinite(downside)
        else 0.0
    )
    return float(sharpe), float(sortino)


def _maximum_drawdown(daily: pd.Series) -> float:
    cumulative = daily.astype(float).cumsum()
    with_origin = pd.concat(
        [pd.Series([0.0], index=[daily.index[0] - pd.Timedelta(days=1)]), cumulative]
    )
    return float((with_origin.cummax() - with_origin).max())


def summarise_candidate(
    ledger: pd.DataFrame,
    *,
    candidate_id: str,
    stream: str,
    start: str | pd.Timestamp = Q2_START,
    end: str | pd.Timestamp = Q2_END,
) -> dict[str, object]:
    """Return finite headline and double-cost metrics for one frozen candidate."""
    current = _return_columns(ledger)
    if "side" not in current:
        raise ValueError("ledger requires side")
    side = pd.to_numeric(current["side"], errors="raise").astype(int)
    if not side.isin((-1, 1)).all():
        raise ValueError("trade side must be -1 or +1")
    daily = daily_net_series(current, start=start, end=end)
    stressed = double_cost_ledger(current)
    stressed_daily = daily_net_series(stressed, start=start, end=end)
    sharpe, sortino = _daily_ratios(daily)
    trades = int(len(current))
    net = float(current["net_return"].sum())
    monthly = daily.resample("MS").sum()
    return {
        "stream": str(stream),
        "candidate_id": str(candidate_id),
        "gross_return": float(current["gross_return"].sum()),
        "cost_return": float(current["cost_return"].sum()),
        "net_return": net,
        "trades": trades,
        "trades_per_day": float(trades / len(daily)),
        "long_trades": int(side.eq(1).sum()),
        "short_trades": int(side.eq(-1).sum()),
        "net_bps_per_trade": float(net * 10_000.0 / trades) if trades else 0.0,
        "win_rate": float(current["net_return"].gt(0.0).mean()) if trades else 0.0,
        "daily_sharpe": sharpe,
        "daily_sortino": sortino,
        "max_drawdown": _maximum_drawdown(daily),
        "positive_months": int(monthly.gt(0.0).sum()),
        "stress_2x_net_return": float(stressed_daily.sum()),
        "stress_2x_daily_sharpe": _daily_ratios(stressed_daily)[0],
        "stress_2x_daily_sortino": _daily_ratios(stressed_daily)[1],
        "stress_2x_max_drawdown": _maximum_drawdown(stressed_daily),
    }


def _require_complete_daily(
    values: pd.Series,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    label: str,
) -> pd.Series:
    if not isinstance(values.index, pd.DatetimeIndex):
        raise ValueError(f"{label} requires a DatetimeIndex")
    current = values.astype(float).copy()
    if current.index.tz is None:
        raise ValueError(f"{label} index must be timezone-aware")
    current.index = current.index.tz_convert("UTC").normalize()
    expected = _daily_grid(start, end)
    if not current.index.equals(expected):
        raise ValueError(f"{label} must contain the exact complete daily grid")
    if not np.isfinite(current.to_numpy()).all():
        raise ValueError(f"{label} values must be finite")
    return current


def monday_week_blocks(
    daily_delta: pd.Series,
    *,
    start: str | pd.Timestamp = Q2_START,
    end: str | pd.Timestamp = Q2_END,
) -> tuple[pd.Series, ...]:
    """Return Monday-Sunday blocks, zero-padding only outside interval bounds."""
    current = _require_complete_daily(
        daily_delta, start=start, end=end, label="daily delta"
    )
    first = current.index[0] - pd.Timedelta(days=current.index[0].weekday())
    last = current.index[-1] + pd.Timedelta(days=6 - current.index[-1].weekday())
    padded_index = pd.date_range(first, last, freq="D", tz="UTC", name="date")
    padded = current.reindex(padded_index, fill_value=0.0)
    if len(padded) % 7:
        raise AssertionError("padded Monday-Sunday grid is not divisible by seven")
    return tuple(padded.iloc[offset : offset + 7] for offset in range(0, len(padded), 7))


def paired_weekly_bootstrap(
    policy_daily: pd.Series,
    control_daily: pd.Series,
    *,
    reps: int = 5_000,
    seed: int = 42,
    estimand: str = PRIMARY_ESTIMAND,
    start: str | pd.Timestamp = Q2_START,
    end: str | pd.Timestamp = Q2_END,
) -> BootstrapResult:
    """Bootstrap fixed Monday-Sunday paired-delta block totals with replacement."""
    if int(reps) <= 0:
        raise ValueError("reps must be positive")
    policy = _require_complete_daily(
        policy_daily, start=start, end=end, label="policy daily series"
    )
    control = _require_complete_daily(
        control_daily, start=start, end=end, label="control daily series"
    )
    delta = policy - control
    blocks = monday_week_blocks(delta, start=start, end=end)
    block_totals = np.asarray([block.sum() for block in blocks], dtype=float)
    generator = np.random.default_rng(int(seed))
    sampled = generator.choice(
        block_totals,
        size=(int(reps), len(block_totals)),
        replace=True,
    ).sum(axis=1)
    lower, upper = np.quantile(sampled, [0.025, 0.975])
    support = bool(lower > 0.0) if estimand == PRIMARY_ESTIMAND else None
    return BootstrapResult(
        estimand=str(estimand),
        point_estimate=float(delta.sum()),
        lower_95=float(lower),
        upper_95=float(upper),
        reps=int(reps),
        seed=int(seed),
        block_count=int(len(block_totals)),
        primary_confirmatory_support=support,
    )


__all__ = [
    "BootstrapResult",
    "DAILY_ANNUALISATION",
    "PRIMARY_ESTIMAND",
    "Q2_END",
    "Q2_START",
    "daily_net_series",
    "double_cost_ledger",
    "monday_week_blocks",
    "paired_weekly_bootstrap",
    "summarise_candidate",
]
