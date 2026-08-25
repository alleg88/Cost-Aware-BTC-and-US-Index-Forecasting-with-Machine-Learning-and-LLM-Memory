"""Causal first-crossing policies and trade counts for Fast-T2 scores."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SCORE_QUANTILES = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

_ENTRY_REQUIRED = frozenset(
    {
        "window_id",
        "decision_id",
        "decision_time",
        "entry_time",
        "active_end_time",
        "side",
        "channel_episode_id",
        "score",
        "r_net",
        "filled",
        "holding_minutes",
    }
)


@dataclass
class EntryPolicyReplay:
    orders: pd.DataFrame
    metrics: dict[str, float | int]
    input_scores: pd.Series
    signal_ids: tuple[str, ...]
    capacity: int | None


def _validate_entries(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_ENTRY_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"entry ledger missing policy columns: {missing}")
    work = frame.copy()
    for column in ("decision_time", "entry_time", "active_end_time"):
        work[column] = pd.to_datetime(work[column], utc=True, errors="raise")
    if not work["decision_id"].is_unique:
        raise ValueError("decision_id must be unique")
    if (work["active_end_time"] < work["entry_time"]).any():
        raise ValueError("active_end_time cannot precede entry_time")
    return work


def first_crossing_entries(
    scored: pd.DataFrame, threshold: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Choose the first causal threshold crossing within each Fast-T2 window."""
    work = _validate_entries(scored)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    work = work.sort_values(["decision_time", "decision_id"], kind="stable")
    entries: list[dict[str, object]] = []
    actions: list[dict[str, object]] = []
    for window_id, group in work.groupby("window_id", sort=False):
        rows = list(group.sort_values("decision_time", kind="stable").itertuples(index=False))
        for position, row in enumerate(rows):
            score = float(row.score)
            if np.isfinite(score) and score >= threshold:
                entries.append(row._asdict())
                actions.append(
                    {
                        "window_id": window_id,
                        "decision_id": row.decision_id,
                        "decision_time": row.decision_time,
                        "score": score,
                        "action": "ENTER",
                    }
                )
                break
            actions.append(
                {
                    "window_id": window_id,
                    "decision_id": row.decision_id,
                    "decision_time": row.decision_time,
                    "score": score,
                    "action": "SKIP" if position == len(rows) - 1 else "WAIT",
                }
            )
    entry_frame = pd.DataFrame(entries, columns=work.columns)
    if not entry_frame.empty:
        entry_frame = entry_frame.sort_values(
            ["entry_time", "score", "decision_id"],
            ascending=[True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        if entry_frame["window_id"].duplicated().any():
            raise AssertionError("a window entered more than once")
    action_frame = pd.DataFrame(
        actions,
        columns=["window_id", "decision_id", "decision_time", "score", "action"],
    )
    return entry_frame, action_frame


def _episode_bootstrap(
    filled: pd.DataFrame, *, replicates: int, seed: int = 42
) -> tuple[float, float]:
    if filled.empty or replicates < 1:
        return np.nan, np.nan
    grouped = filled.groupby("channel_episode_id", sort=False)["r_net"].agg(
        ["sum", "count"]
    )
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarise_trade_activity(
    orders: pd.DataFrame,
    *,
    evaluation_days: float,
    bootstrap_reps: int = 2_000,
    capacity_skips: int = 0,
) -> dict[str, float | int]:
    """Summarise filled trades while retaining submitted/cancelled counts."""
    if not np.isfinite(evaluation_days) or evaluation_days <= 0:
        raise ValueError("evaluation_days must be positive and finite")
    calendar_days = int(round(float(evaluation_days)))
    if not np.isclose(calendar_days, evaluation_days):
        raise ValueError("evaluation_days must be a whole number of days")
    work = orders.copy()
    if work.empty:
        filled = work
    else:
        work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True)
        work["active_end_time"] = pd.to_datetime(work["active_end_time"], utc=True)
        filled = work[work["filled"].astype(bool)].copy()
    daily_observed = (
        filled.assign(day=filled["entry_time"].dt.floor("D"))
        .groupby("day")
        .size()
        if not filled.empty
        else pd.Series(dtype=int)
    )
    daily = np.concatenate(
        [
            daily_observed.to_numpy(dtype=int),
            np.zeros(max(0, calendar_days - len(daily_observed)), dtype=int),
        ]
    )
    if len(daily) > calendar_days:
        raise ValueError("observed trade days exceed evaluation_days")
    low, high = _episode_bootstrap(filled, replicates=bootstrap_reps)
    if filled.empty:
        mean_r = win_rate = mean_hold = np.nan
        total_r = 0.0
        max_drawdown = 0.0
    else:
        mean_r = float(filled["r_net"].mean())
        total_r = float(filled["r_net"].sum())
        win_rate = float(filled["r_net"].gt(0.0).mean())
        mean_hold = float(filled["holding_minutes"].mean())
        ordered = filled.sort_values("active_end_time", kind="stable")
        equity = ordered["r_net"].cumsum().to_numpy(dtype=float)
        peak = np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:]
        max_drawdown = float(np.min(equity - peak))
    concurrent = (
        work["concurrent_at_entry"].to_numpy(dtype=float)
        if "concurrent_at_entry" in work
        else np.array([], dtype=float)
    )
    return {
        "submitted_orders": int(len(work)),
        "filled_trades": int(len(filled)),
        "trades_per_day": float(len(filled) / calendar_days),
        "median_daily_trades": float(np.median(daily)),
        "zero_trade_days": int((daily == 0).sum()),
        "days_one": int((daily == 1).sum()),
        "days_two": int((daily == 2).sum()),
        "days_three_plus": int((daily >= 3).sum()),
        "long_trades": int((filled.get("side", pd.Series(dtype=str)) == "long").sum()),
        "short_trades": int((filled.get("side", pd.Series(dtype=str)) == "short").sum()),
        "mean_concurrent": float(concurrent.mean()) if len(concurrent) else 0.0,
        "max_concurrent": int(concurrent.max()) if len(concurrent) else 0,
        "capacity_skips": int(capacity_skips),
        "mean_net_r": mean_r,
        "total_net_r": total_r,
        "win_rate": win_rate,
        "mean_holding_minutes": mean_hold,
        "max_drawdown_r": max_drawdown,
        "bootstrap_low": low,
        "bootstrap_high": high,
    }


def replay_entry_capacity(
    entries: pd.DataFrame,
    *,
    capacity: int | None,
    evaluation_days: float,
    bootstrap_reps: int = 2_000,
) -> EntryPolicyReplay:
    """Replay one fixed signal ledger without changing any model score."""
    if capacity is not None and capacity < 1:
        raise ValueError("capacity must be positive or None")
    work = _validate_entries(entries)
    input_scores = entries["score"].copy()
    signal_ids = tuple(entries["decision_id"].astype(str))
    work = work.drop(columns=["concurrent_at_entry"], errors="ignore")
    work = work.sort_values(
        ["entry_time", "score", "decision_id"],
        ascending=[True, False, True],
        kind="stable",
    )
    active: list[pd.Timestamp] = []
    accepted: list[dict[str, object]] = []
    capacity_skips = 0
    for row in work.itertuples(index=False):
        active = [end for end in active if end > row.entry_time]
        if capacity is not None and len(active) >= capacity:
            capacity_skips += 1
            continue
        record = row._asdict()
        if bool(row.filled) and row.active_end_time > row.entry_time:
            active.append(row.active_end_time)
        record["concurrent_at_entry"] = len(active)
        accepted.append(record)
    columns = [*work.columns, "concurrent_at_entry"]
    orders = pd.DataFrame(accepted, columns=columns)
    metrics = summarise_trade_activity(
        orders,
        evaluation_days=evaluation_days,
        bootstrap_reps=bootstrap_reps,
        capacity_skips=capacity_skips,
    )
    return EntryPolicyReplay(
        orders=orders.reset_index(drop=True),
        metrics=metrics,
        input_scores=input_scores,
        signal_ids=signal_ids,
        capacity=capacity,
    )


def select_inner_quantile(
    inner_scored: pd.DataFrame,
    *,
    evaluation_days: float,
) -> tuple[float, pd.DataFrame]:
    """Choose a score quantile on an inner chronological validation ledger."""
    finite = pd.to_numeric(inner_scored["score"], errors="coerce").dropna()
    if finite.empty:
        raise ValueError("inner threshold selection needs finite scores")
    rows: list[dict[str, object]] = []
    for quantile in SCORE_QUANTILES:
        threshold = float(finite.quantile(quantile))
        entries, actions = first_crossing_entries(inner_scored, threshold)
        replay = replay_entry_capacity(
            entries,
            capacity=None,
            evaluation_days=evaluation_days,
            bootstrap_reps=0,
        )
        rows.append(
            {
                "quantile": float(quantile),
                "threshold": threshold,
                "wait_actions": int(actions["action"].eq("WAIT").sum()),
                "enter_actions": int(actions["action"].eq("ENTER").sum()),
                "skip_actions": int(actions["action"].eq("SKIP").sum()),
                **replay.metrics,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[
        table["trades_per_day"].ge(1.0)
        & table["long_trades"].gt(0)
        & table["short_trades"].gt(0)
    ]
    source = eligible if not eligible.empty else table
    ranked = source.assign(
        _mean=source["mean_net_r"].fillna(-np.inf),
        _total=source["total_net_r"].fillna(-np.inf),
    ).sort_values(
        ["_mean", "_total", "quantile"],
        ascending=[False, False, True],
        kind="stable",
    )
    return float(ranked.iloc[0]["quantile"]), table
