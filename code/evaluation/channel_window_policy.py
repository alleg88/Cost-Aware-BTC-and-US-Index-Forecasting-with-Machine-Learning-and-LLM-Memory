"""Threshold selection and post-model portfolio replay for Notebook B."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


SELECTION_QUANTILES = (0.00, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
INHERITED_CHANNEL_TRIALS = 100


@dataclass
class PolicyReplay:
    orders: pd.DataFrame
    metrics: dict[str, float | int]
    input_scores: pd.Series
    max_concurrent: int


_POLICY_REQUIRED = frozenset(
    {
        "decision_time",
        "active_end_time",
        "side",
        "score",
        "r_net",
        "filled",
        "window_id",
        "channel_episode_id",
    }
)


def _metrics(
    orders: pd.DataFrame,
    *,
    threshold_crossings: int,
    bucket_skips: int,
    capacity_skips: int,
    concurrency_trace: list[int],
    evaluation_days: float | None,
) -> dict[str, float | int]:
    if orders.empty:
        return {
            "submitted_orders": 0,
            "filled_trades": 0,
            "total_net_r": 0.0,
            "mean_r_net": np.nan,
            "fill_rate": np.nan,
            "trades_per_day": 0.0,
            "long_fills": 0,
            "short_fills": 0,
            "episodes": 0,
            "threshold_crossings": threshold_crossings,
            "bucket_skips": bucket_skips,
            "capacity_skips": capacity_skips,
            "mean_concurrent": 0.0,
            "max_concurrent": 0,
        }
    filled = orders["filled"].astype(bool)
    filled_orders = orders[filled]
    if evaluation_days is None:
        first_day = orders["decision_time"].min().normalize()
        last_day = orders["decision_time"].max().normalize()
        calendar_days = max(
            1.0, float((last_day - first_day) / pd.Timedelta(days=1)) + 1.0
        )
    else:
        calendar_days = float(evaluation_days)
        if not np.isfinite(calendar_days) or calendar_days <= 0:
            raise ValueError("evaluation_days must be positive and finite")
    return {
        "submitted_orders": int(len(orders)),
        "filled_trades": int(filled.sum()),
        "total_net_r": float(orders["r_net"].sum()),
        "mean_r_net": float(orders["r_net"].mean()),
        "fill_rate": float(filled.mean()),
        "trades_per_day": float(filled.sum() / calendar_days),
        "long_fills": int((filled_orders["side"] == "long").sum()),
        "short_fills": int((filled_orders["side"] == "short").sum()),
        "episodes": int(orders["channel_episode_id"].nunique()),
        "threshold_crossings": int(threshold_crossings),
        "bucket_skips": int(bucket_skips),
        "capacity_skips": int(capacity_skips),
        "mean_concurrent": (
            float(np.mean(concurrency_trace)) if concurrency_trace else 0.0
        ),
        "max_concurrent": int(max(concurrency_trace, default=0)),
    }


def replay_capacity(
    scored: pd.DataFrame,
    threshold: float,
    capacity: int | None,
    *,
    evaluation_days: float | None = None,
) -> PolicyReplay:
    """Replay score crossings chronologically without changing any model score."""
    missing = sorted(_POLICY_REQUIRED.difference(scored.columns))
    if missing:
        raise ValueError(f"scored ledger missing policy columns: {missing}")
    if capacity is not None and capacity < 1:
        raise ValueError("capacity must be positive or None")
    input_scores = scored["score"].copy()
    work = scored.copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True)
    work["active_end_time"] = pd.to_datetime(work["active_end_time"], utc=True)
    if work[["decision_time", "active_end_time"]].isna().any().any():
        raise ValueError("policy times cannot be missing")
    if (work["active_end_time"] < work["decision_time"]).any():
        raise ValueError("active order interval ends before its decision")
    work["_input_order"] = np.arange(len(work), dtype=np.int64)
    work = work.sort_values(["decision_time", "_input_order"], kind="stable")

    active: list[pd.Timestamp] = []
    used_buckets: set[pd.Timestamp] = set()
    submitted: list[dict[str, object]] = []
    concurrency_trace: list[int] = []
    threshold_crossings = bucket_skips = capacity_skips = 0
    max_concurrent = 0

    for row in work.itertuples(index=False):
        score = float(row.score)
        if not np.isfinite(score) or score < threshold:
            continue
        threshold_crossings += 1
        active = [end for end in active if end > row.decision_time]
        bucket = row.decision_time.floor("5min")
        if bucket in used_buckets:
            bucket_skips += 1
            continue
        used_buckets.add(bucket)
        if capacity is not None and len(active) >= capacity:
            capacity_skips += 1
            continue

        record = row._asdict()
        record.pop("_input_order", None)
        record["layer_no"] = len(active) + 1
        record["capacity_blocked"] = False
        record["submitted"] = True
        submitted.append(record)
        active.append(row.active_end_time)
        max_concurrent = max(max_concurrent, len(active))
        concurrency_trace.append(len(active))

    order_columns = [*scored.columns, "layer_no", "capacity_blocked", "submitted"]
    orders = pd.DataFrame(submitted)
    if orders.empty:
        orders = pd.DataFrame(columns=order_columns)
    else:
        orders = orders[[column for column in order_columns if column in orders.columns]]
    metrics = _metrics(
        orders,
        threshold_crossings=threshold_crossings,
        bucket_skips=bucket_skips,
        capacity_skips=capacity_skips,
        concurrency_trace=concurrency_trace,
        evaluation_days=evaluation_days,
    )
    return PolicyReplay(
        orders=orders.reset_index(drop=True),
        metrics=metrics,
        input_scores=input_scores,
        max_concurrent=max_concurrent,
    )


def _episode_bootstrap_interval(
    orders: pd.DataFrame,
    *,
    alpha: float,
    replicates: int,
    seed: int = 42,
) -> tuple[float, float]:
    if orders.empty or replicates < 1:
        return np.nan, np.nan
    grouped = orders.groupby("channel_episode_id", sort=False)["r_net"].agg(["sum", "count"])
    if grouped.empty:
        return np.nan, np.nan
    sums = grouped["sum"].to_numpy(dtype=float)
    counts = grouped["count"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(grouped), size=(replicates, len(grouped)))
    means = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return (
        float(np.quantile(means, alpha / 2.0)),
        float(np.quantile(means, 1.0 - alpha / 2.0)),
    )


def sweep_thresholds(
    scored: pd.DataFrame,
    *,
    capacities: tuple[int | None, ...] = (3, 5, None),
    quantiles: tuple[float, ...] = SELECTION_QUANTILES,
    bootstrap_reps: int = 2000,
    inherited_trial_count: int = INHERITED_CHANNEL_TRIALS,
    evaluation_days: float | None = None,
) -> pd.DataFrame:
    """Evaluate a fixed tune-score quantile grid for every capacity sensitivity."""
    finite_scores = pd.to_numeric(scored["score"], errors="coerce").dropna()
    if finite_scores.empty:
        raise ValueError("threshold sweep needs finite scores")
    total_trial_count = inherited_trial_count + len(capacities) * len(quantiles) + 1
    adjusted_alpha = 0.05 / total_trial_count
    rows: list[dict[str, object]] = []
    trial_no = 0
    for capacity in capacities:
        for quantile in quantiles:
            trial_no += 1
            threshold = float(finite_scores.quantile(quantile))
            replay = replay_capacity(
                scored,
                threshold=threshold,
                capacity=capacity,
                evaluation_days=evaluation_days,
            )
            ordinary = _episode_bootstrap_interval(
                replay.orders, alpha=0.05, replicates=bootstrap_reps
            )
            adjusted = _episode_bootstrap_interval(
                replay.orders, alpha=adjusted_alpha, replicates=bootstrap_reps
            )
            rows.append(
                {
                    "trial_id": f"window-policy-{trial_no:03d}",
                    "capacity": capacity,
                    "capacity_label": "unlimited" if capacity is None else str(capacity),
                    "threshold_quantile": float(quantile),
                    "threshold": threshold,
                    **replay.metrics,
                    "bootstrap_low": ordinary[0],
                    "bootstrap_high": ordinary[1],
                    "bonferroni_low": adjusted[0],
                    "bonferroni_high": adjusted[1],
                    "ordinary_alpha": 0.05,
                    "adjusted_alpha": adjusted_alpha,
                    "total_trial_count": total_trial_count,
                }
            )
    return pd.DataFrame(rows)


def _eligible_threshold_rows(
    table: pd.DataFrame,
    capacity: int | None,
    *,
    min_trades_per_day: float = 0.0,
) -> pd.DataFrame:
    if capacity is None:
        capacity_mask = table["capacity"].isna()
    else:
        capacity_mask = table["capacity"].eq(capacity)
    if min_trades_per_day > 0 and "trades_per_day" not in table:
        raise ValueError("frequency-constrained selection needs trades_per_day")
    frequency = (
        table["trades_per_day"].ge(min_trades_per_day)
        if "trades_per_day" in table
        else pd.Series(True, index=table.index)
    )
    return table[
        capacity_mask
        & frequency
        & table["mean_r_net"].gt(0)
        & table["filled_trades"].ge(30)
        & table["long_fills"].ge(10)
        & table["short_fills"].ge(10)
    ].copy()


def choose_tune_threshold(
    table: pd.DataFrame,
    *,
    capacity: int | None = 3,
    min_trades_per_day: float = 0.0,
) -> float:
    """Choose the primary natural threshold; return NaN when support fails."""
    eligible = _eligible_threshold_rows(
        table, capacity, min_trades_per_day=min_trades_per_day
    )
    if eligible.empty:
        return float("nan")
    winner = eligible.sort_values(
        ["total_net_r", "mean_r_net", "threshold"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    return float(winner["threshold"])


def choose_frequency_matched(
    table: pd.DataFrame,
    *,
    target_fills: int,
    capacity: int | None = 3,
) -> pd.Series:
    """Return the supported row closest to another arm's realised fill count."""
    eligible = _eligible_threshold_rows(table, capacity)
    if eligible.empty:
        return pd.Series(dtype=object)
    eligible["_frequency_distance"] = (eligible["filled_trades"] - target_fills).abs()
    return eligible.sort_values(
        ["_frequency_distance", "total_net_r", "threshold"],
        ascending=[True, False, True],
        kind="stable",
    ).iloc[0].drop(labels="_frequency_distance")


def hard_macro_sensitivity(
    scored: pd.DataFrame,
    *,
    threshold: float,
    capacity: int | None,
) -> PolicyReplay:
    """Apply the registered hard-SMA mask to a frozen score stream post hoc."""
    if "macro_alignment" not in scored:
        raise ValueError("macro sensitivity needs macro_alignment")
    original_scores = scored["score"].copy()
    masked = scored.copy()
    masked.loc[~masked["macro_alignment"].eq(1), "score"] = -np.inf
    replay = replay_capacity(masked, threshold=threshold, capacity=capacity)
    replay.input_scores = original_scores
    return replay
