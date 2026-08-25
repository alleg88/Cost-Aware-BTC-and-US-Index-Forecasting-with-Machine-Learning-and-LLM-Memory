"""Economic thresholding, RR sensitivity, and loss-streak diagnostics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.fast_t2_action_policy import (
    first_crossing_entries,
    replay_entry_capacity,
)


ECONOMIC_SCORE_QUANTILES = (0.0, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
SELECTION_NET_R_CLIP = (-5.0, 5.0)


def economic_first_crossing_entries(
    scored: pd.DataFrame,
    *,
    threshold: float,
    min_rr: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply a causal RR eligibility check before the economic crossing."""
    if "rr_proxy" not in scored:
        raise ValueError("economic policy requires rr_proxy")
    if min_rr is not None and (not np.isfinite(min_rr) or min_rr < 0):
        raise ValueError("min_rr must be non-negative and finite")
    work = scored.copy()
    rr = pd.to_numeric(work["rr_proxy"], errors="coerce")
    eligible = rr.notna() if min_rr is None else rr.ge(min_rr)
    original_scores = work.set_index("decision_id")["score"].copy()
    eligibility = pd.Series(
        eligible.to_numpy(dtype=bool), index=work["decision_id"].astype(str)
    )
    work.loc[~eligible, "score"] = -np.inf
    entries, actions = first_crossing_entries(work, threshold)
    if not actions.empty:
        ids = actions["decision_id"].astype(str)
        actions["score"] = ids.map(original_scores)
        actions["rr_eligible"] = ids.map(eligibility).astype(bool)
    return entries, actions


def select_economic_threshold(
    inner_scored: pd.DataFrame,
    *,
    evaluation_days: float,
    min_rr: float | None,
    min_trades_per_day: float = 1.0,
) -> tuple[float, pd.DataFrame]:
    """Select a train-only score quantile under the registered frequency floor."""
    if min_trades_per_day <= 0 or not np.isfinite(min_trades_per_day):
        raise ValueError("min_trades_per_day must be positive and finite")
    finite = pd.to_numeric(inner_scored["score"], errors="coerce").dropna()
    if finite.empty:
        raise ValueError("economic threshold selection needs finite scores")
    rows: list[dict[str, object]] = []
    for quantile in ECONOMIC_SCORE_QUANTILES:
        threshold = float(finite.quantile(quantile))
        entries, actions = economic_first_crossing_entries(
            inner_scored, threshold=threshold, min_rr=min_rr
        )
        replay = replay_entry_capacity(
            entries,
            capacity=None,
            evaluation_days=evaluation_days,
            bootstrap_reps=0,
        )
        filled = replay.orders[replay.orders["filled"].astype(bool)]
        robust_mean = (
            float(filled["r_net"].clip(*SELECTION_NET_R_CLIP).mean())
            if len(filled)
            else np.nan
        )
        metrics = replay.metrics
        frequency_eligible = bool(
            metrics["trades_per_day"] >= min_trades_per_day
            and metrics["long_trades"] > 0
            and metrics["short_trades"] > 0
        )
        rows.append(
            {
                "quantile": float(quantile),
                "threshold": threshold,
                "min_rr": min_rr,
                "robust_mean_net_r": robust_mean,
                "frequency_eligible": frequency_eligible,
                "wait_actions": int(actions["action"].eq("WAIT").sum()),
                "enter_actions": int(actions["action"].eq("ENTER").sum()),
                "skip_actions": int(actions["action"].eq("SKIP").sum()),
                **metrics,
            }
        )
    table = pd.DataFrame(rows)
    eligible = table[table["frequency_eligible"].astype(bool)]
    source = eligible if not eligible.empty else table
    ranked = source.assign(
        _robust=source["robust_mean_net_r"].fillna(-np.inf),
        _raw=source["mean_net_r"].fillna(-np.inf),
        _total=source["total_net_r"].fillna(-np.inf),
    ).sort_values(
        ["_robust", "_raw", "_total", "quantile"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return float(ranked.iloc[0]["quantile"]), table


def summarise_loss_streaks(trades: pd.DataFrame) -> dict[str, float | int]:
    """Describe consecutive non-positive trades in completed-trade order."""
    required = {"active_end_time", "decision_id", "r_net"}
    missing = sorted(required.difference(trades.columns))
    if missing:
        raise ValueError(f"loss-streak ledger missing columns: {missing}")
    if trades.empty:
        return {
            "trades": 0,
            "loss_trades": 0,
            "loss_streak_count": 0,
            "max_loss_streak": 0,
            "mean_loss_streak": 0.0,
            "p95_loss_streak": 0.0,
        }
    work = trades.copy()
    work["active_end_time"] = pd.to_datetime(
        work["active_end_time"], utc=True, errors="raise"
    )
    work = work.sort_values(
        ["active_end_time", "decision_id"], kind="stable"
    )
    streaks: list[int] = []
    current = 0
    for is_loss in work["r_net"].le(0.0):
        if bool(is_loss):
            current += 1
        elif current:
            streaks.append(current)
            current = 0
    if current:
        streaks.append(current)
    values = np.asarray(streaks, dtype=float)
    return {
        "trades": int(len(work)),
        "loss_trades": int(work["r_net"].le(0.0).sum()),
        "loss_streak_count": int(len(values)),
        "max_loss_streak": int(values.max()) if len(values) else 0,
        "mean_loss_streak": float(values.mean()) if len(values) else 0.0,
        "p95_loss_streak": float(np.quantile(values, 0.95)) if len(values) else 0.0,
    }


def mark_frequency_eligibility(
    policies: pd.DataFrame, *, minimum: float = 1.0
) -> pd.DataFrame:
    """Mark each policy row against the frequency and two-sided support gate."""
    required = {"trades_per_day", "long_trades", "short_trades"}
    missing = sorted(required.difference(policies.columns))
    if missing:
        raise ValueError(f"frequency table missing columns: {missing}")
    if not np.isfinite(minimum) or minimum <= 0:
        raise ValueError("minimum frequency must be positive and finite")
    marked = policies.copy()
    marked["frequency_eligible"] = (
        marked["trades_per_day"].ge(minimum)
        & marked["long_trades"].gt(0)
        & marked["short_trades"].gt(0)
    )
    return marked


def select_primary_economic_arm(policies: pd.DataFrame) -> pd.Series:
    """Choose by raw OOF economics after enforcing the registered frequency gate."""
    required = {
        "arm",
        "frequency_eligible",
        "mean_net_r",
        "robust_mean_net_r",
        "total_net_r",
    }
    missing = sorted(required.difference(policies.columns))
    if missing:
        raise ValueError(f"primary economic table missing columns: {missing}")
    if policies.empty:
        raise ValueError("primary economic table cannot be empty")
    eligible = policies[policies["frequency_eligible"].astype(bool)]
    source = eligible if not eligible.empty else policies
    ranked = source.assign(
        _raw=source["mean_net_r"].fillna(-np.inf),
        _robust=source["robust_mean_net_r"].fillna(-np.inf),
        _total=source["total_net_r"].fillna(-np.inf),
    ).sort_values(
        ["_raw", "_robust", "_total", "arm"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    return ranked.iloc[0]
