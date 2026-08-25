"""Fixed zero-EV entry policy for causal Fast-T2 windows."""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_net_r(
    probabilities: np.ndarray,
    *,
    planned_tp_gross_r: np.ndarray,
    timeout_gross_r: np.ndarray,
    planned_cost_r: np.ndarray,
) -> np.ndarray:
    """Return expected net R for class order [SL, TP, timeout]."""
    values = np.asarray(probabilities, dtype=float)
    tp = np.asarray(planned_tp_gross_r, dtype=float)
    timeout = np.asarray(timeout_gross_r, dtype=float)
    cost = np.asarray(planned_cost_r, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("probabilities must have shape (n, 3)")
    if not (len(values) == len(tp) == len(timeout) == len(cost)):
        raise ValueError("EV inputs must align")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("outcome probabilities must sum to one")
    return values[:, 1] * tp - values[:, 0] + values[:, 2] * timeout - cost


def first_positive_ev_entries(
    scored: pd.DataFrame,
    *,
    min_risk_bps: float,
    cancel_after_target: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take the first strictly positive EV decision in each live window."""
    required = {
        "window_id",
        "decision_id",
        "decision_time",
        "ev_score",
        "distance_to_stop_bps",
        "target_reached_before_decision",
        "filled",
    }
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"causal EV scores missing columns: {missing}")
    if not np.isfinite(min_risk_bps) or min_risk_bps <= 0.0:
        raise ValueError("minimum risk must be positive and finite")

    work = scored.copy()
    work["decision_time"] = pd.to_datetime(
        work["decision_time"], utc=True, errors="raise"
    )
    work = work.sort_values(
        ["window_id", "decision_time", "decision_id"], kind="stable"
    )
    entry_indices: list[int] = []
    action_rows: list[dict[str, object]] = []
    for window_id, group in work.groupby("window_id", sort=False):
        terminated = False
        recorded: list[int] = []
        for index, row in group.iterrows():
            stop_cancelled = bool(
                row.get("stop_invalidated_before_decision", False)
            )
            target_cancelled = bool(
                cancel_after_target
                and row.get("target_reached_before_decision", False)
            )
            if stop_cancelled or target_cancelled:
                action = "CANCEL"
                terminated = True
            else:
                risk_ok = bool(
                    np.isfinite(row["distance_to_stop_bps"])
                    and float(row["distance_to_stop_bps"]) >= min_risk_bps
                )
                score = float(row["ev_score"])
                eligible = risk_ok and bool(row["filled"]) and np.isfinite(score)
                if eligible and score > 0.0:
                    action = "ENTER"
                    entry_indices.append(index)
                    terminated = True
                else:
                    action = "WAIT"
            action_rows.append(
                {
                    "window_id": window_id,
                    "decision_id": row["decision_id"],
                    "decision_time": row["decision_time"],
                    "action": action,
                    "ev_score": row["ev_score"],
                    "min_risk_bps": float(min_risk_bps),
                    "cancel_after_target": bool(cancel_after_target),
                }
            )
            recorded.append(len(action_rows) - 1)
            if terminated:
                break
        if not terminated and recorded:
            action_rows[recorded[-1]]["action"] = "SKIP"

    entries = work.loc[entry_indices].copy() if entry_indices else work.iloc[:0].copy()
    actions = pd.DataFrame(action_rows)
    return entries.reset_index(drop=True), actions.reset_index(drop=True)
