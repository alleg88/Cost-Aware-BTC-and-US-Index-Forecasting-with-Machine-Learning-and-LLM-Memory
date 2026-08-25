"""Strict positive-EV policy over complete T1 lifecycles."""
from __future__ import annotations

import numpy as np
import pandas as pd


PRIMARY_MIN_RISK_BPS = 25.0


def first_positive_lifecycle_entries(
    scored: pd.DataFrame, *, phase: str = "all"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enter once per arm at its first valid, strictly positive EV decision."""
    if phase not in {"all", "pre_t2", "post_t2"}:
        raise ValueError("phase must be all, pre_t2, or post_t2")
    required = {
        "arm_id", "decision_id", "decision_time", "decision_phase", "ev_score",
        "distance_to_stop_bps", "filled",
    }
    missing = sorted(required.difference(scored.columns))
    if missing:
        raise ValueError(f"pre-T2 scores missing columns: {missing}")
    work = scored.copy()
    work["decision_time"] = pd.to_datetime(work["decision_time"], utc=True)
    work = work.sort_values(["arm_id", "decision_time", "decision_id"], kind="stable")
    if phase != "all":
        work = work[work["decision_phase"].eq(phase)].copy()
    entries: list[int] = []
    actions: list[dict[str, object]] = []
    for arm_id, group in work.groupby("arm_id", sort=False):
        group_rows = list(group.iterrows())
        terminated = False
        for position, (index, row) in enumerate(group_rows):
            stop_cancelled = bool(row.get("stop_invalidated_before_decision", False))
            score = float(row["ev_score"])
            eligible = (
                not stop_cancelled
                and bool(row["filled"])
                and np.isfinite(score)
                and float(row["distance_to_stop_bps"]) >= PRIMARY_MIN_RISK_BPS
            )
            if stop_cancelled:
                action = "CANCEL"
                terminated = True
            elif eligible and score > 0.0:
                action = "ENTER"
                entries.append(index)
                terminated = True
            else:
                action = "SKIP" if position == len(group_rows) - 1 else "WAIT"
            actions.append(
                {
                    "arm_id": arm_id,
                    "decision_id": row["decision_id"],
                    "decision_time": row["decision_time"],
                    "decision_phase": row["decision_phase"],
                    "ev_score": score,
                    "action": action,
                }
            )
            if terminated:
                break
    entry_frame = work.loc[entries].copy() if entries else work.iloc[:0].copy()
    return entry_frame.reset_index(drop=True), pd.DataFrame(actions)
