"""Causal cost-aware ENTER/WAIT/SKIP policy for Notebook L event windows."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


_SCORE_COLUMNS = ("conservative_net_ev", "enter_advantage_vs_wait")
_LABEL_COLUMNS = frozenset(
    {
        "window_id",
        "channel_episode_id",
        "side",
        "step",
        "decision_time",
        "entry_time",
        "geometry_valid",
        "path_observed",
        "outcome",
        "r_gross",
        "risk_bps",
    }
)


@dataclass(frozen=True)
class CostAwarePolicyConfig:
    """Pre-registered decision threshold and outcome-specific fee schedule."""

    threshold: float = 0.0
    maker_entry_bps: float = 2.0
    maker_tp_exit_bps: float = 2.0
    taker_sl_exit_bps: float = 5.0
    timeout_exit_bps: float = 5.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("threshold must be non-negative and finite")
        for name in (
            "maker_entry_bps",
            "maker_tp_exit_bps",
            "taker_sl_exit_bps",
            "timeout_exit_bps",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and finite")

    def round_trip_bps(self, outcome: str) -> float:
        exits = {
            "tp": self.maker_tp_exit_bps,
            "sl": self.taker_sl_exit_bps,
            "timeout": self.timeout_exit_bps,
        }
        if outcome == "unfilled":
            return 0.0
        if outcome not in exits:
            raise ValueError(f"unsupported observed outcome: {outcome!r}")
        return float(self.maker_entry_bps + exits[outcome])


@dataclass(frozen=True)
class CostAwarePolicyReplay:
    trades: pd.DataFrame
    actions: pd.DataFrame
    attempted_trades: int
    observed_trades: int
    filled_trades: int
    mean_realized_net_r: float
    total_realized_net_r: float


def _attach_scores(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    missing_scores = sorted(
        {"window_id", "step", *_SCORE_COLUMNS}.difference(scores.columns)
    )
    if missing_scores:
        raise ValueError(f"scores missing policy columns: {missing_scores}")
    missing_labels = sorted(_LABEL_COLUMNS.difference(labels.columns))
    if missing_labels:
        raise ValueError(f"labels missing policy columns: {missing_labels}")
    keys = ["window_id", "step"]
    if scores.duplicated(keys).any():
        raise ValueError("scores must be unique by window_id and step")
    if labels.duplicated(keys).any():
        raise ValueError("labels must be unique by window_id and step")
    lookup = scores[[*keys, *_SCORE_COLUMNS]].copy()
    work = labels.merge(
        lookup,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if work["_merge"].ne("both").any():
        absent = work.loc[work["_merge"].ne("both"), keys].to_dict("records")
        raise ValueError(f"scores missing label keys: {absent[:3]}")
    work = work.drop(columns="_merge")
    work["decision_time"] = pd.to_datetime(
        work["decision_time"], utc=True, errors="raise"
    )
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="raise")
    work["conservative_net_ev"] = pd.to_numeric(
        work["conservative_net_ev"], errors="coerce"
    )
    work["enter_advantage_vs_wait"] = pd.to_numeric(
        work["enter_advantage_vs_wait"], errors="coerce"
    )
    work["_input_order"] = np.arange(len(work), dtype=np.int64)
    return work


def _apply_realized_costs(
    trades: pd.DataFrame, config: CostAwarePolicyConfig
) -> pd.DataFrame:
    out = trades.copy()
    out["round_trip_cost_bps"] = np.nan
    out["realized_net_r"] = np.nan
    for index, row in out.iterrows():
        if not bool(row["path_observed"]):
            continue
        if str(row["outcome"]).lower() == "unfilled":
            out.at[index, "round_trip_cost_bps"] = 0.0
            out.at[index, "realized_net_r"] = 0.0
            continue
        risk_bps = float(row["risk_bps"])
        r_gross = float(row["r_gross"])
        if not np.isfinite(risk_bps) or risk_bps <= 0.0:
            raise ValueError("observed trades require positive finite risk_bps")
        if not np.isfinite(r_gross):
            raise ValueError("observed trades require finite r_gross")
        cost_bps = config.round_trip_bps(str(row["outcome"]).lower())
        out.at[index, "round_trip_cost_bps"] = cost_bps
        out.at[index, "realized_net_r"] = r_gross - cost_bps / risk_bps
    return out


def replay_cost_aware_policy(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    config: CostAwarePolicyConfig = CostAwarePolicyConfig(),
) -> CostAwarePolicyReplay:
    """Replay one fixed causal policy without threshold tuning or forced entries.

    Only geometry-valid decisions participate.  A failed condition is WAIT
    until the final valid decision, where it becomes SKIP.  The first joint
    crossing enters and consumes the window even when its path is censored.
    """
    if not isinstance(config, CostAwarePolicyConfig):
        raise TypeError("config must be CostAwarePolicyConfig")
    work = _attach_scores(scores, labels)
    valid = work[work["geometry_valid"].astype(bool)].sort_values(
        ["window_id", "decision_time", "step", "_input_order"], kind="stable"
    )
    actions: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    for _, group in valid.groupby("window_id", sort=False):
        rows = list(group.itertuples(index=False))
        for position, row in enumerate(rows):
            ev = float(row.conservative_net_ev)
            advantage = float(row.enter_advantage_vs_wait)
            enter = bool(
                np.isfinite(ev)
                and np.isfinite(advantage)
                and ev >= config.threshold
                and advantage >= 0.0
            )
            last = position == len(rows) - 1
            action = "ENTER" if enter else ("SKIP" if last else "WAIT")
            actions.append(
                {
                    "window_id": row.window_id,
                    "step": int(row.step),
                    "decision_time": row.decision_time,
                    "conservative_net_ev": ev,
                    "enter_advantage_vs_wait": advantage,
                    "action": action,
                }
            )
            if enter:
                record = row._asdict()
                record.pop("_input_order", None)
                selected.append(record)
                break

    trade_columns = [column for column in work.columns if column != "_input_order"]
    trades = pd.DataFrame(selected, columns=trade_columns)
    trades = _apply_realized_costs(trades, config)
    if not trades.empty:
        trades = trades.sort_values(
            ["entry_time", "window_id"], kind="stable"
        ).reset_index(drop=True)
        if trades["window_id"].duplicated().any():
            raise AssertionError("a window entered more than once")
    action_columns = [
        "window_id",
        "step",
        "decision_time",
        "conservative_net_ev",
        "enter_advantage_vs_wait",
        "action",
    ]
    action_frame = pd.DataFrame(actions, columns=action_columns)
    if not action_frame.empty:
        action_frame = action_frame.sort_values(
            ["decision_time", "window_id", "step"], kind="stable"
        ).reset_index(drop=True)
    observed_orders = (
        trades["path_observed"].astype(bool)
        if not trades.empty
        else pd.Series(dtype=bool)
    )
    filled = (
        observed_orders & trades["outcome"].astype(str).str.lower().isin({"tp", "sl", "timeout"})
        if not trades.empty
        else pd.Series(dtype=bool)
    )
    realized = pd.to_numeric(
        trades.loc[filled, "realized_net_r"], errors="coerce"
    ).dropna()
    return CostAwarePolicyReplay(
        trades=trades,
        actions=action_frame,
        attempted_trades=int(len(trades)),
        observed_trades=int(observed_orders.sum()),
        filled_trades=int(filled.sum()),
        mean_realized_net_r=float(realized.mean()) if len(realized) else np.nan,
        total_realized_net_r=float(realized.sum()) if len(realized) else 0.0,
    )


__all__ = [
    "CostAwarePolicyConfig",
    "CostAwarePolicyReplay",
    "replay_cost_aware_policy",
]
