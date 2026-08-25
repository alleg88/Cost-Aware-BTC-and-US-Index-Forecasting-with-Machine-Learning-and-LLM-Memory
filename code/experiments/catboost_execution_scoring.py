"""Shared fixed-resolution policy scoring and paired-ledger comparison."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.catboost_execution_resolution import policy_choices
from experiments.catboost_matched_ablation import REGIMES, robust_score
from experiments.run_catboost_matched_ablation import safe_signal_mask


def _interval_for_resolution(resolution: str) -> pd.Timedelta:
    if resolution == "1m":
        return pd.Timedelta(minutes=1)
    if resolution == "1s":
        return pd.Timedelta(seconds=1)
    raise ValueError("resolution must be '1m' or '1s'")


def _segment_edges(
    start: pd.Timestamp, end: pd.Timestamp
) -> tuple[pd.Timestamp, ...]:
    monthly = tuple(pd.date_range(start, end, freq="MS"))
    if not monthly or monthly[0] != start:
        monthly = (start, *monthly)
    if monthly[-1] != end:
        monthly = (*monthly, end)
    return monthly


def simulate_policy(
    *,
    bars: pd.DataFrame,
    execution: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resolution: str,
    tau: float,
    tp_bps: int,
    sl_bps: int,
    max_hold: int,
    fee_bps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    indexed = prediction_frame.copy()
    indexed["timestamp"] = pd.to_datetime(indexed["timestamp"], utc=True)
    indexed = indexed.set_index("timestamp").sort_index()
    safe = safe_signal_mask(indexed.index, end_exclusive=end, max_hold=max_hold)
    scope_bars = bars.loc[(bars.index >= start) & (bars.index < end)]
    ledger, per_bar = simulate_bracket_trades_intrabar(
        scope_bars,
        execution,
        indexed.loc[safe, "pred"].astype(int),
        indexed.loc[safe, "confidence"].astype(float),
        tau=float(tau),
        tp_bps=float(tp_bps),
        sl_bps=float(sl_bps),
        max_hold=int(max_hold),
        fee_bps=float(fee_bps),
        expected_interval=_interval_for_resolution(resolution),
        include_audit=True,
    )
    if not np.isclose(
        float(per_bar.sum()), float(ledger["net_return"].sum()), atol=1e-10
    ):
        raise AssertionError("execution return series does not reconcile to ledger")
    return ledger, per_bar


def score_continuous_policy_grid(
    *,
    stage: str,
    width_bps: int,
    candidate_id: int,
    prediction_frame: pd.DataFrame,
    bars: pd.DataFrame,
    execution: pd.DataFrame,
    regimes: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    resolution: str,
    fee_bps: float,
    segment_edges: Sequence[pd.Timestamp] | None = None,
) -> pd.DataFrame:
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    edges = tuple(segment_edges or _segment_edges(start, end))
    if len(edges) < 2:
        raise ValueError("policy scoring needs at least one segment")
    rows: list[dict[str, Any]] = []
    for policy_id, (tau, geometry) in enumerate(policy_choices()):
        tp_bps, sl_bps, max_hold = geometry
        ledger, per_bar = simulate_policy(
            bars=bars,
            execution=execution,
            prediction_frame=prediction_frame,
            start=start,
            end=end,
            resolution=resolution,
            tau=tau,
            tp_bps=tp_bps,
            sl_bps=sl_bps,
            max_hold=max_hold,
            fee_bps=fee_bps,
        )
        summary = economics_summary(per_bar)
        bar_regimes = regimes.reindex(per_bar.index)
        regime_sortino = {
            regime: float(
                economics_summary(per_bar.loc[bar_regimes == regime])["sortino"]
            )
            for regime in REGIMES
        }
        segment_nets = [
            float(per_bar.loc[(per_bar.index >= left) & (per_bar.index < right)].sum())
            for left, right in zip(edges[:-1], edges[1:])
        ]
        ambiguous = int(ledger["ambiguous_touch"].sum())
        row = {
            "stage": stage,
            "resolution": resolution,
            "width_bps": int(width_bps),
            "candidate_id": int(candidate_id),
            "policy_id": int(policy_id),
            "tau": float(tau),
            "tp_bps": int(tp_bps),
            "sl_bps": int(sl_bps),
            "max_hold": int(max_hold),
            "trades": int(len(ledger)),
            "pooled_gross": float(ledger["gross_return"].sum()),
            "pooled_net": float(ledger["net_return"].sum()),
            "pooled_sortino": float(summary["sortino"]),
            "pooled_sharpe": float(summary["sharpe"]),
            "positive_segments": int(sum(value > 0.0 for value in segment_nets)),
            "n_long": int((ledger["side"] == 1).sum()),
            "n_short": int((ledger["side"] == -1).sum()),
            "long_net": float(ledger.loc[ledger["side"] == 1, "net_return"].sum()),
            "short_net": float(ledger.loc[ledger["side"] == -1, "net_return"].sum()),
            "bull_sortino": regime_sortino["bull"],
            "sideways_sortino": regime_sortino["sideways"],
            "bear_sortino": regime_sortino["bear"],
            "ambiguous_exits": ambiguous,
            "ambiguous_share": float(ambiguous / len(ledger)) if len(ledger) else 0.0,
        }
        row["robust_score"] = robust_score(
            pooled_sortino=row["pooled_sortino"],
            pooled_sharpe=row["pooled_sharpe"],
            bull_sortino=row["bull_sortino"],
            sideways_sortino=row["sideways_sortino"],
            bear_sortino=row["bear_sortino"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compare_paired_ledgers(
    one_minute: pd.DataFrame,
    one_second: pd.DataFrame,
) -> tuple[dict[str, int], pd.DataFrame, pd.DataFrame]:
    keys = ["entry_time", "side"]

    def prepared(frame: pd.DataFrame, suffix: str) -> pd.DataFrame:
        required = {*keys, "exit_reason", "net_return", "ambiguous_touch"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"paired ledger misses columns: {sorted(missing)}")
        out = frame.loc[:, [*keys, "exit_reason", "net_return", "ambiguous_touch"]].copy()
        out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
        if out.duplicated(keys).any():
            raise ValueError("paired ledger entry keys must be unique")
        return out.rename(
            columns={
                "exit_reason": f"exit_reason_{suffix}",
                "net_return": f"net_return_{suffix}",
                "ambiguous_touch": f"ambiguous_touch_{suffix}",
            }
        )

    outcomes = prepared(one_minute, "1m").merge(
        prepared(one_second, "1s"),
        on=keys,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    outcomes["match_status"] = outcomes["_merge"].map(
        {"both": "shared", "left_only": "one_minute_only", "right_only": "one_second_only"}
    )
    outcomes = outcomes.drop(columns="_merge")
    shared = outcomes.loc[outcomes["match_status"] == "shared"]
    changed_reason = shared["exit_reason_1m"] != shared["exit_reason_1s"]
    changed_net = ~np.isclose(
        shared["net_return_1m"].to_numpy(dtype=float),
        shared["net_return_1s"].to_numpy(dtype=float),
        atol=1e-12,
        rtol=0.0,
    )
    transitions = (
        shared.groupby(["exit_reason_1m", "exit_reason_1s"], as_index=False)
        .size()
        .rename(columns={"size": "trades"})
    )
    summary = {
        "shared_trades": int(len(shared)),
        "one_minute_only": int((outcomes["match_status"] == "one_minute_only").sum()),
        "one_second_only": int((outcomes["match_status"] == "one_second_only").sum()),
        "changed_exit_reason": int(changed_reason.sum()),
        "changed_net_result": int(changed_net.sum()),
    }
    return summary, outcomes, transitions
