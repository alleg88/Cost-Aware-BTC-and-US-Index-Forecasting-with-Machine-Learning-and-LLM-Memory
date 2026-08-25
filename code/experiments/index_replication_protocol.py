"""Frozen causal contracts shared by the two index replications."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Any, Mapping

import numpy as np
import pandas as pd


CUTOFF = pd.Timestamp("2026-04-01T00:00:00Z")
SELECTION_START = pd.Timestamp("2024-01-01T00:00:00Z")
SELECTION_END = pd.Timestamp("2025-01-01T00:00:00Z")
CALIBRATION_START = SELECTION_END
FORWARD_START = pd.Timestamp("2025-07-01T00:00:00Z")
FORWARD_END = CUTOFF
WIDTHS = (5, 10, 15)
TAUS = (0.00, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
MODEL_NAMES = (
    "logreg",
    "decision_tree",
    "random_forest",
    "svm_linear",
    "xgboost_balanced",
    "catboost_balanced",
    "mlp",
    "lstm",
    "gru",
)
VIX_BASE_COLS = ("vix_log_close", "vix_r1", "vix_r4", "vix_vol20")
VIX_FEATURE_COLS = (*VIX_BASE_COLS, "vix_age_minutes")
MIN_TRADES = 50
MIN_SIDE_TRADES = 15
MIN_POSITIVE_MONTHS = 4


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def build_vix_block(vix_bars: pd.DataFrame) -> pd.DataFrame:
    """Build the fixed VIX block on complete bars, available only at bar close."""
    required = {"close", "available_at", "complete_bar"}
    missing = required.difference(vix_bars.columns)
    if missing:
        raise ValueError(f"VIX bars miss columns: {sorted(missing)}")
    if vix_bars.index.tz is None:
        raise ValueError("VIX bar index must be timezone-aware")
    bars = vix_bars.loc[vix_bars["complete_bar"].astype(bool)].copy().sort_index()
    if bars.empty:
        raise ValueError("VIX complete-bar set is empty")
    available = pd.to_datetime(bars["available_at"], utc=True)
    if not (available > bars.index).all():
        raise ValueError("VIX bars must become available after their bar-open timestamp")
    log_close = np.log(bars["close"].astype(float))
    r1 = log_close.diff()
    block = pd.DataFrame(
        {
            "vix_log_close": log_close,
            "vix_r1": r1,
            "vix_r4": log_close.diff(4),
            "vix_vol20": r1.rolling(20, min_periods=20).std(),
            "vix_bar_open": bars.index,
            "vix_available_at": available.to_numpy(),
        },
        index=bars.index,
    )
    block["vix_available_at"] = pd.to_datetime(block["vix_available_at"], utc=True)
    return block.dropna(subset=list(VIX_BASE_COLS))


def join_completed_vix(index_bars: pd.DataFrame, vix_block: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of join keyed by actual VIX availability, never bar-open."""
    if "decision_time" not in index_bars:
        raise ValueError("index bars require decision_time")
    required = {"vix_available_at", *VIX_BASE_COLS}
    missing = required.difference(vix_block.columns)
    if missing:
        raise ValueError(f"VIX block misses columns: {sorted(missing)}")
    left = index_bars.copy()
    left["decision_time"] = pd.to_datetime(left["decision_time"], utc=True)
    left["__bar_open"] = left.index
    left = left.sort_values("decision_time")
    right = vix_block.reset_index(drop=True).sort_values("vix_available_at")
    joined = pd.merge_asof(
        left,
        right,
        left_on="decision_time",
        right_on="vix_available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    available = pd.to_datetime(joined["vix_available_at"], utc=True)
    leaked = available.notna() & available.gt(joined["decision_time"])
    if leaked.any():
        raise AssertionError("VIX as-of join used a future completed bar")
    joined["vix_age_minutes"] = (
        joined["decision_time"] - available
    ).dt.total_seconds() / 60.0
    joined = joined.set_index("__bar_open")
    joined.index.name = index_bars.index.name
    return joined.reindex(index_bars.index)


@dataclass(frozen=True)
class VixAdmissionDecision:
    selected_base: str
    admitted: bool
    conditions: Mapping[str, bool]
    metrics: Mapping[str, Any]
    family_deltas: tuple[Mapping[str, Any], ...] = ()
    fold_deltas: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _failed_vix_decision(*, actual_rows: int) -> VixAdmissionDecision:
    conditions = {
        "median_positive": False,
        "families_positive_majority_5_of_9": False,
        "folds_positive_3_of_5": False,
        "trade_retention_80pct": False,
    }
    return VixAdmissionDecision(
        selected_base="price",
        admitted=False,
        conditions=conditions,
        metrics={
            "complete_pairing": False,
            "expected_rows": len(MODEL_NAMES) * len(WIDTHS) * 5,
            "actual_rows": int(actual_rows),
        },
    )


def decide_vix_admission(paired: pd.DataFrame) -> VixAdmissionDecision:
    """Apply the frozen four-condition instrument-level 2024 OOF VIX gate."""
    required = {
        "model_name",
        "width_bps",
        "fold_id",
        "price_net",
        "vix_net",
        "price_trades",
        "vix_trades",
        "common_timestamp_hash",
    }
    missing = required.difference(paired.columns)
    if missing:
        raise ValueError(f"paired VIX table misses columns: {sorted(missing)}")
    keys = list(product(MODEL_NAMES, WIDTHS, range(5)))
    actual_keys = list(
        paired[["model_name", "width_bps", "fold_id"]]
        .itertuples(index=False, name=None)
    )
    complete = (
        len(actual_keys) == len(keys)
        and len(set(actual_keys)) == len(keys)
        and set(actual_keys) == set(keys)
        and paired["common_timestamp_hash"].astype(str).str.len().gt(0).all()
    )
    if not complete:
        return _failed_vix_decision(actual_rows=len(paired))

    work = paired.copy()
    numeric = ["price_net", "vix_net", "price_trades", "vix_trades"]
    work[numeric] = work[numeric].apply(pd.to_numeric, errors="coerce")
    if work[numeric].isna().any().any() or (work[["price_trades", "vix_trades"]] < 0).any().any():
        return _failed_vix_decision(actual_rows=len(work))
    work["delta"] = work["vix_net"] - work["price_net"]

    model_width = (
        work.groupby(["model_name", "width_bps"], sort=False)["delta"]
        .sum()
        .rename("delta")
        .reset_index()
    )
    family = (
        model_width.groupby("model_name", sort=False)["delta"]
        .median()
        .reindex(MODEL_NAMES)
    )
    folds = work.groupby("fold_id", sort=True)["delta"].median().reindex(range(5))
    median_family_delta = float(family.median())
    positive_families = int((family > 0).sum())
    positive_folds = int((folds > 0).sum())
    price_trades = float(work["price_trades"].sum())
    retention = float(work["vix_trades"].sum() / price_trades) if price_trades > 0 else 0.0
    conditions = {
        "median_positive": bool(median_family_delta > 0),
        "families_positive_majority_5_of_9": bool(positive_families >= 5),
        "folds_positive_3_of_5": bool(positive_folds >= 3),
        "trade_retention_80pct": bool(retention >= 0.80),
    }
    admitted = all(conditions.values())
    return VixAdmissionDecision(
        selected_base="price_vix" if admitted else "price",
        admitted=admitted,
        conditions=conditions,
        metrics={
            "complete_pairing": True,
            "median_family_delta": median_family_delta,
            "positive_families": positive_families,
            "positive_folds": positive_folds,
            "trade_retention": retention,
            "expected_rows": len(keys),
            "actual_rows": len(work),
        },
        family_deltas=tuple(
            {"model_name": str(model), "net_delta": float(value)}
            for model, value in family.items()
        ),
        fold_deltas=tuple(
            {"fold_id": int(fold), "median_net_delta": float(value)}
            for fold, value in folds.items()
        ),
    )


def _ratio(mean: float, scale: float) -> float:
    return float(mean / scale) if np.isfinite(scale) and scale > 0 else 0.0


def daily_economics(
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
) -> dict[str, Any]:
    """Summarise economics on a common zero-filled UTC calendar-day grid."""
    start_utc, end_utc = _utc(start), _utc(end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")
    returns = per_bar.copy().astype(float)
    returns.index = pd.to_datetime(returns.index, utc=True)
    returns = returns.loc[(returns.index >= start_utc) & (returns.index < end_utc)]
    calendar = pd.date_range(
        start_utc.normalize(), end_utc.normalize(), freq="1D", inclusive="left", tz="UTC"
    )
    daily = returns.resample("1D").sum().reindex(calendar, fill_value=0.0)
    mean = float(daily.mean()) if len(daily) else 0.0
    standard = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    downside = daily.loc[daily < 0]
    downside_scale = (
        float(np.sqrt(np.mean(np.square(downside.to_numpy(dtype=float)))))
        if len(downside)
        else 0.0
    )
    sharpe = _ratio(mean, standard) * np.sqrt(365.0)
    sortino = _ratio(mean, downside_scale) * np.sqrt(365.0)
    equity = (1.0 + daily).cumprod()
    drawdown = equity / equity.cummax() - 1.0 if len(equity) else pd.Series(dtype=float)

    current = ledger.copy()
    if "entry_time" in current:
        current["entry_time"] = pd.to_datetime(current["entry_time"], utc=True)
        current = current.loc[
            (current["entry_time"] >= start_utc) & (current["entry_time"] < end_utc)
        ]
    required = {"side", "gross_return", "net_return"}
    missing = required.difference(current.columns)
    if missing and len(current):
        raise ValueError(f"ledger misses columns: {sorted(missing)}")
    trades = int(len(current))
    gross = float(current.get("gross_return", pd.Series(dtype=float)).sum())
    net = float(current.get("net_return", pd.Series(dtype=float)).sum())
    cost = gross - net
    return {
        "calendar_days": int(len(calendar)),
        "trades": trades,
        "n_long": int((current.get("side", pd.Series(dtype=float)) == 1).sum()),
        "n_short": int((current.get("side", pd.Series(dtype=float)) == -1).sum()),
        "trades_per_day": float(trades / len(calendar)) if len(calendar) else 0.0,
        "gross_return": gross,
        "cost_return": cost,
        "net_return": net,
        "net_bps_per_trade": float(net * 10_000.0 / trades) if trades else 0.0,
        "exposure": float(returns.ne(0.0).mean()) if len(returns) else 0.0,
        "daily_sharpe": float(sharpe),
        "daily_sortino": float(sortino),
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
    }


def _constraint_violation(frame: pd.DataFrame) -> pd.Series:
    return (
        (MIN_TRADES - frame["trades"].astype(int)).clip(lower=0)
        + (MIN_SIDE_TRADES - frame["n_long"].astype(int)).clip(lower=0)
        + (MIN_SIDE_TRADES - frame["n_short"].astype(int)).clip(lower=0)
        + (MIN_POSITIVE_MONTHS - frame["positive_months"].astype(int)).clip(lower=0)
    ).astype(int)


def select_h1_policy(grid: pd.DataFrame) -> pd.Series:
    """Choose one DZ/tau row by constraint, Sortino, net and trade count."""
    required = {
        "width_bps",
        "tau",
        "trades",
        "n_long",
        "n_short",
        "positive_months",
        "daily_sortino",
        "net_return",
    }
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"H1 policy grid misses columns: {sorted(missing)}")
    if grid.empty:
        raise ValueError("H1 policy grid is empty")
    work = grid.copy()
    work["constraint_violation"] = _constraint_violation(work)
    work["eligible"] = work["constraint_violation"].eq(0)
    work["__sortino"] = pd.to_numeric(work["daily_sortino"], errors="coerce").fillna(-np.inf)
    ranked = work.sort_values(
        ["constraint_violation", "__sortino", "net_return", "trades", "width_bps", "tau"],
        ascending=[True, False, False, False, True, True],
        kind="mergesort",
    )
    winner = ranked.iloc[0].drop(labels="__sortino").copy()
    winner["selection_rule"] = "violation,daily_sortino,net_return,trades,width,tau"
    return winner


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm family-wise p-value adjustment, preserving the input index."""
    values = pd.to_numeric(p_values, errors="coerce")
    if values.isna().any() or ((values < 0) | (values > 1)).any():
        raise ValueError("p-values must be finite values in [0, 1]")
    order = values.sort_values(kind="mergesort")
    count = len(order)
    adjusted_ordered = pd.Series(
        np.maximum.accumulate(
            [min(1.0, (count - rank) * float(value)) for rank, value in enumerate(order)]
        ),
        index=order.index,
        dtype=float,
    )
    return adjusted_ordered.reindex(values.index)


__all__ = [
    "CALIBRATION_START",
    "CUTOFF",
    "FORWARD_END",
    "FORWARD_START",
    "MODEL_NAMES",
    "SELECTION_END",
    "SELECTION_START",
    "TAUS",
    "VIX_FEATURE_COLS",
    "VixAdmissionDecision",
    "WIDTHS",
    "build_vix_block",
    "daily_economics",
    "decide_vix_admission",
    "holm_adjust",
    "join_completed_vix",
    "select_h1_policy",
]
