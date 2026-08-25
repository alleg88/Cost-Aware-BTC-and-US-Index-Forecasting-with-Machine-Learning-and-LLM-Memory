"""Host-owned v4 policy masks and immutable Union-first scheduler."""
from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from experiments.xgb_strong_move_admission import (
    decompose_xgb_probabilities,
    load_xgb_panel,
)


Q2_START: Final = pd.Timestamp("2026-04-01", tz="UTC")
POLICY_IDS: Final = (
    "UNION_ONLY",
    "LSTM_HIGH",
    "LSTM_ALL",
    "FUNDING_CONTINUATION",
    "VOLATILITY_RESET",
    "CONTEXT_COMBINED",
    "FIRST_ONLY",
    "THIRD_PLUS_ONLY",
    "XGB_STRONG",
)
POLICY_DESCRIPTIONS: Final = {
    "UNION_ONLY": "Immutable Union only; admit no coverage candidate.",
    "LSTM_HIGH": "Admit every host-labelled HIGH_EXTRA LSTM candidate.",
    "LSTM_ALL": "Admit every host-labelled LSTM coverage candidate.",
    "FUNDING_CONTINUATION": (
        "Admit LONG LOW_EXTRA THIRD_PLUS candidates in POSITIVE funding."
    ),
    "VOLATILITY_RESET": (
        "Admit FIRST candidates in HIGH volatility, FLAT trend and FLAT OI."
    ),
    "CONTEXT_COMBINED": "Union of FUNDING_CONTINUATION and VOLATILITY_RESET.",
    "FIRST_ONLY": "Admit every FIRST candidate.",
    "THIRD_PLUS_ONLY": "Admit every THIRD_PLUS candidate.",
    "XGB_STRONG": (
        "Admit only raw XGBoost move >=0.80 and direction confidence >=0.80 "
        "when XGBoost agrees with the frozen candidate side."
    ),
}

_REQUIRED = {
    "opportunity_id",
    "route",
    "side",
    "decision_time",
    "entry_time",
    "outcome_available_time",
    "confidence_tier",
    "signal_run_bucket",
    "vol_regime",
    "trend_regime",
    "funding_regime",
    "oi_regime",
    "gross_return",
    "net_return",
    "round_trip_cost",
}


def validate_router_opportunities(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"router opportunities lack columns: {missing}")
    output = frame.copy().reset_index(drop=True)
    for column in ("decision_time", "entry_time", "outcome_available_time"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="raise")
    if output["opportunity_id"].duplicated().any():
        raise ValueError("router opportunity IDs must be unique")
    if not output["route"].isin({"UNION_BASE", "COVERAGE_CANDIDATE"}).all():
        raise ValueError("router received an unknown route")
    if not output["side"].isin({"LONG", "SHORT"}).all():
        raise ValueError("router received an unknown side")
    if output["outcome_available_time"].ge(Q2_START).any():
        raise ValueError("Q2 timestamp entered router opportunities")
    if not output["decision_time"].lt(output["outcome_available_time"]).all():
        raise ValueError("router outcome must be strictly later than its decision")
    observed_cost = output["gross_return"].astype(float) - output["net_return"].astype(float)
    if not np.allclose(
        observed_cost,
        output["round_trip_cost"].astype(float),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("router opportunity costs do not reconcile")
    return output


def attach_xgb_support(frame: pd.DataFrame, stage: str) -> tuple[pd.DataFrame, list[Path]]:
    """Attach raw, causal XGBoost support; missing panels never become votes."""

    output = validate_router_opportunities(frame)
    normalized = stage.strip().lower()
    panel_stage = {
        "development": "2024_oof",
        "h1": "h1",
        "forward": "forward",
    }.get(normalized)
    if panel_stage is None:
        raise ValueError("stage must be development, h1 or forward")
    panel, paths = load_xgb_panel(panel_stage)
    raw = decompose_xgb_probabilities(panel)
    raw["xgb_available"] = True
    raw["xgb_side"] = np.where(
        raw["p_long_given_move_raw"].ge(0.5), "LONG", "SHORT"
    )
    raw["xgb_direction_confidence"] = np.maximum(
        raw["p_long_given_move_raw"], 1.0 - raw["p_long_given_move_raw"]
    )
    support = raw.rename(columns={"p_move_raw": "xgb_p_move_raw"})[
        [
            "xgb_available",
            "xgb_side",
            "xgb_direction_confidence",
            "xgb_p_move_raw",
        ]
    ]
    output = output.join(support, on="decision_time")
    output["xgb_available"] = output["xgb_available"].fillna(False).astype(bool)
    return output, paths


def policy_mask(frame: pd.DataFrame, policy_id: str) -> pd.Series:
    if policy_id not in POLICY_IDS:
        raise ValueError(f"unknown router policy: {policy_id}")
    candidate = frame["route"].eq("COVERAGE_CANDIDATE")
    if "block_boundary_eligible" in frame.columns:
        candidate &= frame["block_boundary_eligible"].fillna(False).astype(bool)
    funding = (
        candidate
        & frame["side"].eq("LONG")
        & frame["confidence_tier"].eq("LOW_EXTRA")
        & frame["signal_run_bucket"].eq("THIRD_PLUS")
        & frame["funding_regime"].eq("POSITIVE")
    )
    volatility = (
        candidate
        & frame["signal_run_bucket"].eq("FIRST")
        & frame["vol_regime"].eq("HIGH")
        & frame["trend_regime"].eq("FLAT")
        & frame["oi_regime"].eq("FLAT")
    )
    if policy_id == "UNION_ONLY":
        return pd.Series(False, index=frame.index, dtype=bool)
    if policy_id == "LSTM_HIGH":
        return candidate & frame["confidence_tier"].eq("HIGH_EXTRA")
    if policy_id == "LSTM_ALL":
        return candidate
    if policy_id == "FUNDING_CONTINUATION":
        return funding
    if policy_id == "VOLATILITY_RESET":
        return volatility
    if policy_id == "CONTEXT_COMBINED":
        return funding | volatility
    if policy_id == "FIRST_ONLY":
        return candidate & frame["signal_run_bucket"].eq("FIRST")
    if policy_id == "THIRD_PLUS_ONLY":
        return candidate & frame["signal_run_bucket"].eq("THIRD_PLUS")
    required = {
        "xgb_available",
        "xgb_side",
        "xgb_p_move_raw",
        "xgb_direction_confidence",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"XGB_STRONG support columns are missing: {missing}")
    return (
        candidate
        & frame["xgb_available"].fillna(False).astype(bool)
        & frame["xgb_side"].eq(frame["side"])
        & pd.to_numeric(frame["xgb_p_move_raw"], errors="coerce").ge(0.80)
        & pd.to_numeric(frame["xgb_direction_confidence"], errors="coerce").ge(0.80)
    )


def _overlaps(
    interval: tuple[pd.Timestamp, pd.Timestamp],
    other: tuple[pd.Timestamp, pd.Timestamp],
) -> bool:
    return bool(interval[0] <= other[1] and interval[1] >= other[0])


def apply_router_policy(frame: pd.DataFrame, policy_id: str) -> pd.DataFrame:
    output = validate_router_opportunities(frame)
    eligible = policy_mask(output, policy_id)
    union = output["route"].eq("UNION_BASE")
    output["policy_id"] = policy_id
    output["policy_eligible"] = union | eligible
    output["selected"] = union
    output["skip_reason"] = ""
    output.loc[~union & ~eligible, "skip_reason"] = "POLICY_INELIGIBLE"
    if "block_boundary_eligible" in output.columns:
        boundary_veto = (
            ~union & ~output["block_boundary_eligible"].fillna(False).astype(bool)
        )
        output.loc[boundary_veto, "skip_reason"] = "BLOCK_BOUNDARY"
    union_intervals = [
        (pd.Timestamp(row.entry_time), pd.Timestamp(row.outcome_available_time))
        for row in output.loc[union].itertuples(index=False)
    ]
    admitted: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    candidates = output.loc[eligible].sort_values(
        ["entry_time", "decision_time", "opportunity_id"], kind="stable"
    )
    for row in candidates.itertuples():
        interval = (
            pd.Timestamp(row.entry_time),
            pd.Timestamp(row.outcome_available_time),
        )
        if any(_overlaps(interval, other) for other in union_intervals):
            output.at[row.Index, "skip_reason"] = "OVERLAP_UNION"
            continue
        if any(_overlaps(interval, other) for other in admitted):
            output.at[row.Index, "skip_reason"] = "OVERLAP_CANDIDATE"
            continue
        output.at[row.Index, "selected"] = True
        admitted.append(interval)
    output["action"] = "SKIP"
    output.loc[output["selected"] & output["side"].eq("LONG"), "action"] = "OPEN_LONG"
    output.loc[output["selected"] & output["side"].eq("SHORT"), "action"] = "OPEN_SHORT"
    if not output.loc[union, "selected"].all():
        raise AssertionError("a router policy changed the immutable Union")
    return output


__all__ = [
    "POLICY_DESCRIPTIONS",
    "POLICY_IDS",
    "Q2_START",
    "apply_router_policy",
    "attach_xgb_support",
    "policy_mask",
    "validate_router_opportunities",
]
