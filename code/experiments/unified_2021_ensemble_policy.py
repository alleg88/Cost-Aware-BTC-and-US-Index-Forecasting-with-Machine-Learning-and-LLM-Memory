"""Frozen opportunity/SIDE policy and economics for Notebook 04d."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary


MODEL_NAMES = ("xgboost", "lstm", "svm_linear")
REFRACTORY = pd.Timedelta(minutes=60)


@dataclass(frozen=True, order=True)
class EnsemblePolicy:
    daily_rate_cap: int
    long_regular_threshold: float
    short_regular_threshold: float
    xgb_solo_threshold: float


POLICY_GRID = tuple(
    EnsemblePolicy(rate, long_threshold, short_threshold, solo)
    for rate in (1, 2, 3)
    for long_threshold in (0.55, 0.60, 0.65)
    for short_threshold in (0.55, 0.60, 0.65)
    for solo in (0.70, 0.75)
)
assert len(POLICY_GRID) == 54


def _timedelta(value) -> pd.Timedelta:
    duration = pd.Timedelta(value)
    if duration < pd.Timedelta(0):
        raise ValueError("refractory must be non-negative")
    return duration


def _validate_scores(scores: pd.Series) -> pd.Series:
    if not isinstance(scores, pd.Series):
        raise TypeError("score selector accepts one pandas Series only")
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("scores need a DatetimeIndex")
    if scores.index.tz is None:
        raise ValueError("score timestamps must be timezone-aware")
    if not scores.index.is_monotonic_increasing or not scores.index.is_unique:
        raise ValueError("score timestamps must be unique and increasing")
    output = pd.to_numeric(scores, errors="coerce").astype(float)
    if not np.isfinite(output.to_numpy()).any():
        raise ValueError("scores contain no finite observations")
    return output


def causal_crossings(
    scores: pd.Series,
    threshold: float,
    refractory: str | pd.Timedelta = REFRACTORY,
) -> pd.DatetimeIndex:
    """Replay cold, below-threshold re-arming and refractory consumption."""
    values = _validate_scores(scores)
    cutoff = float(threshold)
    if not np.isfinite(cutoff):
        raise ValueError("threshold must be finite")
    duration = _timedelta(refractory)
    finite = values[np.isfinite(values.to_numpy())]
    if len(finite) < 2:
        return pd.DatetimeIndex([], tz=values.index.tz)
    raw = finite.to_numpy(float)
    # Cold start means the first observation cannot cross.  Thereafter the
    # first >= cutoff value after any < cutoff value is exactly a below-to-above
    # transition; a run of high values cannot re-arm itself.
    candidate_mask = (raw[1:] >= cutoff) & (raw[:-1] < cutoff)
    candidates = pd.DatetimeIndex(finite.index[1:][candidate_mask])
    if len(candidates) < 2:
        return candidates
    # Every candidate resets the refractory clock, even when rejected.  Hence
    # acceptance depends only on the gap from the immediately prior candidate.
    gap = np.diff(candidates.as_unit("ns").asi8)
    accepted = np.concatenate(
        ([True], gap >= int(duration.as_unit("ns").value))
    )
    return candidates[accepted]


def select_score_only_threshold(
    scores: pd.Series,
    daily_rate_cap: int,
    refractory: str | pd.Timedelta = REFRACTORY,
) -> tuple[float, pd.DataFrame]:
    """Choose the highest-count rate-bounded cutoff from scores and time only."""
    values = _validate_scores(scores)
    if int(daily_rate_cap) != daily_rate_cap or daily_rate_cap < 1:
        raise ValueError("daily_rate_cap must be a positive integer")
    finite = values[np.isfinite(values.to_numpy())]
    observed_days = int(finite.index.normalize().nunique())
    rows: list[dict[str, object]] = []
    for cutoff in np.sort(finite.unique()):
        crossings = causal_crossings(finite, float(cutoff), refractory)
        count = len(crossings)
        rows.append(
            {
                "threshold": float(cutoff),
                "crossings": count,
                "observed_days": observed_days,
                "crossings_per_observed_day": count / observed_days,
                "within_rate_cap": count / observed_days <= daily_rate_cap + 1e-12,
            }
        )
    frontier = pd.DataFrame(rows)
    eligible = frontier.loc[frontier["within_rate_cap"]]
    if eligible.empty:
        raise RuntimeError("no score cutoff satisfies the registered rate cap")
    chosen_index = eligible.sort_values(
        ["crossings", "threshold"], ascending=[False, False], kind="stable"
    ).index[0]
    frontier["selected"] = frontier.index == chosen_index
    return float(frontier.loc[chosen_index, "threshold"]), frontier


def _side_votes(row: Mapping[str, object], policy: EnsemblePolicy) -> dict[str, str | None]:
    votes: dict[str, str | None] = {}
    for model in MODEL_NAMES:
        probability = float(row[f"p_long_{model}"])
        if not np.isfinite(probability):
            votes[model] = None
        elif probability >= policy.long_regular_threshold:
            votes[model] = "long"
        elif 1.0 - probability >= policy.short_regular_threshold:
            votes[model] = "short"
        else:
            votes[model] = None
    return votes


def side_route(
    row: Mapping[str, object],
    policy: EnsemblePolicy,
    calibration_guard: Mapping[str, object] | None = None,
) -> tuple[str | None, str]:
    """Apply two-of-three SIDE voting and the strict XGBoost-only exception."""
    del calibration_guard  # No registered policy contains a relaxed threshold.
    votes = _side_votes(row, policy)
    counts = {
        side: sum(vote == side for vote in votes.values())
        for side in ("long", "short")
    }
    majority = next((side for side in ("long", "short") if counts[side] >= 2), None)
    if majority is not None:
        opposite = "short" if majority == "long" else "long"
        opposite_high = any(
            (
                1.0 - float(row[f"p_long_{model}"])
                if opposite == "short"
                else float(row[f"p_long_{model}"])
            )
            >= policy.xgb_solo_threshold
            for model in MODEL_NAMES
        )
        if opposite_high:
            return None, "opposite_veto"
        route = "unanimous" if counts[majority] == 3 else "two_of_three"
        return majority, route

    xgb_probability = float(row["p_long_xgboost"])
    solo_side: str | None = None
    if xgb_probability >= policy.xgb_solo_threshold:
        solo_side = "long"
    elif 1.0 - xgb_probability >= policy.xgb_solo_threshold:
        solo_side = "short"
    if solo_side is None:
        return None, "side_abstain"
    opposite = "short" if solo_side == "long" else "long"
    if any(votes[model] == opposite for model in ("lstm", "svm_linear")):
        return None, "opposite_veto"
    return solo_side, "xgboost_solo"


def _opportunity_score(frame: pd.DataFrame) -> pd.Series:
    columns = [f"p_opportunity_{model}" for model in MODEL_NAMES]
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"missing opportunity probabilities: {missing}")
    return frame[columns].apply(pd.to_numeric, errors="coerce").median(axis=1)


def _fold_values(frame: pd.DataFrame) -> list[int]:
    return (
        sorted(pd.to_numeric(frame["fold_id"], errors="raise").astype(int).unique())
        if "fold_id" in frame
        else [0]
    )


def _path_exit(row: Mapping[str, object], side: str) -> pd.Timestamp | pd.NaT:
    candidates = (
        f"actual_exit_time_{side}",
        f"exit_time_{side}",
    )
    for column in candidates:
        if column in row and pd.notna(row[column]):
            return pd.Timestamp(row[column])
    return pd.NaT


def apply_policy(
    predictions: pd.DataFrame,
    calibration_predictions: pd.DataFrame,
    policy: EnsemblePolicy,
    calibration_guard: Mapping[str, object] | None = None,
    *,
    opportunity_threshold: float | Mapping[int, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Freeze fold thresholds from calibration scores and replay decisions."""
    required = {"row_key", "decision_time", "entry_time"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"prediction frame missing: {missing}")
    scored = predictions.copy()
    scored["decision_time"] = pd.to_datetime(scored["decision_time"], utc=True)
    scored["ensemble_opportunity_score"] = _opportunity_score(scored)
    if "fold_id" not in scored:
        scored["fold_id"] = 0
    if opportunity_threshold is None:
        calibrated = calibration_predictions.copy()
        calibrated["decision_time"] = pd.to_datetime(
            calibrated["decision_time"], utc=True
        )
        calibrated["ensemble_opportunity_score"] = _opportunity_score(calibrated)
        if "fold_id" not in calibrated:
            calibrated["fold_id"] = 0
    else:
        calibrated = pd.DataFrame()

    activation_rows: list[dict[str, object]] = []
    funnel_rows: list[dict[str, object]] = []
    for fold_id in _fold_values(scored):
        fold = scored.loc[scored["fold_id"].eq(fold_id)].sort_values(
            "decision_time", kind="stable"
        )
        if opportunity_threshold is None:
            calibration_fold = calibrated.loc[
                calibrated["fold_id"].eq(fold_id)
            ].sort_values("decision_time", kind="stable")
            if calibration_fold.empty:
                raise ValueError(f"fold {fold_id} has no calibration predictions")
            calibration_scores = pd.Series(
                calibration_fold["ensemble_opportunity_score"].to_numpy(float),
                index=pd.DatetimeIndex(calibration_fold["decision_time"]),
                name="ensemble_opportunity_score",
            )
            threshold, _ = select_score_only_threshold(
                calibration_scores, policy.daily_rate_cap, REFRACTORY
            )
        else:
            threshold = float(
                opportunity_threshold[fold_id]
                if isinstance(opportunity_threshold, Mapping)
                else opportunity_threshold
            )
            if not np.isfinite(threshold):
                raise ValueError("opportunity_threshold must be finite")
        armed = False
        refractory_until: pd.Timestamp | None = None
        open_until: pd.Timestamp | None = None
        for row in fold.to_dict(orient="records"):
            timestamp = pd.Timestamp(row["decision_time"])
            score = float(row["ensemble_opportunity_score"])
            record = {
                **row,
                "opportunity_threshold": threshold,
                "armed_before": armed,
                "threshold_crossing": False,
                "raw_crossing": False,
                "refractory_rejection": False,
                "open_position_rejection": False,
                "selected_side": None,
                "route": None,
                "decision_reason": "not_rearmed",
            }
            if not np.isfinite(score):
                record["decision_reason"] = "missing_opportunity_score"
            elif score < threshold:
                armed = True
                record["decision_reason"] = "below_threshold_rearm"
            elif armed:
                armed = False
                record["threshold_crossing"] = True
                permitted = refractory_until is None or timestamp >= refractory_until
                refractory_until = timestamp + REFRACTORY
                if not permitted:
                    record["refractory_rejection"] = True
                    record["decision_reason"] = "refractory_rejection"
                else:
                    record["raw_crossing"] = True
                    side, route = side_route(row, policy, calibration_guard)
                    record["route"] = route
                    if side is None:
                        record["decision_reason"] = route
                    else:
                        entry_time = pd.Timestamp(row["entry_time"])
                        if open_until is not None and entry_time <= open_until:
                            record["open_position_rejection"] = True
                            record["decision_reason"] = "open_position_rejection"
                        else:
                            exit_time = _path_exit(row, side)
                            if pd.isna(exit_time):
                                record["decision_reason"] = "path_unavailable"
                            else:
                                open_until = pd.Timestamp(exit_time)
                                record["selected_side"] = side
                                record["decision_reason"] = route
                                activation_rows.append(record.copy())
            record["armed_after"] = armed
            record["refractory_until"] = refractory_until
            record["open_until"] = open_until
            funnel_rows.append(record)
    funnel = pd.DataFrame(funnel_rows).sort_values(
        "decision_time", kind="stable"
    ).reset_index(drop=True)
    activations = pd.DataFrame(activation_rows)
    if len(activations):
        activations = activations.sort_values(
            "decision_time", kind="stable"
        ).reset_index(drop=True)
    else:
        activations = funnel.iloc[0:0].copy()
    return activations, funnel


def replay_selected_paths(
    activations: pd.DataFrame, economic_paths: pd.DataFrame
) -> pd.DataFrame:
    """Join the selected paired path and enforce strict single-position replay."""
    if activations.empty:
        columns = list(dict.fromkeys(list(activations.columns) + list(economic_paths.columns)))
        return pd.DataFrame(columns=columns)
    required_activation = {"row_key", "selected_side"}
    required_path = {"row_key", "direction", "entry_time", "actual_exit_time"}
    if not required_activation.issubset(activations.columns):
        raise ValueError("activations need row_key and selected_side")
    if not required_path.issubset(economic_paths.columns):
        raise ValueError("economic paths lack replay columns")
    selected_paths = economic_paths.loc[
        economic_paths["row_key"].isin(activations["row_key"])
    ].copy()
    if "path_complete" in selected_paths:
        selected_paths = selected_paths.loc[
            selected_paths["path_complete"].fillna(False).astype(bool)
        ]
    joined = activations.merge(
        selected_paths,
        left_on=["row_key", "selected_side"],
        right_on=["row_key", "direction"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_path"),
    )
    if joined["entry_time_path" if "entry_time_path" in joined else "entry_time"].isna().any():
        raise AssertionError("an admitted activation has no complete selected path")
    for column in ("entry_time", "actual_exit_time"):
        path_column = f"{column}_path"
        if path_column in joined:
            joined[column] = joined[path_column]
    joined["entry_time"] = pd.to_datetime(joined["entry_time"], utc=True)
    joined["actual_exit_time"] = pd.to_datetime(joined["actual_exit_time"], utc=True)
    joined = joined.sort_values("entry_time", kind="stable")
    accepted: list[int] = []
    previous_exit: pd.Timestamp | None = None
    for index, row in joined.iterrows():
        if previous_exit is not None and row["entry_time"] <= previous_exit:
            continue
        accepted.append(index)
        previous_exit = row["actual_exit_time"]
    return joined.loc[accepted].reset_index(drop=True)


def _bar_frame(bars: pd.DataFrame | pd.Series) -> pd.DataFrame:
    if isinstance(bars, pd.Series):
        frame = bars.rename("close").to_frame()
    else:
        frame = bars.copy()
    if "close" not in frame:
        raise ValueError("bars need a close column")
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("bars need a timezone-aware DatetimeIndex")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError("bar index must be unique and increasing")
    return frame


def _direction_sign(row: Mapping[str, object]) -> float:
    side = row.get("direction", row.get("selected_side", row.get("side")))
    if side in ("long", 1, 1.0):
        return 1.0
    if side in ("short", -1, -1.0):
        return -1.0
    raise ValueError(f"unknown trade direction: {side!r}")


def ledger_to_common_per_bar(
    ledger: pd.DataFrame,
    bars: pd.DataFrame | pd.Series,
) -> pd.Series:
    """Transform native-entry trades to the project's additive M15 equity clock."""
    frame = _bar_frame(bars)
    output = np.zeros(len(frame), dtype=float)
    if ledger.empty:
        return pd.Series(output, index=frame.index, name="candidate_return")
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(float)
    if not np.isfinite(close).all():
        raise ValueError("bar closes must be finite")
    index = pd.DatetimeIndex(frame.index)
    for row in ledger.to_dict(orient="records"):
        entry_time = pd.Timestamp(row["entry_time"])
        exit_value = row.get("actual_exit_time", row.get("exit_time"))
        exit_time = pd.Timestamp(exit_value)
        entry_position = int(index.searchsorted(entry_time, side="right") - 1)
        exit_position = int(index.searchsorted(exit_time, side="right") - 1)
        if entry_position < 0 or exit_position < entry_position or exit_position >= len(index):
            raise ValueError("trade timestamps fall outside the supplied M15 clock")
        entry_price = float(row["entry_price"])
        exit_price = float(row["exit_price"])
        sign = _direction_sign(row)
        marks = np.concatenate(
            ([entry_price], close[entry_position:exit_position], [exit_price])
        )
        increments = sign * np.diff(marks) / entry_price
        output[entry_position : exit_position + 1] += increments
        if "entry_cost_bps" in row or "exit_cost_bps" in row:
            entry_cost = float(row.get("entry_cost_bps", 5.0)) / 10_000.0
            exit_cost = float(row.get("exit_cost_bps", 5.0)) / 10_000.0
        else:
            total_cost = float(row.get("cost_bps", 10.0)) / 10_000.0
            entry_cost = total_cost / 2.0
            exit_cost = total_cost / 2.0
        output[entry_position] -= entry_cost
        output[exit_position] -= exit_cost
    result = pd.Series(output, index=index, name="candidate_return")
    if "net_return" in ledger:
        expected = pd.to_numeric(ledger["net_return"], errors="raise").sum()
        if not np.isclose(result.sum(), expected, rtol=0.0, atol=1e-9):
            raise AssertionError(
                f"per-bar return {result.sum():.12f} does not reconcile with ledger {expected:.12f}"
            )
    return result


def _longest_losing_run(values: Iterable[float]) -> int:
    longest = 0
    current = 0
    for value in values:
        current = current + 1 if value < 0.0 else 0
        longest = max(longest, current)
    return longest


def summarize_candidate(
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    phase: str,
) -> dict[str, object]:
    """Report common economics plus mandatory side and calendar diagnostics."""
    economics = economics_summary(per_bar)
    if ledger.empty:
        direction = pd.Series(dtype="string")
        trade_return = pd.Series(dtype=float)
        entry_time = pd.Series(dtype="datetime64[ns, UTC]")
    else:
        direction_column = next(
            column for column in ("direction", "selected_side", "side") if column in ledger
        )
        direction = ledger[direction_column].replace({1: "long", -1: "short"}).astype(str)
        trade_return = pd.to_numeric(ledger["net_return"], errors="coerce").fillna(0.0)
        entry_time = pd.to_datetime(ledger["entry_time"], utc=True)
    long_mask = direction.eq("long")
    short_mask = direction.eq("short")
    apr_jun = entry_time.dt.month.between(4, 6) if len(entry_time) else pd.Series(dtype=bool)
    observed_quarters = set(
        pd.DatetimeIndex(per_bar.index).tz_convert(None).to_period("Q").astype(str)
    )
    if len(entry_time):
        trade_quarters = entry_time.dt.tz_convert(None).dt.to_period("Q").astype(str)
        both_sides_by_quarter = all(
            bool((trade_quarters.eq(quarter) & long_mask).any())
            and bool((trade_quarters.eq(quarter) & short_mask).any())
            for quarter in observed_quarters
        )
    else:
        both_sides_by_quarter = False
    summary: dict[str, object] = {
        "phase": phase,
        "trades": int(len(ledger)),
        "long_trades": int(long_mask.sum()),
        "short_trades": int(short_mask.sum()),
        "net_return": float(economics["net_return_sum"]),
        "gross_return": float(
            pd.to_numeric(ledger.get("gross_return", pd.Series(dtype=float)), errors="coerce").sum()
        ),
        "long_net_return": float(trade_return.loc[long_mask].sum()),
        "short_net_return": float(trade_return.loc[short_mask].sum()),
        "sharpe": float(economics["sharpe"]),
        "sortino": float(economics["sortino"]),
        "max_drawdown": float(economics["max_drawdown"]),
        "apr_jun_net_return": float(trade_return.loc[apr_jun].sum()) if len(ledger) else 0.0,
        "apr_jun_long_trades": int((apr_jun & long_mask).sum()) if len(ledger) else 0,
        "apr_jun_short_trades": int((apr_jun & short_mask).sum()) if len(ledger) else 0,
        "every_quarter_has_long_and_short": bool(both_sides_by_quarter),
        "longest_losing_run": _longest_losing_run(trade_return.to_numpy(float)),
    }
    if "outcome" in ledger:
        outcomes = ledger["outcome"].astype(str)
        summary["take_profit_rate"] = float(outcomes.eq("take_profit").mean())
        summary["stop_loss_rate"] = float(outcomes.eq("stop_loss").mean())
        summary["timeout_rate"] = float(outcomes.eq("timeout").mean())
    return summary


def evaluate_oof_policy_grid(oof, economic_paths: pd.DataFrame) -> tuple[pd.DataFrame, EnsemblePolicy | None]:
    """Select the trade-maximising policy after mandatory two-sided OOF guards."""
    predictions = oof.predictions
    calibration = oof.calibration_predictions
    fold_ids = sorted(pd.to_numeric(predictions["fold_id"], errors="raise").astype(int).unique())
    threshold_cache: dict[int, dict[int, float]] = {}
    calibration_scored = calibration.copy()
    calibration_scored["decision_time"] = pd.to_datetime(
        calibration_scored["decision_time"], utc=True
    )
    calibration_scored["ensemble_opportunity_score"] = _opportunity_score(
        calibration_scored
    )
    for rate_cap in (1, 2, 3):
        threshold_cache[rate_cap] = {}
        for fold_id in fold_ids:
            fold = calibration_scored.loc[
                calibration_scored["fold_id"].eq(fold_id)
            ].sort_values("decision_time", kind="stable")
            scores = pd.Series(
                fold["ensemble_opportunity_score"].to_numpy(float),
                index=pd.DatetimeIndex(fold["decision_time"]),
            )
            threshold_cache[rate_cap][fold_id], _ = select_score_only_threshold(
                scores, rate_cap
            )
    rows: list[dict[str, object]] = []
    for policy in POLICY_GRID:
        activations, funnel = apply_policy(
            predictions,
            calibration,
            policy,
            opportunity_threshold=threshold_cache[policy.daily_rate_cap],
        )
        ledger = replay_selected_paths(activations, economic_paths)
        if ledger.empty:
            direction = pd.Series(dtype="string")
            trade_return = pd.Series(dtype=float)
            ledger_folds = pd.Series(dtype=int)
        else:
            direction = ledger["direction"].astype(str)
            trade_return = pd.to_numeric(ledger["net_return"], errors="coerce").fillna(0.0)
            ledger_folds = pd.to_numeric(ledger["fold_id"], errors="raise").astype(int)
        side_net = {
            side: float(trade_return.loc[direction.eq(side)].sum())
            for side in ("long", "short")
        }
        positive_folds = {
            side: sum(
                float(
                    trade_return.loc[
                        direction.eq(side) & ledger_folds.eq(fold_id)
                    ].sum()
                )
                > 0.0
                for fold_id in fold_ids
            )
            for side in ("long", "short")
        }
        qualifies = bool(
            side_net["long"] > 0.0
            and side_net["short"] > 0.0
            and positive_folds["long"] >= 3
            and positive_folds["short"] >= 3
        )
        rows.append(
            {
                "daily_rate_cap": policy.daily_rate_cap,
                "long_regular_threshold": policy.long_regular_threshold,
                "short_regular_threshold": policy.short_regular_threshold,
                "xgb_solo_threshold": policy.xgb_solo_threshold,
                "trades": len(ledger),
                "long_trades": int(direction.eq("long").sum()),
                "short_trades": int(direction.eq("short").sum()),
                "net_return": float(trade_return.sum()),
                "long_net_return": side_net["long"],
                "short_net_return": side_net["short"],
                "long_positive_folds": positive_folds["long"],
                "short_positive_folds": positive_folds["short"],
                "raw_crossings": int(funnel["raw_crossing"].sum()),
                "side_abstentions": int(funnel["decision_reason"].eq("side_abstain").sum()),
                "qualifies": qualifies,
            }
        )
    grid = pd.DataFrame(rows)
    qualified = grid.loc[grid["qualifies"]]
    grid["selected"] = False
    if qualified.empty:
        return grid, None
    chosen_index = qualified.sort_values(
        [
            "trades",
            "net_return",
            "daily_rate_cap",
            "long_regular_threshold",
            "short_regular_threshold",
            "xgb_solo_threshold",
        ],
        ascending=[False, False, True, False, False, False],
        kind="stable",
    ).index[0]
    grid.loc[chosen_index, "selected"] = True
    chosen = grid.loc[chosen_index]
    return grid, EnsemblePolicy(
        int(chosen["daily_rate_cap"]),
        float(chosen["long_regular_threshold"]),
        float(chosen["short_regular_threshold"]),
        float(chosen["xgb_solo_threshold"]),
    )


def h1_compatibility_gate(summary: Mapping[str, object]) -> bool:
    return bool(
        int(summary["trades"]) >= 132
        and int(summary["long_trades"]) >= 15
        and int(summary["short_trades"]) >= 15
        and float(summary["net_return"]) >= 0.009687734009
        and float(summary["long_net_return"]) >= 0.0
        and float(summary["short_net_return"]) >= 0.0
        and float(summary["apr_jun_net_return"]) >= 0.0
        and int(summary["apr_jun_long_trades"]) >= 1
        and int(summary["apr_jun_short_trades"]) >= 1
        and float(summary["sortino"]) >= 0.3521029257
        and float(summary["max_drawdown"]) <= 0.047613386266
    )


def forward_promotion_gate(summary: Mapping[str, object]) -> bool:
    return bool(
        int(summary["trades"]) >= 111
        and int(summary["long_trades"]) >= 15
        and int(summary["short_trades"]) >= 15
        and float(summary["net_return"]) >= 0.063149608713
        and float(summary["long_net_return"]) >= 0.0
        and float(summary["short_net_return"]) >= 0.0
        and float(summary["sortino"]) >= 2.1060930178
        and float(summary["max_drawdown"]) <= 0.029796677136
        and bool(summary["every_quarter_has_long_and_short"])
    )


__all__ = [
    "EnsemblePolicy",
    "POLICY_GRID",
    "apply_policy",
    "causal_crossings",
    "evaluate_oof_policy_grid",
    "forward_promotion_gate",
    "h1_compatibility_gate",
    "ledger_to_common_per_bar",
    "replay_selected_paths",
    "select_score_only_threshold",
    "side_route",
    "summarize_candidate",
]
