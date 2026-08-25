"""Causal policy primitives for the Notebook 04c opportunity add-on."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AddonConfig:
    """Frozen policy and execution settings for the 04c experiment."""

    target_activations_per_day: float = 3.0
    refractory_minutes: int = 60
    hold_minutes: int = 120
    target_multiple_b: float = 2.0
    entry_cost_bps: float = 5.0
    exit_cost_bps: float = 5.0


def _validated_scores(scores: pd.Series) -> pd.Series:
    if not isinstance(scores, pd.Series):
        raise TypeError("scores must be a pandas Series")
    if not isinstance(scores.index, pd.DatetimeIndex):
        raise TypeError("scores must use a DatetimeIndex")
    if scores.empty:
        raise ValueError("scores must not be empty")
    index = pd.DatetimeIndex(pd.to_datetime(scores.index, utc=True, errors="raise"))
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("score timestamps must be unique and increasing")
    values = pd.to_numeric(scores, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("scores must be finite")
    return pd.Series(values, index=index, name="opportunity_score")


def causal_crossings(
    scores: pd.Series,
    *,
    threshold: float,
    refractory: pd.Timedelta,
) -> pd.DatetimeIndex:
    """Select cold-start, re-armed score crossings under a global cooldown."""
    series = _validated_scores(scores)
    threshold = float(threshold)
    refractory = pd.Timedelta(refractory)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    if refractory <= pd.Timedelta(0):
        raise ValueError("refractory must be positive")

    armed = False
    last_activation: pd.Timestamp | None = None
    selected: list[pd.Timestamp] = []
    for timestamp, score in series.items():
        if float(score) < threshold:
            armed = True
            continue
        if not armed:
            continue
        armed = False
        if (
            last_activation is not None
            and timestamp - last_activation < refractory
        ):
            continue
        selected.append(timestamp)
        last_activation = timestamp
    return pd.DatetimeIndex(selected, tz="UTC")


def threshold_frontier(
    scores: pd.Series,
    config: AddonConfig = AddonConfig(),
) -> pd.DataFrame:
    """Evaluate every observed score threshold without reading outcome fields."""
    series = _validated_scores(scores)
    if config.target_activations_per_day <= 0.0:
        raise ValueError("target_activations_per_day must be positive")
    refractory = pd.Timedelta(minutes=config.refractory_minutes)
    observed_days = int(series.index.normalize().nunique())
    differences = series.index.to_series().diff().dropna()
    if differences.empty or not differences.eq(pd.Timedelta(minutes=5)).all():
        raise ValueError("threshold frontier requires a complete five-minute grid")
    cooldown_steps = int(np.ceil(refractory / pd.Timedelta(minutes=5)))
    if cooldown_steps <= 0:
        raise ValueError("refractory must span at least one five-minute bar")

    values = series.to_numpy(float)
    thresholds = np.unique(values)
    ring = np.zeros(
        (cooldown_steps + 1, len(thresholds)),
        dtype=np.int32,
    )
    for position in range(len(values) - 1, 0, -1):
        crossing = (
            (values[position - 1] < thresholds)
            & (values[position] >= thresholds)
        )
        ring[position % len(ring)] = np.where(
            crossing,
            1 + ring[(position + cooldown_steps) % len(ring)],
            ring[(position + 1) % len(ring)],
        )
    counts = ring[1 % len(ring)].copy()
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "activations": counts,
            "observed_utc_days": observed_days,
            "activations_per_day": counts / observed_days,
            "selected": False,
        }
    )


def select_frequency_threshold(
    scores: pd.Series,
    config: AddonConfig = AddonConfig(),
) -> tuple[float, pd.DataFrame]:
    """Choose maximum causal activity below the frozen daily-rate ceiling."""
    frontier = threshold_frontier(scores, config)
    eligible = frontier.loc[
        frontier["activations_per_day"].le(config.target_activations_per_day)
    ].sort_values(["activations", "threshold"], ascending=[False, False])
    if eligible.empty:
        raise ValueError("no score threshold satisfies the registered rate")
    threshold = float(eligible.iloc[0]["threshold"])
    frontier.loc[frontier["threshold"].eq(threshold), "selected"] = True
    return threshold, frontier


def h1_access_gate(summary: Mapping[str, object]) -> bool:
    """Return whether H1 evidence permits any forward input to be opened."""
    return bool(
        float(summary["addon_net_return"]) > 0.0
        and int(summary["addon_trades"]) >= 20
        and int(summary["addon_long_trades"]) >= 1
        and int(summary["addon_short_trades"]) >= 1
        and float(summary["apr_jun_addon_net_return"]) >= 0.0
    )


def forward_promotion_gate(summary: Mapping[str, object]) -> bool:
    """Return whether the additive forward evidence promotes Union v2."""
    return bool(
        float(summary["addon_net_return"]) > 0.0
        and int(summary["addon_long_trades"]) >= 1
        and int(summary["addon_short_trades"]) >= 1
        and int(summary["combined_trades"]) > 74
        and float(summary["combined_net_return"])
        > float(summary["union_net_return"])
    )


def align_union_asof(
    activations: pd.DataFrame,
    union_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Attach only M15 Union rows that were tradable by each 5m decision."""
    required = {
        "union_signal",
        "member_conflict",
        "lstm_latent_side",
        "svm_linear_latent_side",
    }
    missing = sorted(required.difference(union_signals.columns))
    if missing:
        raise ValueError(f"Union signals missing columns: {missing}")
    if "decision_time" not in activations:
        raise ValueError("activations missing decision_time")

    left = activations.copy()
    left["decision_time"] = pd.to_datetime(
        left["decision_time"], utc=True, errors="raise"
    )
    if left["decision_time"].duplicated().any():
        raise ValueError("activation decision_time must be unique")

    right = union_signals.copy()
    if "timestamp" in right:
        right["union_timestamp"] = pd.to_datetime(
            right.pop("timestamp"), utc=True, errors="raise"
        )
    elif isinstance(right.index, pd.DatetimeIndex):
        right = right.reset_index(names="union_timestamp")
        right["union_timestamp"] = pd.to_datetime(
            right["union_timestamp"], utc=True, errors="raise"
        )
    else:
        raise ValueError("Union signals require timestamp column or index")
    right["availability_time"] = right["union_timestamp"] + pd.Timedelta(
        minutes=15
    )
    if right["union_timestamp"].duplicated().any():
        raise ValueError("Union timestamps must be unique")

    aligned = pd.merge_asof(
        left.sort_values("decision_time"),
        right.sort_values("availability_time"),
        left_on="decision_time",
        right_on="availability_time",
        direction="backward",
        allow_exact_matches=True,
    )
    available = aligned["union_timestamp"].notna()
    if not (
        aligned.loc[available, "availability_time"]
        <= aligned.loc[available, "decision_time"]
    ).all():
        raise AssertionError("future Union information entered an activation")
    return aligned


def _union_open_mask(
    decision_times: pd.Series,
    union_ledger: pd.DataFrame,
) -> np.ndarray:
    if union_ledger.empty:
        return np.zeros(len(decision_times), dtype=bool)
    if "entry_time" not in union_ledger:
        raise ValueError("Union ledger missing entry_time")
    exit_column = (
        "intrabar_exit_time"
        if "intrabar_exit_time" in union_ledger
        else "exit_time"
    )
    if exit_column not in union_ledger:
        raise ValueError("Union ledger missing an actual exit timestamp")
    starts = pd.DatetimeIndex(
        pd.to_datetime(union_ledger["entry_time"], utc=True, errors="raise")
    ).as_unit("ns").asi8
    ends = pd.DatetimeIndex(
        pd.to_datetime(union_ledger[exit_column], utc=True, errors="raise")
    ).as_unit("ns").asi8
    output = []
    for timestamp in pd.to_datetime(decision_times, utc=True, errors="raise"):
        value = timestamp.value
        output.append(bool(np.any((starts <= value) & (value <= ends))))
    return np.asarray(output, dtype=bool)


def qualify_union_side(
    aligned: pd.DataFrame,
    union_ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the frozen Union-flat and latent-side agreement gate."""
    required = {
        "decision_time",
        "union_signal",
        "member_conflict",
        "lstm_latent_side",
        "svm_linear_latent_side",
    }
    missing = sorted(required.difference(aligned.columns))
    if missing:
        raise ValueError(f"aligned activations missing columns: {missing}")
    output = aligned.copy()
    output["decision_time"] = pd.to_datetime(
        output["decision_time"], utc=True, errors="raise"
    )
    union_open = _union_open_mask(output["decision_time"], union_ledger)
    lstm = pd.to_numeric(output["lstm_latent_side"], errors="coerce").to_numpy(float)
    svm = pd.to_numeric(
        output["svm_linear_latent_side"], errors="coerce"
    ).to_numpy(float)
    finite_agreement = (
        np.isfinite(lstm)
        & np.isfinite(svm)
        & (lstm != 0.0)
        & (svm != 0.0)
        & (lstm == svm)
    )
    decision: list[str] = []
    side = np.zeros(len(output), dtype=np.int8)
    for position, row in enumerate(output.itertuples(index=False)):
        union_timestamp = getattr(row, "union_timestamp", pd.NaT)
        if "union_timestamp" in output and pd.isna(union_timestamp):
            reason = "reject_no_available_union"
        elif not np.isclose(float(row.union_signal), 0.0):
            reason = "reject_union_active"
        elif bool(row.member_conflict):
            reason = "reject_member_conflict"
        elif union_open[position]:
            reason = "reject_union_open"
        elif not finite_agreement[position]:
            reason = "reject_side_disagreement"
        else:
            reason = "accept"
            side[position] = int(np.sign(lstm[position]))
        decision.append(reason)
    output["decision"] = decision
    output["side"] = side
    return output


def select_non_overlapping_addons(paths: pd.DataFrame) -> pd.DataFrame:
    """Greedily retain causal add-ons using each accepted path's actual exit."""
    if paths.empty:
        output = paths.copy()
        output["actual_exit_time"] = pd.Series(dtype="datetime64[ns, UTC]")
        return output
    required = {"decision_time", "bars_held"}
    missing = sorted(required.difference(paths.columns))
    if missing:
        raise ValueError(f"add-on paths missing columns: {missing}")
    work = paths.copy()
    work["decision_time"] = pd.to_datetime(
        work["decision_time"], utc=True, errors="raise"
    )
    if "path_complete" in work:
        work = work.loc[work["path_complete"].astype(bool)].copy()
    bars = pd.to_numeric(work["bars_held"], errors="coerce")
    if bars.isna().any() or bars.le(0).any():
        raise ValueError("complete paths require positive bars_held")
    work["actual_exit_time"] = work["decision_time"] + pd.to_timedelta(
        bars.astype(int) - 1, unit="min"
    )
    accepted: list[int] = []
    last_exit: pd.Timestamp | None = None
    for index, row in work.sort_values("decision_time").iterrows():
        if last_exit is not None and row["decision_time"] <= last_exit:
            continue
        accepted.append(index)
        last_exit = pd.Timestamp(row["actual_exit_time"])
    return work.loc[accepted].sort_values("decision_time").reset_index(drop=True)


def replay_addons(
    qualified: pd.DataFrame,
    minute: pd.DataFrame,
    config: AddonConfig = AddonConfig(),
) -> pd.DataFrame:
    """Replay agreed sides on W's native RR2/120m geometry."""
    accepted = (
        qualified.loc[qualified["decision"].eq("accept")].copy()
        if "decision" in qualified
        else qualified.copy()
    )
    if accepted.empty:
        return pd.DataFrame(
            columns=[
                "decision_time",
                "actual_exit_time",
                "side",
                "gross_return",
                "net_return",
                "net_bps",
                "net_r",
            ]
        )
    required = {
        "decision_time",
        "reference_price",
        "adaptive_barrier_bps",
        "side",
    }
    missing = sorted(required.difference(accepted.columns))
    if missing:
        raise ValueError(f"qualified activations missing columns: {missing}")
    accepted = accepted.sort_values("decision_time").reset_index(drop=True)
    accepted["activation_key"] = accepted["decision_time"].astype(str)
    accepted["window_id"] = accepted["activation_key"]
    accepted["step"] = 0
    accepted["channel_episode_id"] = accepted["activation_key"]
    accepted["channel_side"] = accepted["side"].map({1: "long", -1: "short"})
    if accepted["channel_side"].isna().any():
        raise ValueError("qualified side must be LONG or SHORT")

    from experiments.run_event_window_economic_feasibility import replay_brackets

    paths = replay_brackets(
        accepted,
        minute,
        target_multiples=(config.target_multiple_b,),
        hold_minutes=(config.hold_minutes,),
        entry_cost_bps=config.entry_cost_bps,
        target_exit_cost_bps=config.exit_cost_bps,
        other_exit_cost_bps=config.exit_cost_bps,
    )
    paths = paths.loc[
        paths["direction"].eq(paths["channel_side"])
        & paths["path_complete"].astype(bool)
    ].copy()
    sign = np.where(paths["direction"].eq("long"), 1.0, -1.0)
    paths["gross_return"] = sign * (
        paths["exit_price"].astype(float) / paths["entry_price"].astype(float)
        - 1.0
    )
    paths["net_return"] = paths["gross_return"] - (
        config.entry_cost_bps + config.exit_cost_bps
    ) / 10_000.0
    return select_non_overlapping_addons(paths)


def build_portfolio_series(
    union_per_bar: pd.Series,
    addon_ledger: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Place add-on returns on the immutable Union M15 grid and sum the arms."""
    if not isinstance(union_per_bar.index, pd.DatetimeIndex):
        raise TypeError("Union per-bar returns require a DatetimeIndex")
    union = pd.to_numeric(union_per_bar, errors="raise").astype(float).copy()
    union.index = pd.to_datetime(union.index, utc=True, errors="raise")
    addon = pd.Series(0.0, index=union.index, name="addon_net_return")
    if not addon_ledger.empty:
        required = {"decision_time", "net_return"}
        missing = sorted(required.difference(addon_ledger.columns))
        if missing:
            raise ValueError(f"add-on ledger missing columns: {missing}")
        buckets = pd.to_datetime(
            addon_ledger["decision_time"], utc=True, errors="raise"
        ).dt.floor("15min")
        realised = pd.Series(
            pd.to_numeric(addon_ledger["net_return"], errors="raise").to_numpy(float),
            index=pd.DatetimeIndex(buckets),
        ).groupby(level=0).sum()
        outside = realised.index.difference(addon.index)
        if len(outside):
            raise ValueError("add-on return falls outside the Union grid")
        addon.loc[realised.index] = realised
    combined = union.add(addon, fill_value=0.0).rename("combined_net_return")
    if not np.isclose(addon.sum(), pd.to_numeric(
        addon_ledger.get("net_return", pd.Series(dtype=float)),
        errors="coerce",
    ).sum(), atol=1e-12):
        raise AssertionError("add-on per-bar series does not reconcile")
    return addon, combined


def _minute_occupancy(
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]],
    index: pd.DatetimeIndex,
) -> pd.Series:
    values = pd.Series(0, index=index, dtype=np.int16)
    for start, end in intervals:
        values.loc[(values.index >= start) & (values.index <= end)] += 1
    return values


def concurrency_audit(
    union_ledger: pd.DataFrame,
    addon_ledger: pd.DataFrame,
) -> dict[str, object]:
    """Report overlap while preserving later Union trades."""
    intervals: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []
    if not union_ledger.empty:
        exit_column = (
            "intrabar_exit_time"
            if "intrabar_exit_time" in union_ledger
            else "exit_time"
        )
        for start, end in zip(
            pd.to_datetime(union_ledger["entry_time"], utc=True, errors="raise"),
            pd.to_datetime(union_ledger[exit_column], utc=True, errors="raise"),
            strict=True,
        ):
            intervals.append(("union", start.floor("min"), end.floor("min")))
    if not addon_ledger.empty:
        for start, end in zip(
            pd.to_datetime(addon_ledger["decision_time"], utc=True, errors="raise"),
            pd.to_datetime(
                addon_ledger["actual_exit_time"], utc=True, errors="raise"
            ),
            strict=True,
        ):
            intervals.append(("addon", start.floor("min"), end.floor("min")))
    if not intervals:
        return {
            "max_gross_exposure": 0,
            "union_addon_overlap_minutes": 0,
        }
    index = pd.date_range(
        min(start for _, start, _ in intervals),
        max(end for _, _, end in intervals),
        freq="1min",
        tz="UTC",
    )
    union = _minute_occupancy(
        [(start, end) for arm, start, end in intervals if arm == "union"],
        index,
    )
    addon = _minute_occupancy(
        [(start, end) for arm, start, end in intervals if arm == "addon"],
        index,
    )
    return {
        "max_gross_exposure": int((union + addon).max()),
        "union_addon_overlap_minutes": int(((union > 0) & (addon > 0)).sum()),
    }


__all__ = [
    "AddonConfig",
    "align_union_asof",
    "build_portfolio_series",
    "causal_crossings",
    "concurrency_audit",
    "forward_promotion_gate",
    "h1_access_gate",
    "qualify_union_side",
    "replay_addons",
    "select_non_overlapping_addons",
    "select_frequency_threshold",
    "threshold_frontier",
]
