"""Protected-frequency economic policy for event-window tail models.

The primary policy is fixed: take the first geometry-valid decision whose
calibrated expected value is non-negative.  Frequency protection is expressed
with the frozen integer count and calendar, never a rounded display rate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from evaluation.event_window_economics import replay_first_crossing


REFERENCE_ATTEMPTS = 1448
REFERENCE_CALENDAR_DAYS = 1277
REFERENCE_START = pd.Timestamp("2022-01-01", tz="UTC")
REFERENCE_END = pd.Timestamp("2025-07-01", tz="UTC")
MODEL_SIMPLICITY = ("logreg", "xgboost", "gru", "tcn")


@dataclass(frozen=True)
class TailPolicyResult:
    model_name: str
    trades: pd.DataFrame
    daily: pd.DataFrame
    attempted_trades: int
    observed_trades: int
    trades_per_day: float
    mean_net_r: float
    total_net_r: float
    frequency_noninferior: bool


@dataclass(frozen=True)
class MatchedCountResult(TailPolicyResult):
    threshold: float
    target_attempts: int
    exploratory: bool = True
    can_select: bool = False


@dataclass(frozen=True)
class EpisodeBootstrapResult:
    point_delta_total_net_r: float
    ci_low: float
    ci_high: float
    draws: pd.Series


@dataclass(frozen=True)
class TailModelComparison:
    strongest_comparator: str | None
    chosen_model: str | None
    passes: dict[str, bool]
    rejection_reasons: dict[str, tuple[str, ...]]
    table: pd.DataFrame


def frequency_noninferior(attempts: int, calendar_days: int) -> bool:
    """Apply the frozen integer frequency gate exactly."""
    return bool(
        int(calendar_days) == REFERENCE_CALENDAR_DAYS
        and int(attempts) >= REFERENCE_ATTEMPTS
    )


def _score_frame(scores: Any) -> Any:
    if not isinstance(scores, pd.DataFrame):
        return scores
    if "ev_score" not in scores.columns:
        if "score" not in scores.columns:
            raise ValueError("scores must contain ev_score or score")
        return scores
    if "score" in scores.columns:
        raise ValueError("scores cannot contain both ev_score and score")
    return scores.rename(columns={"ev_score": "score"})


def _model_name(scores: Any, fallback: str) -> str:
    if isinstance(scores, pd.DataFrame):
        if "model" in scores.columns:
            names = scores["model"].dropna().astype(str).unique()
            if len(names) == 1:
                return str(names[0])
        name = scores.attrs.get("model_name")
        if name is not None:
            return str(name)
    return fallback


def _window_universe(labels: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    required = {"window_id", "channel_episode_id", "entry_time"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"labels missing universe columns: {missing}")
    start_time = pd.Timestamp(start)
    end_time = pd.Timestamp(end)
    start_time = (
        start_time.tz_localize("UTC")
        if start_time.tzinfo is None
        else start_time.tz_convert("UTC")
    )
    end_time = (
        end_time.tz_localize("UTC")
        if end_time.tzinfo is None
        else end_time.tz_convert("UTC")
    )
    work = labels[["window_id", "channel_episode_id", "entry_time"]].copy()
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="raise")
    work = work[work["entry_time"].ge(start_time) & work["entry_time"].lt(end_time)]
    episode_counts = work.groupby("window_id")["channel_episode_id"].nunique(dropna=False)
    if episode_counts.gt(1).any():
        raise ValueError("a window cannot span multiple channel episodes")
    return work[["window_id", "channel_episode_id"]].drop_duplicates("window_id")


def replay_positive_ev(
    scores: Any,
    labels: pd.DataFrame,
    calendar_start: Any,
    calendar_end: Any,
    *,
    model_name: str = "model",
) -> TailPolicyResult:
    """Replay the immutable first non-negative-EV policy."""
    replay = replay_first_crossing(
        _score_frame(scores),
        labels,
        threshold=0.0,
        start=calendar_start,
        end=calendar_end,
    )
    attempted = int(replay.summary["attempted_trades"])
    observed = int(replay.summary["observed_trades"])
    calendar_days = len(replay.daily_frequency)
    trades = replay.trades.copy()
    trades.attrs["window_universe"] = _window_universe(
        labels, calendar_start, calendar_end
    )
    return TailPolicyResult(
        model_name=_model_name(scores, model_name),
        trades=trades,
        daily=replay.daily_frequency.copy(),
        attempted_trades=attempted,
        observed_trades=observed,
        trades_per_day=float(attempted / calendar_days) if calendar_days else 0.0,
        mean_net_r=float(replay.summary["mean_net_r"]),
        total_net_r=float(replay.summary["total_net_r"]),
        frequency_noninferior=frequency_noninferior(attempted, len(replay.daily_frequency)),
    )


def replay_same_entries(entries: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Replay RR3 labels on the exact frozen RR2 ``(window_id, step)`` keys."""
    keys = ["window_id", "step"]
    for name, frame in (("entries", entries), ("labels", labels)):
        missing = sorted(set(keys).difference(frame.columns))
        if missing:
            raise ValueError(f"{name} missing entry keys: {missing}")
    if entries.duplicated(keys).any():
        raise ValueError("entries must be unique by window_id and step")
    if labels.duplicated(keys).any():
        raise ValueError("labels must be unique by window_id and step")
    selected = entries[keys].copy()
    selected["_entry_order"] = np.arange(len(selected), dtype=np.int64)
    replay = selected.merge(
        labels,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if replay["_merge"].ne("both").any():
        absent = replay.loc[replay["_merge"].ne("both"), keys].to_dict("records")
        raise ValueError(f"RR3 labels missing selected entry keys: {absent[:3]}")
    return (
        replay.sort_values("_entry_order", kind="stable")
        .drop(columns=["_entry_order", "_merge"])
        .reset_index(drop=True)
    )


def _scored_labels(scores: Any, labels: pd.DataFrame) -> pd.DataFrame:
    score_frame = _score_frame(scores)
    if isinstance(score_frame, pd.DataFrame):
        if {"window_id", "step"} <= set(score_frame.columns):
            lookup = score_frame[["window_id", "step", "score"]].copy()
            if lookup.duplicated(["window_id", "step"]).any():
                raise ValueError("scores must be unique by window_id and step")
            work = labels.merge(
                lookup, on=["window_id", "step"], how="left", validate="one_to_one"
            )
        elif len(score_frame) == len(labels):
            work = labels.copy()
            work["score"] = score_frame["score"].to_numpy()
        else:
            raise ValueError("unkeyed scores must align row-for-row with labels")
    else:
        values = np.asarray(score_frame)
        if values.ndim != 1 or len(values) != len(labels):
            raise ValueError("scores must be one-dimensional and align with labels")
        work = labels.copy()
        work["score"] = values
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    return work


def _matched_threshold(
    scores: Any,
    labels: pd.DataFrame,
    target_attempts: int,
    calendar_start: Any,
    calendar_end: Any,
) -> float:
    if int(target_attempts) != target_attempts or target_attempts < 0:
        raise ValueError("target_attempts must be a non-negative integer")
    work = _scored_labels(scores, labels)
    required = {"window_id", "geometry_valid", "entry_time"}
    missing = sorted(required.difference(work.columns))
    if missing:
        raise ValueError(f"labels missing matched-count columns: {missing}")
    start_time = pd.Timestamp(calendar_start)
    end_time = pd.Timestamp(calendar_end)
    start_time = (
        start_time.tz_localize("UTC")
        if start_time.tzinfo is None
        else start_time.tz_convert("UTC")
    )
    end_time = (
        end_time.tz_localize("UTC")
        if end_time.tzinfo is None
        else end_time.tz_convert("UTC")
    )
    if end_time <= start_time:
        raise ValueError("calendar_end must be later than calendar_start")
    work["entry_time"] = pd.to_datetime(work["entry_time"], utc=True, errors="raise")
    work = work[
        work["entry_time"].ge(start_time)
        & work["entry_time"].lt(end_time)
        & work["geometry_valid"].astype(bool)
        & np.isfinite(work["score"])
    ]
    finite = work.loc[work["score"].ge(0.0), "score"]
    candidates = sorted({0.0, *(float(value) for value in finite.unique())})
    window_max = work.groupby("window_id", sort=False)["score"].max().to_numpy()
    ranked = [
        (abs(int(np.count_nonzero(window_max >= threshold)) - int(target_attempts)), -threshold)
        for threshold in candidates
    ]
    return float(candidates[min(range(len(candidates)), key=ranked.__getitem__)])


def matched_count_diagnostic(
    scores: Any,
    labels: pd.DataFrame,
    target_attempts: int,
    *,
    calendar_start: Any = REFERENCE_START,
    calendar_end: Any = REFERENCE_END,
    model_name: str = "model",
) -> MatchedCountResult:
    """Return the closest full-OOF count as an explicitly post-hoc diagnostic."""
    threshold = _matched_threshold(
        scores, labels, target_attempts, calendar_start, calendar_end
    )
    replay = replay_first_crossing(
        _score_frame(scores),
        labels,
        threshold=threshold,
        start=calendar_start,
        end=calendar_end,
    )
    attempted = int(replay.summary["attempted_trades"])
    calendar_days = len(replay.daily_frequency)
    trades = replay.trades.copy()
    trades.attrs["window_universe"] = _window_universe(
        labels, calendar_start, calendar_end
    )
    return MatchedCountResult(
        model_name=_model_name(scores, model_name),
        trades=trades,
        daily=replay.daily_frequency.copy(),
        attempted_trades=attempted,
        observed_trades=int(replay.summary["observed_trades"]),
        trades_per_day=float(attempted / calendar_days) if calendar_days else 0.0,
        mean_net_r=float(replay.summary["mean_net_r"]),
        total_net_r=float(replay.summary["total_net_r"]),
        frequency_noninferior=frequency_noninferior(
            attempted, len(replay.daily_frequency)
        ),
        threshold=threshold,
        target_attempts=int(target_attempts),
    )


def _replay_mapping(
    replays: Mapping[str, TailPolicyResult] | list[TailPolicyResult] | tuple[TailPolicyResult, ...],
) -> dict[str, TailPolicyResult]:
    if isinstance(replays, Mapping):
        return dict(replays)
    return {replay.model_name: replay for replay in replays}


def paired_window_ledger(
    replays: Mapping[str, TailPolicyResult] | list[TailPolicyResult] | tuple[TailPolicyResult, ...],
) -> pd.DataFrame:
    """Align selected entries on the union of all OOF windows, zero-filling WAIT."""
    arms = _replay_mapping(replays)
    if not arms:
        raise ValueError("at least one replay is required")
    universes: list[pd.DataFrame] = []
    for replay in arms.values():
        universe = replay.trades.attrs.get("window_universe")
        if universe is None:
            universe = replay.trades[["window_id", "channel_episode_id"]]
        universes.append(universe[["window_id", "channel_episode_id"]].copy())
    universe = pd.concat(universes, ignore_index=True).drop_duplicates()
    if universe.groupby("window_id")["channel_episode_id"].nunique(dropna=False).gt(1).any():
        raise ValueError("channel_episode_id disagrees across replay arms")
    ledger = universe.drop_duplicates("window_id").reset_index(drop=True)

    for name, replay in arms.items():
        trades = replay.trades.copy()
        if trades["window_id"].duplicated().any():
            raise ValueError(f"{name} replay contains multiple entries in one window")
        selected = trades.set_index("window_id")
        step = ledger["window_id"].map(selected.get("step", pd.Series(dtype=float)))
        outcome = ledger["window_id"].map(
            selected.get("outcome", pd.Series(dtype=object))
        ).fillna("wait")
        observed = ledger["window_id"].map(
            selected.get("path_observed", pd.Series(dtype=bool))
        ).fillna(False).astype(bool)
        net_r = pd.to_numeric(
            ledger["window_id"].map(selected.get("r_net", pd.Series(dtype=float))),
            errors="coerce",
        )
        net_r = net_r.where(observed & net_r.notna(), 0.0).astype(float)
        ledger[f"{name}_step"] = step
        ledger[f"{name}_outcome"] = outcome
        ledger[f"{name}_net_r"] = net_r

    if len(arms) == 1:
        name = next(iter(arms))
        ledger["model"] = name
        ledger["step"] = ledger[f"{name}_step"]
        ledger["outcome"] = ledger[f"{name}_outcome"]
        ledger["net_r"] = ledger[f"{name}_net_r"]
    return ledger


def episode_pair_bootstrap(
    ledger: pd.DataFrame, draws: int = 2000, seed: int = 42
) -> EpisodeBootstrapResult:
    """Bootstrap paired total-net-R deltas by complete channel episodes."""
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if "channel_episode_id" not in ledger.columns:
        raise ValueError("ledger missing channel_episode_id")
    if {"candidate_net_r", "comparator_net_r"} <= set(ledger.columns):
        delta = ledger["candidate_net_r"] - ledger["comparator_net_r"]
    elif "delta_net_r" in ledger.columns:
        delta = ledger["delta_net_r"]
    else:
        columns = [name for name in ledger if name.endswith("_net_r") and name != "net_r"]
        if len(columns) != 2:
            raise ValueError("ledger must identify exactly two paired net-R arms")
        delta = ledger[columns[0]] - ledger[columns[1]]
    work = pd.DataFrame(
        {
            "channel_episode_id": ledger["channel_episode_id"],
            "delta_net_r": pd.to_numeric(delta, errors="raise").fillna(0.0),
        }
    )
    episode_delta = work.groupby("channel_episode_id", sort=False)["delta_net_r"].sum()
    if episode_delta.empty:
        empty = pd.Series(np.nan, index=pd.RangeIndex(draws), name="delta_total_net_r")
        return EpisodeBootstrapResult(np.nan, np.nan, np.nan, empty)
    values = episode_delta.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sample = rng.integers(0, len(values), size=(draws, len(values)))
    draw_values = values[sample].sum(axis=1)
    draw_series = pd.Series(draw_values, name="delta_total_net_r")
    return EpisodeBootstrapResult(
        point_delta_total_net_r=float(values.sum()),
        ci_low=float(np.quantile(draw_values, 0.025)),
        ci_high=float(np.quantile(draw_values, 0.975)),
        draws=draw_series,
    )


def _value(record: Any, name: str, default: Any = np.nan) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    if isinstance(record, pd.Series):
        return record.get(name, default)
    return getattr(record, name, default)


def _simplicity(name: str) -> tuple[int, str]:
    lowered = name.lower()
    try:
        return MODEL_SIMPLICITY.index(lowered), lowered
    except ValueError:
        return len(MODEL_SIMPLICITY), lowered


def _frequency(record: Any) -> bool:
    explicit = _value(record, "frequency_noninferior", None)
    if explicit is not None:
        return bool(explicit)
    return frequency_noninferior(
        int(_value(record, "attempted_trades", 0)),
        int(_value(record, "calendar_days", REFERENCE_CALENDAR_DAYS)),
    )


def _policy_result(record: Any) -> TailPolicyResult | None:
    if isinstance(record, TailPolicyResult):
        return record
    replay = _value(record, "replay", None)
    return replay if isinstance(replay, TailPolicyResult) else None


def _paired_metrics(candidate: Any, comparator: Any) -> tuple[float, float, float]:
    explicit_delta = float(_value(candidate, "paired_delta_total_net_r", np.nan))
    explicit_low = float(_value(candidate, "paired_delta_ci_low", np.nan))
    explicit_high = float(_value(candidate, "paired_delta_ci_high", np.nan))
    if np.isfinite(explicit_delta) or np.isfinite(explicit_low):
        return explicit_delta, explicit_low, explicit_high
    candidate_replay = _policy_result(candidate)
    comparator_replay = _policy_result(comparator)
    if candidate_replay is None or comparator_replay is None:
        return explicit_delta, explicit_low, explicit_high
    ledger = paired_window_ledger(
        {"candidate": candidate_replay, "comparator": comparator_replay}
    )
    result = episode_pair_bootstrap(ledger, draws=2000, seed=42)
    return result.point_delta_total_net_r, result.ci_low, result.ci_high


def compare_tail_models(
    replays: Mapping[str, Any], frozen_references: Mapping[str, Any]
) -> TailModelComparison:
    """Apply protected-frequency, paired-economics, and simplicity rules."""
    candidates = dict(replays)
    references = dict(frozen_references)
    eligible = {
        **{name: record for name, record in references.items() if _frequency(record)},
        **{name: record for name, record in candidates.items() if _frequency(record)},
    }
    strongest = None
    if eligible:
        strongest = min(
            eligible,
            key=lambda name: (
                -float(_value(eligible[name], "total_net_r", -np.inf)),
                _simplicity(name),
            ),
        )

    passes: dict[str, bool] = {}
    reasons: dict[str, tuple[str, ...]] = {}
    rows: list[dict[str, Any]] = []
    for name, record in candidates.items():
        frequency = _frequency(record)
        mean_net_r = float(_value(record, "mean_net_r", np.nan))
        total_net_r = float(_value(record, "total_net_r", np.nan))
        comparator_pool = {
            **{key: value for key, value in references.items() if _frequency(value)},
            **{
                key: value
                for key, value in candidates.items()
                if _frequency(value) and _simplicity(key) < _simplicity(name)
            },
        }
        comparator_name = None
        if comparator_pool:
            comparator_name = min(
                comparator_pool,
                key=lambda key: (
                    -float(_value(comparator_pool[key], "total_net_r", -np.inf)),
                    _simplicity(key),
                ),
            )
        if comparator_name is None:
            paired_delta = paired_low = paired_high = np.nan
        else:
            paired_delta, paired_low, paired_high = _paired_metrics(
                record, comparator_pool[comparator_name]
            )
        rejected: list[str] = []
        if not frequency:
            rejected.append("frequency")
        if not mean_net_r > 0.0:
            rejected.append("mean_net_r")
        if not total_net_r > 0.0:
            rejected.append("total_net_r")
        if not paired_delta > 0.0:
            rejected.append("paired_delta_total_net_r")
        if not paired_low > 0.0:
            rejected.append("paired_delta_ci_low")
        passed = not rejected
        passes[name] = passed
        reasons[name] = tuple(rejected)
        rows.append(
            {
                "model": name,
                "frequency_noninferior": frequency,
                "mean_net_r": mean_net_r,
                "total_net_r": total_net_r,
                "comparator": comparator_name,
                "paired_delta_total_net_r": paired_delta,
                "paired_delta_ci_low": paired_low,
                "paired_delta_ci_high": paired_high,
                "passes": passed,
                "rejection_reasons": ";".join(rejected),
            }
        )

    passing = sorted((name for name, passed in passes.items() if passed), key=_simplicity)
    chosen = passing[0] if passing else None
    if chosen is not None:
        for name in passing[1:]:
            record = candidates[name]
            if (
                float(_value(record, "total_net_r", np.nan))
                > float(_value(candidates[chosen], "total_net_r", np.nan))
                and float(_value(record, "paired_delta_ci_low", np.nan)) > 0.0
            ):
                chosen = name
    table = pd.DataFrame(rows)
    if not table.empty:
        table = table.sort_values(
            "model", key=lambda series: series.map(_simplicity), kind="stable"
        ).reset_index(drop=True)
    return TailModelComparison(
        strongest_comparator=strongest,
        chosen_model=chosen,
        passes=passes,
        rejection_reasons=reasons,
        table=table,
    )


__all__ = [
    "EpisodeBootstrapResult",
    "MatchedCountResult",
    "REFERENCE_ATTEMPTS",
    "REFERENCE_CALENDAR_DAYS",
    "REFERENCE_END",
    "REFERENCE_START",
    "TailModelComparison",
    "TailPolicyResult",
    "compare_tail_models",
    "episode_pair_bootstrap",
    "frequency_noninferior",
    "matched_count_diagnostic",
    "paired_window_ledger",
    "replay_positive_ev",
    "replay_same_entries",
]
