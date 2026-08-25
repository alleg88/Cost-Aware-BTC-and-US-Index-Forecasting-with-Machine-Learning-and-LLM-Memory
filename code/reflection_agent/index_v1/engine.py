"""Causal opportunity, replay and memory primitives for the index agents."""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from experiments.index_all_model_ensemble import AlignedPanel, combine_probabilities
from experiments.index_replication import simulate_one_bar
from experiments.index_replication_protocol import CUTOFF, FORWARD_START
from reflection_agent.index_v1.config import MODEL_NAMES


PROBABILITY_LABELS = ("short", "flat", "long")
OPPORTUNITY_KEY_COLUMNS = (
    "opportunity_id",
    "signal_bar_open",
    "decision_time",
    "entry_time",
    "exit_time",
)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _assert_before_q2(*values: pd.Timestamp) -> None:
    if any(_utc(value) >= CUTOFF for value in values):
        raise PermissionError("Q2 timestamps are sealed for the USA500 agent")


def _ensemble_prediction(panel: AlignedPanel) -> pd.DataFrame:
    probabilities = combine_probabilities("directional_majority", panel)
    return pd.DataFrame(
        {
            "timestamp": panel.timestamp,
            "y_true": panel.y_true,
            "pred": probabilities.argmax(axis=1).astype(int),
            "confidence": probabilities.max(axis=1),
            "p_short": probabilities[:, 0],
            "p_flat": probabilities[:, 1],
            "p_long": probabilities[:, 2],
            "fit_id": "index-agent:frozen-directional-majority",
        }
    )


def opportunity_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the immutable identity columns for exact arm reconciliation."""
    missing = set(OPPORTUNITY_KEY_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"opportunity keys are incomplete: {sorted(missing)}")
    return frame.loc[:, OPPORTUNITY_KEY_COLUMNS].reset_index(drop=True)


def build_registered_opportunities(
    panel: AlignedPanel,
    bars: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    tau: float,
    cost_bps: float,
    state_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Materialise the frozen directional-majority opportunities and evidence."""
    start_utc, end_utc = _utc(start), _utc(end)
    if end_utc <= start_utc:
        raise ValueError("opportunity interval must be positive")
    _assert_before_q2(start_utc, end_utc - pd.Timedelta(nanoseconds=1))
    if tuple(panel.probabilities) != MODEL_NAMES:
        raise ValueError("opportunity panel changed the frozen model order")
    if len(panel.timestamp):
        _assert_before_q2(pd.Timestamp(panel.timestamp.max()))

    prediction = _ensemble_prediction(panel)
    ledger, _ = simulate_one_bar(
        bars,
        prediction,
        start=start_utc,
        end=end_utc,
        tau=float(tau),
        cost_bps=float(cost_bps),
    )
    if ledger.empty:
        raise ValueError("frozen ensemble produced no registered opportunities")
    ledger = ledger.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
    signal_times = pd.DatetimeIndex(pd.to_datetime(ledger["signal_bar_open"], utc=True))
    positions = panel.timestamp.get_indexer(signal_times)
    if (positions < 0).any():
        raise ValueError("registered signal is absent from the probability panel")

    prefix = "FWD" if start_utc >= FORWARD_START else "H1"
    opportunities = ledger.copy()
    opportunities.insert(
        0,
        "opportunity_id",
        [f"{prefix}-{index:06d}" for index in range(len(opportunities))],
    )
    opportunities["original_side"] = opportunities["side"].astype(int)
    opportunities["outcome_available_at"] = pd.to_datetime(
        opportunities["exit_time"], utc=True
    )
    signal_series = pd.Series(signal_times, index=opportunities.index)
    opportunities["week_start"] = (
        signal_series.dt.normalize()
        - pd.to_timedelta(signal_series.dt.weekday, unit="D")
    )
    opportunities["y_true"] = np.asarray(panel.y_true, dtype=int)[positions]
    for model_index, model in enumerate(MODEL_NAMES):
        values = np.asarray(panel.probabilities[model], dtype=float)[positions]
        if values.shape != (len(opportunities), 3):
            raise ValueError("model probability shape changed at opportunities")
        for class_index, label in enumerate(PROBABILITY_LABELS):
            opportunities[f"m{model_index:02d}_p_{label}"] = values[:, class_index]

    if state_frame is not None:
        required_state = {
            "available_at",
            "state_vix_regime",
            "state_trailing_vol",
            "state_trailing_trend",
        }
        missing_state = required_state.difference(state_frame.columns)
        if missing_state:
            raise ValueError(f"causal state misses columns: {sorted(missing_state)}")
        state = state_frame.copy()
        state.index = pd.to_datetime(state.index, utc=True)
        state["available_at"] = pd.to_datetime(state["available_at"], utc=True)
        if state.index.duplicated().any() or not state.index.is_monotonic_increasing:
            raise ValueError("causal state index must be unique and sorted")
        selected = state.reindex(signal_times)
        if selected[list(required_state)].isna().any().any():
            raise ValueError("causal state is incomplete at a registered opportunity")
        decision = pd.to_datetime(opportunities["decision_time"], utc=True).to_numpy()
        available = pd.to_datetime(selected["available_at"], utc=True).to_numpy()
        if (available > decision).any():
            raise ValueError("causal state was not available by the opportunity decision")
        for column in (
            "state_vix_regime",
            "state_trailing_vol",
            "state_trailing_trend",
        ):
            values = selected[column].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("causal state values must be finite")
            opportunities[column] = values

    if opportunities["opportunity_id"].duplicated().any():
        raise AssertionError("registered opportunity identifiers are not unique")
    ordered = opportunities.sort_values("entry_time", kind="mergesort")
    previous_exit = pd.to_datetime(ordered["exit_time"], utc=True).shift(1)
    current_entry = pd.to_datetime(ordered["entry_time"], utc=True)
    if (current_entry.iloc[1:] < previous_exit.iloc[1:]).any():
        raise AssertionError("registered opportunity schedule overlaps")
    if not np.isfinite(
        opportunities.filter(regex=r"^m\d{2}_p_(short|flat|long)$").to_numpy(float)
    ).all():
        raise ValueError("registered probabilities must be finite")

    ledger.insert(0, "opportunity_id", opportunities["opportunity_id"])
    return opportunities, ledger


def replay_registered_sides(
    opportunities: pd.DataFrame,
    sides: Sequence[int],
    *,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Recompute every LONG/SHORT payoff on the immutable next-M15 prices."""
    if len(sides) != len(opportunities):
        raise ValueError("replay requires one side for every registered opportunity")
    side = np.asarray(sides, dtype=int)
    if not np.isin(side, (-1, 1)).all():
        raise ValueError("every replay decision must be LONG or SHORT")
    if opportunities["opportunity_id"].duplicated().any():
        raise ValueError("registered opportunity identifiers must remain unique")
    _assert_before_q2(pd.to_datetime(opportunities["signal_bar_open"], utc=True).max())

    ledger_columns = [
        "opportunity_id",
        "signal_bar_open",
        "decision_time",
        "entry_bar_open",
        "exit_bar_open",
        "entry_time",
        "exit_time",
        "entry_price",
        "exit_price",
        "confidence",
    ]
    missing = set(ledger_columns).difference(opportunities.columns)
    if missing:
        raise ValueError(f"opportunity replay fields are incomplete: {sorted(missing)}")
    ledger = opportunities.loc[:, ledger_columns].copy()
    entry = ledger["entry_price"].to_numpy(dtype=float)
    exit_ = ledger["exit_price"].to_numpy(dtype=float)
    if not np.isfinite(entry).all() or not np.isfinite(exit_).all() or (entry <= 0).any():
        raise ValueError("execution prices must be positive and finite")
    gross = side * (exit_ / entry - 1.0)
    cost = np.full(len(ledger), float(cost_bps) / 10_000.0, dtype=float)
    ledger["side"] = side
    ledger["gross_return"] = gross
    ledger["cost_return"] = cost
    ledger["net_return"] = gross - cost
    ledger = ledger[
        [
            "opportunity_id",
            "signal_bar_open",
            "decision_time",
            "entry_bar_open",
            "exit_bar_open",
            "entry_time",
            "exit_time",
            "side",
            "entry_price",
            "exit_price",
            "confidence",
            "gross_return",
            "cost_return",
            "net_return",
        ]
    ]
    per_bar = ledger.groupby("entry_bar_open", sort=True)["net_return"].sum()
    per_bar.index = pd.to_datetime(per_bar.index, utc=True)
    per_bar.name = "net_return"
    if not opportunity_keys(ledger).equals(opportunity_keys(opportunities)):
        raise AssertionError("replay changed immutable opportunity keys")
    return ledger, per_bar


def equal_weights() -> tuple[float, ...]:
    """Return the exact safe fallback; it is intentionally off the 0.01 grid."""
    return (1.0 / 9.0,) * 9


def _probability_cube(opportunities: pd.DataFrame) -> np.ndarray:
    arrays = []
    for model_index in range(9):
        columns = [
            f"m{model_index:02d}_p_short",
            f"m{model_index:02d}_p_flat",
            f"m{model_index:02d}_p_long",
        ]
        missing = set(columns).difference(opportunities.columns)
        if missing:
            raise ValueError(f"opportunities miss model probabilities: {sorted(missing)}")
        arrays.append(opportunities.loc[:, columns].to_numpy(dtype=float))
    cube = np.stack(arrays, axis=1)
    if (
        not np.isfinite(cube).all()
        or (cube < 0.0).any()
        or not np.allclose(cube.sum(axis=2), 1.0, rtol=0.0, atol=1e-9)
    ):
        raise ValueError("opportunity probabilities must be finite and normalized")
    return cube


def weighted_sides(
    opportunities: pd.DataFrame, weights: Sequence[float]
) -> np.ndarray:
    """Apply host weights to directional probability margins with tie fallback."""
    values = np.asarray(weights, dtype=float)
    if (
        values.shape != (9,)
        or not np.isfinite(values).all()
        or (values < 0.0).any()
        or abs(float(values.sum()) - 1.0) > 1e-9
    ):
        raise ValueError("weighted decision requires nine non-negative weights summing to one")
    cube = _probability_cube(opportunities)
    margin = cube[:, :, 2] - cube[:, :, 0]
    score = margin @ values
    original = opportunities["original_side"].to_numpy(dtype=int)
    if not np.isin(original, (-1, 1)).all():
        raise ValueError("original ensemble side must be LONG or SHORT")
    return np.where(score > 0.0, 1, np.where(score < 0.0, -1, original)).astype(int)


def constrained_grid_weights(scores: Sequence[float]) -> tuple[float, ...]:
    """Map nine scores to a deterministic 0.01-grid control with a 0.05 floor."""
    values = np.asarray(scores, dtype=float)
    if values.shape != (9,) or not np.isfinite(values).all():
        raise ValueError("control score vector must contain nine finite values")
    shifted = values - float(values.max())
    exponent = np.exp(np.clip(shifted, -50.0, 0.0))
    shares = exponent / exponent.sum()
    raw_extra = shares * 55.0
    extra = np.floor(raw_extra).astype(int)
    remainder = 55 - int(extra.sum())
    order = np.lexsort((np.arange(9), -(raw_extra - extra)))
    for index in order[:remainder]:
        extra[index] += 1
    units = extra + 5
    if units.sum() != 100 or (units < 5).any() or (units > 80).any():
        raise AssertionError("control weight projection violated its fixed constraints")
    return tuple(float(value) / 100.0 for value in units)


def _model_sides(opportunities: pd.DataFrame, model_index: int) -> np.ndarray:
    short = opportunities[f"m{model_index:02d}_p_short"].to_numpy(float)
    long = opportunities[f"m{model_index:02d}_p_long"].to_numpy(float)
    original = opportunities["original_side"].to_numpy(int)
    return np.where(long > short, 1, np.where(long < short, -1, original)).astype(int)


def _week_id(week_start: pd.Timestamp) -> str:
    value = _utc(week_start).isoformat().encode("utf-8")
    return "W-" + hashlib.sha256(value).hexdigest()[:12]


def build_weekly_cards(
    opportunities: pd.DataFrame, *, cost_bps: float
) -> list[dict[str, Any]]:
    """Resolve full-information weekly model cards on immutable opportunities."""
    required = {
        "opportunity_id",
        "week_start",
        "outcome_available_at",
        "original_side",
        "y_true",
        "state_vix_regime",
        "state_trailing_vol",
        "state_trailing_trend",
    }
    missing = required.difference(opportunities.columns)
    if missing:
        raise ValueError(f"weekly memory inputs are incomplete: {sorted(missing)}")
    if opportunities["opportunity_id"].duplicated().any():
        raise ValueError("weekly memory opportunity identifiers must be unique")
    table = opportunities.copy()
    table["week_start"] = pd.to_datetime(table["week_start"], utc=True)
    table["outcome_available_at"] = pd.to_datetime(
        table["outcome_available_at"], utc=True
    )
    if table["outcome_available_at"].max() >= CUTOFF:
        raise PermissionError("Q2 outcomes cannot enter weekly memory")
    cards: list[dict[str, Any]] = []
    for week_start, group in table.groupby("week_start", sort=True):
        current = group.sort_values("signal_bar_open", kind="mergesort").reset_index(drop=True)
        cube = _probability_cube(current)
        equal_side = weighted_sides(current, equal_weights())
        equal_ledger, _ = replay_registered_sides(current, equal_side, cost_bps=cost_bps)
        equal_net = float(equal_ledger["net_return"].sum())
        original_ledger, _ = replay_registered_sides(
            current, current["original_side"].to_numpy(int), cost_bps=cost_bps
        )
        model_statistics = []
        directional_votes = []
        for model_index in range(9):
            side = _model_sides(current, model_index)
            directional_votes.append(side)
            ledger, _ = replay_registered_sides(current, side, cost_bps=cost_bps)
            probabilities = cube[:, model_index, :]
            actual = current["y_true"].to_numpy(dtype=int)
            one_hot = np.eye(3, dtype=float)[actual]
            predicted_class = np.where(side == 1, 2, 0)
            model_statistics.append(
                {
                    "model_index": model_index,
                    "sample_count": int(len(current)),
                    "directional_accuracy": float(np.mean(predicted_class == actual)),
                    "net_return": float(ledger["net_return"].sum()),
                    "long_net_return": float(
                        ledger.loc[ledger["side"].eq(1), "net_return"].sum()
                    ),
                    "short_net_return": float(
                        ledger.loc[ledger["side"].eq(-1), "net_return"].sum()
                    ),
                    "brier_score": float(np.mean(np.square(probabilities - one_hot).sum(axis=1))),
                    "mean_confidence": float(probabilities.max(axis=1).mean()),
                    "marginal_net_vs_equal": float(
                        ledger["net_return"].sum() - equal_net
                    ),
                }
            )
        votes = np.column_stack(directional_votes)
        long_share = (votes == 1).mean(axis=1)
        margins = cube[:, :, 2] - cube[:, :, 0]
        cards.append(
            {
                "week_id": _week_id(pd.Timestamp(week_start)),
                "week_start": _utc(pd.Timestamp(week_start)).isoformat(),
                "available_at": table.loc[group.index, "outcome_available_at"].max().isoformat(),
                "opportunities": int(len(current)),
                "model_statistics": model_statistics,
                "controls": {
                    "original": {
                        "net_return": float(original_ledger["net_return"].sum()),
                        "long_trades": int(original_ledger["side"].eq(1).sum()),
                        "short_trades": int(original_ledger["side"].eq(-1).sum()),
                    },
                    "equal_weight": {
                        "net_return": equal_net,
                        "long_trades": int(equal_ledger["side"].eq(1).sum()),
                        "short_trades": int(equal_ledger["side"].eq(-1).sum()),
                    },
                },
                "market_state": {
                    "vix_regime": float(current["state_vix_regime"].mean()),
                    "trailing_volatility": float(current["state_trailing_vol"].mean()),
                    "trailing_trend": float(current["state_trailing_trend"].mean()),
                },
                "agreement": float(np.maximum(long_share, 1.0 - long_share).mean()),
                "probability_dispersion": float(np.std(margins, axis=1).mean()),
            }
        )
    return cards


def eligible_memory(
    cards: Sequence[dict[str, Any]], decision_time: str | pd.Timestamp
) -> list[dict[str, Any]]:
    """Return only cards whose complete outcome resolved before commitment."""
    cutoff = _utc(decision_time)
    if cutoff >= CUTOFF:
        raise PermissionError("Q2 decisions cannot consume agent memory")
    visible = []
    for card in cards:
        if "available_at" not in card or "week_start" not in card:
            raise ValueError("memory card lacks causal timestamps")
        available = _utc(card["available_at"])
        week_start = _utc(card["week_start"])
        if available >= CUTOFF or week_start >= CUTOFF:
            raise PermissionError("Q2 card entered agent memory")
        if available < cutoff:
            visible.append(copy.deepcopy(card))
    return visible


def shuffled_memory(
    cards: Sequence[dict[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    """Rotate model attribution while preserving every card's causal timing."""
    output: list[dict[str, Any]] = []
    for card_number, original in enumerate(cards):
        card = copy.deepcopy(original)
        rows = list(card.get("model_statistics", []))
        if len(rows) != 9:
            raise ValueError("shuffled memory requires nine model statistics")
        shift = 1 + ((int(seed) + card_number) % 8)
        payloads = [
            {key: value for key, value in row.items() if key != "model_index"}
            for row in rows
        ]
        rotated = payloads[shift:] + payloads[:shift]
        card["model_statistics"] = [
            {"model_index": index, **payload} for index, payload in enumerate(rotated)
        ]
        if "prior_weights" in card:
            weights = list(card["prior_weights"])
            card["prior_weights"] = weights[shift:] + weights[:shift]
        output.append(card)
    return output


def memory_statistics(cards: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the latest one, four and twelve resolved cards by model index."""
    if not cards:
        return []
    ordered = sorted(cards, key=lambda item: _utc(item["week_start"]))
    rows = []
    for model_index in range(9):
        result: dict[str, Any] = {"model_index": model_index}
        for window in (1, 4, 12):
            recent = ordered[-window:]
            values = [
                next(
                    row
                    for row in card["model_statistics"]
                    if int(row["model_index"]) == model_index
                )
                for card in recent
            ]
            samples = sum(int(item["sample_count"]) for item in values)
            weighted_accuracy = (
                sum(float(item["directional_accuracy"]) * int(item["sample_count"]) for item in values)
                / samples
                if samples
                else 0.0
            )
            weighted_brier = (
                sum(float(item["brier_score"]) * int(item["sample_count"]) for item in values)
                / samples
                if samples
                else 0.0
            )
            weighted_confidence = (
                sum(float(item["mean_confidence"]) * int(item["sample_count"]) for item in values)
                / samples
                if samples
                else 0.0
            )
            result[f"rolling_{window}"] = {
                "weeks": len(recent),
                "sample_count": samples,
                "directional_accuracy": float(weighted_accuracy),
                "net_return": float(sum(float(item["net_return"]) for item in values)),
                "brier_score": float(weighted_brier),
                "mean_confidence": float(weighted_confidence),
            }
        rows.append(result)
    return rows


def tune_static_h1_weights(
    h1_opportunities: pd.DataFrame, *, cost_bps: float
) -> tuple[float, ...]:
    """Freeze one constrained vector from model counterfactual H1 Net only."""
    net = []
    for model_index in range(9):
        ledger, _ = replay_registered_sides(
            h1_opportunities,
            _model_sides(h1_opportunities, model_index),
            cost_bps=cost_bps,
        )
        net.append(float(ledger["net_return"].sum()))
    values = np.asarray(net, dtype=float)
    scale = float(values.std(ddof=0))
    scores = (values - float(values.mean())) / scale if scale > 0.0 else np.zeros(9)
    return constrained_grid_weights(scores)


def causal_hedge_schedule(
    cards: Sequence[dict[str, Any]], *, eta: float
) -> dict[str, tuple[float, ...]]:
    """Commit weekly Hedge weights before applying that week's resolved payoffs."""
    cumulative = np.zeros(9, dtype=float)
    schedule: dict[str, tuple[float, ...]] = {}
    for card in sorted(cards, key=lambda item: _utc(item["week_start"])):
        schedule[str(card["week_id"])] = constrained_grid_weights(float(eta) * cumulative)
        rows = sorted(card["model_statistics"], key=lambda item: int(item["model_index"]))
        if [int(item["model_index"]) for item in rows] != list(range(9)):
            raise ValueError("Hedge card changed the nine-model order")
        reward = np.asarray([float(item["net_return"]) for item in rows], dtype=float)
        cumulative += np.tanh(reward / 0.01)
    return schedule


def build_causal_week_states(
    state_frame: pd.DataFrame, week_starts: Sequence[str | pd.Timestamp]
) -> pd.DataFrame:
    """Freeze the latest state strictly available before each weekly decision."""
    required = {
        "available_at",
        "state_vix_regime",
        "state_trailing_vol",
        "state_trailing_trend",
    }
    missing = required.difference(state_frame.columns)
    if missing:
        raise ValueError(f"weekly state source misses columns: {sorted(missing)}")
    state = state_frame.copy()
    state.index = pd.to_datetime(state.index, utc=True)
    state["available_at"] = pd.to_datetime(state["available_at"], utc=True)
    state = state.sort_values("available_at", kind="mergesort")
    rows = []
    for value in sorted({_utc(item) for item in week_starts}):
        if value >= CUTOFF:
            raise PermissionError("Q2 week cannot enter causal state")
        eligible = state.loc[state["available_at"] < value]
        if eligible.empty:
            raise ValueError("no market state was available before weekly commitment")
        current = eligible.iloc[-1]
        rows.append(
            {
                "week_start": value,
                "state_available_at": pd.Timestamp(current["available_at"]),
                "state_vix_regime": float(current["state_vix_regime"]),
                "state_trailing_vol": float(current["state_trailing_vol"]),
                "state_trailing_trend": float(current["state_trailing_trend"]),
            }
        )
    output = pd.DataFrame(rows)
    numeric = output[
        ["state_vix_regime", "state_trailing_vol", "state_trailing_trend"]
    ].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("weekly causal state must be finite")
    return output


def add_control_payoffs(
    cards: Sequence[dict[str, Any]],
    opportunities: pd.DataFrame,
    *,
    control_weights: dict[str, dict[str, Sequence[float]]],
    cost_bps: float,
) -> list[dict[str, Any]]:
    """Attach same-opportunity static/Hedge counterfactuals to resolved cards."""
    table = opportunities.copy()
    table["week_start"] = pd.to_datetime(table["week_start"], utc=True)
    output = []
    for original in cards:
        card = copy.deepcopy(original)
        week_start = _utc(card["week_start"])
        current = table.loc[table["week_start"].eq(week_start)].sort_values(
            "signal_bar_open", kind="mergesort"
        )
        if len(current) != int(card["opportunities"]):
            raise ValueError("control payoff opportunities differ from the memory card")
        for control_id, schedule in control_weights.items():
            if card["week_id"] not in schedule:
                raise ValueError(f"control schedule misses week: {control_id}")
            side = weighted_sides(current, schedule[card["week_id"]])
            ledger, _ = replay_registered_sides(current, side, cost_bps=cost_bps)
            card["controls"][str(control_id)] = {
                "net_return": float(ledger["net_return"].sum()),
                "long_trades": int(ledger["side"].eq(1).sum()),
                "short_trades": int(ledger["side"].eq(-1).sum()),
            }
        output.append(card)
    return output


__all__ = [
    "add_control_payoffs",
    "build_causal_week_states",
    "build_registered_opportunities",
    "build_weekly_cards",
    "causal_hedge_schedule",
    "constrained_grid_weights",
    "eligible_memory",
    "equal_weights",
    "memory_statistics",
    "opportunity_keys",
    "replay_registered_sides",
    "shuffled_memory",
    "tune_static_h1_weights",
    "weighted_sides",
]
