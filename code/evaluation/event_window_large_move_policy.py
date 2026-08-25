"""Causal one-entry-per-window policy for adaptive large moves."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.event_window_large_move_dataset import AdaptiveMoveConfig


@dataclass(frozen=True)
class LargeMovePolicyConfig:
    minimum_conditional_direction_probability: float = 0.55
    minimum_exact_direction_accuracy: float = 0.70
    desired_trades_per_day_low: float = 1.0
    desired_trades_per_day_high: float = 2.0
    threshold_grid: tuple[float, ...] = (
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    )

    def __post_init__(self) -> None:
        if not 0.5 <= self.minimum_conditional_direction_probability < 1.0:
            raise ValueError("conditional direction probability must be in [0.5, 1)")
        if not 0.5 <= self.minimum_exact_direction_accuracy <= 1.0:
            raise ValueError("exact direction accuracy must be in [0.5, 1]")
        if not 0 < self.desired_trades_per_day_low <= self.desired_trades_per_day_high:
            raise ValueError("invalid desired frequency band")
        if not self.threshold_grid or any(
            not 0.0 <= value <= 1.0 for value in self.threshold_grid
        ):
            raise ValueError("threshold grid must contain probabilities")


def _calendar_days(frame: pd.DataFrame) -> int:
    times = pd.to_datetime(frame["decision_time"], utc=True, errors="raise")
    return max(1, int((times.max().normalize() - times.min().normalize()).days + 1))


def _validate_scores(scores: pd.DataFrame) -> pd.DataFrame:
    required = {
        "window_id",
        "channel_episode_id",
        "step",
        "decision_time",
        "p_no_big",
        "p_up_big",
        "p_down_big",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"scores missing columns: {missing}")
    out = scores.copy().reset_index(drop=True)
    out["decision_time"] = pd.to_datetime(out["decision_time"], utc=True, errors="raise")
    if out.duplicated(["window_id", "step"]).any():
        raise ValueError("scores contain duplicate decision keys")
    probabilities = out[["p_no_big", "p_up_big", "p_down_big"]].to_numpy(float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-6
    ):
        raise ValueError("scores require finite probabilities summing to one")
    return out


def _aligned_labels(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    keys = ["window_id", "step"]
    required = {
        *keys,
        "move_code",
        "move_label",
        "adaptive_barrier_bps",
        "terminal_return_bps",
        "model_target_valid",
    }
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"labels missing columns: {missing}")
    work = labels.copy()
    if work.duplicated(keys).any():
        raise ValueError("labels contain duplicate decision keys")
    aligned = scores.merge(work, on=keys, how="left", validate="one_to_one", suffixes=("", "_label"))
    if aligned["move_code"].isna().any():
        raise ValueError("labels do not cover every score row")
    return aligned


def _realise(
    row: pd.Series,
    *,
    predicted_code: int,
    execution: AdaptiveMoveConfig,
) -> dict[str, object]:
    actual = int(row["move_code"])
    barrier = float(row["adaptive_barrier_bps"])
    if predicted_code not in {1, 2} or actual not in {0, 1, 2}:
        raise ValueError("trade realisation requires valid direction classes")
    sign = 1.0 if predicted_code == 1 else -1.0
    if actual == predicted_code:
        outcome = "target"
        gross_bps = barrier
        cost_bps = execution.target_cost_bps
    elif actual in {1, 2}:
        outcome = "opposite_barrier"
        gross_bps = -barrier
        cost_bps = execution.other_cost_bps
    else:
        outcome = "no_big_timeout"
        gross_bps = sign * float(row["terminal_return_bps"])
        cost_bps = execution.other_cost_bps
    net_bps = gross_bps - cost_bps
    return {
        "predicted_direction": "long" if predicted_code == 1 else "short",
        "predicted_code": predicted_code,
        "actual_big": actual in {1, 2},
        "direction_correct": actual == predicted_code,
        "trade_outcome": outcome,
        "gross_bps": float(gross_bps),
        "cost_bps": float(cost_bps),
        "net_bps": float(net_bps),
        "net_r": float(net_bps / barrier),
    }


def replay_large_move_policy(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    threshold: float | None = None,
    threshold_column: str | None = None,
    config: LargeMovePolicyConfig = LargeMovePolicyConfig(),
    execution: AdaptiveMoveConfig = AdaptiveMoveConfig(),
) -> pd.DataFrame:
    """Enter at the first causal row whose predicted direction beats NO_BIG."""
    if (threshold is None) == (threshold_column is None):
        raise ValueError("provide exactly one of threshold or threshold_column")
    if threshold is not None and not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a probability")
    work = _aligned_labels(_validate_scores(scores), labels)
    if not work["model_target_valid"].fillna(False).astype(bool).all():
        raise ValueError("policy scores must exclude invalid target rows")
    if threshold_column is not None and threshold_column not in work:
        raise ValueError(f"missing threshold column: {threshold_column}")
    rows: list[dict[str, object]] = []
    for _, group in work.sort_values(
        ["decision_time", "window_id", "step"], kind="stable"
    ).groupby("window_id", sort=False):
        chosen = None
        for candidate in group.sort_values("step", kind="stable").itertuples(index=False):
            p_no = float(candidate.p_no_big)
            p_up = float(candidate.p_up_big)
            p_down = float(candidate.p_down_big)
            p_big = p_up + p_down
            predicted_code = 1 if p_up >= p_down else 2
            p_direction = max(p_up, p_down)
            conditional = p_direction / p_big if p_big > 0 else 0.0
            row_threshold = (
                float(threshold)
                if threshold is not None
                else float(getattr(candidate, str(threshold_column)))
            )
            barrier = float(candidate.adaptive_barrier_bps)
            if predicted_code == 1:
                p_correct, p_wrong = p_up, p_down
            else:
                p_correct, p_wrong = p_down, p_up
            expected_cost = (
                p_correct * execution.target_cost_bps
                + (p_wrong + p_no) * execution.other_cost_bps
            )
            neutral_timeout_ev_proxy_bps = (
                p_correct - p_wrong
            ) * barrier - expected_cost
            if (
                p_direction >= row_threshold
                and conditional >= config.minimum_conditional_direction_probability
                and neutral_timeout_ev_proxy_bps > 0.0
            ):
                chosen = (
                    candidate,
                    predicted_code,
                    p_big,
                    p_direction,
                    conditional,
                    row_threshold,
                    neutral_timeout_ev_proxy_bps,
                )
                break
        if chosen is None:
            continue
        (
            candidate,
            predicted_code,
            p_big,
            p_direction,
            conditional,
            row_threshold,
            neutral_timeout_ev_proxy_bps,
        ) = chosen
        row = pd.Series(candidate._asdict())
        rows.append(
            {
                **candidate._asdict(),
                "selected_threshold": row_threshold,
                "p_big": p_big,
                "p_direction": p_direction,
                "conditional_direction_probability": conditional,
                "neutral_timeout_ev_proxy_bps": neutral_timeout_ev_proxy_bps,
                **_realise(row, predicted_code=predicted_code, execution=execution),
            }
        )
    if rows:
        return pd.DataFrame(rows)
    empty = work.iloc[:0].copy()
    for column, dtype in (
        ("selected_threshold", "float64"),
        ("p_big", "float64"),
        ("p_direction", "float64"),
        ("conditional_direction_probability", "float64"),
        ("neutral_timeout_ev_proxy_bps", "float64"),
        ("predicted_direction", "object"),
        ("predicted_code", "int64"),
        ("actual_big", "bool"),
        ("direction_correct", "bool"),
        ("trade_outcome", "object"),
        ("gross_bps", "float64"),
        ("cost_bps", "float64"),
        ("net_bps", "float64"),
        ("net_r", "float64"),
    ):
        empty[column] = pd.Series(index=empty.index, dtype=dtype)
    return empty


def policy_summary(selected: pd.DataFrame, scores: pd.DataFrame) -> dict[str, object]:
    days = _calendar_days(scores)
    trades = len(selected)
    return {
        "trades": int(trades),
        "trades_per_day": float(trades / days),
        "zero_trade_days": int(
            days
            - pd.to_datetime(selected.get("decision_time", pd.Series([], dtype="datetime64[ns, UTC]")), utc=True)
            .dt.normalize()
            .nunique()
        ),
        "big_move_precision": float(selected["actual_big"].mean()) if trades else np.nan,
        "directional_precision": float(selected["direction_correct"].mean()) if trades else np.nan,
        "mean_net_bps": float(selected["net_bps"].mean()) if trades else np.nan,
        "total_net_bps": float(selected["net_bps"].sum()) if trades else 0.0,
        "mean_net_r": float(selected["net_r"].mean()) if trades else np.nan,
        "total_net_r": float(selected["net_r"].sum()) if trades else 0.0,
    }


def select_calibration_threshold(
    scores: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    config: LargeMovePolicyConfig = LargeMovePolicyConfig(),
    execution: AdaptiveMoveConfig = AdaptiveMoveConfig(),
) -> tuple[float, pd.DataFrame]:
    """Choose a calibration-only threshold targeting, never forcing, 1-2 trades/day."""
    rows: list[dict[str, object]] = []
    for threshold in config.threshold_grid:
        selected = replay_large_move_policy(
            scores,
            labels,
            threshold=threshold,
            config=config,
            execution=execution,
        )
        rows.append({"threshold": threshold, **policy_summary(selected, scores)})
    frontier = pd.DataFrame(rows)
    in_band = frontier["trades_per_day"].between(
        config.desired_trades_per_day_low,
        config.desired_trades_per_day_high,
        inclusive="both",
    )
    eligible = frontier.loc[in_band]
    if eligible.empty:
        distance = np.maximum(
            config.desired_trades_per_day_low - frontier["trades_per_day"],
            frontier["trades_per_day"] - config.desired_trades_per_day_high,
        ).clip(lower=0.0)
        eligible = frontier.loc[distance.eq(distance.min())]
    positive = eligible.loc[eligible["total_net_r"].gt(0.0)]
    pool = positive if not positive.empty else eligible
    chosen = pool.sort_values(
        ["total_net_r", "directional_precision", "threshold"],
        ascending=[False, False, False],
        kind="stable",
    ).iloc[0]
    frontier["selected"] = frontier["threshold"].eq(float(chosen["threshold"]))
    return float(chosen["threshold"]), frontier


__all__ = [
    "LargeMovePolicyConfig",
    "policy_summary",
    "replay_large_move_policy",
    "select_calibration_threshold",
]
