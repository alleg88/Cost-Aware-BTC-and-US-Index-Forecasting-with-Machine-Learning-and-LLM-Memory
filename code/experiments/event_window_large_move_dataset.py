"""Causal adaptive large-move labels and decision features.

The target is deliberately independent of the old fixed-RR trade outcome.  At
each five-minute decision boundary it asks which adaptive absolute-price
barrier is touched first over the next two hours.  The barrier itself uses only
completed five-minute returns before the decision.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.event_window_tail_dataset import TailDecisionDataset


MOVE_CODES = {"no_big_move": 0, "up_big": 1, "down_big": 2}
OPPORTUNITY_CODES = {"no_opportunity": 0, "opportunity": 1}
_DROP_PREFIXES = (
    "pre_window_mask",
    "active_window_mask",
    "bar_complete",
    "cadence_gap",
)
_DROP_EXACT = frozenset({"window_reason_code", "time_remaining_fraction"})
VOLATILITY_FEATURE_COLUMNS = (
    "adaptive_barrier_bps",
    "past_sigma_5m_bps",
    "past_rv_15_bps",
    "past_rv_30_bps",
    "past_rv_60_bps",
    "past_rv_120_bps",
    "rv_ratio_15_60",
    "rv_ratio_60_120",
    "past_range_15_bps",
    "past_range_60_bps",
    "past_range_120_bps",
    "range_ratio_15_120",
    "past_return_15_bps",
    "past_return_60_bps",
    "past_abs_return_15_bps",
    "past_abs_return_60_bps",
    "volume_ratio_15_120",
    "trade_count_ratio_15_120",
    "taker_imbalance_delta_15_120",
)
_RAW_DIRECTIONAL_SOURCES = {
    "log_return_side": "raw_log_return",
    "cumulative_return_side": "raw_cumulative_return",
    "body_bps_side": "raw_body_bps",
    "taker_imbalance_side": "raw_taker_imbalance",
    "taker_imbalance_mean_3_side": "raw_taker_imbalance_mean_3",
    "taker_imbalance_delta_side": "raw_taker_imbalance_delta",
    "distance_mid_bps_side": "raw_distance_mid_bps",
    "channel_slope_side": "raw_channel_slope",
    "oi_chg_15m_side": "raw_oi_chg_15m",
    "oi_chg_1h_side": "raw_oi_chg_1h",
    "oi_chg_4h_side": "raw_oi_chg_4h",
    "oi_accel_1h_side": "raw_oi_accel_1h",
    "funding_rate_side": "raw_funding_rate",
    "funding_z_side": "raw_funding_z",
    "toptrader_log_ratio_side": "raw_toptrader_log_ratio",
    "taker_log_ratio_side": "raw_taker_log_ratio",
    "price_oi_interaction": "raw_price_oi_interaction",
    "running_favourable_excursion_bps": "raw_signed_favourable_excursion_bps",
    "running_recovery_bps": "raw_signed_recovery_bps",
}


@dataclass(frozen=True)
class AdaptiveMoveConfig:
    minimum_barrier_bps: float = 75.0
    volatility_addon_weight: float = 0.5
    maximum_barrier_bps: float = 250.0
    volatility_lookback_minutes: int = 60
    feature_history_minutes: int = 120
    horizon_minutes: int = 120
    taker_entry_bps: float = 5.0
    maker_target_exit_bps: float = 2.0
    taker_other_exit_bps: float = 5.0

    def __post_init__(self) -> None:
        positive = (
            "minimum_barrier_bps",
            "volatility_addon_weight",
            "maximum_barrier_bps",
            "volatility_lookback_minutes",
            "feature_history_minutes",
            "horizon_minutes",
        )
        for name in positive:
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.maximum_barrier_bps < self.minimum_barrier_bps:
            raise ValueError("maximum barrier cannot be below the minimum")
        if self.feature_history_minutes < self.volatility_lookback_minutes:
            raise ValueError("feature history must cover the volatility lookback")
        if self.volatility_lookback_minutes % 5:
            raise ValueError("volatility lookback must be divisible by five minutes")
        if self.feature_history_minutes < 120:
            raise ValueError("feature history must cover the 120-minute feature horizon")
        for name in (
            "taker_entry_bps",
            "maker_target_exit_bps",
            "taker_other_exit_bps",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")

    @property
    def target_cost_bps(self) -> float:
        return float(self.taker_entry_bps + self.maker_target_exit_bps)

    @property
    def other_cost_bps(self) -> float:
        return float(self.taker_entry_bps + self.taker_other_exit_bps)


@dataclass(frozen=True)
class LargeMoveDecisionDataset:
    decisions: pd.DataFrame
    tabular: np.ndarray
    tabular_features: tuple[str, ...]
    dropped_features: tuple[str, ...]
    feature_set: str


def _utc_minute(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"minute data missing columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("minute data must use a DatetimeIndex")
    out = frame.copy()
    out.index = pd.to_datetime(out.index, utc=True, errors="raise")
    if out.index.has_duplicates or not out.index.is_monotonic_increasing:
        raise ValueError("minute timestamps must be unique and increasing")
    return out


def _continuous(index: pd.DatetimeIndex, start: pd.Timestamp) -> bool:
    if not len(index):
        return False
    expected = pd.date_range(start, periods=len(index), freq="1min", tz="UTC")
    return index.equals(expected)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if np.isfinite(denominator) and denominator > 0 else np.nan


def _history_features(
    history: pd.DataFrame, *, volatility_lookback_minutes: int
) -> dict[str, float]:
    close = history["close"].to_numpy(dtype=float)
    high = history["high"].to_numpy(dtype=float)
    low = history["low"].to_numpy(dtype=float)
    log_returns = np.diff(np.log(close))
    values: dict[str, float] = {}
    five_minute_returns = int(volatility_lookback_minutes // 5)
    completed_five_minute_closes = close[::-5][: five_minute_returns + 1][::-1]
    if len(completed_five_minute_closes) != five_minute_returns + 1:
        raise ValueError("history does not cover the configured completed five-minute returns")
    values["past_sigma_5m_bps"] = float(
        np.std(np.diff(np.log(completed_five_minute_closes)), ddof=1) * 1e4
    )
    for horizon in (15, 30, 60, 120):
        returns = log_returns[-horizon:]
        bars = history.iloc[-horizon:]
        values[f"past_rv_{horizon}_bps"] = float(np.sqrt(np.square(returns).sum()) * 1e4)
        values[f"past_range_{horizon}_bps"] = float(
            (bars["high"].max() / bars["low"].min() - 1.0) * 1e4
        )
    for horizon in (15, 60):
        values[f"past_return_{horizon}_bps"] = float(
            (close[-1] / close[-horizon - 1] - 1.0) * 1e4
        )
        values[f"past_abs_return_{horizon}_bps"] = abs(
            values[f"past_return_{horizon}_bps"]
        )
    values["rv_ratio_15_60"] = _safe_ratio(
        values["past_rv_15_bps"], values["past_rv_60_bps"]
    )
    values["rv_ratio_60_120"] = _safe_ratio(
        values["past_rv_60_bps"], values["past_rv_120_bps"]
    )
    values["range_ratio_15_120"] = _safe_ratio(
        values["past_range_15_bps"], values["past_range_120_bps"]
    )
    count_column = "trade_count" if "trade_count" in history else "count"
    for column, output in (
        ("volume", "volume_ratio_15_120"),
        (count_column, "trade_count_ratio_15_120"),
    ):
        if column in history:
            recent = float(pd.to_numeric(history[column].iloc[-15:], errors="coerce").mean())
            prior = float(pd.to_numeric(history[column].iloc[-120:], errors="coerce").mean())
            values[output] = _safe_ratio(recent, prior)
        else:
            values[output] = np.nan
    if {"taker_buy_base", "volume"}.issubset(history.columns):
        volume = pd.to_numeric(history["volume"], errors="coerce").to_numpy(float)
        taker = pd.to_numeric(history["taker_buy_base"], errors="coerce").to_numpy(float)
        imbalance = np.divide(
            2.0 * taker - volume,
            volume,
            out=np.full(len(volume), np.nan),
            where=np.isfinite(volume) & (volume > 0),
        )
        values["taker_imbalance_delta_15_120"] = float(
            np.nanmean(imbalance[-15:]) - np.nanmean(imbalance[-120:])
        )
    else:
        values["taker_imbalance_delta_15_120"] = np.nan
    return values


def label_adaptive_large_moves(
    base_decisions: pd.DataFrame,
    minute: pd.DataFrame,
    config: AdaptiveMoveConfig = AdaptiveMoveConfig(),
) -> pd.DataFrame:
    """Label first adaptive barrier touch using only a causal past barrier."""
    required = {"window_id", "channel_episode_id", "side", "step", "decision_time"}
    missing = sorted(required.difference(base_decisions.columns))
    if missing:
        raise ValueError(f"base decisions missing columns: {missing}")
    one = _utc_minute(minute)
    decisions = base_decisions.reset_index(drop=True).copy()
    decisions["decision_time"] = pd.to_datetime(
        decisions["decision_time"], utc=True, errors="raise"
    )
    rows: list[dict[str, object]] = []
    history_bars = int(config.feature_history_minutes) + 1
    for row in decisions.itertuples(index=False):
        decision_time = pd.Timestamp(row.decision_time)
        position = int(one.index.searchsorted(decision_time))
        outcome = "censored"
        move_code = -1
        model_target_valid = False
        path_observed = False
        reference = np.nan
        label_end = decision_time
        terminal_return_bps = np.nan
        up_excursion_bps = np.nan
        down_excursion_bps = np.nan
        features = {name: np.nan for name in VOLATILITY_FEATURE_COLUMNS}
        history_start = position - history_bars
        history = one.iloc[max(0, history_start):position]
        history_expected_start = decision_time - pd.Timedelta(
            minutes=config.feature_history_minutes + 1
        )
        history_valid = (
            history_start >= 0
            and len(history) == history_bars
            and _continuous(history.index, history_expected_start)
            and np.isfinite(history[["open", "high", "low", "close"]].to_numpy(float)).all()
        )
        if (
            history_valid
            and position < len(one)
            and one.index[position] == decision_time
            and np.isfinite(float(one.iloc[position]["open"]))
        ):
            features.update(
                _history_features(
                    history,
                    volatility_lookback_minutes=config.volatility_lookback_minutes,
                )
            )
            barrier = float(
                np.clip(
                    max(
                        config.minimum_barrier_bps,
                        config.minimum_barrier_bps
                        + config.volatility_addon_weight
                        * features["past_sigma_5m_bps"]
                        * np.sqrt(config.horizon_minutes / 5.0),
                    ),
                    config.minimum_barrier_bps,
                    config.maximum_barrier_bps,
                )
            )
            features["adaptive_barrier_bps"] = barrier
            reference = float(one.iloc[position]["open"])
            upper = reference * np.exp(barrier / 1e4)
            lower = reference * np.exp(-barrier / 1e4)
            highs: list[float] = []
            lows: list[float] = []
            closes: list[float] = []
            complete = True
            for offset in range(int(config.horizon_minutes)):
                expected_time = decision_time + pd.Timedelta(minutes=offset)
                current = position + offset
                if current >= len(one) or one.index[current] != expected_time:
                    complete = False
                    label_end = expected_time
                    break
                bar = one.iloc[current]
                high, low, close = float(bar["high"]), float(bar["low"]), float(bar["close"])
                if not np.isfinite([high, low, close]).all():
                    complete = False
                    label_end = expected_time
                    break
                highs.append(high)
                lows.append(low)
                closes.append(close)
                up_touch = high >= upper
                down_touch = low <= lower
                label_end = expected_time + pd.Timedelta(minutes=1)
                if up_touch and down_touch:
                    outcome = "ambiguous"
                    path_observed = True
                    break
                if up_touch:
                    outcome, move_code = "up_big", MOVE_CODES["up_big"]
                    model_target_valid = path_observed = True
                    break
                if down_touch:
                    outcome, move_code = "down_big", MOVE_CODES["down_big"]
                    model_target_valid = path_observed = True
                    break
            else:
                if complete:
                    outcome, move_code = "no_big_move", MOVE_CODES["no_big_move"]
                    model_target_valid = path_observed = True
            if highs:
                up_excursion_bps = float(np.log(max(highs) / reference) * 1e4)
                down_excursion_bps = float(np.log(reference / min(lows)) * 1e4)
                terminal_return_bps = float(np.log(closes[-1] / reference) * 1e4)
        rows.append(
            {
                "window_id": row.window_id,
                "channel_episode_id": row.channel_episode_id,
                "side": str(row.side),
                "step": int(row.step),
                "decision_time": decision_time,
                "reference_price": reference,
                "move_label": outcome,
                "move_code": int(move_code),
                "label_start": decision_time,
                "label_end": label_end,
                "model_target_valid": bool(model_target_valid),
                "path_observed": bool(path_observed),
                "terminal_return_bps": terminal_return_bps,
                "future_up_excursion_bps": up_excursion_bps,
                "future_down_excursion_bps": down_excursion_bps,
                **features,
            }
        )
    return add_opportunity_target_contract(pd.DataFrame(rows))


def add_opportunity_target_contract(labels: pd.DataFrame) -> pd.DataFrame:
    """Add separate causal validity for magnitude and direction targets.

    A same-minute double touch proves that a large move occurred, so it is a
    valid positive opportunity label.  Its direction remains unknowable at
    one-minute OHLC resolution.  A censored or gapped path is invalid for both
    targets and invalidates every decision in the same window.
    """
    required = {"window_id", "move_label", "move_code", "path_observed"}
    missing = sorted(required.difference(labels.columns))
    if missing:
        raise ValueError(f"adaptive labels missing target columns: {missing}")
    result = labels.copy()
    positive = result["move_label"].isin(["up_big", "down_big", "ambiguous"])
    negative = result["move_label"].eq("no_big_move")
    result["opportunity_code"] = np.select(
        [positive, negative],
        [OPPORTUNITY_CODES["opportunity"], OPPORTUNITY_CODES["no_opportunity"]],
        default=-1,
    ).astype(np.int8)
    observed = result["path_observed"].astype(bool)
    result["opportunity_row_target_valid"] = observed & result[
        "opportunity_code"
    ].ge(0)
    result["direction_row_target_valid"] = observed & result["move_code"].isin(
        MOVE_CODES.values()
    )
    result["opportunity_incomplete_window"] = result.groupby("window_id")[
        "opportunity_row_target_valid"
    ].transform(lambda values: not bool(values.all())).astype(bool)
    result["direction_incomplete_window"] = result.groupby("window_id")[
        "direction_row_target_valid"
    ].transform(lambda values: not bool(values.all())).astype(bool)
    result["opportunity_target_valid"] = (
        result["opportunity_row_target_valid"]
        & ~result["opportunity_incomplete_window"]
    )
    result["direction_target_valid"] = (
        result["direction_row_target_valid"]
        & ~result["direction_incomplete_window"]
    )
    # Backward-compatible aliases preserve Notebook M's three-class contract.
    result["row_target_valid"] = result["direction_row_target_valid"]
    result["incomplete_window"] = result["direction_incomplete_window"]
    result["model_target_valid"] = result["direction_target_valid"]
    return result


def _drop_feature(name: str) -> bool:
    return name in _DROP_EXACT or any(
        name == prefix or name.startswith(prefix + "_") for prefix in _DROP_PREFIXES
    )


def _drop_opportunity_feature(name: str) -> bool:
    """Return True for features whose meaning depends on LONG versus SHORT."""
    if name == "side_sign" or "_side" in name:
        return True
    directional_prefixes = (
        "distance_adverse_rail_bps",
        "distance_favourable_rail_bps",
        "running_favourable_excursion_bps",
        "running_recovery_bps",
        "bars_since_adverse_extreme",
        "raw_signed_favourable_excursion_bps",
        "raw_signed_recovery_bps",
        "price_oi_interaction",
    )
    if any(name == prefix or name.startswith(prefix + "_") for prefix in directional_prefixes):
        return True
    return name in {
        "window_reason_code",
        "retest_count",
        "structural_risk_bps_known",
        "rail_room_r_known",
    }


def build_large_move_dataset(
    base: TailDecisionDataset,
    labels: pd.DataFrame,
    *,
    feature_set: str = "base",
) -> LargeMoveDecisionDataset:
    """Align adaptive labels and build either base or volatility feature sets."""
    if feature_set not in {"base", "directional", "volatility"}:
        raise ValueError("feature_set must be base, directional or volatility")
    keys = ["window_id", "step"]
    expected = base.decisions[keys].reset_index(drop=True)
    work = labels.reset_index(drop=True).copy()
    if work.duplicated(keys).any() or not expected.equals(work[keys]):
        raise ValueError("adaptive labels must align row-for-row with base decisions")
    keep = [not _drop_feature(name) for name in base.tabular_features]
    kept_names = [
        name for name, retained in zip(base.tabular_features, keep, strict=True) if retained
    ]
    dropped = tuple(
        name for name, retained in zip(base.tabular_features, keep, strict=True) if not retained
    )
    matrix = np.asarray(base.tabular[:, keep], dtype=np.float32)
    additions = ["adaptive_barrier_bps"]
    if feature_set == "volatility":
        additions = list(VOLATILITY_FEATURE_COLUMNS)
    extra = work[additions].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    matrices = [matrix, extra]
    names = [*kept_names, *additions]
    if feature_set in {"directional", "volatility"}:
        lookup = {name: index for index, name in enumerate(base.tabular_features)}
        side_sign = base.tabular[:, lookup["side_sign"]].astype(np.float32)
        raw_columns: list[np.ndarray] = []
        raw_names: list[str] = []
        for source, output in _RAW_DIRECTIONAL_SOURCES.items():
            if source in lookup:
                raw_columns.append(
                    base.tabular[:, lookup[source]].astype(np.float32) * side_sign
                )
                raw_names.append(output)
        for source, output in _RAW_DIRECTIONAL_SOURCES.items():
            for horizon in (3, 6, 12, 24):
                for statistic in ("delta", "mean"):
                    summary_source = f"{source}_{statistic}_{horizon}"
                    if summary_source not in lookup:
                        continue
                    summary_output = f"{output}_{statistic}_{horizon}"
                    if summary_output in raw_names or summary_output in names:
                        continue
                    raw_columns.append(
                        base.tabular[:, lookup[summary_source]].astype(np.float32)
                        * side_sign
                    )
                    raw_names.append(summary_output)
        if raw_columns:
            matrices.append(np.column_stack(raw_columns).astype(np.float32))
            names.extend(raw_names)
        custom: list[np.ndarray] = []
        custom_names: list[str] = []
        if "channel_position_side" in lookup:
            oriented = base.tabular[:, lookup["channel_position_side"]].astype(np.float32)
            custom.append(np.where(side_sign > 0, oriented, 1.0 - oriented))
            custom_names.append("raw_channel_position")
            for horizon in (3, 6, 12, 24):
                delta_name = f"channel_position_side_delta_{horizon}"
                mean_name = f"channel_position_side_mean_{horizon}"
                if delta_name in lookup:
                    custom.append(
                        base.tabular[:, lookup[delta_name]].astype(np.float32) * side_sign
                    )
                    custom_names.append(f"raw_channel_position_delta_{horizon}")
                if mean_name in lookup:
                    oriented_mean = base.tabular[:, lookup[mean_name]].astype(np.float32)
                    custom.append(
                        np.where(side_sign > 0, oriented_mean, 1.0 - oriented_mean)
                    )
                    custom_names.append(f"raw_channel_position_mean_{horizon}")
        if {
            "distance_adverse_rail_bps",
            "distance_favourable_rail_bps",
        }.issubset(lookup):
            adverse = base.tabular[:, lookup["distance_adverse_rail_bps"]].astype(np.float32)
            favourable = base.tabular[:, lookup["distance_favourable_rail_bps"]].astype(np.float32)
            custom.extend(
                (
                    np.where(side_sign > 0, adverse, favourable),
                    np.where(side_sign > 0, -favourable, -adverse),
                )
            )
            custom_names.extend(("raw_distance_lower_bps", "raw_distance_upper_bps"))
            for horizon in (3, 6, 12, 24):
                for statistic in ("delta", "mean", "std"):
                    adverse_name = f"distance_adverse_rail_bps_{statistic}_{horizon}"
                    favourable_name = f"distance_favourable_rail_bps_{statistic}_{horizon}"
                    if adverse_name not in lookup or favourable_name not in lookup:
                        continue
                    adverse_summary = base.tabular[:, lookup[adverse_name]].astype(np.float32)
                    favourable_summary = base.tabular[:, lookup[favourable_name]].astype(np.float32)
                    if statistic == "std":
                        lower = np.where(side_sign > 0, adverse_summary, favourable_summary)
                        upper = np.where(side_sign > 0, favourable_summary, adverse_summary)
                    else:
                        lower = np.where(side_sign > 0, adverse_summary, favourable_summary)
                        upper = np.where(side_sign > 0, -favourable_summary, -adverse_summary)
                    custom.extend((lower, upper))
                    custom_names.extend(
                        (
                            f"raw_distance_lower_bps_{statistic}_{horizon}",
                            f"raw_distance_upper_bps_{statistic}_{horizon}",
                        )
                    )
        if "rsi_channel_side" in lookup:
            oriented = base.tabular[:, lookup["rsi_channel_side"]].astype(np.float32)
            custom.append(np.where(side_sign > 0, oriented, 100.0 - oriented))
            custom_names.append("raw_rsi_channel")
        if custom:
            matrices.append(np.column_stack(custom).astype(np.float32))
            names.extend(custom_names)
    if len(names) != len(set(names)):
        raise AssertionError("large-move feature names must be unique")
    tabular = np.column_stack(matrices).astype(np.float32, copy=False)
    return LargeMoveDecisionDataset(
        decisions=work,
        tabular=tabular,
        tabular_features=tuple(names),
        dropped_features=dropped,
        feature_set=feature_set,
    )


def build_opportunity_dataset(
    base: TailDecisionDataset,
    labels: pd.DataFrame,
    *,
    include_volatility: bool,
) -> LargeMoveDecisionDataset:
    """Build one side-neutral magnitude matrix from physical causal inputs."""
    contracted = add_opportunity_target_contract(labels)
    source = build_large_move_dataset(
        base,
        contracted,
        feature_set="volatility" if include_volatility else "directional",
    )
    keep = [not _drop_opportunity_feature(name) for name in source.tabular_features]
    names = tuple(
        name
        for name, retained in zip(source.tabular_features, keep, strict=True)
        if retained
    )
    removed = tuple(
        name
        for name, retained in zip(source.tabular_features, keep, strict=True)
        if not retained
    )
    if not names or len(names) != len(set(names)):
        raise AssertionError("opportunity features must be non-empty and unique")
    if any(_drop_opportunity_feature(name) for name in names):
        raise AssertionError("side-dependent feature survived opportunity isolation")
    dropped = tuple(dict.fromkeys((*source.dropped_features, *removed)))
    return LargeMoveDecisionDataset(
        decisions=source.decisions,
        tabular=np.asarray(source.tabular[:, keep], dtype=np.float32),
        tabular_features=names,
        dropped_features=dropped,
        feature_set=(
            "opportunity_side_neutral_volatility"
            if include_volatility
            else "opportunity_side_neutral"
        ),
    )


__all__ = [
    "AdaptiveMoveConfig",
    "LargeMoveDecisionDataset",
    "MOVE_CODES",
    "OPPORTUNITY_CODES",
    "VOLATILITY_FEATURE_COLUMNS",
    "add_opportunity_target_contract",
    "build_large_move_dataset",
    "build_opportunity_dataset",
    "label_adaptive_large_moves",
]
