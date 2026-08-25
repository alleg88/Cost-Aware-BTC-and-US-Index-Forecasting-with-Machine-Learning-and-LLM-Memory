"""Causal two-trigger opportunity windows inside direct 5-minute channels.

Trigger 1 (T1) is a wick rejection at the channel edge. Trigger 2 (T2) is a
close through the T1 candle's opposite extreme on one of the next completed
bars. A model may act only from the bar after T2; no future pivot is used.
"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from features.linear_channels import (
    channel_confluence,
    channel_episode_id,
    compute_linear_regression_channels,
    label_channel_regime,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = CODE_ROOT / "data" / "btcusdt_5min_2021_2026.parquet"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "channel_5m_two_trigger" / "cooldown60"
FIGURE_PATH = CODE_ROOT / "notebooks" / "artifacts" / "C_5m_two_trigger_actual.png"
DEV_START = pd.Timestamp("2021-01-01", tz="UTC")
DEV_END = pd.Timestamp("2025-07-01", tz="UTC")


@dataclass(frozen=True)
class TwoTriggerConfig:
    """Frozen screen parameters for the first 5-minute feasibility count."""

    edge_zone: float = 0.30
    min_wick_share: float = 0.30
    confirmation_bars: int = 3
    cooldown_minutes: int = 60
    cadence_minutes: int = 5
    min_r2: float = 0.20


@dataclass(frozen=True)
class DirectChannelConfig:
    windows: tuple[int, ...] = (60, 90, 120)
    primary_window: int = 60
    min_agree: int = 2
    min_r2: float = 0.20
    min_slope_bps_per_hour: float = 5.0
    persist_bars: int = 6
    quantile: float = 0.10


@dataclass(frozen=True)
class TwoTriggerResult:
    windows: pd.DataFrame
    audit: dict[str, int]


WINDOW_COLUMNS = (
    "side",
    "channel_episode_id",
    "t1_time",
    "t2_time",
    "window_start",
    "cooldown_until",
    "confirmation_lag_bars",
    "t1_high",
    "t1_low",
)

FAST_WINDOW_COLUMNS = (
    "side",
    "channel_episode_id",
    "t1_time",
    "t2_time",
    "window_start",
    "cooldown_until",
    "confirmation_lag_minutes",
    "confirmation_buffer_bps",
    "confirmation_price",
    "t1_high",
    "t1_low",
)


def _valid_channel_row(row: object, config: TwoTriggerConfig) -> bool:
    required = (
        "open", "high", "low", "close", "channel_lower", "channel_upper",
        "channel_r2", "channel_confluence", "channel_episode_id",
    )
    if not all(np.isfinite(float(getattr(row, name))) for name in required):
        return False
    if hasattr(row, "minute_count") and int(row.minute_count) != config.cadence_minutes:
        return False
    return (
        float(row.channel_upper) > float(row.channel_lower)
        and float(row.channel_r2) >= config.min_r2
        and bool(row.channel_confluence)
        and row.channel_regime in {"up", "down"}
    )


def _touch_side(row: object, config: TwoTriggerConfig) -> str | None:
    if not _valid_channel_row(row, config):
        return None
    candle_range = float(row.high) - float(row.low)
    if candle_range <= 0:
        return None
    channel_span = float(row.channel_upper) - float(row.channel_lower)
    low_pos = (float(row.low) - float(row.channel_lower)) / channel_span
    high_pos = (float(row.high) - float(row.channel_lower)) / channel_span
    lower_wick = (min(float(row.open), float(row.close)) - float(row.low)) / candle_range
    upper_wick = (float(row.high) - max(float(row.open), float(row.close))) / candle_range

    if row.channel_regime == "up":
        if low_pos <= config.edge_zone and lower_wick >= config.min_wick_share:
            return "long"
    elif high_pos >= 1.0 - config.edge_zone and upper_wick >= config.min_wick_share:
        return "short"
    return None


def detect_two_trigger_windows(
    frame: pd.DataFrame,
    *,
    config: TwoTriggerConfig,
) -> TwoTriggerResult:
    """Detect T1->T2 windows with a global post-confirmation cooldown.

    ``decision_time`` is the close time of each completed 5-minute candle.
    Confirmation is allowed only on the next ``confirmation_bars`` rows and in
    the same channel episode. The resulting ``window_start`` is one cadence
    after T2, which prevents trading on information from an unclosed candle.
    """
    needed = {
        "decision_time", "open", "high", "low", "close", "channel_lower",
        "channel_upper", "channel_r2", "channel_regime",
        "channel_confluence", "channel_episode_id",
    }
    missing = sorted(needed.difference(frame.columns))
    if missing:
        raise ValueError(f"two-trigger frame missing columns: {missing}")

    ordered = frame.sort_values("decision_time", kind="stable").reset_index(drop=True)
    times = pd.to_datetime(ordered["decision_time"], utc=True)
    if times.duplicated().any():
        raise ValueError("decision_time must be unique")

    audit = {
        "candidate_touches": 0,
        "armed_t1": 0,
        "confirmed_t2": 0,
        "expired_t1": 0,
        "invalidated_t1": 0,
        "cooldown_touches": 0,
    }
    windows: list[dict[str, object]] = []
    armed: dict[str, object] | None = None
    cooldown_until: pd.Timestamp | None = None
    cadence = pd.Timedelta(minutes=config.cadence_minutes)
    cooldown = pd.Timedelta(minutes=config.cooldown_minutes)

    for position, row in enumerate(ordered.itertuples(index=False)):
        now = times.iloc[position]
        touch_side = _touch_side(row, config)
        if touch_side is not None:
            audit["candidate_touches"] += 1

        if cooldown_until is not None and now < cooldown_until:
            if touch_side is not None:
                audit["cooldown_touches"] += 1
            armed = None
            continue

        if armed is not None:
            lag = position - int(armed["position"])
            same_context = (
                _valid_channel_row(row, config)
                and row.channel_regime == armed["regime"]
                and row.channel_episode_id == armed["episode"]
            )
            if not same_context:
                audit["invalidated_t1"] += 1
                armed = None
            elif lag > config.confirmation_bars:
                audit["expired_t1"] += 1
                armed = None
            else:
                confirmed = (
                    (armed["side"] == "long" and float(row.close) > float(armed["high"]))
                    or (armed["side"] == "short" and float(row.close) < float(armed["low"]))
                )
                if confirmed:
                    cooldown_until = now + cooldown
                    windows.append(
                        {
                            "side": armed["side"],
                            "channel_episode_id": armed["episode"],
                            "t1_time": armed["time"],
                            "t2_time": now,
                            "window_start": now + cadence,
                            "cooldown_until": cooldown_until,
                            "confirmation_lag_bars": lag,
                            "t1_high": armed["high"],
                            "t1_low": armed["low"],
                        }
                    )
                    audit["confirmed_t2"] += 1
                    armed = None
                    continue

        # A bar that expired or invalidated an old setup may start a fresh one.
        if armed is None and touch_side is not None:
            armed = {
                "side": touch_side,
                "position": position,
                "time": now,
                "high": float(row.high),
                "low": float(row.low),
                "episode": row.channel_episode_id,
                "regime": row.channel_regime,
            }
            audit["armed_t1"] += 1

    out = pd.DataFrame(windows, columns=WINDOW_COLUMNS)
    return TwoTriggerResult(windows=out, audit=audit)


def detect_fast_two_trigger_windows(
    channel_frame: pd.DataFrame,
    minute_bars: pd.DataFrame,
    *,
    config: TwoTriggerConfig,
    confirmation_buffer_bps: float = 0.0,
) -> TwoTriggerResult:
    """Confirm a completed-5m T1 with the first eligible completed 1m close.

    The channel and T1 geometry are frozen at ``t1_time``. Minute-bar index
    values are bar opens, so a row stamped 00:08 is known at 00:09 and can only
    be entered at 00:09, the next one-minute open.
    """
    if confirmation_buffer_bps < 0:
        raise ValueError("confirmation_buffer_bps must be non-negative")
    if not isinstance(minute_bars.index, pd.DatetimeIndex):
        raise TypeError("minute_bars needs a DatetimeIndex of bar opens")
    if "close" not in minute_bars:
        raise ValueError("minute_bars missing close")

    channels = channel_frame.sort_values("decision_time", kind="stable").reset_index(drop=True)
    channel_times = pd.to_datetime(channels["decision_time"], utc=True)
    if channel_times.duplicated().any():
        raise ValueError("channel decision_time must be unique")
    touch_at: dict[pd.Timestamp, object] = {}
    candidate_touches = 0
    for position, row in enumerate(channels.itertuples(index=False)):
        side = _touch_side(row, config)
        if side is None:
            continue
        candidate_touches += 1
        touch_at[channel_times.iloc[position]] = (row, side)

    minute = minute_bars.sort_index(kind="stable")
    if minute.index.has_duplicates:
        raise ValueError("minute_bars index must be unique")
    minute_open = pd.DatetimeIndex(pd.to_datetime(minute.index, utc=True))
    minute_decision = minute_open + pd.Timedelta(minutes=1)

    audit = {
        "candidate_touches": candidate_touches,
        "armed_t1": 0,
        "confirmed_t2": 0,
        "expired_t1": 0,
        "cooldown_touches": 0,
        "overlapping_touches": 0,
        "missing_minute_invalidations": 0,
    }
    windows: list[dict[str, object]] = []
    armed: dict[str, object] | None = None
    cooldown_until: pd.Timestamp | None = None
    confirmation_limit = pd.Timedelta(
        minutes=config.confirmation_bars * config.cadence_minutes
    )
    one_minute = pd.Timedelta(minutes=1)
    cooldown = pd.Timedelta(minutes=config.cooldown_minutes)
    buffer = confirmation_buffer_bps / 10_000.0

    for row, now in zip(minute.itertuples(index=False), minute_decision, strict=True):
        if armed is not None:
            expected = pd.Timestamp(armed["last_minute_time"]) + one_minute
            if now > expected:
                audit["missing_minute_invalidations"] += 1
                armed = None

        if armed is not None:
            expiry = pd.Timestamp(armed["time"]) + confirmation_limit
            if now <= expiry:
                threshold = (
                    float(armed["high"]) * (1.0 + buffer)
                    if armed["side"] == "long"
                    else float(armed["low"]) * (1.0 - buffer)
                )
                confirmed = (
                    (armed["side"] == "long" and float(row.close) > threshold)
                    or (armed["side"] == "short" and float(row.close) < threshold)
                )
                armed["last_minute_time"] = now
                if confirmed:
                    cooldown_until = now + cooldown
                    lag = int((now - pd.Timestamp(armed["time"])) / one_minute)
                    windows.append(
                        {
                            "side": armed["side"],
                            "channel_episode_id": armed["episode"],
                            "t1_time": armed["time"],
                            "t2_time": now,
                            "window_start": now,
                            "cooldown_until": cooldown_until,
                            "confirmation_lag_minutes": lag,
                            "confirmation_buffer_bps": float(confirmation_buffer_bps),
                            "confirmation_price": float(row.close),
                            "t1_high": armed["high"],
                            "t1_low": armed["low"],
                        }
                    )
                    audit["confirmed_t2"] += 1
                    armed = None
            if armed is not None and now >= expiry:
                audit["expired_t1"] += 1
                armed = None

        candidate = touch_at.get(now)
        if candidate is None:
            continue
        channel_row, side = candidate
        if cooldown_until is not None and now < cooldown_until:
            audit["cooldown_touches"] += 1
            continue
        if armed is not None:
            audit["overlapping_touches"] += 1
            continue
        armed = {
            "side": side,
            "time": now,
            "high": float(channel_row.high),
            "low": float(channel_row.low),
            "episode": channel_row.channel_episode_id,
            "last_minute_time": now,
        }
        audit["armed_t1"] += 1

    out = pd.DataFrame(windows, columns=FAST_WINDOW_COLUMNS)
    return TwoTriggerResult(windows=out, audit=audit)


def load_five_minute_bars(
    *,
    start: pd.Timestamp = DEV_START,
    end: pd.Timestamp = DEV_END,
) -> pd.DataFrame:
    """Read only the declared period; the Q2-2026 lockbox is never loaded."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)
    import pyarrow.parquet as pq

    schema = pq.read_schema(DATA_PATH).names
    index_field = "timestamp" if "timestamp" in schema else "__index_level_0__"
    bars = pd.read_parquet(
        DATA_PATH,
        columns=["open", "high", "low", "close", "minute_count"],
        filters=[(index_field, ">=", start), (index_field, "<", end)],
    ).sort_index()
    bars = bars[(bars.index >= start) & (bars.index < end)]
    if bars.index.has_duplicates:
        raise ValueError("5-minute source has duplicate timestamps")
    return bars


def build_direct_channel_frame(
    bars: pd.DataFrame,
    *,
    config: DirectChannelConfig = DirectChannelConfig(),
) -> pd.DataFrame:
    """Build three causal direct-5m channels and their majority regime."""
    if config.primary_window not in config.windows:
        raise ValueError("primary_window must be one of windows")

    regimes: dict[str, pd.Series] = {}
    primary: pd.DataFrame | None = None
    # 5 bps/hour becomes 5/12 bps per 5-minute bar.
    slope_threshold = config.min_slope_bps_per_hour * 5.0 / 60.0 / 10_000.0
    for window in config.windows:
        print(f"fitting direct 5m channel: {window} bars", flush=True)
        fitted = compute_linear_regression_channels(
            bars,
            window=window,
            log_price=True,
            method="quantile",
            quantile=config.quantile,
            require_complete_bars=True,
        )
        name = str(window)
        regimes[name] = label_channel_regime(
            fitted,
            min_slope=slope_threshold,
            min_r2=config.min_r2,
            persist_bars=config.persist_bars,
        )
        if window == config.primary_window:
            primary = fitted[[
                "open", "high", "low", "close", "minute_count",
                "channel_lower", "channel_upper", "channel_r2",
            ]].copy()
        del fitted
        gc.collect()

    assert primary is not None
    regime_frame = pd.DataFrame(regimes, index=bars.index)
    agreement = channel_confluence(
        regime_frame,
        primary=str(config.primary_window),
        min_agree=config.min_agree,
    )
    primary_regime = regime_frame[str(config.primary_window)]
    effective_regime = primary_regime.where(
        agreement["channel_confluence"].astype(bool), "none"
    )
    primary["channel_regime"] = effective_regime
    primary["channel_confluence"] = agreement["channel_confluence"]
    primary["channel_confluence_count"] = agreement["channel_confluence_count"]
    primary["channel_episode_id"] = channel_episode_id(effective_regime)
    # Source timestamps are bar opens. All row values are known five minutes later.
    primary["decision_time"] = primary.index + pd.Timedelta(minutes=5)
    return primary.reset_index(drop=False)


def summarise_windows(
    result: TwoTriggerResult,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame]:
    days = pd.date_range(start.normalize(), end.normalize(), freq="D", inclusive="left")
    if result.windows.empty:
        observed = pd.Series(dtype="int64")
    else:
        observed = (
            result.windows.assign(day=result.windows["t2_time"].dt.floor("D"))
            .groupby("day").size()
        )
    daily = observed.reindex(days, fill_value=0).rename("windows").to_frame()
    total = int(daily["windows"].sum())
    by_side = result.windows["side"].value_counts() if total else pd.Series(dtype=int)
    summary: dict[str, object] = {
        "period_start": start.isoformat(),
        "period_end_exclusive": end.isoformat(),
        "calendar_days": int(len(daily)),
        "windows_total": total,
        "windows_long": int(by_side.get("long", 0)),
        "windows_short": int(by_side.get("short", 0)),
        "windows_per_day": float(total / len(daily)),
        "days_with_window": int((daily["windows"] > 0).sum()),
        "day_coverage": float((daily["windows"] > 0).mean()),
        "days_zero": int((daily["windows"] == 0).sum()),
        "days_one": int((daily["windows"] == 1).sum()),
        "days_two": int((daily["windows"] == 2).sum()),
        "days_three_plus": int((daily["windows"] >= 3).sum()),
        "median_windows_per_day": float(daily["windows"].median()),
        "p90_windows_per_day": float(daily["windows"].quantile(0.90)),
        "max_windows_per_day": int(daily["windows"].max()),
        "unique_channel_episodes": int(
            result.windows["channel_episode_id"].nunique() if total else 0
        ),
        "audit": {key: int(value) for key, value in result.audit.items()},
    }
    return summary, daily


def _draw_candles(ax, sample: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    x = np.arange(len(sample))
    for xi, row in enumerate(sample.itertuples(index=False)):
        color = "#159e75" if row.close >= row.open else "#dc5a5a"
        ax.vlines(xi, row.low, row.high, color=color, linewidth=1.0, zorder=4)
        bottom = min(row.open, row.close)
        height = max(abs(row.close - row.open), float(row.close) * 0.00005)
        ax.add_patch(Rectangle(
            (xi - 0.31, bottom), 0.62, height,
            facecolor=color, edgecolor=color, linewidth=0.8, zorder=5,
        ))


def _representative_window(windows: pd.DataFrame, side: str) -> pd.Series:
    candidates = windows[windows["side"] == side]
    preferred = candidates[candidates["confirmation_lag_bars"] >= 2]
    if not preferred.empty:
        candidates = preferred
    return candidates.iloc[len(candidates) // 2]


def plot_two_trigger_examples(
    channel_frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    output: Path = FIGURE_PATH,
) -> Path:
    """Draw one actual long and one actual short T1/T2 event."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    if not {"long", "short"}.issubset(set(windows["side"])):
        raise ValueError("both long and short examples are required for the figure")
    context = channel_frame.set_index("decision_time", drop=False)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=False)

    for ax, side in zip(axes, ("long", "short"), strict=True):
        event = _representative_window(windows, side)
        left = event.t1_time - pd.Timedelta(minutes=60)
        right = event.t2_time + pd.Timedelta(minutes=75)
        sample = context[(context.index >= left) & (context.index <= right)].copy()
        sample = sample[np.isfinite(sample["channel_lower"])]
        x = np.arange(len(sample))
        time_to_x = {value: index for index, value in enumerate(sample.index)}
        _draw_candles(ax, sample)
        ax.plot(x, sample["channel_lower"], color="#2563eb", lw=1.7, zorder=2)
        ax.plot(x, sample["channel_upper"], color="#2563eb", lw=1.7, zorder=2)
        span = sample["channel_upper"] - sample["channel_lower"]
        if side == "long":
            edge = sample["channel_lower"] + 0.30 * span
            ax.fill_between(x, sample["channel_lower"], edge, color="#fde68a", alpha=0.55)
            edge_label = "нижние 30% канала"
        else:
            edge = sample["channel_upper"] - 0.30 * span
            ax.fill_between(x, edge, sample["channel_upper"], color="#fde68a", alpha=0.55)
            edge_label = "верхние 30% канала"

        t1_x = time_to_x[event.t1_time]
        t2_x = time_to_x[event.t2_time]
        start_x = time_to_x.get(event.window_start, t2_x + 1)
        cooldown_x = min(t2_x + 12, len(sample) - 1)
        ax.axvline(t1_x, color="#7c3aed", ls="--", lw=1.7, zorder=6)
        ax.axvline(t2_x, color="#ea580c", ls="--", lw=1.7, zorder=6)
        ax.axvline(start_x, color="#15803d", ls=":", lw=2.0, zorder=6)
        ax.axvspan(t2_x, cooldown_x, color="#6b7280", alpha=0.10, zorder=0)
        ax.scatter(t1_x, event.t1_low if side == "long" else event.t1_high,
                   marker="o", s=70, color="#7c3aed", zorder=7)
        ax.annotate(
            "T1: касание края\n+ тенью ≥30% свечи",
            xy=(t1_x, event.t1_low if side == "long" else event.t1_high),
            xytext=(t1_x - 8, event.t1_low if side == "long" else event.t1_high),
            arrowprops={"arrowstyle": "->", "color": "#7c3aed"},
            color="#6d28d9", fontsize=9, ha="right",
        )
        comparator = "> high(T1)" if side == "long" else "< low(T1)"
        ax.annotate(
            f"T2: close {comparator}\nчерез {int(event.confirmation_lag_bars)} бар(а)",
            xy=(t2_x, sample.iloc[t2_x]["close"]),
            xytext=(t2_x + 2, sample.iloc[t2_x]["close"]),
            arrowprops={"arrowstyle": "->", "color": "#ea580c"},
            color="#c2410c", fontsize=9,
        )
        ax.text(
            start_x + 0.25,
            ax.get_ylim()[0] + 0.08 * np.ptp(ax.get_ylim()),
            "модель может торговать\nсо следующего бара",
            color="#15803d", fontsize=8.8, va="bottom",
        )
        labels_at = np.linspace(0, len(sample) - 1, min(7, len(sample)), dtype=int)
        ax.set_xticks(labels_at)
        ax.set_xticklabels(
            [sample.index[i].strftime("%d %b\n%H:%M") for i in labels_at], fontsize=8
        )
        direction = "LONG" if side == "long" else "SHORT"
        ax.set_title(
            f"{direction}: фактический BTCUSDT 5m, T1 {event.t1_time:%Y-%m-%d %H:%M} UTC",
            loc="left", fontsize=12, weight="bold",
        )
        ax.set_ylabel("BTCUSDT")
        ax.grid(axis="y", alpha=0.16)
        ax.text(0.995, 0.02, edge_label, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color="#92400e")

    legend = [
        Line2D([0], [0], color="#2563eb", lw=1.8, label="границы прямого 5m-канала"),
        Patch(facecolor="#fde68a", alpha=0.6, label="краевая зона 30%"),
        Line2D([0], [0], color="#7c3aed", ls="--", lw=1.7, label="T1: отбой тенью"),
        Line2D([0], [0], color="#ea580c", ls="--", lw=1.7, label="T2: подтверждение close"),
        Line2D([0], [0], color="#15803d", ls=":", lw=2, label="начало окна модели"),
        Patch(facecolor="#6b7280", alpha=0.12, label="cooldown 60 минут"),
    ]
    fig.suptitle(
        "Два причинных триггера в прямом 5-минутном канале",
        fontsize=16, weight="bold", y=0.985,
    )
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.92, bottom=0.11, hspace=0.32, left=0.07, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def run_dev_screen(*, draw: bool = True) -> dict[str, object]:
    channel_config = DirectChannelConfig()
    trigger_config = TwoTriggerConfig(min_r2=channel_config.min_r2)
    bars = load_five_minute_bars()
    print(f"loaded {len(bars):,} five-minute bars", flush=True)
    channel_frame = build_direct_channel_frame(bars, config=channel_config)
    result = detect_two_trigger_windows(channel_frame, config=trigger_config)
    summary, daily = summarise_windows(result, start=DEV_START, end=DEV_END)
    yearly = (
        result.windows.assign(year=result.windows["t2_time"].dt.year)
        .groupby(["year", "side"]).size().unstack(fill_value=0)
        .reindex(columns=["long", "short"], fill_value=0)
    )
    yearly["total"] = yearly.sum(axis=1)
    days_by_year = daily.assign(year=daily.index.year).groupby("year").size()
    yearly["per_day"] = yearly["total"] / days_by_year
    summary["yearly"] = {
        str(year): {
            "long": int(row.long),
            "short": int(row.short),
            "total": int(row.total),
            "per_day": float(row.per_day),
        }
        for year, row in yearly.iterrows()
    }
    summary["channel_config"] = {
        "windows_bars": list(channel_config.windows),
        "windows_hours": [value * 5 / 60 for value in channel_config.windows],
        "primary_window": channel_config.primary_window,
        "min_agree": channel_config.min_agree,
        "min_r2": channel_config.min_r2,
        "min_slope_bps_per_hour": channel_config.min_slope_bps_per_hour,
        "persist_bars": channel_config.persist_bars,
        "band_quantiles": [channel_config.quantile, 1 - channel_config.quantile],
    }
    summary["trigger_config"] = {
        "edge_zone": trigger_config.edge_zone,
        "min_wick_share": trigger_config.min_wick_share,
        "confirmation_bars": trigger_config.confirmation_bars,
        "cooldown_minutes": trigger_config.cooldown_minutes,
        "window_starts": "next completed 5m bar after T2",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.windows.to_parquet(OUT_DIR / "window_manifest.parquet", index=False)
    daily.to_csv(OUT_DIR / "daily_counts.csv", index_label="day")
    yearly.to_csv(OUT_DIR / "yearly_counts.csv", index_label="year")
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if draw:
        plot_two_trigger_examples(channel_frame, result.windows)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    run_dev_screen(draw=not args.no_plot)


if __name__ == "__main__":
    main()
