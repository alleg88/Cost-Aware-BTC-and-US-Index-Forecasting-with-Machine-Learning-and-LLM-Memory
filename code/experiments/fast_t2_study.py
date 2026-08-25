"""Matched Fast-T2 screen for the direct five-minute channel experiment.

The strict arm changes only at phase two: a T1 made from a completed 5m bar is
confirmed by the first completed native 1m close. The channel, T1 geometry,
15-minute confirmation horizon, and 60-minute cooldown remain frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.five_minute_two_trigger_windows import (
    DATA_PATH,
    DEV_END,
    DEV_START,
    DirectChannelConfig,
    TwoTriggerConfig,
    build_direct_channel_frame,
    detect_fast_two_trigger_windows,
    detect_two_trigger_windows,
    load_five_minute_bars,
    summarise_windows,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
MINUTE_PATH = CODE_ROOT / "data" / "btcusdt_1m_2021_2026.parquet"
OUT_DIR = CODE_ROOT / "experiments" / "cache" / "channel_5m_two_trigger" / "fast_t2"
CHANNEL_CACHE = OUT_DIR / "channel_context.parquet"
EXAMPLE_FIGURE = CODE_ROOT / "notebooks" / "artifacts" / "C_5m_fast_t2_actual.png"
FREQUENCY_FIGURE = CODE_ROOT / "notebooks" / "artifacts" / "C_5m_fast_t2_comparison.png"


def load_minute_bars(
    *,
    start: pd.Timestamp = DEV_START,
    end: pd.Timestamp = DEV_END,
) -> pd.DataFrame:
    """Load native 1m OHLC strictly inside the declared dev period."""
    if not MINUTE_PATH.exists():
        raise FileNotFoundError(MINUTE_PATH)
    minute = pd.read_parquet(
        MINUTE_PATH,
        columns=["open", "high", "low", "close", "count"],
        filters=[("timestamp", ">=", start), ("timestamp", "<", end)],
    ).sort_index()
    minute = minute[(minute.index >= start) & (minute.index < end)]
    if minute.index.has_duplicates:
        raise ValueError("native 1m source has duplicate timestamps")
    return minute


def _yearly_table(windows: pd.DataFrame) -> pd.DataFrame:
    yearly = (
        windows.assign(year=windows["t2_time"].dt.year)
        .groupby(["year", "side"]).size().unstack(fill_value=0)
        .reindex(columns=["long", "short"], fill_value=0)
    )
    yearly["total"] = yearly.sum(axis=1)
    exposure = pd.Series(1, index=pd.date_range(DEV_START, DEV_END, freq="D", inclusive="left"))
    days = exposure.groupby(exposure.index.year).size()
    yearly["per_day"] = yearly["total"] / days
    return yearly


def _lag_counts(windows: pd.DataFrame) -> dict[str, int]:
    counts = windows["confirmation_lag_minutes"].value_counts().sort_index()
    return {str(int(lag)): int(count) for lag, count in counts.items()}


def _strict_confirmation_for_same_t1(
    event: pd.Series,
    channel_by_time: pd.DataFrame,
) -> pd.Timestamp | None:
    future = channel_by_time[
        (channel_by_time.index > event.t1_time)
        & (channel_by_time.index <= event.t1_time + pd.Timedelta(minutes=15))
    ]
    if event.side == "long":
        hit = future[future["close"] > event.t1_high]
    else:
        hit = future[future["close"] < event.t1_low]
    return None if hit.empty else pd.Timestamp(hit.index[0])


def _select_example(
    windows: pd.DataFrame,
    side: str,
    channel_by_time: pd.DataFrame,
) -> tuple[pd.Series, pd.Timestamp | None]:
    candidates = windows[
        (windows["side"] == side) & (windows["confirmation_lag_minutes"] == 4)
    ]
    if candidates.empty:
        candidates = windows[windows["side"] == side]
    for _, event in candidates.iloc[len(candidates) // 3:].iterrows():
        strict_time = _strict_confirmation_for_same_t1(event, channel_by_time)
        if strict_time is None or strict_time > event.t2_time:
            return event, strict_time
    event = candidates.iloc[len(candidates) // 2]
    return event, _strict_confirmation_for_same_t1(event, channel_by_time)


def _draw_minute_candles(ax, sample: pd.DataFrame) -> None:
    from matplotlib.patches import Rectangle

    for xi, row in enumerate(sample.itertuples(index=False)):
        color = "#2563eb" if row.close >= row.open else "#d97706"
        ax.vlines(xi, row.low, row.high, color=color, linewidth=1.0, zorder=4)
        bottom = min(row.open, row.close)
        height = max(abs(row.close - row.open), float(row.close) * 0.000025)
        ax.add_patch(Rectangle(
            (xi - 0.32, bottom), 0.64, height,
            facecolor=color, edgecolor=color, linewidth=0.7, zorder=5,
        ))


def plot_actual_fast_t2(
    minute: pd.DataFrame,
    channel_frame: pd.DataFrame,
    windows: pd.DataFrame,
    *,
    output: Path = EXAMPLE_FIGURE,
) -> Path:
    """Show real long/short Fast-T2 examples whose confirmation lag is 4m."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    channel_by_time = channel_frame.set_index("decision_time", drop=False)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    for ax, side in zip(axes, ("long", "short"), strict=True):
        event, strict_time = _select_example(windows, side, channel_by_time)
        t1_row = channel_by_time.loc[event.t1_time]
        left = event.t1_time - pd.Timedelta(minutes=10)
        right = event.t1_time + pd.Timedelta(minutes=22)
        sample = minute[(minute.index >= left) & (minute.index < right)].copy()
        x = np.arange(len(sample))
        position = {timestamp: i for i, timestamp in enumerate(sample.index)}
        _draw_minute_candles(ax, sample)

        lower = float(t1_row.channel_lower)
        upper = float(t1_row.channel_upper)
        span = upper - lower
        ax.axhline(lower, color="#475569", lw=1.4)
        ax.axhline(upper, color="#475569", lw=1.4)
        if side == "long":
            edge = lower + 0.30 * span
            ax.axhspan(lower, edge, color="#facc15", alpha=0.20)
            threshold = event.t1_high * (1 + event.confirmation_buffer_bps / 10_000)
        else:
            edge = upper - 0.30 * span
            ax.axhspan(edge, upper, color="#facc15", alpha=0.20)
            threshold = event.t1_low * (1 - event.confirmation_buffer_bps / 10_000)
        ax.axhline(threshold, color="#7c3aed", ls="--", lw=1.3)

        t1_x = position[event.t1_time]
        t2_x = position[event.t2_time]
        ax.axvspan(t1_x - 5, t1_x, color="#7c3aed", alpha=0.08)
        ax.axvline(t1_x, color="#7c3aed", ls="--", lw=1.8)
        ax.axvline(t2_x, color="#ea580c", ls="--", lw=1.9)
        ax.axvspan(t2_x, len(sample) - 1, color="#64748b", alpha=0.07)
        if strict_time is not None and strict_time in position:
            ax.axvline(position[strict_time], color="#64748b", ls=":", lw=1.8)
            strict_label = f"5m close: {strict_time:%H:%M}"
        else:
            strict_label = "5m close: нет подтверждения"

        ax.annotate(
            "T1: закрылась 5m-свеча",
            xy=(t1_x, event.t1_low if side == "long" else event.t1_high),
            xytext=(t1_x - 7, event.t1_low if side == "long" else event.t1_high),
            arrowprops={"arrowstyle": "->", "color": "#7c3aed"},
            ha="right", fontsize=9, color="#6d28d9",
        )
        ax.annotate(
            f"Fast-T2: закрытие 1m\nчерез {int(event.confirmation_lag_minutes)} минуты",
            xy=(t2_x, event.confirmation_price),
            xytext=(t2_x + 2, event.confirmation_price),
            arrowprops={"arrowstyle": "->", "color": "#ea580c"},
            fontsize=9, color="#c2410c",
        )
        ax.text(
            0.99, 0.04,
            f"Вход: 1m Open {event.window_start:%H:%M} UTC\n{strict_label}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            color="#334155",
        )
        ticks = np.linspace(0, len(sample) - 1, min(8, len(sample)), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([sample.index[i].strftime("%H:%M") for i in ticks])
        ax.set_ylabel("BTCUSDT")
        ax.grid(axis="y", alpha=0.15)
        ax.set_title(
            f"{side.upper()}: фактические 1m-свечи, T1 {event.t1_time:%Y-%m-%d %H:%M} UTC",
            loc="left", fontsize=12, weight="bold",
        )

    legend = [
        Patch(facecolor="#7c3aed", alpha=0.10, label="пять минут свечи T1"),
        Line2D([0], [0], color="#7c3aed", ls="--", lw=1.8, label="T1 / уровень подтверждения"),
        Line2D([0], [0], color="#ea580c", ls="--", lw=1.9, label="Fast-T2 и вход"),
        Line2D([0], [0], color="#64748b", ls=":", lw=1.8, label="доступное строгое 5m-подтверждение"),
        Patch(facecolor="#64748b", alpha=0.08, label="cooldown 60 минут (показано частично)"),
    ]
    fig.suptitle("Fast-T2: подтверждение закрытой минутной свечой", fontsize=16, weight="bold")
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.91, bottom=0.10, hspace=0.30, left=0.07, right=0.98)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def plot_comparison(
    summaries: dict[str, dict[str, object]],
    windows_by_variant: dict[str, pd.DataFrame],
    *,
    output: Path = FREQUENCY_FIGURE,
) -> Path:
    """Compare opportunity supply and Fast-T2 confirmation latency."""
    import matplotlib.pyplot as plt

    labels = ["Strict 5m", "Fast 1m\n0 bps", "Fast 1m\n+2 bps"]
    keys = ["strict_5m", "fast_0bps", "fast_2bps"]
    colors = ["#64748b", "#2563eb", "#d97706"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [0.82, 1.55]})

    per_day = [float(summaries[key]["windows_per_day"]) for key in keys]
    totals = [int(summaries[key]["windows_total"]) for key in keys]
    bars = axes[0].bar(labels, per_day, color=colors, edgecolor="#334155", linewidth=0.6)
    axes[0].bar_label(
        bars,
        labels=[f"{value:.2f}/день\nN={total:,}" for value, total in zip(per_day, totals)],
        padding=4, fontsize=9,
    )
    axes[0].set_ylim(0, max(per_day) * 1.22)
    axes[0].set_ylabel("подтверждённых окон в день")
    axes[0].set_title("Частота окон", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.16)

    lag_index = np.arange(1, 16)
    width = 0.38
    for offset, key, color, label in (
        (-width / 2, "fast_0bps", colors[1], "Fast 0 bps"),
        (width / 2, "fast_2bps", colors[2], "Fast +2 bps"),
    ):
        counts = windows_by_variant[key]["confirmation_lag_minutes"].value_counts()
        values = np.array([int(counts.get(value, 0)) for value in lag_index])
        axes[1].bar(lag_index + offset, values, width=width, color=color, label=label)
    axes[1].axvline(4, color="#334155", ls="--", lw=1.2, label="пример: 4 минуты")
    axes[1].set_xticks(lag_index)
    axes[1].set_xlabel("минут после закрытия T1")
    axes[1].set_ylabel("число подтверждений")
    axes[1].set_title("Когда приходит Fast-T2", loc="left", weight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.16)

    fig.suptitle(
        "Strict 5m и Fast-T2: matched frequency screen, dev 2021-01—2025-06",
        fontsize=15, weight="bold",
    )
    fig.text(
        0.5, 0.015,
        "Один и тот же прямой канал 60/90/120 и T1; меняется только частота подтверждения T2.",
        ha="center", fontsize=9, color="#475569",
    )
    fig.subplots_adjust(top=0.84, bottom=0.18, left=0.07, right=0.98, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def run() -> dict[str, object]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if CHANNEL_CACHE.exists():
        channel_frame = pd.read_parquet(CHANNEL_CACHE)
        print(f"loaded cached channel context: {len(channel_frame):,} rows", flush=True)
    else:
        bars = load_five_minute_bars()
        print(f"loaded {len(bars):,} five-minute bars from {DATA_PATH.name}", flush=True)
        channel_frame = build_direct_channel_frame(bars, config=DirectChannelConfig())
        channel_frame.to_parquet(CHANNEL_CACHE, index=False)
        print(f"saved {CHANNEL_CACHE}", flush=True)

    minute = load_minute_bars()
    print(f"loaded {len(minute):,} native one-minute bars", flush=True)
    config = TwoTriggerConfig()
    strict = detect_two_trigger_windows(channel_frame, config=config)
    fast_zero = detect_fast_two_trigger_windows(
        channel_frame, minute, config=config, confirmation_buffer_bps=0.0
    )
    fast_two = detect_fast_two_trigger_windows(
        channel_frame, minute, config=config, confirmation_buffer_bps=2.0
    )
    results = {
        "strict_5m": strict,
        "fast_0bps": fast_zero,
        "fast_2bps": fast_two,
    }
    summaries: dict[str, dict[str, object]] = {}
    windows_by_variant: dict[str, pd.DataFrame] = {}
    for key, result in results.items():
        summary, daily = summarise_windows(result, start=DEV_START, end=DEV_END)
        if key.startswith("fast"):
            summary["confirmation_lag_minutes"] = _lag_counts(result.windows)
            summary["confirmations_at_four_minutes"] = int(
                (result.windows["confirmation_lag_minutes"] == 4).sum()
            )
            summary["median_confirmation_minutes"] = float(
                result.windows["confirmation_lag_minutes"].median()
            )
        summary["source"] = "native 1m" if key.startswith("fast") else "completed 5m"
        summaries[key] = summary
        windows_by_variant[key] = result.windows
        variant_dir = OUT_DIR / key
        variant_dir.mkdir(parents=True, exist_ok=True)
        result.windows.to_parquet(variant_dir / "window_manifest.parquet", index=False)
        daily.to_csv(variant_dir / "daily_counts.csv", index_label="day")
        _yearly_table(result.windows).to_csv(variant_dir / "yearly_counts.csv")
        (variant_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    matched_zero = fast_zero.windows.merge(
        strict.windows[["side", "t1_time"]].drop_duplicates(),
        on=["side", "t1_time"], how="inner",
    )
    comparison: dict[str, object] = {
        "period_start": DEV_START.isoformat(),
        "period_end_exclusive": DEV_END.isoformat(),
        "strict_5m": summaries["strict_5m"],
        "fast_0bps": summaries["fast_0bps"],
        "fast_2bps": summaries["fast_2bps"],
        "fast0_strict_same_t1": int(len(matched_zero)),
        "methodology": {
            "t1": "completed 5m edge touch with wick share >=30%",
            "fast_t2": "first completed native 1m close beyond frozen T1 extreme",
            "confirmation_horizon_minutes": 15,
            "cooldown_minutes": 60,
            "entry": "Open of the next native 1m bar at the T2 decision boundary",
        },
    }
    (OUT_DIR / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_actual_fast_t2(minute, channel_frame, fast_two.windows)
    plot_comparison(summaries, windows_by_variant)
    print(json.dumps(comparison, indent=2, ensure_ascii=False), flush=True)
    return comparison


if __name__ == "__main__":
    run()
