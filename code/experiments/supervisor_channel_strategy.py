"""Dev-only feasibility runner for the supervisor's 1h/5m channel strategy.

This is deliberately mechanical. It observes whether strict T1-T2-T3 patterns
naturally produce roughly 3-5 trades/day and positive after-cost economics; it
does not cap frequency, stop trading after a daily result, or train a model.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluation.channel_backtest import backtest_channel_strategy
from features.linear_channels import (
    channel_confluence,
    channel_episode_id,
    compute_linear_regression_channels,
    compute_rsi,
    label_channel_regime,
)
from features.supervisor_channel import (
    SupervisorSignalConfig,
    generate_supervisor_signals,
    project_closed_hourly_channel,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA = CODE_ROOT / "data"
DEFAULT_OUTPUT = CODE_ROOT / "experiments" / "cache" / "supervisor_channel" / "dev"
DEV_START = pd.Timestamp("2021-01-01", tz="UTC")
DEV_END = pd.Timestamp("2025-07-01", tz="UTC")


@dataclass(frozen=True)
class SupervisorChannelConfig:
    symbol: str = "BTCUSDT"
    decision_grid: str = "5min"
    channel_grid: str = "1h"
    channel_window: int = 60
    channel_windows: tuple[int, ...] = (60, 90, 120)
    diagnostic_min_agree: int = 2
    min_r2: float = 0.40
    min_slope_bps_per_hour: float = 5.0
    band_quantile: float = 0.10
    persist_hours: int = 6
    rr_multiples: tuple[float, ...] = (2.0, 3.0, 5.0)
    stop_buffer_bps: float = 5.0
    max_hold_minutes: int = 120
    cost_bps: float = 10.0
    risk_pct: float = 1.0
    frequency_target_low: float = 3.0
    frequency_target_high: float = 5.0
    signal: SupervisorSignalConfig = field(default_factory=SupervisorSignalConfig)


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def protocol_dict(
    config: SupervisorChannelConfig = SupervisorChannelConfig(), *, stage: str = "dev"
) -> dict[str, Any]:
    """Return the frozen protocol before any parquet file is opened."""
    if stage != "dev":
        raise PermissionError("Notebook J is development-only; tune/forward/lockbox stay sealed")
    return {
        "stage": "dev",
        "period_start": DEV_START.isoformat(),
        "period_end_exclusive": DEV_END.isoformat(),
        "decision_grid": config.decision_grid,
        "channel_grid": config.channel_grid,
        "channel_window": config.channel_window,
        "channel_windows_diagnostic": list(config.channel_windows),
        "daily_sma_gate": False,
        "project_hourly_slope_inside_hour": True,
        "side_dataset": "pooled",
        "direction_policy": "up channel -> LONG; down channel -> SHORT",
        "rr_multiples": list(config.rr_multiples),
        "risk_pct_per_trade": config.risk_pct,
        "max_hold_minutes": config.max_hold_minutes,
        "cost_bps_round_trip": config.cost_bps,
        "max_trades_per_day": None,
        "max_concurrent": None,
        "daily_stop_pct": None,
        "profit_shutdown_pct": None,
        "target_frequency_trades_per_day": [
            config.frequency_target_low,
            config.frequency_target_high,
        ],
        "daily_profit_levels_pct": [2.0, 3.0, 5.0],
        "forward_or_lockbox_loaded": False,
        "config": asdict(config),
    }


def _calendar_days(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(_utc(start).normalize(), _utc(end).normalize(), freq="D", inclusive="left")


def summarise_trade_frequency(
    trades: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target_low: float = 3.0,
    target_high: float = 5.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Count every filled entry; the target band is descriptive, never a cap."""
    days = _calendar_days(start, end)
    if trades.empty:
        observed = pd.Series(dtype="int64")
    else:
        entry_day = pd.to_datetime(trades["entry_time"], utc=True).dt.floor("D")
        observed = entry_day.value_counts().sort_index()
    daily = observed.reindex(days, fill_value=0).astype(int).rename("trades").to_frame()
    total = int(daily["trades"].sum())
    rate = total / len(days) if len(days) else float("nan")
    return daily, {
        "calendar_days": int(len(days)),
        "total_trades": total,
        "trades_per_day": float(rate),
        "active_days": int((daily["trades"] > 0).sum()),
        "zero_trade_days": int((daily["trades"] == 0).sum()),
        "median_trades_per_day": float(daily["trades"].median()) if len(days) else float("nan"),
        "inside_target_3_to_5": bool(target_low <= rate <= target_high),
        "target_low": float(target_low),
        "target_high": float(target_high),
        "trade_cap_applied": False,
    }


def summarise_account_pnl(
    trades: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    risk_pct: float = 1.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Report realised UTC-day PnL when one net R equals `risk_pct` percent."""
    days = _calendar_days(start, end)
    if trades.empty:
        pnl = pd.Series(dtype=float)
        counts = pd.Series(dtype="int64")
    else:
        exit_day = pd.to_datetime(trades["exit_time"], utc=True).dt.floor("D")
        trade_pnl = pd.to_numeric(trades["r_net"], errors="coerce") * risk_pct
        pnl = trade_pnl.groupby(exit_day).sum()
        counts = exit_day.value_counts().sort_index()
    daily = pd.DataFrame(index=days)
    daily.index.name = "day"
    daily["realised_pnl_pct"] = pnl.reindex(days, fill_value=0.0).astype(float)
    daily["trades_realised"] = counts.reindex(days, fill_value=0).astype(int)
    daily["cumulative_pnl_pct"] = daily["realised_pnl_pct"].cumsum()
    daily["equity_index"] = (1.0 + daily["realised_pnl_pct"] / 100.0).cumprod()
    summary: dict[str, Any] = {
        "risk_pct_per_trade": float(risk_pct),
        "mean_daily_pnl_pct": float(daily["realised_pnl_pct"].mean()) if len(days) else float("nan"),
        "median_daily_pnl_pct": float(daily["realised_pnl_pct"].median()) if len(days) else float("nan"),
        "total_fixed_risk_pnl_pct": float(daily["realised_pnl_pct"].sum()),
        "positive_days": int((daily["realised_pnl_pct"] > 0).sum()),
        "days_ge_2pct": int((daily["realised_pnl_pct"] >= 2.0).sum()),
        "days_ge_3pct": int((daily["realised_pnl_pct"] >= 3.0).sum()),
        "days_ge_5pct": int((daily["realised_pnl_pct"] >= 5.0).sum()),
        "share_days_ge_2pct": float((daily["realised_pnl_pct"] >= 2.0).mean()) if len(days) else 0.0,
        "share_days_ge_3pct": float((daily["realised_pnl_pct"] >= 3.0).mean()) if len(days) else 0.0,
        "share_days_ge_5pct": float((daily["realised_pnl_pct"] >= 5.0).mean()) if len(days) else 0.0,
        "daily_stop_applied": False,
        "profit_shutdown_applied": False,
    }
    return daily, summary


def summarise_signal_funnel(signals: pd.DataFrame) -> pd.DataFrame:
    """Return one pooled LONG/SHORT funnel while retaining side attribution."""
    rows: list[dict[str, Any]] = []
    for setup_type, prefix in (
        ("edge_rejection", "edge"),
        ("midline_retest", "midline"),
    ):
        for stage_number in (1, 2, 3):
            value_col = f"{prefix}_stage_{stage_number}"
            side_col = f"{prefix}_stage_{stage_number}_side"
            for side in ("long", "short"):
                count = int(((signals[value_col] > 0) & signals[side_col].eq(side)).sum())
                rows.append(
                    {
                        "dataset": "pooled",
                        "setup_type": setup_type,
                        "side": side,
                        "stage": f"T{stage_number}",
                        "count": count,
                    }
                )
        for side, sign in (("long", 1), ("short", -1)):
            count = int(
                (
                    signals["setup_type"].eq(setup_type)
                    & signals["signal"].eq(sign)
                ).sum()
            )
            rows.append(
                {
                    "dataset": "pooled",
                    "setup_type": setup_type,
                    "side": side,
                    "stage": "ENTRY",
                    "count": count,
                }
            )
    return pd.DataFrame(rows)


def summarise_trade_breakdown(
    trades: pd.DataFrame, *, rr_multiple: float
) -> pd.DataFrame:
    """Expose pooled economics by setup and side without creating side datasets."""
    columns = [
        "dataset",
        "rr_multiple",
        "setup_type",
        "side",
        "trades",
        "mean_gross_r",
        "mean_net_r",
        "total_net_r",
        "win_rate",
        "tp_rate",
        "sl_rate",
        "timeout_rate",
        "median_risk_bps",
        "mean_risk_bps",
    ]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    for (setup_type, side), group in trades.groupby(
        ["setup_type", "side"], sort=True, dropna=False
    ):
        outcomes = group["outcome"].value_counts(normalize=True)
        rows.append(
            {
                "dataset": "pooled",
                "rr_multiple": float(rr_multiple),
                "setup_type": setup_type,
                "side": side,
                "trades": int(len(group)),
                "mean_gross_r": float(group["r_gross"].mean()),
                "mean_net_r": float(group["r_net"].mean()),
                "total_net_r": float(group["r_net"].sum()),
                "win_rate": float((group["r_net"] > 0).mean()),
                "tp_rate": float(outcomes.get("tp", 0.0)),
                "sl_rate": float(outcomes.get("sl", 0.0)),
                "timeout_rate": float(outcomes.get("timeout", 0.0)),
                "median_risk_bps": float(group["risk_bps"].median()),
                "mean_risk_bps": float(group["risk_bps"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _load_grid(
    name: str,
    symbol: str,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> pd.DataFrame:
    path = DATA / f"{symbol.lower()}_{name}_2021_2026.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    import pyarrow.parquet as pq

    schema = pq.read_schema(path).names
    index_field = "timestamp" if "timestamp" in schema else "__index_level_0__"
    filters: list[tuple[str, str, pd.Timestamp]] = [(index_field, "<", end)]
    if start is not None:
        filters.insert(0, (index_field, ">=", start))
    frame = pd.read_parquet(path, filters=filters).sort_index()
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path.name}: index must be unique and increasing")
    return frame


def _fit_hourly_context(
    hourly: pd.DataFrame, config: SupervisorChannelConfig
) -> pd.DataFrame:
    regimes: dict[str, pd.Series] = {}
    primary: pd.DataFrame | None = None
    slope_threshold = config.min_slope_bps_per_hour / 10_000.0
    for window in dict.fromkeys(config.channel_windows):
        fitted = compute_linear_regression_channels(
            hourly,
            window=window,
            log_price=True,
            method="quantile",
            quantile=config.band_quantile,
            require_complete_bars=True,
        )
        regimes[str(window)] = label_channel_regime(
            fitted,
            min_slope=slope_threshold,
            min_r2=config.min_r2,
            persist_bars=config.persist_hours,
        )
        if window == config.channel_window:
            primary = fitted
    if primary is None:
        raise ValueError("primary channel_window must be present in channel_windows")
    regime_frame = pd.DataFrame(regimes, index=hourly.index)
    agreement = channel_confluence(
        regime_frame,
        primary=str(config.channel_window),
        min_agree=config.diagnostic_min_agree,
    )
    primary = primary.copy()
    primary["channel_regime"] = regime_frame[str(config.channel_window)]
    primary["channel_episode_id"] = channel_episode_id(primary["channel_regime"])
    primary["channel_confluence"] = agreement["channel_confluence"]
    primary["channel_confluence_count"] = agreement["channel_confluence_count"]
    primary["rsi_channel"] = primary["rsi"]
    return primary


def _prepare_signals(
    hourly: pd.DataFrame,
    bars_5m: pd.DataFrame,
    config: SupervisorChannelConfig,
) -> pd.DataFrame:
    context = _fit_hourly_context(hourly, config)
    frame = project_closed_hourly_channel(
        context,
        bars_5m,
        channel_bar=config.channel_grid,
        ltf_bar=config.decision_grid,
    )
    frame["rsi_5m"] = compute_rsi(frame["close"])
    if "taker_buy_base" in frame and "volume" in frame:
        volume = frame["volume"].replace(0.0, np.nan)
        frame["taker_imbalance"] = 2.0 * frame["taker_buy_base"] / volume - 1.0
    else:
        frame["taker_imbalance"] = np.nan
    signal_config = replace(config.signal, stop_buffer_bps=config.stop_buffer_bps)
    return generate_supervisor_signals(frame, signal_config)


def _attach_trade_metadata(
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    metadata = signals.loc[
        signals["signal"].ne(0),
        ["setup_type", "t1_time", "t2_time", "channel_confluence_count"],
    ].copy()
    metadata.index.name = "signal_time"
    metadata = metadata.reset_index()
    metadata["side"] = np.where(
        signals.loc[signals["signal"].ne(0), "signal"].to_numpy() > 0,
        "long",
        "short",
    )
    link = orders.loc[
        orders["status"].eq("filled"), ["signal_time", "entry_time", "side"]
    ].drop_duplicates(["entry_time", "side"])
    link = link.merge(metadata, on=["signal_time", "side"], how="left", validate="one_to_one")
    return trades.merge(link, on=["entry_time", "side"], how="left", validate="one_to_one")


def _evaluate_rr_variants(
    signals: pd.DataFrame,
    minute: pd.DataFrame,
    config: SupervisorChannelConfig,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[float, dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    details: dict[float, dict[str, Any]] = {}
    num_signals = int(signals["signal"].ne(0).sum())
    for rr in config.rr_multiples:
        result = backtest_channel_strategy(
            signals,
            target_mode="rr",
            stop_mode="swing",
            swing_low_col="signal_swing_low",
            swing_high_col="signal_swing_high",
            stop_buffer_bps=config.stop_buffer_bps,
            rr_multiple=rr,
            min_risk_bps=0.0,
            max_risk_bps=np.inf,
            min_rr=0.0,
            max_hold_minutes=config.max_hold_minutes,
            cost_bps=config.cost_bps,
            entry_mode="market",
            max_trades_per_day=None,
            max_concurrent=None,
            execution_1m=minute,
            episode_col="channel_episode_id",
        )
        trades = _attach_trade_metadata(result["trades_df"], result["orders_df"], signals)
        result["trades_df"] = trades
        daily_frequency, frequency = summarise_trade_frequency(
            trades,
            start=start,
            end=end,
            target_low=config.frequency_target_low,
            target_high=config.frequency_target_high,
        )
        daily_pnl, pnl = summarise_account_pnl(
            trades, start=start, end=end, risk_pct=config.risk_pct
        )
        outcomes = trades["outcome"].value_counts(normalize=True) if not trades.empty else pd.Series(dtype=float)
        skipped = result["skipped"]
        summary_rows.append(
            {
                "rr_multiple": float(rr),
                "signals": num_signals,
                "filled_trades": int(result["num_trades"]),
                "trades_per_day": frequency["trades_per_day"],
                "inside_target_3_to_5": frequency["inside_target_3_to_5"],
                "active_days": frequency["active_days"],
                "zero_trade_days": frequency["zero_trade_days"],
                "win_rate": result["win_rate"],
                "mean_net_r": result["mean_r_net"],
                "total_net_r": float(trades["r_net"].sum()) if not trades.empty else 0.0,
                "profit_factor": result["profit_factor"],
                "tp_rate": float(outcomes.get("tp", 0.0)),
                "sl_rate": float(outcomes.get("sl", 0.0)),
                "timeout_rate": float(outcomes.get("timeout", 0.0)),
                "mean_daily_pnl_pct": pnl["mean_daily_pnl_pct"],
                "days_ge_2pct": pnl["days_ge_2pct"],
                "days_ge_3pct": pnl["days_ge_3pct"],
                "days_ge_5pct": pnl["days_ge_5pct"],
                "share_days_ge_2pct": pnl["share_days_ge_2pct"],
                "share_days_ge_3pct": pnl["share_days_ge_3pct"],
                "share_days_ge_5pct": pnl["share_days_ge_5pct"],
                "skipped_daily_cap": int(skipped["daily_cap"]),
                "skipped_capacity": int(skipped["capacity"]),
                "skipped_geometry": int(skipped["geometry"]),
                "censored": int(skipped["censored"]),
            }
        )
        details[float(rr)] = {
            "result": result,
            "daily_frequency": daily_frequency,
            "frequency": frequency,
            "daily_pnl": daily_pnl,
            "pnl": pnl,
            "breakdown": summarise_trade_breakdown(trades, rr_multiple=rr),
        }
    return pd.DataFrame(summary_rows), details


def _example_context(signals: pd.DataFrame, bars_each_side: int = 12) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for side, sign in (("long", 1), ("short", -1)):
        hits = np.flatnonzero(signals["signal"].to_numpy() == sign)
        if not len(hits):
            continue
        position = int(hits[0])
        lo = max(0, position - bars_each_side)
        hi = min(len(signals), position + bars_each_side + 1)
        piece = signals.iloc[lo:hi].copy()
        piece["example_side"] = side
        piece["example_signal_time"] = signals.index[position]
        pieces.append(piece)
    return pd.concat(pieces) if pieces else signals.iloc[0:0].copy()


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def run_supervisor_channel_study(
    *,
    stage: str = "dev",
    config: SupervisorChannelConfig = SupervisorChannelConfig(),
    output_dir: Path | None = None,
) -> dict[str, Any]:
    protocol = protocol_dict(config, stage=stage)
    start, end = DEV_START, DEV_END
    output = output_dir or DEFAULT_OUTPUT
    output.mkdir(parents=True, exist_ok=True)

    print("[J] loading dev-only 1h / 5m / 1m bars", flush=True)
    hourly = _load_grid(config.channel_grid, config.symbol, start=None, end=end)
    bars_5m = _load_grid(config.decision_grid, config.symbol, start=start, end=end)
    minute = _load_grid("1m", config.symbol, start=start, end=end)
    if max(hourly.index.max(), bars_5m.index.max(), minute.index.max()) >= end:
        raise AssertionError("development loader crossed the frozen end boundary")

    print("[J] fitting projected 1h channels and strict 5m T1-T2-T3", flush=True)
    signals = _prepare_signals(hourly, bars_5m, config)
    signals = signals[(signals.index >= start) & (signals.index < end)]
    funnel = summarise_signal_funnel(signals)

    print("[J] resolving RR2 / RR3 / RR5 on native 1m bars", flush=True)
    rr_summary, details = _evaluate_rr_variants(
        signals, minute, config, start=start, end=end
    )
    primary = rr_summary.loc[rr_summary["rr_multiple"].eq(2.0)].iloc[0]
    result = {
        "forward_or_lockbox_loaded": False,
        "side_dataset": "pooled",
        "input_max_timestamp": max(hourly.index.max(), bars_5m.index.max(), minute.index.max()),
        "signal_rows": int(signals["signal"].ne(0).sum()),
        "primary_rr": 2.0,
        "primary_trades": int(primary["filled_trades"]),
        "primary_trades_per_day": float(primary["trades_per_day"]),
        "frequency_inside_target_3_to_5": bool(primary["inside_target_3_to_5"]),
        "primary_mean_net_r": float(primary["mean_net_r"]),
        "primary_mean_daily_pnl_pct": float(primary["mean_daily_pnl_pct"]),
        "primary_days_ge_2pct": int(primary["days_ge_2pct"]),
        "daily_stop_applied": False,
        "trade_cap_applied": False,
        "profit_shutdown_applied": False,
        "plain_language_conclusion": (
            "Frequency is observed, not capped. The strategy naturally falls "
            + ("inside" if bool(primary["inside_target_3_to_5"]) else "outside")
            + " the 3-5 trades/day target; economics are "
            + ("positive" if float(primary["mean_net_r"]) > 0 else "not positive")
            + " after the frozen 10 bps cost."
        ),
    }

    _write_json(output / "protocol.json", protocol)
    _write_json(output / "result.json", result)
    funnel.to_csv(output / "funnel.csv", index=False)
    rr_summary.to_csv(output / "rr_summary.csv", index=False)
    trade_breakdown = pd.concat(
        [detail["breakdown"] for detail in details.values()], ignore_index=True
    )
    trade_breakdown.to_csv(output / "trade_breakdown.csv", index=False)
    primary_details = details[2.0]
    primary_details["daily_frequency"].to_csv(output / "daily_frequency_rr2.csv")
    primary_details["daily_pnl"].to_csv(output / "daily_pnl_rr2.csv")

    stage_mask = signals[["stage_1", "stage_2", "stage_3"]].any(axis=1)
    signal_columns = [
        "availability_time",
        "open",
        "high",
        "low",
        "close",
        "channel_lower",
        "channel_mid",
        "channel_upper",
        "channel_slope",
        "channel_r2",
        "channel_regime",
        "channel_episode_id",
        "channel_confluence",
        "channel_confluence_count",
        "channel_pos",
        "rsi_channel",
        "rsi_5m",
        "taker_imbalance",
        "stage_1",
        "stage_2",
        "stage_3",
        "signal",
        "setup_type",
        "t1_time",
        "t2_time",
        "signal_swing_low",
        "signal_swing_high",
    ]
    signals.loc[stage_mask, signal_columns].to_parquet(output / "signals.parquet")
    _example_context(signals)[signal_columns + ["example_side", "example_signal_time"]].to_parquet(
        output / "examples.parquet"
    )
    for rr, detail in details.items():
        suffix = str(int(rr)) if float(rr).is_integer() else str(rr).replace(".", "p")
        detail["result"]["trades_df"].to_parquet(output / f"trades_rr{suffix}.parquet")

    print(rr_summary.to_string(index=False), flush=True)
    print(f"[J] artifacts -> {output}", flush=True)
    return {
        "protocol": protocol,
        "result": result,
        "funnel": funnel,
        "rr_summary": rr_summary,
        "trade_breakdown": trade_breakdown,
        "details": details,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_supervisor_channel_study(stage=args.stage, output_dir=args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SupervisorChannelConfig",
    "protocol_dict",
    "run_supervisor_channel_study",
    "summarise_account_pnl",
    "summarise_signal_funnel",
    "summarise_trade_breakdown",
    "summarise_trade_frequency",
]
