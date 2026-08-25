"""Frozen BTC channel-policy replication for USA500 and USATECH.

The wrapper deliberately makes no selection on index outcomes.  It transfers
the completed-H1, window-60 T1-T2-T3 construction and reports RR2 as primary;
RR3/RR5 are sensitivity rows over the same entries.  Signals use M5 decisions,
channels use only completed H1 bars, and every exit is replayed on native M1.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from experiments.supervisor_channel_strategy import (
    SupervisorChannelConfig,
    _evaluate_rr_variants,
    _fit_hourly_context,
    summarise_signal_funnel,
)
from features.linear_channels import compute_rsi
from features.supervisor_channel import (
    generate_supervisor_signals,
    project_closed_hourly_channel,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CODE_ROOT / "data"
CACHE_ROOT = CODE_ROOT / "experiments" / "cache" / "index_channel_replication"
CUTOFF = pd.Timestamp("2026-04-01", tz="UTC")
M5 = pd.Timedelta(minutes=5)
STAGES = {
    "development": (
        pd.Timestamp("2021-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
    ),
    "h1_2025": (
        pd.Timestamp("2025-01-01", tz="UTC"),
        pd.Timestamp("2025-07-01", tz="UTC"),
    ),
    "forward": (
        pd.Timestamp("2025-07-01", tz="UTC"),
        CUTOFF,
    ),
}
STREAMS = {
    "usa500": ("USA500IDXUSD", 2.0),
    "usatech": ("USATECHIDXUSD", 3.0),
}
PROTOCOL_VERSION = "index-channel-transfer-v1"


@dataclass(frozen=True)
class IndexChannelConfig:
    stream: str
    symbol: str
    cost_bps: float
    data_dir: Path = DATA_DIR
    output_root: Path = CACHE_ROOT
    channel_window: int = 60
    channel_windows: tuple[int, ...] = (60, 90, 120)
    rr_multiples: tuple[float, ...] = (2.0, 3.0, 5.0)
    max_hold_minutes: int = 120

    @classmethod
    def for_stream(
        cls,
        stream: str,
        *,
        data_dir: str | Path = DATA_DIR,
        output_base: str | Path = CACHE_ROOT,
    ) -> "IndexChannelConfig":
        if stream not in STREAMS:
            raise ValueError(f"unknown index stream: {stream}")
        symbol, cost = STREAMS[stream]
        return cls(
            stream=stream,
            symbol=symbol,
            cost_bps=cost,
            data_dir=Path(data_dir),
            output_root=Path(output_base) / stream,
        )


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    return value


def _atomic_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_parquet(temporary, index=index)
    os.replace(temporary, path)


def label_signal_funnel(
    funnel: pd.DataFrame, *, stream: str, evaluation_stage: str
) -> pd.DataFrame:
    """Preserve T1/T2/T3/ENTRY while adding the evaluation-period label."""
    if "stage" not in funnel.columns:
        raise ValueError("signal funnel misses its stage column")
    return funnel.rename(columns={"stage": "funnel_stage"}).assign(
        stream=stream, evaluation_stage=evaluation_stage
    )


def channel_protocol(config: IndexChannelConfig) -> dict[str, Any]:
    """Return the immutable transfer contract before index outcomes are read."""
    if config.channel_window != 60 or tuple(config.channel_windows) != (60, 90, 120):
        raise ValueError("index channel transfer must keep BTC windows 60/90/120")
    if tuple(float(value) for value in config.rr_multiples) != (2.0, 3.0, 5.0):
        raise ValueError("index channel transfer must keep RR2/RR3/RR5")
    if int(config.max_hold_minutes) != 120:
        raise ValueError("index channel transfer must keep the 120-minute hold")
    return {
        "protocol_version": PROTOCOL_VERSION,
        "stream": config.stream,
        "symbol": config.symbol,
        "cost_bps_round_trip": float(config.cost_bps),
        "primary": {"window": 60, "rr": 2.0},
        "sensitivities": [
            {"window": 60, "rr": 3.0},
            {"window": 60, "rr": 5.0},
        ],
        "diagnostic_channel_windows": [60, 90, 120],
        "channel_window_unit": "completed_market_hours",
        "stale_context_cutoff_minutes": 60,
        "decision_grid": "5min",
        "channel_grid": "1h",
        "execution_grid": "native_1min",
        "max_hold_minutes": 120,
        "selection": "none_frozen_btc_transfer",
        "later_stage_can_tune": False,
        "vix_used": False,
        "sentiment_used": False,
        "stages": STAGES,
        "q2_2026_loaded": False,
        "source_implementation_sha256": hashlib.sha256(
            (
                inspect.getsource(_fit_hourly_context)
                + inspect.getsource(project_closed_hourly_channel)
                + inspect.getsource(generate_supervisor_signals)
            ).encode("utf-8")
        ).hexdigest(),
    }


def mask_stale_channel_context(
    frame: pd.DataFrame, *, cutoff_minutes: int = 60
) -> pd.DataFrame:
    """Disable an H1 fit after a market closure until a fresh hour closes."""
    if int(cutoff_minutes) < 1:
        raise ValueError("stale-context cutoff must be positive")
    required = {"availability_time", "channel_availability_time", "channel_regime"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"channel projection misses columns: {sorted(missing)}")
    out = frame.copy()
    decision = pd.to_datetime(out["availability_time"], utc=True)
    available = pd.to_datetime(out["channel_availability_time"], utc=True)
    stale = available.isna() | decision.sub(available).ge(
        pd.Timedelta(minutes=int(cutoff_minutes))
    )
    geometry = [
        name
        for name in (
            "channel_mid",
            "channel_upper",
            "channel_lower",
            "channel_slope",
            "channel_r2",
            "channel_width",
            "channel_pos",
            "rsi_channel",
        )
        if name in out.columns
    ]
    out.loc[stale, geometry] = np.nan
    out.loc[stale, "channel_regime"] = "none"
    if "channel_confluence" in out:
        out.loc[stale, "channel_confluence"] = 0
    if "channel_confluence_count" in out:
        out.loc[stale, "channel_confluence_count"] = 0
    out["channel_context_stale"] = stale.astype(bool)
    return out


def _prepare_index_signals(
    hourly: pd.DataFrame,
    bars_5m: pd.DataFrame,
    config: SupervisorChannelConfig,
) -> pd.DataFrame:
    """Adapt the BTC channel clock to completed bars of a session market."""
    if "complete_bar" in hourly and not hourly["complete_bar"].astype(bool).all():
        raise ValueError("index H1 input must contain completed bars only")
    # A scheduled market closure is not a truncated bar.  The loader has already
    # removed genuinely incomplete H1 rows, so the 60-row regression counts the
    # last 60 completed market hours instead of requiring 60 wall-clock hours.
    fitting = hourly.drop(columns=["minute_count"], errors="ignore")
    context = _fit_hourly_context(fitting, config)
    frame = project_closed_hourly_channel(
        context,
        bars_5m,
        channel_bar=config.channel_grid,
        ltf_bar=config.decision_grid,
    )
    frame = mask_stale_channel_context(frame, cutoff_minutes=60)
    frame["rsi_5m"] = compute_rsi(frame["close"])
    if "taker_buy_base" in frame and "volume" in frame:
        volume = frame["volume"].replace(0.0, np.nan)
        frame["taker_imbalance"] = 2.0 * frame["taker_buy_base"] / volume - 1.0
    else:
        frame["taker_imbalance"] = np.nan
    signal_config = replace(config.signal, stop_buffer_bps=config.stop_buffer_bps)
    return generate_supervisor_signals(frame, signal_config)


def validate_closed_channel_context(signals: pd.DataFrame) -> dict[str, Any]:
    """Prove that every fitted H1 context was closed before its M5 decision."""
    required = {
        "signal",
        "availability_time",
        "channel_source_time",
        "channel_availability_time",
    }
    missing = required.difference(signals.columns)
    if missing:
        raise ValueError(f"signals miss context audit columns: {sorted(missing)}")
    decision = pd.to_datetime(signals["availability_time"], utc=True)
    source = pd.to_datetime(signals["channel_source_time"], utc=True)
    available = pd.to_datetime(signals["channel_availability_time"], utc=True)
    valid = source.notna() & available.notna() & decision.notna()
    future = valid & available.gt(decision)
    source_close_mismatch = valid & source.add(pd.Timedelta(hours=1)).ne(available)
    signal_missing = signals["signal"].ne(0) & ~valid
    stale = (
        signals["channel_context_stale"].astype(bool)
        if "channel_context_stale" in signals
        else pd.Series(False, index=signals.index)
    )
    signal_on_stale = signals["signal"].ne(0) & stale
    lags = (decision.loc[valid] - available.loc[valid]) / pd.Timedelta(minutes=1)
    return {
        "passed": bool(
            not future.any()
            and not source_close_mismatch.any()
            and not signal_missing.any()
            and not signal_on_stale.any()
        ),
        "rows": int(len(signals)),
        "valid_context_rows": int(valid.sum()),
        "future_context_rows": int(future.sum()),
        "source_close_mismatch_rows": int(source_close_mismatch.sum()),
        "signal_rows_with_missing_context": int(signal_missing.sum()),
        "stale_context_rows": int(stale.sum()),
        "signal_rows_on_stale_context": int(signal_on_stale.sum()),
        "min_source_availability_lag_minutes": float(lags.min()) if len(lags) else None,
        "max_source_availability_lag_minutes": float(lags.max()) if len(lags) else None,
    }


def stage_signal_frame(
    signals: pd.DataFrame,
    *,
    bars_index: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    hold_minutes: int = 120,
) -> pd.DataFrame:
    """Keep stage-isolated signals with a real next M5 bar and full exit window."""
    work = signals.copy()
    decision = pd.to_datetime(work["availability_time"], utc=True)
    ordered = pd.DatetimeIndex(pd.to_datetime(bars_index, utc=True)).sort_values()
    next_by_bar = pd.Series(ordered[1:], index=ordered[:-1])
    following = pd.Series(work.index, index=work.index).map(next_by_bar)
    consecutive = following.sub(pd.Series(work.index, index=work.index)).eq(M5)
    stage_safe = (
        decision.ge(pd.Timestamp(start))
        & decision.add(pd.Timedelta(minutes=int(hold_minutes))).le(pd.Timestamp(end))
    )
    work.loc[~(consecutive & stage_safe), "signal"] = 0
    return work


def _read_grid(config: IndexChannelConfig, resolution: str) -> pd.DataFrame:
    path = config.data_dir / f"{config.stream}_{resolution}_2021_2026.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    import pyarrow.parquet as pq

    names = pq.read_schema(path).names
    index_field = "timestamp" if "timestamp" in names else "__index_level_0__"
    frame = pd.read_parquet(path, filters=[(index_field, "<", CUTOFF)]).sort_index()
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError(f"{path.name} must have a timezone-aware DatetimeIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{path.name} index must be unique and increasing")
    if frame.empty or frame.index.max() >= CUTOFF:
        raise AssertionError(f"{path.name} crossed or missed the Q2 boundary")
    if "complete_bar" in frame:
        frame = frame.loc[frame["complete_bar"].astype(bool)]
    return frame


def _supervisor_config(config: IndexChannelConfig) -> SupervisorChannelConfig:
    return SupervisorChannelConfig(
        symbol=config.symbol,
        channel_window=config.channel_window,
        channel_windows=config.channel_windows,
        rr_multiples=config.rr_multiples,
        max_hold_minutes=config.max_hold_minutes,
        cost_bps=config.cost_bps,
    )


def run_index_channel_replication(config: IndexChannelConfig) -> dict[str, Any]:
    """Run all fixed stages and persist separate channel evidence per index."""
    protocol = channel_protocol(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(protocol, config.output_root / "protocol.json")
    hourly = _read_grid(config, "1h")
    five = _read_grid(config, "5min")
    minute = _read_grid(config, "1m")
    supervisor = _supervisor_config(config)
    print(f"[{config.stream}] building frozen closed-H1 channel signals", flush=True)
    signals = _prepare_index_signals(hourly, five, supervisor)
    if signals.index.max() >= CUTOFF or signals["availability_time"].max() > CUTOFF:
        raise AssertionError("channel signals crossed Q2-2026")
    context_audit = validate_closed_channel_context(signals)
    if not context_audit["passed"]:
        raise AssertionError(f"closed-H1 channel audit failed: {context_audit}")
    _atomic_json(context_audit, config.output_root / "closed_context_audit.json")

    rr_tables: list[pd.DataFrame] = []
    funnels: list[pd.DataFrame] = []
    breakdowns: list[pd.DataFrame] = []
    primary_ledgers: list[pd.DataFrame] = []
    signal_events: list[pd.DataFrame] = []
    for stage_name, (start, end) in STAGES.items():
        scoped = stage_signal_frame(
            signals,
            bars_index=five.index,
            start=start,
            end=end,
            hold_minutes=config.max_hold_minutes,
        )
        decision = pd.to_datetime(scoped["availability_time"], utc=True)
        in_stage = decision.ge(start) & decision.lt(end)
        funnel = label_signal_funnel(
            summarise_signal_funnel(scoped.loc[in_stage]),
            stream=config.stream,
            evaluation_stage=stage_name,
        )
        funnels.append(funnel)
        print(f"[{config.stream}] replaying {stage_name} RR2/RR3/RR5", flush=True)
        rr_summary, details = _evaluate_rr_variants(
            scoped,
            minute,
            supervisor,
            start=start,
            end=end,
        )
        rr_summary = rr_summary.assign(
            stream=config.stream,
            stage=stage_name,
            primary=rr_summary["rr_multiple"].eq(2.0),
            selected_on_stage=False,
        )
        rr_tables.append(rr_summary)
        for rr, detail in details.items():
            ledger = detail["result"]["trades_df"].copy()
            orders = detail["result"]["orders_df"].copy()
            if not ledger.empty:
                ledger.insert(0, "stage", stage_name)
                ledger.insert(0, "stream", config.stream)
            if not orders.empty:
                orders.insert(0, "stage", stage_name)
                orders.insert(0, "stream", config.stream)
            _atomic_parquet(
                ledger,
                config.output_root / "ledgers" / f"{stage_name}_rr{int(rr)}.parquet",
            )
            _atomic_parquet(
                orders,
                config.output_root / "orders" / f"{stage_name}_rr{int(rr)}.parquet",
            )
            breakdown = detail["breakdown"].assign(
                stream=config.stream, stage=stage_name
            )
            breakdowns.append(breakdown)
            if float(rr) == 2.0:
                primary_ledgers.append(ledger)
                _atomic_parquet(
                    detail["daily_frequency"].rename_axis("date").reset_index(),
                    config.output_root / "daily" / f"{stage_name}_frequency_rr2.parquet",
                )
                _atomic_parquet(
                    detail["daily_pnl"].rename_axis("date").reset_index(),
                    config.output_root / "daily" / f"{stage_name}_pnl_rr2.parquet",
                )
        event_mask = scoped[["stage_1", "stage_2", "stage_3"]].any(axis=1) | scoped[
            "signal"
        ].ne(0)
        events = scoped.loc[event_mask & in_stage].copy()
        events.insert(0, "stage", stage_name)
        events.insert(0, "stream", config.stream)
        signal_events.append(events)

    rr_table = pd.concat(rr_tables, ignore_index=True)
    funnel_table = pd.concat(funnels, ignore_index=True)
    breakdown_table = pd.concat(breakdowns, ignore_index=True)
    events_table = pd.concat(signal_events).sort_index() if signal_events else signals.iloc[0:0]
    ledger_table = (
        pd.concat(primary_ledgers, ignore_index=True)
        if primary_ledgers
        else pd.DataFrame()
    )
    _atomic_parquet(rr_table, config.output_root / "stage_rr_summary.parquet")
    _atomic_parquet(funnel_table, config.output_root / "stage_funnel.parquet")
    _atomic_parquet(breakdown_table, config.output_root / "stage_trade_breakdown.parquet")
    _atomic_parquet(events_table, config.output_root / "signal_events.parquet", index=True)
    _atomic_parquet(ledger_table, config.output_root / "primary_rr2_ledger.parquet")
    primary = rr_table.loc[rr_table["primary"]].copy()
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "stream": config.stream,
        "primary": primary.to_dict("records"),
        "stage_count": len(STAGES),
        "rr_rows": int(len(rr_table)),
        "signal_event_rows": int(len(events_table)),
        "primary_trade_rows": int(len(ledger_table)),
        "closed_context_audit": context_audit,
        "max_input_timestamp": max(hourly.index.max(), five.index.max(), minute.index.max()),
        "max_signal_availability": signals["availability_time"].max(),
        "later_stage_used_for_selection": False,
        "vix_used": False,
        "sentiment_used": False,
        "q2_2026_loaded": False,
    }
    _atomic_json(result, config.output_root / "result.json")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=tuple(STREAMS), required=True)
    args = parser.parse_args(argv)
    result = run_index_channel_replication(IndexChannelConfig.for_stream(args.stream))
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
