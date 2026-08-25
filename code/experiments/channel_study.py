"""Deterministic runner for the linear-regression channel study.

One entry point, one config, no randomness: the same arguments always produce the
same event dataset and the same summary. Everything the study claims has to come
out of here rather than out of a one-off script, so that a number in the write-up
can be traced to a command.

Split discipline is enforced rather than documented. `stage` bounds every read at
its exclusive end; earlier rows are available only for causal feature warm-up:

    dev       2021-01-01 .. 2025-07-01   rules, geometry and thresholds
    tune      2025-07-01 .. 2025-10-01   threshold calibration only
    forward   2025-10-01 .. 2026-04-01   evaluated once, after freezing
    lockbox   2026-04-01 .. 2026-07-01   sealed; refuses to load without --i-am-unsealing

The macro regime needs 200 daily closes before its first value, so daily bars from
2020 are read for warm-up only; no 2020 bar reaches the trading logic.

Run:  python -m experiments.channel_study --stage dev
      python -m experiments.channel_study --stage dev --grid 30min --no-macro
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import numpy as np
import pandas as pd

from features.linear_channels import (
    channel_confluence,
    channel_episode_id,
    compute_linear_regression_channels,
    compute_rsi,
    gate_channel_signals,
    label_channel_regime,
)
from features.channel_faucet import faucet_funnel, generate_channel_faucet_signals
from evaluation.channel_backtest import backtest_channel_strategy
from experiments.channel_event_dataset import (
    build_channel_event_dataset,
    causal_rolling_percentile,
    prepare_positioning_features,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
DATA = CODE_ROOT / "data"
OUT_ROOT = CODE_ROOT / "experiments" / "cache" / "channel_study"
LOCKBOX_MARKER = OUT_ROOT / "LOCKBOX_UNSEALED.json"

STAGES = {
    "dev": ("2021-01-01", "2025-07-01"),
    "tune": ("2025-07-01", "2025-10-01"),
    "forward": ("2025-10-01", "2026-04-01"),
    "lockbox": ("2026-04-01", "2026-07-01"),
}
MACRO_WARMUP_DAYS = 200


def authorise_stage(
    stage: str,
    *,
    unsealing: bool,
    marker: Path = LOCKBOX_MARKER,
    consume: bool = False,
) -> None:
    """Guard the lockbox with an explicit, persistent one-run manifest."""
    if stage != "lockbox":
        return
    if marker.exists():
        raise PermissionError(f"lockbox already unsealed; manifest exists at {marker}")
    if not unsealing:
        raise PermissionError(
            "the lockbox is sealed; unsealing is allowed only for the single final run"
        )
    if consume:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps({
            "stage": stage,
            "unsealed_at_utc": str(pd.Timestamp.now(tz="UTC")),
            "status": "started",
        }, indent=2), encoding="utf-8")


@dataclass(frozen=True)
class ChannelConfig:
    """Every knob the study exposes. Written next to the results so a run is
    reproducible from its own output."""
    symbol: str = "BTCUSDT"
    grid: str = "5min"
    channel_grid: str = "1h"
    channel_window: int = 60
    channel_windows: tuple[int, ...] = (60, 90, 120)
    min_channel_agreement: int = 2
    # Frozen mechanical control. The broader ML-window threshold is configured in
    # Notebook B and must not silently change this historical comparator.
    min_r2: float = 0.400
    min_slope_bps_per_hour: float = 5.0
    band_method: str = "quantile"
    band_quantile: float = 0.10
    band_num_std: float = 2.0
    persist_bars: int = 6
    macro_mode: str = "hard"          # hard | hysteresis | off
    macro_sma_days: int = 200
    macro_hysteresis_pct: float = 2.0
    long_pos_threshold: float = 0.30
    short_pos_threshold: float = 0.70
    rsi_oversold: float = 100.0
    rsi_overbought: float = 0.0
    rsi_percentile_lookback: int = 720
    arm_max_bars: int = 3
    confirm_max_bars: int = 2
    require_confirmation: bool = False
    require_flow: bool = False
    swing_lookback: int = 12
    target_mode: str = "measured"     # measured | rail | rr | pct
    rr_multiple: float = 1.5
    min_risk_bps: float = 40.0
    max_risk_bps: float = 250.0
    min_rr: float = 1.5
    max_hold_bars: int = 288
    cost_bps: float = 10.0
    max_trades_per_day: int | None = None
    max_concurrent: int | None = None
    entry_mode: str = "maker_limit"   # market | maker_limit
    limit_offset_bps: float = 5.0
    fill_window_bars: int = 4
    maker_fee_bps: float | None = 2.0
    taker_fee_bps: float | None = 5.0


def _load_grid(
    name: str,
    symbol: str = "BTCUSDT",
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    path = DATA / f"{symbol.lower()}_{name}_2021_2026.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} missing — run data/build_grids.py")
    filters = []
    if start is not None or end is not None:
        import pyarrow.parquet as pq

        schema_names = pq.read_schema(path).names
        index_field = ("timestamp" if "timestamp" in schema_names
                       else "__index_level_0__")
        if start is not None:
            filters.append((index_field, ">=", start))
        if end is not None:
            filters.append((index_field, "<", end))
    frame = pd.read_parquet(path, filters=filters or None)
    if start is not None:
        frame = frame[frame.index >= start]
    if end is not None:
        frame = frame[frame.index < end]
    return frame


def run_tag(cfg: ChannelConfig, stage: str) -> str:
    """Name an artefact by its full behavioural config, not a lossy subset."""
    payload = json.dumps(asdict(cfg), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
    return (f"{cfg.symbol}_{stage}_{cfg.channel_grid}w{cfg.channel_window}_"
            f"{cfg.grid}_macro-{cfg.macro_mode}_{digest}")


def _daily_closes(cfg: "ChannelConfig", end: pd.Timestamp) -> pd.Series:
    """Daily closes including 2020, used only to warm the macro average."""
    files = sorted((DATA / "raw" / "binance").glob(f"{cfg.symbol}-1d-*.csv"))
    frames = []
    for f in files:
        d = pd.read_csv(f, header=None, usecols=[0, 4], names=["t", "close"])
        d = d[pd.to_numeric(d["t"], errors="coerce").notna()]
        frames.append(d)
    daily_2020 = pd.Series(dtype=float)
    if frames:
        d = pd.concat(frames, ignore_index=True)
        t = pd.to_numeric(d["t"])
        unit = "us" if t.max() > 1e14 else "ms"
        daily_2020 = pd.Series(pd.to_numeric(d["close"]).to_numpy(),
                               index=pd.to_datetime(t, unit=unit, utc=True)).sort_index()
        daily_2020 = daily_2020[daily_2020.index < end]
    later = (_load_grid("1h", cfg.symbol, end=end)["close"]
             .resample("1D").last().dropna())
    return pd.concat([daily_2020[~daily_2020.index.isin(later.index)], later]).sort_index()


def macro_regime(cfg: ChannelConfig, end: pd.Timestamp) -> pd.Series:
    """Daily close against its own long average, readable only the following day."""
    daily = _daily_closes(cfg, end)
    sma = daily.rolling(cfg.macro_sma_days, min_periods=cfg.macro_sma_days).mean()
    if cfg.macro_mode == "off":
        state = pd.Series("any", index=daily.index, dtype=object)
    elif cfg.macro_mode == "hard":
        state = pd.Series(np.where(daily > sma, "bull", "bear"), index=daily.index,
                          dtype=object)
    else:
        band = cfg.macro_hysteresis_pct / 100.0
        raw = np.where(daily > sma * (1 + band), "bull",
                       np.where(daily < sma * (1 - band), "bear", None))
        state = pd.Series(raw, index=daily.index, dtype=object).ffill()
    state[sma.isna()] = "none"
    state.index = state.index + pd.Timedelta(days=1)     # known only once the day closed
    return state.rename("macro_regime")


def build_events(cfg: ChannelConfig, stage: str, *, unsealing: bool = False) -> dict:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; pick one of {sorted(STAGES)}")
    authorise_stage(stage, unsealing=unsealing, consume=(stage == "lockbox"))
    if stage == "lockbox" and not unsealing:
        raise PermissionError(
            "the lockbox is sealed — pass unsealing=True only for the single final run"
        )
    start, end = (pd.Timestamp(t, tz="UTC") for t in STAGES[stage])

    # Every feature load is bounded above by the selected stage.  Earlier history
    # remains available for causal warm-up, but a dev run cannot even read Q2 rows.
    hi = _load_grid(cfg.channel_grid, cfg.symbol, end=end)
    lo = _load_grid(cfg.grid, cfg.symbol, end=end)
    minute = _load_grid("1m", cfg.symbol, start=start, end=end)
    hours_per_bar = pd.Timedelta(cfg.channel_grid).total_seconds() / 3600.0
    decision_delta = pd.Timedelta(cfg.grid)

    windows = tuple(dict.fromkeys(cfg.channel_windows))
    if cfg.channel_window not in windows:
        raise ValueError("primary channel_window must be in channel_windows")
    if not 1 <= cfg.min_channel_agreement <= len(windows):
        raise ValueError("min_channel_agreement must fit channel_windows")

    channels: dict[int, pd.DataFrame] = {}
    for window in windows:
        fitted = compute_linear_regression_channels(
            hi, window=window, num_std=cfg.band_num_std,
            log_price=True, method=cfg.band_method, quantile=cfg.band_quantile,
        )
        # slope is per bar in log units; convert so the threshold reads in bps
        # per hour and stays comparable when the channel grid changes.
        fitted["channel_slope"] = fitted["channel_slope"] / hours_per_bar * 1e4
        fitted["channel_regime"] = label_channel_regime(
            fitted, min_slope=cfg.min_slope_bps_per_hour, min_r2=cfg.min_r2,
            persist_bars=cfg.persist_bars,
        )
        channels[window] = fitted

    ch = channels[cfg.channel_window].copy()
    regimes = pd.DataFrame(
        {str(window): channels[window]["channel_regime"] for window in windows},
        index=hi.index,
    )
    ch = ch.join(channel_confluence(
        regimes, primary=str(cfg.channel_window),
        min_agree=cfg.min_channel_agreement,
    ))
    min_percentile_history = min(72, cfg.rsi_percentile_lookback)
    up_rsi = ch["rsi"].where(ch["channel_regime"] == "up")
    down_rsi = ch["rsi"].where(ch["channel_regime"] == "down")
    ch["rsi_regime_pct"] = causal_rolling_percentile(
        up_rsi, window=cfg.rsi_percentile_lookback,
        min_periods=min_percentile_history,
    ).combine_first(causal_rolling_percentile(
        down_rsi, window=cfg.rsi_percentile_lookback,
        min_periods=min_percentile_history,
    ))
    ch["channel_episode_id"] = channel_episode_id(ch["channel_regime"])
    ch.index = ch.index + pd.Timedelta(hours=hours_per_bar)   # usable once the bar closed

    carry = ["channel_slope", "channel_mid", "channel_upper", "channel_lower",
             "channel_r2", "channel_width", "channel_regime", "channel_episode_id",
             "rsi", "rsi_regime_pct", "channel_confluence_count",
             "channel_confluence"]
    decision_index = lo.index + decision_delta
    aligned_channel = ch[carry].reindex(decision_index, method="ffill")
    aligned_channel.index = lo.index
    frame = lo.join(aligned_channel.rename(columns={"rsi": "rsi_channel"}))
    frame["rsi"] = compute_rsi(frame["close"])
    span = frame["channel_upper"] - frame["channel_lower"]
    frame["channel_pos"] = np.where(span > 0,
                                    (frame["close"] - frame["channel_lower"]) / span, np.nan)

    mac = macro_regime(cfg, end).reindex(decision_index, method="ffill")
    frame["macro_regime"] = mac.to_numpy()
    if cfg.macro_mode != "off":
        block_long = (frame["macro_regime"] != "bull") & (frame["channel_regime"] == "up")
        block_short = (frame["macro_regime"] != "bear") & (frame["channel_regime"] == "down")
        frame.loc[block_long | block_short, "channel_regime"] = "none"
    frame["channel_episode_id"] = channel_episode_id(
        frame["channel_regime"].fillna("none")
    )

    positioning_path = DATA / f"{cfg.symbol.lower()}_positioning_15min_2021_2026.parquet"
    if not positioning_path.exists():
        raise FileNotFoundError(f"{positioning_path.name} missing â€” run data.build_positioning --extended")
    positioning_raw = pd.read_parquet(
        positioning_path, filters=[("timestamp", "<", end)]
    )
    positioning = prepare_positioning_features(positioning_raw)
    aligned_positioning = positioning.reindex(decision_index, method="ffill")
    aligned_positioning.index = lo.index
    frame = frame.join(aligned_positioning)

    frame = frame[(frame.index >= start) & (frame.index < end)]
    if frame.empty:
        raise ValueError(f"no bars in stage {stage}")

    candidates = generate_channel_faucet_signals(
        frame,
        long_pos_threshold=cfg.long_pos_threshold,
        short_pos_threshold=cfg.short_pos_threshold,
        rsi_oversold=cfg.rsi_oversold,
        rsi_overbought=cfg.rsi_overbought,
        arm_max_bars=cfg.arm_max_bars,
        confirm_max_bars=cfg.confirm_max_bars,
        require_confirmation=cfg.require_confirmation,
        require_flow=cfg.require_flow,
        regime_col="channel_regime",
        arm_rsi_col="rsi_channel",
    )
    policy_signals = gate_channel_signals(candidates)
    common_backtest = dict(
        target_mode=cfg.target_mode, stop_mode="swing",
        swing_lookback=cfg.swing_lookback, rr_multiple=cfg.rr_multiple,
        max_hold_bars=cfg.max_hold_bars, cost_bps=cfg.cost_bps,
        entry_mode=cfg.entry_mode, limit_offset_bps=cfg.limit_offset_bps,
        fill_window_bars=cfg.fill_window_bars,
        maker_fee_bps=cfg.maker_fee_bps, taker_fee_bps=cfg.taker_fee_bps,
        regime_col="channel_regime", episode_col="channel_episode_id",
        execution_1m=minute,
    )
    res = backtest_channel_strategy(
        policy_signals,
        min_risk_bps=cfg.min_risk_bps, max_risk_bps=cfg.max_risk_bps,
        min_rr=cfg.min_rr, max_trades_per_day=cfg.max_trades_per_day,
        max_concurrent=cfg.max_concurrent,
        **common_backtest,
    )
    # Labels describe each candidate independently. Portfolio capacity belongs to
    # policy evaluation, not to the training-table sampling mechanism. Confluence
    # remains a varying binary feature here; applying the hard policy gate first
    # would make it a constant and leave ML nothing to learn from.
    label_res = backtest_channel_strategy(
        candidates, min_risk_bps=0.0, max_risk_bps=np.inf, min_rr=0.0,
        max_trades_per_day=None, max_concurrent=None,
        **common_backtest,
    )
    events = build_channel_event_dataset(
        candidates, label_res["orders_df"], swing_lookback=cfg.swing_lookback,
        target_mode=cfg.target_mode, rr_multiple=cfg.rr_multiple,
        min_risk_bps=cfg.min_risk_bps, max_risk_bps=cfg.max_risk_bps,
        min_rr=cfg.min_rr,
    )
    days = max((end - start).days, 1)
    raw_mask = candidates["signal"].ne(0)
    confluent_mask = policy_signals["signal"].ne(0)
    confluence_counts = candidates.loc[raw_mask, "channel_confluence_count"].value_counts()
    res["funnel"] = faucet_funnel(policy_signals)
    res["trades_per_day"] = res["num_trades"] / days
    res["days"] = days
    res["stage"] = stage
    res["num_raw_signals"] = int(raw_mask.sum())
    res["num_confluent_signals"] = int(confluent_mask.sum())
    res["num_confluence_rejected"] = int((raw_mask & ~confluent_mask).sum())
    res["confluence_candidate_counts"] = {
        str(int(level)): int(count) for level, count in confluence_counts.items()
    }
    res["signals"] = policy_signals
    res["events_df"] = events
    res["num_events"] = int(len(events))
    return res


def _summary(res: dict) -> dict:
    return {k: v for k, v in res.items()
            if k not in ("trades_df", "orders_df", "events_df", "signals", "funnel")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="dev", choices=sorted(STAGES))
    ap.add_argument("--symbol", default=ChannelConfig.symbol)
    ap.add_argument("--grid", default=ChannelConfig.grid)
    ap.add_argument("--channel-grid", dest="channel_grid", default=ChannelConfig.channel_grid)
    ap.add_argument("--window", dest="channel_window", type=int,
                    default=ChannelConfig.channel_window)
    ap.add_argument("--channel-windows", dest="channel_windows", type=int, nargs="+",
                    default=ChannelConfig.channel_windows)
    ap.add_argument("--min-channel-agreement", dest="min_channel_agreement", type=int,
                    default=ChannelConfig.min_channel_agreement)
    ap.add_argument("--min-r2", dest="min_r2", type=float, default=ChannelConfig.min_r2)
    ap.add_argument("--min-slope-bps-per-hour", dest="min_slope_bps_per_hour",
                    type=float, default=ChannelConfig.min_slope_bps_per_hour)
    ap.add_argument("--band-method", dest="band_method", default=ChannelConfig.band_method,
                    choices=["quantile", "std"])
    ap.add_argument("--band-quantile", dest="band_quantile", type=float,
                    default=ChannelConfig.band_quantile)
    ap.add_argument("--band-num-std", dest="band_num_std", type=float,
                    default=ChannelConfig.band_num_std)
    ap.add_argument("--persist-bars", dest="persist_bars", type=int,
                    default=ChannelConfig.persist_bars)
    ap.add_argument("--macro", dest="macro_mode", default=ChannelConfig.macro_mode,
                    choices=["hard", "hysteresis", "off"])
    ap.add_argument("--macro-sma-days", dest="macro_sma_days", type=int,
                    default=ChannelConfig.macro_sma_days)
    ap.add_argument("--macro-hysteresis-pct", dest="macro_hysteresis_pct", type=float,
                    default=ChannelConfig.macro_hysteresis_pct)
    ap.add_argument("--long-pos-threshold", dest="long_pos_threshold", type=float,
                    default=ChannelConfig.long_pos_threshold)
    ap.add_argument("--short-pos-threshold", dest="short_pos_threshold", type=float,
                    default=ChannelConfig.short_pos_threshold)
    ap.add_argument("--rsi-oversold", dest="rsi_oversold", type=float,
                    default=ChannelConfig.rsi_oversold)
    ap.add_argument("--rsi-overbought", dest="rsi_overbought", type=float,
                    default=ChannelConfig.rsi_overbought)
    ap.add_argument("--rsi-percentile-lookback", dest="rsi_percentile_lookback",
                    type=int, default=ChannelConfig.rsi_percentile_lookback)
    ap.add_argument("--arm-max-bars", dest="arm_max_bars", type=int,
                    default=ChannelConfig.arm_max_bars)
    ap.add_argument("--confirm-max-bars", dest="confirm_max_bars", type=int,
                    default=ChannelConfig.confirm_max_bars)
    ap.add_argument("--confirmation", dest="require_confirmation",
                    action=argparse.BooleanOptionalAction,
                    default=ChannelConfig.require_confirmation)
    ap.add_argument("--require-flow", dest="require_flow",
                    action=argparse.BooleanOptionalAction, default=ChannelConfig.require_flow)
    ap.add_argument("--swing-lookback", dest="swing_lookback", type=int,
                    default=ChannelConfig.swing_lookback)
    ap.add_argument("--target", dest="target_mode", default=ChannelConfig.target_mode,
                    choices=["measured", "rail", "rr", "pct"])
    ap.add_argument("--rr-multiple", dest="rr_multiple", type=float,
                    default=ChannelConfig.rr_multiple)
    ap.add_argument("--min-risk-bps", dest="min_risk_bps", type=float,
                    default=ChannelConfig.min_risk_bps)
    ap.add_argument("--max-risk-bps", dest="max_risk_bps", type=float,
                    default=ChannelConfig.max_risk_bps)
    ap.add_argument("--min-rr", dest="min_rr", type=float, default=ChannelConfig.min_rr)
    ap.add_argument("--max-hold-bars", dest="max_hold_bars", type=int,
                    default=ChannelConfig.max_hold_bars)
    ap.add_argument("--cost-bps", dest="cost_bps", type=float,
                    default=ChannelConfig.cost_bps)
    ap.add_argument("--max-trades-per-day", dest="max_trades_per_day", type=int,
                    default=ChannelConfig.max_trades_per_day,
                    help="optional cap; omitted means unlimited, including >5/day")
    ap.add_argument("--max-concurrent", dest="max_concurrent", type=int,
                    default=ChannelConfig.max_concurrent,
                    help="optional portfolio capacity; omitted means unlimited")
    ap.add_argument("--entry-mode", dest="entry_mode", default=ChannelConfig.entry_mode,
                    choices=["market", "maker_limit"])
    ap.add_argument("--limit-offset-bps", dest="limit_offset_bps", type=float,
                    default=ChannelConfig.limit_offset_bps)
    ap.add_argument("--fill-window-bars", dest="fill_window_bars", type=int,
                    default=ChannelConfig.fill_window_bars)
    ap.add_argument("--maker-fee-bps", dest="maker_fee_bps", type=float,
                    default=ChannelConfig.maker_fee_bps)
    ap.add_argument("--taker-fee-bps", dest="taker_fee_bps", type=float,
                    default=ChannelConfig.taker_fee_bps)
    ap.add_argument("--out", type=Path, default=OUT_ROOT)
    ap.add_argument("--i-am-unsealing", action="store_true",
                    help="required to touch the lockbox; use once, ever")
    return ap.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> ChannelConfig:
    values = {field.name: getattr(args, field.name) for field in fields(ChannelConfig)}
    values["channel_windows"] = tuple(values["channel_windows"])
    return ChannelConfig(**values)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = config_from_args(args)
    res = build_events(cfg, args.stage, unsealing=args.i_am_unsealing)

    tag = run_tag(cfg, args.stage)
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    res["events_df"].to_parquet(out / "events.parquet", index=False)
    res["trades_df"].to_parquet(out / "trades.parquet", index=False)
    res["orders_df"].to_parquet(out / "orders.parquet", index=False)
    res["funnel"].to_csv(out / "funnel.csv", index=False)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    (out / "summary.json").write_text(
        json.dumps(_summary(res), indent=2, default=str), encoding="utf-8")

    print(f"stage {args.stage}  {res['days']} days")
    print(res["funnel"].to_string(index=False))
    print(f"\ntrades {res['num_trades']:,} ({res['trades_per_day']:.2f}/day) "
          f"over {res['num_episodes']} episodes")
    print(f"E7 labelled candidates {res['num_events']:,}; daily trade cap "
          f"{cfg.max_trades_per_day if cfg.max_trades_per_day is not None else 'unlimited'}")
    if res["num_trades"]:
        print(f"TP-first {res['tp_first_rate']:.1%}  timeout {res['timeout_rate']:.1%}  "
              f"win {res['win_rate']:.1%}")
        print(f"gross R {res['mean_r_gross']:+.3f} (se {res['se_r_gross']:.3f})   "
              f"net R {res['mean_r_net']:+.3f}")
    print(f"\nwritten -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
