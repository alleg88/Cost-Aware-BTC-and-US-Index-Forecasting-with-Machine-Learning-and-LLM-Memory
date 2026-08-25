"""Causal JForex M1 adapter for USA500, USATECH and VOLIDX.

Raw Bid/Ask exports use naive ``Time (EET)`` timestamps.  This module applies
the sealed pre-Q2 cutoff before duplicate/alignment diagnostics, converts the
surviving rows through Europe/Helsinki to UTC, builds midpoint OHLCV, and keeps
crossed-quote diagnostics separate from the non-negative execution spread.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_TIME_COL = "time (eet)"
OHLC = ("open", "high", "low", "close")
INSTRUMENTS = ("USA500IDXUSD", "USATECHIDXUSD", "VOLIDXUSD")
STEMS = {
    "USA500IDXUSD": "usa500",
    "USATECHIDXUSD": "usatech",
    "VOLIDXUSD": "volidx",
}
LOCKBOX_START = pd.Timestamp("2026-04-01T00:00:00Z")
GRIDS = ("5min", "15min", "1h")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value: str | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _find_export(raw_dir: Path, instrument: str, side: str) -> Path:
    matches = sorted(raw_dir.glob(f"{instrument}_1 Min_{side.title()}_*.csv"))
    if len(matches) != 1:
        found = ", ".join(path.name for path in matches) or "none"
        raise FileNotFoundError(
            f"expected one {instrument} 1 Min {side} export in {raw_dir}; found {found}"
        )
    return matches[0]


def _repair_ohlc(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = frame.copy()
    high_floor = out[["open", "close"]].max(axis=1)
    low_ceiling = out[["open", "close"]].min(axis=1)
    invalid = (
        (out["high"] < out["low"])
        | (out["high"] < high_floor)
        | (out["low"] > low_ceiling)
    )
    out["high"] = pd.concat([out["high"], high_floor], axis=1).max(axis=1)
    out["low"] = pd.concat([out["low"], low_ceiling], axis=1).min(axis=1)
    return out, int(invalid.sum())


def _read_side(
    path: Path,
    side: str,
    *,
    end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = pd.read_csv(path)
    source.columns = [str(column).strip().lower() for column in source.columns]
    required = {RAW_TIME_COL, *OHLC, "volume"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")

    local = pd.to_datetime(
        source[RAW_TIME_COL], format="%Y.%m.%d %H:%M:%S", errors="raise"
    )
    utc = (
        local.dt.tz_localize(
            "Europe/Helsinki", ambiguous="infer", nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
    )
    cutoff_mask = utc < end_exclusive
    rows_before = len(source)
    source = source.loc[cutoff_mask, [*OHLC, "volume"]].copy()
    utc = utc.loc[cutoff_mask]

    for column in (*OHLC, "volume"):
        source[column] = pd.to_numeric(source[column], errors="raise")
    if not np.isfinite(source[[*OHLC, "volume"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path.name}: non-finite numeric value")
    if (source[list(OHLC)] <= 0).any().any():
        raise ValueError(f"{path.name}: non-positive price")
    if (source["volume"] < 0).any():
        raise ValueError(f"{path.name}: negative volume")

    source.index = pd.DatetimeIndex(utc, name="timestamp")
    source, repaired = _repair_ohlc(source)
    duplicates = int(source.index.duplicated(keep=False).sum())
    if duplicates:
        raise ValueError(f"{path.name}: duplicate timestamps before Q2: {duplicates}")
    source = source.sort_index()
    source = source.rename(columns={column: f"{column}_{side}" for column in OHLC})
    source = source.rename(columns={"volume": f"volume_{side}"})
    return source, {
        "source_rows": int(rows_before),
        "rows_after_cutoff": int(len(source)),
        "rows_removed_by_cutoff": int(rows_before - len(source)),
        "duplicate_timestamps": duplicates,
        "repaired_ohlc_rows": repaired,
        "source_file": path.name,
    }


def load_index_minutes(
    raw_dir: str | Path,
    instrument: str,
    end_exclusive: str | pd.Timestamp = LOCKBOX_START,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load one exact Bid/Ask M1 pair as causal midpoint bars and an audit."""
    raw_dir = Path(raw_dir)
    cutoff = _utc(end_exclusive)
    if cutoff > LOCKBOX_START:
        raise PermissionError("index loader cannot open Q2 2026 or later")

    bid, bid_audit = _read_side(
        _find_export(raw_dir, instrument, "bid"), "bid", end_exclusive=cutoff
    )
    ask, ask_audit = _read_side(
        _find_export(raw_dir, instrument, "ask"), "ask", end_exclusive=cutoff
    )
    if not bid.index.equals(ask.index):
        difference = bid.index.symmetric_difference(ask.index)
        raise ValueError(
            f"{instrument}: bid/ask timestamp grids differ ({len(difference)} rows)"
        )
    if bid.empty:
        raise ValueError(f"{instrument}: no rows before {cutoff.isoformat()}")

    joined = bid.join(ask, how="inner", validate="one_to_one")
    for column in OHLC:
        joined[column] = (joined[f"{column}_bid"] + joined[f"{column}_ask"]) / 2.0
    repaired_mid, midpoint_repaired = _repair_ohlc(joined[list(OHLC)])
    for column in OHLC:
        joined[column] = repaired_mid[column]

    joined["volume"] = joined[["volume_bid", "volume_ask"]].max(axis=1)
    joined["raw_spread_close"] = joined["close_ask"] - joined["close_bid"]
    joined["raw_spread_close_bps"] = (
        joined["raw_spread_close"] / joined["close"] * 10_000.0
    )
    joined["crossed_close"] = joined["raw_spread_close"] < 0
    joined["execution_spread_bps"] = joined["raw_spread_close_bps"].clip(lower=0.0)
    joined["instrument"] = instrument
    joined["bar_open"] = joined.index
    joined["available_at"] = joined.index + pd.Timedelta(minutes=1)
    joined["minute_count"] = 1
    joined["complete_bar"] = True

    if not joined.index.is_unique or not joined.index.is_monotonic_increasing:
        raise ValueError(f"{instrument}: surviving UTC index must be unique and sorted")
    if joined.index.max() >= cutoff:
        raise AssertionError("post-cutoff row survived index ingestion")

    audit: dict[str, Any] = {
        "instrument": instrument,
        "cutoff_exclusive": cutoff.isoformat(),
        "rows_after_cutoff": int(len(joined)),
        "rows_removed_by_cutoff": int(bid_audit["rows_removed_by_cutoff"]),
        "duplicate_timestamps": 0,
        "start_utc": joined.index.min().isoformat(),
        "end_utc": joined.index.max().isoformat(),
        "crossed_close_rows": int(joined["crossed_close"].sum()),
        "bid_repaired_ohlc_rows": int(bid_audit["repaired_ohlc_rows"]),
        "ask_repaired_ohlc_rows": int(ask_audit["repaired_ohlc_rows"]),
        "midpoint_repaired_ohlc_rows": midpoint_repaired,
        "midpoint_invalid_rows_after_repair": 0,
        "bid_source": bid_audit["source_file"],
        "ask_source": ask_audit["source_file"],
    }
    public_columns = [
        *OHLC,
        "volume",
        "raw_spread_close",
        "raw_spread_close_bps",
        "crossed_close",
        "execution_spread_bps",
        "instrument",
        "bar_open",
        "available_at",
        "minute_count",
        "complete_bar",
    ]
    return joined.loc[:, public_columns], audit


def resample_index_minutes(frame: pd.DataFrame, grid: str) -> pd.DataFrame:
    """Aggregate M1 midpoint bars into left-labelled causal complete-bar grids."""
    if frame.index.tz is None:
        raise ValueError("minute index must be timezone-aware")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("minute index must be unique and sorted")
    duration = pd.Timedelta(grid)
    expected = int(duration / pd.Timedelta(minutes=1))
    if expected < 1:
        raise ValueError("grid must be at least one minute")

    spec: dict[str, tuple[str, str]] = {
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        "volume": ("volume", "sum"),
        "minute_count": ("close", "count"),
    }
    if "raw_spread_close_bps" in frame:
        spec["raw_spread_close_bps"] = ("raw_spread_close_bps", "median")
    if "crossed_close" in frame:
        spec["crossed_close_rows"] = ("crossed_close", "sum")
    out = frame.resample(grid, label="left", closed="left").agg(**spec)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out["bar_open"] = out.index
    out["available_at"] = out.index + duration
    out["complete_bar"] = out["minute_count"].eq(expected)
    if "raw_spread_close_bps" in out:
        out["execution_spread_bps"] = out["raw_spread_close_bps"].clip(lower=0.0)
    if "instrument" in frame and not frame.empty:
        values = frame["instrument"].dropna().unique()
        if len(values) != 1:
            raise ValueError("one resampled grid cannot contain multiple instruments")
        out["instrument"] = values[0]
    if (out["high"] < out[["open", "close"]].max(axis=1)).any():
        raise ValueError(f"{grid}: high below open/close")
    if (out["low"] > out[["open", "close"]].min(axis=1)).any():
        raise ValueError(f"{grid}: low above open/close")
    return out


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def write_index_grids(
    raw_dir: str | Path,
    output_dir: str | Path,
    instrument: str,
    *,
    end_exclusive: str | pd.Timestamp = LOCKBOX_START,
) -> dict[str, Path]:
    """Write one instrument's pre-Q2 M1/M5/M15/H1 grids plus audit JSON."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    minute, audit = load_index_minutes(raw_dir, instrument, end_exclusive=end_exclusive)
    stem = STEMS.get(instrument, instrument.lower().replace("idxusd", ""))
    outputs: dict[str, Path] = {}
    frames = {"1m": minute, **{grid: resample_index_minutes(minute, grid) for grid in GRIDS}}
    grid_audit: dict[str, Any] = {}
    for grid, current in frames.items():
        path = output_dir / f"{stem}_{grid}_2021_2026.parquet"
        current.to_parquet(path, engine="pyarrow", index=True)
        outputs[grid] = path
        grid_audit[grid] = {
            "rows": int(len(current)),
            "complete_rows": int(current["complete_bar"].sum()),
            "start_utc": current.index.min().isoformat(),
            "end_utc": current.index.max().isoformat(),
            "max_available_at": pd.to_datetime(current["available_at"], utc=True)
            .max()
            .isoformat(),
            "output_sha256": _sha256(path),
        }
    audit["grids"] = grid_audit
    audit_path = output_dir / f"{stem}_market_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    outputs["audit"] = audit_path
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path(__file__).parent / "raw")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--instruments", nargs="+", default=list(INSTRUMENTS))
    args = parser.parse_args(argv)
    for instrument in args.instruments:
        outputs = write_index_grids(args.raw_dir, args.output_dir, instrument)
        print(f"{instrument}: " + ", ".join(f"{k}={v.name}" for k, v in outputs.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
