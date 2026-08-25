"""Opaque pre-open sealing and post-open exact Q2 input loaders."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import pandas as pd

from experiments.final_q2_lockbox_contract import Q2_END, Q2_START
from experiments.final_q2_lockbox_state import OpeningIdentity, require_global_opening


BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class OpaqueFileIdentity:
    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": int(self.size), "sha256": self.sha256}


@dataclass(frozen=True)
class OpaqueCopyIdentity:
    source_path: str
    destination_path: str
    source_size: int
    destination_size: int
    source_sha256: str
    destination_sha256: str
    cutoff_exclusive: str
    source_audit_path: str | None = None
    source_audit_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "source_size": int(self.source_size),
            "destination_size": int(self.destination_size),
            "source_sha256": self.source_sha256,
            "destination_sha256": self.destination_sha256,
            "cutoff_exclusive": self.cutoff_exclusive,
            "source_audit_path": self.source_audit_path,
            "source_audit_sha256": self.source_audit_sha256,
        }


def opaque_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def opaque_file_identity(path: str | Path) -> OpaqueFileIdentity:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    return OpaqueFileIdentity(
        path=source.as_posix(),
        size=int(source.stat().st_size),
        sha256=opaque_sha256(source),
    )


def _cutoff_audit(audit: Mapping[str, Any]) -> str:
    cutoff = str(audit.get("cutoff_exclusive", ""))
    if not cutoff or pd.Timestamp(cutoff) != Q2_START:
        raise ValueError("audit cutoff_exclusive must equal the Q2 start")
    for key in ("end_utc", "max_available_at", "maximum_available_time"):
        value = audit.get(key)
        if value is not None and pd.Timestamp(value) >= Q2_START:
            raise ValueError(f"audit {key} is not strictly pre-Q2")
    return Q2_START.isoformat()


def copy_audited_pre_q2(
    source: str | Path,
    destination: str | Path,
    audit: Mapping[str, Any],
    *,
    expected_source_sha256: str | None = None,
    audit_path: str | Path | None = None,
) -> OpaqueCopyIdentity:
    cutoff = _cutoff_audit(audit)
    source_path = Path(source)
    destination_path = Path(destination)
    source_identity = opaque_file_identity(source_path)
    if expected_source_sha256 is None:
        raise ValueError("warm-up copy requires an audit-bound source sha256")
    if source_identity.sha256 != str(expected_source_sha256).lower():
        raise ValueError("warm-up source differs from its audit-bound sha256")
    audit_identity = (
        None if audit_path is None else opaque_file_identity(Path(audit_path))
    )
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists():
        destination_identity = opaque_file_identity(destination_path)
        if (
            destination_identity.size != source_identity.size
            or destination_identity.sha256 != source_identity.sha256
        ):
            raise FileExistsError(
                f"existing warm-up snapshot differs from source: {destination_path}"
            )
    else:
        temporary = destination_path.with_name(destination_path.name + f".tmp.{os.getpid()}")
        if temporary.exists():
            raise FileExistsError(temporary)
        with source_path.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=BUFFER_SIZE)
        if (
            temporary.stat().st_size != source_identity.size
            or opaque_sha256(temporary) != source_identity.sha256
        ):
            raise IOError("opaque warm-up copy failed byte/hash reconciliation")
        os.replace(temporary, destination_path)
        destination_identity = opaque_file_identity(destination_path)
    return OpaqueCopyIdentity(
        source_path=source_identity.path,
        destination_path=destination_identity.path,
        source_size=source_identity.size,
        destination_size=destination_identity.size,
        source_sha256=source_identity.sha256,
        destination_sha256=destination_identity.sha256,
        cutoff_exclusive=cutoff,
        source_audit_path=(None if audit_identity is None else audit_identity.path),
        source_audit_sha256=(
            None if audit_identity is None else audit_identity.sha256
        ),
    )


def collect_q2_archive_identities(
    paths: Sequence[str | Path],
) -> tuple[OpaqueFileIdentity, ...]:
    normalized = sorted({Path(path) for path in paths}, key=lambda path: path.as_posix())
    if len(normalized) != len(paths):
        raise ValueError("Q2 archive paths must be unique")
    return tuple(opaque_file_identity(path) for path in normalized)


def _utc(value: str | pd.Timestamp, *, label: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return stamp.tz_convert("UTC")


def _validate_interval(start: pd.Timestamp, end: pd.Timestamp) -> None:
    if start != Q2_START or end != Q2_END:
        raise PermissionError("decoder interval must equal the registered Q2 lockbox")


def _timestamp_index(
    frame: pd.DataFrame, *, timestamp_column: str
) -> pd.DataFrame:
    if timestamp_column in frame.columns:
        timestamps = pd.to_datetime(frame[timestamp_column], utc=True, errors="raise")
        output = frame.drop(columns=[timestamp_column]).copy()
        output.index = pd.DatetimeIndex(timestamps, name="timestamp")
    elif isinstance(frame.index, pd.DatetimeIndex):
        output = frame.copy()
        index = output.index
        output.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
        output.index.name = "timestamp"
    else:
        raise ValueError(f"input misses timestamp column {timestamp_column!r}")
    if output.empty:
        raise ValueError("Q2 input is empty")
    if output.index.has_duplicates or not output.index.is_monotonic_increasing:
        raise ValueError("Q2 timestamps must be unique and monotonically increasing")
    return output


def _validate_q2_frame(
    frame: pd.DataFrame, *, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    if (frame.index < start).any() or (frame.index >= end).any():
        raise ValueError("Q2 input crosses the registered half-open interval")
    return frame


def load_exact_q2_parquet(
    path: str | Path,
    *,
    identity: OpeningIdentity,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    require_global_opening(identity)
    start_utc, end_utc = _utc(start, label="start"), _utc(end, label="end")
    _validate_interval(start_utc, end_utc)
    frame = _timestamp_index(
        pd.read_parquet(Path(path)), timestamp_column=timestamp_column
    )
    return _validate_q2_frame(frame, start=start_utc, end=end_utc)


def load_q2_parquet_partition(
    path: str | Path,
    *,
    identity: OpeningIdentity,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Read only the registered Q2 row partition from a broader parquet source."""
    require_global_opening(identity)
    start_utc, end_utc = _utc(start, label="start"), _utc(end, label="end")
    _validate_interval(start_utc, end_utc)
    frame = pd.read_parquet(
        Path(path),
        filters=[
            (timestamp_column, ">=", start_utc.to_pydatetime()),
            (timestamp_column, "<", end_utc.to_pydatetime()),
        ],
    )
    output = _timestamp_index(frame, timestamp_column=timestamp_column)
    return _validate_q2_frame(output, start=start_utc, end=end_utc)


def _load_csv(
    path: str | Path,
    *,
    timestamp_column: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    frame = _timestamp_index(
        pd.read_csv(Path(path)), timestamp_column=timestamp_column
    )
    return _validate_q2_frame(frame, start=start, end=end)


def load_exact_q2_csv_pair(
    bid_path: str | Path,
    ask_path: str | Path,
    *,
    identity: OpeningIdentity,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    timestamp_column: str,
) -> pd.DataFrame:
    require_global_opening(identity)
    start_utc, end_utc = _utc(start, label="start"), _utc(end, label="end")
    _validate_interval(start_utc, end_utc)
    bid = _load_csv(
        bid_path,
        timestamp_column=timestamp_column,
        start=start_utc,
        end=end_utc,
    )
    ask = _load_csv(
        ask_path,
        timestamp_column=timestamp_column,
        start=start_utc,
        end=end_utc,
    )
    if not bid.index.equals(ask.index):
        raise ValueError("bid and ask timestamp grids differ")
    return bid.add_suffix("_bid").join(ask.add_suffix("_ask"), how="inner", validate="one_to_one")


def _load_q2_jforex_side(
    path: str | Path,
    side: str,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    from data.index_market import OHLC, RAW_TIME_COL, _repair_ohlc

    source = pd.read_csv(Path(path), dtype=str)
    source.columns = [str(column).strip().lower() for column in source.columns]
    required = {RAW_TIME_COL, *OHLC, "volume"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"JForex {side} source misses columns: {sorted(missing)}")
    local = pd.to_datetime(
        source[RAW_TIME_COL], format="%Y.%m.%d %H:%M:%S", errors="raise"
    )
    timestamps = (
        local.dt.tz_localize(
            "Europe/Helsinki", ambiguous="infer", nonexistent="shift_forward"
        )
        .dt.tz_convert("UTC")
    )
    keep = timestamps.ge(start) & timestamps.lt(end)
    scoped = source.loc[keep, [*OHLC, "volume"]].copy()
    scoped.index = pd.DatetimeIndex(timestamps.loc[keep], name="timestamp")
    if scoped.empty:
        raise ValueError(f"JForex {side} source has no Q2 rows")
    for column in (*OHLC, "volume"):
        scoped[column] = pd.to_numeric(scoped[column], errors="raise")
    values = scoped[[*OHLC, "volume"]].to_numpy(float)
    if not pd.notna(values).all() or not (abs(values) < float("inf")).all():
        raise ValueError(f"JForex {side} Q2 values must be finite")
    if (scoped[list(OHLC)] <= 0.0).any().any() or (scoped["volume"] < 0.0).any():
        raise ValueError(f"JForex {side} Q2 values are outside the valid range")
    scoped, _ = _repair_ohlc(scoped)
    if scoped.index.has_duplicates:
        raise ValueError(f"JForex {side} Q2 timestamps are duplicated")
    scoped = scoped.sort_index()
    return scoped.rename(
        columns={
            **{column: f"{column}_{side}" for column in OHLC},
            "volume": f"volume_{side}",
        }
    )


def load_q2_index_minutes(
    bid_path: str | Path,
    ask_path: str | Path,
    *,
    identity: OpeningIdentity,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    instrument: str,
) -> pd.DataFrame:
    """Decode only Q2 JForex rows and reproduce the causal midpoint M1 adapter."""
    from data.index_market import OHLC, _repair_ohlc

    require_global_opening(identity)
    start_utc, end_utc = _utc(start, label="start"), _utc(end, label="end")
    _validate_interval(start_utc, end_utc)
    bid = _load_q2_jforex_side(
        bid_path, "bid", start=start_utc, end=end_utc
    )
    ask = _load_q2_jforex_side(
        ask_path, "ask", start=start_utc, end=end_utc
    )
    if not bid.index.equals(ask.index):
        raise ValueError("JForex Q2 bid and ask timestamp grids differ")
    joined = bid.join(ask, how="inner", validate="one_to_one")
    for column in OHLC:
        joined[column] = (
            joined[f"{column}_bid"] + joined[f"{column}_ask"]
        ) / 2.0
    repaired, _ = _repair_ohlc(joined[list(OHLC)])
    joined.loc[:, list(OHLC)] = repaired
    joined["volume"] = joined[["volume_bid", "volume_ask"]].max(axis=1)
    joined["raw_spread_close"] = joined["close_ask"] - joined["close_bid"]
    joined["raw_spread_close_bps"] = (
        joined["raw_spread_close"] / joined["close"] * 10_000.0
    )
    joined["crossed_close"] = joined["raw_spread_close"].lt(0.0)
    joined["execution_spread_bps"] = joined["raw_spread_close_bps"].clip(
        lower=0.0
    )
    joined["instrument"] = str(instrument)
    joined["bar_open"] = joined.index
    joined["available_at"] = joined.index + pd.Timedelta(minutes=1)
    joined["minute_count"] = 1
    joined["complete_bar"] = True
    public = [
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
    return _validate_q2_frame(joined.loc[:, public], start=start_utc, end=end_utc)


__all__ = [
    "OpaqueCopyIdentity",
    "OpaqueFileIdentity",
    "collect_q2_archive_identities",
    "copy_audited_pre_q2",
    "load_exact_q2_csv_pair",
    "load_exact_q2_parquet",
    "load_q2_index_minutes",
    "load_q2_parquet_partition",
    "opaque_file_identity",
    "opaque_sha256",
]
