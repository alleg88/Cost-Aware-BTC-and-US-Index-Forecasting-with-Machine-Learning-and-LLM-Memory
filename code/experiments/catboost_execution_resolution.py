"""Frozen contracts for the CatBoost one-minute versus one-second study."""
from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from evaluation.trades_intrabar import simulate_bracket_trades_intrabar
from experiments.catboost_matched_ablation import (
    CALIBRATION_END,
    FORWARD_END,
    SELECTION_END,
    SELECTION_START,
    TAUS,
    WIDTHS,
    economic_ranking_key,
)


PROTOCOL_VERSION = "catboost-execution-resolution-v1"
TP_SL_PAIRS = ((150, 75), (150, 100), (200, 100))
HOLDS = (1, 2)
SIMULATOR_VERSION = hashlib.sha256(
    inspect.getsource(simulate_bracket_trades_intrabar).encode("utf-8")
).hexdigest()[:16]
EXPECTED_ARM_ROWS = {
    "economic_policy_grid_2024": 2970,
    "economic_candidate_winners_2024": 45,
    "selected_candidates_2024": 3,
    "calibration_policy_grid_2025h1": 198,
    "selected_policies_2025h1": 3,
    "forward_monthly": 27,
    "forward_quarterly": 9,
    "forward_summary": 3,
}


def policy_choices() -> tuple[tuple[float, tuple[int, int, int]], ...]:
    """Return the frozen 11 x 3 x 2 policy grid in deterministic order."""
    geometries = tuple(
        (tp_bps, sl_bps, hold)
        for tp_bps, sl_bps in TP_SL_PAIRS
        for hold in HOLDS
    )
    return tuple((tau, geometry) for geometry in geometries for tau in TAUS)


def _canonical(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        timestamp = (
            value.tz_localize("UTC")
            if value.tzinfo is None
            else value.tz_convert("UTC")
        )
        return timestamp.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    return value


def _content_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_policy_fingerprint(
    *,
    stage: str,
    width_bps: int,
    candidate_id: int,
    prediction_fingerprints: Sequence[str],
    m15_fingerprint: str,
    resolution: str,
    execution_data_fingerprint: str,
    fee_bps: float,
    model_name: str | None = None,
) -> str:
    if resolution not in {"1m", "1s"}:
        raise ValueError("resolution must be '1m' or '1s'")
    payload = {
            "protocol_version": PROTOCOL_VERSION,
            "stage": stage,
            "stage_span": {
                "selection": (SELECTION_START, SELECTION_END),
                "calibration": (SELECTION_END, CALIBRATION_END),
                "forward": (CALIBRATION_END, FORWARD_END),
            }.get(stage),
            "scoring_version": "fixed-grid-v1",
            "width_bps": int(width_bps),
            "candidate_id": int(candidate_id),
            "prediction_fingerprints": list(prediction_fingerprints),
            "m15_fingerprint": m15_fingerprint,
            "resolution": resolution,
            "execution_data_fingerprint": execution_data_fingerprint,
            "fee_bps_per_side": float(fee_bps),
            "simulator_version": SIMULATOR_VERSION,
            "policies": policy_choices(),
            "same_source_candle_tie": "stop_first",
        }
    if model_name is not None:
        payload["model_name"] = str(model_name)
    return _content_hash(payload)


def select_economic_candidates(
    winners: pd.DataFrame, *, widths: Sequence[int] = WIDTHS
) -> pd.DataFrame:
    """Select one adequacy-first economic candidate per DZ, without a trade gate."""
    rows = []
    for width in widths:
        choices = winners.loc[winners["width_bps"] == width]
        if choices.empty:
            raise ValueError(f"no economic candidates for DZ{width}")
        selected = min(
            choices.to_dict(orient="records"),
            key=lambda row: economic_ranking_key(row, n_segments=5),
        )
        rows.append({"objective": "economic", **selected})
    return pd.DataFrame(rows).sort_values("width_bps").reset_index(drop=True)


def assert_arm_artifact_counts(actual: Mapping[str, int]) -> None:
    mismatches = {
        name: (expected, actual.get(name))
        for name, expected in EXPECTED_ARM_ROWS.items()
        if actual.get(name) != expected
    }
    if mismatches:
        raise AssertionError(f"artifact row-count mismatch: {mismatches}")


def _utc(value: pd.Timestamp) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


class PartitionedIntrabarStore:
    """Load only execution data needed for a requested half-open UTC span."""

    def __init__(
        self,
        *,
        resolution: str,
        root: Path,
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        if resolution not in {"1m", "1s"}:
            raise ValueError("resolution must be '1m' or '1s'")
        self.resolution = resolution
        self.root = Path(root)
        self._manifest = dict(manifest or {})
        self._records = {
            str(record["month"]): dict(record)
            for record in self._manifest.get("partitions", [])
        }
        self._verified: set[str] = set()
        self._minute_frame: pd.DataFrame | None = None
        self._loaded_months: tuple[str, ...] = ()
        self.data_fingerprint = (
            file_sha256(self.root)
            if resolution == "1m"
            else _content_hash(self._manifest)
        )

    @classmethod
    def one_minute(cls, parquet_path: Path) -> "PartitionedIntrabarStore":
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return cls(resolution="1m", root=path)

    @classmethod
    def one_second(cls, partition_root: Path) -> "PartitionedIntrabarStore":
        root = Path(partition_root)
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("interval") != "1s":
            raise ValueError("one-second manifest interval must be 1s")
        if _utc(pd.Timestamp(manifest["sealed_start"])) != FORWARD_END:
            raise ValueError("one-second manifest sealed boundary is invalid")
        return cls(resolution="1s", root=root, manifest=manifest)

    @property
    def loaded_months(self) -> tuple[str, ...]:
        return self._loaded_months

    def _validate_span(
        self, start: pd.Timestamp, end: pd.Timestamp
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        start, end = _utc(start), _utc(end)
        if end <= start:
            raise ValueError("execution span end must be after start")
        if end > FORWARD_END or start >= FORWARD_END:
            raise ValueError(f"execution span crosses sealed boundary {FORWARD_END}")
        return start, end

    @staticmethod
    def _validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.loc[:, ["open", "high", "low", "close"]].copy()
        frame.index = pd.to_datetime(frame.index, utc=True)
        if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
            raise ValueError("execution timestamps must be sorted and unique")
        if len(frame) and frame.index[-1] >= FORWARD_END:
            raise ValueError(f"execution data crosses sealed boundary {FORWARD_END}")
        return frame

    def load_span(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        start, end = self._validate_span(start, end)
        if self.resolution == "1m":
            if self._minute_frame is None:
                source = pd.read_parquet(self.root)
                source.index = pd.to_datetime(source.index, utc=True)
                source = source.loc[source.index < FORWARD_END]
                self._minute_frame = self._validate_frame(source)
            self._loaded_months = ()
            return self._minute_frame.loc[
                (self._minute_frame.index >= start) & (self._minute_frame.index < end)
            ]

        final_instant = end - pd.Timedelta(nanoseconds=1)
        months = tuple(
            str(period)
            for period in pd.period_range(
                start.tz_localize(None).to_period("M"),
                final_instant.tz_localize(None).to_period("M"),
                freq="M",
            )
        )
        frames = []
        for month in months:
            record = self._records.get(month)
            if record is None:
                raise FileNotFoundError(f"missing 1s partition for {month}")
            path = self.root / str(record["file"])
            if month not in self._verified:
                if file_sha256(path) != record.get("output_sha256"):
                    raise ValueError(f"1s partition hash mismatch for {month}")
                self._verified.add(month)
            frames.append(self._validate_frame(pd.read_parquet(path)))
        self._loaded_months = months
        combined = pd.concat(frames)
        if not combined.index.is_unique:
            raise ValueError("execution timestamps must be unique across partitions")
        return combined.loc[(combined.index >= start) & (combined.index < end)]


def write_run_state(
    path: Path,
    *,
    status: str,
    detail: Mapping[str, Any] | None = None,
) -> None:
    if status not in {"running", "complete", "failed"}:
        raise ValueError("run status must be running, complete, or failed")
    payload = {**dict(detail or {}), "status": status}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.part")
    try:
        temporary.write_text(
            json.dumps(_canonical(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
