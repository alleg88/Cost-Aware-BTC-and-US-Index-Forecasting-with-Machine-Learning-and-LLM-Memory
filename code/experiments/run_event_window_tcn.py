"""Run the bounded, development-only causal event-window TCN study.

The runner is intentionally stricter than an interactive notebook: it rejects
non-development stages before input discovery, bounds every parquet read in
PyArrow, hashes protocol/source/input identity, and updates the Notebook J
reader pointer only after a complete full-development run.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluation.event_window_economics import (
    EventLabelConfig,
    PolicyReplay,
    label_window_steps,
    replay_first_crossing,
    sweep_event_thresholds,
)
from experiments.event_window_dataset import (
    CONTEXT_FEATURES,
    SEQUENCE_FEATURES,
    EventWindowSequences,
    build_event_window_sequences,
)
from experiments.event_window_tcn import (
    OOFSequenceResult,
    TCNConfig,
    run_event_window_oof,
)
from features.event_window_inputs import (
    build_five_minute_feature_frame,
    build_positioning_feature_frame,
)
from features.event_windows import (
    EventWindowConfig,
    build_event_window_manifest,
    build_hourly_channel_context,
    causal_activity_ratio,
    project_hourly_channels,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_tcn"
DEV_START = pd.Timestamp("2021-01-01", tz="UTC")
DEV_END = pd.Timestamp("2025-07-01", tz="UTC")
OOF_START = pd.Timestamp("2022-01-01", tz="UTC")
OOF_END = DEV_END
SMOKE_TRAIN_END = pd.Timestamp("2021-01-15", tz="UTC")
SMOKE_VALID_START = pd.Timestamp("2022-01-01", tz="UTC")
SMOKE_VALID_END = pd.Timestamp("2022-01-15", tz="UTC")
SMOKE_INTERVALS = (
    (DEV_START, SMOKE_TRAIN_END),
    (SMOKE_VALID_START, SMOKE_VALID_END),
)

REQUIRED_ARTIFACTS = (
    "protocol.json",
    "summary.json",
    "window_manifest.parquet",
    "window_audit.csv",
    "feature_profile.csv",
    "high_correlation_pairs.csv",
    "labels_rr2.parquet",
    "labels_rr3.parquet",
    "oof_scores.parquet",
    "fold_audit.csv",
    "threshold_frontier.csv",
    "selected_trades.parquet",
    "daily_frequency.csv",
    "side_year_breakdown.csv",
    "score_deciles.csv",
    "episode_bootstrap.csv",
    "example_windows.parquet",
)

_SOURCE_FILES = (
    CODE_ROOT / "features" / "event_windows.py",
    CODE_ROOT / "features" / "event_window_inputs.py",
    CODE_ROOT / "experiments" / "event_window_dataset.py",
    CODE_ROOT / "evaluation" / "event_window_economics.py",
    CODE_ROOT / "experiments" / "event_window_tcn.py",
    Path(__file__),
)


@dataclass(frozen=True)
class EventWindowStudyConfig:
    symbol: str = "BTCUSDT"
    dev_start: str = "2021-01-01"
    dev_end_exclusive: str = "2025-07-01"
    rr_primary: float = 2.0
    rr_sensitivity: float = 3.0
    risk_pct: float = 1.0
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42
    high_correlation_abs_spearman: float = 0.90
    expected_full_dev_windows: int = 14_510
    window: EventWindowConfig = field(default_factory=EventWindowConfig)
    model: TCNConfig = field(default_factory=TCNConfig)

    def __post_init__(self) -> None:
        start = pd.Timestamp(self.dev_start)
        end = pd.Timestamp(self.dev_end_exclusive)
        if end <= start:
            raise ValueError("dev_end_exclusive must be later than dev_start")
        if self.rr_primary <= 0.0 or self.rr_sensitivity <= 0.0:
            raise ValueError("RR multiples must be positive")
        if self.risk_pct <= 0.0:
            raise ValueError("risk_pct must be positive")
        if self.bootstrap_draws < 1 or self.bootstrap_seed < 0:
            raise ValueError("bootstrap settings are invalid")
        if not 0.0 < self.high_correlation_abs_spearman <= 1.0:
            raise ValueError("correlation threshold must be in (0, 1]")
        if self.expected_full_dev_windows < 1:
            raise ValueError("expected_full_dev_windows must be positive")


@dataclass(frozen=True)
class LoadedInputs:
    minute: pd.DataFrame
    five_minute: pd.DataFrame
    hourly: pd.DataFrame
    positioning: pd.DataFrame
    max_loaded_timestamp: pd.Timestamp
    input_fingerprint: str
    read_start: pd.Timestamp
    read_end_exclusive: pd.Timestamp


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    protocol: dict[str, object]
    summary: dict[str, object]


class ProtocolMismatchError(RuntimeError):
    """Raised when the frozen full-development detector count changes."""


def _utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    return (
        timestamp.tz_localize("UTC")
        if timestamp.tzinfo is None
        else timestamp.tz_convert("UTC")
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_jsonable,
    ).encode("utf-8")


def _sha_payload(payload: object) -> str:
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_version() -> str:
    digest = hashlib.sha256()
    for path in _SOURCE_FILES:
        digest.update(str(path.relative_to(CODE_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol_dict(
    config: EventWindowStudyConfig = EventWindowStudyConfig(),
    *,
    stage: str = "dev",
) -> dict[str, object]:
    """Return the reader-visible frozen protocol, without run-time outcomes."""
    return {
        "stage": stage,
        "symbol": config.symbol,
        "dev_start": config.dev_start,
        "dev_end_exclusive": config.dev_end_exclusive,
        "development_start": config.dev_start,
        "development_end_exclusive": config.dev_end_exclusive,
        "side_dataset": "pooled",
        "decision_cadence": "5min",
        "execution_cadence": "1min",
        "pre_window_bars": config.window.pre_context_bars,
        "active_window_bars": config.window.active_bars,
        "max_trades_per_window": 1,
        "cross_window_capacity": "unlimited",
        "primary_score_threshold": 0.0,
        "expected_full_dev_windows": config.expected_full_dev_windows,
        "desired_trades_per_day": [3.0, 5.0],
        "admissible_trades_per_day": [2.0, 5.0],
        "selection_rr": config.rr_primary,
        "sensitivity_rr": config.rr_sensitivity,
        "sensitivity_can_select": False,
        "entry": "next 1m open at each 5m decision boundary",
        "stop": "12 completed 5m bars plus 5 bps buffer",
        "max_hold_minutes": 120,
        "cost_bps": 10.0,
        "risk_pct_fixed_initial_equity": config.risk_pct,
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_seed": config.bootstrap_seed,
        "high_correlation_abs_spearman": config.high_correlation_abs_spearman,
        "threshold_frontier": "exploratory OOF diagnostic only",
        "forward_or_lockbox_loaded": False,
        "window_config": asdict(config.window),
        "model_config": asdict(config.model),
    }


def _input_paths(config: EventWindowStudyConfig, data_root: Path) -> dict[str, Path]:
    symbol = config.symbol.lower()
    return {
        "minute": data_root / f"{symbol}_1m_2021_2026.parquet",
        "five_minute": data_root / f"{symbol}_5min_2021_2026.parquet",
        "hourly": data_root / f"{symbol}_1h_2021_2026.parquet",
        "positioning": data_root / f"{symbol}_positioning_15min_2021_2026.parquet",
    }


def _fingerprint_inputs(paths: dict[str, Path]) -> str:
    records: list[dict[str, object]] = []
    for name, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(path)
        stat = path.stat()
        records.append(
            {
                "name": name,
                "path": str(path.resolve()),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return _sha_payload(records)


def _load_bounded_parquet(
    path: Path,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Pass both temporal bounds to PyArrow before pandas receives any row."""
    schema = pq.read_schema(path)
    index_field = "timestamp" if "timestamp" in schema.names else "__index_level_0__"
    filters = [(index_field, ">=", start), (index_field, "<", end)]
    frame = pd.read_parquet(path, filters=filters)
    frame.index = pd.to_datetime(frame.index, utc=True, errors="raise")
    frame = frame.sort_index(kind="stable")
    if frame.index.has_duplicates:
        raise ValueError(f"{path.name} contains duplicate timestamps")
    if not frame.empty and (frame.index.min() < start or frame.index.max() >= end):
        raise AssertionError(f"bounded parquet read crossed [{start}, {end}): {path}")
    if "count" in frame and "trade_count" not in frame:
        frame = frame.rename(columns={"count": "trade_count"})
    return frame


def load_inputs(
    config: EventWindowStudyConfig,
    *,
    data_root: Path,
    smoke: bool,
) -> LoadedInputs:
    paths = _input_paths(config, Path(data_root))
    input_fingerprint = _fingerprint_inputs(paths)
    start = _utc(config.dev_start)
    configured_end = _utc(config.dev_end_exclusive)
    if smoke:
        frames = {
            name: pd.concat(
                [
                    _load_bounded_parquet(path, start=left, end=right)
                    for left, right in SMOKE_INTERVALS
                ]
            ).sort_index(kind="stable")
            for name, path in paths.items()
        }
        end = SMOKE_VALID_END
    else:
        end = configured_end
        frames = {
            name: _load_bounded_parquet(path, start=start, end=end)
            for name, path in paths.items()
        }
    for name, frame in frames.items():
        if frame.index.has_duplicates:
            raise ValueError(f"{name} smoke/full input contains duplicate timestamps")
    maxima = [frame.index.max() for frame in frames.values() if not frame.empty]
    maximum = max(maxima) if maxima else pd.Timestamp.min.tz_localize("UTC")
    return LoadedInputs(
        minute=frames["minute"],
        five_minute=frames["five_minute"],
        hourly=frames["hourly"],
        positioning=frames["positioning"],
        max_loaded_timestamp=maximum,
        input_fingerprint=input_fingerprint,
        read_start=start,
        read_end_exclusive=end,
    )


def _atomic_json(path: Path, payload: object, *, attempts: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    content = json.dumps(payload, indent=2, sort_keys=True, default=_jsonable)
    for attempt in range(attempts):
        temporary.write_text(content, encoding="utf-8")
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))


def _publish_latest_dev(
    *,
    output_root: Path,
    run_dir: Path,
    run_hash: str,
    protocol_hash: str,
) -> None:
    _atomic_json(
        output_root / "latest_dev.json",
        {
            "run_hash": run_hash,
            "relative_path": run_dir.relative_to(output_root).as_posix(),
            "protocol_hash": protocol_hash,
        },
    )


class _ArtifactStore:
    def __init__(
        self,
        run_dir: Path,
        *,
        run_hash: str,
        protocol_hash: str,
        input_hash: str,
        source_hash: str,
    ) -> None:
        self.run_dir = run_dir
        self.state_path = run_dir / "run_state.json"
        self.identity = {
            "run_hash": run_hash,
            "protocol_hash": protocol_hash,
            "input_hash": input_hash,
            "source_hash": source_hash,
        }
        run_dir.mkdir(parents=True, exist_ok=True)
        self.state: dict[str, object] = {
            **self.identity,
            "status": "running",
            "artifacts": {},
        }
        if self.state_path.exists():
            existing = json.loads(self.state_path.read_text(encoding="utf-8"))
            if all(existing.get(key) == value for key, value in self.identity.items()):
                self.state = existing
        self._flush()

    def start(self) -> None:
        self.state["status"] = "running"
        self.state.pop("error", None)
        self._flush()

    def _flush(self) -> None:
        _atomic_json(self.state_path, self.state)

    def valid(self, name: str) -> bool:
        path = self.run_dir / name
        record = self.state.get("artifacts", {}).get(name)  # type: ignore[union-attr]
        return bool(
            path.is_file()
            and record
            and _sha_file(path) == record.get("sha256")
        )

    def all_required_valid(self) -> bool:
        return all(self.valid(name) for name in REQUIRED_ARTIFACTS)

    def _record(self, name: str) -> None:
        path = self.run_dir / name
        artifacts = self.state.setdefault("artifacts", {})
        artifacts[name] = {  # type: ignore[index]
            "sha256": _sha_file(path),
            "size": path.stat().st_size,
        }
        self._flush()

    def write_json(self, name: str, payload: object) -> None:
        _atomic_json(self.run_dir / name, payload)
        self._record(name)

    def write_csv(self, name: str, frame: pd.DataFrame) -> None:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        self._record(name)

    def write_frame(self, name: str, frame: pd.DataFrame) -> None:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        self._record(name)

    def frame(self, name: str, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if self.valid(name):
            return pd.read_parquet(self.run_dir / name)
        frame = builder()
        self.write_frame(name, frame)
        return frame

    def csv(self, name: str, builder: Callable[[], pd.DataFrame]) -> pd.DataFrame:
        if self.valid(name):
            return pd.read_csv(self.run_dir / name)
        frame = builder()
        self.write_csv(name, frame)
        return frame

    def complete(self, summary: dict[str, object]) -> None:
        self.state["status"] = "complete"
        self.state["summary"] = summary
        self._flush()

    def fail(self, error: BaseException) -> None:
        self.state["status"] = "failed"
        self.state["error"] = repr(error)
        self._flush()


def episode_bootstrap(
    ledger: pd.DataFrame,
    *,
    draws: int = 2_000,
    seed: int = 42,
) -> pd.DataFrame:
    """Resample complete channel episodes in a paired model/baseline ledger."""
    if draws < 1:
        raise ValueError("draws must be positive")
    required = {"channel_episode_id", "model_net_r", "baseline_net_r"}
    missing = sorted(required.difference(ledger.columns))
    if missing:
        raise ValueError(f"paired ledger missing columns: {missing}")
    grouped = (
        ledger.assign(
            model_net_r=pd.to_numeric(ledger["model_net_r"], errors="coerce").fillna(0.0),
            baseline_net_r=pd.to_numeric(
                ledger["baseline_net_r"], errors="coerce"
            ).fillna(0.0),
        )
        .groupby("channel_episode_id", sort=True)[["model_net_r", "baseline_net_r"]]
        .sum()
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []
    if grouped.empty:
        return pd.DataFrame(
            {
                "draw": np.arange(draws, dtype=int),
                "model_total_net_r": np.zeros(draws),
                "baseline_total_net_r": np.zeros(draws),
                "delta_total_net_r": np.zeros(draws),
            }
        )
    values = grouped.to_numpy(dtype=float)
    for draw in range(draws):
        selected = rng.integers(0, len(values), size=len(values))
        model_total, baseline_total = values[selected].sum(axis=0)
        rows.append(
            {
                "draw": draw,
                "model_total_net_r": float(model_total),
                "baseline_total_net_r": float(baseline_total),
                "delta_total_net_r": float(model_total - baseline_total),
            }
        )
    return pd.DataFrame(rows)


def _empty_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "window_id",
            "channel_episode_id",
            "side",
            "step",
            "decision_time",
            "fold_id",
            "score",
        ]
    )


def _feature_diagnostics(
    sequences: EventWindowSequences,
    correlation_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    layers = (
        (
            "sequence",
            np.asarray(sequences.sequence, dtype=float),
            np.asarray(sequences.sequence_valid, dtype=bool),
            tuple(sequences.sequence_features),
        ),
        (
            "context",
            np.asarray(sequences.context, dtype=float),
            np.asarray(sequences.decision_valid, dtype=bool),
            tuple(sequences.context_features),
        ),
    )
    for layer, tensor, valid, names in layers:
        matrix = tensor[valid] if tensor.size else np.empty((0, len(names)))
        for column, name in enumerate(names):
            values = matrix[:, column] if len(matrix) else np.array([], dtype=float)
            finite = values[np.isfinite(values)]
            profiles.append(
                {
                    "layer": layer,
                    "feature": name,
                    "rows": int(len(values)),
                    "finite_count": int(len(finite)),
                    "missing_count": int(len(values) - len(finite)),
                    "finite_fraction": (
                        float(len(finite) / len(values)) if len(values) else np.nan
                    ),
                    "missing_fraction": (
                        float(1.0 - len(finite) / len(values)) if len(values) else np.nan
                    ),
                    "missing_rate": (
                        float(1.0 - len(finite) / len(values)) if len(values) else np.nan
                    ),
                    "minimum": float(np.min(finite)) if len(finite) else np.nan,
                    "median": float(np.median(finite)) if len(finite) else np.nan,
                    "std": float(np.std(finite)) if len(finite) else np.nan,
                    "maximum": float(np.max(finite)) if len(finite) else np.nan,
                }
            )
        if len(matrix) < 2 or len(names) < 2:
            continue
        correlations = pd.DataFrame(matrix, columns=names).corr(method="spearman")
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                value = correlations.iat[left, right]
                if np.isfinite(value) and abs(value) >= correlation_floor:
                    pairs.append(
                        {
                            "layer": layer,
                            "feature_a": names[left],
                            "feature_b": names[right],
                            "spearman": float(value),
                            "abs_spearman": float(abs(value)),
                        }
                    )
    profile = pd.DataFrame(profiles)
    pair_columns = (
        "layer",
        "feature_a",
        "feature_b",
        "spearman",
        "abs_spearman",
    )
    high_pairs = pd.DataFrame(pairs, columns=pair_columns)
    if not high_pairs.empty:
        high_pairs = high_pairs.sort_values(
            ["abs_spearman", "layer", "feature_a", "feature_b"],
            ascending=[False, True, True, True],
            kind="stable",
        )
    return profile, high_pairs


def _threshold_grid(scores: pd.DataFrame) -> list[float]:
    if "score" not in scores:
        return [0.0]
    finite = pd.to_numeric(scores["score"], errors="coerce")
    finite = finite[np.isfinite(finite) & finite.ge(0.0)]
    if finite.empty:
        return [0.0]
    quantiles = finite.quantile(np.linspace(0.0, 1.0, 101)).to_numpy(dtype=float)
    return sorted({0.0, *(float(value) for value in quantiles if np.isfinite(value))})


def _mark_exploratory_frontier(frontier: pd.DataFrame) -> pd.DataFrame:
    out = frontier.copy()
    primary = pd.to_numeric(out.get("threshold"), errors="coerce").eq(0.0)
    out["primary_registered"] = primary
    out["exploratory"] = ~primary
    out["can_establish_success"] = primary
    out["display_best"] = False
    if out.empty:
        return out
    candidates = out.loc[~primary & out["frequency_admissible"].astype(bool)]
    if candidates.empty:
        candidates = out.loc[~primary]
    if candidates.empty:
        return out
    ranked = candidates.assign(
        _total=pd.to_numeric(out["total_net_r"], errors="coerce").fillna(-np.inf),
        _mean=pd.to_numeric(out["mean_net_r"], errors="coerce").fillna(-np.inf),
    ).sort_values(
        ["_total", "_mean", "threshold"],
        ascending=[False, False, False],
        kind="stable",
    )
    out.loc[ranked.index[0], "display_best"] = True
    return out


def _same_entries_rr3(selected: pd.DataFrame, labels_rr3: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return selected.copy()
    sensitivity_columns = [
        "window_id",
        "step",
        "entry",
        "stop",
        "target",
        "risk_bps",
        "outcome",
        "r_gross",
        "r_net",
        "path_observed",
    ]
    available = [column for column in sensitivity_columns if column in labels_rr3]
    sensitivity = labels_rr3[available].copy()
    if sensitivity.duplicated(["window_id", "step"]).any():
        raise ValueError("RR3 labels must be unique by window_id and step")
    sensitivity = sensitivity.rename(
        columns={
            column: f"rr3_{column}"
            for column in available
            if column not in {"window_id", "step"}
        }
    )
    merged = selected.merge(
        sensitivity,
        on=["window_id", "step"],
        how="left",
        validate="one_to_one",
        indicator="_rr3_match",
    )
    if not merged["_rr3_match"].eq("both").all():
        raise ValueError("every selected RR2 entry must have the same RR3 key")
    merged = merged.drop(columns="_rr3_match")
    if set(zip(merged["window_id"], merged["step"], strict=True)) != set(
        zip(selected["window_id"], selected["step"], strict=True)
    ):
        raise AssertionError("RR3 sensitivity changed selected entry keys")
    return merged


def _daily_report(replay: PolicyReplay, risk_pct: float) -> pd.DataFrame:
    daily = replay.daily_frequency.copy()
    daily["fixed_equity_return_pct"] = daily["net_r"] * float(risk_pct)
    for level in (2, 3, 5):
        daily[f"profit_ge_{level}pct"] = daily["fixed_equity_return_pct"].ge(level)
    return daily.reset_index()


def _paired_ledger(
    scores: pd.DataFrame,
    model: PolicyReplay,
    baseline: PolicyReplay,
) -> pd.DataFrame:
    columns = [
        "window_id",
        "channel_episode_id",
        "model_net_r",
        "baseline_net_r",
    ]
    if scores.empty:
        return pd.DataFrame(columns=columns)
    windows = scores[["window_id", "channel_episode_id"]].drop_duplicates("window_id")

    def totals(replay: PolicyReplay, name: str) -> pd.Series:
        if replay.trades.empty:
            return pd.Series(dtype=float, name=name)
        observed = replay.trades["path_observed"].astype(bool)
        return (
            pd.to_numeric(replay.trades.loc[observed, "r_net"], errors="coerce")
            .groupby(replay.trades.loc[observed, "window_id"])
            .sum()
            .rename(name)
        )

    ledger = windows.merge(
        totals(model, "model_net_r"),
        left_on="window_id",
        right_index=True,
        how="left",
    ).merge(
        totals(baseline, "baseline_net_r"),
        left_on="window_id",
        right_index=True,
        how="left",
    )
    ledger[["model_net_r", "baseline_net_r"]] = ledger[
        ["model_net_r", "baseline_net_r"]
    ].fillna(0.0)
    return ledger.loc[:, columns]


def _score_deciles(scores: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "score_decile",
        "rows",
        "mean_score",
        "mean_net_r",
        "total_net_r",
        "tp_rate",
        "sl_rate",
        "timeout_rate",
    )
    if scores.empty:
        return pd.DataFrame(columns=columns)
    joined = scores[["window_id", "step", "score"]].merge(
        labels[
            ["window_id", "step", "model_target_valid", "r_net", "outcome"]
        ],
        on=["window_id", "step"],
        how="left",
        validate="one_to_one",
    )
    joined = joined[
        joined["model_target_valid"].fillna(False).astype(bool)
        & np.isfinite(pd.to_numeric(joined["score"], errors="coerce"))
        & np.isfinite(pd.to_numeric(joined["r_net"], errors="coerce"))
    ].copy()
    if joined.empty:
        return pd.DataFrame(columns=columns)
    percentile = joined["score"].rank(method="first", pct=True)
    joined["score_decile"] = np.ceil(percentile * 10.0).clip(1, 10).astype(int)
    rows: list[dict[str, object]] = []
    for decile, group in joined.groupby("score_decile", sort=True):
        rows.append(
            {
                "score_decile": int(decile),
                "rows": int(len(group)),
                "mean_score": float(group["score"].mean()),
                "mean_net_r": float(group["r_net"].mean()),
                "total_net_r": float(group["r_net"].sum()),
                "tp_rate": float(group["outcome"].eq("tp").mean()),
                "sl_rate": float(group["outcome"].eq("sl").mean()),
                "timeout_rate": float(group["outcome"].eq("timeout").mean()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _side_year_breakdown(
    model: PolicyReplay,
    baseline: PolicyReplay,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for policy_name, replay in (("model", model), ("first_decision_baseline", baseline)):
        if replay.trades.empty:
            continue
        work = replay.trades.copy()
        work["year"] = pd.to_datetime(work["entry_time"], utc=True).dt.year
        for (side, year), group in work.groupby(["side", "year"], sort=True):
            observed = group[group["path_observed"].astype(bool)]
            r_values = pd.to_numeric(observed["r_net"], errors="coerce").dropna()
            rows.append(
                {
                    "policy": policy_name,
                    "side": side,
                    "year": int(year),
                    "attempted_trades": int(len(group)),
                    "observed_trades": int(len(observed)),
                    "mean_net_r": float(r_values.mean()) if len(r_values) else np.nan,
                    "total_net_r": float(r_values.sum()) if len(r_values) else 0.0,
                }
            )
    return pd.DataFrame(
        rows,
        columns=(
            "policy",
            "side",
            "year",
            "attempted_trades",
            "observed_trades",
            "mean_net_r",
            "total_net_r",
        ),
    )


def _example_windows(
    manifest: pd.DataFrame,
    five: pd.DataFrame,
    selected: pd.DataFrame,
    config: EventWindowConfig,
) -> pd.DataFrame:
    columns = (
        "window_id",
        "side",
        "source_bar_time",
        "decision_time",
        "segment",
        "relative_step",
        "open",
        "high",
        "low",
        "close",
        "channel_lower",
        "channel_mid",
        "channel_upper",
        "selected",
    )
    if manifest.empty:
        return pd.DataFrame(columns=columns)
    chosen_ids = list(dict.fromkeys(selected.get("window_id", pd.Series(dtype=object))))
    for window_id in manifest["window_id"]:
        if window_id not in chosen_ids:
            chosen_ids.append(window_id)
        if len(chosen_ids) >= 6:
            break
    chosen = manifest[manifest["window_id"].isin(chosen_ids[:6])]
    selected_keys = (
        set(zip(selected["window_id"], selected["step"], strict=False))
        if {"window_id", "step"} <= set(selected.columns)
        else set()
    )
    cadence = pd.Timedelta(config.bar)
    rows: list[dict[str, object]] = []
    for record in chosen.itertuples(index=False):
        start = _utc(record.window_start)
        first_source = start - (config.pre_context_bars + 1) * cadence
        last_source = start + (config.active_bars - 2) * cadence
        view = five.loc[first_source:last_source]
        for source_time, bar in view.iterrows():
            decision_time = source_time + cadence
            active_step = int((decision_time - start) / cadence)
            active = 0 <= active_step < config.active_bars
            rows.append(
                {
                    "window_id": record.window_id,
                    "side": record.side,
                    "source_bar_time": source_time,
                    "decision_time": decision_time,
                    "segment": "active" if active else "pre",
                    "relative_step": active_step,
                    "open": bar.get("open", np.nan),
                    "high": bar.get("high", np.nan),
                    "low": bar.get("low", np.nan),
                    "close": bar.get("close", np.nan),
                    "channel_lower": bar.get("channel_lower", np.nan),
                    "channel_mid": bar.get("channel_mid", np.nan),
                    "channel_upper": bar.get("channel_upper", np.nan),
                    "selected": (record.window_id, active_step) in selected_keys,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    return float(numerator / denominator)


def _execute_dev_flow(
    config: EventWindowStudyConfig,
    loaded: LoadedInputs,
    store: _ArtifactStore,
    *,
    smoke: bool,
) -> dict[str, object]:
    hourly_context = build_hourly_channel_context(loaded.hourly, config.window)
    projected = project_hourly_channels(loaded.five_minute, hourly_context)
    projected["activity_ratio"] = causal_activity_ratio(projected, config.window)
    five_features = build_five_minute_feature_frame(projected)
    positioning_features = build_positioning_feature_frame(loaded.positioning)

    def make_manifest() -> pd.DataFrame:
        frame = build_event_window_manifest(
            projected,
            config.window,
            symbol=config.symbol,
        )
        if frame.empty:
            return frame
        start = _utc(config.dev_start)
        end = _utc(config.dev_end_exclusive)
        frame["window_start"] = pd.to_datetime(frame["window_start"], utc=True)
        return frame[
            frame["window_start"].ge(start) & frame["window_start"].lt(end)
        ].reset_index(drop=True)

    manifest = store.frame("window_manifest.parquet", make_manifest)
    actual_windows = int(len(manifest))
    window_audit = pd.DataFrame(
        [
            {
                "metric": "full_dev_manifest_windows",
                "actual": actual_windows,
                "expected": config.expected_full_dev_windows,
                "difference": actual_windows - config.expected_full_dev_windows,
                "exact_match": actual_windows == config.expected_full_dev_windows,
                "smoke_bypass": bool(smoke),
            }
        ]
    )
    store.write_csv("window_audit.csv", window_audit)
    if not smoke and actual_windows != config.expected_full_dev_windows:
        raise ProtocolMismatchError(
            "protocol mismatch: expected exactly "
            f"{config.expected_full_dev_windows:,} full-development windows, "
            f"observed {actual_windows:,}"
        )

    sequences = build_event_window_sequences(
        manifest,
        five_features,
        positioning_features,
        config.window,
    )
    profile, high_pairs = _feature_diagnostics(
        sequences,
        config.high_correlation_abs_spearman,
    )
    store.write_csv("feature_profile.csv", profile)
    store.write_csv("high_correlation_pairs.csv", high_pairs)

    primary_label_config = EventLabelConfig(rr_multiple=config.rr_primary)
    sensitivity_label_config = EventLabelConfig(rr_multiple=config.rr_sensitivity)
    labels_rr2 = store.frame(
        "labels_rr2.parquet",
        lambda: label_window_steps(
            sequences,
            five_features,
            loaded.minute,
            primary_label_config,
        ),
    )
    labels_rr3 = store.frame(
        "labels_rr3.parquet",
        lambda: label_window_steps(
            sequences,
            five_features,
            loaded.minute,
            sensitivity_label_config,
        ),
    )

    model_config = (
        replace(config.model, epochs=2, patience=min(config.model.patience, 2))
        if smoke
        else config.model
    )
    if store.valid("oof_scores.parquet") and store.valid("fold_audit.csv"):
        scores = pd.read_parquet(store.run_dir / "oof_scores.parquet")
        fold_audit = pd.read_csv(store.run_dir / "fold_audit.csv")
    else:
        oof = run_event_window_oof(sequences, labels_rr2, model_config)
        scores = oof.scores if not oof.scores.empty else _empty_scores()
        fold_audit = oof.fold_audit
        if fold_audit.empty and len(fold_audit.columns) == 0:
            fold_audit = pd.DataFrame(
                columns=(
                    "fold_id",
                    "train_windows",
                    "inner_train_windows",
                    "early_stop_windows",
                    "valid_windows",
                    "train_valid_episode_overlap",
                    "live_label_overlap",
                    "inner_early_live_label_overlap",
                    "train_end",
                    "valid_start",
                    "valid_end",
                )
            )
        store.write_frame("oof_scores.parquet", scores)
        store.write_csv("fold_audit.csv", fold_audit)

    evaluation_end = SMOKE_VALID_END if smoke else OOF_END
    thresholds = _threshold_grid(scores)
    frontier = sweep_event_thresholds(
        scores,
        labels_rr2,
        start=OOF_START,
        end=evaluation_end,
        thresholds=thresholds,
    )
    frontier = _mark_exploratory_frontier(frontier)
    store.write_csv("threshold_frontier.csv", frontier)
    primary = replay_first_crossing(
        scores,
        labels_rr2,
        threshold=0.0,
        start=OOF_START,
        end=evaluation_end,
    )
    baseline_scores = scores[["window_id", "step"]].copy()
    baseline_scores["score"] = 0.0
    baseline = replay_first_crossing(
        baseline_scores,
        labels_rr2,
        threshold=0.0,
        start=OOF_START,
        end=evaluation_end,
    )
    selected = _same_entries_rr3(primary.trades, labels_rr3)
    store.write_frame("selected_trades.parquet", selected)

    daily = _daily_report(primary, config.risk_pct)
    store.write_csv("daily_frequency.csv", daily)
    side_year = _side_year_breakdown(primary, baseline)
    store.write_csv("side_year_breakdown.csv", side_year)
    deciles = _score_deciles(scores, labels_rr2)
    store.write_csv("score_deciles.csv", deciles)

    paired = _paired_ledger(scores, primary, baseline)
    draws = 20 if smoke else config.bootstrap_draws
    bootstrap = episode_bootstrap(
        paired,
        draws=draws,
        seed=config.bootstrap_seed,
    )
    store.write_csv("episode_bootstrap.csv", bootstrap)
    examples = _example_windows(manifest, projected, primary.trades, config.window)
    store.write_frame("example_windows.parquet", examples)

    attempted = int(primary.summary["attempted_trades"])
    observed = int(primary.summary["observed_trades"])
    selected_observed = (
        primary.trades[primary.trades["path_observed"].astype(bool)]
        if attempted
        else primary.trades
    )
    observed_r = pd.to_numeric(
        selected_observed.get("r_net", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    positive = float(observed_r[observed_r > 0.0].sum())
    negative = float(-observed_r[observed_r < 0.0].sum())
    outcomes = selected_observed.get("outcome", pd.Series(dtype=object))
    delta_total = float(paired["model_net_r"].sum() - paired["baseline_net_r"].sum())
    delta_draws = pd.to_numeric(bootstrap["delta_total_net_r"], errors="coerce")
    ci_low = float(delta_draws.quantile(0.025))
    ci_high = float(delta_draws.quantile(0.975))
    full_days = (
        sum(int((right - left) / pd.Timedelta("1D")) for left, right in SMOKE_INTERVALS)
        if smoke
        else int(
            (_utc(config.dev_end_exclusive) - _utc(config.dev_start))
            / pd.Timedelta("1D")
        )
    )
    trades_per_day = float(primary.summary["trades_per_day"])
    mean_net_r = float(observed_r.mean()) if len(observed_r) else np.nan
    total_net_r = float(observed_r.sum()) if len(observed_r) else 0.0
    frequency_ok = bool(2.0 <= trades_per_day <= 5.0)
    rejection_reasons: list[str] = []
    if scores.empty:
        rejection_reasons.append("no_oof_scores")
    if not frequency_ok:
        rejection_reasons.append("threshold_zero_frequency_outside_2_to_5_per_day")
    if not (np.isfinite(mean_net_r) and mean_net_r > 0.0):
        rejection_reasons.append("threshold_zero_mean_net_r_not_positive")
    if total_net_r <= 0.0:
        rejection_reasons.append("threshold_zero_total_net_r_not_positive")
    if delta_total <= 0.0:
        rejection_reasons.append("paired_increment_over_baseline_not_positive")
    success = not rejection_reasons
    positioning_missing_index = (
        sequences.context_features.index("positioning_missing")
        if "positioning_missing" in sequences.context_features
        else None
    )
    oi_missing_index = (
        sequences.context_features.index("oi_missing")
        if "oi_missing" in sequences.context_features
        else None
    )
    valid_context = np.asarray(sequences.decision_valid, dtype=bool)
    if (
        positioning_missing_index is not None
        and oi_missing_index is not None
        and valid_context.any()
    ):
        positioning_missing = sequences.context[:, :, positioning_missing_index][valid_context]
        oi_missing = sequences.context[:, :, oi_missing_index][valid_context]
        oi_coverage = float(
            np.mean((positioning_missing == 0.0) & (oi_missing == 0.0))
        )
    else:
        oi_coverage = np.nan
    geometry_count = (
        int(labels_rr2["geometry_valid"].astype(bool).sum())
        if "geometry_valid" in labels_rr2
        else 0
    )
    target_count = (
        int(labels_rr2["model_target_valid"].astype(bool).sum())
        if "model_target_valid" in labels_rr2
        else 0
    )
    rr3_observed_mask = (
        selected["rr3_path_observed"].fillna(False).astype(bool)
        if "rr3_path_observed" in selected
        else pd.Series(False, index=selected.index)
    )
    rr3_values = pd.to_numeric(
        selected.loc[rr3_observed_mask, "rr3_r_net"]
        if "rr3_r_net" in selected
        else pd.Series(dtype=float),
        errors="coerce",
    ).dropna()
    rr3_outcomes = (
        selected.loc[rr3_observed_mask, "rr3_outcome"]
        if "rr3_outcome" in selected
        else pd.Series(dtype=object)
    )
    summary: dict[str, object] = {
        "run_hash": store.identity["run_hash"],
        "protocol_hash": store.identity["protocol_hash"],
        "stage": "dev",
        "smoke": bool(smoke),
        "forward_or_lockbox_loaded": False,
        "side_dataset": "pooled",
        "max_trades_per_window": 1,
        "manifest_windows": actual_windows,
        "windows_per_calendar_day": float(actual_windows / full_days),
        "window_calendar_days": full_days,
        "tensor_windows": int(len(sequences.metadata)),
        "tensor_decisions": int(valid_context.sum()),
        "oi_coverage_decisions": oi_coverage if np.isfinite(oi_coverage) else None,
        "rr2_label_rows": int(len(labels_rr2)),
        "geometry_valid_labels": geometry_count,
        "model_target_valid_labels": target_count,
        "censored_labels": int(geometry_count - target_count),
        "oof_score_rows": int(len(scores)),
        "oof_fold_rows": int(len(fold_audit)),
        "threshold_zero_is_registered_primary": True,
        "nonzero_threshold_frontier_is_exploratory": True,
        "primary_score_threshold": 0.0,
        "attempted_trades": attempted,
        "observed_trades": observed,
        "trades_per_calendar_day": trades_per_day,
        "zero_trade_days": int((daily["attempted_trades"] == 0).sum()),
        "mean_net_r": mean_net_r if np.isfinite(mean_net_r) else None,
        "total_net_r": total_net_r,
        "profit_factor": _safe_ratio(positive, negative),
        "tp_rate": float(outcomes.eq("tp").mean()) if observed else 0.0,
        "sl_rate": float(outcomes.eq("sl").mean()) if observed else 0.0,
        "timeout_rate": float(outcomes.eq("timeout").mean()) if observed else 0.0,
        "censored_rate": float((attempted - observed) / attempted) if attempted else 0.0,
        "rr3_same_entries": True,
        "rr3_attempted_trades": attempted,
        "rr3_observed_trades": int(rr3_observed_mask.sum()),
        "rr3_mean_net_r": float(rr3_values.mean()) if len(rr3_values) else None,
        "rr3_total_net_r": float(rr3_values.sum()) if len(rr3_values) else 0.0,
        "rr3_tp_rate": float(rr3_outcomes.eq("tp").mean()) if len(rr3_outcomes) else 0.0,
        "rr3_sl_rate": float(rr3_outcomes.eq("sl").mean()) if len(rr3_outcomes) else 0.0,
        "rr3_timeout_rate": (
            float(rr3_outcomes.eq("timeout").mean()) if len(rr3_outcomes) else 0.0
        ),
        "baseline_total_net_r": float(paired["baseline_net_r"].sum()),
        "baseline_mean_net_r_per_oof_window": (
            float(paired["baseline_net_r"].sum() / len(paired)) if len(paired) else 0.0
        ),
        "model_mean_net_r_per_oof_window": (
            float(paired["model_net_r"].sum() / len(paired)) if len(paired) else 0.0
        ),
        "paired_delta_total_net_r": delta_total,
        "paired_delta_mean_net_r": float(delta_total / len(paired)) if len(paired) else 0.0,
        "paired_delta_ci_low": ci_low,
        "paired_delta_ci_high": ci_high,
        "days_profit_ge_2pct": int(daily["profit_ge_2pct"].sum()),
        "days_profit_ge_3pct": int(daily["profit_ge_3pct"].sum()),
        "days_profit_ge_5pct": int(daily["profit_ge_5pct"].sum()),
        "frequency_admissible_2_to_5": frequency_ok,
        "preferred_frequency_3_to_5": bool(3.0 <= trades_per_day <= 5.0),
        "threshold_zero_success": bool(success),
        "rejection_reasons": rejection_reasons,
        "read_start": loaded.read_start,
        "read_end_exclusive": loaded.read_end_exclusive,
        "max_loaded_timestamp": loaded.max_loaded_timestamp,
        "resumed": False,
    }
    store.write_json("summary.json", summary)
    return summary


def run_event_window_study(
    *,
    stage: str = "dev",
    config: EventWindowStudyConfig = EventWindowStudyConfig(),
    data_root: Path = DEFAULT_DATA_ROOT,
    output_root: Path = RUN_ROOT,
    smoke: bool = False,
) -> RunResult:
    """Build or resume the complete development-only event-window study."""
    if stage != "dev":
        raise PermissionError(
            "event-window runner is development-only; tune, forward and Q2 remain sealed"
        )
    if _utc(config.dev_start) != DEV_START or _utc(config.dev_end_exclusive) != DEV_END:
        raise ValueError("the frozen development boundary is [2021-01-01, 2025-07-01)")

    data_root = Path(data_root)
    output_root = Path(output_root)
    protocol = protocol_dict(config, stage=stage)
    source_hash = source_version()
    paths = _input_paths(config, data_root)
    input_hash = _fingerprint_inputs(paths)
    protocol_hash = _sha_payload(protocol)
    run_hash = _sha_payload(
        {
            "protocol_hash": protocol_hash,
            "source_hash": source_hash,
            "input_hash": input_hash,
        }
    )[:20]
    mode = "smoke" if smoke else "full"
    run_dir = output_root / run_hash / mode
    store = _ArtifactStore(
        run_dir,
        run_hash=run_hash,
        protocol_hash=protocol_hash,
        input_hash=input_hash,
        source_hash=source_hash,
    )
    if store.state.get("status") == "complete" and store.all_required_valid():
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        summary["resumed"] = True
        if not smoke:
            _publish_latest_dev(
                output_root=output_root,
                run_dir=run_dir,
                run_hash=run_hash,
                protocol_hash=protocol_hash,
            )
        return RunResult(run_dir=run_dir, protocol=protocol, summary=summary)

    store.start()
    protocol_artifact = {
        **protocol,
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
        "mode": mode,
        "effective_epochs": 2 if smoke else config.model.epochs,
        "effective_bootstrap_draws": 20 if smoke else config.bootstrap_draws,
        "effective_read_intervals": (
            [[left, right] for left, right in SMOKE_INTERVALS]
            if smoke
            else [[DEV_START, DEV_END]]
        ),
    }
    store.write_json("protocol.json", protocol_artifact)
    try:
        loaded = load_inputs(config, data_root=data_root, smoke=smoke)
        if loaded.input_fingerprint != input_hash:
            raise RuntimeError("input metadata changed while the bounded run was starting")
        if loaded.max_loaded_timestamp >= loaded.read_end_exclusive:
            raise AssertionError("loaded data crossed the exclusive study boundary")
        protocol_artifact.update(
            {
                "read_start": loaded.read_start,
                "read_end_exclusive": loaded.read_end_exclusive,
                "max_loaded_timestamp": loaded.max_loaded_timestamp,
            }
        )
        store.write_json("protocol.json", protocol_artifact)
        summary = _execute_dev_flow(config, loaded, store, smoke=smoke)
        store.complete(summary)
    except BaseException as error:
        store.fail(error)
        raise

    if not smoke:
        _publish_latest_dev(
            output_root=output_root,
            run_dir=run_dir,
            run_hash=run_hash,
            protocol_hash=protocol_hash,
        )
    return RunResult(run_dir=run_dir, protocol=protocol, summary=summary)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=RUN_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_event_window_study(
        stage=args.stage,
        data_root=args.data_root,
        output_root=args.output_root,
        smoke=args.smoke,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATA_ROOT",
    "REQUIRED_ARTIFACTS",
    "RUN_ROOT",
    "EventWindowStudyConfig",
    "LoadedInputs",
    "ProtocolMismatchError",
    "RunResult",
    "episode_bootstrap",
    "load_inputs",
    "main",
    "parse_args",
    "protocol_dict",
    "run_event_window_study",
]
