"""Guarded and resumable runner for causal channel-window Notebook B.

The runner exposes development and, after an explicit protocol freeze, later
stages.  It never exposes a lockbox-unseal switch.  Every parquet read has a hard
exclusive upper bound before pandas receives the frame.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from evaluation.channel_window_policy import choose_tune_threshold, sweep_thresholds
from evaluation.channel_window_validation import VALID_BLOCKS, expanding_purged_folds
from experiments.channel_event_dataset import prepare_positioning_features
from experiments.channel_window_dataset import (
    DecisionFrames,
    WindowLabelConfig,
    build_decision_candidates,
    label_decision_candidates,
)
from experiments.channel_window_models import (
    OOFResult,
    logreg_continuation,
    run_catboost_oof,
    run_logreg_oof,
    score_frozen_model,
)
from features.channel_windows import build_channel_window_manifest
from features.linear_channels import (
    channel_confluence,
    channel_episode_id,
    compute_linear_regression_channels,
    label_channel_regime,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
RUN_DIR = CODE_ROOT / "experiments" / "cache" / "channel_window_ml"

STAGE_BOUNDS = {
    "dev": ("2021-01-01", "2025-07-01"),
    "tune": ("2025-07-01", "2025-10-01"),
    "forward": ("2025-10-01", "2026-04-01"),
    "lockbox": ("2026-04-01", "2026-07-01"),
}

_SOURCE_FILES = (
    CODE_ROOT / "features" / "channel_windows.py",
    CODE_ROOT / "evaluation" / "channel_backtest.py",
    CODE_ROOT / "evaluation" / "channel_window_validation.py",
    CODE_ROOT / "evaluation" / "channel_window_policy.py",
    CODE_ROOT / "experiments" / "channel_window_dataset.py",
    CODE_ROOT / "experiments" / "channel_window_models.py",
    Path(__file__),
)


@dataclass(frozen=True)
class WindowMLConfig:
    stage: str = "dev"
    symbol: str = "BTCUSDT"
    cadences: Sequence[str] = ("5min", "1min")
    architectures: Sequence[str] = ("pooled", "separate")
    channel_windows: Sequence[int] = (60, 90, 120)
    primary_window: int = 60
    min_r2: float = 0.20
    zone: float = 0.30
    max_window_minutes: int = 120
    require_confluence: bool = True
    min_trades_per_day: float = 0.0
    capacities: Sequence[int | None] = (3, 5, None)
    random_seed: int = 42
    inherited_trial_count: int = 160
    protocol_hash: str | None = None


@dataclass(frozen=True)
class LoadedInputs:
    minute: pd.DataFrame
    five_minute: pd.DataFrame
    fifteen_minute: pd.DataFrame
    hourly: pd.DataFrame
    positioning: pd.DataFrame
    max_loaded_timestamp: pd.Timestamp
    input_fingerprint: str


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    manifest: dict[str, object]
    summary: dict[str, object]


def _jsonable(value):
    if isinstance(value, (pd.Timestamp, Path)):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


def _canonical(payload: object) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_jsonable
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


def _protocol_payload(config: WindowMLConfig) -> dict[str, object]:
    payload = asdict(config)
    payload.pop("stage")
    payload.pop("protocol_hash")
    return payload


def protocol_identity(config: WindowMLConfig, source_hash: str) -> str:
    return _sha_payload({"protocol": _protocol_payload(config), "source": source_hash})


def run_identity(config: WindowMLConfig, input_hash: str, source_hash: str) -> str:
    """Hash full behaviour and inputs; smoke is intentionally not an argument."""
    return _sha_payload(
        {"config": asdict(config), "input": input_hash, "source": source_hash}
    )[:20]


def _input_paths(config: WindowMLConfig, data_root: Path) -> dict[str, Path]:
    symbol = config.symbol.lower()
    return {
        "minute": data_root / f"{symbol}_1m_2021_2026.parquet",
        "five_minute": data_root / f"{symbol}_5min_2021_2026.parquet",
        "fifteen_minute": data_root / f"{symbol}_15min_2021_2026.parquet",
        "hourly": data_root / f"{symbol}_1h_2021_2026.parquet",
        "positioning": data_root / f"{symbol}_positioning_15min_2021_2026.parquet",
    }


def _fingerprint_inputs(paths: dict[str, Path]) -> str:
    records = []
    for name, path in sorted(paths.items()):
        if not path.exists():
            raise FileNotFoundError(path)
        stat = path.stat()
        records.append(
            {"name": name, "path": str(path.resolve()), "size": stat.st_size,
             "mtime_ns": stat.st_mtime_ns}
        )
    return _sha_payload(records)


def _load_bounded_parquet(
    path: Path,
    *,
    start: pd.Timestamp | None,
    end: pd.Timestamp,
) -> pd.DataFrame:
    schema = pq.read_schema(path)
    index_field = "timestamp" if "timestamp" in schema.names else "__index_level_0__"
    filters: list[tuple[str, str, pd.Timestamp]] = [(index_field, "<", end)]
    if start is not None:
        filters.insert(0, (index_field, ">=", start))
    frame = pd.read_parquet(path, filters=filters)
    if start is not None:
        frame = frame[frame.index >= start]
    frame = frame[frame.index < end]
    if not frame.empty and frame.index.max() >= end:
        raise AssertionError(f"bounded read crossed {end}: {path.name}")
    return frame


def _max_loaded(frames: Sequence[pd.DataFrame]) -> pd.Timestamp:
    maxima = [pd.Timestamp(frame.index.max()) for frame in frames if not frame.empty]
    if not maxima:
        return pd.Timestamp.min.tz_localize("UTC")
    return max(maxima)


def load_inputs(
    config: WindowMLConfig,
    *,
    data_root: Path,
    decision_start: pd.Timestamp,
    stage_end: pd.Timestamp,
    smoke: bool,
) -> LoadedInputs:
    paths = _input_paths(config, data_root)
    fingerprint = _fingerprint_inputs(paths)
    short_start = None
    positioning_start = None
    if smoke:
        short_start = decision_start - pd.Timedelta(days=2)
        positioning_start = decision_start - pd.Timedelta(days=8)
    minute = _load_bounded_parquet(
        paths["minute"], start=short_start, end=stage_end
    )
    five = _load_bounded_parquet(
        paths["five_minute"], start=short_start, end=stage_end
    )
    fifteen = _load_bounded_parquet(
        paths["fifteen_minute"],
        start=(decision_start - pd.Timedelta(days=7) if smoke else None),
        end=stage_end,
    )
    hourly = _load_bounded_parquet(paths["hourly"], start=None, end=stage_end)
    positioning = _load_bounded_parquet(
        paths["positioning"], start=positioning_start, end=stage_end
    )
    maximum = _max_loaded((minute, five, fifteen, hourly, positioning))
    return LoadedInputs(
        minute=minute,
        five_minute=five,
        fifteen_minute=fifteen,
        hourly=hourly,
        positioning=positioning,
        max_loaded_timestamp=maximum,
        input_fingerprint=fingerprint,
    )


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable), encoding="utf-8"
    )
    temporary.replace(path)


class ArtifactStore:
    def __init__(self, run_dir: Path, *, run_hash: str, protocol_hash: str):
        self.run_dir = run_dir
        self.run_hash = run_hash
        self.protocol_hash = protocol_hash
        self.state_path = run_dir / "run_state.json"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.state = {
            "status": "running",
            "run_hash": run_hash,
            "protocol_hash": protocol_hash,
            "artifacts": {},
        }
        if self.state_path.exists():
            existing = json.loads(self.state_path.read_text(encoding="utf-8"))
            if (existing.get("run_hash") == run_hash
                    and existing.get("protocol_hash") == protocol_hash):
                self.state = existing
                self.state["status"] = "running"
        self._flush()

    def _flush(self) -> None:
        _atomic_json(self.state_path, self.state)

    def valid(self, name: str) -> bool:
        path = self.run_dir / name
        recorded = self.state.get("artifacts", {}).get(name)
        return bool(path.exists() and recorded and _sha_file(path) == recorded["sha256"])

    def _record(self, name: str) -> None:
        path = self.run_dir / name
        self.state.setdefault("artifacts", {})[name] = {
            "sha256": _sha_file(path), "size": path.stat().st_size,
        }
        self._flush()

    def frame(self, name: str, builder) -> pd.DataFrame:
        path = self.run_dir / name
        if self.valid(name):
            return pd.read_parquet(path)
        frame = builder()
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        self._record(name)
        return frame

    def write_frame(self, name: str, frame: pd.DataFrame) -> None:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        self._record(name)

    def write_csv(self, name: str, frame: pd.DataFrame) -> None:
        path = self.run_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
        self._record(name)

    def write_json(self, name: str, payload: object) -> None:
        _atomic_json(self.run_dir / name, payload)
        self._record(name)

    def complete(self, summary: dict[str, object]) -> None:
        self.state["status"] = "complete"
        self.state["summary"] = summary
        self._flush()

    def fail(self, error: BaseException) -> None:
        self.state["status"] = "failed"
        self.state["error"] = repr(error)
        self._flush()


def _channel_context(
    hourly: pd.DataFrame, config: WindowMLConfig
) -> pd.DataFrame:
    windows = tuple(dict.fromkeys(int(value) for value in config.channel_windows))
    if config.primary_window not in windows:
        raise ValueError("primary_window must be present in channel_windows")
    if 180 in windows:
        raise ValueError("channel window 180 is outside the frozen protocol")
    fitted: dict[int, pd.DataFrame] = {}
    regimes = {}
    for window in windows:
        channel = compute_linear_regression_channels(
            hourly,
            window=window,
            log_price=True,
            method="quantile",
            quantile=0.10,
        )
        channel["channel_slope"] = channel["channel_slope"] * 1e4
        channel["channel_regime"] = label_channel_regime(
            channel,
            min_slope=5.0,
            min_r2=config.min_r2,
            persist_bars=6,
        )
        fitted[window] = channel
        regimes[str(window)] = channel["channel_regime"]
    primary = fitted[config.primary_window].copy()
    agreement = channel_confluence(
        pd.DataFrame(regimes, index=hourly.index),
        primary=str(config.primary_window),
        min_agree=2,
    )
    primary = primary.join(agreement)
    primary["channel_episode_id"] = channel_episode_id(primary["channel_regime"])
    return primary


def _manifest_frame(
    fifteen: pd.DataFrame, hourly_context: pd.DataFrame
) -> pd.DataFrame:
    carry = [
        "channel_slope", "channel_mid", "channel_upper", "channel_lower",
        "channel_r2", "channel_width", "channel_regime", "channel_episode_id",
        "channel_confluence_count", "channel_confluence",
    ]
    available_hourly = hourly_context[carry].copy()
    available_hourly.index = available_hourly.index + pd.Timedelta("1h")
    availability = fifteen.index + pd.Timedelta("15min")
    aligned = available_hourly.reindex(availability, method="ffill")
    aligned.index = fifteen.index
    frame = fifteen.join(aligned)
    span = frame["channel_upper"] - frame["channel_lower"]
    frame["channel_pos"] = np.where(
        span > 0, (frame["close"] - frame["channel_lower"]) / span, np.nan
    )
    frame["availability_time"] = availability
    return frame


def _positioning_context(raw: pd.DataFrame) -> pd.DataFrame:
    if {"funding_rate", "sum_open_interest"}.issubset(raw.columns):
        return prepare_positioning_features(raw)
    required = {"funding_z", "oi_chg_4h"}
    if not required.issubset(raw.columns):
        raise ValueError("positioning input lacks raw or prepared funding/OI columns")
    out = raw.copy()
    if "positioning_stale" not in out:
        out["positioning_stale"] = False
    if "positioning_age_min" not in out:
        out["positioning_age_min"] = 0.0
    return out


def _fold_records(events: pd.DataFrame) -> list[dict[str, object]]:
    if events.empty:
        return []
    return [
        {
            "fold_id": fold.fold_id,
            "train_rows": int(len(fold.train)),
            "valid_rows": int(len(fold.valid)),
            "train_end": fold.train_end.isoformat(),
            "valid_start": fold.valid_start.isoformat(),
            "valid_end": fold.valid_end.isoformat(),
        }
        for fold in expanding_purged_folds(events)
    ]


def _empty_labelled(candidates: pd.DataFrame) -> pd.DataFrame:
    out = candidates.copy()
    for column in (
        "order_status", "filled", "entry_time", "exit_time", "active_end_time",
        "entry", "stop", "target", "outcome", "r_net", "label_start",
        "label_end", "label_net_positive",
    ):
        if column not in out:
            out[column] = pd.Series(dtype=float if column in {"r_net"} else object)
    return out


def _result_from_artifacts(
    predictions: pd.DataFrame,
    audit: pd.DataFrame,
    architecture: str,
    model_kind: str,
) -> OOFResult:
    return OOFResult(
        predictions=predictions,
        audit=audit,
        feature_columns=(),
        architecture=architecture,
        model_kind=model_kind,
    )


def _policy_score_results(
    logreg_result: OOFResult,
    catboost_result: OOFResult | None,
) -> tuple[OOFResult, ...]:
    """Keep the linear baseline in the economic contest when CatBoost runs."""
    if catboost_result is None:
        return (logreg_result,)
    return (logreg_result, catboost_result)


def select_dev_arm(
    policy: pd.DataFrame,
    *,
    primary_capacity: int = 3,
    min_trades_per_day: float = 0.0,
) -> dict[str, object] | None:
    """Freeze one supported arm on dev; the threshold itself remains tune-only."""
    required = {
        "cadence", "architecture", "model_kind", "capacity", "threshold",
        "threshold_quantile", "mean_r_net", "total_net_r", "filled_trades",
        "long_fills", "short_fills",
    }
    missing = sorted(required.difference(policy.columns))
    if missing:
        if policy.empty:
            return None
        raise ValueError(f"dev policy table missing selection columns: {missing}")
    if min_trades_per_day > 0 and "trades_per_day" not in policy:
        raise ValueError("frequency-constrained dev selection needs trades_per_day")
    frequency = (
        policy["trades_per_day"].ge(min_trades_per_day)
        if "trades_per_day" in policy
        else pd.Series(True, index=policy.index)
    )
    eligible = policy[
        policy["capacity"].eq(primary_capacity)
        & frequency
        & policy["mean_r_net"].gt(0)
        & policy["filled_trades"].ge(30)
        & policy["long_fills"].ge(10)
        & policy["short_fills"].ge(10)
    ]
    if eligible.empty:
        return None
    winner = eligible.sort_values(
        ["total_net_r", "mean_r_net", "threshold"],
        ascending=[False, False, True],
        kind="stable",
    ).iloc[0]
    return {
        "cadence": str(winner["cadence"]),
        "architecture": str(winner["architecture"]),
        "model_kind": str(winner["model_kind"]),
        "primary_capacity": int(primary_capacity),
        "dev_diagnostic_threshold": float(winner["threshold"]),
        "dev_diagnostic_quantile": float(winner["threshold_quantile"]),
        "dev_mean_r_net": float(winner["mean_r_net"]),
        "dev_total_net_r": float(winner["total_net_r"]),
        "dev_filled_trades": int(winner["filled_trades"]),
        "dev_long_fills": int(winner["long_fills"]),
        "dev_short_fills": int(winner["short_fills"]),
        "dev_trades_per_day": float(winner["trades_per_day"])
        if "trades_per_day" in winner else None,
    }


def _build_stage_datasets(
    config: WindowMLConfig,
    loaded: LoadedInputs,
    store: ArtifactStore,
    decision_start: pd.Timestamp,
    stage_end: pd.Timestamp,
    *,
    cadences: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, dict[str, int]]]:
    """Build one bounded stage with identical causal feature and label logic."""
    hourly_context = _channel_context(loaded.hourly, config)
    full_manifest_frame = _manifest_frame(loaded.fifteen_minute, hourly_context)

    def make_manifest() -> pd.DataFrame:
        manifest = build_channel_window_manifest(
            full_manifest_frame,
            symbol=config.symbol,
            zone=config.zone,
            max_duration=f"{config.max_window_minutes}min",
            require_confluence=config.require_confluence,
        )
        if manifest.empty:
            return manifest
        return manifest[
            (manifest["eligible_end_time"] > decision_start)
            & (manifest["window_start"] < stage_end)
        ].reset_index(drop=True)

    manifest = store.frame("window_manifest.parquet", make_manifest)
    daily = loaded.hourly["close"].resample("1D", label="left", closed="left").last().dropna()
    daily = daily.to_frame("close")
    frames = DecisionFrames(
        minute=loaded.minute,
        five_minute=loaded.five_minute,
        fifteen_minute=loaded.fifteen_minute,
        hourly=hourly_context,
        daily=daily,
        positioning=_positioning_context(loaded.positioning),
    )

    labelled_by_cadence: dict[str, pd.DataFrame] = {}
    counts: dict[str, dict[str, int]] = {}
    fold_manifest: dict[str, list[dict[str, object]]] = {}
    for cadence in tuple(cadences or config.cadences):
        suffix = "5m" if cadence == "5min" else "1m"

        def make_candidates(cadence=cadence) -> pd.DataFrame:
            candidates = build_decision_candidates(manifest, frames, cadence=cadence)
            if "decision_time" in candidates:
                candidates = candidates[
                    (candidates["decision_time"] >= decision_start)
                    & (candidates["decision_time"] < stage_end)
                ]
            return candidates.reset_index(drop=True)

        candidates = store.frame(f"decision_candidates_{suffix}.parquet", make_candidates)

        def make_labels(candidates=candidates) -> pd.DataFrame:
            if candidates.empty:
                return _empty_labelled(candidates)
            return label_decision_candidates(
                candidates, loaded.minute, config=WindowLabelConfig()
            )

        labelled = store.frame(f"labelled_{suffix}.parquet", make_labels)
        labelled_by_cadence[cadence] = labelled
        counts[cadence] = {
            "candidates": int(len(candidates)),
            "labelled": int(len(labelled)),
            "filled": int(labelled["filled"].astype(bool).sum()) if "filled" in labelled else 0,
        }
        fold_manifest[cadence] = _fold_records(labelled)
    store.write_json("fold_manifest.json", fold_manifest)
    return manifest, labelled_by_cadence, counts


def _execute_dev_pipeline(
    config: WindowMLConfig,
    loaded: LoadedInputs,
    store: ArtifactStore,
    decision_start: pd.Timestamp,
    stage_end: pd.Timestamp,
) -> dict[str, object]:
    manifest, labelled_by_cadence, counts = _build_stage_datasets(
        config, loaded, store, decision_start, stage_end
    )

    continuation: dict[str, bool] = {}
    policy_tables: list[pd.DataFrame] = []
    for cadence, events in labelled_by_cadence.items():
        if events.empty:
            continue
        suffix = "5m" if cadence == "5min" else "1m"
        for architecture in config.architectures:
            log_name = f"oof_logreg_{suffix}_{architecture}.parquet"
            audit_name = f"audit_logreg_{suffix}_{architecture}.csv"
            if store.valid(log_name) and store.valid(audit_name):
                predictions = pd.read_parquet(store.run_dir / log_name)
                audit = pd.read_csv(store.run_dir / audit_name)
                log_result = _result_from_artifacts(
                    predictions, audit, architecture, "logreg"
                )
            else:
                log_result = run_logreg_oof(events, architecture=architecture)
                store.write_frame(log_name, log_result.predictions)
                store.write_csv(audit_name, log_result.audit)
            arm = f"{cadence}_{architecture}"
            allowed = logreg_continuation(log_result)
            continuation[arm] = allowed
            cat_result: OOFResult | None = None
            if allowed:
                cat_name = f"oof_catboost_{suffix}_{architecture}.parquet"
                cat_audit_name = f"audit_catboost_{suffix}_{architecture}.csv"
                if store.valid(cat_name) and store.valid(cat_audit_name):
                    cat_predictions = pd.read_parquet(store.run_dir / cat_name)
                    cat_audit = pd.read_csv(store.run_dir / cat_audit_name)
                    cat_result = _result_from_artifacts(
                        cat_predictions, cat_audit, architecture, "catboost_regressor"
                    )
                else:
                    cat_result = run_catboost_oof(events, architecture=architecture)
                    store.write_frame(cat_name, cat_result.predictions)
                    store.write_csv(cat_audit_name, cat_result.audit)
            for score_result in _policy_score_results(log_result, cat_result):
                if not score_result.predictions.empty:
                    table = sweep_thresholds(
                        score_result.predictions,
                        capacities=tuple(config.capacities),
                        inherited_trial_count=config.inherited_trial_count,
                        evaluation_days=sum(
                            float((end - start) / pd.Timedelta("1D"))
                            for start, end in VALID_BLOCKS
                        ),
                    )
                    table.insert(0, "cadence", cadence)
                    table.insert(1, "architecture", architecture)
                    table.insert(2, "model_kind", score_result.model_kind)
                    policy_tables.append(table)

    store.write_json("continuation.json", continuation)
    policy = pd.concat(policy_tables, ignore_index=True) if policy_tables else pd.DataFrame()
    store.write_csv("trial_ledger.csv", policy)
    store.write_csv("thresholds.csv", policy)
    store.write_csv("capacity_results.csv", policy)
    selected_arm = select_dev_arm(
        policy, min_trades_per_day=config.min_trades_per_day
    )
    store.write_json(
        "dev_freeze.json",
        {
            "promotion": selected_arm is not None,
            "selected_arm": selected_arm,
            "selection_scope": "dev OOF only",
            "threshold_status": "diagnostic only; tune chooses the frozen threshold",
            "minimum_trades_per_day": config.min_trades_per_day,
        },
    )
    summary = {
        "stage": config.stage,
        "windows": int(len(manifest)),
        "counts": counts,
        "continuation": continuation,
        "catboost_arms": int(sum(continuation.values())),
        "policy_rows": int(len(policy)),
        "selected_arm": selected_arm,
        "minimum_trades_per_day": config.min_trades_per_day,
    }
    return summary


def _frozen_dev_run(out_root: Path, protocol_hash: str) -> tuple[Path, dict[str, object]]:
    frozen_path = out_root / "frozen_protocols" / f"{protocol_hash}.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen.get("protocol_hash") != protocol_hash:
        raise PermissionError("frozen protocol manifest hash mismatch")
    dev_run_hash = frozen.get("dev_run_hash")
    if not dev_run_hash:
        raise PermissionError("frozen protocol manifest has no dev run")
    dev_dir = out_root / str(dev_run_hash) / "full"
    required = (
        dev_dir / "run_state.json",
        dev_dir / "protocol_manifest.json",
        dev_dir / "dev_freeze.json",
    )
    if not all(path.exists() for path in required):
        raise PermissionError("frozen dev run is incomplete")
    state = json.loads(required[0].read_text(encoding="utf-8"))
    manifest = json.loads(required[1].read_text(encoding="utf-8"))
    if state.get("status") != "complete" or manifest.get("stage") != "dev":
        raise PermissionError("frozen dev run is not a completed dev stage")
    if manifest.get("protocol_hash") != protocol_hash:
        raise PermissionError("dev artifact protocol does not match tune protocol")
    return dev_dir, json.loads(required[2].read_text(encoding="utf-8"))


def _execute_tune_pipeline(
    config: WindowMLConfig,
    loaded: LoadedInputs,
    store: ArtifactStore,
    decision_start: pd.Timestamp,
    stage_end: pd.Timestamp,
    out_root: Path,
) -> dict[str, object]:
    """Score one dev-frozen arm and choose only its global tune threshold."""
    if not config.protocol_hash:
        raise PermissionError("tune requires a frozen dev protocol hash")
    dev_dir, dev_freeze = _frozen_dev_run(out_root, config.protocol_hash)
    selected = dev_freeze.get("selected_arm")
    if not dev_freeze.get("promotion") or not isinstance(selected, dict):
        raise PermissionError("dev produced no supported arm for tune")
    cadence = str(selected["cadence"])
    architecture = str(selected["architecture"])
    model_kind = str(selected["model_kind"])
    if cadence not in config.cadences or architecture not in config.architectures:
        raise PermissionError("frozen arm is outside the supplied protocol config")

    manifest, labelled_by_cadence, counts = _build_stage_datasets(
        config,
        loaded,
        store,
        decision_start,
        stage_end,
        cadences=(cadence,),
    )
    suffix = "5m" if cadence == "5min" else "1m"
    dev_labels_path = dev_dir / f"labelled_{suffix}.parquet"
    if not dev_labels_path.exists():
        raise PermissionError(f"frozen dev labels missing: {dev_labels_path.name}")
    dev_events = pd.read_parquet(dev_labels_path)
    tune_events = labelled_by_cadence[cadence]
    if dev_events.empty or tune_events.empty:
        raise ValueError("frozen scoring requires non-empty dev and tune labels")
    dev_times = pd.to_datetime(dev_events["decision_time"], utc=True)
    tune_times = pd.to_datetime(tune_events["decision_time"], utc=True)
    if dev_times.max() >= decision_start or tune_times.min() < decision_start:
        raise AssertionError("dev/tune decision boundary overlap")
    if tune_times.max() >= stage_end:
        raise AssertionError("tune decisions crossed the exclusive stage end")

    scored = score_frozen_model(
        dev_events,
        tune_events,
        architecture=architecture,
        model_kind=model_kind,
    )
    score_name = f"frozen_scores_{suffix}_{architecture}.parquet"
    store.write_frame(score_name, scored)
    policy = sweep_thresholds(
        scored,
        capacities=tuple(config.capacities),
        inherited_trial_count=config.inherited_trial_count,
        evaluation_days=float((stage_end - decision_start) / pd.Timedelta("1D")),
    )
    policy.insert(0, "cadence", cadence)
    policy.insert(1, "architecture", architecture)
    policy.insert(2, "model_kind", model_kind)
    store.write_csv("trial_ledger.csv", policy)
    store.write_csv("thresholds.csv", policy)
    store.write_csv("capacity_results.csv", policy)

    primary_capacity = int(selected["primary_capacity"])
    threshold = choose_tune_threshold(
        policy,
        capacity=primary_capacity,
        min_trades_per_day=config.min_trades_per_day,
    )
    promoted = bool(np.isfinite(threshold))
    diagnostic: dict[str, object] | None = None
    if promoted:
        chosen = policy[
            policy["capacity"].eq(primary_capacity)
            & np.isclose(policy["threshold"].astype(float), threshold)
        ].sort_values(
            ["total_net_r", "mean_r_net", "threshold_quantile"],
            ascending=[False, False, True],
            kind="stable",
        ).iloc[0]
        diagnostic = {
            "threshold_quantile": float(chosen["threshold_quantile"]),
            "mean_r_net": float(chosen["mean_r_net"]),
            "total_net_r": float(chosen["total_net_r"]),
            "filled_trades": int(chosen["filled_trades"]),
            "long_fills": int(chosen["long_fills"]),
            "short_fills": int(chosen["short_fills"]),
            "bootstrap_low": float(chosen["bootstrap_low"]),
            "bootstrap_high": float(chosen["bootstrap_high"]),
            "bonferroni_low": float(chosen["bonferroni_low"]),
            "bonferroni_high": float(chosen["bonferroni_high"]),
        }
    threshold_freeze = {
        "promotion": promoted,
        "protocol_hash": config.protocol_hash,
        "dev_run_hash": dev_dir.parent.name,
        "selected_arm": selected,
        "primary_capacity": primary_capacity,
        "threshold": float(threshold) if promoted else None,
        "tune_diagnostic": diagnostic,
        "selection_scope": "2025-07-01 through 2025-10-01 only",
    }
    store.write_json("threshold_freeze.json", threshold_freeze)
    return {
        "stage": config.stage,
        "windows": int(len(manifest)),
        "counts": counts,
        "selected_arm": selected,
        "threshold_promoted": promoted,
        "threshold": float(threshold) if promoted else None,
        "policy_rows": int(len(policy)),
    }


def _guard_stage(config: WindowMLConfig, out_root: Path) -> None:
    if config.stage not in STAGE_BOUNDS:
        raise ValueError(f"unknown stage: {config.stage}")
    if config.stage == "lockbox":
        raise PermissionError(
            "2026 Q2 remains sealed; this runner intentionally has no unseal path"
        )
    if config.stage == "tune" and not config.protocol_hash:
        raise PermissionError("tune requires a frozen dev protocol hash")
    if config.stage == "forward" and not config.protocol_hash:
        raise PermissionError("forward requires a frozen protocol hash")
    if config.stage in {"tune", "forward"}:
        frozen = out_root / "frozen_protocols" / f"{config.protocol_hash}.json"
        if not frozen.exists():
            raise PermissionError(f"frozen protocol manifest not found: {frozen}")


def run_experiment(
    config: WindowMLConfig,
    *,
    out_root: Path = RUN_DIR,
    data_root: Path = DEFAULT_DATA_ROOT,
    smoke: bool = False,
    resume: bool = True,
    loaded_inputs: LoadedInputs | None = None,
) -> RunResult:
    """Run or safely resume one config-hashed channel-window experiment."""
    out_root = Path(out_root)
    data_root = Path(data_root)
    _guard_stage(config, out_root)
    stage_start, stage_end_text = STAGE_BOUNDS[config.stage]
    decision_start = pd.Timestamp(stage_start, tz="UTC")
    stage_end = pd.Timestamp(stage_end_text, tz="UTC")
    if smoke:
        if config.stage != "dev":
            raise ValueError("smoke mode is defined only inside dev")
        decision_start = pd.Timestamp("2021-08-01", tz="UTC")
        stage_end = pd.Timestamp("2022-02-01", tz="UTC")

    source_hash = source_version()
    if loaded_inputs is None:
        paths = _input_paths(config, data_root)
        input_hash = _fingerprint_inputs(paths)
    else:
        input_hash = loaded_inputs.input_fingerprint
    run_hash = run_identity(config, input_hash, source_hash)
    protocol_hash = protocol_identity(config, source_hash)
    if config.protocol_hash and config.protocol_hash != protocol_hash:
        raise PermissionError("provided frozen protocol hash does not match this source/config")
    run_dir = out_root / run_hash / ("smoke" if smoke else "full")
    state_path = run_dir / "run_state.json"
    manifest_path = run_dir / "protocol_manifest.json"
    summary_path = run_dir / "summary.json"
    if resume and state_path.exists() and manifest_path.exists() and summary_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if state.get("status") == "complete" and state.get("run_hash") == run_hash:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            return RunResult(run_dir, manifest, {**summary, "resumed": True})

    loaded = loaded_inputs or load_inputs(
        config,
        data_root=data_root,
        decision_start=decision_start,
        stage_end=stage_end,
        smoke=smoke,
    )
    maximum = pd.Timestamp(loaded.max_loaded_timestamp)
    if maximum.tzinfo is None:
        maximum = maximum.tz_localize("UTC")
    else:
        maximum = maximum.tz_convert("UTC")
    if maximum >= stage_end:
        raise AssertionError(
            f"stage boundary violated: loaded {maximum} at or after {stage_end}"
        )
    manifest = {
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "input_hash": input_hash,
        "source_version": source_hash,
        "stage": config.stage,
        "smoke": smoke,
        "decision_start": decision_start.isoformat(),
        "exclusive_stage_end": stage_end.isoformat(),
        "max_loaded_timestamp": maximum.isoformat(),
        "config": asdict(config),
    }
    store = ArtifactStore(run_dir, run_hash=run_hash, protocol_hash=protocol_hash)
    store.write_json("protocol_manifest.json", manifest)
    try:
        if config.stage == "dev":
            summary = _execute_dev_pipeline(
                config, loaded, store, decision_start, stage_end
            )
        elif config.stage == "tune":
            summary = _execute_tune_pipeline(
                config, loaded, store, decision_start, stage_end, out_root
            )
        else:
            raise NotImplementedError(
                "forward scoring remains blocked until a tune threshold is frozen"
            )
        summary = {**summary, "run_hash": run_hash, "protocol_hash": protocol_hash}
        store.write_json("summary.json", summary)
        store.complete(summary)
        if config.stage == "dev" and not smoke:
            freeze = out_root / "frozen_protocols" / f"{protocol_hash}.json"
            if not freeze.exists():
                _atomic_json(
                    freeze,
                    {"protocol_hash": protocol_hash, "dev_run_hash": run_hash,
                     "summary": summary},
                )
        return RunResult(run_dir, manifest, summary)
    except BaseException as error:
        store.fail(error)
        raise


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_BOUNDS), default="dev")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cadence", action="append", choices=("5min", "1min"))
    parser.add_argument("--architecture", action="append", choices=("pooled", "separate"))
    parser.add_argument("--protocol-hash")
    parser.add_argument("--soft-confluence", action="store_true")
    parser.add_argument("--min-trades-per-day", type=float, default=0.0)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-root", type=Path, default=RUN_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = WindowMLConfig(
        stage=args.stage,
        cadences=tuple(args.cadence or WindowMLConfig.cadences),
        architectures=tuple(args.architecture or WindowMLConfig.architectures),
        protocol_hash=args.protocol_hash,
        require_confluence=not args.soft_confluence,
        min_trades_per_day=args.min_trades_per_day,
    )
    result = run_experiment(
        config,
        out_root=args.out_root,
        data_root=args.data_root,
        smoke=args.smoke,
        resume=args.resume,
    )
    print(json.dumps({"run_dir": str(result.run_dir), **result.summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
