"""Resumable development-only runner for Notebook K tail-model comparison."""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from experiments.channel_rebuild_contract import recomputed_handoffs, selected_run_hash

from evaluation.event_window_tail_policy import (
    episode_pair_bootstrap,
    matched_count_diagnostic,
    paired_window_ledger,
    replay_positive_ev,
    replay_same_entries,
)
from experiments.event_window_dataset import build_event_window_sequences
from experiments.event_window_tail_dataset import (
    TailDecisionDataset,
    build_tail_decision_dataset,
)
from experiments.event_window_tail_neural import TailNeuralConfig
from experiments.event_window_tail_oof import (
    TailOOFConfig,
    TailOOFResult,
    _fit_hash,
    _half_open_uniqueness,
    _outer_folds,
    _validated_decisions,
    assert_identical_outer_score_keys,
    chronological_inner_partitions,
    run_tail_fold,
)
from experiments.event_window_tail_tabular import XGBoostTailConfig
from experiments.run_event_window_tcn import REQUIRED_ARTIFACTS as FROZEN_J_ARTIFACTS
from experiments.run_event_window_tcn import (
    EventWindowStudyConfig,
    load_inputs,
)
from features.event_window_inputs import (
    build_five_minute_feature_frame,
    build_positioning_feature_frame,
)
from features.event_windows import (
    build_hourly_channel_context,
    causal_activity_ratio,
    project_hourly_channels,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = CODE_ROOT / "data"
FROZEN_J_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_tcn"
RUN_ROOT = CODE_ROOT / "experiments" / "cache" / "event_window_tail_models"
FROZEN_J_RUN_HASH = "2c6e19d5eae7ba9ffaaa"
DEV_END = pd.Timestamp("2025-07-01", tz="UTC")

REQUIRED_ARTIFACTS = (
    "protocol.json",
    "frozen_j_contract.json",
    "feature_profile.csv",
    "high_correlation_pairs.csv",
    "oof_logreg.parquet",
    "oof_xgboost.parquet",
    "oof_tcn.parquet",
    "oof_gru.parquet",
    "fold_audit.csv",
    "calibration_audit.csv",
    "natural_policy_results.csv",
    "matched_rate_results.csv",
    "matched_count_results.csv",
    "selected_trades_rr2.parquet",
    "selected_trades_rr3.parquet",
    "daily_frequency.csv",
    "side_year_breakdown.csv",
    "score_deciles.csv",
    "paired_comparisons.csv",
    "episode_bootstrap.csv",
    "example_window.parquet",
    "summary.json",
)

_SOURCE_FILES = (
    CODE_ROOT / "experiments" / "event_window_tail_dataset.py",
    CODE_ROOT / "experiments" / "event_window_tail_calibration.py",
    CODE_ROOT / "experiments" / "event_window_tail_tabular.py",
    CODE_ROOT / "experiments" / "event_window_tail_neural.py",
    CODE_ROOT / "experiments" / "event_window_tail_oof.py",
    CODE_ROOT / "evaluation" / "event_window_tail_policy.py",
    Path(__file__),
)


class ProtocolMismatchError(RuntimeError):
    """Raised when the frozen Notebook J handoff changes."""


@dataclass(frozen=True)
class FrozenJArtifacts:
    run_hash: str
    run_dir: Path
    manifest_sha256: str
    input_hash: str
    protocol: dict[str, object]
    summary: dict[str, object]
    manifest: pd.DataFrame
    labels_rr2: pd.DataFrame
    labels_rr3: pd.DataFrame
    direct_scores: pd.DataFrame
    selected_trades: pd.DataFrame


@dataclass(frozen=True)
class TailStudyConfig:
    frozen_j_run_hash: str = FROZEN_J_RUN_HASH
    dev_start: str = "2021-01-01"
    dev_end_exclusive: str = "2025-07-01"
    models: tuple[str, ...] = ("logreg", "xgboost", "tcn", "gru")
    reference_attempts: int = 1_448
    reference_calendar_days: int = 1_277
    bootstrap_draws: int = 2_000
    bootstrap_seed: int = 42


@dataclass(frozen=True)
class TailRunResult:
    run_dir: Path
    protocol: dict[str, object]
    summary: dict[str, object]
    recomputed: tuple[tuple[str, str], ...]


def _jsonable(value: object) -> object:
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


def source_version() -> str:
    digest = hashlib.sha256()
    for path in _SOURCE_FILES:
        if not path.is_file():
            continue
        digest.update(str(path.relative_to(CODE_ROOT)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def protocol_dict(
    config: TailStudyConfig = TailStudyConfig(),
    *,
    stage: str = "dev",
) -> dict[str, object]:
    return {
        "stage": stage,
        "development_start": config.dev_start,
        "development_end_exclusive": config.dev_end_exclusive,
        "models": list(config.models),
        "frozen_j_run_hash": config.frozen_j_run_hash,
        "selection_rr": 2.0,
        "sensitivity_rr": 3.0,
        "primary_policy": "first calibrated EV >= 0",
        "max_trades_per_window": 1,
        "cross_window_capacity": "unlimited",
        "reference_attempts": config.reference_attempts,
        "reference_calendar_days": config.reference_calendar_days,
        "reference_trades_per_day": (
            config.reference_attempts / config.reference_calendar_days
        ),
        "bootstrap_draws": config.bootstrap_draws,
        "bootstrap_seed": config.bootstrap_seed,
        "forward_or_lockbox_loaded": False,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_jsonable),
        encoding="utf-8",
    )
    temporary.replace(path)


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
        if self.state_path.is_file():
            existing = _read_json(self.state_path)
            if all(existing.get(key) == value for key, value in self.identity.items()):
                self.state = existing
        self._flush()

    def _flush(self) -> None:
        _atomic_json(self.state_path, self.state)

    def start(self) -> None:
        self.state["status"] = "running"
        self.state.pop("error", None)
        self._flush()

    def valid(self, name: str) -> bool:
        path = self.run_dir / name
        artifacts = self.state.get("artifacts", {})
        record = artifacts.get(name) if isinstance(artifacts, dict) else None
        return bool(
            path.is_file()
            and isinstance(record, dict)
            and record.get("sha256") == _sha256(path)
            and int(record.get("size", -1)) == path.stat().st_size
        )

    def all_required_valid(self) -> bool:
        return all(self.valid(name) for name in REQUIRED_ARTIFACTS)

    def _record(self, name: str) -> None:
        path = self.run_dir / name
        artifacts = self.state.setdefault("artifacts", {})
        if not isinstance(artifacts, dict):
            raise RuntimeError("artifact registry is invalid")
        artifacts[name] = {"sha256": _sha256(path), "size": path.stat().st_size}
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

    def complete(self, summary: dict[str, object]) -> None:
        self.state["status"] = "complete"
        self.state["summary"] = summary
        self._flush()

    def fail(self, error: BaseException) -> None:
        self.state["status"] = "failed"
        self.state["error"] = repr(error)
        self._flush()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolMismatchError(f"invalid frozen JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise ProtocolMismatchError(f"frozen JSON artifact is not an object: {path.name}")
    return value


def _decision_keys(frame: pd.DataFrame, *, name: str) -> set[tuple[object, int]]:
    required = {"window_id", "step"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ProtocolMismatchError(f"{name} missing decision keys: {missing}")
    if frame.duplicated(["window_id", "step"]).any():
        raise ProtocolMismatchError(f"{name} contains duplicate decision keys")
    return set(
        zip(
            frame["window_id"],
            pd.to_numeric(frame["step"], errors="raise").astype(int),
            strict=True,
        )
    )


def load_frozen_j_artifacts(
    run_root: Path = FROZEN_J_ROOT,
) -> FrozenJArtifacts:
    """Validate every frozen Notebook J artifact before exposing its frames."""
    root = Path(run_root)
    pointer = _read_json(root / "latest_dev.json")
    expected_hash = selected_run_hash(pointer, FROZEN_J_RUN_HASH)
    if pointer.get("run_hash") != expected_hash:
        raise ProtocolMismatchError("frozen Notebook J run hash changed")
    expected_relative = f"{expected_hash}/full"
    if pointer.get("relative_path") != expected_relative:
        raise ProtocolMismatchError("frozen Notebook J pointer must select the full run")
    run_dir = (root / expected_relative).resolve()
    if not run_dir.is_relative_to(root.resolve()):
        raise ProtocolMismatchError("frozen Notebook J path escaped its run root")
    state = _read_json(run_dir / "run_state.json")
    if state.get("status") != "complete" or state.get("run_hash") != expected_hash:
        raise ProtocolMismatchError("frozen Notebook J run is not complete")
    if state.get("protocol_hash") != pointer.get("protocol_hash"):
        raise ProtocolMismatchError("frozen Notebook J protocol identity changed")
    records = state.get("artifacts")
    if not isinstance(records, dict):
        raise ProtocolMismatchError("frozen Notebook J artifact registry is missing")
    for name in FROZEN_J_ARTIFACTS:
        path = run_dir / name
        record = records.get(name)
        if not path.is_file() or not isinstance(record, dict):
            raise ProtocolMismatchError(f"frozen artifact is missing: {name}")
        if record.get("sha256") != _sha256(path):
            raise ProtocolMismatchError(f"frozen artifact hash changed: {name}")
        if int(record.get("size", -1)) != path.stat().st_size:
            raise ProtocolMismatchError(f"frozen artifact size changed: {name}")

    protocol = _read_json(run_dir / "protocol.json")
    summary = _read_json(run_dir / "summary.json")
    if protocol.get("forward_or_lockbox_loaded") is not False:
        raise ProtocolMismatchError("frozen Notebook J opened a later period")
    if summary.get("forward_or_lockbox_loaded") is not False:
        raise ProtocolMismatchError("frozen Notebook J summary opened a later period")
    maximum = pd.Timestamp(protocol.get("max_loaded_timestamp"))
    maximum = maximum.tz_localize("UTC") if maximum.tzinfo is None else maximum.tz_convert("UTC")
    if maximum >= DEV_END:
        raise ProtocolMismatchError("frozen Notebook J crossed the development boundary")

    manifest = pd.read_parquet(run_dir / "window_manifest.parquet")
    if len(manifest) != 14_510 or "window_id" not in manifest:
        raise ProtocolMismatchError("frozen Notebook J manifest must contain 14,510 windows")
    if manifest["window_id"].duplicated().any():
        raise ProtocolMismatchError("frozen Notebook J manifest window IDs are duplicated")
    labels_rr2 = pd.read_parquet(run_dir / "labels_rr2.parquet")
    labels_rr3 = pd.read_parquet(run_dir / "labels_rr3.parquet")
    if _decision_keys(labels_rr2, name="RR2") != _decision_keys(labels_rr3, name="RR3"):
        raise ProtocolMismatchError("frozen RR2 and RR3 decision keys differ")
    direct_scores = pd.read_parquet(run_dir / "oof_scores.parquet")
    selected = pd.read_parquet(run_dir / "selected_trades.parquet")
    if int(summary.get("manifest_windows", -1)) != 14_510:
        raise ProtocolMismatchError("frozen Notebook J summary manifest count changed")
    expected_attempts = len(selected) if recomputed_handoffs() else 1_448
    if int(summary.get("attempted_trades", -1)) != expected_attempts:
        raise ProtocolMismatchError("frozen Notebook J reference attempt count changed")
    return FrozenJArtifacts(
        run_hash=expected_hash,
        run_dir=run_dir,
        manifest_sha256=_sha256(run_dir / "window_manifest.parquet"),
        input_hash=str(state.get("input_hash", "")),
        protocol=protocol,
        summary=summary,
        manifest=manifest,
        labels_rr2=labels_rr2,
        labels_rr3=labels_rr3,
        direct_scores=direct_scores,
        selected_trades=selected,
    )


def _subset_labels_for_sequences(
    sequences,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    valid_keys = {
        (sequences.metadata.iloc[row]["window_id"], int(step))
        for row in range(len(sequences.metadata))
        for step in np.flatnonzero(sequences.decision_valid[row])
    }
    key_series = list(zip(labels["window_id"], labels["step"], strict=True))
    subset = labels.loc[[key in valid_keys for key in key_series]].copy()
    actual = set(subset[["window_id", "step"]].itertuples(index=False, name=None))
    if actual != valid_keys:
        raise ProtocolMismatchError("frozen labels do not cover the rebuilt decision keys")
    return subset.reset_index(drop=True)


def _validate_loaded_input_fingerprint(
    loaded,
    frozen: FrozenJArtifacts,
    *,
    required: bool,
) -> None:
    if required and loaded.input_fingerprint != frozen.input_hash:
        raise ProtocolMismatchError("current bounded inputs do not match frozen Notebook J")


def _build_tail_dataset(
    frozen: FrozenJArtifacts,
    *,
    data_root: Path,
    smoke: bool,
    validate_legacy_input_fingerprint: bool = True,
) -> tuple[TailDecisionDataset, pd.DataFrame, object]:
    base_config = EventWindowStudyConfig()
    loaded = load_inputs(base_config, data_root=Path(data_root), smoke=smoke)
    _validate_loaded_input_fingerprint(
        loaded,
        frozen,
        required=validate_legacy_input_fingerprint,
    )
    if loaded.max_loaded_timestamp >= loaded.read_end_exclusive:
        raise AssertionError("bounded inputs crossed their exclusive end")
    hourly_context = build_hourly_channel_context(loaded.hourly, base_config.window)
    projected = project_hourly_channels(loaded.five_minute, hourly_context)
    projected["activity_ratio"] = causal_activity_ratio(projected, base_config.window)
    five_features = build_five_minute_feature_frame(projected)
    positioning_features = build_positioning_feature_frame(loaded.positioning)
    sequences = build_event_window_sequences(
        frozen.manifest,
        five_features,
        positioning_features,
        base_config.window,
    )
    if not smoke and len(sequences.metadata) != 14_510:
        raise ProtocolMismatchError("rebuilt tensor windows differ from frozen Notebook J")
    labels_rr2 = _subset_labels_for_sequences(sequences, frozen.labels_rr2)
    labels_rr3 = _subset_labels_for_sequences(sequences, frozen.labels_rr3)
    dataset = build_tail_decision_dataset(
        sequences,
        labels_rr2,
        rr=2.0,
        cost_bps=10.0,
    )
    return dataset, labels_rr3, loaded


def _feature_diagnostics(
    dataset: TailDecisionDataset,
    *,
    correlation_threshold: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values = np.asarray(dataset.tabular, dtype=np.float32)
    profile_rows: list[dict[str, object]] = []
    for column, name in enumerate(dataset.tabular_features):
        observed = values[:, column]
        finite = observed[np.isfinite(observed)]
        profile_rows.append(
            {
                "feature": name,
                "rows": len(observed),
                "finite_fraction": float(np.isfinite(observed).mean()),
                "missing_fraction": float(1.0 - np.isfinite(observed).mean()),
                "mean": float(np.mean(finite)) if finite.size else np.nan,
                "std": float(np.std(finite)) if finite.size else np.nan,
            }
        )
    profile = pd.DataFrame(profile_rows)
    sample_size = min(20_000, len(values))
    if sample_size < 2:
        return profile, pd.DataFrame(
            columns=["feature_a", "feature_b", "spearman", "sample_rows"]
        )
    rng = np.random.default_rng(42)
    positions = (
        np.arange(len(values))
        if sample_size == len(values)
        else np.sort(rng.choice(len(values), size=sample_size, replace=False))
    )
    sample = pd.DataFrame(values[positions], columns=dataset.tabular_features)
    correlation = sample.corr(method="spearman", min_periods=max(20, sample_size // 20))
    pairs: list[dict[str, object]] = []
    for left in range(len(correlation.columns)):
        for right in range(left + 1, len(correlation.columns)):
            value = float(correlation.iat[left, right])
            if np.isfinite(value) and abs(value) >= correlation_threshold:
                pairs.append(
                    {
                        "feature_a": correlation.columns[left],
                        "feature_b": correlation.columns[right],
                        "spearman": value,
                        "sample_rows": sample_size,
                    }
                )
    high_pairs = pd.DataFrame(
        pairs,
        columns=["feature_a", "feature_b", "spearman", "sample_rows"],
    )
    if not high_pairs.empty:
        high_pairs = high_pairs.reindex(
            high_pairs["spearman"].abs().sort_values(ascending=False).index
        ).reset_index(drop=True)
    return profile, high_pairs


def _checkpoint_paths(run_dir: Path, model: str, fold_id: str) -> tuple[Path, Path]:
    directory = run_dir / "checkpoints" / model
    return directory / f"{fold_id}.parquet", directory / f"{fold_id}_audit.json"


def _checkpoint_identity(
    store: _ArtifactStore,
    frozen: FrozenJArtifacts,
    *,
    model: str,
    fold_id: str,
) -> dict[str, object]:
    return {
        **store.identity,
        "frozen_j_run_hash": frozen.run_hash,
        "frozen_manifest_sha256": frozen.manifest_sha256,
        "model": model,
        "fold_id": fold_id,
    }


def _load_fold_checkpoint(
    store: _ArtifactStore,
    frozen: FrozenJArtifacts,
    *,
    model: str,
    fold_id: str,
) -> TailOOFResult | None:
    score_path, audit_path = _checkpoint_paths(store.run_dir, model, fold_id)
    if not score_path.is_file() or not audit_path.is_file():
        return None
    try:
        audit = _read_json(audit_path)
    except ProtocolMismatchError:
        return None
    expected = _checkpoint_identity(
        store, frozen, model=model, fold_id=fold_id
    )
    if any(audit.get(key) != value for key, value in expected.items()):
        return None
    if audit.get("score_sha256") != _sha256(score_path):
        return None
    try:
        scores = pd.read_parquet(score_path)
        fold_audit = pd.DataFrame([audit["fold_audit"]])
        calibration_audit = pd.DataFrame([audit["calibration_audit"]])
    except (OSError, KeyError, ValueError):
        return None
    required = {
        "model", "fold_id", "window_id", "step", "p_sl", "p_tp",
        "p_timeout", "ev_score", "matched_rate_threshold",
    }
    if not required.issubset(scores.columns):
        return None
    if scores.duplicated(["window_id", "step"]).any():
        return None
    return TailOOFResult(model, scores, fold_audit, calibration_audit)


def _write_fold_checkpoint(
    store: _ArtifactStore,
    frozen: FrozenJArtifacts,
    result: TailOOFResult,
) -> None:
    fold_id = str(result.fold_audit.iloc[0]["fold_id"])
    score_path, audit_path = _checkpoint_paths(
        store.run_dir, result.model_name, fold_id
    )
    score_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = score_path.with_suffix(score_path.suffix + ".tmp")
    result.scores.to_parquet(temporary, index=False)
    temporary.replace(score_path)
    payload = {
        **_checkpoint_identity(
            store,
            frozen,
            model=result.model_name,
            fold_id=fold_id,
        ),
        "score_sha256": _sha256(score_path),
        "score_rows": len(result.scores),
        "fold_audit": result.fold_audit.iloc[0].to_dict(),
        "calibration_audit": result.calibration_audit.iloc[0].to_dict(),
    }
    _atomic_json(audit_path, payload)


def _migrate_compatible_fold_checkpoint(
    store: _ArtifactStore,
    frozen: FrozenJArtifacts,
    *,
    model: str,
    fold,
    dataset: TailDecisionDataset,
    config: TailOOFConfig,
) -> TailOOFResult | None:
    """Reuse a prior fold only when its fit-data hash is exactly unchanged."""
    decisions = _validated_decisions(dataset)
    partitions = chronological_inner_partitions(decisions, fold.train, config.fold)
    weights = _half_open_uniqueness(decisions, partitions.fit)
    expected_fit_hash = _fit_hash(
        model, dataset, decisions, partitions.fit, weights, config
    )
    run_root = store.run_dir.parent.parent
    candidates = sorted(
        run_root.glob(f"*/full/checkpoints/{model}/{fold.fold_id}_audit.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    expected_keys = decisions.iloc[fold.valid][["window_id", "step"]].copy()
    for audit_path in candidates:
        if audit_path.parents[2].resolve() == store.run_dir.resolve():
            continue
        try:
            audit = _read_json(audit_path)
        except ProtocolMismatchError:
            continue
        if (
            audit.get("protocol_hash") != store.identity["protocol_hash"]
            or audit.get("input_hash") != store.identity["input_hash"]
            or audit.get("frozen_j_run_hash") != frozen.run_hash
            or audit.get("frozen_manifest_sha256") != frozen.manifest_sha256
            or audit.get("model") != model
            or audit.get("fold_id") != fold.fold_id
        ):
            continue
        fold_record = audit.get("fold_audit")
        calibration_record = audit.get("calibration_audit")
        if not isinstance(fold_record, dict) or not isinstance(calibration_record, dict):
            continue
        if fold_record.get("fit_model_hash") != expected_fit_hash:
            continue
        score_path = audit_path.with_name(f"{fold.fold_id}.parquet")
        if not score_path.is_file() or audit.get("score_sha256") != _sha256(score_path):
            continue
        try:
            old_scores = pd.read_parquet(score_path)
            scores = expected_keys.merge(
                old_scores,
                on=["window_id", "step"],
                how="left",
                validate="one_to_one",
            )
        except (OSError, ValueError):
            continue
        if scores["ev_score"].isna().any() or len(scores) != len(expected_keys):
            continue
        fold_record = dict(fold_record)
        fold_record["outer_rows"] = len(scores)
        result = TailOOFResult(
            model,
            scores,
            pd.DataFrame([fold_record]),
            pd.DataFrame([calibration_record]),
        )
        _write_fold_checkpoint(store, frozen, result)
        return result
    return None


def _run_all_oof(
    dataset: TailDecisionDataset,
    frozen: FrozenJArtifacts,
    store: _ArtifactStore,
    *,
    smoke: bool,
) -> tuple[list[TailOOFResult], tuple[tuple[str, str], ...]]:
    oof_config = TailOOFConfig()
    if smoke:
        oof_config = replace(
            oof_config,
            xgboost=replace(oof_config.xgboost, n_estimators=20),
            neural=replace(oof_config.neural, epochs=2, patience=2),
        )
    folds = [
        fold
        for fold in _outer_folds(dataset.decisions)
        if len(fold.train) and len(fold.valid)
    ]
    if not folds:
        raise RuntimeError("no non-empty OOF folds are available")
    model_results: list[TailOOFResult] = []
    recomputed: list[tuple[str, str]] = []
    for model in ("logreg", "xgboost", "tcn", "gru"):
        fold_results: list[TailOOFResult] = []
        for fold in folds:
            result = _load_fold_checkpoint(
                store, frozen, model=model, fold_id=fold.fold_id
            )
            if result is None:
                result = _migrate_compatible_fold_checkpoint(
                    store,
                    frozen,
                    model=model,
                    fold=fold,
                    dataset=dataset,
                    config=oof_config,
                )
                if result is None:
                    result = run_tail_fold(model, fold, dataset, oof_config)
                    _write_fold_checkpoint(store, frozen, result)
                    recomputed.append((model, fold.fold_id))
            fold_results.append(result)
        combined = TailOOFResult(
            model_name=model,
            scores=pd.concat([value.scores for value in fold_results], ignore_index=True),
            fold_audit=pd.concat(
                [value.fold_audit for value in fold_results], ignore_index=True
            ),
            calibration_audit=pd.concat(
                [value.calibration_audit for value in fold_results], ignore_index=True
            ),
        )
        combined_scores = combined.scores.sort_values(
            ["decision_time", "window_id", "step"], kind="stable"
        ).reset_index(drop=True)
        combined = TailOOFResult(
            model, combined_scores, combined.fold_audit, combined.calibration_audit
        )
        store.write_frame(f"oof_{model}.parquet", combined.scores)
        model_results.append(combined)
    assert_identical_outer_score_keys(model_results)
    return model_results, tuple(recomputed)


def _evaluation_bounds(
    scores: pd.DataFrame,
    *,
    smoke: bool,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if not smoke:
        return pd.Timestamp("2022-01-01", tz="UTC"), DEV_END
    times = pd.to_datetime(scores["decision_time"], utc=True, errors="raise")
    start = times.min().floor("D")
    end = times.max().floor("D") + pd.Timedelta("1D")
    return start, end


def _natural_row(name: str, replay) -> dict[str, object]:
    observed = replay.trades.loc[
        replay.trades.get("path_observed", pd.Series(False, index=replay.trades.index))
        .fillna(False)
        .astype(bool)
    ]
    outcomes = observed.get("outcome", pd.Series(dtype=object))
    return {
        "model": name,
        "attempted_trades": replay.attempted_trades,
        "observed_trades": replay.observed_trades,
        "trades_per_day": replay.trades_per_day,
        "zero_trade_days": int((replay.daily["attempted_trades"] == 0).sum()),
        "mean_net_r": replay.mean_net_r,
        "total_net_r": replay.total_net_r,
        "tp_rate": float(outcomes.eq("tp").mean()) if len(observed) else 0.0,
        "sl_rate": float(outcomes.eq("sl").mean()) if len(observed) else 0.0,
        "timeout_rate": float(outcomes.eq("timeout").mean()) if len(observed) else 0.0,
        "frequency_noninferior": replay.frequency_noninferior,
    }


def _same_key_scores(
    reference: pd.DataFrame,
    keys: pd.DataFrame,
) -> pd.DataFrame:
    columns = [
        column
        for column in (
            "window_id", "channel_episode_id", "side", "step", "decision_time",
            "fold_id", "score",
        )
        if column in reference.columns
    ]
    selected = keys[["window_id", "step"]].merge(
        reference[columns],
        on=["window_id", "step"],
        how="left",
        validate="one_to_one",
    )
    if selected["score"].isna().any():
        raise ProtocolMismatchError("frozen direct-TCN scores do not match the K OOF keys")
    return selected


def _score_deciles(
    results: list[TailOOFResult],
    labels: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    observed = labels.loc[labels["model_target_valid"].astype(bool)].copy()
    for result in results:
        merged = result.scores[["window_id", "step", "ev_score"]].merge(
            observed[["window_id", "step", "r_net"]],
            on=["window_id", "step"],
            how="inner",
            validate="one_to_one",
        )
        if merged.empty:
            continue
        ranked = merged["ev_score"].rank(method="first")
        merged["score_decile"] = pd.qcut(ranked, 10, labels=False, duplicates="drop") + 1
        grouped = merged.groupby("score_decile", observed=True).agg(
            decisions=("r_net", "size"),
            mean_ev=("ev_score", "mean"),
            mean_net_r=("r_net", "mean"),
            total_net_r=("r_net", "sum"),
        ).reset_index()
        grouped.insert(0, "model", result.model_name)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _example_window(
    results: list[TailOOFResult],
    selected: pd.DataFrame,
) -> pd.DataFrame:
    selected_new = selected.loc[selected["model"].isin({"logreg", "xgboost", "tcn", "gru"})]
    if not selected_new.empty:
        window_id = selected_new.iloc[0]["window_id"]
    else:
        window_id = results[0].scores.iloc[0]["window_id"]
    base = results[0].scores.loc[
        results[0].scores["window_id"].eq(window_id),
        ["window_id", "channel_episode_id", "side", "step", "decision_time"],
    ].copy()
    for result in results:
        model_scores = result.scores.loc[
            result.scores["window_id"].eq(window_id),
            ["window_id", "step", "ev_score"],
        ].rename(columns={"ev_score": f"ev_{result.model_name}"})
        base = base.merge(model_scores, on=["window_id", "step"], how="left")
        chosen = set(
            selected_new.loc[
                selected_new["model"].eq(result.model_name)
                & selected_new["window_id"].eq(window_id),
                "step",
            ].astype(int)
        )
        base[f"selected_{result.model_name}"] = base["step"].isin(chosen)
    return base.sort_values("step", kind="stable").reset_index(drop=True)


def _paired_comparison(
    candidate_name: str,
    candidate_replay,
    comparator_name: str,
    comparator_replay,
    *,
    draws: int,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    ledger = paired_window_ledger(
        {"candidate": candidate_replay, "comparator": comparator_replay}
    )
    bootstrap = episode_pair_bootstrap(ledger, draws=draws, seed=seed)
    row = {
        "candidate": candidate_name,
        "comparator": comparator_name,
        "paired_delta_total_net_r": bootstrap.point_delta_total_net_r,
        "paired_delta_ci_low": bootstrap.ci_low,
        "paired_delta_ci_high": bootstrap.ci_high,
    }
    draw_frame = pd.DataFrame(
        {
            "candidate": candidate_name,
            "comparator": comparator_name,
            "draw": np.arange(len(bootstrap.draws)),
            "delta_total_net_r": bootstrap.draws.to_numpy(),
        }
    )
    return row, draw_frame


def _policy_artifacts(
    results: list[TailOOFResult],
    frozen: FrozenJArtifacts,
    dataset: TailDecisionDataset,
    labels_rr3: pd.DataFrame,
    config: TailStudyConfig,
    *,
    smoke: bool,
) -> dict[str, object]:
    common_keys = results[0].scores[["window_id", "step"]]
    start, end = _evaluation_bounds(results[0].scores, smoke=smoke)
    labels_rr2 = dataset.decisions.copy()
    labels_rr3_work = labels_rr3.copy()
    new_replays: dict[str, object] = {}
    natural_rows: list[dict[str, object]] = []
    selected_rr2: list[pd.DataFrame] = []
    selected_rr3: list[pd.DataFrame] = []
    matched_rate_rows: list[dict[str, object]] = []
    matched_count_rows: list[dict[str, object]] = []

    frozen_direct_scores = _same_key_scores(frozen.direct_scores, common_keys)
    frozen_first_scores = frozen_direct_scores.copy()
    frozen_first_scores["score"] = 0.0
    reference_replays = {
        "first_entry": replay_positive_ev(
            frozen_first_scores, labels_rr2, start, end, model_name="first_entry"
        ),
        "frozen_j_tcn": replay_positive_ev(
            frozen_direct_scores, labels_rr2, start, end, model_name="frozen_j_tcn"
        ),
    }
    for name, replay in reference_replays.items():
        natural_rows.append(_natural_row(name, replay))

    for result in results:
        replay = replay_positive_ev(
            result.scores,
            labels_rr2,
            start,
            end,
            model_name=result.model_name,
        )
        new_replays[result.model_name] = replay
        natural_rows.append(_natural_row(result.model_name, replay))
        trades = replay.trades.copy()
        trades.attrs = {}
        trades.insert(0, "model", result.model_name)
        selected_rr2.append(trades)
        rr3 = replay_same_entries(replay.trades, labels_rr3_work)
        rr3.insert(0, "model", result.model_name)
        selected_rr3.append(rr3)

        causal_scores = result.scores.copy()
        causal_scores["ev_score"] = (
            causal_scores["ev_score"] - causal_scores["matched_rate_threshold"]
        )
        causal = replay_positive_ev(
            causal_scores,
            labels_rr2,
            start,
            end,
            model_name=result.model_name,
        )
        causal_row = _natural_row(result.model_name, causal)
        causal_row.update(
            {
                "threshold_mode": "fold_local_calibration",
                "exploratory": True,
                "can_select": False,
            }
        )
        matched_rate_rows.append(causal_row)

        matched = matched_count_diagnostic(
            result.scores,
            labels_rr2,
            config.reference_attempts,
            calendar_start=start,
            calendar_end=end,
            model_name=result.model_name,
        )
        matched_count_rows.append(
            {
                **_natural_row(result.model_name, matched),
                "threshold": matched.threshold,
                "target_attempts": matched.target_attempts,
                "exploratory": matched.exploratory,
                "can_select": matched.can_select,
            }
        )

    natural = pd.DataFrame(natural_rows)
    reference_names = ["first_entry", "frozen_j_tcn"]
    eligible_references = natural.loc[
        natural["model"].isin(reference_names)
        & natural["frequency_noninferior"].astype(bool)
    ]
    if eligible_references.empty:
        strongest_name = "frozen_j_tcn"
    else:
        strongest_name = str(
            eligible_references.sort_values("total_net_r", ascending=False).iloc[0]["model"]
        )
    strongest = reference_replays[strongest_name]
    paired_rows: list[dict[str, object]] = []
    bootstrap_frames: list[pd.DataFrame] = []
    draws = 20 if smoke else config.bootstrap_draws
    for model, replay in new_replays.items():
        row, draw_frame = _paired_comparison(
            model,
            replay,
            strongest_name,
            strongest,
            draws=draws,
            seed=config.bootstrap_seed,
        )
        paired_rows.append(row)
        bootstrap_frames.append(draw_frame)
    paired = pd.DataFrame(paired_rows)
    natural = natural.merge(
        paired.rename(columns={"candidate": "model"}),
        on="model",
        how="left",
    )
    new_rows = natural["model"].isin(new_replays)
    natural.loc[new_rows, "passes_registered_rule"] = (
        natural.loc[new_rows, "frequency_noninferior"].astype(bool)
        & natural.loc[new_rows, "mean_net_r"].gt(0.0)
        & natural.loc[new_rows, "total_net_r"].gt(0.0)
        & natural.loc[new_rows, "paired_delta_total_net_r"].gt(0.0)
        & natural.loc[new_rows, "paired_delta_ci_low"].gt(0.0)
    )
    natural["rejection_reasons"] = ""
    for index in natural.index[new_rows]:
        reasons: list[str] = []
        row = natural.loc[index]
        if not bool(row["frequency_noninferior"]):
            reasons.append("frequency")
        if not float(row["mean_net_r"]) > 0.0:
            reasons.append("mean_net_r")
        if not float(row["total_net_r"]) > 0.0:
            reasons.append("total_net_r")
        if not float(row["paired_delta_total_net_r"]) > 0.0:
            reasons.append("paired_delta")
        if not float(row["paired_delta_ci_low"]) > 0.0:
            reasons.append("paired_ci")
        natural.at[index, "rejection_reasons"] = ";".join(reasons)

    simplicity = ("logreg", "xgboost", "gru", "tcn")
    passing = [
        model
        for model in simplicity
        if bool(
            natural.loc[natural["model"].eq(model), "passes_registered_rule"].fillna(False).iloc[0]
        )
    ]
    chosen = passing[0] if passing else None
    temporal_justified = chosen in {"gru", "tcn"}
    selected2 = pd.concat(selected_rr2, ignore_index=True) if selected_rr2 else pd.DataFrame()
    selected3 = pd.concat(selected_rr3, ignore_index=True) if selected_rr3 else pd.DataFrame()

    daily = pd.concat(
        [replay.daily.assign(model=name).reset_index(names="date") for name, replay in new_replays.items()],
        ignore_index=True,
    )
    if not selected2.empty:
        side_year = selected2.copy()
        side_year["year"] = pd.to_datetime(side_year["entry_time"], utc=True).dt.year
        side_year = side_year.loc[side_year["path_observed"].astype(bool)].groupby(
            ["model", "side", "year"], observed=True
        ).agg(
            trades=("r_net", "size"),
            mean_net_r=("r_net", "mean"),
            total_net_r=("r_net", "sum"),
        ).reset_index()
    else:
        side_year = pd.DataFrame()
    return {
        "natural": natural,
        "matched_rate": pd.DataFrame(matched_rate_rows),
        "matched_count": pd.DataFrame(matched_count_rows),
        "selected_rr2": selected2,
        "selected_rr3": selected3,
        "daily": daily,
        "side_year": side_year,
        "score_deciles": _score_deciles(results, labels_rr2),
        "paired": paired,
        "bootstrap": pd.concat(bootstrap_frames, ignore_index=True),
        "example": _example_window(results, selected2),
        "strongest_comparator": strongest_name,
        "chosen_model": chosen,
        "temporal_complexity_justified": temporal_justified,
        "rr3_same_entries": True,
        "evaluation_start": start,
        "evaluation_end_exclusive": end,
    }


def run_event_window_tail_study(
    *,
    stage: str = "dev",
    config: TailStudyConfig = TailStudyConfig(),
    data_root: Path = DEFAULT_DATA_ROOT,
    run_root: Path = RUN_ROOT,
    frozen_j_root: Path = FROZEN_J_ROOT,
    smoke: bool = False,
) -> TailRunResult:
    """Build or resume the Notebook K development contest."""
    if stage != "dev":
        raise PermissionError(
            "event-window tail runner is development-only; later periods remain sealed"
        )
    if config.frozen_j_run_hash != FROZEN_J_RUN_HASH:
        raise ValueError("the frozen Notebook J run hash cannot be changed")
    if config.models != ("logreg", "xgboost", "tcn", "gru"):
        raise ValueError("the registered four-model contest cannot be changed")
    if config.dev_start != "2021-01-01" or config.dev_end_exclusive != "2025-07-01":
        raise ValueError("the development boundary is frozen")

    frozen = load_frozen_j_artifacts(Path(frozen_j_root))
    protocol = protocol_dict(config, stage=stage)
    source_hash = source_version()
    input_hash = frozen.input_hash
    protocol_hash = _sha_payload(protocol)
    run_hash = _sha_payload(
        {
            "protocol_hash": protocol_hash,
            "source_hash": source_hash,
            "input_hash": input_hash,
            "frozen_manifest_sha256": frozen.manifest_sha256,
        }
    )[:20]
    mode = "smoke" if smoke else "full"
    output_root = Path(run_root)
    run_dir = output_root / run_hash / mode
    store = _ArtifactStore(
        run_dir,
        run_hash=run_hash,
        protocol_hash=protocol_hash,
        input_hash=input_hash,
        source_hash=source_hash,
    )
    if store.state.get("status") == "complete" and store.all_required_valid():
        summary = _read_json(run_dir / "summary.json")
        returned = {**summary, "resumed": True}
        if not smoke:
            _atomic_json(
                output_root / "latest_dev.json",
                {
                    "run_hash": run_hash,
                    "relative_path": run_dir.relative_to(output_root).as_posix(),
                    "protocol_hash": protocol_hash,
                },
            )
        return TailRunResult(run_dir, protocol, returned, ())

    store.start()
    protocol_artifact: dict[str, object] = {
        **protocol,
        "run_hash": run_hash,
        "protocol_hash": protocol_hash,
        "source_hash": source_hash,
        "input_hash": input_hash,
        "mode": mode,
        "smoke": bool(smoke),
        "dev_start": config.dev_start,
        "dev_end_exclusive": config.dev_end_exclusive,
        "selection_rule": "first calibrated EV >= 0",
        "effective_xgboost_estimators": 20 if smoke else 300,
        "effective_neural_epochs": 2 if smoke else 40,
        "effective_bootstrap_draws": 20 if smoke else config.bootstrap_draws,
    }
    store.write_json("protocol.json", protocol_artifact)
    recomputed: tuple[tuple[str, str], ...] = ()
    try:
        dataset, labels_rr3, loaded = _build_tail_dataset(
            frozen,
            data_root=Path(data_root),
            smoke=smoke,
        )
        protocol_artifact.update(
            {
                "read_start": loaded.read_start,
                "read_end_exclusive": loaded.read_end_exclusive,
                "max_loaded_timestamp": loaded.max_loaded_timestamp,
            }
        )
        store.write_json("protocol.json", protocol_artifact)
        store.write_json(
            "frozen_j_contract.json",
            {
                "run_hash": frozen.run_hash,
                "run_dir_relative": frozen.run_dir.relative_to(Path(frozen_j_root).resolve()).as_posix(),
                "manifest_sha256": frozen.manifest_sha256,
                "manifest_windows": len(frozen.manifest),
                "rr2_rows": len(frozen.labels_rr2),
                "rr3_rows": len(frozen.labels_rr3),
                "reference_attempts": int(frozen.summary["attempted_trades"]),
                "reference_total_net_r": frozen.summary.get("total_net_r"),
                "forward_or_lockbox_loaded": False,
            },
        )
        profile, correlations = _feature_diagnostics(dataset)
        store.write_csv("feature_profile.csv", profile)
        store.write_csv("high_correlation_pairs.csv", correlations)
        results, recomputed = _run_all_oof(
            dataset,
            frozen,
            store,
            smoke=smoke,
        )
        fold_audit = pd.concat(
            [result.fold_audit for result in results], ignore_index=True
        )
        calibration_audit = pd.concat(
            [result.calibration_audit for result in results], ignore_index=True
        )
        overlap_columns = [name for name in fold_audit if "overlap" in name]
        if overlap_columns and not fold_audit[overlap_columns].eq(0).all().all():
            raise AssertionError("Notebook K fold leakage audit failed")
        store.write_csv("fold_audit.csv", fold_audit)
        store.write_csv("calibration_audit.csv", calibration_audit)

        policy = _policy_artifacts(
            results,
            frozen,
            dataset,
            labels_rr3,
            config,
            smoke=smoke,
        )
        store.write_csv("natural_policy_results.csv", policy["natural"])
        store.write_csv("matched_rate_results.csv", policy["matched_rate"])
        store.write_csv("matched_count_results.csv", policy["matched_count"])
        store.write_frame("selected_trades_rr2.parquet", policy["selected_rr2"])
        store.write_frame("selected_trades_rr3.parquet", policy["selected_rr3"])
        store.write_csv("daily_frequency.csv", policy["daily"])
        store.write_csv("side_year_breakdown.csv", policy["side_year"])
        store.write_csv("score_deciles.csv", policy["score_deciles"])
        store.write_csv("paired_comparisons.csv", policy["paired"])
        store.write_csv("episode_bootstrap.csv", policy["bootstrap"])
        store.write_frame("example_window.parquet", policy["example"])

        natural = policy["natural"]
        model_rows = natural.loc[natural["model"].isin(config.models)].copy()
        rr3_selected = policy["selected_rr3"]
        rr3_summary: dict[str, dict[str, object]] = {}
        for model in config.models:
            selected = rr3_selected.loc[rr3_selected["model"].eq(model)]
            observed = selected.loc[selected["path_observed"].fillna(False).astype(bool)]
            values = pd.to_numeric(observed.get("r_net", pd.Series(dtype=float)), errors="coerce").dropna()
            rr3_summary[model] = {
                "attempted_trades": len(selected),
                "observed_trades": len(observed),
                "mean_net_r": float(values.mean()) if len(values) else None,
                "total_net_r": float(values.sum()) if len(values) else 0.0,
            }
        summary: dict[str, object] = {
            "run_hash": run_hash,
            "protocol_hash": protocol_hash,
            "stage": "dev",
            "smoke": bool(smoke),
            "models": list(config.models),
            "forward_or_lockbox_loaded": False,
            "frozen_j_run_hash": frozen.run_hash,
            "manifest_windows": len(frozen.manifest),
            "tensor_windows": len(dataset.sequences.metadata),
            "tensor_decisions": len(dataset.decisions),
            "oof_score_rows_per_model": {
                result.model_name: len(result.scores) for result in results
            },
            "all_model_outer_keys_identical": True,
            "fold_overlap_max": int(fold_audit[overlap_columns].max().max()) if overlap_columns else 0,
            "reference_attempts": config.reference_attempts,
            "reference_calendar_days": config.reference_calendar_days,
            "strongest_comparator": policy["strongest_comparator"],
            "chosen_model": policy["chosen_model"],
            "temporal_complexity_justified": policy["temporal_complexity_justified"],
            "rr3_same_entries": policy["rr3_same_entries"],
            "rr3_models": rr3_summary,
            "model_results": model_rows.set_index("model").to_dict("index"),
            "evaluation_start": policy["evaluation_start"],
            "evaluation_end_exclusive": policy["evaluation_end_exclusive"],
            "read_start": loaded.read_start,
            "read_end_exclusive": loaded.read_end_exclusive,
            "max_loaded_timestamp": loaded.max_loaded_timestamp,
            "resumed": False,
        }
        store.write_json("summary.json", summary)
        if not store.all_required_valid():
            missing = [name for name in REQUIRED_ARTIFACTS if not store.valid(name)]
            raise RuntimeError(f"Notebook K artifact validation failed: {missing}")
        store.complete(summary)
    except BaseException as error:
        store.fail(error)
        raise

    if not smoke:
        _atomic_json(
            output_root / "latest_dev.json",
            {
                "run_hash": run_hash,
                "relative_path": run_dir.relative_to(output_root).as_posix(),
                "protocol_hash": protocol_hash,
            },
        )
    return TailRunResult(run_dir, protocol, summary, recomputed)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="dev")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    result = run_event_window_tail_study(
        stage=arguments.stage,
        smoke=arguments.smoke,
    )
    print(result.run_dir)
    print(json.dumps(result.summary, indent=2, sort_keys=True, default=_jsonable))
    return 0


__all__ = [
    "FROZEN_J_ARTIFACTS",
    "FROZEN_J_RUN_HASH",
    "REQUIRED_ARTIFACTS",
    "FrozenJArtifacts",
    "ProtocolMismatchError",
    "TailRunResult",
    "TailStudyConfig",
    "load_frozen_j_artifacts",
    "main",
    "parse_args",
    "protocol_dict",
    "run_event_window_tail_study",
]


if __name__ == "__main__":
    raise SystemExit(main())
