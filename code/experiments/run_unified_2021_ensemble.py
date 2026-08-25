"""Staged runner for the bounded Notebook 04d three-model experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from experiments.unified_2021_ensemble_data import (
    UNIFIED_FEATURES,
    UnifiedDataConfig,
    UnifiedDataset,
    assert_information_contract,
    build_unified_dataset,
    make_blocking_fold_manifest,
)
from experiments.unified_2021_ensemble_models import (
    MODEL_NAMES,
    FoldPredictionResult,
    OOFResult,
    UnifiedModelConfig,
    fit_cross_calibrated_fold,
    fit_historical_snapshot,
    score_historical_snapshot,
    sha256_keys,
)
from experiments.unified_2021_ensemble_policy import (
    POLICY_GRID,
    EnsemblePolicy,
    apply_policy,
    evaluate_oof_policy_grid,
    forward_promotion_gate,
    h1_compatibility_gate,
    ledger_to_common_per_bar,
    replay_selected_paths,
    select_score_only_threshold,
    summarize_candidate,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "unified_2021_ensemble"
UNION_CACHE = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"
LOCKBOX_START = pd.Timestamp("2026-04-01", tz="UTC")
DEVELOPMENT_START = pd.Timestamp("2021-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")
H1_END = pd.Timestamp("2025-07-01", tz="UTC")

UNION_REFERENCE = {
    "h1": {
        "trades": 88,
        "net_return": 0.009687734009,
        "sortino": 0.3521029257,
        "max_drawdown": 0.047613386266,
    },
    "forward": {
        "trades": 74,
        "net_return": 0.063149608713,
        "sortino": 2.1060930178,
        "max_drawdown": 0.029796677136,
    },
}


@dataclass(frozen=True)
class SourcePaths:
    m15: Path = CODE_ROOT / "data" / "btcusdt_15min_2021_2026.parquet"
    minute: Path = CODE_ROOT / "data" / "btcusdt_1m_2021_2026.parquet"
    positioning: Path = (
        CODE_ROOT / "data" / "btcusdt_positioning_15min_2021_2026.parquet"
    )


@dataclass(frozen=True)
class SourceBundle:
    m15: pd.DataFrame
    minute: pd.DataFrame
    positioning: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp
    source_identities: dict[str, dict[str, object]]

    @property
    def max_loaded_timestamp(self) -> pd.Timestamp:
        maxima = [
            pd.Timestamp(identity["max_timestamp"])
            for identity in self.source_identities.values()
            if identity.get("max_timestamp") is not None
        ]
        return max(maxima) if maxima else self.start


@dataclass(frozen=True)
class UnionReference:
    stage: str
    summary: dict[str, object]
    dependency_hashes: dict[str, str]
    root: Path


@dataclass
class DevelopmentResult:
    selected_policy: EnsemblePolicy | None
    opportunity_threshold: float | None
    summary: dict[str, object]
    work_dir: Path
    protocol: dict[str, object]
    source_identities: dict[str, dict[str, object]] = field(default_factory=dict)


@dataclass
class StageResult:
    stage: str
    summary: dict[str, object]
    work_dir: Path
    selected_policy: EnsemblePolicy
    opportunity_threshold: float
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    per_bar: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    funnel: pd.DataFrame = field(default_factory=pd.DataFrame)
    refit_audit: pd.DataFrame = field(default_factory=pd.DataFrame)
    gate_passed: bool = False
    source_identities: dict[str, dict[str, object]] = field(default_factory=dict)


def _utc_boundary(value, name: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware")
    return timestamp.tz_convert("UTC")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), separators=(",", ":")).encode())
    digest.update(str(tuple(str(dtype) for dtype in frame.dtypes)).encode())
    hashed = pd.util.hash_pandas_object(frame, index=True, categorize=True)
    digest.update(hashed.to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def _source_identity(name: str, path: Path, frame: pd.DataFrame) -> dict[str, object]:
    return {
        "name": name,
        "path": str(path.resolve()),
        "rows": len(frame),
        "min_timestamp": frame.index.min().isoformat() if len(frame) else None,
        "max_timestamp": frame.index.max().isoformat() if len(frame) else None,
        "bounded_sha256": _frame_sha256(frame),
    }


def _read_bounded(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    schema_names = pq.read_schema(path).names
    if "timestamp" in schema_names:
        timestamp_field = "timestamp"
    elif "__index_level_0__" in schema_names:
        timestamp_field = "__index_level_0__"
    else:
        raise ValueError(f"{path.name} has no parquet timestamp index field")
    frame = pd.read_parquet(
        path,
        filters=[(timestamp_field, ">=", start), (timestamp_field, "<", end)],
    )
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{path.name} needs a DatetimeIndex")
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    frame.index = index
    if frame.empty:
        raise ValueError(f"bounded source {path.name} is empty")
    if frame.index.min() < start or frame.index.max() >= end:
        raise AssertionError(f"bounded read escaped [{start}, {end})")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"{path.name} index must be unique and increasing")
    return frame


def load_bounded_sources(
    start,
    end,
    paths: SourcePaths = SourcePaths(),
) -> SourceBundle:
    """Read all inputs through explicit half-open filters below Q2-2026."""
    lower = _utc_boundary(start, "start")
    upper = _utc_boundary(end, "end")
    if lower >= upper:
        raise ValueError("source interval must be increasing")
    if upper > LOCKBOX_START:
        raise ValueError("Q2-2026 lockbox cannot be opened")
    m15_end = upper - pd.Timedelta(minutes=15)
    if m15_end <= lower:
        raise ValueError("bounded interval is too short for one completed M15 decision")
    m15 = _read_bounded(paths.m15, lower, m15_end)
    minute = _read_bounded(paths.minute, lower, upper)
    positioning = _read_bounded(paths.positioning, lower, m15_end)
    identities = {
        "m15": _source_identity("m15", paths.m15, m15),
        "minute": _source_identity("minute", paths.minute, minute),
        "positioning": _source_identity("positioning", paths.positioning, positioning),
    }
    maximum = max(pd.Timestamp(item["max_timestamp"]) for item in identities.values())
    if maximum >= LOCKBOX_START:
        raise AssertionError("a bounded source reached the Q2-2026 lockbox")
    return SourceBundle(m15, minute, positioning, lower, upper, identities)


def verify_frozen_union(
    stage: str,
    root: Path = UNION_CACHE,
) -> UnionReference:
    """Verify immutable Union files before accepting its reference metrics."""
    if stage not in UNION_REFERENCE:
        raise ValueError(f"unknown Union stage: {stage}")
    union_root = Path(root).resolve()
    manifest_path = union_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    dependencies: dict[str, str] = {}
    for filename, expected in manifest.get("artifact_hashes", {}).items():
        path = (union_root / filename).resolve()
        if path.parent != union_root or not path.is_file():
            raise AssertionError(f"Union artifact is missing or escaped root: {filename}")
        actual = _sha256_file(path)
        if actual != expected:
            raise AssertionError(f"Union artifact hash mismatch: {filename}")
        dependencies[filename] = actual
    dependencies["manifest.json"] = _sha256_file(manifest_path)
    summary_path = union_root / "summary.csv"
    summary_frame = pd.read_csv(summary_path)
    selected = summary_frame.loc[summary_frame["phase"].eq(stage)]
    if len(selected) != 1:
        raise AssertionError(f"Union summary needs exactly one {stage} row")
    summary = selected.iloc[0].to_dict()
    for metric, expected in UNION_REFERENCE[stage].items():
        actual = summary[metric]
        if metric == "trades":
            if int(actual) != int(expected):
                raise AssertionError(f"Union {stage} {metric} drifted")
        # Registered references are frozen to 10-12 decimal places while the
        # immutable CSV retains full precision; tolerate only that publication
        # rounding, not an economically meaningful drift.
        elif not np.isclose(float(actual), float(expected), rtol=0.0, atol=5e-11):
            raise AssertionError(f"Union {stage} {metric} drifted")
    return UnionReference(stage, summary, dependencies, union_root)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def freeze_protocol(
    data_config: UnifiedDataConfig = UnifiedDataConfig(),
    model_config: UnifiedModelConfig = UnifiedModelConfig(),
) -> dict[str, object]:
    """Return the deterministic protocol declaration used to key checkpoints."""
    payload: dict[str, object] = {
        "protocol_version": "unified-2021-ensemble-v1",
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_end_exclusive": DEVELOPMENT_END.isoformat(),
        "h1_end_exclusive": H1_END.isoformat(),
        "lockbox_start": LOCKBOX_START.isoformat(),
        "lockbox_2026_q2_used": False,
        "data_config": asdict(data_config),
        "model_config": asdict(model_config),
        "feature_names": list(UNIFIED_FEATURES),
        "policy_grid": [asdict(policy) for policy in POLICY_GRID],
        "union_reference": UNION_REFERENCE,
        "h1_label": "observed_development_walk_forward",
        "forward_label": "conditional_development_forward",
    }
    payload["protocol_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json(path: Path, payload: object) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _dataset_from_bundle(
    bundle: SourceBundle,
    config: UnifiedDataConfig = UnifiedDataConfig(),
) -> UnifiedDataset:
    dataset = build_unified_dataset(
        bundle.m15, bundle.minute, bundle.positioning, config
    )
    decision_time = pd.to_datetime(dataset.decisions["decision_time"], utc=True)
    keep = decision_time.ge(bundle.start) & decision_time.lt(bundle.end)
    if not keep.all():
        positions = np.flatnonzero(keep.to_numpy(bool))
        decisions = dataset.decisions.iloc[positions].reset_index(drop=True)
        dataset = UnifiedDataset(
            decisions=decisions,
            tabular=dataset.tabular[positions],
            sequences=dataset.sequences[positions],
            feature_names=dataset.feature_names,
            economic_paths=dataset.economic_paths.loc[
                dataset.economic_paths["row_key"].isin(decisions["row_key"])
            ].reset_index(drop=True),
        )
    return dataset


def _checkpoint_identity(protocol: dict[str, object], bundle: SourceBundle) -> str:
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "sources": bundle.source_identities,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _save_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _load_fold_checkpoint(
    path: Path,
    identity: str,
    expected_keys: pd.Series,
) -> FoldPredictionResult | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("identity") != identity:
        return None
    result = payload.get("result")
    if not isinstance(result, FoldPredictionResult):
        return None
    if sha256_keys(result.test_predictions["row_key"]) != sha256_keys(expected_keys):
        return None
    return result


def _combine_fold_results(results: list[FoldPredictionResult]) -> OOFResult:
    predictions = pd.concat(
        [result.test_predictions for result in results], ignore_index=True
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)
    if predictions["row_key"].duplicated().any():
        raise AssertionError("outer-test decisions overlap across checkpoints")
    calibration = pd.concat(
        [result.calibration_predictions for result in results], ignore_index=True
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)
    metrics = pd.concat(
        [result.calibration_metrics for result in results], ignore_index=True
    )
    reliability = pd.concat(
        [result.reliability_bins for result in results], ignore_index=True
    )
    fit_audit = pd.concat([result.fit_audit for result in results], ignore_index=True)
    key_hash = sha256_keys(predictions["row_key"])
    key_audit = pd.DataFrame(
        {"model": MODEL_NAMES, "rows": len(predictions), "keys_sha256": key_hash}
    )
    return OOFResult(
        predictions,
        calibration,
        metrics,
        reliability,
        key_audit,
        fit_audit,
    )


def _run_checkpointed_oof(
    dataset: UnifiedDataset,
    manifest: pd.DataFrame,
    model_config: UnifiedModelConfig,
    checkpoint_dir: Path,
    identity: str,
) -> OOFResult:
    results: list[FoldPredictionResult] = []
    for fold_id in sorted(manifest["fold_id"].unique()):
        expected = manifest.loc[
            manifest["fold_id"].eq(fold_id) & manifest["role"].eq("test"),
            "row_key",
        ]
        checkpoint = checkpoint_dir / f"fold_{int(fold_id)}_{identity[:16]}.pkl"
        result = _load_fold_checkpoint(checkpoint, identity, expected)
        if result is None:
            result = fit_cross_calibrated_fold(
                dataset, manifest, int(fold_id), model_config
            )
            _save_pickle(checkpoint, {"identity": identity, "result": result})
        results.append(result)
    return _combine_fold_results(results)


def _bars_for_period(
    bundle: SourceBundle, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    bars = bundle.m15.loc[(bundle.m15.index >= start) & (bundle.m15.index < end), ["close"]]
    if bars.empty:
        raise ValueError(f"no M15 bars for [{start}, {end})")
    return bars


def _selected_policy_payload(
    policy: EnsemblePolicy | None,
    threshold: float | None,
) -> dict[str, object]:
    return {
        "selected": policy is not None,
        "policy": asdict(policy) if policy is not None else None,
        "opportunity_threshold": threshold,
    }


def run_development(
    bundle: SourceBundle,
    protocol: dict[str, object],
    work_dir: Path,
) -> DevelopmentResult:
    """Complete and freeze all 2021-2024 selection before H1 is reachable."""
    root = Path(work_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "frozen_protocol.json", protocol)
    _write_json(
        root / "run_state.json",
        {
            "stage": "development",
            "h1_loaded": False,
            "forward_loaded": False,
            "lockbox_2026_q2_used": False,
        },
    )
    declared_grid = pd.DataFrame([asdict(policy) for policy in POLICY_GRID])
    _write_csv(root / "declared_policy_grid.csv", declared_grid)

    data_config = UnifiedDataConfig()
    model_config = UnifiedModelConfig()
    dataset = _dataset_from_bundle(bundle, data_config)
    feature_audit = assert_information_contract(dataset)
    manifest = make_blocking_fold_manifest(
        dataset.decisions,
        dataset.decisions[["row_key", "path_complete", "label_end"]],
        data_config,
    )
    _write_parquet(root / "decision_dataset.parquet", dataset.decisions)
    _write_parquet(root / "economic_labels.parquet", dataset.economic_paths)
    _write_csv(root / "fold_audit.csv", manifest)
    _write_csv(root / "feature_audit.csv", feature_audit)

    identity = _checkpoint_identity(protocol, bundle)
    oof = _run_checkpointed_oof(
        dataset,
        manifest,
        model_config,
        root / "checkpoints" / "development",
        identity,
    )
    _write_parquet(root / "oof_predictions.parquet", oof.predictions)
    _write_csv(root / "calibration_metrics.csv", oof.calibration_metrics)
    _write_csv(root / "reliability_bins.csv", oof.reliability_bins)
    _write_csv(root / "leakage_audit.csv", oof.fit_audit)
    _write_csv(root / "model_key_audit.csv", oof.model_key_audit)

    policy_grid, selected = evaluate_oof_policy_grid(oof, dataset.economic_paths)
    _write_csv(root / "policy_grid.csv", policy_grid)
    if selected is None:
        payload = _selected_policy_payload(None, None)
        payload["decision"] = "development_fail_keep_union_v1"
        _write_json(root / "selected_policy.json", payload)
        summary = {
            "decision": "development_fail_keep_union_v1",
            "development_policy_found": False,
            "qualifying_policies": 0,
            "oof_rows": len(oof.predictions),
        }
        return DevelopmentResult(
            None, None, summary, root, protocol, bundle.source_identities
        )

    activations, _ = apply_policy(
        oof.predictions, oof.calibration_predictions, selected
    )
    ledger = replay_selected_paths(activations, dataset.economic_paths)
    per_bar = ledger_to_common_per_bar(
        ledger, _bars_for_period(bundle, DEVELOPMENT_START, DEVELOPMENT_END)
    )
    development_summary = summarize_candidate(ledger, per_bar, "development_oof")

    snapshot = fit_historical_snapshot(dataset, DEVELOPMENT_END, model_config)
    calibration = snapshot.calibration_predictions.sort_values(
        "decision_time", kind="stable"
    )
    score = calibration[
        [f"p_opportunity_{model}" for model in MODEL_NAMES]
    ].median(axis=1)
    score.index = pd.DatetimeIndex(calibration["decision_time"])
    opportunity_threshold, threshold_frontier = select_score_only_threshold(
        score, selected.daily_rate_cap
    )
    _write_csv(root / "final_threshold_frontier.csv", threshold_frontier)
    payload = _selected_policy_payload(selected, opportunity_threshold)
    payload.update(
        {
            "decision": "development_pass_policy_frozen",
            "fit_max_label_end": snapshot.fit_max_label_end,
            "calibration_start": snapshot.calibration_start,
            "calibration_max_label_end": snapshot.calibration_max_label_end,
        }
    )
    _write_json(root / "selected_policy.json", payload)
    summary = {
        **development_summary,
        "decision": "development_pass_policy_frozen",
        "development_policy_found": True,
        "qualifying_policies": int(policy_grid["qualifies"].sum()),
        "opportunity_threshold": opportunity_threshold,
    }
    return DevelopmentResult(
        selected,
        opportunity_threshold,
        summary,
        root,
        protocol,
        bundle.source_identities,
    )


def _stage_bounds(stage: str) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    if stage == "h1":
        return DEVELOPMENT_END, H1_END, "observed_development_walk_forward"
    if stage == "forward":
        return H1_END, LOCKBOX_START, "conditional_development_forward"
    raise ValueError(f"unknown walk-forward stage: {stage}")


def _monthly_checkpoint(
    path: Path,
    identity: str,
) -> dict[str, object] | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    return payload if payload.get("identity") == identity else None


def _monthly_summaries(
    stage: str,
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for month_start in pd.date_range(start, end, freq="MS", inclusive="left"):
        month_end = min(month_start + pd.offsets.MonthBegin(1), end)
        month_bar = per_bar.loc[(per_bar.index >= month_start) & (per_bar.index < month_end)]
        if ledger.empty:
            month_ledger = ledger.copy()
        else:
            entry = pd.to_datetime(ledger["entry_time"], utc=True)
            month_ledger = ledger.loc[entry.ge(month_start) & entry.lt(month_end)]
        row = summarize_candidate(month_ledger, month_bar, f"{stage}_month")
        row["month"] = month_start.strftime("%Y-%m")
        rows.append(row)
    return pd.DataFrame(rows)


def _side_metrics(ledger: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for side in ("long", "short"):
        if ledger.empty:
            selected = ledger
        else:
            direction = ledger["direction"].astype(str)
            selected = ledger.loc[direction.eq(side)]
        net = pd.to_numeric(selected.get("net_return", pd.Series(dtype=float)), errors="coerce")
        rows.append(
            {
                "side": side,
                "trades": len(selected),
                "net_return": float(net.sum()),
                "win_rate": float(net.gt(0.0).mean()) if len(net) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def run_walk_forward(
    stage: str,
    bundle: SourceBundle,
    selected_policy: EnsemblePolicy,
    opportunity_threshold: float,
    work_dir: Path,
) -> StageResult:
    """Run fixed-policy monthly causal refits over H1 or conditional forward."""
    stage_start, stage_end, stage_label = _stage_bounds(stage)
    if bundle.end < stage_end:
        raise ValueError(f"{stage} bundle ends before its registered boundary")
    if bundle.end > LOCKBOX_START:
        raise ValueError("Q2-2026 lockbox cannot be opened")
    root = Path(work_dir).resolve()
    dataset = _dataset_from_bundle(bundle, UnifiedDataConfig())
    decision_time = pd.to_datetime(dataset.decisions["decision_time"], utc=True)
    model_config = UnifiedModelConfig()
    protocol = freeze_protocol(UnifiedDataConfig(), model_config)
    base_identity = _checkpoint_identity(protocol, bundle)
    prediction_parts: list[pd.DataFrame] = []
    calibration_parts: list[pd.DataFrame] = []
    refit_rows: list[dict[str, object]] = []
    for month_start in pd.date_range(
        stage_start, stage_end, freq="MS", inclusive="left"
    ):
        month_end = min(month_start + pd.offsets.MonthBegin(1), stage_end)
        month_identity = hashlib.sha256(
            f"{base_identity}|{stage}|{month_start.isoformat()}".encode()
        ).hexdigest()
        checkpoint = (
            root
            / "checkpoints"
            / stage
            / f"{month_start.strftime('%Y_%m')}_{month_identity[:16]}.pkl"
        )
        cached = _monthly_checkpoint(checkpoint, month_identity)
        if cached is None:
            snapshot = fit_historical_snapshot(dataset, month_start, model_config)
            positions = np.flatnonzero(
                decision_time.ge(month_start).to_numpy(bool)
                & decision_time.lt(month_end).to_numpy(bool)
                & dataset.decisions["path_complete"].fillna(False).to_numpy(bool)
            )
            predictions = score_historical_snapshot(snapshot, dataset, positions)
            calibration = snapshot.calibration_predictions
            audit = {
                "scored_month_start": month_start,
                "scored_month_end_exclusive": month_end,
                "fit_max_label_end": snapshot.fit_max_label_end,
                "calibration_start": snapshot.calibration_start,
                "calibration_max_label_end": snapshot.calibration_max_label_end,
                "calibration_max_decision_time": pd.to_datetime(
                    calibration["decision_time"], utc=True
                ).max(),
                "scored_rows": len(predictions),
            }
            cached = {
                "identity": month_identity,
                "predictions": predictions,
                "calibration": calibration,
                "audit": audit,
            }
            _save_pickle(checkpoint, cached)
        audit = dict(cached["audit"])
        if not pd.Timestamp(audit["fit_max_label_end"]) < month_start:
            raise AssertionError("monthly fit labels cross the scored month")
        if not pd.Timestamp(audit["calibration_max_label_end"]) < month_start:
            raise AssertionError("monthly calibration labels cross the scored month")
        if not pd.Timestamp(audit["calibration_max_decision_time"]) < month_start:
            raise AssertionError("monthly calibration decisions cross the scored month")
        prediction_parts.append(cached["predictions"])
        calibration_parts.append(cached["calibration"])
        refit_rows.append(audit)
    predictions = pd.concat(prediction_parts, ignore_index=True).sort_values(
        "decision_time", kind="stable"
    ).reset_index(drop=True)
    if predictions["row_key"].duplicated().any():
        raise AssertionError("monthly predictions revise an earlier decision")
    calibration = pd.concat(calibration_parts, ignore_index=True)
    activations, funnel = apply_policy(
        predictions,
        calibration,
        selected_policy,
        opportunity_threshold=opportunity_threshold,
    )
    ledger = replay_selected_paths(activations, dataset.economic_paths)
    bars = _bars_for_period(bundle, stage_start, stage_end)
    per_bar = ledger_to_common_per_bar(ledger, bars)
    summary = summarize_candidate(ledger, per_bar, stage_label)
    summary["stage"] = stage
    summary["stage_label"] = stage_label
    summary["opportunity_threshold"] = opportunity_threshold
    summary["max_scored_timestamp"] = (
        pd.to_datetime(predictions["decision_time"], utc=True).max()
        if len(predictions)
        else None
    )
    if summary["max_scored_timestamp"] is not None and summary["max_scored_timestamp"] >= LOCKBOX_START:
        raise AssertionError("walk-forward score reached the Q2-2026 lockbox")
    gate_passed = (
        h1_compatibility_gate(summary)
        if stage == "h1"
        else forward_promotion_gate(summary)
    )
    summary["gate_passed"] = gate_passed
    refit_audit = pd.DataFrame(refit_rows)
    monthly = _monthly_summaries(stage, ledger, per_bar, stage_start, stage_end)
    side_metrics = _side_metrics(ledger)

    _write_parquet(root / f"{stage}_predictions.parquet", predictions)
    _write_parquet(root / f"{stage}_trade_ledger.parquet", ledger)
    _write_parquet(
        root / f"{stage}_per_bar.parquet",
        per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
    )
    _write_parquet(root / f"{stage}_funnel.parquet", funnel)
    _write_csv(root / f"{stage}_refit_audit.csv", refit_audit)
    _write_csv(root / f"{stage}_summary.csv", pd.DataFrame([summary]))
    _write_csv(root / f"{stage}_monthly.csv", monthly)
    _write_csv(root / f"{stage}_side_metrics.csv", side_metrics)
    return StageResult(
        stage=stage,
        summary=summary,
        work_dir=root,
        selected_policy=selected_policy,
        opportunity_threshold=opportunity_threshold,
        predictions=predictions,
        ledger=ledger,
        per_bar=per_bar,
        funnel=funnel,
        refit_audit=refit_audit,
        gate_passed=gate_passed,
        source_identities=bundle.source_identities,
    )


def remove_stale_forward_artifacts(cache_dir: Path) -> None:
    """Remove only registered direct-child forward outputs from one cache."""
    root = Path(cache_dir).resolve()
    if not root.exists():
        return
    for path in root.glob("forward_*"):
        resolved = path.resolve()
        if resolved.parent != root:
            raise AssertionError("stale forward target escaped configured cache")
        if resolved.is_file():
            resolved.unlink()


def _remove_stale_h1_artifacts(cache_dir: Path) -> None:
    root = Path(cache_dir).resolve()
    if not root.exists():
        return
    for path in root.glob("h1_*"):
        resolved = path.resolve()
        if resolved.parent == root and resolved.is_file():
            resolved.unlink()


def maybe_run_h1(
    development: DevelopmentResult,
    loader: Callable[[], SourceBundle],
) -> StageResult | None:
    if development.selected_policy is None or development.opportunity_threshold is None:
        return None
    bundle = loader()
    return run_walk_forward(
        "h1",
        bundle,
        development.selected_policy,
        development.opportunity_threshold,
        development.work_dir,
    )


def maybe_run_forward(
    h1: StageResult | None,
    loader: Callable[[], SourceBundle],
    cache_dir: Path,
) -> StageResult | None:
    if h1 is None or not h1.gate_passed:
        remove_stale_forward_artifacts(cache_dir)
        return None
    bundle = loader()
    return run_walk_forward(
        "forward",
        bundle,
        h1.selected_policy,
        h1.opportunity_threshold,
        h1.work_dir,
    )


def _artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name != "manifest.json" and path.suffix in {
            ".json",
            ".csv",
            ".parquet",
        }:
            hashes[path.name] = _sha256_file(path)
    return hashes


def run(
    cache_dir: Path = CACHE,
    source_paths: SourcePaths = SourcePaths(),
) -> dict[str, object]:
    """Execute the full gated experiment once and return its final summary."""
    root = Path(cache_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    union_h1 = verify_frozen_union("h1")
    protocol = freeze_protocol()
    development_bundle = load_bounded_sources(
        DEVELOPMENT_START, DEVELOPMENT_END, source_paths
    )
    development = run_development(development_bundle, protocol, root)
    loaded_sources: dict[str, object] = {"development": development.source_identities}
    h1: StageResult | None = None
    forward: StageResult | None = None
    union_forward: UnionReference | None = None
    if development.selected_policy is None:
        _remove_stale_h1_artifacts(root)
        remove_stale_forward_artifacts(root)
        decision = "development_fail_keep_union_v1"
    else:
        h1 = maybe_run_h1(
            development,
            lambda: load_bounded_sources(DEVELOPMENT_START, H1_END, source_paths),
        )
        if h1 is None:
            decision = "development_fail_keep_union_v1"
        else:
            loaded_sources["h1"] = h1.source_identities
            if not h1.gate_passed:
                remove_stale_forward_artifacts(root)
                decision = "h1_fail_keep_union_v1"
            else:
                union_forward = verify_frozen_union("forward")
                forward = maybe_run_forward(
                    h1,
                    lambda: load_bounded_sources(
                        DEVELOPMENT_START, LOCKBOX_START, source_paths
                    ),
                    root,
                )
                if forward is None:
                    decision = "h1_fail_keep_union_v1"
                else:
                    loaded_sources["forward"] = forward.source_identities
                    decision = (
                        "promote_ensemble_v2_candidate"
                        if forward.gate_passed
                        else "forward_fail_keep_union_v1"
                    )
    maxima = [
        pd.Timestamp(identity["max_timestamp"])
        for stage_sources in loaded_sources.values()
        for identity in stage_sources.values()
        if identity.get("max_timestamp") is not None
    ]
    max_loaded = max(maxima)
    if max_loaded >= LOCKBOX_START:
        raise AssertionError("final source audit reached Q2-2026")
    summary: dict[str, object] = {
        "decision": decision,
        "development": development.summary,
        "h1": h1.summary if h1 is not None else None,
        "forward": forward.summary if forward is not None else None,
        "h1_loaded": h1 is not None,
        "h1_compatibility_passed": bool(h1 and h1.gate_passed),
        "forward_loaded": forward is not None,
        "forward_promoted": bool(forward and forward.gate_passed),
        "lockbox_2026_q2_used": False,
        "max_loaded_timestamp": max_loaded.isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "union_h1": union_h1.summary,
        "union_forward": union_forward.summary if union_forward else None,
    }
    _write_json(root / "summary.json", summary)
    _write_json(
        root / "run_state.json",
        {
            "stage": "complete",
            "decision": decision,
            "h1_loaded": h1 is not None,
            "forward_loaded": forward is not None,
            "lockbox_2026_q2_used": False,
        },
    )
    union_dependencies = dict(union_h1.dependency_hashes)
    if union_forward is not None:
        union_dependencies.update(union_forward.dependency_hashes)
    manifest = {
        "protocol_sha256": protocol["protocol_sha256"],
        "bounded_sources": loaded_sources,
        "union_dependency_hashes": union_dependencies,
        "artifact_hashes": _artifact_hashes(root),
        "maximum_loaded_timestamp": max_loaded.isoformat(),
        "h1_loaded": h1 is not None,
        "forward_loaded": forward is not None,
        "lockbox_2026_q2_used": False,
    }
    _write_json(root / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    args = parser.parse_args(argv)
    summary = run(args.cache_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE",
    "DevelopmentResult",
    "SourceBundle",
    "SourcePaths",
    "StageResult",
    "UnionReference",
    "freeze_protocol",
    "load_bounded_sources",
    "main",
    "maybe_run_forward",
    "maybe_run_h1",
    "remove_stale_forward_artifacts",
    "run",
    "run_development",
    "run_walk_forward",
    "verify_frozen_union",
]
