"""Staged, checkpointed runner for Notebook 04h Union episode re-entry."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from evaluation.economics import economics_summary
from experiments.run_unified_2021_ensemble import (
    SourceBundle,
    SourcePaths,
    load_bounded_sources,
    verify_frozen_union,
)
from experiments.unified_2021_ensemble_data import (
    UNIFIED_FEATURES,
    UnifiedDataConfig,
    UnifiedDataset,
    build_unified_dataset,
)
from experiments.union_v1_episode_reentry_data import (
    attach_union_dead_zone_targets,
    load_frozen_union_reentry_dataset,
)
from experiments.union_v1_episode_reentry_models import (
    UnionFoldPredictions,
    UnionReentryModelConfig,
    combine_union_reentry_folds,
    fit_union_reentry_fold,
)
from experiments.union_v1_episode_reentry_policy import (
    FEE_BPS_PER_SIDE,
    MAX_HOLD_BARS,
    SL_BPS,
    TP_BPS,
    build_union_v1_style_signals,
    evaluate_reentry_development,
    replay_union_control_and_reentry,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "union_v1_episode_reentry"
FROZEN_04D = CODE_ROOT / "experiments" / "cache" / "unified_2021_ensemble"
UNION_CACHE = CODE_ROOT / "experiments" / "cache" / "qualified_union_v1"
DEVELOPMENT_START = pd.Timestamp("2021-01-01", tz="UTC")
DEVELOPMENT_END = pd.Timestamp("2025-01-01", tz="UTC")
H1_END = pd.Timestamp("2025-07-01", tz="UTC")
LOCKBOX_START = pd.Timestamp("2026-04-01", tz="UTC")

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


@dataclass
class ReentryStageRun:
    stage: str
    summary: dict[str, object]
    gate_passed: bool
    output_root: Path
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals: pd.DataFrame = field(default_factory=pd.DataFrame)
    control_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    reentry_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)
    candidate_ledger: pd.DataFrame = field(default_factory=pd.DataFrame)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _key_hash(keys: Iterable[object]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        encoded = str(key).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def fold_checkpoint_identity(payload: dict[str, object]) -> str:
    """Hash every declared input that can change one fold result."""
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def freeze_protocol(
    model_config: UnionReentryModelConfig = UnionReentryModelConfig(),
) -> dict[str, object]:
    """Return the single registered experiment; there is no outcome-driven grid."""
    frozen_manifest = FROZEN_04D / "manifest.json"
    union = verify_frozen_union("h1", UNION_CACHE)
    code_files = [
        Path(__file__).resolve(),
        CODE_ROOT / "experiments" / "union_v1_episode_reentry_data.py",
        CODE_ROOT / "experiments" / "union_v1_episode_reentry_models.py",
        CODE_ROOT / "experiments" / "union_v1_episode_reentry_policy.py",
    ]
    payload: dict[str, object] = {
        "protocol_version": "union-v1-style-episode-reentry-v1",
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_end_exclusive": DEVELOPMENT_END.isoformat(),
        "h1_end_exclusive": H1_END.isoformat(),
        "lockbox_start": LOCKBOX_START.isoformat(),
        "lockbox_2026_q2_used": False,
        "n_splits": 5,
        "train_fraction": 0.8,
        "embargo_bars": 8,
        "features": list(UNIFIED_FEATURES),
        "targets": {
            "lstm_dz55": "next_exact_M15_close_return_dead_zone_55bps",
            "svm_linear_dz75": "next_exact_M15_close_return_dead_zone_75bps",
        },
        "models": ["lstm_dz55", "svm_linear_dz75"],
        "model_config": asdict(model_config),
        "signal_rule": {
            "lstm_tau": 0.75,
            "svm_tau": 0.0,
            "combiner": "all_active_same_side_with_opposite_veto",
        },
        "candidate_rules": ["one_earliest_extra_per_same_side_episode"],
        "policy_grid": [],
        "execution": {
            "tp_bps": TP_BPS,
            "sl_bps": SL_BPS,
            "max_hold_m15_bars": MAX_HOLD_BARS,
            "fee_bps_per_side": FEE_BPS_PER_SIDE,
            "expected_execution_interval_seconds": 60,
        },
        "development_gates": [
            "candidate_trades_at_least_ceil_1.15_control",
            "control_positive_total_long_short_and_three_folds",
            "candidate_noninferior_total_long_short",
            "incremental_nonnegative_total_long_short_and_three_folds",
            "candidate_at_least_15_long_and_15_short",
            "all_reconciliation_audits_clean",
        ],
        "h1_label": "observed_development_validation_walk_forward",
        "forward_label": "conditional_development_forward",
        "frozen_04d_manifest_sha256": _sha256_file(frozen_manifest),
        "union_dependency_hashes": union.dependency_hashes,
        "union_reference": UNION_REFERENCE,
        "code_sha256": {
            path.name: _sha256_file(path) for path in code_files if path.is_file()
        },
    }
    payload["protocol_sha256"] = fold_checkpoint_identity(payload)
    return payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
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


def _write_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary, path)


def _frame_identity(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "rows": len(frame),
        "min_timestamp": frame.index.min().isoformat() if len(frame) else None,
        "max_timestamp": frame.index.max().isoformat() if len(frame) else None,
    }


def _dataset_hash(dataset: UnifiedDataset) -> str:
    digest = hashlib.sha256()
    digest.update(_key_hash(dataset.decisions["row_key"]).encode("ascii"))
    digest.update("|".join(dataset.feature_names).encode("utf-8"))
    values = np.asarray(dataset.tabular, dtype=np.float32).copy()
    values[~np.isfinite(values)] = np.nan
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _load_checkpoint(
    path: Path,
    identity: str,
    expected_keys: pd.Series,
) -> UnionFoldPredictions | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if payload.get("identity") != identity:
        return None
    result = payload.get("result")
    if not isinstance(result, UnionFoldPredictions):
        return None
    if _key_hash(result.predictions["row_key"]) != _key_hash(expected_keys):
        return None
    return result


def _checkpointed_predictions(
    dataset: UnifiedDataset,
    manifest: pd.DataFrame,
    model_config: UnionReentryModelConfig,
    protocol: dict[str, object],
    checkpoint_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_hash = _dataset_hash(dataset)
    results: list[UnionFoldPredictions] = []
    for fold_id in sorted(manifest["fold_id"].unique()):
        expected = manifest.loc[
            manifest["fold_id"].eq(fold_id) & manifest["role"].eq("test"),
            "row_key",
        ]
        identity = fold_checkpoint_identity(
            {
                "protocol_sha256": protocol["protocol_sha256"],
                "data_sha256": data_hash,
                "fold_id": int(fold_id),
                "fold_manifest_sha256": _key_hash(
                    manifest.loc[manifest["fold_id"].eq(fold_id), "row_key"]
                ),
                "expected_test_keys_sha256": _key_hash(expected),
                "model_config": asdict(model_config),
                "scheduler": protocol["candidate_rules"],
                "code_sha256": protocol["code_sha256"],
            }
        )
        checkpoint = checkpoint_dir / f"fold_{int(fold_id)}_{identity[:16]}.pkl"
        result = _load_checkpoint(checkpoint, identity, expected)
        if result is None:
            print(f"development fold {int(fold_id) + 1}/5: fit")
            result = fit_union_reentry_fold(dataset, manifest, int(fold_id), model_config)
            _write_pickle(checkpoint, {"identity": identity, "result": result})
        else:
            print(f"development fold {int(fold_id) + 1}/5: checkpoint")
        results.append(result)
    return combine_union_reentry_folds(results)


def _annotate_ledger(ledger: pd.DataFrame, signals: pd.DataFrame) -> pd.DataFrame:
    output = ledger.copy()
    signal_index = signals.set_index("decision_time")
    output["row_key"] = output["signal_time"].map(signal_index["row_key"])
    output["fold_id"] = output["signal_time"].map(signal_index["fold_id"]).astype("Int64")
    output["position_overlap"] = False
    output["round_trip_cost"] = output["gross_return"] - output["net_return"]
    output["cost_match"] = np.isclose(
        output["round_trip_cost"], 2.0 * FEE_BPS_PER_SIDE / 10_000.0, atol=1e-12
    )
    if "execution_interval_seconds" in output:
        output["path_complete"] = output["execution_interval_seconds"].eq(60.0)
    else:
        output["path_complete"] = False
    return output


def _ledger_summary(
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    *,
    audit_clean: bool,
) -> dict[str, object]:
    side = pd.to_numeric(ledger.get("side", pd.Series(dtype=float)), errors="coerce")
    gross = pd.to_numeric(
        ledger.get("gross_return", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    net = pd.to_numeric(
        ledger.get("net_return", pd.Series(dtype=float)), errors="coerce"
    ).fillna(0.0)
    economics = economics_summary(per_bar)
    return {
        "trades": int(len(ledger)),
        "long_trades": int((side > 0).sum()),
        "short_trades": int((side < 0).sum()),
        "gross_return": float(gross.sum()),
        "cost_return": float((gross - net).sum()),
        "net_return": float(net.sum()),
        "long_net_return": float(net.loc[side > 0].sum()),
        "short_net_return": float(net.loc[side < 0].sum()),
        "sortino": float(economics["sortino"]),
        "sharpe": float(economics["sharpe"]),
        "max_drawdown": float(economics["max_drawdown"]),
        "audit_clean": bool(audit_clean),
    }


def _fold_metrics(
    manifest: pd.DataFrame,
    control: pd.DataFrame,
    reentry: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fold_id in sorted(manifest["fold_id"].unique()):
        control_net = float(
            control.loc[control["fold_id"].eq(fold_id), "net_return"].sum()
        )
        incremental_net = float(
            reentry.loc[reentry["fold_id"].eq(fold_id), "net_return"].sum()
        )
        rows.append(
            {
                "fold_id": int(fold_id),
                "control_trades": int(control["fold_id"].eq(fold_id).sum()),
                "reentry_trades": int(reentry["fold_id"].eq(fold_id).sum()),
                "control_net_return": control_net,
                "incremental_net_return": incremental_net,
                "candidate_net_return": control_net + incremental_net,
                "audit_clean": True,
            }
        )
    return pd.DataFrame(rows)


def _episode_table(signals: pd.DataFrame, control: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    active = signals.loc[signals["episode_id"].notna()].copy()
    if active.empty:
        return pd.DataFrame(
            columns=[
                "episode_id",
                "side",
                "start_time",
                "end_time",
                "qualified_bars",
                "control_trades",
                "skipped_qualified_bars",
                "accepted_reentry",
            ]
        )
    control_count = control.groupby("episode_id", dropna=True).size()
    extra_count = extra.groupby("episode_id", dropna=True).size()
    table = active.groupby("episode_id", as_index=False).agg(
        side=("union_signal", "first"),
        start_time=("decision_time", "min"),
        end_time=("decision_time", "max"),
        qualified_bars=("decision_time", "size"),
    )
    table["control_trades"] = table["episode_id"].map(control_count).fillna(0).astype(int)
    table["skipped_qualified_bars"] = table["qualified_bars"] - table["control_trades"]
    table["accepted_reentry"] = table["episode_id"].map(extra_count).fillna(0).astype(int)
    return table


def _returns_frame(per_bar: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(per_bar.index),
            "net_return": per_bar.to_numpy(float),
        }
    )


def _artifact_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if (
            path.is_file()
            and path.name != "manifest.json"
            and path.suffix in {".json", ".csv", ".parquet"}
        ):
            hashes[path.name] = _sha256_file(path)
    return hashes


def _development_artifact_hashes(root: Path) -> dict[str, str]:
    names = {"protocol.json", "frozen_protocol.json"}
    hashes: dict[str, str] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if (
            path.is_file()
            and (path.name.startswith("development_") or path.name in names)
            and path.name != "development_artifacts.json"
            and path.suffix in {".json", ".csv", ".parquet"}
        ):
            hashes[path.name] = _sha256_file(path)
    return hashes


def _real_development_inputs(
    source_paths: SourcePaths,
) -> tuple[UnifiedDataset, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    dataset, manifest, frozen_audit = load_frozen_union_reentry_dataset(FROZEN_04D)
    bundle = load_bounded_sources(DEVELOPMENT_START, DEVELOPMENT_END, source_paths)
    source_audit: dict[str, object] = {
        "frozen_04d": frozen_audit,
        **bundle.source_identities,
    }
    return dataset, manifest, bundle.m15, bundle.minute, source_audit


def run_development(
    *,
    output_root: str | Path = CACHE,
    dataset: UnifiedDataset | None = None,
    manifest: pd.DataFrame | None = None,
    m15: pd.DataFrame | None = None,
    minute: pd.DataFrame | None = None,
    model_config: UnionReentryModelConfig = UnionReentryModelConfig(),
    source_audit: dict[str, object] | None = None,
    protocol: dict[str, object] | None = None,
    source_paths: SourcePaths = SourcePaths(),
) -> ReentryStageRun:
    """Fit/resume five development folds and evaluate the one registered candidate."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    provided = [dataset is not None, manifest is not None, m15 is not None, minute is not None]
    if any(provided) and not all(provided):
        raise ValueError("Injected development dataset, manifest, M15 and M1 must be complete")
    if not any(provided):
        dataset, manifest, m15, minute, loaded_audit = _real_development_inputs(source_paths)
        source_audit = loaded_audit
    assert dataset is not None and manifest is not None and m15 is not None and minute is not None
    source_audit = source_audit or {
        "m15": _frame_identity(m15),
        "minute": _frame_identity(minute),
    }
    protocol = protocol or freeze_protocol(model_config)
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "frozen_protocol.json", protocol)
    _write_json(root / "development_source_audit.json", source_audit)
    _write_parquet(root / "development_fold_manifest.parquet", manifest)
    _write_json(
        root / "run_state.json",
        {
            "stage": "development",
            "h1_loaded": False,
            "forward_loaded": False,
            "lockbox_2026_q2_used": False,
        },
    )
    decision_time = pd.to_datetime(dataset.decisions["decision_time"], utc=True)
    maxima = [decision_time.max(), pd.Timestamp(m15.index.max()), pd.Timestamp(minute.index.max())]
    maximum_loaded = max(maxima)
    if maximum_loaded >= DEVELOPMENT_END:
        raise AssertionError("Development input reached 2025 or later")

    predictions, training_audit = _checkpointed_predictions(
        dataset,
        manifest,
        model_config,
        protocol,
        root / "checkpoints" / "development",
    )
    signals = build_union_v1_style_signals(predictions)
    replay = replay_union_control_and_reentry(signals, m15, minute)
    control = _annotate_ledger(replay.control_ledger, signals)
    reentry = _annotate_ledger(replay.reentry_ledger, signals)
    candidate = pd.concat([control, reentry], ignore_index=True).sort_values(
        ["entry_time", "route"], kind="stable"
    ).reset_index(drop=True)
    if candidate["entry_time"].duplicated().any():
        candidate["position_overlap"] = candidate["entry_time"].duplicated(keep=False)
    clean_control = bool(
        control["cost_match"].all()
        and control["path_complete"].all()
        and control["trade_key"].is_unique
    )
    clean_reentry = bool(
        reentry["cost_match"].all()
        and reentry["path_complete"].all()
        and reentry["trade_key"].is_unique
    )
    clean_candidate = bool(
        clean_control
        and clean_reentry
        and candidate["trade_key"].is_unique
        and not candidate["position_overlap"].any()
        and set(control["trade_key"]).issubset(set(candidate["trade_key"]))
    )
    control_summary = _ledger_summary(
        control, replay.control_returns, audit_clean=clean_control
    )
    reentry_summary = _ledger_summary(
        reentry, replay.reentry_returns, audit_clean=clean_reentry
    )
    candidate_summary = _ledger_summary(
        candidate, replay.candidate_returns, audit_clean=clean_candidate
    )
    folds = _fold_metrics(manifest, control, reentry)
    gates = evaluate_reentry_development(
        control_summary, candidate_summary, reentry_summary, folds
    )
    episode_table = _episode_table(signals, control, reentry)
    calendar_span_days = max(
        float((signals["decision_time"].max() - signals["decision_time"].min() + pd.Timedelta(minutes=15)) / pd.Timedelta(days=1)),
        1.0 / 96.0,
    )
    evaluation_days = len(predictions) / 96.0
    summary: dict[str, object] = {
        "stage": "development",
        "decision": gates["decision"],
        "development_pass": bool(gates["development_pass"]),
        "oof_rows": len(predictions),
        "evaluation_days": evaluation_days,
        "calendar_span_days": calendar_span_days,
        "control_trades": control_summary["trades"],
        "candidate_trades": candidate_summary["trades"],
        "reentry_trades": reentry_summary["trades"],
        "required_candidate_trades": gates["required_candidate_trades"],
        "trade_count_increase": int(candidate_summary["trades"]) - int(control_summary["trades"]),
        "trade_count_increase_fraction": (
            (int(candidate_summary["trades"]) / int(control_summary["trades"]) - 1.0)
            if int(control_summary["trades"]) else 0.0
        ),
        "control_trades_per_day": int(control_summary["trades"]) / evaluation_days,
        "candidate_trades_per_day": int(candidate_summary["trades"]) / evaluation_days,
        "control_net_return": control_summary["net_return"],
        "candidate_net_return": candidate_summary["net_return"],
        "incremental_net_return": reentry_summary["net_return"],
        "control": control_summary,
        "candidate": candidate_summary,
        "incremental": reentry_summary,
        "gates": gates,
        "maximum_loaded_timestamp": maximum_loaded.isoformat(),
        "h1_loaded": False,
        "forward_loaded": False,
        "lockbox_2026_q2_used": False,
    }
    _write_parquet(root / "development_oof_predictions.parquet", predictions)
    _write_csv(root / "development_training_audit.csv", training_audit)
    _write_parquet(root / "development_signals.parquet", signals)
    _write_parquet(root / "development_episode_table.parquet", episode_table)
    _write_parquet(root / "development_selected_reentries.parquet", replay.selected_reentries)
    _write_parquet(root / "development_control_ledger.parquet", control)
    _write_parquet(root / "development_reentry_ledger.parquet", reentry)
    _write_parquet(root / "development_candidate_ledger.parquet", candidate)
    _write_parquet(root / "development_control_returns.parquet", _returns_frame(replay.control_returns))
    _write_parquet(root / "development_reentry_returns.parquet", _returns_frame(replay.reentry_returns))
    _write_parquet(root / "development_candidate_returns.parquet", _returns_frame(replay.candidate_returns))
    _write_csv(root / "development_fold_metrics.csv", folds)
    _write_csv(
        root / "development_route_metrics.csv",
        pd.DataFrame(
            [
                {"route": "union_control", **control_summary},
                {"route": "episode_reentry", **reentry_summary},
                {"route": "candidate_combined", **candidate_summary},
            ]
        ),
    )
    _write_json(root / "development_gates.json", gates)
    _write_json(root / "development_summary.json", summary)
    _write_json(
        root / "development_artifacts.json",
        {
            "protocol_sha256": protocol["protocol_sha256"],
            "artifact_hashes": _development_artifact_hashes(root),
            "lockbox_2026_q2_used": False,
        },
    )
    return ReentryStageRun(
        "development",
        summary,
        bool(gates["development_pass"]),
        root,
        predictions,
        signals,
        control,
        reentry,
        candidate,
    )


def load_h1_sources(source_paths: SourcePaths = SourcePaths()) -> SourceBundle:
    return load_bounded_sources(DEVELOPMENT_START, H1_END, source_paths)


def load_forward_sources(source_paths: SourcePaths = SourcePaths()) -> SourceBundle:
    return load_bounded_sources(DEVELOPMENT_START, LOCKBOX_START, source_paths)


def _dataset_from_bundle(bundle: SourceBundle) -> UnifiedDataset:
    dataset = build_unified_dataset(
        bundle.m15, bundle.minute, bundle.positioning, UnifiedDataConfig()
    )
    return attach_union_dead_zone_targets(dataset)


def _monthly_manifest(
    dataset: UnifiedDataset,
    start: pd.Timestamp,
    end: pd.Timestamp,
    fold_id: int,
) -> pd.DataFrame:
    decisions = dataset.decisions.reset_index(drop=True)
    decision_time = pd.to_datetime(decisions["decision_time"], utc=True)
    target_time = pd.to_datetime(decisions["union_target_time"], utc=True)
    test_mask = decision_time.ge(start) & decision_time.lt(end)
    test_positions = np.flatnonzero(test_mask.to_numpy(bool))
    if not len(test_positions):
        raise ValueError(f"No monthly Union decisions in [{start}, {end})")
    fit_eligible = decision_time.lt(start) & target_time.lt(start)
    fit_positions = np.flatnonzero(fit_eligible.to_numpy(bool))
    if len(fit_positions) <= 8:
        raise ValueError("Monthly Union refit needs history plus embargo")
    fit_positions = fit_positions[:-8]
    stop = int(test_positions.max()) + 1
    positions = np.arange(0, stop, dtype=np.int64)
    role = np.full(stop, "target_censored", dtype=object)
    role[fit_positions] = "fit"
    role[test_positions] = "test"
    return pd.DataFrame(
        {
            "fold_id": fold_id,
            "row_key": decisions.iloc[positions]["row_key"].to_numpy(),
            "position": positions,
            "decision_time": decision_time.iloc[positions].to_numpy(),
            "union_target_time": target_time.iloc[positions].to_numpy(),
            "target_dz55": decisions.iloc[positions]["target_dz55"].to_numpy(),
            "target_dz75": decisions.iloc[positions]["target_dz75"].to_numpy(),
            "role": role,
        }
    )


def _observed_stage_gate(
    stage: str,
    candidate: dict[str, object],
    incremental: dict[str, object],
    clean: bool,
) -> dict[str, object]:
    reference = UNION_REFERENCE[stage]
    minimum_trades = 102 if stage == "h1" else 86
    gates = {
        "minimum_trades": int(candidate["trades"]) >= minimum_trades,
        "net_noninferior_to_union": float(candidate["net_return"]) >= float(reference["net_return"]),
        "long_nonnegative": float(candidate["long_net_return"]) >= 0.0,
        "short_nonnegative": float(candidate["short_net_return"]) >= 0.0,
        "incremental_total_nonnegative": float(incremental["net_return"]) >= 0.0,
        "incremental_long_nonnegative": float(incremental["long_net_return"]) >= 0.0,
        "incremental_short_nonnegative": float(incremental["short_net_return"]) >= 0.0,
        "sortino_noninferior_to_union": float(candidate["sortino"]) >= float(reference["sortino"]),
        "drawdown_noninferior_to_union": float(candidate["max_drawdown"]) <= float(reference["max_drawdown"]),
        "audit_clean": bool(clean),
    }
    gates[f"{stage}_pass"] = bool(all(gates.values()))
    return gates


def _run_observed_stage(
    stage: str,
    bundle: SourceBundle,
    output_root: Path,
    protocol: dict[str, object],
    model_config: UnionReentryModelConfig,
) -> ReentryStageRun:
    if bundle.end > LOCKBOX_START:
        raise AssertionError("Observed stage reached the Q2 lockbox")
    stage_start, stage_end = (
        (DEVELOPMENT_END, H1_END) if stage == "h1" else (H1_END, LOCKBOX_START)
    )
    dataset = _dataset_from_bundle(bundle)
    month_starts = pd.date_range(stage_start, stage_end, freq="MS", inclusive="left")
    results: list[UnionFoldPredictions] = []
    for month_id, month_start in enumerate(month_starts):
        month_start = pd.Timestamp(month_start)
        month_end = min(month_start + pd.offsets.MonthBegin(1), stage_end)
        monthly = _monthly_manifest(dataset, month_start, month_end, month_id)
        print(f"{stage} month {month_id + 1}/{len(month_starts)}: fit")
        results.append(fit_union_reentry_fold(dataset, monthly, month_id, model_config))
    predictions, _ = combine_union_reentry_folds(results)
    signals = build_union_v1_style_signals(predictions)
    m15 = bundle.m15.loc[(bundle.m15.index >= stage_start) & (bundle.m15.index < stage_end)]
    minute = bundle.minute.loc[(bundle.minute.index >= stage_start) & (bundle.minute.index < stage_end)]
    replay = replay_union_control_and_reentry(signals, m15, minute)
    control = _annotate_ledger(replay.control_ledger, signals)
    reentry = _annotate_ledger(replay.reentry_ledger, signals)
    candidate = pd.concat([control, reentry], ignore_index=True).sort_values(
        ["entry_time", "route"], kind="stable"
    ).reset_index(drop=True)
    clean = bool(
        not candidate["entry_time"].duplicated().any()
        and candidate["trade_key"].is_unique
        and control["cost_match"].all()
        and reentry["cost_match"].all()
        and control["path_complete"].all()
        and reentry["path_complete"].all()
    )
    control_summary = _ledger_summary(control, replay.control_returns, audit_clean=clean)
    reentry_summary = _ledger_summary(reentry, replay.reentry_returns, audit_clean=clean)
    candidate_summary = _ledger_summary(candidate, replay.candidate_returns, audit_clean=clean)
    gates = _observed_stage_gate(stage, candidate_summary, reentry_summary, clean)
    passed = bool(gates[f"{stage}_pass"])
    maximum_loaded = bundle.max_loaded_timestamp
    summary = {
        "stage": stage,
        "decision": f"{stage}_{'pass_candidate' if passed else 'fail_keep_union_v1'}",
        "control": control_summary,
        "candidate": candidate_summary,
        "incremental": reentry_summary,
        "gates": gates,
        "maximum_loaded_timestamp": maximum_loaded.isoformat(),
        "lockbox_2026_q2_used": False,
    }
    _write_parquet(output_root / f"{stage}_predictions.parquet", predictions)
    _write_parquet(output_root / f"{stage}_signals.parquet", signals)
    _write_parquet(output_root / f"{stage}_control_ledger.parquet", control)
    _write_parquet(output_root / f"{stage}_reentry_ledger.parquet", reentry)
    _write_parquet(output_root / f"{stage}_candidate_ledger.parquet", candidate)
    _write_json(output_root / f"{stage}_summary.json", summary)
    return ReentryStageRun(
        stage, summary, passed, output_root, predictions, signals, control, reentry, candidate
    )


def run_h1(
    *,
    output_root: str | Path = CACHE,
    bundle: SourceBundle | None = None,
    protocol: dict[str, object] | None = None,
    model_config: UnionReentryModelConfig = UnionReentryModelConfig(),
    source_paths: SourcePaths = SourcePaths(),
) -> ReentryStageRun:
    root = Path(output_root).resolve()
    return _run_observed_stage(
        "h1",
        bundle or load_h1_sources(source_paths),
        root,
        protocol or freeze_protocol(model_config),
        model_config,
    )


def run_forward(
    *,
    output_root: str | Path = CACHE,
    bundle: SourceBundle | None = None,
    protocol: dict[str, object] | None = None,
    model_config: UnionReentryModelConfig = UnionReentryModelConfig(),
    source_paths: SourcePaths = SourcePaths(),
) -> ReentryStageRun:
    root = Path(output_root).resolve()
    return _run_observed_stage(
        "forward",
        bundle or load_forward_sources(source_paths),
        root,
        protocol or freeze_protocol(model_config),
        model_config,
    )


def _remove_stage_files(root: Path, stage: str) -> None:
    for path in root.glob(f"{stage}_*"):
        resolved = path.resolve()
        if resolved.parent != root or not resolved.is_file():
            raise AssertionError("Stale-stage cleanup escaped output root")
        resolved.unlink()


def run_experiment(
    *,
    output_root: str | Path = CACHE,
    source_paths: SourcePaths = SourcePaths(),
) -> dict[str, object]:
    """Run development first and make every later loader gate-unreachable on failure."""
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol()
    _write_json(root / "protocol.json", protocol)
    development = run_development(
        output_root=root, protocol=protocol, source_paths=source_paths
    )
    h1: ReentryStageRun | None = None
    forward: ReentryStageRun | None = None
    if not development.gate_passed:
        _remove_stage_files(root, "h1")
        _remove_stage_files(root, "forward")
        decision = "development_fail_keep_union_v1"
    else:
        h1 = run_h1(output_root=root, protocol=protocol, source_paths=source_paths)
        if not h1.gate_passed:
            _remove_stage_files(root, "forward")
            decision = "h1_fail_keep_union_v1"
        else:
            forward = run_forward(
                output_root=root, protocol=protocol, source_paths=source_paths
            )
            decision = (
                "union_v1_reentry_candidate"
                if forward.gate_passed
                else "forward_fail_keep_union_v1"
            )
    maximum_loaded = max(
        pd.Timestamp(stage.summary["maximum_loaded_timestamp"])
        for stage in (development, h1, forward)
        if stage is not None
    )
    if maximum_loaded >= LOCKBOX_START:
        raise AssertionError("Final source audit reached Q2-2026")
    summary: dict[str, object] = {
        "decision": decision,
        "development": development.summary,
        "h1": h1.summary if h1 is not None else None,
        "forward": forward.summary if forward is not None else None,
        "h1_loaded": h1 is not None,
        "forward_loaded": forward is not None,
        "maximum_loaded_timestamp": maximum_loaded.isoformat(),
        "lockbox_2026_q2_used": False,
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
    manifest = {
        "protocol_sha256": protocol["protocol_sha256"],
        "union_dependency_hashes": protocol["union_dependency_hashes"],
        "artifact_hashes": _artifact_hashes(root),
        "maximum_loaded_timestamp": maximum_loaded.isoformat(),
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
    summary = run_experiment(output_root=args.cache_dir)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE",
    "ReentryStageRun",
    "fold_checkpoint_identity",
    "freeze_protocol",
    "load_forward_sources",
    "load_h1_sources",
    "main",
    "run_development",
    "run_experiment",
    "run_forward",
    "run_h1",
]
