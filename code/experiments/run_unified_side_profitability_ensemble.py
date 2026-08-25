"""Run the gated Notebook 04e side-profitability ensemble experiment."""
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

from experiments.run_unified_2021_ensemble import (
    H1_END,
    LOCKBOX_START,
    SourceBundle,
    SourcePaths,
    _artifact_hashes,
    _bars_for_period,
    _dataset_from_bundle,
    _sha256_file,
    _write_csv,
    _write_json,
    _write_parquet,
    load_bounded_sources,
    verify_frozen_union,
)
from experiments.unified_2021_ensemble_data import UNIFIED_FEATURES
from experiments.unified_2021_ensemble_models import (
    UnifiedModelConfig,
    sha256_keys,
)
from experiments.unified_2021_ensemble_policy import (
    forward_promotion_gate,
    h1_compatibility_gate,
    ledger_to_common_per_bar,
    replay_selected_paths,
    summarize_candidate,
)
from experiments.unified_side_profitability_data import (
    DEVELOPMENT_END,
    PROFITABILITY_HEADS,
    attach_profitability_targets,
    load_frozen_development_dataset,
    make_four_role_manifest,
)
from experiments.unified_side_profitability_models import (
    ProfitabilityFoldResult,
    fit_profitability_fold,
)
from experiments.unified_side_profitability_policy import (
    POLICY_GRID,
    FoldPolicySelection,
    apply_frozen_fold_policy,
    evaluate_development_ledger,
    select_fold_policy,
    xgb_satellite_ablation,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "unified_side_profitability_ensemble"
FROZEN_DEVELOPMENT_CACHE = CODE_ROOT / "experiments" / "cache" / "unified_2021_ensemble"
DEVELOPMENT_START = pd.Timestamp("2021-01-01", tz="UTC")
ROLES = ("fit", "probability_calibration", "policy_selection", "test")


@dataclass
class DevelopmentRun:
    summary: dict[str, object]
    ledger: pd.DataFrame
    fold_selections: list[FoldPolicySelection]
    source_audit: dict[str, object]
    work_dir: Path


@dataclass
class StageRun:
    stage: str
    summary: dict[str, object]
    ledger: pd.DataFrame
    source_audit: dict[str, object]
    work_dir: Path
    gate_passed: bool
    fold_selections: list[FoldPolicySelection] = field(default_factory=list)


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def freeze_protocol(
    model_config: UnifiedModelConfig = UnifiedModelConfig(),
) -> dict[str, object]:
    """Declare every tunable choice before outer-test or later-stage scoring."""
    payload: dict[str, object] = {
        "protocol_version": "side-profitability-ensemble-v2",
        "development_start": DEVELOPMENT_START.isoformat(),
        "development_end_exclusive": DEVELOPMENT_END.isoformat(),
        "h1_end_exclusive": H1_END.isoformat(),
        "forward_end_exclusive": LOCKBOX_START.isoformat(),
        "lockbox_2026_q2_used": False,
        "targets": list(PROFITABILITY_HEADS),
        "roles": list(ROLES),
        "purge_embargo_m15_bars": 8,
        "refractory_minutes": 60,
        "feature_names": list(UNIFIED_FEATURES),
        "feature_count": len(UNIFIED_FEATURES),
        "feature_source": "immutable_notebook_04d_development_cache",
        "model_config": asdict(model_config),
        "policy_grid": [asdict(policy) for policy in POLICY_GRID],
        "development_gates": {
            "non_regression_trades": 87,
            "promotion_trades": 132,
            "minimum_trades_per_side": 15,
            "minimum_positive_folds": 3,
            "minimum_xgb_solo_trades": 10,
            "minimum_policy_selection_passes": 4,
        },
        "h1_gate": "frozen_notebook_04d_union_compatibility",
        "forward_gate": "frozen_notebook_04d_union_promotion",
        "lob_used": False,
    }
    payload["protocol_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def _frame_sha256(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_json(list(frame.columns)).encode("utf-8"))
    digest.update(_canonical_json([str(value) for value in frame.dtypes]).encode("utf-8"))
    digest.update(
        pd.util.hash_pandas_object(frame, index=True, categorize=True)
        .to_numpy(np.uint64)
        .tobytes()
    )
    return digest.hexdigest()


def _checkpoint_identity(
    protocol: dict[str, object],
    source_audit: dict[str, object],
    manifest_sha256: str,
    expected_outer_keys: Iterable[object],
) -> str:
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "source_manifest_sha256": source_audit.get("source_manifest_sha256"),
        "artifact_hashes": source_audit.get("artifact_hashes", {}),
        "manifest_sha256": manifest_sha256,
        "outer_keys_sha256": sha256_keys(expected_outer_keys),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _save_checkpoint(path: Path, identity: str, result: ProfitabilityFoldResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(
            {"identity": identity, "result": result},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    os.replace(temporary, path)


def _load_checkpoint(
    path: Path,
    identity: str,
    expected_outer_keys: Iterable[object],
) -> ProfitabilityFoldResult | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    result = payload.get("result")
    if payload.get("identity") != identity or not isinstance(
        result, ProfitabilityFoldResult
    ):
        return None
    if sha256_keys(result.test_predictions["row_key"]) != sha256_keys(
        expected_outer_keys
    ):
        return None
    return result


def _selection_row(selection: FoldPolicySelection) -> dict[str, object]:
    return {
        "fold_id": selection.fold_id,
        "source_role": selection.source_role,
        "selection_passed": selection.selection_passed,
        "regular_side_rate_cap": selection.policy.regular_side_rate_cap,
        "xgb_side_rate_cap": selection.policy.xgb_side_rate_cap,
        **selection.thresholds,
    }


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    useful = [frame for frame in frames if not frame.empty]
    return pd.concat(useful, ignore_index=True) if useful else pd.DataFrame()


def _ledger_group_metrics(ledger: pd.DataFrame, column: str) -> pd.DataFrame:
    if ledger.empty or column not in ledger:
        return pd.DataFrame(columns=[column, "trades", "net_return", "positive"])
    rows = []
    for key, group in ledger.groupby(column, sort=True):
        net = float(pd.to_numeric(group["net_return"], errors="raise").sum())
        rows.append(
            {column: key, "trades": len(group), "net_return": net, "positive": net > 0.0}
        )
    return pd.DataFrame(rows)


def _write_development_artifacts(
    root: Path,
    protocol: dict[str, object],
    source_audit: dict[str, object],
    manifest: pd.DataFrame,
    results: list[ProfitabilityFoldResult],
    selections: list[FoldPolicySelection],
    activations: pd.DataFrame,
    funnel: pd.DataFrame,
    ledger: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    _write_json(root / "protocol.json", protocol)
    _write_json(root / "frozen_protocol.json", protocol)
    _write_json(root / "source_audit.json", source_audit)
    _write_parquet(root / "fold_manifest.parquet", manifest)
    _write_csv(
        root / "target_prevalence.csv",
        pd.DataFrame(
            [
                {"target": target, "prevalence": float(pd.to_numeric(
                    _concat([result.test_predictions for result in results])[target],
                    errors="raise",
                ).mean())}
                for target in PROFITABILITY_HEADS
            ]
        ),
    )
    _write_parquet(
        root / "oof_predictions.parquet",
        _concat([result.test_predictions for result in results]),
    )
    _write_parquet(
        root / "policy_predictions.parquet",
        _concat([result.policy_predictions for result in results]),
    )
    _write_parquet(
        root / "probability_calibration_predictions.parquet",
        _concat([result.calibration_predictions for result in results]),
    )
    _write_csv(
        root / "calibration_metrics.csv",
        _concat([result.calibration_metrics for result in results]),
    )
    _write_csv(
        root / "reliability_bins.csv",
        _concat([result.reliability_bins for result in results]),
    )
    _write_csv(
        root / "leakage_audit.csv",
        _concat([result.fit_audit for result in results]),
    )
    _write_csv(root / "fold_policy_selections.csv", pd.DataFrame(map(_selection_row, selections)))
    _write_csv(
        root / "policy_grid.csv",
        _concat([selection.policy_grid for selection in selections]),
    )
    _write_csv(
        root / "threshold_frontier.csv",
        _concat([selection.threshold_frontier.assign(fold_id=selection.fold_id) for selection in selections]),
    )
    _write_parquet(root / "development_activations.parquet", activations)
    _write_parquet(root / "development_funnel.parquet", funnel)
    _write_parquet(root / "development_trade_ledger.parquet", ledger)
    _write_csv(root / "development_side_metrics.csv", _ledger_group_metrics(ledger, "direction"))
    _write_csv(root / "development_fold_metrics.csv", _ledger_group_metrics(ledger, "fold_id"))
    _write_csv(root / "development_route_metrics.csv", _ledger_group_metrics(ledger, "route"))
    _write_json(root / "xgb_satellite_ablation.json", xgb_satellite_ablation(ledger))
    _write_json(root / "development_summary.json", summary)


def run_development(
    root: Path = CACHE,
    protocol: dict[str, object] | None = None,
    frozen_root: Path = FROZEN_DEVELOPMENT_CACHE,
    model_config: UnifiedModelConfig = UnifiedModelConfig(),
) -> DevelopmentRun:
    """Train/select on four inner roles and score each outer test exactly once."""
    work_dir = Path(root).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    frozen_protocol = protocol or freeze_protocol(model_config)
    _write_json(
        work_dir / "run_state.json",
        {
            "stage": "development",
            "h1_loaded": False,
            "forward_loaded": False,
            "lockbox_2026_q2_used": False,
        },
    )
    dataset, source_audit = load_frozen_development_dataset(frozen_root)
    manifest = make_four_role_manifest(dataset)
    manifest_sha256 = _frame_sha256(manifest)
    source_audit = dict(source_audit, manifest_sha256=manifest_sha256)

    results: list[ProfitabilityFoldResult] = []
    selections: list[FoldPolicySelection] = []
    activation_parts: list[pd.DataFrame] = []
    funnel_parts: list[pd.DataFrame] = []
    ledger_parts: list[pd.DataFrame] = []
    for fold_id in sorted(pd.to_numeric(manifest["fold_id"], errors="raise").unique()):
        fold_id = int(fold_id)
        expected_keys = manifest.loc[
            manifest["fold_id"].eq(fold_id) & manifest["role"].eq("test"),
            "row_key",
        ]
        identity = _checkpoint_identity(
            frozen_protocol, source_audit, manifest_sha256, expected_keys
        )
        checkpoint = (
            work_dir
            / "checkpoints"
            / "development"
            / f"fold_{fold_id}_{identity[:16]}.pkl"
        )
        result = _load_checkpoint(checkpoint, identity, expected_keys)
        if result is None:
            result = fit_profitability_fold(
                dataset, manifest, fold_id, model_config
            )
            _save_checkpoint(checkpoint, identity, result)
        selection = select_fold_policy(
            result.policy_predictions, dataset.economic_paths, fold_id
        )
        activations, funnel = apply_frozen_fold_policy(
            result.test_predictions, selection
        )
        ledger = replay_selected_paths(activations, dataset.economic_paths)
        results.append(result)
        selections.append(selection)
        activation_parts.append(activations)
        funnel_parts.append(funnel)
        ledger_parts.append(ledger)

    activations = _concat(activation_parts).sort_values(
        "decision_time", kind="stable"
    ).reset_index(drop=True)
    funnel = _concat(funnel_parts).sort_values(
        "decision_time", kind="stable"
    ).reset_index(drop=True)
    ledger = _concat(ledger_parts).sort_values("entry_time", kind="stable").reset_index(drop=True)
    summary = evaluate_development_ledger(ledger, selections)
    test_predictions = _concat([result.test_predictions for result in results])
    test_time = pd.to_datetime(test_predictions["decision_time"], utc=True)
    observed_days = int(test_time.dt.normalize().nunique())
    net = pd.to_numeric(ledger.get("net_return", pd.Series(dtype=float)), errors="raise")
    gross = pd.to_numeric(ledger.get("gross_return", pd.Series(dtype=float)), errors="raise")
    summary.update(
        {
            "phase": "development_oof",
            "oof_rows": len(test_predictions),
            "observed_days": observed_days,
            "trades_per_observed_day": len(ledger) / observed_days,
            "gross_return": float(gross.sum()),
            "cost_return": float(gross.sum() - net.sum()),
            "maximum_scored_timestamp": test_time.max().isoformat(),
            "h1_loaded": False,
            "forward_loaded": False,
            "lockbox_2026_q2_used": False,
        }
    )
    _write_development_artifacts(
        work_dir,
        frozen_protocol,
        source_audit,
        manifest,
        results,
        selections,
        activations,
        funnel,
        ledger,
        summary,
    )
    return DevelopmentRun(summary, ledger, selections, source_audit, work_dir)


def _monthly_manifest(dataset, month_start: pd.Timestamp, month_end: pd.Timestamp, fold_id: int) -> pd.DataFrame:
    decisions = dataset.decisions
    decision_time = pd.to_datetime(decisions["decision_time"], utc=True)
    label_end = pd.to_datetime(decisions["label_end"], utc=True)
    complete = decisions["path_complete"].fillna(False).to_numpy(bool)
    history = np.flatnonzero(
        complete & decision_time.lt(month_start).to_numpy(bool) & label_end.lt(month_start).to_numpy(bool)
    )
    test = np.flatnonzero(
        complete
        & decision_time.ge(month_start).to_numpy(bool)
        & decision_time.lt(month_end).to_numpy(bool)
    )
    if len(history) < 100 or not len(test):
        raise ValueError(f"insufficient causal rows for {month_start:%Y-%m}")
    probability_start = int(np.floor(len(history) * 0.70))
    policy_start = int(np.floor(len(history) * 0.85))
    if probability_start <= 8 or policy_start - probability_start <= 8 or len(history) - policy_start <= 8:
        raise ValueError("history is too short for four purged roles")
    probability_time = decision_time.iloc[history[probability_start]]
    policy_time = decision_time.iloc[history[policy_start]]
    fit = history[: probability_start - 8]
    fit = fit[label_end.iloc[fit].lt(probability_time).to_numpy(bool)]
    calibration = history[probability_start : policy_start - 8]
    calibration = calibration[label_end.iloc[calibration].lt(policy_time).to_numpy(bool)]
    policy = history[policy_start : -8]
    policy = policy[label_end.iloc[policy].lt(month_start).to_numpy(bool)]
    parts = []
    for role, positions in (
        ("fit", fit),
        ("probability_calibration", calibration),
        ("policy_selection", policy),
        ("test", test),
    ):
        if not len(positions):
            raise ValueError(f"monthly manifest has no {role} rows")
        parts.append(
            pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "position": positions,
                    "row_key": decisions.iloc[positions]["row_key"].astype(str).to_numpy(),
                    "role": role,
                }
            )
        )
    manifest = pd.concat(parts, ignore_index=True).sort_values("position", kind="stable")
    for earlier, later in zip(ROLES, ROLES[1:]):
        left = manifest.loc[manifest["role"].eq(earlier), "position"].to_numpy(int)
        right = manifest.loc[manifest["role"].eq(later), "position"].to_numpy(int)
        if not label_end.iloc[left].max() < decision_time.iloc[right].min():
            raise AssertionError(f"monthly {earlier} labels cross {later}")
    return manifest.reset_index(drop=True)


def _monthly_summary_rows(
    stage: str,
    ledger: pd.DataFrame,
    per_bar: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    entry = pd.to_datetime(ledger.get("entry_time", pd.Series(dtype="datetime64[ns, UTC]")), utc=True)
    for month_start in pd.date_range(start, end, freq="MS", inclusive="left"):
        month_end = min(month_start + pd.offsets.MonthBegin(1), end)
        month_ledger = ledger.loc[entry.ge(month_start) & entry.lt(month_end)] if len(ledger) else ledger
        month_bar = per_bar.loc[(per_bar.index >= month_start) & (per_bar.index < month_end)]
        row = summarize_candidate(month_ledger, month_bar, f"{stage}_month")
        row["month"] = month_start.strftime("%Y-%m")
        rows.append(row)
    return pd.DataFrame(rows)


def run_walk_forward(
    stage: str,
    bundle: SourceBundle,
    root: Path = CACHE,
    protocol: dict[str, object] | None = None,
    model_config: UnifiedModelConfig = UnifiedModelConfig(),
) -> StageRun:
    """Run causal monthly four-role refits over H1 or bounded forward."""
    if stage == "h1":
        stage_start, stage_end = DEVELOPMENT_END, H1_END
    elif stage == "forward":
        stage_start, stage_end = H1_END, LOCKBOX_START
    else:
        raise ValueError(f"unknown stage: {stage}")
    if bundle.end < stage_end or bundle.end > LOCKBOX_START:
        raise ValueError(f"invalid bounded source for {stage}")
    work_dir = Path(root).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    frozen_protocol = protocol or freeze_protocol(model_config)
    dataset = attach_profitability_targets(_dataset_from_bundle(bundle))
    selections: list[FoldPolicySelection] = []
    results: list[ProfitabilityFoldResult] = []
    activation_parts: list[pd.DataFrame] = []
    funnel_parts: list[pd.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for fold_id, month_start in enumerate(
        pd.date_range(stage_start, stage_end, freq="MS", inclusive="left")
    ):
        month_end = min(month_start + pd.offsets.MonthBegin(1), stage_end)
        manifest = _monthly_manifest(dataset, month_start, month_end, fold_id)
        expected_keys = manifest.loc[manifest["role"].eq("test"), "row_key"]
        stage_source = {
            "source_manifest_sha256": _canonical_json(bundle.source_identities),
            "artifact_hashes": {},
        }
        manifest_sha256 = _frame_sha256(manifest)
        identity = _checkpoint_identity(
            frozen_protocol, stage_source, manifest_sha256, expected_keys
        )
        checkpoint = (
            work_dir
            / "checkpoints"
            / stage
            / f"{month_start:%Y_%m}_{identity[:16]}.pkl"
        )
        result = _load_checkpoint(checkpoint, identity, expected_keys)
        if result is None:
            result = fit_profitability_fold(dataset, manifest, fold_id, model_config)
            _save_checkpoint(checkpoint, identity, result)
        selection = select_fold_policy(
            result.policy_predictions, dataset.economic_paths, fold_id
        )
        activations, funnel = apply_frozen_fold_policy(result.test_predictions, selection)
        results.append(result)
        selections.append(selection)
        activation_parts.append(activations)
        funnel_parts.append(funnel)
        fit_max = pd.to_datetime(
            result.fit_audit["fit_max_label_end"], utc=True
        ).max()
        if not fit_max < month_start:
            raise AssertionError("monthly fit labels cross scored month")
        audit_rows.append(
            {
                "fold_id": fold_id,
                "month_start": month_start,
                "month_end_exclusive": month_end,
                "fit_max_label_end": fit_max,
                "policy_selection_passed": selection.selection_passed,
                "scored_rows": len(result.test_predictions),
            }
        )
    predictions = _concat([result.test_predictions for result in results]).sort_values(
        "decision_time", kind="stable"
    ).reset_index(drop=True)
    activations = _concat(activation_parts).sort_values("decision_time", kind="stable").reset_index(drop=True)
    funnel = _concat(funnel_parts).sort_values("decision_time", kind="stable").reset_index(drop=True)
    ledger = replay_selected_paths(activations, dataset.economic_paths)
    bars = _bars_for_period(bundle, stage_start, stage_end)
    per_bar = ledger_to_common_per_bar(ledger, bars)
    summary = summarize_candidate(
        ledger,
        per_bar,
        "observed_development_walk_forward" if stage == "h1" else "conditional_development_forward",
    )
    summary.update(
        {
            "stage": stage,
            "maximum_scored_timestamp": (
                pd.to_datetime(predictions["decision_time"], utc=True).max().isoformat()
                if len(predictions)
                else None
            ),
            "xgb_satellite": xgb_satellite_ablation(ledger),
        }
    )
    gate_passed = h1_compatibility_gate(summary) if stage == "h1" else forward_promotion_gate(summary)
    summary["gate_passed"] = gate_passed
    _write_parquet(work_dir / f"{stage}_predictions.parquet", predictions)
    _write_parquet(work_dir / f"{stage}_activations.parquet", activations)
    _write_parquet(work_dir / f"{stage}_funnel.parquet", funnel)
    _write_parquet(work_dir / f"{stage}_trade_ledger.parquet", ledger)
    _write_parquet(
        work_dir / f"{stage}_per_bar.parquet",
        per_bar.rename("net_return").rename_axis("timestamp").reset_index(),
    )
    _write_csv(work_dir / f"{stage}_refit_audit.csv", pd.DataFrame(audit_rows))
    _write_csv(work_dir / f"{stage}_policy_selections.csv", pd.DataFrame(map(_selection_row, selections)))
    _write_csv(work_dir / f"{stage}_side_metrics.csv", _ledger_group_metrics(ledger, "direction"))
    _write_csv(work_dir / f"{stage}_route_metrics.csv", _ledger_group_metrics(ledger, "route"))
    _write_csv(work_dir / f"{stage}_monthly.csv", _monthly_summary_rows(stage, ledger, per_bar, stage_start, stage_end))
    _write_json(work_dir / f"{stage}_summary.json", summary)
    return StageRun(stage, summary, ledger, bundle.source_identities, work_dir, gate_passed, selections)


def _remove_stage_artifacts(root: Path, stage: str) -> None:
    resolved_root = Path(root).resolve()
    if not resolved_root.exists():
        return
    for path in resolved_root.glob(f"{stage}_*"):
        resolved = path.resolve()
        if resolved.parent != resolved_root:
            raise AssertionError("stage artifact escaped experiment root")
        if resolved.is_file():
            resolved.unlink()


def _union_summary(reference: object) -> object:
    return getattr(reference, "summary", None)


def _maximum_loaded_timestamp(development: DevelopmentRun, stages: list[StageRun]) -> pd.Timestamp:
    values = [pd.Timestamp(development.source_audit["maximum_decision_time"])]
    for stage in stages:
        values.extend(
            pd.Timestamp(identity["max_timestamp"])
            for identity in stage.source_audit.values()
            if identity.get("max_timestamp") is not None
        )
    maximum = max(values)
    if maximum >= LOCKBOX_START:
        raise AssertionError("experiment reached the Q2-2026 lockbox")
    return maximum


def run_experiment(
    root: Path = CACHE,
    frozen_root: Path = FROZEN_DEVELOPMENT_CACHE,
    source_paths: SourcePaths = SourcePaths(),
    model_config: UnifiedModelConfig = UnifiedModelConfig(),
) -> dict[str, object]:
    """Execute development, H1 and forward only through their frozen gates."""
    work_dir = Path(root).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol(model_config)
    union_h1 = verify_frozen_union("h1")
    development = run_development(
        work_dir, protocol, frozen_root, model_config
    )
    h1: StageRun | None = None
    forward: StageRun | None = None
    union_forward: object | None = None
    if not bool(development.summary.get("development_pass", False)):
        _remove_stage_artifacts(work_dir, "h1")
        _remove_stage_artifacts(work_dir, "forward")
        decision = "development_fail_keep_union_v1"
    else:
        h1_bundle = load_bounded_sources(DEVELOPMENT_START, H1_END, source_paths)
        h1 = run_walk_forward("h1", h1_bundle, work_dir, protocol, model_config)
        if not h1.gate_passed:
            _remove_stage_artifacts(work_dir, "forward")
            decision = "h1_fail_keep_union_v1"
        else:
            union_forward = verify_frozen_union("forward")
            forward_bundle = load_bounded_sources(
                DEVELOPMENT_START, LOCKBOX_START, source_paths
            )
            forward = run_walk_forward(
                "forward", forward_bundle, work_dir, protocol, model_config
            )
            decision = (
                "promote_side_profitability_ensemble"
                if forward.gate_passed
                else "forward_fail_keep_union_v1"
            )
    stages = [stage for stage in (h1, forward) if stage is not None]
    maximum = _maximum_loaded_timestamp(development, stages)
    summary = {
        "decision": decision,
        "development": development.summary,
        "h1": h1.summary if h1 is not None else None,
        "forward": forward.summary if forward is not None else None,
        "h1_loaded": h1 is not None,
        "h1_compatibility_passed": bool(h1 and h1.gate_passed),
        "forward_loaded": forward is not None,
        "forward_promoted": bool(forward and forward.gate_passed),
        "lockbox_2026_q2_used": False,
        "lob_used": False,
        "maximum_loaded_timestamp": maximum.isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "union_h1": _union_summary(union_h1),
        "union_forward": _union_summary(union_forward) if union_forward else None,
    }
    _write_json(work_dir / "summary.json", summary)
    _write_json(
        work_dir / "run_state.json",
        {
            "stage": "complete",
            "decision": decision,
            "h1_loaded": h1 is not None,
            "forward_loaded": forward is not None,
            "lockbox_2026_q2_used": False,
        },
    )
    union_dependencies = dict(getattr(union_h1, "dependency_hashes", {}))
    if union_forward is not None:
        union_dependencies.update(
            getattr(union_forward, "dependency_hashes", {})
        )
    manifest = {
        "protocol_sha256": protocol["protocol_sha256"],
        "frozen_development_source": development.source_audit,
        "h1_sources": h1.source_audit if h1 else None,
        "forward_sources": forward.source_audit if forward else None,
        "union_dependency_hashes": union_dependencies,
        "artifact_hashes": _artifact_hashes(work_dir),
        "maximum_loaded_timestamp": maximum.isoformat(),
        "h1_loaded": h1 is not None,
        "forward_loaded": forward is not None,
        "lockbox_2026_q2_used": False,
    }
    _write_json(work_dir / "manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CACHE)
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_DEVELOPMENT_CACHE)
    args = parser.parse_args(argv)
    summary = run_experiment(args.root, args.frozen_root)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE",
    "DevelopmentRun",
    "FROZEN_DEVELOPMENT_CACHE",
    "StageRun",
    "freeze_protocol",
    "main",
    "run_development",
    "run_experiment",
    "run_walk_forward",
]
