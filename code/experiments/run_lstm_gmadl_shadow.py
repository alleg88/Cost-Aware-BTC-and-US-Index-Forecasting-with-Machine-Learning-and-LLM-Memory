"""Run Notebook 04g, a paired development-only GMADL LSTM replacement."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.lstm_gmadl_shadow import (
    GMADL_ALPHA,
    GMADL_BETA,
    GMADL_LAMBDA,
    evaluate_shadow_admission,
    fit_paired_lstm_shadow,
)
from experiments.run_unified_2021_ensemble import (
    _artifact_hashes,
    _sha256_file,
    _write_csv,
    _write_json,
    _write_parquet,
)
from experiments.run_unified_expected_net_ensemble import (
    CACHE as CONTROL_CACHE,
    FROZEN_DEVELOPMENT_CACHE,
    _canonical_json,
    _development_audit,
    _ledger_group_metrics,
)
from experiments.unified_2021_ensemble_models import UnifiedModelConfig, sha256_keys
from experiments.unified_2021_ensemble_policy import replay_selected_paths
from experiments.unified_expected_net_data import (
    EXPECTED_NET_TARGETS,
    load_frozen_expected_net_dataset,
)
from experiments.unified_expected_net_models import (
    NonNegativeAffineCalibrator,
    _score_lstm_positions,
    shared_target_scale,
)
from experiments.unified_expected_net_policy import (
    evaluate_expected_net_development,
    fixed_policy_activations,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
CACHE = CODE_ROOT / "experiments" / "cache" / "lstm_gmadl_shadow"
MODEL_NAMES = ("xgboost", "lstm", "svm_linear")
LSTM_COLUMNS = (
    "raw_long_lstm",
    "raw_short_lstm",
    "pred_long_lstm",
    "pred_short_lstm",
)


@dataclass
class ShadowFoldResult:
    control_predictions: pd.DataFrame
    candidate_predictions: pd.DataFrame
    calibration_metrics: pd.DataFrame
    paired_training_audit: dict[str, object]
    side_choice_audit: pd.DataFrame
    fit_audit: pd.DataFrame
    target_scale_bps: float


def freeze_protocol(
    model_config: UnifiedModelConfig = UnifiedModelConfig(),
) -> dict[str, object]:
    """Freeze the sole paired seed/loss replacement before reading outcomes."""
    implementation_files = (
        "lstm_gmadl_shadow.py",
        "run_lstm_gmadl_shadow.py",
    )
    payload: dict[str, object] = {
        "protocol_version": "lstm-gmadl-shadow-v1",
        "development_only": True,
        "control_loss": "mse_long_plus_mse_short",
        "candidate_loss": "control_plus_0.25_gmadl",
        "gmadl": {
            "alpha": GMADL_ALPHA,
            "beta": GMADL_BETA,
            "lambda": GMADL_LAMBDA,
        },
        "seeds": [model_config.seed],
        "model_config": asdict(model_config),
        "lstm_role": "replacement_not_fourth_vote",
        "reuse_04f_xgboost_svm": True,
        "same_manifest_features_rows_scale_state_order_optimizer_epochs_clip": True,
        "separate_later_affine_calibration_by_arm_and_side": True,
        "policy": "exact_04f_two_of_three_positive_expected_net",
        "sweep": [],
        "h1_access_allowed": False,
        "forward_access_allowed": False,
        "lockbox_2026_q2_used": False,
        "shadow_only": True,
        "implementation_hashes": {
            filename: _sha256_file(CODE_ROOT / "experiments" / filename)
            for filename in implementation_files
        },
    }
    payload["protocol_sha256"] = hashlib.sha256(
        _canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return payload


def replace_lstm_predictions(
    base: pd.DataFrame,
    replacement: pd.DataFrame,
    *,
    arm: str,
) -> pd.DataFrame:
    """Replace exactly one LSTM LONG/SHORT pair while preserving XGB/SVM."""
    if arm not in {"control", "candidate"}:
        raise ValueError("arm must be control or candidate")
    required = {"row_key", *LSTM_COLUMNS}
    missing = sorted(required.difference(replacement.columns))
    if missing:
        raise ValueError(f"LSTM replacement lacks columns: {missing}")
    if base["row_key"].astype(str).duplicated().any() or replacement[
        "row_key"
    ].astype(str).duplicated().any():
        raise ValueError("replacement keys must be unique")
    base_keys = base["row_key"].astype(str).tolist()
    replacement_index = replacement.assign(
        row_key=replacement["row_key"].astype(str)
    ).set_index("row_key")
    if set(base_keys) != set(replacement_index.index):
        raise AssertionError("replacement and base keys differ")
    ordered = replacement_index.loc[base_keys]
    output = base.copy().reset_index(drop=True)
    for column in LSTM_COLUMNS:
        values = pd.to_numeric(ordered[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"{arm} {column} must be finite")
        output[column] = values
    output["lstm_arm"] = arm
    if any(
        column.startswith(("pred_long_lstm_", "pred_short_lstm_"))
        for column in output.columns
    ):
        raise AssertionError("both LSTM arms escaped into one policy frame")
    return output


def _verify_control_dependencies(root: Path) -> dict[str, object]:
    control_root = Path(root).resolve()
    manifest_path = control_root / "manifest.json"
    summary_path = control_root / "summary.json"
    protocol_path = control_root / "protocol.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if manifest.get("lockbox_2026_q2_used") or summary.get("lockbox_2026_q2_used"):
        raise AssertionError("04f dependency reports Q2-2026 lockbox use")
    verified = {}
    for filename, expected in manifest["artifact_hashes"].items():
        path = (control_root / filename).resolve()
        if path.parent != control_root or not path.is_file():
            raise AssertionError(f"04f dependency is missing: {filename}")
        actual = _sha256_file(path)
        if actual != expected:
            raise AssertionError(f"04f dependency hash mismatch: {filename}")
        verified[filename] = actual
    if manifest["protocol_sha256"] != summary["protocol_sha256"]:
        raise AssertionError("04f protocol and summary hashes differ")
    if protocol["protocol_sha256"] != summary["protocol_sha256"]:
        raise AssertionError("04f frozen protocol does not match its summary")
    return {
        "control_root": str(control_root),
        "control_manifest_sha256": _sha256_file(manifest_path),
        "control_summary_sha256": _sha256_file(summary_path),
        "control_protocol_sha256": protocol["protocol_sha256"],
        "verified_artifacts": len(verified),
        "artifact_hashes": verified,
        "control_summary": summary,
        "control_manifest": manifest,
    }


def _role_positions(fold: pd.DataFrame, role: str) -> np.ndarray:
    positions = fold.loc[fold["role"].eq(role), "position"].to_numpy(np.int64)
    if not len(positions):
        raise ValueError(f"fold has no {role} rows")
    return positions


def _ordered_control_rows(
    control_oof: pd.DataFrame,
    row_keys: Iterable[object],
) -> pd.DataFrame:
    expected = [str(key) for key in row_keys]
    indexed = control_oof.assign(
        row_key=control_oof["row_key"].astype(str)
    ).set_index("row_key")
    if len(expected) != len(set(expected)) or not set(expected).issubset(indexed.index):
        raise AssertionError("04f OOF rows do not cover the requested fold keys")
    return indexed.loc[expected].reset_index()


def _manifest_matches_dataset(dataset, manifest: pd.DataFrame) -> bool:
    """Validate frozen row content without depending on Parquet dtype round-trips."""
    required = {"fold_id", "position", "row_key", "role"}
    if not required.issubset(manifest.columns) or manifest.empty:
        return False
    try:
        positions = pd.to_numeric(manifest["position"], errors="raise").to_numpy(
            np.int64
        )
    except (TypeError, ValueError):
        return False
    if (positions < 0).any() or (positions >= len(dataset.decisions)).any():
        return False
    expected = dataset.decisions.iloc[positions]["row_key"].astype(str).to_numpy()
    actual = manifest["row_key"].astype(str).to_numpy()
    return bool(
        np.array_equal(expected, actual)
        and manifest["fold_id"].notna().all()
        and manifest["role"].notna().all()
    )


def _calibration_metric(
    fold_id: int,
    arm: str,
    side: str,
    target: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    calibrator: NonNegativeAffineCalibrator,
) -> dict[str, object]:
    raw_error = raw - target
    calibrated_error = calibrated - target
    return {
        "fold_id": fold_id,
        "arm": arm,
        "side": side,
        "rows": len(target),
        "raw_mae_bps": float(np.mean(np.abs(raw_error))),
        "calibrated_mae_bps": float(np.mean(np.abs(calibrated_error))),
        "raw_rmse_bps": float(np.sqrt(np.mean(raw_error * raw_error))),
        "calibrated_rmse_bps": float(
            np.sqrt(np.mean(calibrated_error * calibrated_error))
        ),
        "slope": calibrator.slope,
        "intercept_bps": calibrator.intercept,
    }


def _value_weighted_side_choice(
    truth: np.ndarray,
    predictions: np.ndarray,
    threshold_bps: float,
) -> tuple[float, float, float, int]:
    true_spread = truth[:, 0] - truth[:, 1]
    predicted_spread = predictions[:, 0] - predictions[:, 1]
    eligible = np.abs(true_spread) >= threshold_bps
    eligible &= np.abs(true_spread) > np.finfo(float).eps
    weight = np.abs(true_spread[eligible])
    if not len(weight) or float(weight.sum()) <= 0.0:
        return float("nan"), 0.0, 0.0, 0
    correct = np.sign(predicted_spread[eligible]) == np.sign(true_spread[eligible])
    numerator = float(np.sum(weight * correct))
    denominator = float(weight.sum())
    return numerator / denominator, numerator, denominator, int(eligible.sum())


def _fit_shadow_fold(
    dataset,
    manifest: pd.DataFrame,
    control_oof: pd.DataFrame,
    fold_id: int,
    config: UnifiedModelConfig,
) -> ShadowFoldResult:
    fold = manifest.loc[manifest["fold_id"].eq(fold_id)].sort_values(
        "position", kind="stable"
    )
    if fold.empty:
        raise ValueError(f"manifest has no fold {fold_id}")
    fit_positions = _role_positions(fold, "fit")
    calibration_positions = _role_positions(fold, "probability_calibration")
    preflight_positions = _role_positions(fold, "fixed_policy_preflight")
    test_positions = _role_positions(fold, "test")
    history_start = int(fold["position"].min())
    history_stop = int(fit_positions.max()) + 1
    history_positions = np.arange(history_start, history_stop, dtype=np.int64)
    fit_mask = np.isin(history_positions, fit_positions)
    target = dataset.decisions.loc[:, list(EXPECTED_NET_TARGETS)].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy(float)
    target_scale_bps = shared_target_scale(
        target[fit_positions, 0], target[fit_positions, 1]
    )
    scaled_target = target / target_scale_bps
    paired = fit_paired_lstm_shadow(
        dataset.tabular[history_positions],
        scaled_target[history_positions],
        fit_mask,
        config=config,
    )

    raw_calibration = {
        "control": _score_lstm_positions(
            paired.control, dataset, calibration_positions, history_start
        )
        * target_scale_bps,
        "candidate": _score_lstm_positions(
            paired.candidate, dataset, calibration_positions, history_start
        )
        * target_scale_bps,
    }
    calibrators: dict[tuple[str, str], NonNegativeAffineCalibrator] = {}
    metric_rows: list[dict[str, object]] = []
    for arm in ("control", "candidate"):
        for side_index, side in enumerate(("long", "short")):
            truth = target[calibration_positions, side_index]
            raw = raw_calibration[arm][:, side_index]
            calibrator = NonNegativeAffineCalibrator.fit(raw, truth)
            calibrated = calibrator.predict(raw)
            calibrators[(arm, side)] = calibrator
            metric_rows.append(
                _calibration_metric(
                    fold_id,
                    arm,
                    side,
                    truth,
                    raw,
                    calibrated,
                    calibrator,
                )
            )

    base = _ordered_control_rows(
        control_oof,
        dataset.decisions.iloc[test_positions]["row_key"],
    )
    arm_frames: dict[str, pd.DataFrame] = {}
    side_choice_rows: list[dict[str, object]] = []
    fit_spread = target[fit_positions, 0] - target[fit_positions, 1]
    top_quartile_threshold = float(np.quantile(np.abs(fit_spread), 0.75))
    for arm, model in (("control", paired.control), ("candidate", paired.candidate)):
        raw = (
            _score_lstm_positions(model, dataset, test_positions, history_start)
            * target_scale_bps
        )
        replacement = pd.DataFrame(
            {
                "row_key": dataset.decisions.iloc[test_positions]["row_key"].astype(str),
                "raw_long_lstm": raw[:, 0],
                "raw_short_lstm": raw[:, 1],
                "pred_long_lstm": calibrators[(arm, "long")].predict(raw[:, 0]),
                "pred_short_lstm": calibrators[(arm, "short")].predict(raw[:, 1]),
            }
        )
        frame = replace_lstm_predictions(base, replacement, arm=arm)
        arm_frames[arm] = frame
        predicted = frame[["pred_long_lstm", "pred_short_lstm"]].to_numpy(float)
        accuracy, numerator, denominator, rows = _value_weighted_side_choice(
            target[test_positions], predicted, top_quartile_threshold
        )
        side_choice_rows.append(
            {
                "fold_id": fold_id,
                "arm": arm,
                "fit_top_quartile_threshold_bps": top_quartile_threshold,
                "test_rows": rows,
                "weighted_correct": numerator,
                "weight_sum": denominator,
                "value_weighted_accuracy": accuracy,
            }
        )

    control_delta = np.max(
        np.abs(
            arm_frames["control"].loc[:, list(LSTM_COLUMNS)].to_numpy(float)
            - base.loc[:, list(LSTM_COLUMNS)].to_numpy(float)
        )
    )
    control_matches = bool(control_delta <= 1e-6)
    fit_keys = set(dataset.decisions.iloc[fit_positions]["row_key"].astype(str))
    later = {
        "probability_calibration": set(
            dataset.decisions.iloc[calibration_positions]["row_key"].astype(str)
        ),
        "fixed_policy_preflight": set(
            dataset.decisions.iloc[preflight_positions]["row_key"].astype(str)
        ),
        "test": set(dataset.decisions.iloc[test_positions]["row_key"].astype(str)),
    }
    fit_audit = pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "model": "lstm",
                "target": target_name,
                "fit_rows": len(fit_positions),
                "fit_keys_sha256": sha256_keys(
                    dataset.decisions.iloc[fit_positions]["row_key"]
                ),
                "target_scale_bps": target_scale_bps,
                "probability_calibration_overlap": bool(
                    fit_keys.intersection(later["probability_calibration"])
                ),
                "fixed_policy_preflight_overlap": bool(
                    fit_keys.intersection(later["fixed_policy_preflight"])
                ),
                "test_overlap": bool(fit_keys.intersection(later["test"])),
            }
            for target_name in EXPECTED_NET_TARGETS
        ]
    )
    training_audit = {
        "fold_id": fold_id,
        "seed": paired.seed,
        "initial_state_sha256": paired.initial_state_sha256,
        "control_initial_state_sha256": paired.control.training_audit[
            "initial_state_sha256"
        ],
        "candidate_initial_state_sha256": paired.candidate.training_audit[
            "initial_state_sha256"
        ],
        "batch_order_sha256": paired.batch_order_sha256,
        "control_batch_order_sha256": paired.control.training_audit[
            "batch_order_sha256"
        ],
        "candidate_batch_order_sha256": paired.candidate.training_audit[
            "batch_order_sha256"
        ],
        "initial_state_match": (
            paired.control.training_audit["initial_state_sha256"]
            == paired.candidate.training_audit["initial_state_sha256"]
            == paired.initial_state_sha256
        ),
        "batch_order_match": (
            paired.control.training_audit["batch_order_sha256"]
            == paired.candidate.training_audit["batch_order_sha256"]
            == paired.batch_order_sha256
        ),
        "target_scale_bps": target_scale_bps,
        "alpha": paired.alpha,
        "beta": paired.beta,
        "lambda_gmadl": paired.lambda_gmadl,
        "control_max_abs_delta_from_04f": float(control_delta),
        "control_matches_04f": control_matches,
    }
    return ShadowFoldResult(
        control_predictions=arm_frames["control"],
        candidate_predictions=arm_frames["candidate"],
        calibration_metrics=pd.DataFrame(metric_rows),
        paired_training_audit=training_audit,
        side_choice_audit=pd.DataFrame(side_choice_rows),
        fit_audit=fit_audit,
        target_scale_bps=target_scale_bps,
    )


def _checkpoint_identity(
    protocol: dict[str, object],
    dependency_audit: dict[str, object],
    manifest_sha256: str,
    outer_keys: Iterable[object],
) -> str:
    payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "control_manifest_sha256": dependency_audit["control_manifest_sha256"],
        "control_protocol_sha256": dependency_audit["control_protocol_sha256"],
        "manifest_sha256": manifest_sha256,
        "outer_keys_sha256": sha256_keys(outer_keys),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _save_checkpoint(path: Path, identity: str, result: ShadowFoldResult) -> None:
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
    expected_keys: Iterable[object],
) -> ShadowFoldResult | None:
    if not path.is_file():
        return None
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    result = payload.get("result")
    if payload.get("identity") != identity or not isinstance(result, ShadowFoldResult):
        return None
    if sha256_keys(result.control_predictions["row_key"]) != sha256_keys(expected_keys):
        return None
    if sha256_keys(result.candidate_predictions["row_key"]) != sha256_keys(
        expected_keys
    ):
        return None
    return result


def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
    useful = [frame for frame in frames if not frame.empty]
    return pd.concat(useful, ignore_index=True) if useful else pd.DataFrame()


def _arm_ledger(
    results: list[ShadowFoldResult],
    arm: str,
    economic_paths: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    activation_parts = []
    funnel_parts = []
    for result in results:
        predictions = getattr(result, f"{arm}_predictions")
        activations, funnel = fixed_policy_activations(predictions)
        activation_parts.append(activations)
        funnel_parts.append(funnel)
    activations = _concat(activation_parts)
    if not activations.empty:
        activations = activations.sort_values(
            "decision_time", kind="stable"
        ).reset_index(drop=True)
    funnel = _concat(funnel_parts)
    if not funnel.empty:
        funnel = funnel.sort_values("decision_time", kind="stable").reset_index(
            drop=True
        )
    ledger = replay_selected_paths(activations, economic_paths)
    return ledger, activations, funnel


def _complete_summary(
    ledger: pd.DataFrame,
    results: list[ShadowFoldResult],
    economic_paths: pd.DataFrame,
    observed_days: int,
) -> dict[str, object]:
    audit = _development_audit(ledger, results, economic_paths)
    summary = evaluate_expected_net_development(ledger, audit)
    gross = pd.to_numeric(
        ledger.get("gross_return", pd.Series(dtype=float)), errors="raise"
    )
    net = pd.to_numeric(
        ledger.get("net_return", pd.Series(dtype=float)), errors="raise"
    )
    summary.update(
        {
            "observed_days": observed_days,
            "trades_per_observed_day": len(ledger) / observed_days,
            "gross_return": float(gross.sum()),
            "cost_return": float(gross.sum() - net.sum()),
            **audit,
        }
    )
    return summary


def _fold_deltas(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for fold_id in range(5):
        control_net = float(
            pd.to_numeric(
                control.loc[control["fold_id"].eq(fold_id), "net_return"],
                errors="raise",
            ).sum()
        )
        candidate_net = float(
            pd.to_numeric(
                candidate.loc[candidate["fold_id"].eq(fold_id), "net_return"],
                errors="raise",
            ).sum()
        )
        rows.append(
            {
                "fold_id": fold_id,
                "control_net_return": control_net,
                "candidate_net_return": candidate_net,
                "net_delta": candidate_net - control_net,
            }
        )
    return pd.DataFrame(rows)


def _aggregate_side_choice(audit: pd.DataFrame, arm: str) -> float:
    selected = audit.loc[audit["arm"].eq(arm)]
    numerator = float(pd.to_numeric(selected["weighted_correct"], errors="raise").sum())
    denominator = float(pd.to_numeric(selected["weight_sum"], errors="raise").sum())
    if denominator <= 0.0:
        raise AssertionError(f"{arm} top-quartile diagnostic has no weight")
    return numerator / denominator


def run_experiment(
    root: Path = CACHE,
    control_root: Path = CONTROL_CACHE,
    frozen_root: Path = FROZEN_DEVELOPMENT_CACHE,
    model_config: UnifiedModelConfig = UnifiedModelConfig(),
) -> dict[str, object]:
    """Run one paired seed on 04f development rows and never access later data."""
    work_dir = Path(root).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    protocol = freeze_protocol(model_config)
    _write_json(
        work_dir / "run_state.json",
        {
            "stage": "development_shadow",
            "h1_loaded": False,
            "forward_loaded": False,
            "lockbox_2026_q2_used": False,
        },
    )
    dependency_audit = _verify_control_dependencies(control_root)
    control_path = Path(control_root).resolve()
    dataset, source_audit = load_frozen_expected_net_dataset(frozen_root)
    manifest = pd.read_parquet(control_path / "fold_manifest.parquet")
    if not _manifest_matches_dataset(dataset, manifest):
        raise AssertionError("04g manifest rows differ from the frozen dataset")
    manifest_sha256 = dependency_audit["artifact_hashes"]["fold_manifest.parquet"]
    control_oof = pd.read_parquet(control_path / "oof_predictions.parquet")
    expected_test_keys = manifest.loc[manifest["role"].eq("test"), "row_key"]
    if set(control_oof["row_key"].astype(str)) != set(expected_test_keys.astype(str)):
        raise AssertionError("04f OOF predictions and manifest test keys differ")

    results: list[ShadowFoldResult] = []
    fold_ids = sorted(pd.to_numeric(manifest["fold_id"], errors="raise").unique())
    for fold_id_value in fold_ids:
        fold_id = int(fold_id_value)
        outer_keys = manifest.loc[
            manifest["fold_id"].eq(fold_id) & manifest["role"].eq("test"),
            "row_key",
        ]
        identity = _checkpoint_identity(
            protocol, dependency_audit, manifest_sha256, outer_keys
        )
        checkpoint = (
            work_dir
            / "checkpoints"
            / "development"
            / f"fold_{fold_id}_{identity[:16]}.pkl"
        )
        result = _load_checkpoint(checkpoint, identity, outer_keys)
        if result is None:
            print(f"04g paired fold {fold_id + 1}/{len(fold_ids)}: fitting", flush=True)
            result = _fit_shadow_fold(
                dataset, manifest, control_oof, fold_id, model_config
            )
            _save_checkpoint(checkpoint, identity, result)
        else:
            print(f"04g paired fold {fold_id + 1}/{len(fold_ids)}: checkpoint", flush=True)
        results.append(result)

    training_audit = pd.DataFrame(
        [result.paired_training_audit for result in results]
    )
    if not training_audit["initial_state_match"].astype(bool).all():
        raise AssertionError("paired initial states differ")
    if not training_audit["batch_order_match"].astype(bool).all():
        raise AssertionError("paired batch orders differ")
    if not training_audit["control_matches_04f"].astype(bool).all():
        raise AssertionError("paired control failed to reproduce 04f LSTM predictions")

    control_predictions = _concat(
        [result.control_predictions for result in results]
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)
    candidate_predictions = _concat(
        [result.candidate_predictions for result in results]
    ).sort_values("decision_time", kind="stable").reset_index(drop=True)
    control_ledger, control_activations, control_funnel = _arm_ledger(
        results, "control", dataset.economic_paths
    )
    candidate_ledger, candidate_activations, candidate_funnel = _arm_ledger(
        results, "candidate", dataset.economic_paths
    )
    test_time = pd.to_datetime(control_predictions["decision_time"], utc=True)
    observed_days = int(test_time.dt.normalize().nunique())
    control_summary = _complete_summary(
        control_ledger, results, dataset.economic_paths, observed_days
    )
    candidate_summary = _complete_summary(
        candidate_ledger, results, dataset.economic_paths, observed_days
    )
    frozen_control_summary = dependency_audit["control_summary"]["development"]
    control_reconciles = bool(
        len(control_ledger) == int(frozen_control_summary["trades"])
        and np.isclose(
            float(control_ledger["net_return"].sum()),
            float(frozen_control_summary["net_return"]),
            rtol=0.0,
            atol=1e-12,
        )
    )
    if not control_reconciles:
        raise AssertionError("paired control ledger does not reproduce 04f")

    fold_deltas = _fold_deltas(control_ledger, candidate_ledger)
    side_choice = _concat([result.side_choice_audit for result in results])
    control_accuracy = _aggregate_side_choice(side_choice, "control")
    candidate_accuracy = _aggregate_side_choice(side_choice, "candidate")
    admission = evaluate_shadow_admission(
        frozen_control_summary,
        candidate_summary,
        fold_deltas,
        control_top_quartile_accuracy=control_accuracy,
        candidate_top_quartile_accuracy=candidate_accuracy,
    )
    decision = (
        "shadow_admit_for_reflection"
        if admission["shadow_admit_for_reflection"]
        else "shadow_reject_keep_standard_lstm_and_union_v1"
    )
    summary = {
        "decision": decision,
        "control": control_summary,
        "candidate": candidate_summary,
        "admission": admission,
        "control_reconciles_04f": control_reconciles,
        "shadow_only": True,
        "h1_loaded": False,
        "forward_loaded": False,
        "lockbox_2026_q2_used": False,
        "maximum_loaded_timestamp": source_audit["maximum_decision_time"],
        "maximum_scored_timestamp": test_time.max().isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "control_protocol_sha256": dependency_audit["control_protocol_sha256"],
    }

    _write_json(work_dir / "protocol.json", protocol)
    _write_json(work_dir / "dependency_audit.json", dependency_audit)
    _write_json(work_dir / "source_audit.json", source_audit)
    _write_parquet(work_dir / "fold_manifest.parquet", manifest)
    _write_parquet(work_dir / "control_predictions.parquet", control_predictions)
    _write_parquet(work_dir / "candidate_predictions.parquet", candidate_predictions)
    _write_csv(
        work_dir / "calibration_metrics.csv",
        _concat([result.calibration_metrics for result in results]),
    )
    _write_csv(work_dir / "paired_training_audit.csv", training_audit)
    _write_csv(work_dir / "side_choice_audit.csv", side_choice)
    _write_parquet(work_dir / "control_activations.parquet", control_activations)
    _write_parquet(work_dir / "candidate_activations.parquet", candidate_activations)
    _write_parquet(work_dir / "control_funnel.parquet", control_funnel)
    _write_parquet(work_dir / "candidate_funnel.parquet", candidate_funnel)
    _write_parquet(work_dir / "control_trade_ledger.parquet", control_ledger)
    _write_parquet(work_dir / "candidate_trade_ledger.parquet", candidate_ledger)
    _write_csv(work_dir / "fold_deltas.csv", fold_deltas)
    _write_csv(
        work_dir / "control_side_metrics.csv",
        _ledger_group_metrics(control_ledger, "direction"),
    )
    _write_csv(
        work_dir / "candidate_side_metrics.csv",
        _ledger_group_metrics(candidate_ledger, "direction"),
    )
    _write_csv(
        work_dir / "control_route_metrics.csv",
        _ledger_group_metrics(control_ledger, "route"),
    )
    _write_csv(
        work_dir / "candidate_route_metrics.csv",
        _ledger_group_metrics(candidate_ledger, "route"),
    )
    _write_json(work_dir / "control_summary.json", control_summary)
    _write_json(work_dir / "candidate_summary.json", candidate_summary)
    _write_json(work_dir / "summary.json", summary)
    _write_json(
        work_dir / "run_state.json",
        {
            "stage": "complete",
            "decision": decision,
            "shadow_only": True,
            "h1_loaded": False,
            "forward_loaded": False,
            "lockbox_2026_q2_used": False,
        },
    )
    manifest_payload = {
        "protocol_sha256": protocol["protocol_sha256"],
        "control_protocol_sha256": dependency_audit["control_protocol_sha256"],
        "control_dependency_hashes": {
            "manifest.json": dependency_audit["control_manifest_sha256"],
            "summary.json": dependency_audit["control_summary_sha256"],
            **dependency_audit["artifact_hashes"],
        },
        "artifact_hashes": _artifact_hashes(work_dir),
        "maximum_loaded_timestamp": source_audit["maximum_decision_time"],
        "shadow_only": True,
        "h1_loaded": False,
        "forward_loaded": False,
        "lockbox_2026_q2_used": False,
    }
    _write_json(work_dir / "manifest.json", manifest_payload)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=CACHE)
    parser.add_argument("--control-root", type=Path, default=CONTROL_CACHE)
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_DEVELOPMENT_CACHE)
    args = parser.parse_args(argv)
    summary = run_experiment(args.root, args.control_root, args.frozen_root)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CACHE",
    "CONTROL_CACHE",
    "ShadowFoldResult",
    "freeze_protocol",
    "main",
    "replace_lstm_predictions",
    "run_experiment",
]
