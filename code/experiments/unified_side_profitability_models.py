"""Cross-calibrated LONG/SHORT profitability heads for Notebook 04e."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments.unified_2021_ensemble_data import UnifiedDataset
from experiments.unified_2021_ensemble_models import (
    MODEL_NAMES,
    SigmoidCalibrator,
    UnifiedModelConfig,
    _calibration_metric_row,
    _make_model,
    _reliability_rows,
    _score_positions,
    sha256_keys,
)
from experiments.unified_side_profitability_data import PROFITABILITY_HEADS


@dataclass
class ProfitabilityFoldResult:
    test_predictions: pd.DataFrame
    policy_predictions: pd.DataFrame
    calibration_predictions: pd.DataFrame
    calibration_metrics: pd.DataFrame
    reliability_bins: pd.DataFrame
    fit_audit: pd.DataFrame


def _target(decisions: pd.DataFrame, head: str) -> np.ndarray:
    if head not in PROFITABILITY_HEADS:
        raise KeyError(f"unknown profitability head: {head}")
    values = pd.to_numeric(decisions[head], errors="raise").to_numpy(np.int64)
    if not np.isin(values, (0, 1)).all():
        raise ValueError(f"{head} must be binary")
    return values


def _role_positions(fold: pd.DataFrame, role: str) -> np.ndarray:
    positions = fold.loc[fold["role"].eq(role), "position"].to_numpy(np.int64)
    if not len(positions):
        raise ValueError(f"fold has no {role} rows")
    return positions


def _discrimination_metrics(
    target: np.ndarray,
    raw: np.ndarray,
    probability: np.ndarray,
) -> dict[str, float]:
    if np.unique(target).size < 2:
        return {
            "raw_roc_auc": float("nan"),
            "calibrated_roc_auc": float("nan"),
            "raw_pr_auc": float("nan"),
            "calibrated_pr_auc": float("nan"),
        }
    return {
        "raw_roc_auc": float(roc_auc_score(target, raw)),
        "calibrated_roc_auc": float(roc_auc_score(target, probability)),
        "raw_pr_auc": float(average_precision_score(target, raw)),
        "calibrated_pr_auc": float(average_precision_score(target, probability)),
    }


def _score_role(
    dataset: UnifiedDataset,
    positions: np.ndarray,
    history_start: int,
    models: dict[tuple[str, str], object],
    calibrators: dict[tuple[str, str], SigmoidCalibrator],
    config: UnifiedModelConfig,
    fold_id: int,
    source_role: str,
) -> pd.DataFrame:
    output = dataset.decisions.iloc[positions].copy().reset_index(drop=True)
    output.insert(0, "source_role", source_role)
    output.insert(0, "fold_id", fold_id)
    for head in PROFITABILITY_HEADS:
        side = "long" if head == "long_profitable" else "short"
        for model_name in MODEL_NAMES:
            raw = _score_positions(
                models[(model_name, head)],
                dataset,
                positions,
                history_start,
                config.sequence_length,
            )
            output[f"raw_{side}_{model_name}"] = raw
            output[f"p_{side}_{model_name}"] = calibrators[
                (model_name, head)
            ].predict_proba(raw)
    return output


def fit_profitability_fold(
    dataset: UnifiedDataset,
    manifest: pd.DataFrame,
    fold_id: int,
    config: UnifiedModelConfig = UnifiedModelConfig(),
) -> ProfitabilityFoldResult:
    """Fit six raw heads, calibrate later, and score policy/test once."""
    fold = manifest.loc[manifest["fold_id"].eq(fold_id)].sort_values(
        "position", kind="stable"
    )
    if fold.empty:
        raise ValueError(f"manifest has no fold {fold_id}")
    expected = dataset.decisions.iloc[fold["position"].to_numpy(np.int64)][
        "row_key"
    ].astype(str).to_numpy()
    if not np.array_equal(expected, fold["row_key"].astype(str).to_numpy()):
        raise AssertionError("manifest positions do not match dataset keys")

    fit_positions = _role_positions(fold, "fit")
    calibration_positions = _role_positions(fold, "probability_calibration")
    policy_positions = _role_positions(fold, "policy_selection")
    test_positions = _role_positions(fold, "test")
    history_start = int(fold["position"].min())
    history_stop = int(fit_positions.max()) + 1
    history_positions = np.arange(history_start, history_stop, dtype=np.int64)
    fit_mask = np.isin(history_positions, fit_positions)
    if not fit_mask.any():
        raise ValueError("fold has no fit rows in its history block")

    decisions = dataset.decisions
    models: dict[tuple[str, str], object] = {}
    audit_rows: list[dict[str, object]] = []
    fit_keys = decisions.iloc[fit_positions]["row_key"].astype(str)
    later_key_sets = {
        "probability_calibration": set(
            decisions.iloc[calibration_positions]["row_key"].astype(str)
        ),
        "policy_selection": set(
            decisions.iloc[policy_positions]["row_key"].astype(str)
        ),
        "test": set(decisions.iloc[test_positions]["row_key"].astype(str)),
    }
    for head in PROFITABILITY_HEADS:
        target = _target(decisions, head)
        for model_name in MODEL_NAMES:
            model = _make_model(model_name, config)
            model.fit(
                dataset.tabular[history_positions],
                target[history_positions],
                sample_mask=fit_mask,
            )
            models[(model_name, head)] = model
            fit_key_set = set(fit_keys)
            audit_rows.append(
                {
                    "fold_id": fold_id,
                    "model": model_name,
                    "head": head,
                    "fit_rows": len(fit_positions),
                    "fit_keys_sha256": sha256_keys(fit_keys),
                    "fit_min_decision_time": decisions.iloc[fit_positions][
                        "decision_time"
                    ].min(),
                    "fit_max_label_end": pd.to_datetime(
                        decisions.iloc[fit_positions]["label_end"], utc=True
                    ).max(),
                    "probability_calibration_overlap": bool(
                        fit_key_set.intersection(
                            later_key_sets["probability_calibration"]
                        )
                    ),
                    "policy_selection_overlap": bool(
                        fit_key_set.intersection(later_key_sets["policy_selection"])
                    ),
                    "test_overlap": bool(
                        fit_key_set.intersection(later_key_sets["test"])
                    ),
                }
            )

    calibrators: dict[tuple[str, str], SigmoidCalibrator] = {}
    calibration = dataset.decisions.iloc[calibration_positions].copy().reset_index(
        drop=True
    )
    calibration.insert(0, "source_role", "probability_calibration")
    calibration.insert(0, "fold_id", fold_id)
    metric_rows: list[dict[str, object]] = []
    reliability_rows: list[dict[str, object]] = []
    for head in PROFITABILITY_HEADS:
        side = "long" if head == "long_profitable" else "short"
        target = _target(decisions, head)[calibration_positions]
        for model_name in MODEL_NAMES:
            raw = _score_positions(
                models[(model_name, head)],
                dataset,
                calibration_positions,
                history_start,
                config.sequence_length,
            )
            calibrator = SigmoidCalibrator().fit(raw, target)
            probability = calibrator.predict_proba(raw)
            calibrators[(model_name, head)] = calibrator
            calibration[f"raw_{side}_{model_name}"] = raw
            calibration[f"p_{side}_{model_name}"] = probability
            metric_row = _calibration_metric_row(
                    fold_id,
                    model_name,
                    head,
                    target,
                    raw,
                    probability,
                )
            metric_row.update(_discrimination_metrics(target, raw, probability))
            metric_rows.append(
                metric_row
            )
            reliability_rows.extend(
                _reliability_rows(
                    fold_id,
                    model_name,
                    head,
                    target,
                    probability,
                )
            )

    policy = _score_role(
        dataset,
        policy_positions,
        history_start,
        models,
        calibrators,
        config,
        fold_id,
        "policy_selection",
    )
    test = _score_role(
        dataset,
        test_positions,
        history_start,
        models,
        calibrators,
        config,
        fold_id,
        "test",
    )
    return ProfitabilityFoldResult(
        test_predictions=test,
        policy_predictions=policy,
        calibration_predictions=calibration,
        calibration_metrics=pd.DataFrame(metric_rows),
        reliability_bins=pd.DataFrame(reliability_rows),
        fit_audit=pd.DataFrame(audit_rows),
    )


__all__ = ["ProfitabilityFoldResult", "fit_profitability_fold"]
