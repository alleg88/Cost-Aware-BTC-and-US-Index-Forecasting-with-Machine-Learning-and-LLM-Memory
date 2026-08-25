"""All-nine-model Logistic Regression stacking for Notebooks 03 and 03a.

Notebook 03 owns 2024-OOF/H1 construction and policy selection. Notebook 03a
reports the unchanged July-2025-to-March-2026 forward replay.  The 2026-Q2
lockbox is never accessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from experiments.all_model_sentiment_raw import ARMS, prepare_arm
from experiments.all_model_sentiment_scoreboard import ARM_LABELS
from experiments.baseline_model_zoo_1m import FORWARD_END, FORWARD_START
from experiments.catboost_execution_scoring import simulate_policy
from experiments.catboost_matched_ablation import economic_ranking_key
from experiments.correlation_ensemble import (
    FEE_BPS,
    LOOKBACK_DAYS,
    PROBABILITY_COLUMNS,
    _feature_matrix,
    _fit_stack,
    _h1_economic_rows,
    _h1_model_rank,
    _variant_probabilities,
    load_aligned_stage,
    probabilities_to_frame,
)
from experiments.raw_hold_control import MODEL_LABELS, MODEL_NAMES
from experiments.run_catboost_matched_ablation import (
    _atomic_json,
    _atomic_parquet,
    summarize_forward_evidence,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "all_model_stacking"
WIDTHS = (55, 65, 75)
ENSEMBLE_VARIANTS = ("soft_vote", "consensus", "stack")
VARIANTS = ("best_single", *ENSEMBLE_VARIANTS)
H1_STAGES = tuple(f"calibration_2025_{month:02d}" for month in range(1, 7))
ALL_MODELS = " | ".join(MODEL_NAMES)
CLASS_LABELS = {0: "short", 1: "flat", 2: "long"}


def _rank(row: Mapping[str, Any]) -> tuple[float, ...]:
    return economic_ranking_key(row, n_segments=6)


def _best_h1_single(arm: str, width_bps: int) -> dict[str, Any]:
    ranking = _h1_model_rank(arm, width_bps=width_bps)
    winner = min(ranking.to_dict("records"), key=_rank)
    winner.update(
        {
            "sentiment_arm": arm,
            "Arm": ARM_LABELS[arm],
            "variant": "best_single",
            "selected_base_model": str(winner["model_name"]),
            "all_base_models": ALL_MODELS,
        }
    )
    return winner


def _select_policy(grid: pd.DataFrame) -> dict[str, Any]:
    if len(grid) != 33:
        raise ValueError("each ensemble Arm/DZ/variant must contain 33 H1 policies")
    return min(grid.to_dict("records"), key=_rank)


def _select_dz(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) != len(WIDTHS):
        raise ValueError("each Arm/variant must contain all three dead zones")
    winner = min((dict(row) for row in rows), key=_rank)
    winner["dz_selection_rule"] = "adequacy, robust score, Sortino, net return, trades, policy"
    return winner


def _coefficient_rows(meta: Any, arm: str, width_bps: int) -> list[dict[str, Any]]:
    logistic = meta.named_steps["logisticregression"]
    feature_names = [
        f"{MODEL_LABELS[model]} — {direction}"
        for model in MODEL_NAMES
        for direction in ("P(short)", "P(long)")
    ]
    rows = []
    for class_position, class_id in enumerate(logistic.classes_):
        for feature_position, feature_name in enumerate(feature_names):
            rows.append(
                {
                    "sentiment_arm": arm,
                    "Arm": ARM_LABELS[arm],
                    "width_bps": int(width_bps),
                    "class_id": int(class_id),
                    "class_label": CLASS_LABELS[int(class_id)],
                    "feature": feature_name,
                    "coefficient": float(logistic.coef_[class_position, feature_position]),
                    "intercept": float(logistic.intercept_[class_position]),
                }
            )
    return rows


def run(output_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json({"status": "running"}, output_root / "run_state.json")

    grids: list[pd.DataFrame] = []
    per_dz_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    forward_rows: list[pd.DataFrame] = []
    monthly_rows: list[pd.DataFrame] = []
    quarterly_rows: list[pd.DataFrame] = []
    prediction_root = output_root / "predictions"

    try:
        for arm in ARMS:
            prepared = prepare_arm(arm)
            final_meta: dict[int, Any] = {}

            for width_bps in WIDTHS:
                oof_time, oof_y, oof_probs, _ = load_aligned_stage(
                    arm, MODEL_NAMES, "oof_2024", width_bps=width_bps
                )
                stack_X = _feature_matrix(oof_probs, MODEL_NAMES)
                stack_y = oof_y.to_numpy(dtype=int)
                h1_predictions = {variant: [] for variant in ENSEMBLE_VARIANTS}
                h1_fit_ids = {variant: [] for variant in ENSEMBLE_VARIANTS}

                for stage in H1_STAGES:
                    timestamp, y_true, probabilities, refits = load_aligned_stage(
                        arm, MODEL_NAMES, stage, width_bps=width_bps
                    )
                    meta = _fit_stack(stack_X, stack_y)
                    for variant in ENSEMBLE_VARIANTS:
                        fit_id = (
                            f"stack-causal-through-{stage}-w{width_bps}"
                            if variant == "stack"
                            else f"{variant}:w{width_bps}:" + "|".join(refits.values())
                        )
                        values = _variant_probabilities(
                            variant,
                            probabilities,
                            MODEL_NAMES,
                            best_single=MODEL_NAMES[0],
                            stack_model=meta if variant == "stack" else None,
                        )
                        h1_predictions[variant].append(
                            probabilities_to_frame(
                                timestamp=timestamp,
                                y_true=y_true,
                                probabilities=values,
                                refit_id=fit_id,
                            )
                        )
                        h1_fit_ids[variant].append(fit_id)
                    stack_X = np.vstack([stack_X, _feature_matrix(probabilities, MODEL_NAMES)])
                    stack_y = np.concatenate([stack_y, y_true.to_numpy(dtype=int)])

                final_meta[width_bps] = _fit_stack(stack_X, stack_y)
                coefficient_rows.extend(
                    _coefficient_rows(final_meta[width_bps], arm, width_bps)
                )
                per_dz_rows.append(_best_h1_single(arm, width_bps))

                for variant in ENSEMBLE_VARIANTS:
                    combined = pd.concat(h1_predictions[variant], ignore_index=True).sort_values("timestamp")
                    _atomic_parquet(
                        combined,
                        prediction_root / arm / f"w{width_bps}_{variant}_h1.parquet",
                    )
                    grid = _h1_economic_rows(
                        arm=arm,
                        variant=variant,
                        prediction=combined,
                        prepared=prepared,
                        fit_id="|".join(h1_fit_ids[variant]),
                        width_bps=width_bps,
                    )
                    grid["Arm"] = ARM_LABELS[arm]
                    grid["all_base_models"] = ALL_MODELS
                    grid["selected_base_model"] = None
                    grids.append(grid)
                    winner = _select_policy(grid)
                    winner.update(
                        {
                            "sentiment_arm": arm,
                            "Arm": ARM_LABELS[arm],
                            "variant": variant,
                            "all_base_models": ALL_MODELS,
                            "selected_base_model": None,
                        }
                    )
                    per_dz_rows.append(winner)

            arm_per_dz = [row for row in per_dz_rows if row["sentiment_arm"] == arm]
            for variant in VARIANTS:
                candidates = [row for row in arm_per_dz if row["variant"] == variant]
                candidate_rows.append(_select_dz(candidates))

            arm_candidates = [row for row in candidate_rows if row["sentiment_arm"] == arm]
            forward_cache: dict[int, tuple[pd.Series, pd.Series, dict[str, np.ndarray], dict[str, str]]] = {}
            for policy in arm_candidates:
                width_bps = int(policy["width_bps"])
                if width_bps not in forward_cache:
                    forward_cache[width_bps] = load_aligned_stage(
                        arm, MODEL_NAMES, "forward", width_bps=width_bps
                    )
                timestamp, y_true, probabilities, refits = forward_cache[width_bps]
                variant = str(policy["variant"])
                if variant == "best_single":
                    selected_model = str(policy["selected_base_model"])
                    values = probabilities[selected_model]
                    fit_id = f"best_single:{selected_model}:{refits[selected_model]}"
                    model_set = selected_model
                else:
                    values = _variant_probabilities(
                        variant,
                        probabilities,
                        MODEL_NAMES,
                        best_single=MODEL_NAMES[0],
                        stack_model=final_meta[width_bps] if variant == "stack" else None,
                    )
                    fit_id = (
                        f"stack-final-2024oof-plus-2025h1-w{width_bps}"
                        if variant == "stack"
                        else f"{variant}:w{width_bps}:" + "|".join(refits.values())
                    )
                    model_set = ALL_MODELS
                prediction = probabilities_to_frame(
                    timestamp=timestamp,
                    y_true=y_true,
                    probabilities=values,
                    refit_id=fit_id,
                )
                _atomic_parquet(
                    prediction,
                    prediction_root / arm / f"w{width_bps}_{variant}_forward.parquet",
                )
                ledger, per_bar = simulate_policy(
                    bars=prepared.bars,
                    execution=prepared.minute,
                    prediction_frame=prediction,
                    start=FORWARD_START,
                    end=FORWARD_END,
                    resolution="1m",
                    tau=float(policy["tau"]),
                    tp_bps=int(policy["tp_bps"]),
                    sl_bps=int(policy["sl_bps"]),
                    max_hold=int(policy["max_hold"]),
                    fee_bps=FEE_BPS,
                )
                evidence_policy = {
                    **policy,
                    "objective": variant,
                    "fit_id": fit_id,
                }
                monthly, quarterly, summary = summarize_forward_evidence(
                    per_bar=per_bar,
                    ledger=ledger,
                    regimes=prepared.regimes,
                    policy=evidence_policy,
                )
                for frame in (monthly, quarterly, summary):
                    frame["sentiment_arm"] = arm
                    frame["Arm"] = ARM_LABELS[arm]
                    frame["variant"] = variant
                    frame["model_set"] = model_set
                    frame["selected_base_model"] = policy.get("selected_base_model")
                    frame["lookback_days"] = LOOKBACK_DAYS
                monthly_rows.append(monthly)
                quarterly_rows.append(quarterly)
                forward_rows.append(summary)

        tables = {
            "h1_policy_grid": pd.concat(grids, ignore_index=True),
            "h1_selected_per_dz": pd.DataFrame(per_dz_rows),
            "h1_selected_candidates": pd.DataFrame(candidate_rows),
            "meta_coefficients": pd.DataFrame(coefficient_rows),
            "forward_monthly": pd.concat(monthly_rows, ignore_index=True),
            "forward_quarterly": pd.concat(quarterly_rows, ignore_index=True),
            "forward_summary": pd.concat(forward_rows, ignore_index=True),
        }
        expected = {
            "h1_policy_grid": 891,
            "h1_selected_per_dz": 36,
            "h1_selected_candidates": 12,
            "meta_coefficients": 486,
            "forward_monthly": 108,
            "forward_quarterly": 36,
            "forward_summary": 12,
        }
        for name, count in expected.items():
            if len(tables[name]) != count:
                raise AssertionError(f"{name}: expected {count} rows, found {len(tables[name])}")
            _atomic_parquet(tables[name], output_root / f"{name}.parquet")
            tables[name].to_csv(output_root / f"{name}.csv", index=False)

        manifest = {
            "protocol": "all-nine-logistic-stacking-v1",
            "base_models": list(MODEL_NAMES),
            "meta_learner": "StandardScaler + L2 LogisticRegression(C=0.1, class_weight=balanced)",
            "meta_features": "P(short) and P(long) from each base model (18 total)",
            "widths": list(WIDTHS),
            "sentiment_arms": list(ARMS),
            "lookback_days": LOOKBACK_DAYS,
            "h1_policy_count_per_ensemble_dz": 33,
            "development_data": "2024 blocking OOF plus causal 2025 H1",
            "forward_period": "2025-07-01 to 2026-04-01 exclusive",
            "lockbox_2026_q2_used": False,
            "artifact_rows": {name: len(frame) for name, frame in tables.items()},
        }
        _atomic_json(manifest, output_root / "manifest.json")
        _atomic_json({"status": "complete", **manifest}, output_root / "run_state.json")
        return manifest
    except Exception as exc:
        _atomic_json({"status": "failed", "error": repr(exc)}, output_root / "run_state.json")
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
