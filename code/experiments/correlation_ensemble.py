"""Correlation-selected DZ65 ensembles for Notebook 03.

Model membership is learned only from 2024 blocking-OOF and causal 2025-H1
predictions.  July 2025 onward is used once, after the subset, combiner and
execution policy have been frozen.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.all_model_sentiment_policy import DEFAULT_ROOT as POLICY_ROOT
from experiments.all_model_sentiment_raw import ARMS, DEFAULT_ROOT as RAW_ROOT, prepare_arm
from experiments.all_model_sentiment_scoreboard import ARM_LABELS
from experiments.baseline_model_zoo_1m import (
    CALIBRATION_START,
    FORWARD_END,
    FORWARD_START,
    _economic_row,
    policy_choices_for_hold,
    select_h1_policy_rows,
)
from experiments.catboost_execution_scoring import simulate_policy
from experiments.raw_hold_control import MODEL_LABELS, MODEL_NAMES
from experiments.run_catboost_matched_ablation import (
    _atomic_json,
    _atomic_parquet,
    summarize_forward_evidence,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = CODE_ROOT / "experiments" / "cache" / "tuning" / "correlation_ensemble_dz65"
WIDTH_BPS = 65
LOOKBACK_DAYS = 180
FEE_BPS = 5.0
SCORE_CORRELATION_THRESHOLD = 0.85
ERROR_CORRELATION_THRESHOLD = 0.75
VARIANTS = ("best_single", "soft_vote", "consensus", "stack")
MONTHS = tuple(pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS"))
PROBABILITY_COLUMNS = ("p_short", "p_flat", "p_long")


def redundancy_components(
    score_correlation: pd.DataFrame,
    error_correlation: pd.DataFrame,
    *,
    score_threshold: float = SCORE_CORRELATION_THRESHOLD,
    error_threshold: float = ERROR_CORRELATION_THRESHOLD,
) -> tuple[tuple[str, ...], ...]:
    """Return connected components of jointly redundant model pairs."""
    if not score_correlation.index.equals(score_correlation.columns):
        raise ValueError("score correlation must be a square labelled matrix")
    if not score_correlation.index.equals(error_correlation.index) or not error_correlation.index.equals(error_correlation.columns):
        raise ValueError("score and error correlation labels must match")
    names = list(score_correlation.index.astype(str))
    adjacent = (
        score_correlation.to_numpy(dtype=float) >= float(score_threshold)
    ) & (error_correlation.to_numpy(dtype=float) >= float(error_threshold))
    seen: set[int] = set()
    components: list[tuple[str, ...]] = []
    for start in range(len(names)):
        if start in seen:
            continue
        stack, group = [start], []
        seen.add(start)
        while stack:
            current = stack.pop()
            group.append(names[current])
            for neighbour in np.flatnonzero(adjacent[current]):
                neighbour = int(neighbour)
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(tuple(sorted(group)))
    return tuple(sorted(components, key=lambda group: (group[0], len(group))))


def correlation_order(score_correlation: pd.DataFrame) -> tuple[str, ...]:
    """Order heatmap labels with average-linkage clustering."""
    if len(score_correlation) <= 1:
        return tuple(score_correlation.index.astype(str))
    distance = 1.0 - score_correlation.clip(-1.0, 1.0).to_numpy(dtype=float)
    np.fill_diagonal(distance, 0.0)
    condensed = squareform(distance, checks=False)
    order = leaves_list(linkage(condensed, method="average"))
    names = score_correlation.index.astype(str).to_numpy()
    return tuple(names[order])


def silhouette_components(
    score_correlation: pd.DataFrame,
) -> tuple[tuple[tuple[str, ...], ...], float]:
    """Choose the number of correlation clusters by maximum silhouette."""
    names = score_correlation.index.astype(str).to_numpy()
    if len(names) < 3:
        return (tuple(names),), 0.0
    distance = 1.0 - score_correlation.clip(-1.0, 1.0).to_numpy(dtype=float)
    np.fill_diagonal(distance, 0.0)
    hierarchy = linkage(squareform(distance, checks=False), method="average")
    candidates: list[tuple[float, int, np.ndarray]] = []
    for requested_clusters in range(2, len(names)):
        labels = fcluster(hierarchy, requested_clusters, criterion="maxclust")
        actual_clusters = len(np.unique(labels))
        if actual_clusters < 2 or actual_clusters >= len(names):
            continue
        score = float(silhouette_score(distance, labels, metric="precomputed"))
        candidates.append((score, -actual_clusters, labels))
    if not candidates:
        return (tuple(names),), 0.0
    score, _, labels = max(candidates, key=lambda item: (item[0], item[1]))
    groups = [tuple(sorted(names[labels == label])) for label in sorted(np.unique(labels))]
    return tuple(sorted(groups, key=lambda group: (group[0], len(group)))), score


def probabilities_to_frame(
    *, timestamp: pd.Series, y_true: pd.Series, probabilities: np.ndarray, refit_id: str
) -> pd.DataFrame:
    """Build the common prediction-cache schema needed by execution scoring."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("ensemble probabilities must have three columns")
    probabilities = np.clip(probabilities, 0.0, None)
    totals = probabilities.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("ensemble probabilities must have positive row sums")
    probabilities = probabilities / totals
    pred = probabilities.argmax(axis=1)
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamp, utc=True),
            "y_true": np.asarray(y_true, dtype=int),
            "pred": pred.astype(int),
            "confidence": probabilities.max(axis=1),
            "p_short": probabilities[:, 0],
            "p_flat": probabilities[:, 1],
            "p_long": probabilities[:, 2],
            "refit_id": str(refit_id),
        }
    )


def _one_prediction(path: Path, *, width_bps: int = WIDTH_BPS) -> pd.DataFrame:
    paths = sorted(Path(path).glob(f"w{int(width_bps)}_*.parquet"))
    if len(paths) != 1:
        raise ValueError(f"expected one prediction parquet in {path}, found {len(paths)}")
    frame = pd.read_parquet(paths[0]).copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    if not frame["width_bps"].astype(int).eq(int(width_bps)).all():
        raise ValueError(f"prediction dead zone changed in {paths[0]}")
    return frame.sort_values("timestamp").reset_index(drop=True)


def _model_stage(
    arm: str, model_name: str, stage: str, *, width_bps: int = WIDTH_BPS
) -> pd.DataFrame:
    if stage == "oof_2024":
        paths = sorted((RAW_ROOT / arm / model_name / "predictions").glob(f"w{int(width_bps)}_candidate_00_fold_*.parquet"))
        if len(paths) != 5:
            raise ValueError(f"{arm}/{model_name}: expected five 2024 OOF folds")
        frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    elif stage.startswith("calibration_2025_"):
        frame = _one_prediction(
            POLICY_ROOT / arm / model_name / "stage_predictions" / stage,
            width_bps=width_bps,
        )
    elif stage == "forward":
        frame = _one_prediction(
            RAW_ROOT / arm / model_name / "stage_predictions" / "raw_forward",
            width_bps=width_bps,
        )
    else:
        raise ValueError(f"unknown prediction stage: {stage}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    required = {"timestamp", "y_true", *PROBABILITY_COLUMNS, "refit_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"prediction stage misses columns: {sorted(missing)}")
    return frame.sort_values("timestamp").reset_index(drop=True)


def load_aligned_stage(
    arm: str,
    model_names: Sequence[str],
    stage: str,
    *,
    width_bps: int = WIDTH_BPS,
) -> tuple[pd.Series, pd.Series, dict[str, np.ndarray], dict[str, str]]:
    """Inner-align one prediction stage across model families."""
    merged: pd.DataFrame | None = None
    refits: dict[str, str] = {}
    for model_name in model_names:
        frame = _model_stage(arm, model_name, stage, width_bps=width_bps)
        refits[model_name] = "|".join(sorted(frame["refit_id"].astype(str).unique()))
        keep = frame.loc[:, ["timestamp", "y_true", *PROBABILITY_COLUMNS]].rename(
            columns={
                "y_true": f"{model_name}__y_true",
                **{column: f"{model_name}__{column}" for column in PROBABILITY_COLUMNS},
            }
        )
        merged = keep if merged is None else merged.merge(keep, on="timestamp", how="inner", validate="one_to_one")
    if merged is None or merged.empty:
        raise ValueError(f"no aligned predictions for {arm}/{stage}")
    truth_columns = [f"{model_name}__y_true" for model_name in model_names]
    if not merged[truth_columns].eq(merged[truth_columns[0]], axis=0).all().all():
        raise ValueError(f"target labels differ across models for {arm}/{stage}")
    probabilities = {
        model_name: merged[[f"{model_name}__{column}" for column in PROBABILITY_COLUMNS]].to_numpy(dtype=float)
        for model_name in model_names
    }
    return merged["timestamp"], merged[truth_columns[0]].astype(int), probabilities, refits


def _feature_matrix(probabilities: Mapping[str, np.ndarray], models: Sequence[str]) -> np.ndarray:
    # Flat probability is implied by P(short)+P(flat)+P(long)=1.  Keeping only
    # the two independent directional probabilities avoids exact collinearity.
    return np.column_stack([probabilities[model][:, [0, 2]] for model in models])


def _fit_stack(X: np.ndarray, y: np.ndarray):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            l1_ratio=0.0,
            max_iter=2000,
            random_state=42,
        ),
    ).fit(X, y)


def _variant_probabilities(
    variant: str,
    probabilities: Mapping[str, np.ndarray],
    selected_models: Sequence[str],
    *,
    best_single: str,
    stack_model: Any | None = None,
) -> np.ndarray:
    arrays = [probabilities[model] for model in selected_models]
    if variant == "best_single":
        return probabilities[best_single]
    if variant == "soft_vote":
        return np.mean(arrays, axis=0)
    if variant == "consensus":
        votes = np.stack([array.argmax(axis=1) for array in arrays], axis=1)
        unanimous = np.all(votes == votes[:, [0]], axis=1)
        output = np.zeros((len(votes), 3), dtype=float)
        output[:, 1] = 1.0
        output[unanimous] = np.eye(3)[votes[unanimous, 0]]
        return output
    if variant == "stack":
        if stack_model is None:
            raise ValueError("stack variant requires a fitted meta-model")
        output = np.zeros((len(arrays[0]), 3), dtype=float)
        predicted = stack_model.predict_proba(_feature_matrix(probabilities, selected_models))
        for source, label in enumerate(stack_model.classes_):
            output[:, int(label)] = predicted[:, source]
        return output
    raise ValueError(f"unknown ensemble variant: {variant}")


def _h1_economic_rows(
    *,
    arm: str,
    variant: str,
    prediction: pd.DataFrame,
    prepared: Any,
    fit_id: str,
    width_bps: int = WIDTH_BPS,
) -> pd.DataFrame:
    rows = []
    month_edges = pd.date_range(CALIBRATION_START, FORWARD_START, freq="MS")
    for policy_id, (tau, geometry) in enumerate(policy_choices_for_hold(1)):
        tp_bps, sl_bps, hold_bars = geometry
        ledger, per_bar = simulate_policy(
            bars=prepared.bars,
            execution=prepared.minute,
            prediction_frame=prediction,
            start=CALIBRATION_START,
            end=FORWARD_START,
            resolution="1m",
            tau=float(tau),
            tp_bps=int(tp_bps),
            sl_bps=int(sl_bps),
            max_hold=int(hold_bars),
            fee_bps=FEE_BPS,
        )
        segment_nets = [
            float(per_bar.loc[(per_bar.index >= left) & (per_bar.index < right)].sum())
            for left, right in zip(month_edges[:-1], month_edges[1:])
        ]
        row = _economic_row(
            model_name=f"{arm}__{variant}",
            width_bps=int(width_bps),
            candidate_id=0,
            policy_id=policy_id,
            tau=float(tau),
            tp_bps=int(tp_bps),
            sl_bps=int(sl_bps),
            hold_bars=int(hold_bars),
            ledgers=(ledger,),
            returns=(per_bar,),
            segment_nets=segment_nets,
            regimes=prepared.regimes,
        )
        row.update(
            {
                "sentiment_arm": arm,
                "variant": variant,
                "lookback_days": LOOKBACK_DAYS,
                "fit_id": fit_id,
                "monthly_fit_count": 6,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _h1_model_rank(arm: str, *, width_bps: int = WIDTH_BPS) -> pd.DataFrame:
    rows = []
    for model_name in MODEL_NAMES:
        frame = pd.read_parquet(POLICY_ROOT / arm / model_name / "selected_policies_2025h1.parquet")
        row = frame.loc[frame["width_bps"].astype(int) == int(width_bps)].copy()
        if len(row) != 1:
            raise ValueError(f"{arm}/{model_name}: expected one DZ65 H1 policy")
        rows.append(row.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def _ranking_tuple(row: Mapping[str, Any]) -> tuple[float, ...]:
    from experiments.catboost_matched_ablation import economic_ranking_key

    return economic_ranking_key(row, n_segments=6)


def _select_representatives(
    components: Sequence[Sequence[str]], h1_rank: pd.DataFrame
) -> tuple[str, ...]:
    indexed = {row["model_name"]: row for row in h1_rank.to_dict("records")}
    selected = []
    for component in components:
        selected.append(min(component, key=lambda model: _ranking_tuple(indexed[model])))
    return tuple(selected)


def run(output_root: Path = DEFAULT_ROOT) -> dict[str, Any]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _atomic_json({"status": "running"}, output_root / "run_state.json")
    score_long, error_long, cluster_rows, selected_rows = [], [], [], []
    grids, selected_policies, forward_rows, monthly_rows, quarterly_rows = [], [], [], [], []
    prediction_root = output_root / "predictions"

    try:
        for arm in ARMS:
            prepared = prepare_arm(arm)
            stage_names = ("oof_2024", *(f"calibration_2025_{month:02d}" for month in range(1, 7)))
            development = [load_aligned_stage(arm, MODEL_NAMES, stage) for stage in stage_names]
            scores = pd.DataFrame(
                {
                    model: np.concatenate([item[2][model][:, 2] - item[2][model][:, 0] for item in development])
                    for model in MODEL_NAMES
                }
            )
            truth = np.concatenate([item[1].to_numpy(dtype=int) for item in development])
            errors = pd.DataFrame(
                {
                    model: np.concatenate([item[2][model].argmax(axis=1) for item in development]) != truth
                    for model in MODEL_NAMES
                }
            ).astype(float)
            score_corr = scores.corr(method="spearman").fillna(0.0)
            error_corr = errors.corr(method="spearman").fillna(0.0)
            for model_name in MODEL_NAMES:
                score_corr.loc[model_name, model_name] = 1.0
                error_corr.loc[model_name, model_name] = 1.0
            order = correlation_order(score_corr)
            components, silhouette = silhouette_components(score_corr)
            h1_rank = _h1_model_rank(arm)
            selected = _select_representatives(components, h1_rank)
            best_single = min(selected, key=lambda model: _ranking_tuple(h1_rank.loc[h1_rank["model_name"] == model].iloc[0].to_dict()))

            for left in MODEL_NAMES:
                for right in MODEL_NAMES:
                    score_long.append({"sentiment_arm": arm, "model_a": left, "model_b": right, "correlation": score_corr.loc[left, right], "heatmap_order_a": order.index(left), "heatmap_order_b": order.index(right)})
                    error_long.append({"sentiment_arm": arm, "model_a": left, "model_b": right, "correlation": error_corr.loc[left, right]})
            for cluster_id, component in enumerate(components, 1):
                representative = next(model for model in selected if model in component)
                for model in component:
                    cluster_rows.append({"sentiment_arm": arm, "cluster_id": cluster_id, "model_name": model, "representative": representative, "selected": model == representative, "cluster_size": len(component)})
            selected_rows.append({"sentiment_arm": arm, "Arm": ARM_LABELS[arm], "selected_count": len(selected), "selected_models": " | ".join(selected), "best_single": best_single, "cluster_count": len(components), "silhouette": silhouette, "selection_rule": "maximum silhouette on Spearman directional-score distance"})

            oof_time, oof_y, oof_probs, _ = development[0]
            X_train = _feature_matrix(oof_probs, selected)
            y_train = oof_y.to_numpy(dtype=int)
            h1_predictions: dict[str, list[pd.DataFrame]] = {variant: [] for variant in VARIANTS}
            h1_fit_ids: dict[str, list[str]] = {variant: [] for variant in VARIANTS}
            stack_training_X, stack_training_y = X_train.copy(), y_train.copy()
            for month_index, stage in enumerate(stage_names[1:], 1):
                timestamp, y_true, probabilities, refits = development[month_index]
                meta = _fit_stack(stack_training_X, stack_training_y)
                for variant in VARIANTS:
                    fit_id = (
                        f"stack-causal-through-{stage}" if variant == "stack" else
                        f"{variant}:{'|'.join(refits[model] for model in selected)}"
                    )
                    values = _variant_probabilities(variant, probabilities, selected, best_single=best_single, stack_model=meta if variant == "stack" else None)
                    h1_predictions[variant].append(probabilities_to_frame(timestamp=timestamp, y_true=y_true, probabilities=values, refit_id=fit_id))
                    h1_fit_ids[variant].append(fit_id)
                stack_training_X = np.vstack([stack_training_X, _feature_matrix(probabilities, selected)])
                stack_training_y = np.concatenate([stack_training_y, y_true.to_numpy(dtype=int)])

            for variant in VARIANTS:
                combined = pd.concat(h1_predictions[variant], ignore_index=True).sort_values("timestamp")
                _atomic_parquet(combined, prediction_root / arm / f"{variant}_h1.parquet")
                grid = _h1_economic_rows(arm=arm, variant=variant, prediction=combined, prepared=prepared, fit_id="|".join(h1_fit_ids[variant]))
                grids.append(grid)

            arm_grid = pd.concat(grids[-len(VARIANTS):], ignore_index=True)
            arm_selected = select_h1_policy_rows(arm_grid)
            arm_selected["sentiment_arm"] = arm
            arm_selected["variant"] = arm_selected["model_name"].str.split("__", n=1).str[1]
            arm_selected["selected_models"] = " | ".join(selected)
            selected_policies.append(arm_selected)

            forward_time, forward_y, forward_probs, forward_refits = load_aligned_stage(arm, MODEL_NAMES, "forward")
            final_meta = _fit_stack(stack_training_X, stack_training_y)
            for variant in VARIANTS:
                policy = arm_selected.loc[arm_selected["variant"] == variant].iloc[0].to_dict()
                values = _variant_probabilities(variant, forward_probs, selected, best_single=best_single, stack_model=final_meta if variant == "stack" else None)
                fit_id = (
                    "stack-final-2024oof-plus-2025h1" if variant == "stack" else
                    f"{variant}:{'|'.join(forward_refits[model] for model in selected)}"
                )
                prediction = probabilities_to_frame(timestamp=forward_time, y_true=forward_y, probabilities=values, refit_id=fit_id)
                _atomic_parquet(prediction, prediction_root / arm / f"{variant}_forward.parquet")
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
                evidence_policy = {**policy, "objective": variant, "fit_id": fit_id}
                monthly, quarterly, summary = summarize_forward_evidence(per_bar=per_bar, ledger=ledger, regimes=prepared.regimes, policy=evidence_policy)
                for frame in (monthly, quarterly, summary):
                    frame["sentiment_arm"] = arm
                    frame["Arm"] = ARM_LABELS[arm]
                    frame["variant"] = variant
                    frame["selected_models"] = " | ".join(selected)
                    frame["selected_count"] = len(selected)
                    frame["best_single"] = best_single
                monthly_rows.append(monthly)
                quarterly_rows.append(quarterly)
                forward_rows.append(summary)

        tables = {
            "score_correlations": pd.DataFrame(score_long),
            "error_correlations": pd.DataFrame(error_long),
            "clusters": pd.DataFrame(cluster_rows),
            "selected_models": pd.DataFrame(selected_rows),
            "h1_policy_grid": pd.concat(grids, ignore_index=True),
            "selected_policies": pd.concat(selected_policies, ignore_index=True),
            "forward_monthly": pd.concat(monthly_rows, ignore_index=True),
            "forward_quarterly": pd.concat(quarterly_rows, ignore_index=True),
            "forward_summary": pd.concat(forward_rows, ignore_index=True),
        }
        expected = {"score_correlations": 243, "error_correlations": 243, "selected_models": 3, "h1_policy_grid": 396, "selected_policies": 12, "forward_monthly": 108, "forward_quarterly": 36, "forward_summary": 12}
        for name, count in expected.items():
            if len(tables[name]) != count:
                raise AssertionError(f"{name}: expected {count} rows, found {len(tables[name])}")
        for name, frame in tables.items():
            _atomic_parquet(frame, output_root / f"{name}.parquet")
            frame.to_csv(output_root / f"{name}.csv", index=False)
        manifest = {
            "protocol": "correlation-selected-ensemble-dz65-v1",
            "width_bps": WIDTH_BPS,
            "lookback_days": LOOKBACK_DAYS,
            "development_correlation_data": "2024 blocking OOF plus causal 2025 H1",
            "forward_period": "2025-07-01 to 2026-04-01 exclusive",
            "lockbox_2026_q2_used": False,
            "cluster_count_rule": "maximum silhouette over 2..8 average-linkage clusters",
            "error_correlation_role": "reported audit only; not used to choose cluster count",
            "variants": list(VARIANTS),
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
